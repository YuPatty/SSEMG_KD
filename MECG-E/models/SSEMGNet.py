# /data/member1/user_howardshih/shihsemg/MECG-E/models/SSEMGNet.py
# =============================================================================
# SSEMGNet.py (SSEMG-Net model implementation)/data/member1/user_howardshih/shihsemg/MECG-E/models/SSEMGNet.py
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import InstanceNorm2d
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from functools import partial
from mamba_ssm.models.mixer_seq_simple import Block, _init_weights
from mamba_ssm.modules.mamba_simple import Mamba
import mamba_ssm.modules.mamba_simple as _mamba_simple_module
from mamba_ssm.ops.triton.layer_norm import RMSNorm as _TritonRMSNorm

# ── CPU 相容性修補（第二處，比 RMSNorm 更隱蔽）──────────────────
# 實測發現：即使 Mamba(..., use_fast_path=False)，mamba_simple.py 內部呼叫
# causal_conv1d_fn 的判斷邏輯，並不是看 use_fast_path，而是獨立判斷
# 「causal_conv1d_fn 這個名稱在模組命名空間裡是不是 None」——只要環境裡
# 裝了 causal_conv1d 套件（這個名稱因此不是 None），不管 use_fast_path
# 設什麼，都會被觸發並在 CPU tensor 上崩潰。
# 同樣地，當 use_fast_path=False 時，Mamba 仍會呼叫 selective_scan_fn，
# 而它底層綁死了 selective_scan_cuda，導致在 CPU 報錯。
# 修法：沒有 CUDA 時，將 causal_conv1d 設為 None 觸發 fallback；
# 並將 selective_scan_fn 強制替換為純 PyTorch 實作的 selective_scan_ref。
if not torch.cuda.is_available():
    _mamba_simple_module.causal_conv1d_fn = None
    _mamba_simple_module.causal_conv1d_update = None
    
    # 替換 selective_scan_fn 為 PyTorch 參考實作
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
        _mamba_simple_module.selective_scan_fn = selective_scan_ref
    except ImportError:
        pass

# ── CPU 相容性修補 ──────────────────────────────────────────
# 原本的 RMSNorm 是從 mamba_ssm.ops.triton.layer_norm 匯入的 Triton 版本，
# 即使 Block 的 fused_add_norm=False，只要呼叫 self.norm(x) 內部一樣會觸發
# Triton kernel（RuntimeError: invalid argument to exchangeDevice），CPU
# 環境完全無法執行。這裡另外實作一個數學等價、純 PyTorch 的 RMSNorm，只在
# 沒有 CUDA 的環境自動切換使用，GPU 環境完全不受影響（維持原本的 Triton
# 加速路徑，速度不變、不影響你已經測過的 28.26ms 那組結果）。
class _PlainRMSNorm(nn.Module):
    """跟 mamba_ssm 的 Triton RMSNorm 數學上等價的純 PyTorch 版本，
    參數命名（self.weight）刻意保持一致，確保 checkpoint 能用 strict=True 載入。

    呼叫簽章比照實際崩潰堆疊確認過的用法（fused_add_norm=False 時，
    Block.forward 只會用單一位置參數呼叫 self.norm(x)，不會用到
    residual/prenorm 這些只有 fused 路徑才需要的關鍵字參數）。"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(dtype)


def _get_norm_cls(device_type: str):
    """依裝置自動選擇：CUDA 用原本的 Triton RMSNorm（不影響既有 GPU 效能），
    CPU 用上面的純 PyTorch 版本（唯一能在 CPU 執行的路徑）。"""
    return _TritonRMSNorm if device_type == 'cuda' else _PlainRMSNorm


RMSNorm = _TritonRMSNorm  # 保留原本的名稱以維持向後相容（其他地方若直接 import RMSNorm 仍可用）

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


class _LearnableSigmoidBase(nn.Module):
    """
    共同邏輯：把 raw 參數經 softplus → 永遠 >0，
    再加一個可設定的最小斜率 min_slope 避免太扁平。
    """
    def __init__(self, in_features: int, beta: float = 1.0,
                 min_slope: float = 0.05):
        super().__init__()
        self.beta = beta
        self.min_slope = float(min_slope)

        # raw 以 0 起始，softplus(0)=0.693…
        self.raw = nn.Parameter(torch.zeros(1, in_features))

    def _positive_slope(self):
        # softplus 保證 >0；再加 min_slope 作「地板」
        return self.min_slope + F.softplus(self.raw)

    def extra_repr(self) -> str:
        return f'beta={self.beta}, min_slope={self.min_slope}'


class LearnableSigmoid_1d(_LearnableSigmoidBase):
    def forward(self, x):
        slope = self._positive_slope()             # [1, F]
        return self.beta * torch.sigmoid(slope * x)


class LearnableSigmoid_2d(_LearnableSigmoidBase):
    def forward(self, x):                          # x: [B, T, F]
        slope = self._positive_slope()             # [1, F] → broadcast
        return self.beta * torch.sigmoid(slope * x)

        
def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)
    return mag, pha, com

def mag_pha_stft_loss(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    real_part = stft_spec.real
    imag_part = stft_spec.imag
    stft_spec = torch.stack((real_part, imag_part), dim=-1)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:,:,:,1] + (1e-10), stft_spec[:,:,:,0] + (1e-5))
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)
    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # mag/pha 預期為 [B, F, T]
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device, dtype=com.dtype if com.is_floating_point() else torch.float32)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)
    return wav

class MambaBlock(nn.Module):
    def __init__(self, in_channels, n_layer=1, bidirectional=False):
        super(MambaBlock, self).__init__()
        # ── CPU 相容性修補 ──
        # 原本 use_fast_path=True 強制 Mamba 核心的 selective scan 一定走
        # Triton/CUDA 加速路徑，CPU 上會直接崩潰。這裡在「建構模型的當下」
        # 先偵測有沒有 CUDA，沒有的話就把 use_fast_path 設 False，讓 Mamba
        # 自動切換成官方內建的純 PyTorch fallback（slow_forward），可以在
        # CPU 上跑，只是速度慢很多——這是「能不能跑」與「跑多快」的取捨，
        # 不是要用這個路徑真的拿去部署，純粹是為了量測 CPU 延遲這個對照數字。
        # GPU 環境完全不受影響：torch.cuda.is_available()=True 時，
        # use_fast_path 仍是 True，跟原本行為一致。
        use_cuda = torch.cuda.is_available()
        norm_cls_selected = partial(_TritonRMSNorm, eps=1e-5) if use_cuda else partial(_PlainRMSNorm, eps=1e-5)
        use_fast_path_selected = True if use_cuda else False

        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4,
                                       use_fast_path=use_fast_path_selected),
                    mlp_cls=nn.Identity, 
                    norm_cls=norm_cls_selected,
                    fused_add_norm=False,
                )
            )
        self.backward_blocks = None  # ← 預設 None，避免 AttributeError
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4,
                                           use_fast_path=use_fast_path_selected),
                        mlp_cls=nn.Identity,
                        norm_cls=norm_cls_selected,
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x):
        for_residual = None
        forward_f = x.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(x, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)
        
        return residual
 

class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        
        self.h = h
        # --- 安全檢查：GroupNorm 分組需整除 ----------------
        if h.get('use_gn', False):
            assert h.dense_channel % h.get('gn_groups', 8) == 0, \
                f"dense_channel={h.dense_channel} 不能被 gn_groups={h.gn_groups} 整除"
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                (nn.GroupNorm(h.get('gn_groups', 8), h.dense_channel, affine=True)
                if h.get('use_gn', False)
                else nn.InstanceNorm2d(h.dense_channel, affine=True)),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        # ── (1) 首層：Conv → GN / IN → PReLU ──
        gn_or_in = (
            nn.GroupNorm(h.get('gn_groups', 8), h.dense_channel, affine=True)
            if h.get('use_gn', False)
            else nn.InstanceNorm2d(h.dense_channel, affine=True)
        )
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            gn_or_in,
            nn.PReLU(h.dense_channel)
        )
        
        depth = h.get('edepth', 4)
        self.dense_block = DenseBlock(h, depth=depth)  # ← 用 depth 變數

        gn_or_in2 = (
            nn.GroupNorm(h.get('gn_groups', 8), h.dense_channel, affine=True)
            if h.get('use_gn', False)
            else nn.InstanceNorm2d(h.dense_channel, affine=True)
        )
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            gn_or_in2,
            nn.PReLU(h.dense_channel)
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        depth = h.get('mdepth', 4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        F = h.n_fft // 2 + 1
        self.lsigmoid = LearnableSigmoid_2d(F, beta=h.beta)

        # ---- 可選 gate：只有 mask_hp_cut_hz > 0 才建立 ----
        self.cut_bin = -1
        cut_hz = h.get('mask_hp_cut_hz', None)
        try:
            cut_hz = float(cut_hz) if cut_hz is not None else None
        except Exception:
            cut_hz = None

        if (cut_hz is not None) and (cut_hz > 0):
            sr   = float(h.get('sampling_rate', 1000))
            nfft = int(h.get('n_fft', 512))
            df   = sr / nfft
            cut_bin = int(np.floor(cut_hz / df + 1e-9))
            cut_bin = max(0, min(cut_bin, F))

            low_gate = torch.zeros(1, 1, 1, F)
            low_gate[..., :cut_bin] = 1.0
            self.register_buffer('low_gate', low_gate)  # ← 只在需要時註冊
            self.cut_bin = cut_bin
        # 不需要 gate 就不註冊；forward 會用 getattr 檢查

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)          # [B, 1, T, F]
        m_tf = self.lsigmoid(x.squeeze(1))   # [B, T, F] → (0,1)

        m_tf = m_tf.unsqueeze(1)       # [B, 1, T, F]
        low_gate = getattr(self, 'low_gate', None)
        if low_gate is not None:
            m_tf = m_tf * low_gate + (1.0 - low_gate)   # 低頻用可學，高頻固定 1
        return m_tf



class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        depth = h.get('pdepth',4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

class ComplexDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(ComplexDecoder, self).__init__()
        depth = h.get('pdepth',4)
        self.dense_block = DenseBlock(h, depth=depth)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.cat((x_r, x_i), dim=1)
        return x

class TSMambaBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h


        self.time_mamba = MambaBlock(h.dense_channel, 1, bidirectional=True)
        self.freq_mamba = MambaBlock(h.dense_channel, 1, bidirectional=True)

        self.tlinear = nn.ConvTranspose1d(h.dense_channel * 2,
                                          h.dense_channel, 1)
        if h.get('fmamba', True):
            self.flinear = nn.ConvTranspose1d(h.dense_channel * 2,
                                              h.dense_channel, 1)

    def forward(self, x):                       # x: [B,C,T,F/2]
        B, C, T, Fh = x.size()

        # ── 時軸 ─────────────────────────────
        xt = x.permute(0, 3, 2, 1).reshape(B*Fh, T, C)
        xt = self.time_mamba(xt)
        xt = self.tlinear(xt.permute(0, 2, 1)).permute(0, 2, 1)
        xt = xt.reshape(B, Fh, T, C)

        # ── 頻軸（可選）──────────────────────
        if self.h.get('fmamba', True):
            xf = xt.permute(0, 2, 1, 3).reshape(B*T, Fh, C)
            xf = self.freq_mamba(xf)
            xf = self.flinear(xf.permute(0, 2, 1)).permute(0, 2, 1)
            xt = xf.reshape(B, T, Fh, C).permute(0, 2, 1, 3)

        return xt.permute(0, 3, 2, 1)           # 回到 [B,C,T,F/2]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# === 在 _mrstft_loss 之外建立 cache（全域即可） ===
_STFT_WINDOW_CACHE = {}

# === 取代原本的 _mrstft_loss（和輔助函式） ==============================

def _get_hann(win, device, dtype):
    """Hann 視窗快取輔助。"""
    key = (win, str(device), str(dtype))
    if key not in _STFT_WINDOW_CACHE:
        _STFT_WINDOW_CACHE[key] = torch.hann_window(win, device=device, dtype=dtype)
    return _STFT_WINDOW_CACHE[key]

def _stft_mag(wav, n_fft, hop, win, center=True, eps=1e-9):
    """回傳 |STFT|，形狀 [B, F, T]，Hann 視窗有快取。"""
    device, dtype = wav.device, wav.dtype
    window = _get_hann(win, device, dtype)
    X = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=center, pad_mode='reflect', normalized=False, return_complex=True
    )
    mag = torch.clamp(X.abs(), min=eps)  # 避免 log(0)
    return mag

def _spectral_convergence(x_mag, y_mag):
    """
    Lsc = || |Y|-|X| ||_F / || |Y| ||_F
    逐 batch 計算 Frobenius ratio 後取平均。
    """
    B = y_mag.size(0)
    num = torch.linalg.norm((y_mag - x_mag).reshape(B, -1), ord=2, dim=1)
    den = torch.linalg.norm(y_mag.reshape(B, -1), ord=2, dim=1) + 1e-12
    return (num / den).mean()

def _log_mag_l1(x_mag, y_mag, log_eps=1e-7):
    """
    Lmag = (1/N) * || log|X| - log|Y| ||_1
    這裡直接用 elementwise L1 的 mean（等價於除以 N 再求平均）。
    """
    return (torch.log(x_mag + log_eps) - torch.log(y_mag + log_eps)).abs().mean()

def _mrstft_loss(
    wav_pred: torch.Tensor,
    wav_ref: torch.Tensor,
    cfg,
    # 程式碼原 scales（用戶指定以程式碼為準）
    scales=((256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024))
):
    """
    Multi-Resolution STFT loss（與論文一致）：
      對每個尺度 m 計算 Ls^(m) = Lsc + Lmag，最後取平均：
        L_mrstft = (1/M) * sum_m Ls^(m)
    可由 cfg 指定權重與 eps（選用）。
    """
    device = wav_pred.device
    assert wav_pred.shape == wav_ref.shape and wav_pred.dim() == 2, \
        f"Expect [B, T] waveform; got {tuple(wav_pred.shape)}"

    # 可從設定帶入（若無則預設 1.0 / 1e-7）
    w_sc   = float(cfg.get("w_sc", 1.0))
    w_log  = float(cfg.get("w_log_mag", 1.0))
    log_eps = float(cfg.get("mrstft_log_eps", 1e-7))
    eps     = float(cfg.get("mrstft_eps", 1e-9))

    total = torch.tensor(0.0, device=device)
    M = len(scales)

    for n_fft, hop, win in scales:
        # 只用幅度（論文的 Lsc 與 Lmag 都在 |STFT|）
        X_mag = _stft_mag(wav_pred, n_fft, hop, win, center=True, eps=eps)
        Y_mag = _stft_mag(wav_ref , n_fft, hop, win, center=True, eps=eps)

        l_sc  = _spectral_convergence(X_mag, Y_mag) if w_sc  != 0.0 else 0.0
        l_mag = _log_mag_l1        (X_mag, Y_mag, log_eps) if w_log != 0.0 else 0.0

        total = total + (w_sc * l_sc + w_log * l_mag)

    return total / M

class SSEMGNet(nn.Module):
    def __init__(self, config):
        super(SSEMGNet, self).__init__()

        # ---- ablation flags ---------------------------------
        ab_cfg = config.get('ablation', {})
        self.use_mrstft  = bool(ab_cfg.get('use_mrstft',  True))
        self.use_entropy = bool(ab_cfg.get('use_entropy', True))
        # ---- phase ablation flag ---------------------------------
        ab_cfg = config.get('ablation', {})
        self.use_phase = bool(ab_cfg.get('use_phase', True))
        
        h = AttrDict(config['model'])
        self.fea        = h.get('fea', 'pha')
        self.h          = h
        self.norm       = h.norm
        self.loss_fn    = h.loss_fn.split('+') if isinstance(h.loss_fn, str) else list(h.loss_fn)
        self.num_tscblocks = h.num_tscblocks

        # ──────────────────────────────────────────────────────
        # 1) 共用元件
        # ──────────────────────────────────────────────────────
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSMamba = nn.ModuleList([
            TSMambaBlock(h) for _ in range(h.num_tscblocks)
        ])

        self.mask_decoder   = MaskDecoder(h, out_channel=1)

        # → ❶ 先統一建立 ComplexDecoder，之後任何 fea 都能用
        self.complex_decoder = ComplexDecoder(h, out_channel=1)

        # ──────────────────────────────────────────────────────
        # 2) 依 fea 額外元件
        # ──────────────────────────────────────────────────────
        if self.fea == 'cpx':
            pass                           # 早已擁有 complex_decoder

        elif self.fea == 'wav':
            self.encoder = nn.Conv1d(
                1, (h.n_fft // 2 + 1) * 2,
                h.win_size, h.hop_size,
                padding=h.win_size // 2
            )
            self.decoder = nn.ConvTranspose1d(
                (h.n_fft // 2 + 1) * 2, 1,
                h.win_size, h.hop_size,
                padding=h.win_size // 2, output_padding=0
            )
            # complex_decoder 已有，這裡不用再重建

        elif self.fea == 'pha':
            self.phase_decoder = PhaseDecoder(h, out_channel=1)
            # phase estimation ablation:
            #   use_phase=True  -> predict phase via PhaseDecoder (original)
            #   use_phase=False -> reuse noisy phase (no PhaseDecoder)
            if self.use_phase:
                self.phase_decoder = PhaseDecoder(h, out_channel=1)
            else:
                self.phase_decoder = None
            # 同樣沿用共用的 complex_decoder（只在 'con' loss 時用得到）

        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

    def forward(self, clean_audio, noisy_audio): # [B, F, T]
        if clean_audio.ndim == 4 and clean_audio.shape[1] == 2:
            return self.forward_spectrogram(clean_audio, noisy_audio)

        if self.norm=='1':
            norm_factor = torch.sqrt(noisy_audio.shape[-1] / torch.sum(noisy_audio ** 2.0, -1, keepdim=True))
        elif self.norm=='2':
            norm_factor = 1 / noisy_audio.abs().max(-1, keepdim=True)[0]
        else:
            norm_factor = torch.ones((noisy_audio.shape[0],1,1),device=noisy_audio.device)

        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = (noisy_audio * norm_factor).squeeze(1)
        
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 

        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]

        if self.fea=='cpx':
            x = noisy_com.permute(0, 3, 2, 1) # [B, 2, T, F]
        elif self.fea=='pha':
            noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1) # [B, 1, T, F]
            x = torch.cat((noisy_mag, noisy_pha), dim=1) # [B, 2, T, F]
        elif self.fea=='wav':
            x = self.encoder(noisy_audio.unsqueeze(1))
            B, C, T = x.shape
            x = x.view(B, 2, -1, T).permute(0, 1, 3, 2)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSMamba[i](x)
        
        mag_g = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)

        if self.fea=='cpx':
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack((mag_g*torch.cos(noisy_pha),
                                    mag_g*torch.sin(noisy_pha)), dim=-1)
            com_g = com_g + com_d
            # mag_g = torch.abs(torch.complex(com_g[...,0], com_g[...,1]))
            pha_g = torch.angle(torch.complex(com_g[...,0], com_g[...,1]))
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='pha':
            # phase ablation:
            #   if use_phase: predict phase
            #   else: reuse noisy_pha
            if self.use_phase and (self.phase_decoder is not None):
                pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
            else:
                pha_g = noisy_pha
            com_g = torch.stack((mag_g*torch.cos(pha_g),
                                        mag_g*torch.sin(pha_g)), dim=-1)
            audio_g = mag_pha_istft(mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
        elif self.fea=='wav':
            com_d = self.complex_decoder(x).permute(0, 1, 3, 2).reshape(B, C, T)
            audio_g = self.decoder(com_d).squeeze(1)
            _, _, com_g = mag_pha_stft_loss(audio_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor) 
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented!")

        loss_gen_all = 0

        # Time Loss
        if 'time' in self.loss_fn:
            loss_time = F.l1_loss(clean_audio, audio_g, reduction='none')
            loss_time = (loss_time/norm_factor.squeeze(-1)).mean()
            loss_gen_all += loss_time * 0.5

        # L2 Complex Loss
        if 'com' in self.loss_fn:
            loss_com = F.mse_loss(clean_com, com_g, reduction='none') * 2
            loss_com = (loss_com/norm_factor.unsqueeze(-1)).mean()
            loss_gen_all += loss_com * 0.5

        # Consistancy Loss
        if 'con' in self.loss_fn:
            _, _, com_con = mag_pha_stft_loss(audio_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor)
            loss_con = F.mse_loss(com_g, com_con, reduction='none') * 2
            loss_con = (loss_con/norm_factor.unsqueeze(-1)).mean()
            loss_gen_all += loss_con * 0.5
        
        return loss_gen_all

    def forward_spectrogram(self, clean_spec, noisy_spec):
        """
        clean_spec, noisy_spec: [B, 2, F, T]
        依 config['model']['fea']：
          - 'pha'：通道 2 表示 [mag(已壓縮), pha]
          - 'cpx'：通道 2 表示 [real, imag]（對應壓縮後的 mag*cos, mag*sin）
        """

        # --------- helper: ISTFT 一律用 float32 -----------
        def istft32(mag_FT, pha_FT):
            return mag_pha_istft(
                mag_FT.float(), pha_FT.float(),
                n_fft=self.h.n_fft,
                hop_size=self.h.hop_size,
                win_size=self.h.win_size,
                compress_factor=self.h.compress_factor
            )

        # Step 1: 統一成 [B, 2, T, F]
        x_noisy  = noisy_spec.permute(0, 1, 3, 2).contiguous()   # [B, 2, T, F]
        x_clean  = clean_spec.permute(0, 1, 3, 2).contiguous()   # [B, 2, T, F]

        # ---- 一次性 debug（只在第一個 batch 的第一次 forward 印）----
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            with torch.no_grad():
                a, b = x_noisy[:,0], x_noisy[:,1]
                neg_ratio = (a < 0).float().mean().item()
                pha_max   = b.abs().max().item()
                print(f"[debug] x_noisy ch0 neg_ratio={neg_ratio:.3f} (mag應該≈0)， ch1 |max|={pha_max:.3f} (pha應該≈3.14)")
                print(f"[debug] expect fea='{self.fea}' ; x_noisy shape={tuple(x_noisy.shape)} , x_clean shape={tuple(x_clean.shape)}")

        # Step 2: 從 noisy 取出 mag/pha 或 real/imag（皆為 [B, T, F]）
        # 若 fea='pha' 但 ch0 有大量負值、或 ch1 振幅遠超出 pi，視為 (real,imag)，自動轉成 (mag,pha)
        if self.fea == 'cpx':
            noisy_real_TF, noisy_imag_TF = x_noisy[:, 0], x_noisy[:, 1]   # [B, T, F]
            mag_TF = torch.sqrt(noisy_real_TF**2 + noisy_imag_TF**2 + 1e-12)
            pha_TF = torch.atan2(noisy_imag_TF, noisy_real_TF)
            x_input = x_noisy                                          # [B, 2, T, F]
        elif self.fea == 'pha':
            ch0, ch1 = x_noisy[:, 0], x_noisy[:, 1]
            need_convert = ((ch0 < 0).float().mean() > 0.05) or (ch1.abs().max() > 3.6)
            if need_convert:
                # 看起來資料其實是 (real,imag)；自動轉
                noisy_real_TF, noisy_imag_TF = ch0, ch1
                mag_TF = torch.sqrt(noisy_real_TF**2 + noisy_imag_TF**2 + 1e-12)
                pha_TF = torch.atan2(noisy_imag_TF, noisy_real_TF)
            else:
                mag_TF, pha_TF = ch0, ch1
            x_input = torch.stack([mag_TF, pha_TF], dim=1)              # [B, 2, T, F]
        else:
            raise NotImplementedError("Only 'cpx' and 'pha' are supported in forward_spectrogram.")

        # Step 3: 編碼器 → Mamba Block(s)
        x_feat = self.dense_encoder(x_input)                             # [B, C, T, F/2]
        feat_list = []                                                   # ★ KD: 收集每層 TF-Bi-Mamba 輸出
        for i in range(self.num_tscblocks):
            x_feat = self.TSMamba[i](x_feat)                             # [B, C, T, F/2]
            feat_list.append(x_feat)
        self.last_feats = feat_list                                      # ★ KD: 供 feature-based distillation 使用

        # Step 4: 幅度遮罩預測與應用（mask_out 要是 [B,1,T,F]）
        mask_out = self.mask_decoder(x_feat)                             # [B, 1, T, F]
        if mask_out.shape[1:] != (1, mag_TF.shape[1], mag_TF.shape[2]):
            raise RuntimeError(
                f"mask_out shape mismatch: got {tuple(mask_out.shape)}, "
                f"expected [B,1,T,F]=[B,1,{mag_TF.shape[1]},{mag_TF.shape[2]}]"
            )
        mag_g_TF = (mag_TF.unsqueeze(1) * mask_out).squeeze(1)           # [B, T, F]

        # 轉為 [B, F, T]
        mag_g_FT    = mag_g_TF.permute(0, 2, 1).contiguous()             # [B, F, T]
        mag_noisy_FT= mag_TF.permute(0, 2, 1).contiguous()               # [B, F, T]
        pha_FT      = pha_TF.permute(0, 2, 1).contiguous()               # [B, F, T]
        assert mag_g_FT.shape[-2] == self.h.n_fft//2+1, "F dim mismatch for ISTFT"

        # ---------- 熵正則  (放在取到 mask_out 之後即可)  ---------------
        mask_entropy = torch.tensor(0.0, device=x_feat.device)
        if self.use_entropy:
            mask_entropy = -(mask_out * (mask_out + 1e-8).log() +
                            (1 - mask_out) * (1 - mask_out + 1e-8).log()).mean()


        # Step 5: com_g 與 pha_g
        if self.fea == 'cpx':
            com_d = self.complex_decoder(x_feat).permute(0, 3, 2, 1).contiguous()  # [B, F, T, 2]
            com_g = torch.stack([
                mag_g_FT * torch.cos(pha_FT),
                mag_g_FT * torch.sin(pha_FT)
            ], dim=-1)                                                              # [B, F, T, 2]
            com_g = com_g + com_d
            pha_g = torch.atan2(com_g[..., 1], com_g[..., 0])                       # [B, F, T]
        else:  # 'pha'
            # phase ablation:
            #   if use_phase: predict phase
            #   else: reuse noisy phase (pha_FT)
            if self.use_phase and (self.phase_decoder is not None):
                pha_g_TF = self.phase_decoder(x_feat).squeeze(1)                    # [B, T, F]
                pha_g = pha_g_TF.permute(0, 2, 1).contiguous()                      # [B, F, T]
            else:
                pha_g = pha_FT                           # [B, F, T]
            com_g = torch.stack([
                mag_g_FT * torch.cos(pha_g),
                mag_g_FT * torch.sin(pha_g)
            ], dim=-1)                                                              # [B, F, T, 2]

        # === 準備 clean 的頻譜（[B, F, T, 2]） ===
        if self.fea == 'cpx':
            cr_TF, ci_TF = x_clean[:,0], x_clean[:,1]
            c_mag_FT = torch.sqrt(cr_TF**2 + ci_TF**2 + 1e-12).permute(0,2,1).contiguous()
            c_pha_FT = torch.atan2(ci_TF, cr_TF).permute(0,2,1).contiguous()
        else:
            c_mag_FT = x_clean[:,0].permute(0,2,1).contiguous()
            c_pha_FT = x_clean[:,1].permute(0,2,1).contiguous()
        clean_com = torch.stack([c_mag_FT*torch.cos(c_pha_FT),
                                 c_mag_FT*torch.sin(c_pha_FT)], dim=-1)             # [B, F, T, 2]

        # === norm_factor（選擇性；和原始 forward 對齊） ===
        norm_factor = None
        if self.norm in ('1', '2') and any(k in self.loss_fn for k in ('time', 'com', 'con')):
            noisy_wav_ref = istft32(mag_noisy_FT, pha_FT)                            # [B, T]
            if self.norm == '1':
                norm_factor = torch.sqrt(noisy_wav_ref.shape[-1] /
                                         torch.sum(noisy_wav_ref ** 2.0, -1, keepdim=True))   # [B,1]
            else:
                norm_factor = 1 / noisy_wav_ref.abs().max(-1, keepdim=True)[0]               # [B,1]

        # Step 6: Loss ── 權重（報告/優化都不再用 H）
        w_cfg = self.h.get('loss_weights', {})
        W_TIME = w_cfg.get('time', 0.20)
        W_COM  = w_cfg.get('com' , 0.60)
        W_CON  = w_cfg.get('con' , 0.20)
        W_MR   = w_cfg.get('mr'  , 0.08)
        # W_H    = w_cfg.get('H'   , 1e-2)   # ← 不再使用於 objective

        # 是否需要波形（time 或 MR 任一需要就做 ISTFT）
        need_wav = ('time' in self.loss_fn) or self.use_mrstft
        wav_g = wav_c = None
        if need_wav:
            wav_g = istft32(mag_g_FT, pha_g)     # [B,T]
            wav_c = istft32(c_mag_FT, c_pha_FT)  # [B,T]

        # 先設零，方便記錄
        loss_time = torch.tensor(0., device=x_feat.device)
        loss_com  = torch.tensor(0., device=x_feat.device)
        loss_con  = torch.tensor(0., device=x_feat.device)
        loss_mr   = torch.tensor(0., device=x_feat.device)

        # Time
        if 'time' in self.loss_fn:
            loss_time = F.l1_loss(wav_c, wav_g, reduction='none')
            if norm_factor is not None:
                loss_time = loss_time / norm_factor.squeeze(-1)
            loss_time = loss_time.mean()

        # MR-STFT（由 ablation 控）
        if self.use_mrstft:
            if wav_g is None or wav_c is None:
                wav_g = istft32(mag_g_FT, pha_g)
                wav_c = istft32(c_mag_FT, c_pha_FT)
            loss_mr = _mrstft_loss(wav_g, wav_c, self.h)

        # Complex
        if 'com' in self.loss_fn:
            loss_com = F.mse_loss(clean_com, com_g, reduction='none') * 2.0
            if norm_factor is not None:
                loss_com = loss_com / norm_factor.unsqueeze(-1)
            loss_com = loss_com.mean()

        # Consistency
        if 'con' in self.loss_fn:
            com_con = self.complex_decoder(x_feat).permute(0, 3, 2, 1)
            loss_con = F.mse_loss(com_g, com_con, reduction='none') * 2.0
            if norm_factor is not None:
                loss_con = loss_con / norm_factor.unsqueeze(-1)
            loss_con = loss_con.mean()

        # 報告/優化的核心 loss（★ 不含 entropy）
        if not self.use_mrstft:
            W_MR = 0.0
        loss_core = (W_TIME * loss_time +
                     W_COM  * loss_com  +
                     W_CON  * loss_con  +
                     W_MR   * loss_mr)

        # entropy 僅計算與記錄，不參與 backward/選最佳
        mask_entropy = torch.tensor(0.0, device=x_feat.device)
        if self.use_entropy:
            mask_entropy = -(mask_out * (mask_out + 1e-8).log() +
                             (1 - mask_out) * (1 - mask_out + 1e-8).log()).mean()

        # === 對外約定 ===
        # 1) 回傳的 loss 就是核心 loss（★無 entropy）
        # 2) last_report_loss = loss_core（pipeline 用它算 train/val/scheduler）
        # 3) last_losses 仍記錄 entropy 給觀察
        self.last_losses = {
            "time":    loss_time.detach(),
            "com":     loss_com.detach(),
            "con":     loss_con.detach(),
            "mr":      loss_mr.detach(),
            "entropy": mask_entropy.detach(),
        }
        self.last_report_loss = loss_core.detach()

        # 額外保留最後一個 batch 的波形（若後續要算 MF）
        if need_wav:
            self.last_wavs = {"pred": wav_g.detach().float(),
                              "clean": wav_c.detach().float()}
        else:
            self.last_wavs = None

        # ★ KD: 暫存 response-level 輸出（mask / mag / phase / complex spectrogram）
        #   注意：這裡刻意「不」 detach，讓呼叫端自己決定 teacher 要 detach、student 要保留 graph
        self.last_outputs = {
            "mask":     mask_out,     # [B, 1, T, F]
            "mag_g_FT": mag_g_FT,     # [B, F, T]
            "pha_g":    pha_g,        # [B, F, T]
            "com_g":    com_g,        # [B, F, T, 2]
        }

        return loss_core   # ← ★ 只回核心四項，用它做 backward


# Backward-compatible alias for legacy training scripts/checkpoints.
MECGE = SSEMGNet
