# ────────────────────────────────────────────────────
# MECG-E/models/StudentNet.py
# 跨架構 Student：完全不 import mamba_ssm，可在沒有 CUDA/Triton 的環境跑。
#
# 設計原則：
#   - DenseEncoder / MaskDecoder / PhaseDecoder / ComplexDecoder 與 loss 計算
#     邏輯直接沿用 SSEMGNet.py（本來就是純 CNN，不依賴 Mamba，照抄無風險）
#   - 唯一替換的是中間 4 層 TSMambaBlock → TSConvBlock
#     用「深度可分離卷積（depthwise separable conv）」做時間軸/頻率軸建模，
#     對應 ULde-net 的架構設計精神（grouped conv + depthwise separable conv
#     取代原本較重的序列建模模組）
#   - forward_spectrogram() 的輸出介面（self.last_feats / self.last_outputs /
#     回傳 loss_core）跟 teacher 完全一致，distill_loss.py 不用改就能吃兩邊
# ────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2),
            int((kernel_size[1]*dilation[1] - dilation[1])/2))


class _LearnableSigmoidBase(nn.Module):
    def __init__(self, in_features: int, beta: float = 1.0, min_slope: float = 0.05):
        super().__init__()
        self.beta = beta
        self.min_slope = float(min_slope)
        self.raw = nn.Parameter(torch.zeros(1, in_features))

    def _positive_slope(self):
        return self.min_slope + F.softplus(self.raw)


class LearnableSigmoid_2d(_LearnableSigmoidBase):
    def forward(self, x):                          # x: [B, T, F]
        slope = self._positive_slope()
        return self.beta * torch.sigmoid(slope * x)


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device,
                                     dtype=com.dtype if com.is_floating_point() else torch.float32)
    wav = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size,
                       window=hann_window, center=center)
    return wav


# ── MR-STFT loss：直接複製一份，刻意「不」從 SSEMGNet.py import ──
#    因為 SSEMGNet.py 開頭有 `from mamba_ssm...`，只要 import 它就會
#    強制要求 CUDA/Triton，這樣 student 就沒有真正脫離 Mamba 依賴了。
_STFT_WINDOW_CACHE = {}

def _get_hann(win, device, dtype):
    key = (win, str(device), str(dtype))
    if key not in _STFT_WINDOW_CACHE:
        _STFT_WINDOW_CACHE[key] = torch.hann_window(win, device=device, dtype=dtype)
    return _STFT_WINDOW_CACHE[key]

def _stft_mag(wav, n_fft, hop, win, center=True, eps=1e-9):
    device, dtype = wav.device, wav.dtype
    window = _get_hann(win, device, dtype)
    X = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window,
                   center=center, pad_mode='reflect', normalized=False, return_complex=True)
    return torch.clamp(X.abs(), min=eps)

def _spectral_convergence(x_mag, y_mag):
    B = y_mag.size(0)
    num = torch.linalg.norm((y_mag - x_mag).reshape(B, -1), ord=2, dim=1)
    den = torch.linalg.norm(y_mag.reshape(B, -1), ord=2, dim=1) + 1e-12
    return (num / den).mean()

def _log_mag_l1(x_mag, y_mag, log_eps=1e-7):
    return (torch.log(x_mag + log_eps) - torch.log(y_mag + log_eps)).abs().mean()

def _mrstft_loss(wav_pred, wav_ref, cfg,
                  scales=((256, 64, 256), (512, 128, 512), (1024, 256, 1024))):
    device = wav_pred.device
    w_sc = float(cfg.get("w_sc", 1.0))
    w_log = float(cfg.get("w_log_mag", 1.0))
    log_eps = float(cfg.get("mrstft_log_eps", 1e-7))
    eps = float(cfg.get("mrstft_eps", 1e-9))
    total = torch.tensor(0.0, device=device)
    for n_fft, hop, win in scales:
        X_mag = _stft_mag(wav_pred, n_fft, hop, win, eps=eps)
        Y_mag = _stft_mag(wav_ref, n_fft, hop, win, eps=eps)
        l_sc = _spectral_convergence(X_mag, Y_mag) if w_sc != 0.0 else 0.0
        l_mag = _log_mag_l1(X_mag, Y_mag, log_eps) if w_log != 0.0 else 0.0
        total = total + (w_sc*l_sc + w_log*l_mag)
    return total / len(scales)


# ══════════════════════════════════════════════════════════
# 純 CNN 版 encoder/decoder（跟 SSEMGNet.py 裡的定義邏輯一致，
# 這裡重新宣告一份是為了讓 StudentNet.py 完全不 import mamba_ssm）
# ══════════════════════════════════════════════════════════
class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super().__init__()
        if h.get('use_gn', False):
            assert h['dense_channel'] % h.get('gn_groups', 8) == 0, \
                f"dense_channel={h['dense_channel']} 不能被 gn_groups 整除"
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(h['dense_channel']*(i+1), h['dense_channel'], kernel_size,
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                (nn.GroupNorm(h.get('gn_groups', 8), h['dense_channel'], affine=True)
                 if h.get('use_gn', False) else nn.InstanceNorm2d(h['dense_channel'], affine=True)),
                nn.PReLU(h['dense_channel'])
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
        super().__init__()
        gn_or_in = (nn.GroupNorm(h.get('gn_groups', 8), h['dense_channel'], affine=True)
                    if h.get('use_gn', False) else nn.InstanceNorm2d(h['dense_channel'], affine=True))
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h['dense_channel'], (1, 1)), gn_or_in, nn.PReLU(h['dense_channel'])
        )
        self.dense_block = DenseBlock(h, depth=h.get('edepth', 4))
        gn_or_in2 = (nn.GroupNorm(h.get('gn_groups', 8), h['dense_channel'], affine=True)
                    if h.get('use_gn', False) else nn.InstanceNorm2d(h['dense_channel'], affine=True))
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h['dense_channel'], h['dense_channel'], (1, 3), (1, 2)),
            gn_or_in2, nn.PReLU(h['dense_channel'])
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get('mdepth', 4))
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h['dense_channel'], h['dense_channel'], (1, 3), (1, 2)),
            nn.Conv2d(h['dense_channel'], out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        Fbins = h['n_fft'] // 2 + 1
        self.lsigmoid = LearnableSigmoid_2d(Fbins, beta=h.get('beta', 1.0))

        self.cut_bin = -1
        cut_hz = h.get('mask_hp_cut_hz', None)
        try:
            cut_hz = float(cut_hz) if cut_hz is not None else None
        except Exception:
            cut_hz = None
        if cut_hz is not None and cut_hz > 0:
            sr, nfft = float(h.get('sampling_rate', 1000)), int(h.get('n_fft', 512))
            df = sr / nfft
            cut_bin = int(np.floor(cut_hz / df + 1e-9))
            cut_bin = max(0, min(cut_bin, Fbins))
            low_gate = torch.zeros(1, 1, 1, Fbins)
            low_gate[..., :cut_bin] = 1.0
            self.register_buffer('low_gate', low_gate)
            self.cut_bin = cut_bin

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        m_tf = self.lsigmoid(x.squeeze(1)).unsqueeze(1)
        low_gate = getattr(self, 'low_gate', None)
        if low_gate is not None:
            m_tf = m_tf * low_gate + (1.0 - low_gate)
        return m_tf


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get('pdepth', 4))
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h['dense_channel'], h['dense_channel'], (1, 3), (1, 2)),
            nn.InstanceNorm2d(h['dense_channel'], affine=True), nn.PReLU(h['dense_channel'])
        )
        self.phase_conv_r = nn.Conv2d(h['dense_channel'], out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h['dense_channel'], out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


class ComplexDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get('pdepth', 4))
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h['dense_channel'], h['dense_channel'], (1, 3), (1, 2)),
            nn.InstanceNorm2d(h['dense_channel'], affine=True), nn.PReLU(h['dense_channel'])
        )
        self.phase_conv_r = nn.Conv2d(h['dense_channel'], out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h['dense_channel'], out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        return torch.cat((self.phase_conv_r(x), self.phase_conv_i(x)), dim=1)


# ══════════════════════════════════════════════════════════
# 核心替換：TSConvBlock 取代 TSMambaBlock
# → 對應 ULde-net 的 depthwise separable conv 設計
# ══════════════════════════════════════════════════════════
class DepthwiseSeparableConv1d(nn.Module):
    """分組卷積（depthwise）+ 1x1 卷積（pointwise），ULde-net 的核心壓縮技巧。"""
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = get_padding(kernel_size, dilation)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                                    dilation=dilation, groups=channels)   # groups=channels → depthwise
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.PReLU(channels)

    def forward(self, x):          # x: [B, C, L]
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class TSConvBlock(nn.Module):
    """
    Cross-architecture 替換 TSMambaBlock，I/O shape 保持一致 [B,C,T,F/2]，
    這樣 distill_loss.py 的 feature-based KD（1x1 conv 投影對齊）不用改。
    時間軸用兩層不同 dilation 的 depthwise separable conv 擴大感受野，
    近似 Mamba 的長距離依賴建模能力（雖然理論上限不同，但足以做 KD baseline）。
    """
    def __init__(self, h):
        super().__init__()
        C = h['dense_channel']
        self.time_conv1 = DepthwiseSeparableConv1d(C, kernel_size=5, dilation=1)
        self.time_conv2 = DepthwiseSeparableConv1d(C, kernel_size=5, dilation=4)
        self.use_fconv = h.get('fmamba', True)   # 沿用同名 config key，語意上代表「頻率軸也建模」
        if self.use_fconv:
            self.freq_conv = DepthwiseSeparableConv1d(C, kernel_size=5, dilation=1)

    def forward(self, x):   # [B,C,T,F/2]
        B, C, T, Fh = x.shape

        xt = x.permute(0, 3, 1, 2).reshape(B * Fh, C, T)     # [B*Fh, C, T]
        xt = self.time_conv1(xt)
        xt = self.time_conv2(xt)
        xt = xt.reshape(B, Fh, C, T).permute(0, 2, 3, 1)     # [B, C, T, Fh]

        if self.use_fconv:
            xf = xt.permute(0, 2, 1, 3).reshape(B * T, C, Fh)
            xf = self.freq_conv(xf)
            xt = xf.reshape(B, T, C, Fh).permute(0, 2, 1, 3)

        return xt   # [B, C, T, F/2]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# ══════════════════════════════════════════════════════════
# StudentSSEMGNet：介面與 SSEMGNet 一致（loss 計算邏輯照抄，
# 唯一差異是中間層用 TSConvBlock，且完全不碰 mamba_ssm）
# ══════════════════════════════════════════════════════════
class StudentSSEMGNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ab_cfg = config.get('ablation', {})
        self.use_mrstft = bool(ab_cfg.get('use_mrstft', True))
        self.use_entropy = bool(ab_cfg.get('use_entropy', True))
        self.use_phase = bool(ab_cfg.get('use_phase', True))

        h = config['model']
        self.fea = h.get('fea', 'pha')
        self.h = h
        self.norm = h.get('norm', False)
        self.loss_fn = h['loss_fn'].split('+') if isinstance(h['loss_fn'], str) else list(h['loss_fn'])
        self.num_tscblocks = h['num_tscblocks']

        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.TSConv = nn.ModuleList([TSConvBlock(h) for _ in range(h['num_tscblocks'])])
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.complex_decoder = ComplexDecoder(h, out_channel=1)

        if self.fea == 'pha' and self.use_phase:
            self.phase_decoder = PhaseDecoder(h, out_channel=1)
        else:
            self.phase_decoder = None

    def forward(self, clean_spec, noisy_spec):
        return self.forward_spectrogram(clean_spec, noisy_spec)

    def forward_spectrogram(self, clean_spec, noisy_spec):
        """跟 SSEMGNet.forward_spectrogram 邏輯完全一致，只是 TSMamba → TSConv。"""
        def istft32(mag_FT, pha_FT):
            return mag_pha_istft(mag_FT.float(), pha_FT.float(),
                                  n_fft=self.h['n_fft'], hop_size=self.h['hop_size'],
                                  win_size=self.h['win_size'], compress_factor=self.h['compress_factor'])

        x_noisy = noisy_spec.permute(0, 1, 3, 2).contiguous()
        x_clean = clean_spec.permute(0, 1, 3, 2).contiguous()

        if self.fea == 'cpx':
            noisy_real_TF, noisy_imag_TF = x_noisy[:, 0], x_noisy[:, 1]
            mag_TF = torch.sqrt(noisy_real_TF**2 + noisy_imag_TF**2 + 1e-12)
            pha_TF = torch.atan2(noisy_imag_TF, noisy_real_TF)
            x_input = x_noisy
        elif self.fea == 'pha':
            ch0, ch1 = x_noisy[:, 0], x_noisy[:, 1]
            need_convert = ((ch0 < 0).float().mean() > 0.05) or (ch1.abs().max() > 3.6)
            if need_convert:
                noisy_real_TF, noisy_imag_TF = ch0, ch1
                mag_TF = torch.sqrt(noisy_real_TF**2 + noisy_imag_TF**2 + 1e-12)
                pha_TF = torch.atan2(noisy_imag_TF, noisy_real_TF)
            else:
                mag_TF, pha_TF = ch0, ch1
            x_input = torch.stack([mag_TF, pha_TF], dim=1)
        else:
            raise NotImplementedError("Only 'cpx' and 'pha' are supported.")

        x_feat = self.dense_encoder(x_input)
        feat_list = []
        for i in range(self.num_tscblocks):
            x_feat = self.TSConv[i](x_feat)
            feat_list.append(x_feat)
        self.last_feats = feat_list

        mask_out = self.mask_decoder(x_feat)
        mag_g_TF = (mag_TF.unsqueeze(1) * mask_out).squeeze(1)
        mag_g_FT = mag_g_TF.permute(0, 2, 1).contiguous()
        mag_noisy_FT = mag_TF.permute(0, 2, 1).contiguous()
        pha_FT = pha_TF.permute(0, 2, 1).contiguous()

        if self.fea == 'cpx':
            com_d = self.complex_decoder(x_feat).permute(0, 3, 2, 1).contiguous()
            com_g = torch.stack([mag_g_FT*torch.cos(pha_FT), mag_g_FT*torch.sin(pha_FT)], dim=-1) + com_d
            pha_g = torch.atan2(com_g[..., 1], com_g[..., 0])
        else:
            if self.use_phase and self.phase_decoder is not None:
                pha_g_TF = self.phase_decoder(x_feat).squeeze(1)
                pha_g = pha_g_TF.permute(0, 2, 1).contiguous()
            else:
                pha_g = pha_FT
            com_g = torch.stack([mag_g_FT*torch.cos(pha_g), mag_g_FT*torch.sin(pha_g)], dim=-1)

        if self.fea == 'cpx':
            cr_TF, ci_TF = x_clean[:, 0], x_clean[:, 1]
            c_mag_FT = torch.sqrt(cr_TF**2 + ci_TF**2 + 1e-12).permute(0, 2, 1).contiguous()
            c_pha_FT = torch.atan2(ci_TF, cr_TF).permute(0, 2, 1).contiguous()
        else:
            c_mag_FT = x_clean[:, 0].permute(0, 2, 1).contiguous()
            c_pha_FT = x_clean[:, 1].permute(0, 2, 1).contiguous()
        clean_com = torch.stack([c_mag_FT*torch.cos(c_pha_FT), c_mag_FT*torch.sin(c_pha_FT)], dim=-1)

        norm_factor = None
        if self.norm in ('1', '2') and any(k in self.loss_fn for k in ('time', 'com', 'con')):
            noisy_wav_ref = istft32(mag_noisy_FT, pha_FT)
            if self.norm == '1':
                norm_factor = torch.sqrt(noisy_wav_ref.shape[-1] / torch.sum(noisy_wav_ref**2.0, -1, keepdim=True))
            else:
                norm_factor = 1 / noisy_wav_ref.abs().max(-1, keepdim=True)[0]

        w_cfg = self.h.get('loss_weights', {})
        W_TIME, W_COM = w_cfg.get('time', 0.20), w_cfg.get('com', 0.60)
        W_CON, W_MR = w_cfg.get('con', 0.20), w_cfg.get('mr', 0.08)

        need_wav = ('time' in self.loss_fn) or self.use_mrstft
        wav_g = wav_c = None
        if need_wav:
            wav_g = istft32(mag_g_FT, pha_g)
            wav_c = istft32(c_mag_FT, c_pha_FT)

        loss_time = torch.tensor(0., device=x_feat.device)
        loss_com = torch.tensor(0., device=x_feat.device)
        loss_con = torch.tensor(0., device=x_feat.device)
        loss_mr = torch.tensor(0., device=x_feat.device)

        if 'time' in self.loss_fn:
            loss_time = F.l1_loss(wav_c, wav_g, reduction='none')
            if norm_factor is not None:
                loss_time = loss_time / norm_factor.squeeze(-1)
            loss_time = loss_time.mean()

        if self.use_mrstft:
            loss_mr = _mrstft_loss(wav_g, wav_c, self.h)

        if 'com' in self.loss_fn:
            loss_com = F.mse_loss(clean_com, com_g, reduction='none') * 2.0
            if norm_factor is not None:
                loss_com = loss_com / norm_factor.unsqueeze(-1)
            loss_com = loss_com.mean()

        if 'con' in self.loss_fn:
            com_con = self.complex_decoder(x_feat).permute(0, 3, 2, 1)
            loss_con = F.mse_loss(com_g, com_con, reduction='none') * 2.0
            if norm_factor is not None:
                loss_con = loss_con / norm_factor.unsqueeze(-1)
            loss_con = loss_con.mean()

        if not self.use_mrstft:
            W_MR = 0.0
        loss_core = W_TIME*loss_time + W_COM*loss_com + W_CON*loss_con + W_MR*loss_mr

        mask_entropy = torch.tensor(0.0, device=x_feat.device)
        if self.use_entropy:
            mask_entropy = -(mask_out*(mask_out+1e-8).log() + (1-mask_out)*(1-mask_out+1e-8).log()).mean()

        self.last_losses = {"time": loss_time.detach(), "com": loss_com.detach(),
                            "con": loss_con.detach(), "mr": loss_mr.detach(),
                            "entropy": mask_entropy.detach()}
        self.last_report_loss = loss_core.detach()
        self.last_wavs = ({"pred": wav_g.detach().float(), "clean": wav_c.detach().float()}
                          if need_wav else None)
        self.last_outputs = {"mask": mask_out, "mag_g_FT": mag_g_FT, "pha_g": pha_g, "com_g": com_g}

        return loss_core
