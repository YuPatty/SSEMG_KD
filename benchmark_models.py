"""
benchmark_models.py
──────────────────────────────────────────────────────────────
統一量測 SDEMG / MSEMG / FCN / MECG-E (Teacher & Student)
的參數量、FLOPs(MACs)、GPU / CPU 推論時間。

【重要修正】
1. 完全移除全域 Monkeypatch，統一透過 set_fast_path 切換 Mamba fast_path。
2. 測量 CPU / FLOPs 時將 use_fast_path 設為 False（避免 CUDA 核心依賴崩潰）。
3. 測量 GPU 時將 use_fast_path 設為 True（確保量測到真實優化後的 CUDA Kernel 速度）。
4. MECG-E 採用端到端 (Waveform In -> Waveform Out) 推論封裝。
──────────────────────────────────────────────────────────────
"""
import argparse
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import types

ROOT = Path(__file__).resolve().parent

# ★ 固定 CPU thread 數，避免跟其他 process 搶核心造成 thread contention。
# 預設抓「實體核心數的一半」，可用環境變數 BENCH_NUM_THREADS 覆蓋。
# 固定下來之後，同一台機器上多次重跑的數字才有可比性；否則 PyTorch 預設
# 吃滿所有可見核心，一旦跟別的 process 競爭，延遲會暴增且完全不可預測。
_default_threads = max(1, (os.cpu_count() or 4) // 2)
_num_threads = int(os.environ.get("BENCH_NUM_THREADS", _default_threads))
torch.set_num_threads(_num_threads)
print(f"[info] torch.set_num_threads({_num_threads})  "
      f"(os.cpu_count()={os.cpu_count()}, 可用 BENCH_NUM_THREADS 環境變數覆蓋)")

# ★ 關鍵修正：一定要在任何 mamba_ssm patch 之前，就把 SDEMG/MSEMG 底下
# 本地(vendored)的 mamba_ssm 資料夾加進 sys.path。
# 原本的版本把這個 path insert 動作放在 bench_msemg()/bench_student()/bench_teacher()
# 函式『裡面』，但檔案最上面這些 patch 是模組載入當下就執行——結果就是
# patch 執行的時候 Python 根本還找不到 mamba_ssm，全部靜默失敗(只印警告訊息)，
# 之後 bench_msemg() 真正 import 到的是完全『沒被 patch 過』的原始版本，
# CPU/FLOPs 量測時一樣會如先前一樣直接因為 CUDA-only 運算而崩潰。
for _sub in ["SDEMG", "MSEMG"]:
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ==========================================
# 修補 selective_scan_fn 以支援 CPU / fvcore FLOPs Tracing
# ==========================================
try:
    import mamba_ssm.ops.selective_scan_interface as ssi
    if hasattr(ssi, "selective_scan_fn") and hasattr(ssi, "selective_scan_ref"):
        _orig_selective_scan_fn = ssi.selective_scan_fn

        def _safe_selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
            if not u.is_cuda:
                # 張量在 CPU 上時，自動切換至純 PyTorch 實作 (selective_scan_ref)
                return ssi.selective_scan_ref(
                    u, delta, A, B, C, D=D, z=z, 
                    delta_bias=delta_bias, delta_softplus=delta_softplus, 
                    return_last_state=return_last_state
                )
            # 張量在 CUDA 上時，維持原生 CUDA Kernel 呼叫
            return _orig_selective_scan_fn(
                u, delta, A, B, C, D=D, z=z, 
                delta_bias=delta_bias, delta_softplus=delta_softplus, 
                return_last_state=return_last_state
            )

        # 覆蓋介面層與所有已載入模組
        ssi.selective_scan_fn = _safe_selective_scan_fn
        for mod_name, mod in list(sys.modules.items()):
            if mod and hasattr(mod, "selective_scan_fn"):
                setattr(mod, "selective_scan_fn", _safe_selective_scan_fn)
                
        print("[info] Successfully patched selective_scan_fn for CPU & FLOPs benchmarking.")
except Exception as e:
    print(f"[warning] Could not patch selective_scan_fn: {e}")

# ==========================================
# 修補 mamba_ssm 缺少的/不相容 CPU 的 Triton RMSNorm 模組
# (強制替換為純 PyTorch 實作，徹底避開 Triton JIT 在 CPU 下的編譯崩潰)
# ==========================================
class PyTorchRMSNorm(nn.Module):
    """RMSNorm dispatcher(支援 CPU / GPU 與殘差項傳遞)。

    重要：Mamba 的 Block 在『模型建構當下』就把 self.norm = RMSNorm(...) 蓋好、
    變成一個固定的 nn.Module instance，不是每次 forward 才查一次的函式。
    所以「事後把模組層級的 RMSNorm 這個 class 換掉」對已經蓋好的 instance
    完全沒有作用（跟 causal_conv1d_fn 那種每次呼叫才查一次的 free function 不同）。

    正確做法：讓這個 class 本身在 forward() 裡動態判斷「現在該不該用真正的
    Triton fused kernel」，這樣不管 instance 是什麼時候蓋出來的，切換
    set_fast_path() 都會在下一次 forward 立刻生效。
    """
    _fast_path_enabled = False  # 由 set_fast_path() 統一控制的全域開關
    _real_rms_norm_fn = None    # 真正的 Triton rms_norm_fn，第一次被覆蓋時備份

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = None

    def _naive_forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual.float() if residual_in_fp32 else x + residual
            residual_out = x
        else:
            residual_out = None
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out = (x_f32 * torch.rsqrt(variance + self.eps)).to(input_dtype) * self.weight
        if self.bias is not None:
            out = out + self.bias
        if prenorm:
            return out, residual_out
        return out

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if (
            PyTorchRMSNorm._fast_path_enabled
            and PyTorchRMSNorm._real_rms_norm_fn is not None
            and x.is_cuda
        ):
            return PyTorchRMSNorm._real_rms_norm_fn(
                x, self.weight, self.bias, residual=residual, eps=self.eps,
                prenorm=prenorm, residual_in_fp32=residual_in_fp32,
            )
        return self._naive_forward(x, residual=residual, prenorm=prenorm,
                                    residual_in_fp32=residual_in_fp32)

# 強制覆蓋帶底線 (SSEMGNet) 與無底線 (MSEMG) 兩種路徑
# 把每個模組原本的 rms_norm_fn（真正的 Triton fused kernel）備份起來，
# 交給 PyTorchRMSNorm._real_rms_norm_fn 在 forward() 時動態呼叫，
# 再把 RMSNorm 這個 class 換成 PyTorchRMSNorm（class 本身只需要換一次，
# 真正決定用哪條運算路徑的邏輯在 PyTorchRMSNorm.forward() 裡動態判斷）。
for mod_name in ["mamba_ssm.ops.triton.layer_norm", "mamba_ssm.ops.triton.layernorm"]:
    try:
        mod = __import__(mod_name, fromlist=["RMSNorm", "rms_norm_fn"])
    except ImportError:
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod

    real_fn = getattr(mod, "rms_norm_fn", None)
    if real_fn is not None and PyTorchRMSNorm._real_rms_norm_fn is None:
        PyTorchRMSNorm._real_rms_norm_fn = real_fn

    mod.RMSNorm = PyTorchRMSNorm

print("[info] RMSNorm 已改為動態 dispatcher：set_fast_path(True) 時走真正的 Triton kernel，"
      "False 時走純 PyTorch 版，且對已建構的 layer instance 同樣即時生效。")

# ==========================================
# 修補 mamba_ssm 新舊版本 Block 匯入與參數簽名差異
# ==========================================
import inspect

try:
    import mamba_ssm.modules.mamba_simple as mamba_simple
    
    # 嘗試抓取新版 mixer_seq_simple 的 Block 或舊版 block.Block
    try:
        from mamba_ssm.models.mixer_seq_simple import Block as OriginalBlock
    except ImportError:
        from mamba_ssm.modules.block import Block as OriginalBlock

    # 動態檢查底層 OriginalBlock.__init__ 是否接受 mlp_cls 參數
    sig = inspect.signature(OriginalBlock.__init__)
    has_mlp_cls = "mlp_cls" in sig.parameters

    class PatchedBlock(OriginalBlock):
        """動態相容新舊版 mamba_ssm 的 Block 參數簽名"""
        def __init__(self, *args, **kwargs):
            if has_mlp_cls:
                # 新版 Block: 若缺少 mlp_cls 則自動補上 nn.Identity
                if "mlp_cls" not in kwargs:
                    if len(args) == 3:
                        dim, mixer_cls, norm_cls = args
                        args = (dim, mixer_cls)
                        kwargs["mlp_cls"] = nn.Identity
                        kwargs["norm_cls"] = norm_cls
                    else:
                        kwargs["mlp_cls"] = nn.Identity
            else:
                # 舊版 Block: 不支援 mlp_cls 參數，不論是否傳入一律強制安全剔除
                kwargs.pop("mlp_cls", None)

            super().__init__(*args, **kwargs)

    mamba_simple.Block = PatchedBlock
    
    # 全域同步覆蓋所有已載入與未來可能載入的 mamba_ssm 模組層級 Block
    for mod_name in ["mamba_ssm.modules.mamba_simple", "mamba_ssm.modules.block", "mamba_ssm.models.mixer_seq_simple"]:
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], "Block", PatchedBlock)

    print("[info] Successfully patched Block with backwards-compatible signature.")
except Exception as e:
    print(f"[warning] Could not patch mamba_ssm Block: {e}")
    
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 修補 causal_conv1d 以支援 CPU / fvcore FLOPs Tracing
# ==========================================
try:
    import causal_conv1d
    import causal_conv1d_interface
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    _orig_causal_conv1d_fn = causal_conv1d_interface.causal_conv1d_fn

    def _safe_causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, 
                               return_final_states=False, final_states_out=None, activation=None):
        if not x.is_cuda:
            # 張量在 CPU 上時，自動切換至純 PyTorch 實作 (causal_conv1d_ref)
            return causal_conv1d_ref(
                x, weight, bias=bias, 
                initial_states=initial_states, 
                return_final_states=return_final_states, 
                final_states_out=final_states_out, 
                activation=activation
            )
        # 張量在 CUDA 上時，維持原生 CUDA Kernel 呼叫
        return _orig_causal_conv1d_fn(
            x, weight, bias=bias, 
            seq_idx=seq_idx, 
            initial_states=initial_states, 
            return_final_states=return_final_states, 
            final_states_out=final_states_out, 
            activation=activation
        )

    # 全域覆蓋介面
    causal_conv1d_interface.causal_conv1d_fn = _safe_causal_conv1d_fn
    causal_conv1d.causal_conv1d_fn = _safe_causal_conv1d_fn
    print("[info] Successfully patched causal_conv1d for CPU & FLOPs benchmarking.")
except Exception as e:
    print(f"[warning] Could not patch causal_conv1d: {e}")


def human(n):
    if n >= 1e9: return f"{n/1e9:.3f} G"
    if n >= 1e6: return f"{n/1e6:.3f} M"
    if n >= 1e3: return f"{n/1e3:.3f} K"
    return str(n)

def set_fast_path(model, enable: bool):
    """動態開啟或關閉 Mamba 模組的 CUDA fast_path、causal_conv1d 優化、以及 RMSNorm 的
    Triton fused kernel。三者現在統一由這個函式控制，enable=False 時全部退回可在 CPU
    上安全執行、且 fvcore 可追蹤的慢速版本；enable=True 時全部換回真正優化過的版本，
    用來量測『真實』的 GPU 推論速度。"""
    # 0. RMSNorm dispatcher 的全域開關（對所有已建構的 layer instance 立即生效）
    PyTorchRMSNorm._fast_path_enabled = enable

    # 1. 切換模型中所有 Module 的 use_fast_path 屬性
    for m in model.modules():
        if hasattr(m, "use_fast_path"):
            m.use_fast_path = enable

    # 2. 控制已載入的 Mamba 模組內部的 causal_conv1d_fn
    # - CPU / FLOPs 模式 (enable=False): 設為 None，強迫 Mamba 自動退回原生 PyTorch nn.Conv1d (不依賴 CUDA)
    # - GPU 模式 (enable=True): 恢復原本的 CUDA 核心函式，獲得極速 C++ 加速
    for mod_name, mod in list(sys.modules.items()):
        if mod and ("mamba_simple" in mod_name or "mamba2" in mod_name):
            # 首次遇到時，先將原始的 CUDA 函式備份起來
            if not hasattr(mod, "_orig_causal_conv1d_fn"):
                orig = getattr(mod, "causal_conv1d_fn", None)
                setattr(mod, "_orig_causal_conv1d_fn", orig)
            
            orig_fn = getattr(mod, "_orig_causal_conv1d_fn")
            if orig_fn is not None:
                setattr(mod, "causal_conv1d_fn", orig_fn if enable else None)

    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_flops(model, inputs):
    from fvcore.nn import FlopCountAnalysis
    fca = FlopCountAnalysis(model, inputs)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total()

def time_inference(model, make_inputs, device, repeats, warmup=5, verbose_tag=None):
    """量測單次 forward 的延遲。

    改動重點（原本用『整圈總時間 / repeats』取平均，只要中間被別的 process
    搶走 CPU 一次，單一離群值就會直接污染平均值，而且完全看不出來）：
      1. 每一次 forward 各自計時，回傳全部樣本的『中位數』而非平均值 —— 中位數
         對離群值不敏感，偶發的一次資源競爭不會大幅拉動結果。
      2. 額外印出 min/median/max，讓污染在當下就看得到，不用等事後起疑才去查。
      3. CPU 計時改用 time.perf_counter()（單調、高解析度），取代 time.time()
         （會受系統時鐘調整影響、解析度較低）。
    """
    model = model.to(device)
    model.eval()
    inputs = make_inputs(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)

        samples_ms = []
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                start_event.record()
                model(*inputs)
                end_event.record()
                torch.cuda.synchronize()
                samples_ms.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(*inputs)
                t1 = time.perf_counter()
                samples_ms.append((t1 - t0) * 1000.0)

    med = statistics.median(samples_ms)
    if verbose_tag and len(samples_ms) > 1:
        lo, hi = min(samples_ms), max(samples_ms)
        stdev = statistics.pstdev(samples_ms)
        # max/median 比值大代表量測期間有離群的慢跑（很可能是外部資源競爭），
        # 這時候印出警告，提醒這次數字可能不能信。
        ratio = hi / med if med > 0 else float("inf")
        flag = "  ⚠️ 疑似受外部干擾（max/median 比值過高，建議重跑）" if ratio > 3 else ""
        print(f"[timing] {verbose_tag} [{device}] median={med:.3f}ms "
              f"min={lo:.3f}ms max={hi:.3f}ms std={stdev:.3f}ms{flag}")
    return med

# ==========================================
# MECG-E 專用：端到端純推論 Wrapper (Waveform In -> Waveform Out)
# ==========================================
class MECGE_InferenceWrapper(nn.Module):
    def __init__(self, m, h):
        super().__init__()
        self.m = m
        self.h = h
        self.register_buffer('window', torch.hann_window(h['win_size']))
        
    def forward(self, wav):
        # 1. STFT
        wav = wav.squeeze(1)
        X = torch.stft(wav, n_fft=self.h['n_fft'], hop_length=self.h['hop_size'], 
                       win_length=self.h['win_size'], window=self.window, 
                       center=True, pad_mode='reflect', return_complex=True)
        mag, pha = X.abs(), X.angle()
        mag_comp = torch.pow(mag, self.h['compress_factor'])
        
        # 2. 轉換維度以符合模型輸入
        spec = torch.stack([mag_comp, pha], dim=1)
        x_noisy = spec.permute(0, 1, 3, 2).contiguous() # [B, 2, T, F]
        x_input = torch.stack([x_noisy[:, 0], x_noisy[:, 1]], dim=1)
        
        # 3. 核心網路正向傳播 (跳過 Loss 運算)
        x_feat = self.m.dense_encoder(x_input)
        
        if hasattr(self.m, 'TSConv'): blocks = self.m.TSConv
        elif hasattr(self.m, 'TSMamba'): blocks = self.m.TSMamba
        elif hasattr(self.m, 'TSMambaBlock'): blocks = self.m.TSMambaBlock
        else: blocks = [m for k, m in self.m.named_children() if isinstance(m, nn.ModuleList)][0]
        
        for blk in blocks: 
            x_feat = blk(x_feat)
            
        mask_out = self.m.mask_decoder(x_feat)
        mag_g_TF = (x_noisy[:, 0].unsqueeze(1) * mask_out).squeeze(1)
        mag_g_FT = mag_g_TF.permute(0, 2, 1).contiguous()
        
        if getattr(self.m, 'phase_decoder', None) is not None:
            pha_g = self.m.phase_decoder(x_feat).squeeze(1).permute(0, 2, 1).contiguous()
        else:
            pha_g = x_noisy[:, 1].permute(0, 2, 1).contiguous()
            
        # 4. ISTFT
        mag_uncomp = torch.pow(mag_g_FT, (1.0 / self.h['compress_factor']))
        com = torch.complex(mag_uncomp * torch.cos(pha_g), mag_uncomp * torch.sin(pha_g))
        wav_out = torch.istft(com, self.h['n_fft'], hop_length=self.h['hop_size'], 
                              win_length=self.h['win_size'], window=self.window, center=True)
        return wav_out.unsqueeze(1)


# ==========================================
# 基準測試函式
# ==========================================
def bench_sdemg(seconds, fs, gpu_repeats, cpu_repeats, feats=64):
    sys.path.insert(0, str(ROOT / "SDEMG"))
    from deep_filter_model import ConditionalModel

    # 動態讀取實際推論步數，比照 main.py 的解析順序：
    #   denoise() 內部 self.denoise_timesteps = default(denoise_timesteps, self.num_timesteps)
    #   其中 self.num_timesteps 來自 GaussianDiffusion1D(timesteps=exp_cfg['sampling_steps'])
    # 也就是說：cfg 的 denoise_timesteps 若非空值就優先採用，否則退回 sampling_steps。
    # 寫死 50 在 cfg 沒被改動前是對的，但一旦你調整超參數就會悄悄算錯，故改成動態讀取。
    sampling_steps = 50
    cfg_path = ROOT / "SDEMG" / "cfg" / "default.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            exp_cfg = yaml.safe_load(f)
        denoise_ts = exp_cfg.get("denoise_timesteps")
        sampling_steps = int(denoise_ts) if denoise_ts else int(exp_cfg.get("sampling_steps", 50))
    else:
        print(f"[warning] 找不到 {cfg_path}，sampling_steps 使用預設值 50，"
              "請確認這跟你實際訓練用的設定一致。")

    L = int(seconds * fs)
    model = ConditionalModel(feats=feats)
    params = count_params(model)
    print(f"[info] SDEMG: feats={feats} -> params={params:,} "
          f"({'對齊 MSEMG 論文 Table II 報告數字(1,233,857)' if feats == 64 else '不是文獻報告的版本，注意標註'})")
    print(f"[info] SDEMG: 完整推論步數 = {sampling_steps}（讀自 cfg/default.yaml）")

    def make_inputs(device):
        x = torch.randn(1, 1, L, device=device)
        cond = torch.randn(1, 1, L, device=device)
        noise_scale = torch.tensor([0.5], device=device)
        return x, noise_scale, cond

    flops = count_flops(model.to("cpu"), make_inputs("cpu"))
    cpu_ms = time_inference(model, make_inputs, "cpu", cpu_repeats, verbose_tag="SDEMG")
    gpu_ms = time_inference(model, make_inputs, "cuda", gpu_repeats, verbose_tag="SDEMG") if torch.cuda.is_available() else None

    return {
        "name": "SDEMG", "params": params, "flops_single_step": flops, "flops_full": flops * sampling_steps,
        "sampling_steps": sampling_steps, "cpu_ms_single_step": cpu_ms, "cpu_ms_full": cpu_ms * sampling_steps,
        "gpu_ms_single_step": gpu_ms, "gpu_ms_full": (gpu_ms * sampling_steps) if gpu_ms else None
    }

def bench_msemg(seconds, fs, gpu_repeats, cpu_repeats):
    sys.path.insert(0, str(ROOT / "MSEMG"))
    from model import EMGMAMBA

    L = int(seconds * fs)
    model = EMGMAMBA(in_channels=64, feats=64, n_layer=1)
    params = count_params(model)

    def make_inputs(device): return (torch.randn(1, 1, L, device=device),)

    # 1. 關閉 fast_path 以利 CPU / FLOPs 計算
    set_fast_path(model, False)
    flops = count_flops(model.to("cpu"), make_inputs("cpu"))
    cpu_ms = time_inference(model, make_inputs, "cpu", cpu_repeats, verbose_tag="MSEMG")

    # 2. 開啟 fast_path 以進行真實的 GPU 速度量測
    set_fast_path(model, True)
    gpu_ms = time_inference(model, make_inputs, "cuda", gpu_repeats, verbose_tag="MSEMG") if torch.cuda.is_available() else None

    return {
        "name": "MSEMG", "params": params, "flops_single_step": flops, "flops_full": flops,
        "sampling_steps": 1, "cpu_ms_single_step": cpu_ms, "cpu_ms_full": cpu_ms,
        "gpu_ms_single_step": gpu_ms, "gpu_ms_full": gpu_ms
    }

def bench_fcn(seconds, fs, gpu_repeats, cpu_repeats):
    import importlib.util
    fcn_model_path = ROOT / "ECG-removal-from-sEMG-by-FCN" / "main" / "model" / "FCN.py"

    spec = importlib.util.spec_from_file_location("fcn_module", fcn_model_path)
    fcn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcn_module)
    model = getattr(fcn_module, "FCN_01" if hasattr(fcn_module, "FCN_01") else "FCN")()

    L = int(seconds * fs)
    params = count_params(model)

    # FCN 的輸入慣例不確定是 (B, L) 還是 (B, 1, L)，用小張量各試一次，
    # 用真的能跑通的那個形狀，而不是憑空假設，避免整個函式在這裡直接 crash。
    model.eval()
    with torch.no_grad():
        try:
            _ = model(torch.randn(1, L))
            input_ndim = 2
        except Exception as e2d:
            try:
                _ = model(torch.randn(1, 1, L))
                input_ndim = 3
            except Exception as e3d:
                raise RuntimeError(
                    f"FCN 模型輸入形狀既不吃 (1,{L}) 也不吃 (1,1,{L})，"
                    f"2D 錯誤: {e2d}；3D 錯誤: {e3d}；請自行確認 FCN.py 的 forward() 期待的輸入維度。"
                )
    print(f"[info] FCN: 自動偵測到輸入形狀為 {input_ndim}D")

    def make_inputs(device):
        if input_ndim == 2:
            return (torch.randn(1, L, device=device),)
        return (torch.randn(1, 1, L, device=device),)

    flops = count_flops(model.to("cpu"), make_inputs("cpu"))
    cpu_ms = time_inference(model, make_inputs, "cpu", cpu_repeats, verbose_tag="FCN")
    gpu_ms = time_inference(model, make_inputs, "cuda", gpu_repeats, verbose_tag="FCN") if torch.cuda.is_available() else None

    return {
        "name": "FCN", "params": params, "flops_single_step": flops, "flops_full": flops,
        "sampling_steps": 1, "cpu_ms_single_step": cpu_ms, "cpu_ms_full": cpu_ms,
        "gpu_ms_single_step": gpu_ms, "gpu_ms_full": gpu_ms
    }

def bench_student(seconds, fs, gpu_repeats, cpu_repeats):
    import importlib.util
    student_path = Path("/home/taes10056/SSEMG-Net/MECG-E/models/StudentNet.py")
    if not student_path.exists():
        student_path = ROOT.parent / "SSEMG-Net" / "MECG-E" / "models" / "StudentNet.py"
        
    spec = importlib.util.spec_from_file_location("student_mod", student_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    
    h = {'dense_channel': 32, 'n_fft': 512, 'hop_size': 128, 'win_size': 512, 'compress_factor': 0.5, 
         'fea': 'pha', 'norm': False, 'loss_fn': 'time+com+con', 'num_tscblocks': 2, 'edepth': 4, 
         'mdepth': 4, 'pdepth': 4, 'beta': 1.0, 'fmamba': True, 'use_gn': True, 'gn_groups': 8}
    config = {'model': h, 'ablation': {'use_mrstft': True, 'use_entropy': True, 'use_phase': True}}
    
    model = getattr(student_module, "StudentSSEMGNet")(config).eval()
    params = count_params(model)
    wrapped_model = MECGE_InferenceWrapper(model, h)

    def make_inputs(device): return (torch.randn(1, 1, int(seconds * fs), device=device),)

    # 1. 關閉 fast_path 以利 CPU / FLOPs 計算
    set_fast_path(model, False)
    flops = count_flops(wrapped_model.to("cpu"), make_inputs("cpu"))
    cpu_ms = time_inference(wrapped_model, make_inputs, "cpu", cpu_repeats, verbose_tag="Student")

    # 2. 開啟 fast_path 以進行真實的 GPU 速度量測
    set_fast_path(model, True)
    gpu_ms = time_inference(wrapped_model, make_inputs, "cuda", gpu_repeats, verbose_tag="Student") if torch.cuda.is_available() else None

    return {
        "name": "MECG-E (Student)", "params": params, "flops_single_step": flops, "flops_full": flops,
        "sampling_steps": 1, "cpu_ms_single_step": cpu_ms, "cpu_ms_full": cpu_ms,
        "gpu_ms_single_step": gpu_ms, "gpu_ms_full": gpu_ms
    }

def bench_teacher(seconds, fs, gpu_repeats, cpu_repeats):
    import importlib.util
    teacher_path = Path("/home/taes10056/SSEMG-Net/MECG-E/models/SSEMGNet.py")
    if not teacher_path.exists():
        teacher_path = ROOT.parent / "SSEMG-Net" / "MECG-E" / "models" / "SSEMGNet.py"
            
    spec = importlib.util.spec_from_file_location("teacher_mod", teacher_path)
    teacher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher_module)
    
    h = {'dense_channel': 64, 'n_fft': 512, 'hop_size': 128, 'win_size': 512, 'compress_factor': 0.5, 
         'fea': 'pha', 'norm': False, 'loss_fn': 'time+com+con', 'num_tscblocks': 4, 'edepth': 4, 
         'mdepth': 4, 'pdepth': 4, 'beta': 1.0, 'fmamba': True, 'use_gn': True, 'gn_groups': 8}
    config = {'model': h, 'ablation': {'use_mrstft': True, 'use_entropy': True, 'use_phase': True}}
    
    model = getattr(teacher_module, "SSEMGNet")(config).eval()
    params = count_params(model)
    wrapped_model = MECGE_InferenceWrapper(model, h)

    def make_inputs(device): return (torch.randn(1, 1, int(seconds * fs), device=device),)

    # 1. 關閉 fast_path 以利 CPU / FLOPs 計算
    set_fast_path(model, False)
    flops = count_flops(wrapped_model.to("cpu"), make_inputs("cpu"))
    cpu_ms = time_inference(wrapped_model, make_inputs, "cpu", cpu_repeats, verbose_tag="Teacher")

    # 2. 開啟 fast_path 以進行真實的 GPU 速度量測
    set_fast_path(model, True)
    gpu_ms = time_inference(wrapped_model, make_inputs, "cuda", gpu_repeats, verbose_tag="Teacher") if torch.cuda.is_available() else None

    return {
        "name": "MECG-E (Teacher)", "params": params, "flops_single_step": flops, "flops_full": flops,
        "sampling_steps": 1, "cpu_ms_single_step": cpu_ms, "cpu_ms_full": cpu_ms,
        "gpu_ms_single_step": gpu_ms, "gpu_ms_full": gpu_ms
    }

def print_table(rows):
    header = f"{'Method':<18}{'Params':>12}{'GFLOPS(full)':>14}{'GPU(ms)':>12}{'CPU(ms)':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        gpu = f"{r['gpu_ms_full']:.2f}" if r["gpu_ms_full"] is not None else "N/A(無GPU)"
        gflops_full = (r['flops_full'] * 2) / 1e9
        print(f"{r['name']:<18}{human(r['params']):>12}{gflops_full:>14.3f}{gpu:>12}{r['cpu_ms_full']:>12.2f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--fs", type=int, default=1000)
    p.add_argument("--gpu-repeats", type=int, default=50)
    # 10 次樣本取中位數容易被單一離群值撐大（只要 10 個裡有 1~2 個被外部干擾，
    # 中位數還是可能落在被污染的那一側）。30 次能讓中位數更穩健，跑起來也不會太久。
    p.add_argument("--cpu-repeats", type=int, default=30)
    p.add_argument("--cpu-threads", type=int, default=None,
                    help="覆蓋 torch.set_num_threads()；不指定則用 BENCH_NUM_THREADS 環境變數"
                         "或 os.cpu_count()//2 的預設值")
    p.add_argument("--models", nargs="+", default=["sdemg", "msemg", "fcn", "student", "teacher"])
    p.add_argument("--sdemg-feats", type=int, default=64, choices=[64, 128],
                    help="SDEMG 的 feats 設定：64 對齊 MSEMG 論文 Table II 報告的參數量(1,233,857)；"
                         "128 對齊 SDEMG repo 目前 main.py 訓練腳本的預設值(參數量會變成4,925,313，"
                         "與文獻報告數字不同，需自行在報告中註明)。")
    args = p.parse_args()

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        print(f"[info] torch.set_num_threads({args.cpu_threads})  (由 --cpu-threads 覆蓋)")

    print(f"[info] device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu only'}")
    print(f"[info] segment: {args.seconds}s @ {args.fs}Hz = {int(args.seconds*args.fs)} samples\n")

    rows = []
    if "sdemg" in args.models: print("[info] benchmarking SDEMG ..."); rows.append(bench_sdemg(args.seconds, args.fs, args.gpu_repeats, args.cpu_repeats, feats=args.sdemg_feats))
    if "msemg" in args.models: print("[info] benchmarking MSEMG ..."); rows.append(bench_msemg(args.seconds, args.fs, args.gpu_repeats, args.cpu_repeats))
    if "fcn" in args.models: print("[info] benchmarking FCN ..."); rows.append(bench_fcn(args.seconds, args.fs, args.gpu_repeats, args.cpu_repeats))
    if "teacher" in args.models: print("[info] benchmarking Teacher ..."); rows.append(bench_teacher(args.seconds, args.fs, args.gpu_repeats, args.cpu_repeats))
    if "student" in args.models: print("[info] benchmarking Student ..."); rows.append(bench_student(args.seconds, args.fs, args.gpu_repeats, args.cpu_repeats))

    print()
    print_table(rows)