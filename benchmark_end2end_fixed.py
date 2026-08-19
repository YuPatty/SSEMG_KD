import torch
import torch.nn as nn
import yaml
import math
import time
import sys
import os
import argparse
from datetime import datetime

# ── 修正 CPU thread oversubscription ──
# len(os.sched_getaffinity(0)) 讀的是這個 process 實際被 cgroup/cpuset 分配到
# 的核心數（等同 `nproc` 看到的數字），跟 lscpu / /proc/cpuinfo 看到的「底層
# 實體主機總核心數」是兩回事。PyTorch 的 intra-op thread pool 如果沒有明確
# 設定，很容易照後者去開執行緒（例如在 nproc=4 但 lscpu 顯示 36 核心的機器
# 上，可能開出接近 36 個執行緒），導致大量執行緒擠在少數幾顆真正可用的核心
# 上互搶，是先前多次量測出現「同一份 config 在不同 pod 上延遲差幾十倍」、
# 「離群值連續好幾個 iteration 一起出現」的可能根因之一。這裡在任何 torch
# 運算之前，就把執行緒數明確釘死在這個 process 實際可用的核心數。
try:
    _n_visible_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # 非 Linux 平台沒有 sched_getaffinity，退回 os.cpu_count()
    _n_visible_cpus = os.cpu_count() or 1
torch.set_num_threads(_n_visible_cpus)
torch.set_num_interop_threads(1)
print(f"[info] 偵測到本 process 實際可用核心數 (os.sched_getaffinity) = {_n_visible_cpus}，"
      f"已將 torch.set_num_threads() 釘死在這個數字，避免執行緒 oversubscription")

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet, set_model_device_mode
from models.StudentNet import StudentSSEMGNet


# ── 確保純推論與 End-to-End 量測的 Wrapper ──
class MECGE_InferenceWrapper(nn.Module):
    def __init__(self, m, h):
        super().__init__()
        self.m = m
        self.h = h
        self.register_buffer('window', torch.hann_window(h['win_size']))

    def forward(self, wav):
        wav = wav.squeeze(1)
        X = torch.stft(wav, n_fft=self.h['n_fft'], hop_length=self.h['hop_size'],
                        win_length=self.h['win_size'], window=self.window,
                        center=True, pad_mode='reflect', return_complex=True)
        mag, pha = X.abs(), X.angle()
        mag_comp = torch.pow(mag, self.h['compress_factor'])

        spec = torch.stack([mag_comp, pha], dim=1)
        x_noisy = spec.permute(0, 1, 3, 2).contiguous()
        x_input = torch.stack([x_noisy[:, 0], x_noisy[:, 1]], dim=1)

        x_feat = self.m.dense_encoder(x_input)

        if hasattr(self.m, 'TSConv'):
            blocks = self.m.TSConv
        elif hasattr(self.m, 'TSMamba'):
            blocks = self.m.TSMamba
        elif hasattr(self.m, 'TSMambaBlock'):
            blocks = self.m.TSMambaBlock
        else:
            blocks = [m for k, m in self.m.named_children() if isinstance(m, nn.ModuleList)][0]

        for blk in blocks:
            x_feat = blk(x_feat)

        mask_out = self.m.mask_decoder(x_feat)
        mag_g_TF = (x_noisy[:, 0].unsqueeze(1) * mask_out).squeeze(1)
        mag_g_FT = mag_g_TF.permute(0, 2, 1).contiguous()

        if getattr(self.m, 'phase_decoder', None) is not None:
            pha_g = self.m.phase_decoder(x_feat).squeeze(1).permute(0, 2, 1).contiguous()
        else:
            pha_g = x_noisy[:, 1].permute(0, 2, 1).contiguous()

        mag_uncomp = torch.pow(mag_g_FT, (1.0 / self.h['compress_factor']))
        com = torch.complex(mag_uncomp * torch.cos(pha_g), mag_uncomp * torch.sin(pha_g))
        wav_out = torch.istft(com, self.h['n_fft'], hop_length=self.h['hop_size'],
                               win_length=self.h['win_size'], window=self.window, center=True)
        return wav_out.unsqueeze(1)


def get_signal_duration_sec(cfg, n_frames=79):
    """
    依 config 動態計算 n_frames 幀頻譜對應的實際訊號長度（秒）：
        訊號長度(秒) = (幀數 × hop_size) / sampling_rate
    這樣 --seconds 不用自己手算、也不會跟訓練時實際用的語音長度對不上
    （之前 --seconds 預設寫死 2.0，跟你的資料實際 ~10 秒的長度差了 5 倍，
    量出來的延遲/RTF 會嚴重失真）。
    注意：假設 STFT 沒有額外 padding；若用 center=True，真實輸入波形長度
    會比這個公式多約一個 window_length，算出來的 RTF 會偏樂觀一點點。
    """
    sr = float(cfg['model'].get('sampling_rate', 1000))
    hop_size = float(cfg['model']['hop_size'])
    return (n_frames * hop_size) / sr


def _percentile(sorted_samples, pct):
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * pct
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_samples[int(k)]
    return sorted_samples[f] * (c - k) + sorted_samples[c] * (k - f)


def count_flops(wrapped_model, input_tensor, tag=""):
    """
    用 fvcore 算端到端（含 STFT/ISTFT）的 GFLOPs。
    ⚠️ 已知限制：fvcore 只認得標準的 nn.Linear/Conv 等 op，Mamba 的
    selective_scan_fn / causal_conv1d_fn，以及 torch.stft/istft 底層用到
    的 FFT 相關 op，都不是它認得的標準 op，會被「靜默跳過（記成 0 FLOPs）」。
    這裡會把被跳過的 op 印出來，讓你知道有沒有漏算；GFLOPs 這欄只能同架構
    內比較相對大小，不能拿來跟非 Mamba 架構的模型做絕對算力比較。

    FLOPs 一律在 CPU 上算（fvcore 用一般 Python hook，不需要 GPU）。
    呼叫前務必先用 set_model_device_mode(wrapped_model.m, "cpu") 把模型
    的 fast-path 明確設成 CPU 模式，不然如果模型是在有 GPU 的機器上建構
    的，內部仍然是 Triton RMSNorm + use_fast_path=True，Triton kernel
    沒辦法被 fvcore 追蹤，會直接崩潰。
    """
    from fvcore.nn import FlopCountAnalysis
    # ⚠️ 修正：必須先 .to("cpu") 把參數實際搬過去，再呼叫 set_model_device_mode，
    # 否則呼叫當下模型參數還在原本的裝置上，fast-path 旗標會跟參數實際所在
    # 裝置不同步，導致後續 forward 時 tensor device 不一致而失敗。
    wrapped_model = wrapped_model.to("cpu").eval()
    set_model_device_mode(wrapped_model.m, "cpu")
    fca = FlopCountAnalysis(wrapped_model, input_tensor.to("cpu"))
    fca.unsupported_ops_warnings(True)
    fca.uncalled_modules_warnings(False)
    flops = fca.total()
    unsupported = fca.unsupported_ops()
    if unsupported:
        print(f"[warning] {tag}: fvcore 未計入下列 op 的 FLOPs（GFLOPs 會被低估）: "
              f"{dict(unsupported)}")
    return flops


def time_inference(model, device, seconds, fs, num_iter=200, num_warmup=30, time_budget_sec=30.0,
                    dump_samples_path=None, tag=""):
    """
    回傳 (p50_ms, p10_ms, p90_ms)。

    ⚠️ CPU 上的 Mamba（尤其 Bi-Mamba，序列長度 ≈ 秒數×fs/hop_size，可能
    上千甚至上萬）會走純 Python 的 selective_scan_ref，一次 forward 可能
    要好幾秒甚至更久。固定跑 num_iter 次會導致單一模型卡上數小時，所以
    這裡是「時間預算到了就停，但至少收集 3 筆」，避免無限期卡住。

    若指定 dump_samples_path，會把「每一次 iteration」的原始樣本
    (wall-clock 時間戳 + 耗時 ms，未排序、依實際執行順序) 以 CSV 附加寫入
    該檔案，欄位為 tag,device,iter_idx,wall_clock_iso,elapsed_ms。
    這樣事後如果某次量測出現離群值 (例如 p90 遠大於 p50)，可以對照
    wall_clock 時間戳去查當下 nvidia-smi/htop 的系統負載，找出是哪一次
    iteration 被什麼東西卡住，而不是只看聚合後的 p10/p50/p90。
    """
    model.eval()
    # ⚠️ 修正：同樣必須先 .to(device) 把參數實際搬過去，再呼叫
    # set_model_device_mode，順序反過來會導致 fast-path 旗標跟參數實際
    # 所在裝置不同步（例如 count_flops() 已經把同一個 wrapped 物件搬到
    # CPU 並設成 CPU 模式，這裡若在搬到 cuda 之前就呼叫 set_model_device_mode，
    # 呼叫當下模型參數還在 CPU 上，容易在 forward 時噴 tensor device 不一致
    # 的 RuntimeError，被下面的 except 吃掉，最終在表格裡顯示 N/A）。
    model = model.to(device)
    set_model_device_mode(model.m, device)

    inputs = torch.randn(1, 1, int(seconds * fs), device=device)

    try:
        with torch.no_grad():
            warmup_deadline = time.perf_counter() + min(time_budget_sec, 15.0)
            warmup_count = 0
            while warmup_count < num_warmup and time.perf_counter() < warmup_deadline:
                _ = model(inputs)
                warmup_count += 1
            if device.type == 'cuda':
                torch.cuda.synchronize()
        if warmup_count < num_warmup:
            print(f"[note] {device}: warm-up 在時間預算內只跑了 {warmup_count}/{num_warmup} 次"
                  f"（此模型在此裝置上單次 forward 較慢）")
    except RuntimeError as e:
        print(f"[跳過] Warm-up 階段失敗或不支援 ({device}): {e}")
        return None, None, None

    samples_ms = []
    raw_records = []  # (iter_idx, wall_clock_iso, elapsed_ms) — 依實際執行順序，不排序
    deadline = time.perf_counter() + time_budget_sec

    with torch.no_grad():
        if device.type == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            while len(samples_ms) < num_iter:
                wall_ts = datetime.now().isoformat(timespec='milliseconds')
                starter.record()
                _ = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)
                samples_ms.append(elapsed)
                raw_records.append((len(samples_ms) - 1, wall_ts, elapsed))
                if len(samples_ms) >= 3 and time.perf_counter() > deadline:
                    break
        else:
            while len(samples_ms) < num_iter:
                wall_ts = datetime.now().isoformat(timespec='milliseconds')
                t0 = time.perf_counter()
                _ = model(inputs)
                t1 = time.perf_counter()
                elapsed = (t1 - t0) * 1000.0
                samples_ms.append(elapsed)
                raw_records.append((len(samples_ms) - 1, wall_ts, elapsed))
                if len(samples_ms) >= 3 and time.perf_counter() > deadline:
                    break

    if len(samples_ms) < num_iter:
        print(f"[note] {device}: 在時間預算 {time_budget_sec:.0f}s 內只收集到 "
              f"{len(samples_ms)}/{num_iter} 筆樣本（p10/p90 區間會比較不精確，但不會卡住）")

    if dump_samples_path:
        file_exists = os.path.isfile(dump_samples_path)
        with open(dump_samples_path, "a", newline="") as f:
            if not file_exists:
                f.write("tag,device,iter_idx,wall_clock_iso,elapsed_ms\n")
            for iter_idx, wall_ts, elapsed in raw_records:
                f.write(f"{tag},{device},{iter_idx},{wall_ts},{elapsed:.4f}\n")
        # 順便標出這次量測裡的離群值，直接印在終端機，不用等事後開檔案查
        elapsed_sorted = sorted(e for _, _, e in raw_records)
        p50_tmp = _percentile(elapsed_sorted, 0.5)
        outliers = [(idx, ts, e) for idx, ts, e in raw_records if e > p50_tmp * 5]
        if outliers:
            print(f"[note] {tag}@{device}: 發現 {len(outliers)} 筆離群樣本 (> 5x p50={p50_tmp:.2f}ms)：")
            for idx, ts, e in outliers[:10]:
                print(f"         iter={idx}  wall_clock={ts}  elapsed={e:.2f}ms")
        print(f"[info] 原始樣本已寫入 {dump_samples_path}")

    samples_ms.sort()
    return _percentile(samples_ms, 0.5), _percentile(samples_ms, 0.1), _percentile(samples_ms, 0.9)


def human(n):
    if n >= 1e6:
        return f"{n/1e6:.3f} M"
    if n >= 1e3:
        return f"{n/1e3:.3f} K"
    return str(n)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--devices', nargs='+', choices=['cpu', 'cuda'], default=['cpu', 'cuda'])
    p.add_argument('--teacher_config', default='config/config_spectrogram_v19_tt_mask.yaml')
    p.add_argument('--student_configs', nargs='+', required=True)
    p.add_argument('--n_frames', type=int, default=79,
                    help="用來自動推算訊號長度的幀數，跟訓練時實際用的長度保持一致")
    p.add_argument('--seconds', type=float, default=None,
                    help="不指定時，會依各 config 的 hop_size/sampling_rate 跟 --n_frames "
                         "自動算出真實訊號長度；指定的話則所有模型都用同一個秒數")
    p.add_argument('--num_iter', type=int, default=200)
    p.add_argument('--num_warmup', type=int, default=30)
    p.add_argument('--time_budget_sec', type=float, default=30.0)
    p.add_argument('--skip_teacher', action='store_true')
    p.add_argument('--dump_samples', type=str, default=None,
                    help="指定路徑後，會把每一次 iteration 的原始樣本 "
                         "(wall-clock 時間戳 + 耗時) 依實際執行順序附加寫入這個 CSV，"
                         "方便事後對照離群值出現在什麼時間點，用來排查是不是被背景 "
                         "process 干擾。例如 --dump_samples logs/latency_samples.csv")
    args = p.parse_args()

    device_list = [torch.device(d) for d in args.devices if d == 'cpu' or (d == 'cuda' and torch.cuda.is_available())]
    print(f"Devices: {[str(d) for d in device_list]}\n")

    # ── 記錄硬體/環境資訊，方便事後追溯「這批數字是在哪台機器、多少核心下量的」──
    # 不同 pod/節點的 CPU 配額可能差異很大 (nproc vs lscpu 看到的實體核心數常常
    # 對不上)，這幾行不影響量測，純粹是把量測當下的環境釘進 log/報告，避免之後
    # 拿不同硬體規格量到的數字互相比較卻不自知。
    try:
        import subprocess
        hostname = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    except Exception:
        hostname = "unknown"
    print(f"[env] hostname = {hostname}")
    print(f"[env] os.sched_getaffinity 可用核心數 = {_n_visible_cpus}  (nproc 應該回報同樣的數字)")
    try:
        loadavg = os.getloadavg()
        print(f"[env] loadavg (1/5/15 min) = {loadavg}")
    except (OSError, AttributeError):
        pass
    print()

    rows = []  # (name, params, flops, duration_sec, {device: (p50,p10,p90)})

    def bench_one(cfg_path, is_teacher):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        duration_sec = args.seconds if args.seconds is not None else get_signal_duration_sec(cfg, args.n_frames)
        fs = cfg['model'].get('sampling_rate', 1000)
        name = "Teacher (Mamba)" if is_teacher else os.path.splitext(os.path.basename(cfg_path))[0]
        print(f"[{name}] 訊號長度: {duration_sec:.3f}s (fs={fs})")

        model = SSEMGNet(cfg) if is_teacher else StudentSSEMGNet(cfg)
        wrapped = MECGE_InferenceWrapper(model, cfg['model'])
        params = count_params(model)

        dummy_input = torch.randn(1, 1, int(duration_sec * fs))
        print(f"Measuring {name} FLOPs...")
        try:
            flops = count_flops(wrapped, dummy_input, tag=name)
        except Exception as e:
            print(f"[warning] {name}: fvcore FLOPs 追蹤失敗 ({e})")
            flops = None

        print(f"Measuring {name} End-to-End latency...")
        latencies = {
            d: time_inference(wrapped, d, duration_sec, fs, args.num_iter, args.num_warmup,
                               args.time_budget_sec, dump_samples_path=args.dump_samples, tag=name)
            for d in device_list
        }
        rows.append((name, params, flops, duration_sec, latencies))
        del model, wrapped

    if not args.skip_teacher:
        bench_one(args.teacher_config, is_teacher=True)

    for cfg_path in args.student_configs:
        bench_one(cfg_path, is_teacher=False)

    # ── 印出總表 ──
    col_w = 26
    header = f"{'Model':<24}{'Params':>12}{'GFLOPs':>10}"
    for dn in device_list:
        header += f"{str(dn) + ' E2E (ms)':>{col_w}}{'RTF-' + str(dn):>12}"
    print("\n" + "=" * len(header))
    print("      端到端推論效能總表 (含 STFT/ISTFT，訊號長度依各 config 自動計算)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, params, flops, duration_sec, latencies in rows:
        gflops_s = f"{(flops * 2) / 1e9:.3f}" if flops is not None else "N/A"
        duration_ms = duration_sec * 1000.0
        line = f"{name:<24}{human(params):>12}{gflops_s:>10}"
        for device in device_list:
            p50, p10, p90 = latencies[device]
            if p50 is None:
                line += f"{'N/A':>{col_w}}{'N/A':>12}"
            else:
                lat_str = f"{p50:.2f} [{p10:.1f}-{p90:.1f}]"
                line += f"{lat_str:>{col_w}}{p50 / duration_ms:>12.4f}"
        print(line)
    print("=" * len(header))
    print("GFLOPs 註：fvcore 未計入 Mamba selective_scan/causal_conv1d 與 STFT/ISTFT 底層 FFT")
    print("      op 的計算量，只能同架構內比較相對大小，不能跨架構比較絕對算力。")
    print("RTF < 1.0 代表推論比訊號播放時間快（可即時處理）。")


if __name__ == '__main__':
    main()