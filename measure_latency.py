import torch
import yaml
import numpy as np
import time
import sys
import os
import argparse

# ── 這裡不再需要任何額外的 RMSNorm/causal_conv1d/selective_scan monkey-patch ──
# 之前這裡有一段「無條件」覆寫 mamba_ssm.ops.triton.layer_norm.RMSNorm 的 patch，
# 問題是它不分裝置一律套用，導致 GPU 環境也被誤套用 CPU fallback 版本，
# 汙染了 GPU 的測量結果（這是造成上次 GPU 數字跟先前報告對不起來的原因）。
# 現在 models/SSEMGNet.py 內部已經有「依裝置自動判斷」的正確版本
# （torch.cuda.is_available() 為 True 時用原本的 Triton 加速路徑，
#   為 False 時才切換成 CPU 相容的 fallback），不需要在這裡重複、
# 更不能用無條件版本蓋掉它。

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet
from models.StudentNet import StudentSSEMGNet


def get_signal_duration_ms(cfg, n_frames=79):
    """
    依 config 動態計算 n_frames 幀頻譜對應的實際訊號長度（毫秒），
    不要寫死成固定秒數——之前寫死 1.0 秒是錯的，正確算法：
        訊號長度(秒) = (幀數 × hop_size) / sampling_rate
    在本專案設定下（sr=1000Hz, hop_size=128, n_frames=79）約為 10.112 秒，
    不是 1 秒，兩者相差 10 倍，RTF 算出來會差 10 倍，不可以直接假設。
    """
    sr = float(cfg['model'].get('sampling_rate', 1000))
    hop_size = float(cfg['model']['hop_size'])
    duration_sec = (n_frames * hop_size) / sr
    return duration_sec * 1000.0


def measure_model_latency(model, device, num_iter=200, num_warmup=50):
    """
    回傳 (mean_ms, std_ms)，若模型在此裝置上完全無法執行，回傳 (None, None)
    並印出原因，而不是讓整支腳本崩潰。
    """
    model.eval()

    clean_x = torch.abs(torch.randn(1, 2, 257, 79)).to(device)
    noisy_x = torch.abs(torch.randn(1, 2, 257, 79)).to(device)
    try:
        with torch.no_grad():
            _ = model(clean_x, noisy_x)
    except RuntimeError as e:
        msg = str(e)
        if "invalid argument to exchangeDevice" in msg or "is_cuda" in msg or "CUDA" in msg or "Triton" in msg:
            print(f"[跳過] 此模型在裝置 '{device}' 上無法執行（非 shape 問題，"
                  f"是底層 kernel 不支援此裝置）：{msg.splitlines()[0]}")
            return None, None
        else:
            clean_x = torch.abs(torch.randn(1, 2, 79, 257)).to(device)
            noisy_x = torch.abs(torch.randn(1, 2, 79, 257)).to(device)
            try:
                with torch.no_grad():
                    _ = model(clean_x, noisy_x)
            except RuntimeError as e2:
                print(f"[跳過] 兩種 shape 都無法執行：{e2}")
                return None, None

    try:
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(clean_x, noisy_x)
    except RuntimeError as e:
        print(f"[跳過] Warm-up 階段失敗：{e}")
        return None, None

    latencies = []
    if device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(num_iter):
                starter.record()
                _ = model(clean_x, noisy_x)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(num_iter):
                t0 = time.perf_counter()
                _ = model(clean_x, noisy_x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    return np.mean(latencies), np.std(latencies)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                    help="不指定時自動偵測；--device cpu 會強制在 CPU 上測")
    p.add_argument('--teacher_config', default='config/config_spectrogram_v19_tt_mask.yaml')
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml')
    p.add_argument('--n_frames', type=int, default=79)
    args = p.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.teacher_config) as f:
        t_cfg = yaml.safe_load(f)
    with open(args.student_config) as f:
        s_cfg = yaml.safe_load(f)

    signal_duration_ms = get_signal_duration_ms(t_cfg, args.n_frames)
    print(f"訊號物理長度: {signal_duration_ms:.2f} ms ({signal_duration_ms/1000:.4f} 秒)"
          f"  [= {args.n_frames}幀 × hop_size{t_cfg['model']['hop_size']} / "
          f"sr{t_cfg['model']['sampling_rate']}]\n")

    teacher = SSEMGNet(t_cfg).to(device)
    student = StudentSSEMGNet(s_cfg).to(device)

    print("Measuring Teacher (Mamba) Latency...")
    t_mean, t_std = measure_model_latency(teacher, device)

    print("Measuring Student (Conv) Latency...")
    s_mean, s_std = measure_model_latency(student, device)

    print("\n==============================================")
    print(f"      推論延遲比較 Inference Latency (Batch Size = 1, device={device})")
    print("==============================================")
    if t_mean is not None:
        rtf_t = t_mean / signal_duration_ms
        print(f"Teacher (Mamba): {t_mean:8.2f} ms ± {t_std:5.2f} ms   RTF = {rtf_t:.4f}"
              f"  {'✅' if rtf_t < 1.0 else '❌'}")
    else:
        print(f"Teacher (Mamba): 無法在 {device} 上執行")

    if s_mean is not None:
        rtf_s = s_mean / signal_duration_ms
        print(f"Student (Conv):  {s_mean:8.2f} ms ± {s_std:5.2f} ms   RTF = {rtf_s:.4f}"
              f"  {'✅' if rtf_s < 1.0 else '❌'}")
    else:
        print(f"Student (Conv):  無法在 {device} 上執行")

    if t_mean is not None and s_mean is not None:
        print(f"👉 加速比 (Speedup): {t_mean / s_mean:.2f}x 倍快")
    elif t_mean is None and s_mean is not None:
        print("👉 Teacher 在此裝置上完全無法運行，Student 是唯一可行選項")
    print("==============================================")


if __name__ == '__main__':
    main()
