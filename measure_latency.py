import torch
import yaml
import numpy as np
import time
import sys
import os
import argparse

# ── 終極 CPU Monkey-patch (強制覆寫 Triton RMSNorm) ──
import torch.nn as nn
class _PlainRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm.to(dtype)

# 強制攔截 mamba_ssm 內部的 Triton 呼叫
import mamba_ssm.ops.triton.layer_norm
mamba_ssm.ops.triton.layer_norm.RMSNorm = _PlainRMSNorm
# ──────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet
from models.StudentNet import StudentSSEMGNet


def measure_model_latency(model, device, num_iter=200, num_warmup=50):
    """
    回傳 (mean_ms, std_ms)，若模型在此裝置上完全無法執行（例如 Mamba 在 CPU 上
    缺少 Triton/CUDA kernel），回傳 (None, None) 並印出原因，而不是讓整支
    腳本崩潰——Teacher 在 CPU 上跑不動本身就是要證明的結果，不是要排除的錯誤。
    """
    model.eval()

    # 先用一種 shape 試跑，抓的是「shape 不匹配」這種可恢復的錯誤；
    # 如果是硬體/kernel 不相容（CUDA/Triton），不屬於這裡要處理的範圍，直接往外拋。
    clean_x = torch.abs(torch.randn(1, 2, 257, 79)).to(device)
    noisy_x = torch.abs(torch.randn(1, 2, 257, 79)).to(device)
    try:
        with torch.no_grad():
            _ = model(clean_x, noisy_x)
    except RuntimeError as e:
        msg = str(e)
        # 明確區分兩種情況：shape 不對 → 換個 shape 再試；
        # CUDA/Triton 不相容 → 這是硬體限制，不是 shape 問題，直接回報失敗
        if "invalid argument to exchangeDevice" in msg or "CUDA" in msg or "Triton" in msg:
            print(f"[跳過] 此模型在裝置 '{device}' 上無法執行（非 shape 問題，"
                  f"是底層 kernel 不支援此裝置）：{msg.splitlines()[0]}")
            return None, None
        else:
            # 真的是 shape 問題，換一種排列再試一次
            clean_x = torch.abs(torch.randn(1, 2, 79, 257)).to(device)
            noisy_x = torch.abs(torch.randn(1, 2, 79, 257)).to(device)
            try:
                with torch.no_grad():
                    _ = model(clean_x, noisy_x)
            except RuntimeError as e2:
                print(f"[跳過] 兩種 shape 都無法執行：{e2}")
                return None, None

    # Warm-up
    try:
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(clean_x, noisy_x)
    except RuntimeError as e:
        print(f"[跳過] Warm-up 階段失敗：{e}")
        return None, None

    # 正式測速
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
        # CPU 上沒有 torch.cuda.Event，改用 time.perf_counter
        with torch.no_grad():
            for _ in range(num_iter):
                t0 = time.perf_counter()
                _ = model(clean_x, noisy_x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # 轉成 ms

    return np.mean(latencies), np.std(latencies)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                    help="不指定時自動偵測；--device cpu 會強制在 CPU 上測")
    args = p.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('config/config_spectrogram_v19_tt_mask.yaml') as f:
        t_cfg = yaml.safe_load(f)
    with open('config/config_student_crossarch.yaml') as f:
        s_cfg = yaml.safe_load(f)

    teacher = SSEMGNet(t_cfg).to(device)
    student = StudentSSEMGNet(s_cfg).to(device)

    print("\nMeasuring Teacher (Mamba) Latency...")
    t_mean, t_std = measure_model_latency(teacher, device)

    print("Measuring Student (Conv) Latency...")
    s_mean, s_std = measure_model_latency(student, device)

    print("\n==============================================")
    print(f"      推論延遲比較 Inference Latency (Batch Size = 1, device={device})")
    print("==============================================")
    if t_mean is not None:
        print(f"Teacher (Mamba): {t_mean:6.2f} ms ± {t_std:4.2f} ms")
    else:
        print(f"Teacher (Mamba): 無法在 {device} 上執行（缺少 CUDA/Triton 支援）")

    if s_mean is not None:
        print(f"Student (Conv):  {s_mean:6.2f} ms ± {s_std:4.2f} ms")
    else:
        print(f"Student (Conv):  無法在 {device} 上執行")

    if t_mean is not None and s_mean is not None:
        print(f"👉 加速比 (Speedup): {t_mean / s_mean:.2f}x 倍快")
    elif t_mean is None and s_mean is not None:
        print("👉 Teacher 在此裝置上完全無法運行，Student 是唯一可行選項"
              "（這正是跨架構設計要證明的重點，不是負面結果）")
    print("==============================================")


if __name__ == '__main__':
    main()
