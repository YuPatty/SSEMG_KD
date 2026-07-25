# ────────────────────────────────────────────────────
# measure_student_rtf.py
# 量測 StudentSSEMGNet（我們真正訓練出來的模型，不是簡化版 TSConv）的
# 推論延遲與 Real-Time Factor (RTF)。
#
# 用法：
#   python measure_student_rtf.py --device cpu
#   python measure_student_rtf.py --device cuda
# ────────────────────────────────────────────────────
import os, sys, argparse
import time
import yaml
import torch

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.StudentNet import StudentSSEMGNet   # noqa: E402  # 不依賴 mamba_ssm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--student_config", default="config/config_student_crossarch.yaml")
    p.add_argument("--student_weights", default=None,
                    help="指定 checkpoint 路徑（例如 model_weight/student_production_annealed_60.pth）。"
                         "不指定的話用隨機初始化權重測速度（速度不受權重影響，但如果你想同時"
                         "確認 checkpoint 能不能正常載入，建議指定這個參數）")
    p.add_argument("--num_runs", type=int, default=100)
    p.add_argument("--num_warmup", type=int, default=10)
    p.add_argument("--n_frames", type=int, default=79)
    p.add_argument("--signal_length_sec", type=float, default=None,
                    help="不指定的話會依 config 的 sampling_rate/hop_size/n_frames 動態算出"
                         "正確的訊號長度（本專案設定下約 10.112 秒，不是 1 秒——之前這裡"
                         "預設寫死 1.0 是錯的，已修正為動態計算，避免同樣的錯誤再發生）")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"🚀 開始測量 StudentSSEMGNet 在 {args.device.upper()} 上的 RTF...")

    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    if args.signal_length_sec is None:
        sr = float(student_cfg['model'].get('sampling_rate', 1000))
        hop_size = float(student_cfg['model']['hop_size'])
        args.signal_length_sec = (args.n_frames * hop_size) / sr
        print(f"訊號長度未指定，依 config 動態算出：{args.signal_length_sec:.4f} 秒 "
              f"[= {args.n_frames}幀 × hop_size{hop_size} / sr{sr}]")

    model = StudentSSEMGNet(student_cfg).to(device)
    model.eval()

    if args.student_weights:
        state = torch.load(args.student_weights, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        # strict=True：只要有任何 key 對不上就會直接報錯，不會靜默跳過，
        # 這正是要避免上次「架構對不上但沒發現」問題的關鍵
        print(f"✅ 已載入 checkpoint：{args.student_weights}（strict=True 驗證通過）")
    else:
        print("⚠️ 未指定 --student_weights，使用隨機初始化權重（僅測速度，不代表真實模型的輸出品質）")

    # 輸入 shape 依 config 的 input_channels/input_freq_bins/input_time_bins
    B = 1
    C = student_cfg.get('input_channels', 2)
    Fbin = student_cfg.get('input_freq_bins', student_cfg['model']['n_fft'] // 2 + 1)
    T = student_cfg.get('input_time_bins', 79)
    print(f"輸入 shape: [B={B}, C={C}, F={Fbin}, T={T}]")

    clean = torch.randn(B, C, Fbin, T).to(device)
    noisy = torch.randn(B, C, Fbin, T).to(device)

    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(clean, noisy)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(args.num_runs):
            _ = model(clean, noisy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    total_time = time.time() - start_time

    avg_latency_sec = total_time / args.num_runs
    avg_latency_ms = avg_latency_sec * 1000
    rtf = avg_latency_sec / args.signal_length_sec

    n_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 50)
    print(f"📊 StudentSSEMGNet 效能報告（{args.device.upper()}）")
    print("=" * 50)
    print(f"參數量: {n_params:,}（應為 392,075，若不同代表 config 或架構跟先前不一致）")
    print(f"單次推論延遲 (Latency): {avg_latency_ms:.2f} ms")
    print(f"訊號物理長度 (Signal): {args.signal_length_sec:.2f} sec")
    print(f"⭐ Real-Time Factor (RTF): {rtf:.5f}")
    print("=" * 50)
    if rtf < 1.0:
        print("✅ 結論：達成即時運算要求 (RTF < 1)")
    else:
        print("❌ 結論：無法達成即時運算")


if __name__ == "__main__":
    main()
