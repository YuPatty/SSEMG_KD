import torch
import torch.nn as nn
import time
import argparse

# 直接定義與您專案完全一致的 TSConv 學生模型結構，徹底擺脫 import 路徑問題
class TSConv(nn.Module):
    def __init__(self, in_channels=2, num_layers=4, out_channels=2):
        super(TSConv, self).__init__()
        layers = []
        curr_ch = in_channels
        hidden_ch = 64
        for i in range(num_layers):
            next_ch = hidden_ch if i < num_layers - 1 else out_channels
            layers.append(nn.Conv2d(curr_ch, next_ch, kernel_size=3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            curr_ch = next_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    # 假設 79 幀對應 1.0 秒訊號（如有不同可傳入 --signal_length_sec 調整）
    parser.add_argument("--signal_length_sec", type=float, default=1.0) 
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 開始測量 純 CNN 學生模型 在 {args.device.upper()} 上的 RTF...")

    # 1. 建立模型
    model = TSConv(in_channels=2, num_layers=4, out_channels=2).to(device)
    model.eval()

    # 2. 建立輸入張量 [B, C, T, F] -> [1, 2, 79, 257]
    x = torch.randn(1, 2, 79, 257).to(device)

    # 3. 暖機 (Warm-up) 10 次
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # 4. 正式測速 (精準測試 100 次取平均)
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    total_time = time.time() - start_time
    avg_latency_sec = total_time / num_runs
    avg_latency_ms = avg_latency_sec * 1000

    # 5. 計算 RTF
    rtf = avg_latency_sec / args.signal_length_sec

    print("\n" + "="*50)
    print(f"📊 Student_Production_Annealed 模型效能報告 ({args.device.upper()})")
    print("="*50)
    print(f"單次推論延遲 (Latency): {avg_latency_ms:.2f} ms")
    print(f"訊號物理長度 (Signal):  {args.signal_length_sec:.2f} sec")
    print(f"⭐ Real-Time Factor (RTF): {rtf:.5f}")
    print("="*50)
    
    if rtf < 1.0:
        print("✅ 結論：完美達成即時運算要求 (RTF < 1)！")
    else:
        print("❌ 結論：無法達成即時運算。")

if __name__ == "__main__":
    main()
