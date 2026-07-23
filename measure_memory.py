import torch
import torch.nn as nn
from torchinfo import summary

# 定義學生模型
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
    model = TSConv()
    batch_size = 1
    # 你的輸入大小：[Batch, Channels, Time, Freq]
    input_shape = (batch_size, 2, 79, 257)
    
    print("\n" + "="*60)
    print("🧠 Student (TSConv) 模型資源佔用分析報告")
    print("="*60)
    
    # 呼叫 torchinfo 進行深度分析
    summary(model, input_size=input_shape, 
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=3)

if __name__ == "__main__":
    main()
