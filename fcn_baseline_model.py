# ────────────────────────────────────────────────────
# fcn_baseline_model.py
# 移植自 eric-wang135/ECG-removal-from-sEMG-by-FCN（main/model/FCN.py 的 FCN_01）
# 原始碼來源：https://github.com/eric-wang135/ECG-removal-from-sEMG-by-FCN
# 對應論文：Wang et al., "ECG Artifact Removal from Single-Channel Surface EMG
#           Using Fully Convolutional Networks", ICASSP 2023 (arXiv:2210.13271)
#
# 這裡逐字保留原始架構定義（encoder/decoder 的 channel 數、kernel size、stride
# 完全比照原始碼），只是搬進本專案，方便直接用你們現有的資料/訓練慣例接上，
# 不需要另外 clone 整個原始 repo。
#
# 原始訓練配方（來自原 repo 的 main.py / Trainer.py 預設值）：
#   loss_fn='l1', optimizer='adam', lr=0.0001, batch_size=16, epochs=100
#   訓練方式：pred = model(noisy); loss = L1(pred, clean) —— 純粹逼近 GT，
#   原始碼裡從頭到尾沒有 teacher/蒸餾的概念。
# ────────────────────────────────────────────────────
import torch
import torch.nn as nn


class conv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift):
        super().__init__()
        self.conv_1d = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, frame_size, shift),
            nn.BatchNorm1d(out_channel),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv_1d(x)


class deconv_1d(nn.Module):
    def __init__(self, in_channel, out_channel, frame_size, shift, out_pad=0):
        super().__init__()
        self.deconv_1d = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, frame_size, shift, output_padding=out_pad),
            nn.BatchNorm1d(out_channel),
            nn.ELU(),
        )

    def forward(self, x):
        return self.deconv_1d(x)


class FCN_01(nn.Module):
    """
    輸入/輸出: [B, L]（原始時域波形，單通道）。
    跟 SSEMG-Net/StudentSSEMGNet 完全不同的資料表示方式（時域 vs 頻域），
    這是刻意保留的，不做任何修改，確保跟原論文的架構完全一致，
    比較才有意義。
    """
    def __init__(self):
        super().__init__()
        self.frame_size = 16
        self.encoder = nn.Sequential(
            conv_1d(1, 80, self.frame_size, 2),
            conv_1d(80, 40, self.frame_size, 2),
            conv_1d(40, 20, self.frame_size, 1),
        )
        self.decoder = nn.Sequential(
            deconv_1d(20, 20, self.frame_size, 1),
            deconv_1d(20, 40, self.frame_size, 2),
            deconv_1d(40, 80, self.frame_size, 2),
            nn.ConvTranspose1d(80, 1, self.frame_size, 1),
        )

    def forward(self, emg):
        f = self.encoder(emg.unsqueeze(1))
        out = self.decoder(f)
        return out[:, :, :emg.shape[1]].squeeze(1)
