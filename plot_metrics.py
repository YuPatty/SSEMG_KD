import pandas as pd
import matplotlib
matplotlib.use('Agg') # 確保在無 GUI 伺服器上能正常存圖
import matplotlib.pyplot as plt
import os

# 讀取剛剛 60 圈退火的 Log
log_file = 'log_production_annealed_60.csv'
if not os.path.exists(log_file):
    print(f"找不到 {log_file}，請確認檔名！")
    exit()

df = pd.read_csv(log_file)

# 建立 2x2 的畫布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Metrics for Student_Production_Annealed_60', fontsize=18, fontweight='bold')

# 1. Alpha 退火曲線
axes[0, 0].plot(df['epoch'], df['alpha'], color='purple', linewidth=2.5)
axes[0, 0].set_title('KD Weight (Alpha) Annealing', fontsize=14)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Alpha Value')
axes[0, 0].grid(True, linestyle='--', alpha=0.7)

# 2. Train GT vs Train KD
axes[0, 1].plot(df['epoch'], df['train_gt'], label='Train GT Loss', color='#1f77b4', linewidth=2)
axes[0, 1].plot(df['epoch'], df['train_kd'], label='Train KD Loss', color='#ff7f0e', linewidth=2)
axes[0, 1].set_title('Training Losses (GT vs KD)', fontsize=14)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend(fontsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)

# 3. Validation GT Loss
axes[1, 0].plot(df['epoch'], df['val_gt'], color='#2ca02c', linewidth=2.5)
axes[1, 0].set_title('Validation GT Loss', fontsize=14)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, linestyle='--', alpha=0.7)

# 4. Validation RMSE_MF
axes[1, 1].plot(df['epoch'], df['val_mf'], color='#d62728', linewidth=2.5)
axes[1, 1].set_title('Validation Frequency Error (RMSE_MF)', fontsize=14)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Error (Hz)')
axes[1, 1].grid(True, linestyle='--', alpha=0.7)

# 調整排版並存檔
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 確保 results 資料夾存在
os.makedirs('results', exist_ok=True)
save_path = 'results/annealed_60_metrics.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✨ 繪圖大功告成！高畫質圖片已儲存至：{save_path}")
