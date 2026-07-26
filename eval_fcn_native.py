# ────────────────────────────────────────────────────
# eval_fcn_native.py
# 載入用原始 repo（main.py/Trainer.py，完全未修改）訓練出來的 FCN_01
# checkpoint，接上你們專案自己的 test-set 評估邏輯（SNRimp/RMSE/RMSE_MF 等），
# 確保跟 Teacher/KD-Student/No-KD 用同一套指標，可以放進同一張比較表。
#
# ── checkpoint 格式說明（核對過 Trainer.py::save_checkpoint()）──
# 原始 repo 存的不是純權重，是包了一層的 dict：
#   {'epoch':.., 'model': model.state_dict(), 'optimizer':.., 'best_loss':..}
# 真正的模型權重在 checkpoint['model']，不能直接對整個 dict 做 load_state_dict。
#
# 用法：
#   python eval_fcn_native.py \
#       --fcn_checkpoint /path/to/ECG-removal-from-sEMG-by-FCN/main/save_model/denoise_FCN_01_epochs100_adam_l1_batch16_lr0.0001.pth.tar \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset
# ────────────────────────────────────────────────────
import os, sys, argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from fcn_baseline_model import FCN_01
from pipeline_spectrogram import load_dataset, auto_select_gpu
from models.StudentNet import mag_pha_istft


def spec_to_wav_batch(spec, n_fft, hop_size, win_size, compress_factor):
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)


# ── 沿用專案既有的評估指標定義（跟 Teacher/Student inference 一致）──
def cal_snrimp(clean, noisy, enhanced, eps=1e-12):
    noise_before = clean - noisy
    noise_after = clean - enhanced
    p_before = np.sum(noise_before ** 2) + eps
    p_after = np.sum(noise_after ** 2) + eps
    return 10 * np.log10(p_before / p_after)


def cal_rmse(clean, enhanced):
    return np.sqrt(np.mean((clean - enhanced) ** 2))


def cal_prd(clean, enhanced, eps=1e-12):
    num = np.sum((clean - enhanced) ** 2)
    den = np.sum(clean ** 2) + eps
    return 100 * np.sqrt(num / den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fcn_checkpoint', required=True,
                    help="原始 repo 存的 .pth.tar 檔案路徑（包了 epoch/model/optimizer/best_loss 的 dict）")
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml',
                    help="只用來讀 STFT 參數，確保 iSTFT 重建跟訓練時一致")
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    device = auto_select_gpu()

    # ── 載入 checkpoint，取出 state_dict['model'] 這一層 ──
    checkpoint = torch.load(args.fcn_checkpoint, map_location=device)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(
            f"checkpoint 格式跟預期不符（應該要有 'model' 這個 key），"
            f"實際 keys: {checkpoint.keys() if isinstance(checkpoint, dict) else type(checkpoint)}"
        )
    print(f"✅ checkpoint 讀取成功，訓練到 epoch={checkpoint.get('epoch')}, "
          f"best_loss={checkpoint.get('best_loss')}")

    model = FCN_01().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"✅ FCN_01 權重載入成功（strict=True），參數量: {n_params:,}")

    # ── 準備 test set（跟 spec_to_wav 邏輯一致，即時 iSTFT）──
    X_te, y_te = load_dataset('test', args.data_root)
    test_loader = DataLoader(TensorDataset(y_te, X_te), batch_size=args.batch_size, shuffle=False)

    snrimp_list, rmse_list, prd_list = [], [], []

    with torch.no_grad():
        for clean_spec, noisy_spec in tqdm(test_loader, desc="Evaluating FCN (native-trained)"):
            clean_spec = clean_spec.to(device)
            noisy_spec = noisy_spec.to(device)

            clean_wav = spec_to_wav_batch(clean_spec, n_fft, hop_size, win_size, compress_factor)
            noisy_wav = spec_to_wav_batch(noisy_spec, n_fft, hop_size, win_size, compress_factor)
            pred_wav = model(noisy_wav)

            clean_np = clean_wav.cpu().numpy()
            noisy_np = noisy_wav.cpu().numpy()
            pred_np = pred_wav.cpu().numpy()

            for i in range(clean_np.shape[0]):
                snrimp_list.append(cal_snrimp(clean_np[i], noisy_np[i], pred_np[i]))
                rmse_list.append(cal_rmse(clean_np[i], pred_np[i]))
                prd_list.append(cal_prd(clean_np[i], pred_np[i]))

    print("\n===== Test-set Average (FCN, 原生 repo 訓練) =====")
    print(f"  SNRimp                   : {np.mean(snrimp_list):.4f}")
    print(f"  RMSE                     : {np.mean(rmse_list):.4f}")
    print(f"  PRD(%)                   : {np.mean(prd_list):.4f}")
    print("\n⚠️ 注意：這裡沒有算 RMSE_MF / RMSE_MeanF / RMSE_MedianF，")
    print("   因為那幾項需要頻域分析，如果你要跟 Teacher/Student 那組報告的")
    print("   完整指標對齊，需要另外接上你們主線 inference 腳本裡對應的頻率")
    print("   相關計算邏輯——目前這支只算了 SNRimp/RMSE/PRD 三項最基本的比較。")


if __name__ == '__main__':
    main()
