# ────────────────────────────────────────────────────
# prepare_fcn_data.py
# 把 train/valid_spectrogram.pt 轉成原始 FCN repo（eric-wang135/
# ECG-removal-from-sEMG-by-FCN）的 Load_model.py::Load_data() 預期的資料格式，
# 讓他的 Trainer.py/main.py/Load_model.py 完全不用修改就能直接訓練。
#
# ── 原始 repo 的資料格式規格（來自 Load_model.py 逐行核對）──
#   Load_data() 做的事：
#     train_paths = get_filepaths(f'{train_path}/noisy', '.pt')
#     val_paths   = get_filepaths(f'{train_path}/val', '.pt')
#     clean_path  = train_path + train_clean   ← 注意：字串相加，不是路徑合併！
#                   所以 train_path 這個參數結尾必須自己帶 '/'，
#                   例如 train_path='./fcn_data/'，train_clean='clean'，
#                   相加後才會變成合法路徑 './fcn_data/clean'
#     CustomDataset 用「檔名」去 noisy/val 資料夾比對 clean 資料夾裡的同名檔案，
#     不是用順序比對，所以 train 和 val 的 clean 檔案要放在同一個 clean 資料夾，
#     用不同檔名前綴（train_/val_）避免衝突。
#
#   最終要產生：
#     {train_path}noisy/train_000000.pt   ← 每個樣本各自一個檔案（時域波形）
#     {train_path}clean/train_000000.pt   ← 檔名跟 noisy 對應
#     {train_path}val/val_000000.pt
#     {train_path}clean/val_000000.pt     ← 跟 train 的 clean 共用同一個資料夾
#
# ⚠️ 這裡產生的時域波形，是用專案既有的 mag_pha_istft() 從 spectrogram
# 重建回來的，不是真正原始的時域錄製資料（原因跟 train_fcn_baseline.py
# 裡說明的一樣：你的資料集只保留了 STFT 頻譜，沒有另存原始波形）。
#
# ⚠️ 注意：這會產生大量小檔案（每個樣本一個 .pt），如果 train_spectrogram.pt
# 樣本數很多，磁碟 inode 數量和寫入時間都可能不小，建議先用 --limit 測一個
# 小子集，確認流程沒問題，再跑全量轉換。
#
# 用法：
#   python prepare_fcn_data.py \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset --out_dir ./fcn_data/ --limit 200   # 先測小樣本
#   python prepare_fcn_data.py \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset --out_dir ./fcn_data/               # 確認沒問題後跑全量
# ────────────────────────────────────────────────────
import os, sys, argparse
import yaml
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from pipeline_spectrogram import load_dataset
from models.StudentNet import mag_pha_istft   # 純函式，不依賴 mamba_ssm


def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    """spec: [2, F, T]（單一樣本，channel0=mag, channel1=phase）→ [L] 時域波形"""
    mag = spec[0].unsqueeze(0)   # [1, F, T]
    pha = spec[1].unsqueeze(0)
    wav = mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)
    return wav.squeeze(0)        # [L]


def check_folder(path):
    """比照原始 repo util.py 的行為：傳入完整檔案路徑，自動建立其上層資料夾"""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)


def convert_split(X, Y, out_dir, prefix, n_fft, hop_size, win_size, compress_factor,
                   noisy_subdir, limit=None):
    """
    X: noisy spectrogram [N, 2, F, T]，Y: clean spectrogram [N, 2, F, T]
    輸出：
      {out_dir}{noisy_subdir}/{prefix}_{i:06d}.pt   ← noisy 時域波形
      {out_dir}clean/{prefix}_{i:06d}.pt            ← clean 時域波形（統一放 clean 資料夾）
    """
    n = X.shape[0] if limit is None else min(limit, X.shape[0])
    for i in tqdm(range(n), desc=f"converting {prefix}"):
        noisy_wav = spec_to_wav(X[i], n_fft, hop_size, win_size, compress_factor)
        clean_wav = spec_to_wav(Y[i], n_fft, hop_size, win_size, compress_factor)

        noisy_path = os.path.join(out_dir, noisy_subdir, f"{prefix}_{i:06d}.pt")
        clean_path = os.path.join(out_dir, "clean", f"{prefix}_{i:06d}.pt")

        check_folder(noisy_path)
        check_folder(clean_path)
        torch.save(noisy_wav.clone(), noisy_path)
        torch.save(clean_wav.clone(), clean_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml')
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--out_dir', default='./fcn_data/',
                    help="⚠️ 結尾要帶 '/'，因為原始 repo 的 Load_data() 用字串相加"
                         "（train_path + train_clean）組合出 clean 資料夾路徑，"
                         "不是用 os.path.join，沒有結尾斜線會拼出錯誤路徑")
    p.add_argument('--limit', type=int, default=None,
                    help="只轉換前 N 筆樣本，先用小樣本測試整條流程再跑全量")
    args = p.parse_args()

    if not args.out_dir.endswith('/'):
        args.out_dir += '/'
        print(f"[提醒] --out_dir 沒有結尾斜線，已自動補上：{args.out_dir}")

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    print("載入 train_spectrogram.pt / valid_spectrogram.pt ...")
    X_tr, y_tr = load_dataset('train', args.data_root)
    X_va, y_va = load_dataset('valid', args.data_root)

    print(f"train: {X_tr.shape[0]} 筆, valid: {X_va.shape[0]} 筆"
          + (f"（本次限制只轉前 {args.limit} 筆）" if args.limit else ""))

    convert_split(X_tr, y_tr, args.out_dir, prefix="train",
                  n_fft=n_fft, hop_size=hop_size, win_size=win_size,
                  compress_factor=compress_factor, noisy_subdir="noisy", limit=args.limit)
    convert_split(X_va, y_va, args.out_dir, prefix="val",
                  n_fft=n_fft, hop_size=hop_size, win_size=win_size,
                  compress_factor=compress_factor, noisy_subdir="val", limit=args.limit)

    print(f"\n✅ 轉換完成，輸出在 {args.out_dir}")
    print(f"   {args.out_dir}noisy/train_*.pt")
    print(f"   {args.out_dir}val/val_*.pt")
    print(f"   {args.out_dir}clean/(train_*.pt 和 val_*.pt 都在這裡)")
    print(f"\n接下來用原始 repo 的 main.py 訓練，參數對應：")
    print(f"   --train_path {args.out_dir}")
    print(f"   --train_clean clean")
    print(f"   --model FCN_01")


if __name__ == '__main__':
    main()
