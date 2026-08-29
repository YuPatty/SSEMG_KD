# ────────────────────────────────────────────────────
# build_snr_labels.py
# 重建 {split}_spectrogram.pt 裡每個 index 對應的原始 SNR 檔位。
#
# 原理：make_dataset_spectrogram.py 用 sorted(glob(...)) 收集檔案，順序固定、
# 可重現；只要 processed/semg/{split}/spectrogram_cf05/ 資料夾內容沒變過，
# 重新跑一次同樣的收集 + skip 邏輯，就能得到跟當初疊 tensor 時完全一致的順序，
# 藉此標出每個 index 屬於哪個 SNR，而不用重新產生 .pt 本身。
#
# ⚠️ 重要：這個重建結果只在「spectrogram_cf05/ 資料夾內容沒有變動過」的前提下
# 才準確。腳本最後會用長度比對做 sanity check，如果對不上，代表資料夾已經變過，
# 這個重建方法不可信，要找當初真正產生 test_spectrogram.pt 時的原始檔案清單。
#
# 用法：
#   python build_snr_labels.py \
#       --data_root /home/taes10056/SSEMG-Net/semg_data/processed \
#       --split test \
#       --spectrogram_pt /home/taes10056/SSEMG-Net/dataset/test_spectrogram.pt \
#       --out test_snr_labels.json
# ────────────────────────────────────────────────────
import argparse
import json
import os
import re
from glob import glob

import torch
from tqdm import tqdm


def _to_2FT(t):
    """跟 make_dataset_spectrogram.py 的 _to_2FT 完全一致，只是為了複製同樣的
    shape 檢查邏輯，不是為了真的轉換資料。"""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    if t.ndim != 3:
        raise ValueError(f"Expect 3D tensor, got {tuple(t.shape)}")
    if t.shape[-1] == 2:
        t = t.permute(2, 1, 0)
    elif t.shape[0] == 2 and t.shape[1] != 2:
        c, a, b = t.shape
        if a > b:
            t = t.permute(0, 2, 1)
    return t.contiguous().float()


def extract_snr(noisy_path, noisy_root):
    """從路徑推出 SNR 檔位。noisy_root 底下第一層資料夾就是 SNR
    （spectrogram_cf05/{SNR}/{ECG_ID}/*.pt），跟你 find 出來的
    -10/-12/-14/-2/-4/-6/-8/0 這幾個資料夾對應。"""
    rel = os.path.relpath(noisy_path, noisy_root)
    first_part = rel.split(os.sep)[0]
    m = re.match(r"^-?\d+$", first_part)
    if not m:
        raise ValueError(f"無法從路徑解析 SNR：{noisy_path}（第一層資料夾={first_part}）")
    return int(first_part)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True,
                    help='對應 make_dataset_spectrogram.py 的 --data_root，'
                         '例如 .../semg_data/processed')
    p.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    p.add_argument('--noisy_dir', default='spectrogram_cf05')
    p.add_argument('--clean_dir', default='spectrogram_cf05_clean')
    p.add_argument('--ext', default='.pt')
    p.add_argument('--spectrogram_pt', required=True,
                    help='對應的 {split}_spectrogram.pt，用來做長度 sanity check')
    p.add_argument('--out', default=None,
                    help='輸出的標籤檔路徑，預設 {split}_snr_labels.json')
    args = p.parse_args()

    split_dir = os.path.join(args.data_root, args.split)
    noisy_root = os.path.join(split_dir, args.noisy_dir)
    clean_root = os.path.join(split_dir, args.clean_dir)

    pattern = os.path.join(noisy_root, "**", f"*{args.ext}")
    noisy_files = sorted(glob(pattern, recursive=True))
    if not noisy_files:
        raise FileNotFoundError(f"在 {noisy_root} 底下找不到任何 {args.ext} 檔案，路徑對嗎？")

    print(f"[info] 找到 {len(noisy_files)} 個 noisy 檔案，開始重建順序 + SNR 標籤...")

    snr_labels = []
    ref_shape = None
    skipped = 0

    for noisy_path in tqdm(noisy_files, desc="rebuild"):
        rel = os.path.relpath(noisy_path, noisy_root)
        clean_path = os.path.join(clean_root, rel)
        if not os.path.exists(clean_path):
            fname = os.path.basename(noisy_path)
            alt_clean = os.path.join(clean_root, fname)
            if os.path.exists(alt_clean):
                clean_path = alt_clean
            else:
                skipped += 1
                continue

        try:
            n = _to_2FT(torch.load(noisy_path, map_location='cpu'))
        except Exception:
            skipped += 1
            continue

        if ref_shape is None:
            ref_shape = tuple(n.shape)
        elif tuple(n.shape) != ref_shape:
            # 跟 make_dataset_spectrogram.py 的 shape mismatch skip 邏輯一致
            skipped += 1
            continue

        snr = extract_snr(noisy_path, noisy_root)
        snr_labels.append(snr)

    print(f"[info] 重建完成：{len(snr_labels)} 筆有效樣本，跳過 {skipped} 筆")

    # ---- sanity check：長度要跟 {split}_spectrogram.pt 的樣本數一致 ----
    X, Y = torch.load(args.spectrogram_pt, map_location='cpu')
    n_actual = X.shape[0]
    if len(snr_labels) != n_actual:
        print(f"⚠️ 警告：重建出 {len(snr_labels)} 筆，但 {args.spectrogram_pt} 裡實際有 "
              f"{n_actual} 筆。兩者對不上，代表 spectrogram_cf05/ 資料夾內容在產生 "
              f"{args.spectrogram_pt} 之後被改動過（新增/刪除/覆蓋過檔案），"
              f"這個標籤重建結果不可信，請不要拿去用。")
    else:
        print(f"✓ sanity check 通過：{len(snr_labels)} 筆對上 {n_actual} 筆，順序可信。")

    out_path = args.out or f"{args.split}_snr_labels.json"
    with open(out_path, 'w') as f:
        json.dump(snr_labels, f)
    print(f"[info] 已存到 {out_path}")

    # 順便印一下每個 SNR 檔位各有幾筆，方便你確認資料分布跟資料夾內容對得上
    from collections import Counter
    counts = Counter(snr_labels)
    print("[info] 各 SNR 檔位樣本數：")
    for snr in sorted(counts.keys(), reverse=True):
        print(f"    {snr:>4} dB : {counts[snr]}")


if __name__ == '__main__':
    main()
