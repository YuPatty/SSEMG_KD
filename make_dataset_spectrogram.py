# /data/member1/user_howardshih/shihsemg/make_dataset_spectrogram.py
import os
import argparse
import json
from glob import glob

import torch
from tqdm import tqdm


def _to_2FT(t):
    """
    將輸入 spectrogram 張量正規化為 [2, F, T] 形狀。
    允許輸入為：
      - [T, F, 2]
      - [2, F, T]
      - [2, T, F]
    其他形狀會丟出 ValueError。
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)

    if t.ndim != 3:
        raise ValueError(f"Expect 3D tensor for a single spectrogram, got shape {tuple(t.shape)}")

    # 三種常見情況
    if t.shape[-1] == 2:             # [T, F, 2]
        t = t.permute(2, 1, 0)       # -> [2, F, T]
    elif t.shape[0] == 2 and t.shape[1] != 2:  # [2, F, T] (already OK) 或 [2, T, F]
        # 判斷第二維可能是 F 或 T，無法自動辨識，只能盡量用規則：
        # 通常 F << T（例如 F~257/257，T較大），我們用「F < T」來猜測
        c, a, b = t.shape
        if a <= b:   # 假設 a 是 F, b 是 T
            # 視為 [2, F, T]
            pass
        else:
            # 視為 [2, T, F] -> [2, F, T]
            t = t.permute(0, 2, 1)
    else:
        raise ValueError(
            f"Unsupported spectrogram shape {tuple(t.shape)}. "
            f"Expect [T, F, 2] or [2, F, T] or [2, T, F]."
        )

    # dtype 與 contiguous 處理
    t = t.contiguous().float()
    return t


def collect_spectrogram_pairs(split_dir, noisy_dirname, clean_dirname, ext=".pt"):
    """
    收集 (noisy_spec, clean_spec) 路徑配對清單。
    noisy_dir 目錄層級預期： split/noisy_dir/{SNR}/{ECG_ID}/*.pt
    clean_dir 目錄層級預期： split/clean_dir/*.pt   或   split/clean_dir/{同名層級}/.pt
    若 clean 與 noisy 為同名階層（建議），則會直接以檔名匹配。
    """
    noisy_root = os.path.join(split_dir, noisy_dirname)
    clean_root = os.path.join(split_dir, clean_dirname)

    if not os.path.isdir(noisy_root):
        raise FileNotFoundError(f"Noisy root not found: {noisy_root}")
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"Clean root not found: {clean_root}")

    # 尋找 noisy 端所有檔案
    pattern = os.path.join(noisy_root, "**", f"*{ext}")
    noisy_files = sorted(glob(pattern, recursive=True))

    pairs = []
    missing_clean = 0

    for noisy_path in noisy_files:
        # 將 noisy_path 相對於 noisy_root 的相對路徑，拿去 clean_root 對應
        rel = os.path.relpath(noisy_path, noisy_root)
        clean_path = os.path.join(clean_root, rel)

        # 若 clean 不是同層級結構，也允許在 clean_root 直接用檔名匹配
        if not os.path.exists(clean_path):
            fname = os.path.basename(noisy_path)
            alt_clean = os.path.join(clean_root, fname)
            if os.path.exists(alt_clean):
                clean_path = alt_clean

        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
        else:
            missing_clean += 1

    if missing_clean > 0:
        print(f"[Warning] {missing_clean} clean files not found under '{clean_root}' (by relative path or filename).")

    return pairs


# === 替換 save_tensor_dataset() 整段 =================================
def save_tensor_dataset(pairs, save_path, expect_min=1, sidecar_meta_path=None):
    ref = _to_2FT(torch.load(pairs[0][0], map_location='cpu'))
    N, C, F, T = len(pairs), *ref.shape
    X = torch.empty((N, C, F, T), dtype=torch.float32)   # → float32
    Y = torch.empty_like(X)

    valid = 0
    for noisy_p, clean_p in tqdm(pairs, desc="stack"):
        try:
            n = _to_2FT(torch.load(noisy_p, map_location='cpu'))
            c = _to_2FT(torch.load(clean_p, map_location='cpu'))
            if n.shape != (C, F, T):
                print(f"[shape skip] {noisy_p}")
                continue
            X[valid], Y[valid] = n, c
            valid += 1
        except Exception as e:
            print(f"[skip] {noisy_p} ({e})")

    X, Y = X[:valid], Y[:valid]
    torch.save([X, Y], save_path)           # 預設 zip 格式，I/O 足夠快
    print(f"✔ saved {valid} pairs  float32  -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='processed/semg')
    parser.add_argument('--noisy_dir', type=str, default='spectrogram_cf05')
    parser.add_argument('--clean_dir', type=str, default='spectrogram_cf05_clean')
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--ext', type=str, default='.pt', help="spectrogram file extension, default .pt")
    parser.add_argument('--save_meta', action='store_true', help="save a JSON sidecar file with dataset meta")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing split: {split}")
        split_dir = os.path.join(args.data_root, split)
        pairs = collect_spectrogram_pairs(split_dir, args.noisy_dir, args.clean_dir, ext=args.ext)

        save_path = os.path.join(args.output_dir, f"{split}_spectrogram.pt")
        meta_path = os.path.join(args.output_dir, f"{split}_spectrogram.meta.json") if args.save_meta else None
        save_tensor_dataset(pairs, save_path, sidecar_meta_path=meta_path)


if __name__ == "__main__":
    main()