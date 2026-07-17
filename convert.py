# /data/member1/user_howardshih/shihsemg/convert.py

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from spectrogram_utils import mag_pha_stft

# STFT parameters
n_fft = 512
hop_size = 128
win_size = 512
compress_factor = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hann_win = torch.hann_window(win_size, dtype=torch.float32, device=device)

def _wav2spec(arg):
    """Convert a single .npy waveform to spectrogram and save as .pt"""
    in_f, out_f = arg
    x = np.load(in_f)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mag, pha, _ = mag_pha_stft(
            x_tensor, n_fft, hop_size, win_size, compress_factor
        )
        spec = torch.stack([mag.squeeze(0).T, pha.squeeze(0).T], dim=-1) \
                   .half().cpu()
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    torch.save(spec, out_f)

def process_noisy(input_dir, output_dir, max_workers=None):
    """Process noisy .npy files under nested dirs into spectrogram .pt"""
    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".npy"):
                in_f = os.path.join(root, f)
                rel = os.path.relpath(in_f, input_dir)[:-4] + ".pt"
                out_f = os.path.join(output_dir, rel)
                tasks.append((in_f, out_f))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_wav2spec, tasks), total=len(tasks),
                  desc=f"NOISY STFT→pt ({os.path.basename(output_dir)})"))

def process_clean(input_dir, output_dir, max_workers=None):
    """Process clean .npy files (flat dir) into spectrogram .pt"""
    tasks = []
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        if f.endswith(".npy") and "_sti" not in f:
            in_f = os.path.join(input_dir, f)
            out_f = os.path.join(output_dir, f.replace(".npy", ".pt"))
            tasks.append((in_f, out_f))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_wav2spec, tasks), total=len(tasks),
                  desc=f"CLEAN STFT→pt ({os.path.basename(output_dir)})"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sEMG noisy and clean waveforms (.npy) to spectrogram (.pt)"
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "valid", "test"],
        help="Dataset split to process: train, valid, or test"
    )
    args = parser.parse_args()

    base_in = f"semg_data/processed/{args.split}"
    noisy_in  = os.path.join(base_in, "noisy")
    clean_in  = os.path.join(base_in, "clean")
    noisy_out = os.path.join(base_in, "spectrogram_cf05")
    clean_out = os.path.join(base_in, "spectrogram_cf05_clean")

    print(f"🚀 Converting {args.split} NOISY waveforms to spectrogram...")
    process_noisy(noisy_in, noisy_out, max_workers=os.cpu_count()*2)

    print(f"🚀 Converting {args.split} CLEAN waveforms to spectrogram...")
    process_clean(clean_in, clean_out, max_workers=os.cpu_count()*2)

'''
# Convert train split
python3 convert.py --split train

# Convert valid split
python3 convert.py --split valid

# Convert test split
python3 convert.py --split test
'''