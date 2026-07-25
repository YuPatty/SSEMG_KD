# ────────────────────────────────────────────────────
# train_fcn_baseline.py
# 用 FCN_01（移植自 eric-wang135/ECG-removal-from-sEMG-by-FCN）當作額外的
# baseline student，訓練方式完全比照原始 repo 的配方：純粹用 L1 loss 逼近
# ground truth 時域波形，不接觸 teacher，不做任何蒸餾。
#
# ── 為什麼要即時做 iSTFT，而不是另存一份時域資料集 ──
# 你的 train/valid/test_spectrogram.pt 只存了 STFT 頻譜（[N,2,F,T]），
# 沒有保留原始時域波形。這裡直接在讀 batch 的時候，用專案裡本來就有的
# mag_pha_istft() 把 noisy/clean spectrogram 即時轉回時域波形，不用另外
# 產生一份新的、跟現有 8.2GB/36.9GB 一樣龐大的中間檔案。
#
# ⚠️ 注意：這樣重建出來的波形，跟真正「原始」的時域資料相比，會有極輕微的
# STFT/iSTFT round-trip 誤差（尤其邊界的 overlap-add 效應），不是 bit-identical，
# 但用的是跟你們整條 pipeline 訓練時算 time-domain loss 一致的參數與方法，
# 應該足夠接近，不影響這個 baseline 實驗的參考價值。
#
# 用法：
#   python train_fcn_baseline.py \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset --epochs 100 --batch_size 16 --lr 1e-4
# ────────────────────────────────────────────────────
import os, sys, csv, argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from fcn_baseline_model import FCN_01
from pipeline_spectrogram import load_dataset, auto_select_gpu
from models.StudentNet import mag_pha_istft   # 純函式，不依賴 mamba_ssm


def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    """
    spec: [B, 2, F, T]（channel0=mag, channel1=phase，對應 fea='pha' 設定）
    回傳: [B, L] 時域波形
    """
    mag = spec[:, 0]   # [B, F, T]
    pha = spec[:, 1]   # [B, F, T]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--student_config', default='config/config_student_crossarch.yaml',
                    help="只用來讀 n_fft/hop_size/win_size/compress_factor/sampling_rate 等"
                         "STFT 參數，確保 iSTFT 重建方式跟你們主線訓練一致")
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--epochs', type=int, default=100,
                    help="原始 repo 預設 100，比你們主線訓練的 30~60 長，"
                         "沿用原論文配方，不要隨意縮短")
    p.add_argument('--batch_size', type=int, default=16, help="原始 repo 預設值")
    p.add_argument('--lr', type=float, default=1e-4, help="原始 repo 預設值")
    p.add_argument('--patience', type=int, default=15,
                    help="原始 repo 的 Trainer.train() 用 epoch_count<15 當 early stop 條件")
    p.add_argument('--loss_fn', choices=['l1', 'l2'], default='l1', help="原始 repo 預設 l1")
    p.add_argument('--log_csv', default='log_fcn_baseline.csv')
    p.add_argument('--model_save', default='model_weight/fcn_baseline.pth')
    args = p.parse_args()

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    device = auto_select_gpu()
    model = FCN_01().to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[FCN baseline] 參數量: {n_params:,}（原論文架構，未經任何壓縮調整）")

    criterion = torch.nn.L1Loss() if args.loss_fn == 'l1' else torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_tr, y_tr = load_dataset('train', args.data_root)
    X_va, y_va = load_dataset('valid', args.data_root)
    num_workers = max(0, (os.cpu_count() or 0) // 2)
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(TensorDataset(y_tr, X_tr), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin)
    valid_loader = DataLoader(TensorDataset(y_va, X_va), batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=use_pin)

    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    with open(args.log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    best_val = float('inf')
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"[FCN baseline] epoch {ep}/{args.epochs}")
        for step, (clean_spec, noisy_spec) in enumerate(bar, start=1):
            clean_spec = clean_spec.to(device, non_blocking=True)
            noisy_spec = noisy_spec.to(device, non_blocking=True)

            with torch.no_grad():
                # iSTFT 重建波形本身不需要梯度（GT/輸入都是固定的），
                # 只有 FCN 的 forward 才需要梯度
                clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
                noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)

            pred_wav = model(noisy_wav)
            loss = criterion(pred_wav, clean_wav)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.detach())
            bar.set_postfix(loss=f"{running/step:.5g}")

        train_loss = running / max(1, len(train_loader))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for clean_spec, noisy_spec in valid_loader:
                clean_spec = clean_spec.to(device, non_blocking=True)
                noisy_spec = noisy_spec.to(device, non_blocking=True)
                clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
                noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)
                pred_wav = model(noisy_wav)
                val_running += float(criterion(pred_wav, clean_wav))
        val_loss = val_running / max(1, len(valid_loader))

        print(f"★ Epoch {ep} | train_loss={train_loss:.5g} | val_loss={val_loss:.5g}")
        with open(args.log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), args.model_save)
            print(f"✓ new best FCN baseline (val_loss={best_val:.5g}) saved to {args.model_save}")
        else:
            no_improve += 1
            print(f"✗ no improvement ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


if __name__ == '__main__':
    main()
