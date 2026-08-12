# ────────────────────────────────────────────────────
# train_msemg_baseline.py
# 用 EMGMAMBA（來自 yt-tony-liu/MSEMG 的 model.py）當作額外的 no-KD baseline，
# 訓練方式跟 train_fcn_baseline.py 完全同一套精神：純粹用 L1/L2 loss 逼近
# ground truth 時域波形，不接觸 teacher，不做任何蒸餾。
#
# ── 跟 train_fcn_baseline.py 的差異 ──
# MSEMG 原始 repo 沒有附 cfg/，也沒有文件化的官方訓練配方（不像 FCN 那篇論文
# main.py/Trainer.py 裡有寫死的預設值可以直接沿用）。這裡的 epochs/batch_size/
# lr/loss_fn 預設值是「跟你們主線訓練慣例接近」的合理值，不是 MSEMG 原論文的
# 官方配方——這點務必知道，不要當成跟 FCN baseline 一樣「逐字複刻原論文」。
#
# EMGMAMBA 需要 mamba_ssm（Triton kernel），必須有 CUDA 環境才能跑。
#
# 用法：
#   python train_msemg_baseline.py \
#       --msemg_repo /home/taes10056/MSEMG \
#       --student_config config_student_crossarch.yaml \
#       --data_root dataset --epochs 100 --batch_size 16 --lr 1e-4 \
#       --feats 64 --n_layer 1
# ────────────────────────────────────────────────────
import os, sys, csv, argparse, importlib.util
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))

from pipeline_spectrogram import load_dataset, auto_select_gpu
from models.StudentNet import mag_pha_istft   # 純函式，不依賴 mamba_ssm


def load_module_from_path(module_name, file_path):
    """用 importlib 從指定路徑載入模組，不污染 sys.path（避免跟 SSEMG_KD 自己的
    utils.py / dataset.py 撞名，MSEMG repo 裡也有同名檔案）。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    """spec: [B, 2, F, T] → 回傳 [B, L] 時域波形"""
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--msemg_repo', required=True, help='clone 下來的 MSEMG repo 路徑')
    p.add_argument('--student_config', default='config_student_crossarch.yaml',
                    help="只用來讀 n_fft/hop_size/win_size/compress_factor 等 STFT 參數，"
                         "確保 iSTFT 重建方式跟主線訓練一致")
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--loss_fn', choices=['l1', 'l2'], default='l1')
    p.add_argument('--feats', type=int, default=64, help='EMGMAMBA 的 hidden channels，對應 model.py 內建 demo 預設值')
    p.add_argument('--in_channels', type=int, default=64, help='MambaBlock 的 in_channels 參數（沿用 model.py 預設）')
    p.add_argument('--n_layer', type=int, default=1)
    p.add_argument('--log_csv', default='log_msemg_baseline.csv')
    p.add_argument('--model_save', default='model_weight/msemg_baseline.pth')
    args = p.parse_args()

    device = auto_select_gpu()
    if device.type != 'cuda':
        raise RuntimeError("EMGMAMBA 依賴 mamba_ssm 的 Triton kernel，必須在有 CUDA 的機器上跑。")

    model_module_path = os.path.join(args.msemg_repo, 'model.py')
    msemg_model = load_module_from_path('msemg_model', model_module_path)
    EMGMAMBA = msemg_model.EMGMAMBA

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    model = EMGMAMBA(in_channels=args.in_channels, feats=args.feats, n_layer=args.n_layer).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[MSEMG baseline] 參數量: {n_params:,}（feats={args.feats}, n_layer={args.n_layer}）")

    criterion = torch.nn.L1Loss() if args.loss_fn == 'l1' else torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_tr, y_tr = load_dataset('train', args.data_root)
    X_va, y_va = load_dataset('valid', args.data_root)
    num_workers = max(0, (os.cpu_count() or 0) // 2)
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(TensorDataset(y_tr, X_tr), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin)
    valid_loader = DataLoader(TensorDataset(y_va, X_va), batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin)
    # ⚠️ valid_loader 這裡刻意用 drop_last=True（跟 train_fcn_baseline.py 不同）：
    # EMGMAMBA 靠 mamba_ssm 的 Triton kernel 做 forward，如果 batch shape 中途變動
    # （例如最後一個不滿的 batch），Triton 會重新 JIT 編譯 kernel，在某些環境下
    # （舊 kernel / k8s pod 的 pids_limit 較嚴格）容易觸發
    # "LLVM ERROR: pthread_join failed" 之類的當機。固定 batch shape 可以避免這個問題。

    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    with open(args.log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    best_val = float('inf')
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"[MSEMG baseline] epoch {ep}/{args.epochs}")
        for step, (clean_spec, noisy_spec) in enumerate(bar, start=1):
            clean_spec = clean_spec.to(device, non_blocking=True)
            noisy_spec = noisy_spec.to(device, non_blocking=True)

            with torch.no_grad():
                clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
                noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)

            pred_wav = model(noisy_wav.unsqueeze(1)).squeeze(1)   # EMGMAMBA 要求 (B,1,L)
            minL = min(clean_wav.size(-1), pred_wav.size(-1))
            loss = criterion(pred_wav[:, :minL], clean_wav[:, :minL])

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
                pred_wav = model(noisy_wav.unsqueeze(1)).squeeze(1)
                minL = min(clean_wav.size(-1), pred_wav.size(-1))
                val_running += float(criterion(pred_wav[:, :minL], clean_wav[:, :minL]))
        val_loss = val_running / max(1, len(valid_loader))

        print(f"★ Epoch {ep} | train_loss={train_loss:.5g} | val_loss={val_loss:.5g}")
        with open(args.log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), args.model_save)
            print(f"✓ new best MSEMG baseline (val_loss={best_val:.5g}) saved to {args.model_save}")
        else:
            no_improve += 1
            print(f"✗ no improvement ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


if __name__ == '__main__':
    main()
