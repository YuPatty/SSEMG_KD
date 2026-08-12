# ────────────────────────────────────────────────────
# train_sdemg_baseline.py
# 用 ConditionalModel + GaussianDiffusion1D（來自 yt-tony-liu/SDEMG 的
# deep_filter_model.py / ddpm_1d.py）當作額外的 no-KD baseline。
#
# ⚠️ SDEMG 跟 FCN / MSEMG 不一樣，是 diffusion model，不能直接套
# train_fcn_baseline.py 那種「pred = model(noisy); loss = L1(pred, clean)」
# 的訓練方式。這裡改用 SDEMG 原始碼裡 GaussianDiffusion1D.forward() 內建的
# diffusion loss（p_losses，對加噪後的訊號預測噪聲），沒有重新推導公式，
# 直接呼叫 SDEMG 原始的 ddpm_1d.py，避免手刻 diffusion loss 卻推錯的風險。
#
# 訓練配方沿用 SDEMG repo 自帶的 cfg/default.yaml（DiffuEMG_10sec_EP40_SS50）：
#   train_epochs=40, batch_size=64, condition=True, sampling_steps(timesteps)=50,
#   beta_schedule='cosine', objective='pred_noise', loss_function='l2', lr=1e-4
# 這些是 SDEMG 有文件化的官方預設值，不是我自己猜的。
#
# ── 注意：這裡監控的 val_loss 是 diffusion loss（單步 noise 預測誤差），
# 不是最終去噪音質。真正的去噪品質（SNR/PRD 等）要跑完整的 reverse diffusion
# sampling（denoise()，50 步），運算量大很多，建議另外寫 inference 腳本、
# 只在訓練完之後對 test set 跑一次，不要放進每個 epoch 的 validation loop。
#
# 用法：
#   python train_sdemg_baseline.py \
#       --sdemg_repo /home/taes10056/SDEMG \
#       --student_config config_student_crossarch.yaml \
#       --data_root dataset --epochs 40 --batch_size 64 --lr 1e-4
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


def load_module_from_path(module_name, file_path, extra_sys_path=None):
    """用 importlib 從指定路徑載入模組，避免跟 SSEMG_KD 自己的 utils.py 撞名。
    ddpm_1d.py 內部有 `from utils import default`，所以載入它之前要暫時把
    SDEMG repo 路徑塞進 sys.path，載入完畢後立刻移除，不留在全域。"""
    added = False
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module   # ddpm_1d.py 内部 import 需要能找到自己
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(extra_sys_path)


def spec_to_wav(spec, n_fft, hop_size, win_size, compress_factor):
    """spec: [B, 2, F, T] → 回傳 [B, L] 時域波形"""
    mag = spec[:, 0]
    pha = spec[:, 1]
    return mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sdemg_repo', required=True, help='clone 下來的 SDEMG repo 路徑')
    p.add_argument('--student_config', default='config_student_crossarch.yaml',
                    help="只用來讀 n_fft/hop_size/win_size/compress_factor 等 STFT 參數")
    p.add_argument('--data_root', default='dataset')

    # 以下預設值沿用 SDEMG repo 自帶的 cfg/default.yaml
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--feats', type=int, default=128)
    p.add_argument('--seq_length', type=int, default=10000)
    p.add_argument('--sampling_steps', type=int, default=50, help='對應 diffusion timesteps')
    p.add_argument('--objective', choices=['pred_noise', 'pred_x0', 'pred_v'], default='pred_noise')
    p.add_argument('--loss_function', choices=['l1', 'l2'], default='l2')
    p.add_argument('--beta_schedule', choices=['linear', 'cosine', 'quad'], default='cosine')
    p.add_argument('--condition', action='store_true', default=True,
                    help='ConditionalModel 需要 cond 輸入，SDEMG 原始預設就是 True，不要關掉')

    p.add_argument('--log_csv', default='log_sdemg_baseline.csv')
    p.add_argument('--model_save', default='model_weight/sdemg_baseline.pth')
    args = p.parse_args()

    device = auto_select_gpu()

    dfm = load_module_from_path('sdemg_deep_filter_model',
                                 os.path.join(args.sdemg_repo, 'deep_filter_model.py'))
    ddpm = load_module_from_path('sdemg_ddpm_1d',
                                  os.path.join(args.sdemg_repo, 'ddpm_1d.py'),
                                  extra_sys_path=args.sdemg_repo)
    ConditionalModel = dfm.ConditionalModel
    GaussianDiffusion1D = ddpm.GaussianDiffusion1D

    with open(args.student_config) as f:
        cfg = yaml.safe_load(f)
    n_fft = cfg['model']['n_fft']
    hop_size = cfg['model']['hop_size']
    win_size = cfg['model']['win_size']
    compress_factor = cfg['model']['compress_factor']

    denoiser = ConditionalModel(feats=args.feats)
    diffusion = GaussianDiffusion1D(
        denoiser,
        seq_length=args.seq_length,
        timesteps=args.sampling_steps,
        objective=args.objective,
        loss_function=args.loss_function,
        beta_schedule=args.beta_schedule,
        condition=args.condition,
    ).to(device)

    n_params = sum(p_.numel() for p_ in denoiser.parameters())
    print(f"[SDEMG baseline] ConditionalModel 參數量: {n_params:,}（feats={args.feats}, "
          f"timesteps={args.sampling_steps}, objective={args.objective}）")

    optim = torch.optim.Adam(diffusion.parameters(), lr=args.lr)

    X_tr, y_tr = load_dataset('train', args.data_root)
    X_va, y_va = load_dataset('valid', args.data_root)
    
    num_workers = 0
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(TensorDataset(y_tr, X_tr), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin)
    valid_loader = DataLoader(TensorDataset(y_va, X_va), batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin)

    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    with open(args.log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_diffusion_loss', 'val_diffusion_loss'])

    best_val = float('inf')
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        diffusion.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"[SDEMG baseline] epoch {ep}/{args.epochs}")
        for step, (clean_spec, noisy_spec) in enumerate(bar, start=1):
            clean_spec = clean_spec.to(device, non_blocking=True)
            noisy_spec = noisy_spec.to(device, non_blocking=True)

            with torch.no_grad():
                clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
                noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)
                minL = min(clean_wav.size(-1), noisy_wav.size(-1), args.seq_length)
                # GaussianDiffusion1D 要求輸入長度剛好等於 seq_length，多截少補
                clean_wav = _fit_length(clean_wav, args.seq_length)
                noisy_wav = _fit_length(noisy_wav, args.seq_length)

            # GaussianDiffusion1D.forward(clean_img, noisy_img) 要求 (B, C=1, L)
            loss = diffusion(clean_wav.unsqueeze(1), noisy_wav.unsqueeze(1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += float(loss.detach())
            bar.set_postfix(loss=f"{running/step:.5g}")

        train_loss = running / max(1, len(train_loader))

        diffusion.eval()
        val_running = 0.0
        with torch.no_grad():
            for clean_spec, noisy_spec in valid_loader:
                clean_spec = clean_spec.to(device, non_blocking=True)
                noisy_spec = noisy_spec.to(device, non_blocking=True)
                clean_wav = spec_to_wav(clean_spec, n_fft, hop_size, win_size, compress_factor)
                noisy_wav = spec_to_wav(noisy_spec, n_fft, hop_size, win_size, compress_factor)
                clean_wav = _fit_length(clean_wav, args.seq_length)
                noisy_wav = _fit_length(noisy_wav, args.seq_length)
                val_running += float(diffusion(clean_wav.unsqueeze(1), noisy_wav.unsqueeze(1)))
        val_loss = val_running / max(1, len(valid_loader))

        print(f"★ Epoch {ep} | train_diffusion_loss={train_loss:.5g} | val_diffusion_loss={val_loss:.5g}")
        with open(args.log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, train_loss, val_loss])

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            # 只存 denoiser (ConditionalModel) 的 state_dict，跟 param_count 腳本、
            # 你們現有的 checkpoint 慣例（只存 model 本體）保持一致
            torch.save(denoiser.state_dict(), args.model_save)
            print(f"✓ new best SDEMG baseline (val_diffusion_loss={best_val:.5g}) saved to {args.model_save}")
        else:
            no_improve += 1
            print(f"✗ no improvement ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


def _fit_length(wav, target_len):
    """裁切或右側 zero-pad 到 target_len，確保符合 GaussianDiffusion1D 的
    `assert n == seq_length` 要求。"""
    L = wav.size(-1)
    if L == target_len:
        return wav
    if L > target_len:
        return wav[:, :target_len]
    pad = target_len - L
    return torch.nn.functional.pad(wav, (0, pad))


if __name__ == '__main__':
    main()