# ────────────────────────────────────────────────────
# train_sdemg_baseline.py  (patched)
# 用 ConditionalModel + GaussianDiffusion1D（來自 yt-tony-liu/SDEMG 的
# deep_filter_model.py / ddpm_1d.py）當作額外的 no-KD baseline。
#
# ⚠️ 這個版本補上了跟 SDEMG 官方 trainer.py 對齊的兩個機制（先前版本漏掉的）：
#   1. ReduceLROnPlateau(factor=0.5, patience=4) —— val_loss 連續 4 epoch 沒進步就把 lr 減半
#   2. clip_grad_norm_(parameters, max_norm=1.0) —— 每個 batch 都做梯度裁剪
# 這兩個都是直接對照 SDEMG 官方 trainer.py 第 105 / 168 / 190 行加上去的，
# 除此之外訓練邏輯（diffusion loss、資料來源、early stopping）都跟原本版本一樣，
# 只改動這兩處，方便做「補上這兩個機制前後」的乾淨對照實驗。
#
# 訓練配方沿用 SDEMG repo 自帶的 cfg/default.yaml（DiffuEMG_10sec_EP40_SS50）：
#   train_epochs=40, batch_size=64, condition=True, sampling_steps(timesteps)=50,
#   beta_schedule='cosine', objective='pred_noise', loss_function='l2', lr=1e-4
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

# 同 train_msemg_baseline.py 的修正：ROOT_DIR 改用 --ssemg_net_root 明確指定，
# 腳本檔案本身放在任何地方都能跑，不用跟 pipeline_spectrogram.py / MECG-E/
# 放在同一層資料夾。不傳這個參數時退回原本行為（假設腳本就放在 SSEMG-Net 裡）。
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--ssemg_net_root', default=None)
_pre_args, _ = _p.parse_known_args()
ROOT_DIR = _pre_args.ssemg_net_root or os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
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
    p.add_argument('--ssemg_net_root', default=None,
                    help='SSEMG-Net repo 的路徑（用來找 pipeline_spectrogram.py / MECG-E/）。'
                         '不指定時預設腳本檔案自己所在的資料夾。')
    p.add_argument('--student_config', default='config_student_crossarch.yaml',
                    help="只用來讀 n_fft/hop_size/win_size/compress_factor 等 STFT 參數")
    p.add_argument('--data_root', default='dataset')

    # 以下預設值沿用 SDEMG repo 自帶的 cfg/default.yaml
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15,
                    help='early stopping patience（官方 trainer.py 其實沒有 early stop，'
                         '這裡保留只是安全網，40 epoch 內基本不會被觸發）')
    p.add_argument('--feats', type=int, default=64,
                    help='ConditionalModel 的 hidden channels。預設 64，對齊 MSEMG 論文 '
                         'Table II 報告之比較基準（1,233,857 參數）；官方 repo main.py '
                         '目前預設是 128（4.93M 參數），與此不同，除非你刻意要重現那個版本，'
                         '否則不要改回 128。')
    p.add_argument('--seq_length', type=int, default=10000)
    p.add_argument('--sampling_steps', type=int, default=50, help='對應 diffusion timesteps')
    p.add_argument('--objective', choices=['pred_noise', 'pred_x0', 'pred_v'], default='pred_noise')
    p.add_argument('--loss_function', choices=['l1', 'l2'], default='l2')
    p.add_argument('--beta_schedule', choices=['linear', 'cosine', 'quad'], default='cosine')
    p.add_argument('--condition', action='store_true', default=True,
                    help='ConditionalModel 需要 cond 輸入，SDEMG 原始預設就是 True，不要關掉')

    # ── 新增：跟官方 trainer.py 對齊的兩個機制 ──
    p.add_argument('--lr_patience', type=int, default=4,
                    help='ReduceLROnPlateau 的 patience，對齊官方 trainer.py 第105行')
    p.add_argument('--lr_factor', type=float, default=0.5,
                    help='ReduceLROnPlateau 的 factor，對齊官方 trainer.py 第105行')
    p.add_argument('--grad_clip_norm', type=float, default=1.0,
                    help='gradient clipping 的 max_norm，對齊官方 trainer.py 第168行')

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
    # ── 新增 1/2：對齊官方 trainer.py 的 ReduceLROnPlateau ──
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )

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
        csv.writer(f).writerow(['epoch', 'train_diffusion_loss', 'val_diffusion_loss', 'lr'])

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
                # GaussianDiffusion1D 要求輸入長度剛好等於 seq_length，多截少補
                clean_wav = _fit_length(clean_wav, args.seq_length)
                noisy_wav = _fit_length(noisy_wav, args.seq_length)

            # GaussianDiffusion1D.forward(clean_img, noisy_img) 要求 (B, C=1, L)
            loss = diffusion(clean_wav.unsqueeze(1), noisy_wav.unsqueeze(1))

            optim.zero_grad()
            loss.backward()
            # ── 新增 2/2：對齊官方 trainer.py 的 gradient clipping ──
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip_norm)
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

        # ── 新增：對齊官方 trainer.py 第190行，用 val_loss 驅動 lr scheduler ──
        lr_scheduler.step(val_loss)
        current_lr = optim.param_groups[0]['lr']

        print(f"★ Epoch {ep} | train_diffusion_loss={train_loss:.5g} | "
              f"val_diffusion_loss={val_loss:.5g} | lr={current_lr:.2e}")
        with open(args.log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, train_loss, val_loss, current_lr])

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            # 只存 denoiser (ConditionalModel) 的 state_dict，跟你們現有的 checkpoint 慣例
            # （只存 model 本體）保持一致；官方 test() 用的也是非 EMA 的原始權重，這裡對齊。
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

