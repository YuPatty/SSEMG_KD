# ────────────────────────────────────────────────────
# /data/member1/user_howardshih/shihsemg/pipeline_spectrogram.py 
# ────────────────────────────────────────────────────
import os, re, sys, csv, subprocess, torch, math
from contextlib import nullcontext
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── FP32 版本就通通不要 AMP ───────────────────

def _autocast_ctx(device):            # 依舊包一層，之後呼叫不用改
    return nullcontext()

# ── 讓 MECGE 可以 import ───────────────────────────
ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
# from models.SSEMGNet import SSEMGNet           # noqa: E402

# ═══════════════════════════════════════════════════
# utils
# ═══════════════════════════════════════════════════
def load_dataset(split, data_root):
    """
    讀取 {train|valid|test}_spectrogram.pt，回傳 (X, y)。
    期望 shape: X, y ∈ [N, 2, F, T]
    """
    f = torch.load(os.path.join(data_root, f'{split}_spectrogram.pt'), map_location='cpu', mmap=True)
    X, y = f[0], f[1]  # X=noisy , y=clean
    assert X.ndim == 4 and y.ndim == 4 and X.shape[1] == 2 and y.shape[1] == 2, \
        f"Expect X,y as [N,2,F,T], got X={tuple(X.shape)}, y={tuple(y.shape)}"
    assert X.shape == y.shape, f"X/y shape mismatch: {tuple(X.shape)} vs {tuple(y.shape)}"
    return X, y

def auto_select_gpu(min_used=100, idle_util=5):
    """
    1. 先用 nvidia-smi 抓所有卡的 free/used/util
    2. 優先選「幾乎閒置」(used < min_used & util < idle_util)
    3. 仍找不到就挑 free mem 最大
    4. 任一步驟失敗 → fallback CPU
    """
    if not torch.cuda.is_available():
        print("[Auto GPU] no CUDA detected → CPU")
        return torch.device('cpu')

    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,memory.free,memory.used,utilization.gpu',
            '--format=csv,nounits,noheader'          # ← 只有 format 沒有 mat !
        ]
        out = subprocess.check_output(cmd, encoding='utf-8')
        gpus = []
        for line in out.strip().splitlines():
            idx, free, used, util = map(int, re.split(r',\s*', line))
            gpus.append({'idx': idx, 'free': free, 'used': used, 'util': util})

        idle = [g for g in gpus if g['used'] < min_used and g['util'] < idle_util]
        best = max(idle, key=lambda g: g['free']) if idle else max(gpus, key=lambda g: g['free'])

        os.environ['CUDA_VISIBLE_DEVICES'] = str(best['idx'])
        torch.cuda.init()
        _ = torch.zeros(1).cuda()   # 實際測試這張卡能不能配置記憶體
        print(f"[Auto GPU] GPU{best['idx']} free={best['free']}MiB used={best['used']}MiB util={best['util']}%")
        return torch.device('cuda')
    except Exception as e:           # 捕獲任何錯誤 → CPU
        print(f"[Auto GPU] fallback to CPU ({e})")
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        return torch.device('cpu')

def _get_model_savepath(cfg, data_root):
    """
    盡量還原原始 MECGE 的命名：
    model_weight/{experiment}_{n_type}_nv{nv}_weights.pth
    若缺欄位則退回 data_root/best_model.pt
    """
    exp = (
        (cfg.get('exp') or {}).get('name') or
        cfg.get('experiment') or
        'exp'
    )
    n_type = (
        (cfg.get('data') or {}).get('noise_type') or
        cfg.get('n_type') or
        'mix'
    )
    nv = (
        (cfg.get('data') or {}).get('nv') or
        cfg.get('nv') or
        0
    )
    mw_dir = os.path.join(ROOT_DIR, 'model_weight')
    os.makedirs(mw_dir, exist_ok=True)
    path = os.path.join(mw_dir, f"{exp}_{n_type}_nv{nv}_weights.pth")
    # 若使用者希望簡單存一份在 data_root：
    fallback = os.path.join(data_root, 'best_model.pt')
    return path, fallback
def _median_frequency(wav: torch.Tensor, sr: int, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """
    wav: [B, T] float32 (cpu 或 cuda 皆可)
    回傳每個樣本的 MF（Hz）→ [B]
    """
    device = wav.device
    window = torch.hann_window(win, device=device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, center=True, return_complex=True)   # [B, F, TT]
    power_f = (spec.abs() ** 2).sum(dim=-1)  # [B, F]，對 time 聚合
    cum = torch.cumsum(power_f, dim=1)
    total = cum[:, -1:].clamp_min(1e-12)
    thr = 0.5 * total
    # 找到第一個 cum >= thr 的頻Bin
    idx = (cum >= thr).float().argmax(dim=1)  # [B]
    # bin → Hz
    freqs = idx.float() * (sr / n_fft)
    return freqs

def _batch_mf_err(wav_pred: torch.Tensor, wav_clean: torch.Tensor, metrics_cfg: dict) -> float:
    """
    計算一個 batch 的 MF 誤差（|mf(pred) - mf(clean)| 的 batch mean）
    """
    sr   = int(metrics_cfg.get('sampling_rate', 1000))
    nfft = int(metrics_cfg.get('n_fft', 256))
    hop  = int(metrics_cfg.get('hop_size', 32))
    win  = int(metrics_cfg.get('win_size', 128))

    # 放到同一個 device 做 STFT（快）
    device = wav_pred.device
    mf_p = _median_frequency(wav_pred,  sr, nfft, hop, win)
    mf_c = _median_frequency(wav_clean, sr, nfft, hop, win)
    return (mf_p - mf_c).abs().mean().item()

# ═══════════════════════════════════════════════════
def run_training(cfg, data_root):
    # ── 讀取 batch/epochs/accum/amp ─────────────────
    BATCH = int(cfg.get('train', {}).get('batch_size', 8))
    EPOCHS = int(cfg.get('train', {}).get('epochs', 100))
    ACCUM = int(cfg.get('train', {}).get('accum', 1))          # 累積梯度，預設 1（不累積）

    # ── Device / Model ───────────────────────────────
    device = auto_select_gpu()
    model = SSEMGNet(cfg).to(device)

    # ----------- Optimizer & regularizer ---------------------------------
    lr           = float(cfg['train']['lr'])
    clip_max_norm= float(cfg['train']['clip_grad'])

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lsigmoid' in n and 'raw' in n:    # 只對 sigmoid 的 raw 參數加 L2
            p.regularizer = cfg['model'].get('lsig_L2', 5e-5)
            no_decay.append(p)

        elif p.ndim > 1 and 'weight' in n:
            decay.append(p)                   # 有 weight-decay
        else:
            no_decay.append(p)

    optim = torch.optim.AdamW(
        [{'params': decay,    'weight_decay': 1e-3},
        {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr, betas=(0.8, 0.99)
    )
    # ─── 動態學習率（與 v19 相同） ─────────────────────────
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )



    print ("Loading dataset...")




    X_tr, y_tr = load_dataset('train', data_root)
    X_va, y_va = load_dataset('valid', data_root)

    num_workers = max(0, (os.cpu_count() or 0) // 2)
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(
        TensorDataset(y_tr, X_tr),  # (clean, noisy) —— 忠於原始 MECGE
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0)
    )
    valid_loader = DataLoader(
        TensorDataset(y_va, X_va),
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=(num_workers > 0)
    )

    # ── Log CSV（簡單沿用） ─────────────────────────
    # —— use exp.name 作為實驗簽名，存到 dataset/logs/<exp.name>/train_log.csv
    exp = ((cfg.get('exp') or {}).get('name') or
        cfg.get('experiment') or 'exp')
    log_dir = os.path.join(data_root, 'logs', str(exp))
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, 'train_log.csv')

    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch','train_loss','val_loss',
            'tr_time','tr_com','tr_con','tr_mr','tr_H',
            'va_time','va_com','va_con','va_mr','va_H',
            'val_mf','lsigmoid_slope'
        ])
    print(f"[log] writing logs to {log_csv}")

    no_improve_epochs  = 0        # ← 新增：連續沒進步的計數
    PATIENCE           = 10        # ← 想要連續幾個 epoch 才停
    model_save, fallback_save = _get_model_savepath(cfg, data_root)
    print(f"[ckpt] will save to: {model_save}")

    best_val = float('inf')
    best_mf = float('inf')
    best_by_mf_path = model_save.replace('_weights.pth', '_best_by_mf.pth')
    dbg_every = int(cfg.get('train', {}).get('dbg_every', 5000))

    # 額外列印基本資訊（方便核對）
    print(f"Train set: N={X_tr.shape[0]}, per-sample={tuple(X_tr.shape[1:])}")
    print(f"Valid set: N={X_va.shape[0]}, per-sample={tuple(X_va.shape[1:])}")
    print(f"Steps/epoch: {len(train_loader)} (batch={BATCH}, drop_last=True)")

    # ════════════ Epoch loop ════════════
    for ep in range(1, EPOCHS + 1):
        
        # ===== train =========================================================
        model.train()
        running_core = 0.0
        optim.zero_grad(set_to_none=True)

        did_opt_step = False          # NEW: 本 epoch 是否真的做過 optimizer.step()

        with tqdm(train_loader, desc=f"Epoch {ep} [Train]", dynamic_ncols=True) as bar:
            for step, (clean_b, noisy_b) in enumerate(bar, 1):
                # ===== quick-probe：每 3000 step 測一次梯度是否爆、param 是否更新 =====
                if step % 5000 == 1: # ----- Δ|W| probe -----
                    w = model.dense_encoder.dense_conv_1[0].weight
                    prev = getattr(w, "_prev_norm", None)
                    now  = w.detach().float().norm().item()
                    if prev is not None:
                        delta = now - prev
                        if abs(delta) > 3e-3:
                            print(f"[probe] Δ|W|={delta:+.3e}")
                    w._prev_norm = now

                clean_b = clean_b.to(device, non_blocking=True)  # [B, 2, F, T]
                noisy_b = noisy_b.to(device, non_blocking=True)  # [B, 2, F, T]

                ctx = nullcontext()
                with ctx:
                    core = model(clean_b, noisy_b)   # ★ 回傳四項核心 loss

                    # ====== epoch 前 3 代逐漸開啟正則 (warm-up) =====
                    reg_factor = min(1.0, ep / 3.0)   # ep=1→0.33, ep=3→1
                    reg = 0.0
                    for p in model.parameters():
                        if getattr(p, 'regularizer', 0):
                            reg += p.regularizer * (p**2).mean()
                    loss = (core + reg_factor * reg) / ACCUM              # ③ 再除以 ACCUM

                loss.backward()

                # 累積梯度：每 ACCUM 次更新一次
                if step % ACCUM == 0 or step == len(bar):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                    optim.step()
                    # 追加保險：強制正值 & 上下限，避免數值爆
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if 'lsigmoid' in name and 'raw' in name:
                                # raw 本身可無限大、但若你怕梯度爆，可限制：(-10, 10) 左右
                                p.clamp_(min=-8.0, max=8.0)
                    did_opt_step = True
                    optim.zero_grad(set_to_none=True)

                 # save indiv. losses
                ldict = getattr(model, "last_losses", {})
                time_l = float(ldict.get("time", 0.0))
                com_l  = float(ldict.get("com" , 0.0))
                con_l  = float(ldict.get("con" , 0.0))
                mr_l   = float(ldict.get("mr"  , 0.0))
                h_l    = float(ldict.get("entropy", 0.0))

                # ★ 累積核心四項（不含 reg）
                running_core += float(getattr(model, "last_report_loss", core).item())
                bar.set_postfix(
                    L_core=f"{running_core/step:.4g}",  # running 就是四項平均
                    t=f"{time_l:.3e}", c=f"{com_l:.3e}",
                        n=f"{con_l:.3e}", mr=f"{mr_l:.3e}", H=f"{h_l:.3e}"
                )

                # ---- 中途監控（可選；模組不存在則跳過） ----------
                # 取代原來 PRINT_EVERY 塊
                if dbg_every and step % dbg_every == 0:
                    with torch.no_grad():
                        # ❶ 重新 forward 一次取得 feat (不用回傳值)
                        dbg_x = noisy_b[:1].permute(0, 1, 3, 2)   # [1,2,T,F]
                        feat = model.dense_encoder(dbg_x)
                        for blk in model.TSMamba:
                            feat = blk(feat)

                        mask = model.mask_decoder(feat)           # [1,1,T,F]
                        slope = model.mask_decoder.lsigmoid._positive_slope()   # 取實際 >0 斜率

                        slope_mean = slope.mean().item()
                        slope_min = slope.min().item()
                        slope_max = slope.max().item()
                        current_lr = optim.param_groups[0]['lr']
                        H_dbg      = h_l


                        print(
                            f"[dbg] E{ep} S{step} | "
                            f"loss={running_core/step:.4g} | "
                            f"mask μ={mask.mean():.3f} σ={mask.std():.3f} max={mask.max():.3f} | "
                            f"slope μ={slope_mean:.3f} min={slope_min:.3f} max={slope_max:.3f} | "
                            f"lr={current_lr:.2e} | H={H_dbg:.3f}"
                        ) 


        # （建議）在每個 epoch 結束後，快速檢查權重是否有變動
        try:
            with torch.no_grad():
                checksum = 0.0
                for p in model.parameters():
                    checksum += p.detach().abs().sum().item()
            print(f"[dbg] epoch {ep} param |W|_1 checksum = {checksum:.6e}")
        except Exception:
            pass

        # ===== validation ====================================================
        model.eval()
        val_loss = 0.0
        va_time = va_com = va_con = va_mr = va_H = 0.0
        n_val_batches = 0
        val_mf_sum = 0.0
        n_val_samples = 0

        vctx = nullcontext()
        with torch.no_grad(), vctx:
            for clean_b, noisy_b in valid_loader:
                clean_b = clean_b.to(device, non_blocking=True)
                noisy_b = noisy_b.to(device, non_blocking=True)
                core = model(clean_b, noisy_b)
                val_loss += float(getattr(model, "last_report_loss", core).item())

                # 取各分項（模型 forward 會填 last_losses）
                ldict = getattr(model, "last_losses", {})
                va_time += float(ldict.get("time", 0.0))
                va_com  += float(ldict.get("com",  0.0))
                va_con  += float(ldict.get("con",  0.0))
                va_mr   += float(ldict.get("mr",   0.0))
                va_H    += float(ldict.get("entropy", 0.0))
                n_val_batches += 1

                # --- 取波形來估計 MF 誤差（模型 forward 會填 last_wavs） ---
                last_wavs = getattr(model, "last_wavs", None)
                if last_wavs is not None:
                    wav_p = last_wavs["pred"]   # [B,T] on device
                    wav_c = last_wavs["clean"]  # [B,T] on device
                    # 在同一 device 上做 STFT（較快）
                    mf_err = _batch_mf_err(wav_p, wav_c, cfg.get('metrics', {}))
                    val_mf_sum += mf_err * wav_p.shape[0]
                    n_val_samples += wav_p.shape[0]

        den_batches = max(1, n_val_batches)
        val_loss /= den_batches
        va_time  /= den_batches
        va_com   /= den_batches
        va_con   /= den_batches
        va_mr    /= den_batches
        va_H     /= den_batches
        val_mf   = (val_mf_sum / max(1, n_val_samples)) if n_val_samples > 0 else float('nan')

        # ---- 監控 slope (learnable-sigmoid) --------------------------------
        try:
            slope_val = model.mask_decoder.lsigmoid._positive_slope().mean().item()
        except Exception:
            slope_val = float('nan')

        train_loss_epoch = running_core / max(1, len(train_loader))   # ★ 只四項
        print(f"★ Epoch {ep} done | train={train_loss_epoch:.4f} | val={val_loss:.4f} | "
              f"va: time={va_time:.3e}, com={va_com:.3e}, con={va_con:.3e}, mr={va_mr:.3e}, H={va_H:.3e} | "
              f"val_mf={val_mf:.4f} Hz | slope={slope_val:.6f}")

        # ---- CSV log ---------------------------------------------------------
        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                ep, train_loss_epoch, val_loss,
                time_l, com_l, con_l, mr_l, h_l,         # train（最後一個 batch 快照）
                va_time, va_com, va_con, va_mr, va_H,     # valid（epoch 平均）
                val_mf, slope_val
            ])

        # ---- best by val_loss ------------------------------------------------
        if val_loss < best_val:
            best_val = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save)
            print(f"✓ new best (val={best_val:.4f}) saved to {model_save}")
        else:
            no_improve_epochs += 1
            print(f"✗ no improvement ({no_improve_epochs}/{PATIENCE})")

        # ---- best by MF ------------------------------------------------------
        if not math.isnan(val_mf) and val_mf < best_mf:
            best_mf = val_mf
            torch.save(model.state_dict(), best_by_mf_path)
            print(f"✓ new best MF (val_mf={best_mf:.4f} Hz) saved to {best_by_mf_path}")

        prev_lr = optim.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optim.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f" >>> lr reduced: {prev_lr:.2e} → {new_lr:.2e}")
        else:
            print(f" >>> lr stays : {new_lr:.2e}")

        # 每個 epoch 都覆蓋一份 "last"
        last_path = model_save.replace('_weights.pth', '_last.pth')
        torch.save(model.state_dict(), last_path)

        if no_improve_epochs >= PATIENCE:
            print(f"⏹ Early stopping: val_loss did not improve for {PATIENCE} consecutive epochs.")
            break

# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    import yaml, argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='yaml config path')
    p.add_argument('--data_root', default='.', help='where *_spectrogram.pt are')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_training(cfg, args.data_root)
