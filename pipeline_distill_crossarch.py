# ────────────────────────────────────────────────────
# pipeline_distill_crossarch.py
# 跨架構版：Teacher = SSEMGNet (TF-Bi-Mamba), Student = StudentSSEMGNet (純 CNN)
# 新增功能：
#   --feature_noise : 啟用加噪衰減蒸餾（模擬 Teacher Trajectory）
#   --noise_std_start / --noise_std_end / --noise_schedule : 控制噪聲衰減
#   --layer_curriculum : 啟用層級課程，逐步加入更深層的 teacher 特徵
#   Teacher Trajectory 模式仍保留（需 --trajectory + checkpoint 列表）
# ────────────────────────────────────────────────────
import os, sys, csv, argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pipeline_spectrogram import load_dataset, auto_select_gpu, _batch_mf_err

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet
from models.StudentNet import StudentSSEMGNet

from distill_loss import (FeatureProjector, compute_kd_loss, compute_kd_loss_annealed,
                          annealed_alpha, get_trajectory_active_indices,
                          compute_trajectory_kd_batch, get_noise_std,
                          get_layer_curriculum_indices)


def run_crossarch_distillation(teacher_cfg, student_cfg, teacher_weights_path, data_root,
                                total_epochs=30, batch_size=4, accum=4, lr=3e-4,
                                clip_grad=5.0, patience=5, log_csv='distill_crossarch_log.csv',
                                model_save='model_weight/student_crossarch.pth',
                                kd_weight_mode='annealed', fixed_alpha=0.5,
                                w_feature=0.5, w_response=1.0,
                                include_relation=False, w_relation=20.0,
                                resp_weights=None, no_kd=False,
                                anneal_schedule='linear', anneal_exp_k=5.0,
                                teacher_layer_selection='first', 
                                feature_layer_weights=None,
                                two_stage=False, two_stage_ratio=0.7,
                                use_trajectory=False,
                                teacher_ckpt_list=None,
                                trajectory_weights=None,
                                trajectory_curriculum='linear_grow',
                                # 新增參數 ──
                                use_feature_noise=False,
                                noise_std_start=0.5,
                                noise_std_end=0.0,
                                noise_schedule='linear',
                                use_layer_curriculum=False):
    if resp_weights is None:
        resp_weights = {"mask": 1.0, "mag": 1.0, "pha": 1.0, "com": 1.0}

    device = auto_select_gpu()

    teacher = None
    projector = None
    teacher_state_dicts = None

    if not no_kd:
        teacher = SSEMGNet(teacher_cfg).to(device)
        if use_trajectory and teacher_ckpt_list is not None:
            teacher_state_dicts = [torch.load(p, map_location='cpu') for p in teacher_ckpt_list]
            print(f"[Trajectory] Loaded {len(teacher_state_dicts)} teacher checkpoints.")
        else:
            teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

    student = StudentSSEMGNet(student_cfg).to(device)

    n_params_s = sum(p.numel() for p in student.parameters())
    if teacher is not None and not use_trajectory:
        n_params_t = sum(p.numel() for p in teacher.parameters())
        print(f"[CrossKD] Teacher(Mamba) params: {n_params_t/1e6:.2f}M | "
              f"Student(Conv) params: {n_params_s/1e6:.2f}M | "
              f"Compression: {n_params_t/max(1,n_params_s):.2f}x")
    elif use_trajectory:
        print(f"[Trajectory KD] Teacher checkpoints: {len(teacher_state_dicts)} | "
              f"Student(Conv) params: {n_params_s/1e6:.2f}M")
    else:
        print(f"[NoKD baseline] Student(Conv) params: {n_params_s/1e6:.2f}M "
              f"| training WITHOUT teacher, GT loss only")

    if not no_kd:
        t_ch = teacher_cfg['model']['dense_channel']
        s_ch = student_cfg['model']['dense_channel']
        n_align_blocks = min(teacher_cfg['model']['num_tscblocks'],
                              student_cfg['model']['num_tscblocks'])
        projector = FeatureProjector(student_ch=s_ch, teacher_ch=t_ch,
                                      n_blocks=n_align_blocks).to(device)

    trainable_params = list(student.parameters())
    if projector is not None:
        trainable_params += list(projector.parameters())

    optim = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.8, 0.99), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    X_tr, y_tr = load_dataset('train', data_root)
    X_va, y_va = load_dataset('valid', data_root)
    num_workers = max(0, (os.cpu_count() or 0) // 2)
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(TensorDataset(y_tr, X_tr), batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))
    valid_loader = DataLoader(TensorDataset(y_va, X_va), batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))

    os.makedirs(os.path.dirname(model_save), exist_ok=True)
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_gt', 'train_kd', 'alpha', 'val_gt', 'val_mf',
             'resp_mask', 'resp_mag', 'resp_pha', 'resp_com', 'feat_loss', 'rel_loss',
             'noise_std']
        )

    best_val = float('inf')
    no_improve = 0

    # 預先取得 teacher 總層數，供層級課程使用
    teacher_total_layers = teacher_cfg['model']['num_tscblocks'] if not no_kd else 4

    for ep in range(1, total_epochs + 1):
        student.train()
        if projector is not None:
            projector.train()
        running_gt, running_kd = 0.0, 0.0
        running_info = {"resp_mask": 0.0, "resp_mag": 0.0, "resp_pha": 0.0,
                        "resp_com": 0.0, "feat_loss": 0.0, "rel_loss": 0.0}

        # ── 計算當前 noise_std ──
        if use_feature_noise:
            cur_noise_std = get_noise_std(ep, total_epochs, noise_std_start,
                                          noise_std_end, noise_schedule)
        else:
            cur_noise_std = 0.0

        # ── 計算當前層級對齊索引 ──
        if use_layer_curriculum:
            all_layers = list(range(teacher_total_layers))
            cur_teacher_indices = get_layer_curriculum_indices(ep, total_epochs, all_layers)
        else:
            cur_teacher_indices = teacher_layer_selection

        mode_tag = "NoKD" if no_kd else (
            "TrajKD" if use_trajectory else (
                f"CrossKD:{kd_weight_mode}" + ("+noise" if use_feature_noise else "") +
                ("+layerCur" if use_layer_curriculum else "")
            )
        )
        bar = tqdm(train_loader, desc=f"[{mode_tag}] epoch {ep}/{total_epochs}")

        for step, (clean_b, noisy_b) in enumerate(bar, start=1):
            clean_b = clean_b.to(device, non_blocking=True)
            noisy_b = noisy_b.to(device, non_blocking=True)

            gt_loss = student(clean_b, noisy_b)

            if no_kd:
                total_loss = gt_loss
                kd_loss = torch.tensor(0.0, device=device)
                alpha = 0.0
                info = {}
            elif two_stage:
                phase1_epochs = int(total_epochs * two_stage_ratio)
                if ep <= phase1_epochs:
                    with torch.no_grad():
                        teacher(clean_b, noisy_b)
                    kd_loss, _, info = compute_kd_loss(
                        student, teacher, projector=projector, alpha=1.0,
                        w_feature=w_feature, w_response=w_response,
                        include_relation=include_relation, w_relation=w_relation,
                        resp_weights=resp_weights,
                        teacher_layer_selection=cur_teacher_indices,
                        feature_layer_weights=feature_layer_weights,
                        noise_std=cur_noise_std)
                    total_loss = kd_loss
                    alpha = 1.0
                else:
                    total_loss = gt_loss
                    kd_loss = torch.tensor(0.0, device=device)
                    alpha = 0.0
                    info = {}
            elif use_trajectory and teacher_state_dicts is not None:
                active_idx = get_trajectory_active_indices(
                    ep, total_epochs, len(teacher_state_dicts),
                    mode=trajectory_curriculum
                )
                traj_kd, _ = compute_trajectory_kd_batch(
                    student, teacher, teacher_state_dicts,
                    clean_b, noisy_b,
                    projector,
                    trajectory_weights=trajectory_weights,
                    active_indices=active_idx,
                    w_feature=w_feature, w_response=w_response,
                    include_relation=include_relation,
                    w_relation=w_relation,
                    resp_weights=resp_weights,
                    teacher_layer_selection=cur_teacher_indices,
                    feature_layer_weights=feature_layer_weights,
                    device=device,
                    noise_std=cur_noise_std)
                if kd_weight_mode == 'annealed':
                    alpha = annealed_alpha(ep, total_epochs,
                                           schedule=anneal_schedule, exp_k=anneal_exp_k)
                else:
                    alpha = fixed_alpha
                kd_loss = traj_kd
                total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss
                info = {"kd_loss": float(kd_loss.detach()), "alpha": alpha, "noise_std": cur_noise_std}
            else:
                # 傳統單一 teacher + 混合蒸餾
                with torch.no_grad():
                    teacher(clean_b, noisy_b)
                if kd_weight_mode == 'annealed':
                    kd_loss, alpha, info = compute_kd_loss_annealed(
                        student, teacher, epoch=ep, total_epochs=total_epochs, projector=projector,
                        w_feature=w_feature, w_response=w_response,
                        include_relation=include_relation, w_relation=w_relation,
                        resp_weights=resp_weights, schedule=anneal_schedule, exp_k=anneal_exp_k,
                        teacher_layer_selection=cur_teacher_indices,
                        feature_layer_weights=feature_layer_weights,
                        noise_std=cur_noise_std)
                else:
                    kd_loss, alpha, info = compute_kd_loss(
                        student, teacher, projector=projector, alpha=fixed_alpha,
                        w_feature=w_feature, w_response=w_response,
                        include_relation=include_relation, w_relation=w_relation,
                        resp_weights=resp_weights,
                        teacher_layer_selection=cur_teacher_indices,
                        feature_layer_weights=feature_layer_weights,
                        noise_std=cur_noise_std)
                total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss

            (total_loss / accum).backward()

            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad)
                optim.step()
                optim.zero_grad(set_to_none=True)

            running_gt += float(gt_loss.detach())
            running_kd += float(kd_loss.detach())
            for k in running_info:
                v = info.get(k)
                running_info[k] += (v if v is not None else 0.0)
            bar.set_postfix(gt=f"{running_gt/step:.4g}", kd=f"{running_kd/step:.4g}",
                            alpha=f"{alpha:.2f}", noise=f"{cur_noise_std:.2f}")

        student.eval()
        val_gt, val_mf_sum, n_val_samples, n_batches = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for clean_b, noisy_b in valid_loader:
                clean_b = clean_b.to(device, non_blocking=True)
                noisy_b = noisy_b.to(device, non_blocking=True)
                gt_loss = student(clean_b, noisy_b)
                val_gt += float(gt_loss.detach())
                n_batches += 1
                last_wavs = getattr(student, "last_wavs", None)
                if last_wavs is not None:
                    mf_err = _batch_mf_err(last_wavs["pred"], last_wavs["clean"],
                                            student_cfg.get('metrics', {}))
                    val_mf_sum += mf_err * last_wavs["pred"].shape[0]
                    n_val_samples += last_wavs["pred"].shape[0]

        val_gt /= max(1, n_batches)
        val_mf = (val_mf_sum / n_val_samples) if n_val_samples > 0 else float('nan')
        train_gt_epoch = running_gt / max(1, len(train_loader))
        train_kd_epoch = running_kd / max(1, len(train_loader))

        print(f"★ Epoch {ep} | train_gt={train_gt_epoch:.4f} | train_kd={train_kd_epoch:.4f} "
              f"| alpha={alpha:.2f} | noise={cur_noise_std:.3f} | val_gt={val_gt:.4f} | val_mf={val_mf:.4f} Hz")

        with open(log_csv, 'a', newline='') as f:
            row = [ep, train_gt_epoch, train_kd_epoch, alpha, val_gt, val_mf]
            row += [running_info[k] / max(1, len(train_loader))
                    for k in ["resp_mask", "resp_mag", "resp_pha", "resp_com", "feat_loss", "rel_loss"]]
            row += [cur_noise_std]
            csv.writer(f).writerow(row)

        scheduler.step(val_gt)

        is_phase_1 = (two_stage and ep <= int(total_epochs * two_stage_ratio))

        if val_gt < best_val:
            best_val = val_gt
            no_improve = 0
            torch.save(student.state_dict(), model_save)
            print(f"✓ new best student (val_gt={best_val:.4f}) saved to {model_save}")
        else:
            if not is_phase_1:
                no_improve += 1
                print(f"✗ no improvement ({no_improve}/{patience})")
            else:
                print(f"ℹ [Phase 1] val_gt plateau expected")

        if not is_phase_1 and no_improve >= patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_config', required=True)
    p.add_argument('--teacher_weights', default=None)
    p.add_argument('--student_config', required=True)
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--no_kd', action='store_true')
    p.add_argument('--kd_weight_mode', choices=['fixed', 'annealed'], default='annealed')
    p.add_argument('--fixed_alpha', type=float, default=0.5)
    p.add_argument('--anneal_schedule', choices=['linear', 'cosine', 'exponential'], default='linear')
    p.add_argument('--anneal_exp_k', type=float, default=5.0)
    p.add_argument('--w_feature', type=float, default=0.5)
    p.add_argument('--w_response', type=float, default=1.0)
    p.add_argument('--include_relation', action='store_true')
    p.add_argument('--w_relation', type=float, default=20.0)
    p.add_argument('--teacher_indices', type=int, nargs='+', default=None)
    p.add_argument('--w_resp_mask', type=float, default=3.5)
    p.add_argument('--w_resp_mag', type=float, default=5.0)
    p.add_argument('--w_resp_pha', type=float, default=0.3)
    p.add_argument('--w_resp_com', type=float, default=2.3)
    p.add_argument('--log_csv', default='distill_crossarch_log.csv')
    p.add_argument('--model_save', default='model_weight/student_crossarch.pth')
    p.add_argument('--feature_layer_weights', type=float, nargs='+', default=None)
    p.add_argument('--two_stage', action='store_true')
    p.add_argument('--two_stage_ratio', type=float, default=0.7)
    p.add_argument('--patience', type=int, default=10, help='early stopping patience')

    # Trajectory 參數（原功能保留）
    p.add_argument('--trajectory', action='store_true')
    p.add_argument('--teacher_ckpt_list', nargs='+', default=None)
    p.add_argument('--trajectory_weights', nargs='+', type=float, default=None)
    p.add_argument('--trajectory_curriculum', choices=['all', 'linear_grow'], default='linear_grow')

    # 新增：退化教師蒸餾（加噪衰減）
    p.add_argument('--feature_noise', action='store_true',
                   help='對 teacher 中間層特徵加噪，並隨 epoch 衰減（模擬 Teacher Trajectory）')
    p.add_argument('--noise_std_start', type=float, default=0.5,
                   help='初始噪聲標準差（預設 0.5）')
    p.add_argument('--noise_std_end', type=float, default=0.0,
                   help='最終噪聲標準差（預設 0.0）')
    p.add_argument('--noise_schedule', choices=['linear', 'cosine', 'exponential'], default='linear',
                   help='噪聲衰減曲線（同 alpha 排程）')

    # 新增：層級課程
    p.add_argument('--layer_curriculum', action='store_true',
                   help='啟用層級課程：前期只用淺層 teacher，逐步加入深層')
    
    args = p.parse_args()

    if not args.no_kd and not args.trajectory and args.teacher_weights is None:
        p.error("必須指定 --teacher_weights（或使用 --no_kd 或 --trajectory）")

    with open(args.teacher_config) as f:
        teacher_cfg = yaml.safe_load(f)
    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    default_log = 'log_student_no_kd.csv' if args.no_kd else 'distill_crossarch_log.csv'
    default_save = 'model_weight/student_no_kd.pth' if args.no_kd else 'model_weight/student_crossarch.pth'
    log_csv = args.log_csv if args.log_csv != 'distill_crossarch_log.csv' else default_log
    model_save = args.model_save if args.model_save != 'model_weight/student_crossarch.pth' else default_save

    layer_selection = args.teacher_indices if args.teacher_indices is not None else 'first'

    run_crossarch_distillation(
        teacher_cfg, student_cfg, args.teacher_weights, args.data_root,
        total_epochs=args.epochs, batch_size=args.batch_size, accum=args.accum,
        kd_weight_mode=args.kd_weight_mode, fixed_alpha=args.fixed_alpha,
        w_feature=args.w_feature, w_response=args.w_response,
        include_relation=args.include_relation, w_relation=args.w_relation,
        log_csv=log_csv, model_save=model_save,
        resp_weights={"mask": args.w_resp_mask, "mag": args.w_resp_mag,
                      "pha": args.w_resp_pha, "com": args.w_resp_com},
        no_kd=args.no_kd, anneal_schedule=args.anneal_schedule, anneal_exp_k=args.anneal_exp_k,
        teacher_layer_selection=layer_selection,
        feature_layer_weights=args.feature_layer_weights,
        two_stage=args.two_stage, two_stage_ratio=args.two_stage_ratio,
        use_trajectory=args.trajectory,
        teacher_ckpt_list=args.teacher_ckpt_list,
        trajectory_weights=args.trajectory_weights,
        trajectory_curriculum=args.trajectory_curriculum,
        use_feature_noise=args.feature_noise,
        noise_std_start=args.noise_std_start,
        noise_std_end=args.noise_std_end,
        noise_schedule=args.noise_schedule,
        use_layer_curriculum=args.layer_curriculum,
        patience=args.patience
    )
