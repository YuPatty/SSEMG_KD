# pipeline_distill_crossarch.py — Teacher = SSEMGNet (TF-Bi-Mamba), Student = StudentSSEMGNet (純 CNN)
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
                                use_feature_noise=False,
                                noise_std_start=0.5,
                                noise_std_end=0.0,
                                noise_schedule='linear',
                                use_layer_curriculum=False,
                                schedule_epochs=None,   # None → fallback to total_epochs
                                use_similarity=False,
                                w_similarity=1.0,
                                similarity_student_idx=-1,
                                similarity_teacher_idx=-1,
                                use_swa=False, swa_dir=None,
                                seed=None,
                                resume=False,
                                resume_path=None,
                                use_amp=False,
                                num_workers=0,
                                cudnn_benchmark=True):
    if resp_weights is None:
        resp_weights = {"mask": 1.0, "mag": 1.0, "pha": 1.0, "com": 1.0}

    # 固定隨機種子（須放在任何模型/DataLoader 建立之前）
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[seed] 已固定 random/numpy/torch seed = {seed}")

    if schedule_epochs is None:
        schedule_epochs = total_epochs
    if schedule_epochs > total_epochs:
        print(f"[warning] --schedule_epochs ({schedule_epochs}) > --epochs ({total_epochs})，"
              f"排程永遠走不完，自動裁切成 {total_epochs}")
        schedule_epochs = total_epochs
    if schedule_epochs != total_epochs:
        print(f"[Schedule] 排程長度={schedule_epochs} epoch，訓練總長度={total_epochs} epoch。"
              f"epoch > {schedule_epochs} 之後，alpha/noise/layer curriculum 全部凍結在排程終點值，"
              f"純粹用該終點設定繼續訓練（不會再變化）。")

    device = auto_select_gpu()

    # cuDNN autotuning（不影響計算結果，只挑最快的 kernel 實作）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cudnn_benchmark:
            print("[speed] 已啟用 torch.backends.cudnn.benchmark=True（固定 shape 下自動挑最快 kernel）")

    # AMP 混合精度（FP16 forward + FP32 GradScaler）
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))
    if use_amp:
        if device.type == 'cuda':
            print(f"[speed] 已啟用 AMP混合精度訓練 (autocast dtype={amp_dtype})")
        else:
            print("[speed] --use_amp 有指定，但目前裝置是 CPU，AMP 只在 CUDA 上有意義，已自動略過")

    teacher = None
    projector = None
    teacher_state_dicts = None

    if not no_kd:
        teacher = SSEMGNet(teacher_cfg).to(device)
        if use_trajectory and teacher_ckpt_list is not None:
            teacher_state_dicts = [torch.load(p, map_location='cpu', mmap=True) for p in teacher_ckpt_list]
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

    # resume_path 預設與 model_save 同目錄，檔名不同（避免與純 state_dict 混淆）
    if resume_path is None:
        base = model_save[:-4] if model_save.endswith('.pth') else model_save
        resume_path = base + '.resume_state.pth'

    start_epoch = 1
    best_val = float('inf')
    no_improve = 0
    resumed = False

    if resume and os.path.exists(resume_path):
        print(f"[resume] 找到 {resume_path}，從中接續訓練狀態...")
        state = torch.load(resume_path, map_location=device)
        student.load_state_dict(state['student_state_dict'])
        if projector is not None and state.get('projector_state_dict') is not None:
            projector.load_state_dict(state['projector_state_dict'])
        optim.load_state_dict(state['optim_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        if use_amp and state.get('scaler_state_dict') is not None:
            scaler.load_state_dict(state['scaler_state_dict'])
        best_val = state['best_val']
        no_improve = state['no_improve']
        start_epoch = state['epoch'] + 1
        # 還原 RNG 狀態以接續訓練軌跡
        if 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.type == 'cuda' and 'cuda_rng_state' in state and state['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng_state'].cpu(), device=device)
        if 'numpy_rng_state' in state and state['numpy_rng_state'] is not None:
            import numpy as np
            np.random.set_state(state['numpy_rng_state'])
        if 'python_rng_state' in state and state['python_rng_state'] is not None:
            import random
            random.setstate(state['python_rng_state'])
        resumed = True
        print(f"[resume] 已還原到 epoch {state['epoch']} 結束時的狀態，"
              f"從 epoch {start_epoch} 繼續（best_val={best_val:.4f}, no_improve={no_improve}）")
    elif resume:
        print(f"[resume] --resume 有指定，但 {resume_path} 不存在，視為全新訓練從 epoch 1 開始")

    X_tr, y_tr = load_dataset('train', data_root)
    X_va, y_va = load_dataset('valid', data_root)
    use_pin = (device.type == 'cuda')

    train_loader = DataLoader(TensorDataset(y_tr, X_tr), batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))
    valid_loader = DataLoader(TensorDataset(y_va, X_va), batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers, pin_memory=use_pin,
                              persistent_workers=(num_workers > 0))

    os.makedirs(os.path.dirname(model_save), exist_ok=True)

    if use_swa:
        if swa_dir is None:
            base = model_save[:-4] if model_save.endswith('.pth') else model_save
            swa_dir = base + '_swa_ckpts'
        os.makedirs(swa_dir, exist_ok=True)
        print(f"[SWA] 排程走完後每個 epoch 的 checkpoint 會存到 {swa_dir}/")

    # resume 時用 append 模式接續寫，避免清空既有訓練曲線
    if resumed:
        log_csv_mode = 'a'
        print(f"[resume] {log_csv} 用 append 模式接續寫入，不會清空先前紀錄")
    else:
        log_csv_mode = 'w'
    if log_csv_mode == 'w':
        with open(log_csv, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'train_gt', 'train_kd', 'alpha', 'val_gt', 'val_mf',
                 'resp_mask', 'resp_mag', 'resp_pha', 'resp_com', 'feat_loss', 'rel_loss',
                 'noise_std', 'sim_loss']
            )

    # 預先取得 teacher 總層數，供層級課程使用
    teacher_total_layers = teacher_cfg['model']['num_tscblocks'] if not no_kd else 4

    if resumed and no_improve >= patience:
        print(f"[resume] 還原後的 no_improve({no_improve}) 已經 >= patience({patience})，"
              f"代表上次中斷前其實已經該觸發 early stopping 了，這裡直接結束、不再繼續訓練。")
        return

    for ep in range(start_epoch, total_epochs + 1):
        student.train()
        if projector is not None:
            projector.train()
        running_gt, running_kd = 0.0, 0.0
        running_info = {"resp_mask": 0.0, "resp_mag": 0.0, "resp_pha": 0.0,
                        "resp_com": 0.0, "feat_loss": 0.0, "rel_loss": 0.0,
                        "sim_loss": 0.0}

        # 排程用的有效 epoch 封頂在 schedule_epochs（超過後凍結在排程終點）
        sched_ep = min(ep, schedule_epochs)

        # ── 計算當前 noise_std ──
        if use_feature_noise:
            cur_noise_std = get_noise_std(sched_ep, schedule_epochs, noise_std_start,
                                          noise_std_end, noise_schedule)
        else:
            cur_noise_std = 0.0

        # ── 計算當前層級對齊索引 ──
        if use_layer_curriculum:
            all_layers = list(range(teacher_total_layers))
            cur_teacher_indices = get_layer_curriculum_indices(sched_ep, schedule_epochs, all_layers)
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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == 'cuda')):
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
                            noise_std=cur_noise_std,
                            use_similarity=use_similarity, w_similarity=w_similarity,
                            similarity_student_idx=similarity_student_idx,
                            similarity_teacher_idx=similarity_teacher_idx)
                        total_loss = kd_loss
                        alpha = 1.0
                    else:
                        total_loss = gt_loss
                        kd_loss = torch.tensor(0.0, device=device)
                        alpha = 0.0
                        info = {}
                elif use_trajectory and teacher_state_dicts is not None:
                    active_idx = get_trajectory_active_indices(
                        sched_ep, schedule_epochs, len(teacher_state_dicts),
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
                        alpha = annealed_alpha(sched_ep, schedule_epochs,
                                               schedule=anneal_schedule, exp_k=anneal_exp_k)
                    else:
                        alpha = fixed_alpha
                    kd_loss = traj_kd
                    total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss
                    info = {"kd_loss": float(kd_loss.detach()), "alpha": alpha, "noise_std": cur_noise_std}
                else:
                    # 排程結束且 alpha=0 時跳過 teacher forward，退化成純 GT loss 微調
                    skip_kd_this_step = False
                    if kd_weight_mode == 'annealed':
                        alpha_probe = annealed_alpha(sched_ep, schedule_epochs,
                                                     schedule=anneal_schedule, exp_k=anneal_exp_k)
                        if sched_ep >= schedule_epochs and alpha_probe <= 1e-8:
                            skip_kd_this_step = True

                    if skip_kd_this_step:
                        total_loss = gt_loss
                        kd_loss = torch.tensor(0.0, device=device)
                        alpha = 0.0
                        info = {}
                    else:
                        with torch.no_grad():
                            teacher(clean_b, noisy_b)
                        if kd_weight_mode == 'annealed':
                            kd_loss, alpha, info = compute_kd_loss_annealed(
                                student, teacher, epoch=sched_ep, total_epochs=schedule_epochs, projector=projector,
                                w_feature=w_feature, w_response=w_response,
                                include_relation=include_relation, w_relation=w_relation,
                                resp_weights=resp_weights, schedule=anneal_schedule, exp_k=anneal_exp_k,
                                teacher_layer_selection=cur_teacher_indices,
                                feature_layer_weights=feature_layer_weights,
                                noise_std=cur_noise_std,
                                use_similarity=use_similarity, w_similarity=w_similarity,
                                similarity_student_idx=similarity_student_idx,
                                similarity_teacher_idx=similarity_teacher_idx)
                        else:
                            kd_loss, alpha, info = compute_kd_loss(
                                student, teacher, projector=projector, alpha=fixed_alpha,
                                w_feature=w_feature, w_response=w_response,
                                include_relation=include_relation, w_relation=w_relation,
                                resp_weights=resp_weights,
                                teacher_layer_selection=cur_teacher_indices,
                                feature_layer_weights=feature_layer_weights,
                                noise_std=cur_noise_std,
                                use_similarity=use_similarity, w_similarity=w_similarity,
                                similarity_student_idx=similarity_student_idx,
                                similarity_teacher_idx=similarity_teacher_idx)
                        total_loss = (1.0 - alpha) * gt_loss + alpha * kd_loss

            # AMP 下用 GradScaler 縮放 loss 再 backward，避免 FP16 梯度 underflow
            scaler.scale(total_loss / accum).backward()

            if step % accum == 0 or step == len(train_loader):
                # clip 前需先 unscale_，否則梯度範數被 scaler 放大過
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad)
                scaler.step(optim)
                scaler.update()
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
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                              enabled=(use_amp and device.type == 'cuda')):
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
            row += [running_info["sim_loss"] / max(1, len(train_loader))]
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

        # 排程結束後每 epoch 額外存一份 checkpoint，供訓練後做 SWA 平均
        if use_swa and sched_ep >= schedule_epochs:
            swa_ckpt_path = os.path.join(swa_dir, f"epoch{ep}_valgt{val_gt:.5f}.pth")
            torch.save(student.state_dict(), swa_ckpt_path)

        # 每 epoch 存一份完整訓練狀態供 resume 用（覆蓋寫，只留最新一份）
        resume_state = {
            'epoch': ep,
            'student_state_dict': student.state_dict(),
            'projector_state_dict': (projector.state_dict() if projector is not None else None),
            'optim_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': (scaler.state_dict() if use_amp else None),
            'best_val': best_val,
            'no_improve': no_improve,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (torch.cuda.get_rng_state(device) if device.type == 'cuda' else None),
        }
        try:
            import numpy as np
            resume_state['numpy_rng_state'] = np.random.get_state()
        except Exception:
            resume_state['numpy_rng_state'] = None
        try:
            import random
            resume_state['python_rng_state'] = random.getstate()
        except Exception:
            resume_state['python_rng_state'] = None
        torch.save(resume_state, resume_path)

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
    p.add_argument('--schedule_epochs', type=int, default=None,
                   help='alpha/noise/layer curriculum 排程長度；預設等於 --epochs（向下相容）。'
                        '設成比 --epochs 小的值，可以讓排程照原節奏走完後，'
                        '用排程終點設定繼續訓練，不影響已驗證過的排程動態。')
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

    # SWA checkpoint 收集
    p.add_argument('--swa', action='store_true',
                   help='排程走完後（sched_ep >= schedule_epochs）每個 epoch 額外存一份 '
                        'checkpoint 到 --swa_dir，供訓練結束後做權重平均。不影響原本 '
                        'best-checkpoint 的存法，是額外多存，兩者互不影響。')
    p.add_argument('--swa_dir', default=None,
                   help='SWA checkpoint 存放資料夾，預設是 model_save 去掉副檔名 + "_swa_ckpts"')

    p.add_argument('--trajectory', action='store_true')
    p.add_argument('--teacher_ckpt_list', nargs='+', default=None)
    p.add_argument('--trajectory_weights', nargs='+', type=float, default=None)
    p.add_argument('--trajectory_curriculum', choices=['all', 'linear_grow'], default='linear_grow')

    p.add_argument('--feature_noise', action='store_true',
                   help='對 teacher 中間層特徵加噪，並隨 epoch 衰減（模擬 Teacher Trajectory）')
    p.add_argument('--noise_std_start', type=float, default=0.5,
                   help='初始噪聲標準差（預設 0.5）')
    p.add_argument('--noise_std_end', type=float, default=0.0,
                   help='最終噪聲標準差（預設 0.0）')
    p.add_argument('--noise_schedule', choices=['linear', 'cosine', 'exponential'], default='linear',
                   help='噪聲衰減曲線（同 alpha 排程）')

    p.add_argument('--layer_curriculum', action='store_true',
                   help='啟用層級課程：前期只用淺層 teacher，逐步加入深層')

    # Similarity-Preserving KD
    p.add_argument('--similarity_preserving', action='store_true',
                   help='啟用 Similarity-Preserving KD（batch 內樣本兩兩相似度矩陣對齊，'
                        '不需要 FeatureProjector、不受 block 數限制）')
    p.add_argument('--w_similarity', type=float, default=1.0,
                   help='Similarity-Preserving loss 權重（預設 1.0，未經調參，建議自行掃描）')
    p.add_argument('--similarity_student_idx', type=int, default=-1,
                   help='取學生 last_feats 的第幾層算相似度矩陣，預設 -1（最後一層）')
    p.add_argument('--similarity_teacher_idx', type=int, default=-1,
                   help='取 teacher 全部 feats 的第幾層算相似度矩陣，預設 -1（最深層）')
    p.add_argument('--seed', type=int, default=None,
                   help='固定 random/numpy/torch 的隨機種子，讓重跑盡量逼近同一次訓練軌跡。'
                        '不指定時完全不設 seed(向下相容)。GPU 上某些 cuDNN kernel 本身非'
                        '完全確定性，設了 seed 只能大幅縮小、不能保證 100% 消除重跑間的差異。')

    p.add_argument('--resume', action='store_true',
                   help='啟用後，如果 --resume_path 指定的檔案存在，會從裡面還原 model/'
                        'optimizer/scheduler/epoch/best_val/no_improve/RNG 狀態接續訓練；'
                        '檔案不存在則視同全新訓練（不報錯）。這個檔案跟 --model_save 存的'
                        '純 state_dict 是分開的兩個檔案，--model_save 那份格式維持不變，'
                        '可以繼續直接餵給 inference_student.py。')
    p.add_argument('--resume_path', default=None,
                   help='resume 用的完整訓練狀態檔案路徑；不指定時預設是 '
                        '--model_save 去掉副檔名 + ".resume_state.pth"')

    # 加速選項（不改變計算結果）
    p.add_argument('--use_amp', action='store_true',
                   help='啟用 AMP 混合精度訓練 (FP16 autocast + GradScaler)。'
                        '不改變計算圖或 batch 組成，只是數值精度略有誤差，'
                        '通常遠小於訓練本身的隨機性雜訊。只在 CUDA 上有效果，'
                        'CPU 上會自動略過。')
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader 平行預取的 worker 數量，預設 0（沿用舊行為）。'
                        '純粹是資料怎麼餵進去的機制，不影響訓練結果。因為資料是'
                        '一次性整包 load 進記憶體，效益可能有限，但改了無害。')
    p.add_argument('--no_cudnn_benchmark', action='store_true',
                   help='關閉 torch.backends.cudnn.benchmark（預設是開啟的）。'
                        '如果需要追求 bit-exact 重現性可以關掉，但這個非確定性'
                        '通常遠小於本專案已經在容忍的其他雜訊來源。')

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
        patience=args.patience,
        schedule_epochs=args.schedule_epochs,
        use_similarity=args.similarity_preserving,
        w_similarity=args.w_similarity,
        similarity_student_idx=args.similarity_student_idx,
        similarity_teacher_idx=args.similarity_teacher_idx,
        use_swa=args.swa,
        swa_dir=args.swa_dir,
        seed=args.seed,
        resume=args.resume,
        resume_path=args.resume_path,
        use_amp=args.use_amp,
        num_workers=args.num_workers,
        cudnn_benchmark=(not args.no_cudnn_benchmark)
    )
