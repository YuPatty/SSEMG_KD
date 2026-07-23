# ────────────────────────────────────────────────────
# pipeline_distill_crossarch.py
# 跨架構版：Teacher = SSEMGNet (TF-Bi-Mamba, 需 mamba_ssm/CUDA)
#           Student = StudentSSEMGNet (depthwise separable conv, 無 mamba_ssm 依賴)
#
# 跟 pipeline_distill.py（同架構版）的差異：
#   - Student 換成 StudentSSEMGNet，import 不會觸發 mamba_ssm
#   - 其餘（teacher 凍結、distill_loss.py 的 response+feature+annealing）完全共用
#
# 使用方式：
#   python pipeline_distill_crossarch.py \
#       --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
#       --teacher_weights model_weight/xxx_weights.pth \
#       --student_config config/config_student_crossarch.yaml \
#       --data_root dataset
# ────────────────────────────────────────────────────
import os, sys, csv, argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pipeline_spectrogram import load_dataset, auto_select_gpu, _batch_mf_err

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet            # noqa: E402  # teacher，需要 mamba_ssm
from models.StudentNet import StudentSSEMGNet   # noqa: E402  # student，不需要 mamba_ssm

from distill_loss import FeatureProjector, compute_kd_loss, compute_kd_loss_annealed


def run_crossarch_distillation(teacher_cfg, student_cfg, teacher_weights_path, data_root,
                                total_epochs=30, batch_size=4, accum=4, lr=3e-4,
                                clip_grad=5.0, patience=5, log_csv='distill_crossarch_log.csv',
                                model_save='model_weight/student_crossarch.pth',
                                kd_weight_mode='fixed', fixed_alpha=0.5,
                                w_feature=0.3, include_relation=False, w_relation=0.3,
                                resp_weights=None, no_kd=False):
    """
    no_kd: True 時完全不建構/載入 teacher，student 只用自己的 GT loss 訓練
           （total_loss = gt_loss，相當於 alpha 恆為 0）。用來回答「student
           架構不靠 teacher 蒸餾，自己能學到多好」這個 baseline 問題——
           如果 KD 版本相對這個 baseline 沒有明顯提升，代表蒸餾本身沒有
           真的發揮作用，架構縮小才是主要因素。
    kd_weight_mode: 'fixed'（EDA 論文精神，alpha 固定）或 'annealed'（PreFallKD 精神，
                    alpha 隨 epoch 遞減）。哪個比較好目前沒有定論，建議兩種都跑一次，
                    比較 log_csv 裡的 val_gt/val_mf 曲線，用結果決定，而不是預設哪個對。
    w_feature / include_relation / w_relation: 見 distill_loss.py 的說明，
                    數值都建議做 ablation（例如 w_feature ∈ {0.1, 0.3, 0.5}）。
    resp_weights: dict，{"mask":.., "mag":.., "pha":.., "com":..}，用來平衡四個
                    response 分量的梯度貢獻量級（實測發現不加權時 phase 分量會主導，
                    mask/mag 幾乎被蓋過）。預設 None 時 distill_loss.py 內部全部設 1.0。
    """
    if resp_weights is None:
        resp_weights = {"mask": 1.0, "mag": 1.0, "pha": 1.0, "com": 1.0}

    device = auto_select_gpu()

    # ── Teacher：no_kd=True 時完全跳過，不建構、不載入、不佔顯存 ──────
    teacher = None
    projector = None
    if not no_kd:
        teacher = SSEMGNet(teacher_cfg).to(device)
        teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # ── Student：StudentSSEMGNet（跨架構）──────────────────
    student = StudentSSEMGNet(student_cfg).to(device)

    n_params_s = sum(p.numel() for p in student.parameters())
    if teacher is not None:
        n_params_t = sum(p.numel() for p in teacher.parameters())
        print(f"[CrossKD] Teacher(Mamba) params: {n_params_t/1e6:.2f}M | "
              f"Student(Conv) params: {n_params_s/1e6:.2f}M | "
              f"Compression: {n_params_t/max(1,n_params_s):.2f}x")
    else:
        print(f"[NoKD baseline] Student(Conv) params: {n_params_s/1e6:.2f}M "
              f"| training WITHOUT teacher, GT loss only")

    # ── Feature projector：no_kd=True 時不需要（沒有 teacher 特徵可對齊）──
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
             'resp_mask', 'resp_mag', 'resp_pha', 'resp_com', 'feat_loss', 'rel_loss']
        )

    best_val = float('inf')
    no_improve = 0

    for ep in range(1, total_epochs + 1):
        student.train()
        if projector is not None:
            projector.train()
        running_gt, running_kd = 0.0, 0.0
        running_info = {"resp_mask": 0.0, "resp_mag": 0.0, "resp_pha": 0.0,
                        "resp_com": 0.0, "feat_loss": 0.0, "rel_loss": 0.0}
        mode_tag = "NoKD" if no_kd else f"CrossKD:{kd_weight_mode}"
        bar = tqdm(train_loader, desc=f"[{mode_tag}] epoch {ep}/{total_epochs}")

        for step, (clean_b, noisy_b) in enumerate(bar, start=1):
            clean_b = clean_b.to(device, non_blocking=True)
            noisy_b = noisy_b.to(device, non_blocking=True)

            gt_loss = student(clean_b, noisy_b)

            if no_kd:
                # ── baseline：完全不碰 teacher，student 自己用 GT loss 學 ──
                total_loss = gt_loss
                kd_loss = torch.tensor(0.0, device=device)
                alpha = 0.0
                info = {}
            else:
                with torch.no_grad():
                    teacher(clean_b, noisy_b)

                if kd_weight_mode == 'annealed':
                    kd_loss, alpha, info = compute_kd_loss_annealed(
                        student, teacher, epoch=ep, total_epochs=total_epochs, projector=projector,
                        w_feature=w_feature, include_relation=include_relation, w_relation=w_relation,
                        resp_weights=resp_weights
                    )
                else:  # 'fixed'
                    kd_loss, alpha, info = compute_kd_loss(
                        student, teacher, projector=projector, alpha=fixed_alpha,
                        w_feature=w_feature, include_relation=include_relation, w_relation=w_relation,
                        resp_weights=resp_weights
                    )
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
                if v is not None:
                    running_info[k] += v
            bar.set_postfix(gt=f"{running_gt/step:.4g}", kd=f"{running_kd/step:.4g}", alpha=f"{alpha:.2f}")

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
        # 取出該圈平均的子項 loss
        avg_info = {k: v / max(1, len(train_loader)) for k, v in running_info.items()}

        print(f"★ Epoch {ep} | train_gt={train_gt_epoch:.4f} | train_kd={train_kd_epoch:.4f} "
              f"| alpha={alpha:.2f} | val_gt={val_gt:.4f} | val_mf={val_mf:.4f} Hz")

        # =========================================================================
        # 🔥 【自動防呆檢查與佔比監控儀表板】 🔥
        # 將 check_loss_balance.py 邏輯內嵌，動態換算加權後的真實貢獻百分比
        # =========================================================================
        w_rel_actual = w_relation if include_relation else 0.0
        weighted_vals = {
            'mask': avg_info['resp_mask'] * resp_weights['mask'],
            'mag':  avg_info['resp_mag']  * resp_weights['mag'],
            'pha':  avg_info['resp_pha']  * resp_weights['pha'],
            'com':  avg_info['resp_com']  * resp_weights['com'],
            'feat': avg_info['feat_loss'] * w_feature,
            'rel':  avg_info['rel_loss']  * w_rel_actual
        }
        total_weighted = sum(weighted_vals.values())
        
        if total_weighted > 0:
            pcts = {k: (v / total_weighted) * 100 for k, v in weighted_vals.items()}
            print(f"   [Gradient Balance] Mask:{pcts['mask']:.1f}% | Mag:{pcts['mag']:.1f}% | Pha:{pcts['pha']:.1f}% | Com:{pcts['com']:.1f}% | Feat:{pcts['feat']:.1f}% | Rel:{pcts['rel']:.1f}%")
            
            # 針對第一圈進行嚴格的斷頭防呆
            if ep == 1:
                max_comp = max(pcts, key=pcts.get)
                # 設定 75% 為暴君門檻，您可以依據容忍度調整
                if pcts[max_comp] > 75.0:
                    print(f"\n🚨 [嚴重警告] 第 1 圈檢測到 '{max_comp}' 佔據了 {pcts[max_comp]:.1f}% 的梯度，超過 75% 安全閾值！")
                    print("🚨 該項權重設定過高，已嚴重排擠其他特徵的學習空間。")
                    print("🚨 為避免浪費算力白跑 30 圈，訓練已自動終止 (sys.exit)。請調降該項權重後重新執行。")
                    sys.exit(1)
        # =========================================================================

        with open(log_csv, 'a', newline='') as f:
            row = [ep, train_gt_epoch, train_kd_epoch, alpha, val_gt, val_mf]
            row += [avg_info[k] for k in ["resp_mask", "resp_mag", "resp_pha", "resp_com", "feat_loss", "rel_loss"]]
            csv.writer(f).writerow(row)

        scheduler.step(val_gt)

        if val_gt < best_val:
            best_val = val_gt
            no_improve = 0
            torch.save(student.state_dict(), model_save)
            print(f"✓ new best student (val_gt={best_val:.4f}) saved to {model_save}")
        else:
            no_improve += 1
            print(f"✗ no improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_config', required=True,
                    help="即使 --no_kd 也需要提供（只用來讀 dense_channel 等共用設定，"
                         "不會真的建構/載入 teacher 權重）")
    p.add_argument('--teacher_weights', default=None,
                    help="--no_kd 時可省略；否則必填")
    p.add_argument('--student_config', required=True)
    p.add_argument('--data_root', default='dataset')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--no_kd', action='store_true',
                    help="Baseline：完全不用 teacher，student 只靠自己的 GT loss 訓練。"
                         "用來回答『不靠蒸餾，student 架構自己能學多好』這個問題，"
                         "拿來跟 KD 版本比較才能證明蒸餾本身有沒有實質貢獻。")
    # ── §②④ 消融實驗參數：不要假設預設值就是最好的，實際掃過再決定 ──
    p.add_argument('--kd_weight_mode', choices=['fixed', 'annealed'], default='fixed',
                    help="GT/KD 權重策略：fixed(EDA論文精神) 或 annealed(PreFallKD精神)，"
                         "建議兩者都跑一次比較 val_gt/val_mf 曲線")
    p.add_argument('--fixed_alpha', type=float, default=0.5)
    p.add_argument('--w_feature', type=float, default=0.3,
                    help="建議掃 {0.1, 0.3, 0.5} 找出適合 SSEMG-Net teacher-student gap 的值")
    p.add_argument('--include_relation', action='store_true',
                    help="加入 relation-based loss（跨時間自相似矩陣對齊），"
                         "建議先跑一次 with/without 對照，尤其看低SNR樣本差異")
    p.add_argument('--w_relation', type=float, default=20.0,
                    help="實測發現 w_relation=0.3（EDA論文預設）時，relation loss 加權後"
                         "貢獻僅0.15~0.67%%，幾乎沒有實際作用。20.0 是從 epoch5/epoch10 "
                         "真實訓練數據反推、讓它跟其他項同量級的起點。"
                         "注意：rel_loss 衰減速度比其他項快（訓練中期後可能又變得不顯著），"
                         "這不是一勞永逸的解，建議訓練時持續監控 rel_loss 的加權貢獻比例。")
    # ── resp_weights：實測發現不加權時 phase 分量貢獻約60%，mask/mag 加起來僅5%（見對話紀錄），
    #    以下預設值是從 epoch5 真實 log 反推、讓四項貢獻大致相等的起點，非最終答案，
    #    建議重跑 ablation 驗證這組值是否真的改善 val_gt/val_mf，而不是照搬 ──
    p.add_argument('--w_resp_mask', type=float, default=3.5)
    p.add_argument('--w_resp_mag', type=float, default=5.0)
    p.add_argument('--w_resp_pha', type=float, default=0.3)
    p.add_argument('--w_resp_com', type=float, default=2.3)
    p.add_argument('--log_csv', default='distill_crossarch_log.csv')
    p.add_argument('--model_save', default='model_weight/student_crossarch.pth')
    args = p.parse_args()

    if not args.no_kd and args.teacher_weights is None:
        p.error("--teacher_weights 是必填的，除非你指定 --no_kd")

    with open(args.teacher_config) as f:
        teacher_cfg = yaml.safe_load(f)
    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    default_log = 'log_student_no_kd.csv' if args.no_kd else 'distill_crossarch_log.csv'
    default_save = 'model_weight/student_no_kd.pth' if args.no_kd else 'model_weight/student_crossarch.pth'
    log_csv = args.log_csv if args.log_csv != 'distill_crossarch_log.csv' else default_log
    model_save = args.model_save if args.model_save != 'model_weight/student_crossarch.pth' else default_save

    run_crossarch_distillation(
        teacher_cfg, student_cfg, args.teacher_weights, args.data_root,
        total_epochs=args.epochs, batch_size=args.batch_size, accum=args.accum,
        kd_weight_mode=args.kd_weight_mode, fixed_alpha=args.fixed_alpha,
        w_feature=args.w_feature, include_relation=args.include_relation,
        w_relation=args.w_relation, log_csv=log_csv, model_save=model_save,
        resp_weights={"mask": args.w_resp_mask, "mag": args.w_resp_mag,
                      "pha": args.w_resp_pha, "com": args.w_resp_com},
        no_kd=args.no_kd
    )
