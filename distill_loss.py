# distill_loss.py — Knowledge-distillation loss for SSEMG-Net (Mamba teacher → cross-arch student)
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """1x1 conv：把 student 中間層通道數投影到 teacher 通道數，供逐層 MSE 對齊。"""
    def __init__(self, student_ch: int, teacher_ch: int, n_blocks: int):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(student_ch, teacher_ch, kernel_size=1)
            for _ in range(n_blocks)
        ])

    def forward(self, student_feats):
        assert len(student_feats) == len(self.projs), (
            f"student_feats has {len(student_feats)} blocks but projector "
            f"was built for {len(self.projs)} blocks"
        )
        return [proj(f) for proj, f in zip(self.projs, student_feats)]


def response_distill_loss(student_out: dict, teacher_out: dict):
    """Response-based KD loss：mask/mag/com 用 MSE，phase 用 1-cos(Δφ) circular distance。"""
    l_mask = F.mse_loss(student_out["mask"], teacher_out["mask"])
    l_mag = F.mse_loss(student_out["mag_g_FT"], teacher_out["mag_g_FT"])

    dphi = student_out["pha_g"] - teacher_out["pha_g"]
    l_pha = (1.0 - torch.cos(dphi)).mean()

    l_com = F.mse_loss(student_out["com_g"], teacher_out["com_g"])

    return {"mask": l_mask, "mag": l_mag, "pha": l_pha, "com": l_com}


def feature_distill_loss(student_feats, teacher_feats, projector: FeatureProjector,
                         layer_weights=None, noise_std=0.0):
    """Feature-based KD loss：逐層 MSE 加權加總；noise_std>0 時對 teacher 特徵加高斯噪聲（噪聲衰減蒸餾）。"""
    proj_feats = projector(student_feats)
    if layer_weights is None:
        layer_weights = [1.0] * len(proj_feats)
    assert len(layer_weights) == len(proj_feats), \
        f"layer_weights 長度({len(layer_weights)})需跟對齊層數({len(proj_feats)})一致"
    total = 0.0
    for sf, tf, w in zip(proj_feats, teacher_feats, layer_weights):
        # 對 teacher 特徵加噪
        if noise_std > 0:
            tf_noisy = tf + torch.randn_like(tf) * noise_std
        else:
            tf_noisy = tf
        total = total + w * F.mse_loss(sf, tf_noisy)
    return total


def relation_distill_loss(student_feats, teacher_feats):
    """Relation-based KD loss（time-flow / frequency-flow 雙流 Gram matrix，參考 I2SRF-TFCKD）。"""
    time_losses, freq_losses = [], []
    for sf, tf in zip(student_feats, teacher_feats):
        Bs, Cs, T, Fh = sf.shape
        Bt, Ct, _, _ = tf.shape

        # ── 時間流：[B, T, T] ──
        sf_t = sf.permute(0, 2, 1, 3).reshape(Bs, T, Cs * Fh)
        tf_t = tf.permute(0, 2, 1, 3).reshape(Bt, T, Ct * Fh)
        sf_t = F.normalize(sf_t, dim=-1, eps=1e-8)
        tf_t = F.normalize(tf_t, dim=-1, eps=1e-8)
        G_s_t = torch.bmm(sf_t, sf_t.transpose(1, 2))
        G_t_t = torch.bmm(tf_t, tf_t.transpose(1, 2))
        time_losses.append(F.mse_loss(G_s_t, G_t_t))

        # ── 頻率流：[B, F, F] ──
        sf_f = sf.permute(0, 3, 1, 2).reshape(Bs, Fh, Cs * T)
        tf_f = tf.permute(0, 3, 1, 2).reshape(Bt, Fh, Ct * T)
        sf_f = F.normalize(sf_f, dim=-1, eps=1e-8)
        tf_f = F.normalize(tf_f, dim=-1, eps=1e-8)
        G_s_f = torch.bmm(sf_f, sf_f.transpose(1, 2))
        G_t_f = torch.bmm(tf_f, tf_f.transpose(1, 2))
        freq_losses.append(F.mse_loss(G_s_f, G_t_f))

    time_loss = sum(time_losses) / len(time_losses)
    freq_loss = sum(freq_losses) / len(freq_losses)
    return time_loss + freq_loss


def similarity_preserving_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                                eps: float = 1e-8):
    """Similarity-Preserving KD（Tung & Mori, 2019）：batch 內樣本兩兩相似度矩陣 G=normalize(X)@normalize(X)^T 對齊（MSE），
    student/teacher 各取一層即可、形狀不需相同，不受 block 數限制，對 1-block 架構也有效。"""
    Bs = student_feat.shape[0]
    Bt = teacher_feat.shape[0]
    assert Bs == Bt, f"batch size 不一致：student={Bs}, teacher={Bt}"

    s_flat = student_feat.reshape(Bs, -1)
    t_flat = teacher_feat.reshape(Bt, -1)

    s_norm = F.normalize(s_flat, dim=1, eps=eps)
    t_norm = F.normalize(t_flat, dim=1, eps=eps)

    G_s = s_norm @ s_norm.t()  # [B, B]
    G_t = t_norm @ t_norm.t()  # [B, B]

    return F.mse_loss(G_s, G_t)


def compute_kd_loss(student, teacher, projector: FeatureProjector,
                     alpha: float = 0.5, w_feature: float = 0.5, w_response: float = 1.0,
                     include_relation: bool = False, w_relation: float = 20.0,
                     resp_weights: dict = None,
                     teacher_layer_selection: str = 'first', feature_layer_weights=None,
                     noise_std=0.0,
                     # ── 新增：Similarity-Preserving KD ──
                     use_similarity: bool = False, w_similarity: float = 1.0,
                     similarity_student_idx: int = -1, similarity_teacher_idx: int = -1):
    """kd_loss = w_response·L_response + w_feature·L_feature (+ w_relation·L_relation) (+ w_similarity·L_similarity)"""
    if resp_weights is None:
        resp_weights = {"mask": 1.0, "mag": 1.0, "pha": 1.0, "com": 1.0}

    teacher_out = {k: v.detach() for k, v in teacher.last_outputs.items()}
    all_teacher_feats = [f.detach() for f in teacher.last_feats]

    n_student_blocks = len(student.last_feats)
    if isinstance(teacher_layer_selection, list):
        teacher_feats = [all_teacher_feats[i] for i in teacher_layer_selection]
    elif teacher_layer_selection == 'first':
        teacher_feats = all_teacher_feats[:n_student_blocks]
    elif teacher_layer_selection == 'last':
        teacher_feats = all_teacher_feats[-n_student_blocks:]
    else:
        raise ValueError(f"未知的 teacher_layer_selection: {teacher_layer_selection}")
    
    resp_losses = response_distill_loss(student.last_outputs, teacher_out)
    resp_loss = (resp_weights["mask"] * resp_losses["mask"] +
                 resp_weights["mag"]  * resp_losses["mag"] +
                 resp_weights["pha"]  * resp_losses["pha"] +
                 resp_weights["com"]  * resp_losses["com"])

    feat_loss = feature_distill_loss(student.last_feats, teacher_feats, projector,
                                      layer_weights=feature_layer_weights,
                                      noise_std=noise_std)

    kd_loss = w_response * resp_loss + w_feature * feat_loss

    rel_loss = None
    if include_relation:
        rel_loss = relation_distill_loss(student.last_feats, teacher_feats)
        kd_loss = kd_loss + w_relation * rel_loss

    sim_loss = None
    if use_similarity:
        s_idx = similarity_student_idx if -len(student.last_feats) <= similarity_student_idx < len(student.last_feats) else -1
        t_idx = similarity_teacher_idx if -len(all_teacher_feats) <= similarity_teacher_idx < len(all_teacher_feats) else -1
        sim_loss = similarity_preserving_loss(student.last_feats[s_idx], all_teacher_feats[t_idx])
        kd_loss = kd_loss + w_similarity * sim_loss

    info = {
        "kd_loss": float(kd_loss.detach()),
        "resp_mask": float(resp_losses["mask"].detach()),
        "resp_mag": float(resp_losses["mag"].detach()),
        "resp_pha": float(resp_losses["pha"].detach()),
        "resp_com": float(resp_losses["com"].detach()),
        "feat_loss": float(feat_loss.detach()) if torch.is_tensor(feat_loss) else float(feat_loss),
        "rel_loss": (float(rel_loss.detach()) if rel_loss is not None else None),
        "sim_loss": (float(sim_loss.detach()) if sim_loss is not None else None),
        "alpha": alpha,
        "noise_std": noise_std,
    }
    return kd_loss, alpha, info


def annealed_alpha(epoch: int, total_epochs: int,
                    alpha_start: float = 1.0, alpha_end: float = 0.0,
                    schedule: str = 'linear', exp_k: float = 5.0) -> float:
    frac = min(1.0, epoch / max(1, total_epochs))
    if schedule == 'linear':
        shape = frac
    elif schedule == 'cosine':
        import math
        shape = 0.5 * (1 - math.cos(math.pi * frac))
    elif schedule == 'exponential':
        import math
        raw = 1 - math.exp(-exp_k * frac)
        norm = 1 - math.exp(-exp_k)
        shape = raw / norm if norm > 1e-9 else frac
    else:
        raise ValueError(f"未知的 schedule 類型: {schedule}")
    return alpha_start + (alpha_end - alpha_start) * shape


def compute_kd_loss_annealed(student, teacher, epoch: int, total_epochs: int,
                              projector: FeatureProjector, w_feature: float = 0.5,
                              w_response: float = 1.0,
                              include_relation: bool = False, w_relation: float = 20.0,
                              resp_weights: dict = None, schedule: str = 'linear',
                              exp_k: float = 5.0,
                              teacher_layer_selection: str = 'first', feature_layer_weights=None,
                              noise_std=0.0,
                              use_similarity: bool = False, w_similarity: float = 1.0,
                              similarity_student_idx: int = -1, similarity_teacher_idx: int = -1):
    """Annealed alpha + 可選 noise_std / similarity-preserving 傳遞。"""
    kd_loss, _, info = compute_kd_loss(student, teacher, projector, alpha=0.0,
                                        w_feature=w_feature, w_response=w_response,
                                        include_relation=include_relation,
                                        w_relation=w_relation, resp_weights=resp_weights,
                                        teacher_layer_selection=teacher_layer_selection,
                                        feature_layer_weights=feature_layer_weights,
                                        noise_std=noise_std,
                                        use_similarity=use_similarity, w_similarity=w_similarity,
                                        similarity_student_idx=similarity_student_idx,
                                        similarity_teacher_idx=similarity_teacher_idx)
    alpha = annealed_alpha(epoch, total_epochs, schedule=schedule, exp_k=exp_k)
    info["alpha"] = alpha
    return kd_loss, alpha, info


def get_trajectory_active_indices(epoch, total_epochs, num_teachers,
                                  mode='linear_grow'):
    if mode == 'all' or num_teachers <= 1:
        return list(range(num_teachers))
    import math
    frac = min(1.0, epoch / max(1, total_epochs))
    n_active = max(1, math.ceil(frac * num_teachers))
    return list(range(n_active))


def compute_trajectory_kd_batch(student, teacher, state_dicts, clean_b, noisy_b,
                                projector, trajectory_weights=None,
                                active_indices=None,
                                w_feature=0.5, w_response=1.0,
                                include_relation=False, w_relation=20.0,
                                resp_weights=None,
                                teacher_layer_selection='first',
                                feature_layer_weights=None,
                                device='cuda', noise_std=0.0):
    """軌跡蒸餾 + noise_std 支援。"""
    if trajectory_weights is None:
        trajectory_weights = [1.0] * len(state_dicts)
    if active_indices is None:
        active_indices = list(range(len(state_dicts)))
    total_kd = 0.0
    for idx in active_indices:
        teacher.load_state_dict(state_dicts[idx])
        teacher.to(device)
        teacher.eval()
        with torch.no_grad():
            teacher(clean_b, noisy_b)
        kd_loss_i, _, _ = compute_kd_loss(
            student, teacher, projector, alpha=0.0,
            w_feature=w_feature, w_response=w_response,
            include_relation=include_relation, w_relation=w_relation,
            resp_weights=resp_weights,
            teacher_layer_selection=teacher_layer_selection,
            feature_layer_weights=feature_layer_weights,
            noise_std=noise_std)
        total_kd += trajectory_weights[idx] * kd_loss_i
    return total_kd, []


def get_noise_std(epoch, total_epochs, start=0.5, end=0.0, schedule='linear'):
    """
    回傳當前 epoch 的噪聲標準差 (float)，從 start 衰減到 end。
    schedule: 'linear', 'cosine', 'exponential'（同 annealed_alpha 的曲線類型）
    """
    frac = min(1.0, epoch / max(1, total_epochs))
    if schedule == 'linear':
        shape = frac
    elif schedule == 'cosine':
        import math
        shape = 0.5 * (1 - math.cos(math.pi * frac))
    elif schedule == 'exponential':
        import math
        raw = 1 - math.exp(-5.0 * frac)   # 固定 k=5，可擴充
        norm = 1 - math.exp(-5.0)
        shape = raw / norm if norm > 1e-9 else frac
    else:
        raise ValueError(f"未知的 schedule: {schedule}")
    return start + (end - start) * shape


def get_layer_curriculum_indices(epoch, total_epochs, all_teacher_indices=[0,1,2,3]):
    """回傳當前 epoch 應對齊的 teacher 層索引（逐步加入更深層）。"""
    if len(all_teacher_indices) <= 1:
        return all_teacher_indices
    import math
    frac = min(1.0, epoch / max(1, total_epochs))
    n_active = max(1, math.ceil(frac * len(all_teacher_indices)))
    return all_teacher_indices[:n_active]
