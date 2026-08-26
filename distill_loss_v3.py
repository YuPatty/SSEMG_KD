# ────────────────────────────────────────────────────
# distill_loss.py
# Knowledge-distillation loss for SSEMG-Net (Mamba teacher → smaller/cross-arch student)
#
# 新增功能：
#   - 加噪衰減蒸餾：對 teacher 中間層特徵加入隨 epoch 遞減的高斯噪聲，
#     模擬從早期欠擬合到晚期精確的軌跡，無需中間 checkpoint。
#   - 層級課程蒸餾：隨訓練進行，逐步加入更深層的 teacher 特徵。
#   - 所有原有功能（fixed/annealed alpha、response/feature/relation loss、
#     選擇性層對齊）完全保留，向後相容。
# ────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjector(nn.Module):
    """
    把 student 每一層中間輸出（channel = student_ch）投影到 teacher 的
    channel 數（teacher_ch），才能逐點做 loss 對齊。1x1 conv，對應
    AgriKD / FitNets / EDA 論文 Eq.7 的 φ_i。
    n_blocks = min(student 層數, teacher 層數)，只對齊兩者共有的層數。
    """
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
    """
    Response-based (output-level) KD loss。
    mask / mag / complex：MSE（比照 EDA 論文 Eq.6 的精神）。
    phase：1-cos(Δφ) circular distance（SSEMG-Net 特有，論文沒有這個分量）。
    teacher_out 假設呼叫端已經 detach。
    """
    l_mask = F.mse_loss(student_out["mask"], teacher_out["mask"])
    l_mag = F.mse_loss(student_out["mag_g_FT"], teacher_out["mag_g_FT"])

    dphi = student_out["pha_g"] - teacher_out["pha_g"]
    l_pha = (1.0 - torch.cos(dphi)).mean()

    l_com = F.mse_loss(student_out["com_g"], teacher_out["com_g"])

    return {"mask": l_mask, "mag": l_mag, "pha": l_pha, "com": l_com}


def feature_distill_loss(student_feats, teacher_feats, projector: FeatureProjector,
                         layer_weights=None, noise_std=0.0):
    """
    Feature-based（中間層）KD loss。純 MSE，逐層加權加總。

    新增 noise_std：對 teacher_feats 加入高斯噪聲，強度由外部控制。
    用於「退化教師蒸餾」：早期 noise_std 較大，讓學生模仿模糊表徵；
    後期 noise_std 遞減至 0，對齊精確表徵。
    """
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
    """
    Relation-based KD loss（雙流版本，參考 I2SRF-TFCKD 的 time-flow / frequency-flow
    設計，比先前版本更完整）。
    """
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
    """
    Similarity-Preserving KD（Tung & Mori, 2019；本專案首次加入）。

    跟上面的 relation_distill_loss 不一樣：這個算的是「同一個 batch 裡，
    樣本兩兩之間」的相似度結構，不是單一樣本內部的時間/頻率自相關。

    輸入可以是學生、教師『各自任一層』的 feature map，形狀甚至不用相同
    （不需要 channel 數對齊、不需要 FeatureProjector、不需要逐 block 配對）：
        student_feat: [B, C_s, T_s, F_s]（任意 C_s/T_s/F_s）
        teacher_feat: [B, C_t, T_t, F_t]（任意 C_t/T_t/F_t，可以跟學生完全不同）

    做法：
      1. 每個樣本攤平成一維向量 [B, D]（D 各自算，不用相等）。
      2. 算 batch 內樣本兩兩的內積相似度矩陣 G = normalize(X) @ normalize(X).T，
         得到 [B, B]，形狀只跟 batch size 有關，跟 D 無關。
      3. 讓學生的 G_s 去逼近教師的 G_t（MSE）。

    這正是它對 1-block 這種小架構友善的原因：不管學生只有 1 個 block 還是
    4 個，student_feat 永遠可以是「學生唯一/最後一層」的 feature，
    不會有 zip() 截斷或 projector 維度對不上的問題。
    """
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
    """
    kd_loss = w_response × response_loss + w_feature × feature_loss
              (+ w_relation × relation_loss) (+ w_similarity × similarity_loss)

    新增 noise_std：傳遞給 feature_distill_loss，對 teacher 中間層加噪。

    新增 use_similarity/w_similarity：Similarity-Preserving KD（見
    similarity_preserving_loss 的說明）。跟 feature_loss/relation_loss 不同，
    這個不需要 projector、不受 n_align_blocks（min(student blocks, teacher
    blocks)）限制，student/teacher 各自只取一層 feature（預設各自最後一層），
    對 1-block 這種小架構也完整有效。
    """
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
        # 不透過 projector、不用跟 teacher_feats（受 n_student_blocks 截斷的清單）配對，
        # 直接各自從「學生自己的 last_feats」跟「teacher 完整的 all_teacher_feats」
        # 取一層即可，索引超出範圍時自動退回最後一層，避免設錯索引直接崩潰。
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


# ── annealed_alpha 不變 ──
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


# ── Teacher Trajectory 輔助（保留原功能） ──
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


# ── 新增：噪聲標準差衰減函數 ──
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


# ── 新增：層級課程動態索引 ──
def get_layer_curriculum_indices(epoch, total_epochs, all_teacher_indices=[0,1,2,3]):
    """回傳當前 epoch 應對齊的 teacher 層索引（逐步加入更深層）。"""
    if len(all_teacher_indices) <= 1:
        return all_teacher_indices
    import math
    frac = min(1.0, epoch / max(1, total_epochs))
    n_active = max(1, math.ceil(frac * len(all_teacher_indices)))
    return all_teacher_indices[:n_active]