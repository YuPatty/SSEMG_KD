# ────────────────────────────────────────────────────
# distill_loss.py
# Knowledge-distillation loss for SSEMG-Net (Mamba teacher → smaller/cross-arch student)
#
# ── 最終決定（不是機械套用單一論文，而是依 SSEMG-Net 的實際情況判斷）──
#
#   1) GT/KD 權重：採 EDA Denoising KD 論文（Lee et al., 2026）的固定 0.5/0.5，
#      捨棄 PreFallKD 的 annealing。
#      理由：annealing 是分類任務的設計（怕 student 依賴 teacher 的 soft label
#      學不到自己的決策邊界）。SSEMG-Net 是回歸型訊號重建，teacher 的連續輸出
#      本身就是穩定的監督訊號，EDA 論文是同性質任務（生理訊號回歸去噪）且已用
#      下游任務（CNS-OT 預測）驗證固定權重有效，是更貼近的先例。
#
#   2) Feature loss：改用純 MSE（EDA 論文 Eq.7），捨棄 cosine similarity。
#      理由：FeatureProjector 的 1x1 conv 本身就能學會縮放，不需要用 cosine
#      similarity 迴避尺度問題，那樣反而丟失頻譜能量大小這個可能有意義的資訊。
#
#   3) Response loss：mask / magnitude / complex 三項比照 EDA 論文用 MSE，
#      但 phase 這項維持 circular distance（1-cosΔφ），不跟論文用 MSE。
#      理由：這不是風格選擇，是正確性問題——phase 是圓周量（-π 和 +π 是同一個
#      角度），論文的輸出是單一時域訊號沒有這個分量，若照抄 MSE 會在 wrap 邊界
#      算出錯誤的巨大 loss。SSEMG-Net 有 phase 分支，這裡必須用適合圓周量的距離。
#
#   4) Feature 對齊層數：對齊 min(teacher blocks, student blocks) 層，逐層 MSE
#      加總（不取平均），比照論文 Eq.7 的加總寫法。
#
#   PreFallKD 的 annealing 仍保留在檔案最後，作為「你想比較兩種策略」時的
#   替代選項，但不是預設呼叫的版本。
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
    #  1times1 的 Convolution 做 Linear Projection
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


def feature_distill_loss(student_feats, teacher_feats, projector: FeatureProjector):
    """
    Feature-based（中間層）KD loss。純 MSE，逐層加總（比照 EDA 論文 Eq.7）。
    teacher_feats 假設呼叫端已經 detach。
    """
    proj_feats = projector(student_feats)
    total = 0.0
    for sf, tf in zip(proj_feats, teacher_feats):
        total = total + F.mse_loss(sf, tf)
    return total


def relation_distill_loss(student_feats, teacher_feats):
    """
    Relation-based KD loss（RKD / DIST 風格的自相似矩陣對齊）。

    ── 為什麼需要這項 ──
    Response/feature loss 都是 pointwise 對齊：只要求「這個時頻點的值
    /特徵長得像」，沒有要求「跨時間幀的關聯結構像」。但 TF-Bi-Mamba 相對於
    depthwise separable conv 的核心優勢正是建立跨時間幀的長距離依賴——
    depthwise conv 的感受野是局部、有限的（受 kernel size/dilation 限制），
    這正是 student 選它的理由（省算力），但也代表它結構上學不到 Mamba 的
    全域依賴，除非用額外 loss 明確去逼近。這一項就是做這件事。

    ── 做法 ──
    對每一層 feature map，把每個時間步的特徵向量（跨 channel × frequency）
    做 L2 normalize，算出 [T, T] 的自相似（cosine）矩陣，代表「哪些時間幀
    彼此關聯」。比較 teacher/student 各自的這個關係矩陣，而不是比較
    絕對數值——因此完全不需要 channel projector，對 teacher/student
    channel 數不同完全不敏感，這是它跟 feature_distill_loss 互補的地方。
    """
    losses = []
    for sf, tf in zip(student_feats, teacher_feats):
        Bs, Cs, T, Fh = sf.shape
        Bt, Ct, _, _ = tf.shape
        sf_v = sf.permute(0, 2, 1, 3).reshape(Bs, T, Cs * Fh)
        tf_v = tf.permute(0, 2, 1, 3).reshape(Bt, T, Ct * Fh)
        sf_v = F.normalize(sf_v, dim=-1, eps=1e-8)
        tf_v = F.normalize(tf_v, dim=-1, eps=1e-8)
        G_s = torch.bmm(sf_v, sf_v.transpose(1, 2))     # [B, T, T]
        G_t = torch.bmm(tf_v, tf_v.transpose(1, 2))     # [B, T, T]
        losses.append(F.mse_loss(G_s, G_t))
    return sum(losses) / len(losses)


def compute_kd_loss(student, teacher, projector: FeatureProjector,
                     alpha: float = 0.5, w_feature: float = 0.3,
                     include_relation: bool = False, w_relation: float = 0.3,
                     resp_weights: dict = None):
    """
    kd_loss = response_loss + w_feature*feature_loss (+ w_relation*relation_loss)
    total   = (1-alpha)*gt_loss + alpha*kd_loss   ← alpha 這裡當「固定值」用，
              若要 annealing 由呼叫端在每個 epoch 算好 alpha 再傳進來即可
              （這個函式本身不管 alpha 是不是隨 epoch 變，只負責照給定值算）。

    w_feature / w_relation / alpha 的預設值取自 EDA 論文 Eq.4-5，
    **不是**已驗證適合 SSEMG-Net 的最佳值，建議依 §④ 的建議掃參數。

    resp_weights: dict，可指定 {"mask":.., "mag":.., "pha":.., "com":..}
                  各分項的權重，預設全部 1.0（直接相加，不做任何假設）。
                  務必搭配 info 裡回傳的各分項數值檢查量級是否失衡（§③）。

    include_relation: 是否加入 relation_distill_loss（跨時間自相似矩陣對齊，
                       見該函式說明——這項不需要 projector）。
    """
    if resp_weights is None:
        resp_weights = {"mask": 1.0, "mag": 1.0, "pha": 1.0, "com": 1.0}

    teacher_out = {k: v.detach() for k, v in teacher.last_outputs.items()}
    teacher_feats = [f.detach() for f in teacher.last_feats]

    resp_losses = response_distill_loss(student.last_outputs, teacher_out)
    resp_loss = (resp_weights["mask"] * resp_losses["mask"] +
                 resp_weights["mag"]  * resp_losses["mag"] +
                 resp_weights["pha"]  * resp_losses["pha"] +
                 resp_weights["com"]  * resp_losses["com"])

    feat_loss = feature_distill_loss(student.last_feats, teacher_feats, projector)

    kd_loss = resp_loss + w_feature * feat_loss

    rel_loss = None
    if include_relation:
        rel_loss = relation_distill_loss(student.last_feats, teacher_feats)
        kd_loss = kd_loss + w_relation * rel_loss

    info = {
        "kd_loss": float(kd_loss.detach()),
        # ★ §③ 量級診斷：訓練時把這幾個值印出來/存進 log，
        #   確認沒有某一項因為數值量級大而主導梯度方向
        "resp_mask": float(resp_losses["mask"].detach()),
        "resp_mag": float(resp_losses["mag"].detach()),
        "resp_pha": float(resp_losses["pha"].detach()),
        "resp_com": float(resp_losses["com"].detach()),
        "feat_loss": float(feat_loss.detach()) if torch.is_tensor(feat_loss) else float(feat_loss),
        "rel_loss": (float(rel_loss.detach()) if rel_loss is not None else None),
        "alpha": alpha,
    }
    return kd_loss, alpha, info


# ══════════════════════════════════════════════════════════
# GT/KD 權重策略：fixed vs annealed
# ------------------------------------------------------------
# 之前這裡寫「回歸任務不需要 annealing」，是沒有實驗支撐的斷言，已修正。
# annealing 要不要用是訓練動態的問題，跟任務是分類或回歸無關——真正該做的
# 是跑一次 fixed vs annealed 的對照實驗，用驗證集結果決定，而不是用推論
# 跳過驗證。以下兩個函式功能對等，差別只在 alpha 是否隨 epoch 變化，
# 透過 pipeline 的 --kd_weight_mode 參數切換，兩者都會把結果記錄到
# log CSV，方便你事後比較。
# ══════════════════════════════════════════════════════════
def annealed_alpha(epoch: int, total_epochs: int,
                    alpha_start: float = 1.0, alpha_end: float = 0.0) -> float:
    frac = min(1.0, epoch / max(1, total_epochs))
    return alpha_start + (alpha_end - alpha_start) * frac


def compute_kd_loss_annealed(student, teacher, epoch: int, total_epochs: int,
                              projector: FeatureProjector, w_feature: float = 0.3,
                              include_relation: bool = False, w_relation: float = 0.3,
                              resp_weights: dict = None):
    """跟 compute_kd_loss 邏輯相同，差別只在 alpha 隨 epoch 遞減。"""
    kd_loss, _, info = compute_kd_loss(student, teacher, projector, alpha=0.0,
                                        w_feature=w_feature, include_relation=include_relation,
                                        w_relation=w_relation, resp_weights=resp_weights)
    alpha = annealed_alpha(epoch, total_epochs)
    info["alpha"] = alpha
    return kd_loss, alpha, info
