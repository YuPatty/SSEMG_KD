# ────────────────────────────────────────────────────
# distill_loss.py
# Knowledge-distillation loss for SSEMG-Net (Mamba teacher → smaller/cross-arch student)
#
# ── 最終決定（不是機械套用單一論文，而是依 SSEMG-Net 的實際情況判斷）──
#
#   1) GT/KD 權重排程：Fixed vs Annealed 兩者都保留，透過消融實驗決定。
#      ⚠️ 歷史更正：這裡原本寫「捨棄 annealing，採 EDA 論文的固定 0.5/0.5」，
#      是短訓練回合（30 epoch）消融後下的初步結論。後續用完整的 60-epoch
#      生產訓練重新比較，Annealed 在 val_gt 與 val_mf 上都反超 Fixed，
#      最終生產設定改採 Annealed（alpha 隨 epoch 由 1 遞減至 0）。
#      這個逆轉本身也是方法論教訓：消融實驗的訓練長度必須跟正式生產訓練
#      一致，短訓練回合測出的結論不一定適用於長訓練回合。
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
#   Annealed 的 alpha 排程支援多種曲線形狀（linear/cosine/exponential），
#   見 annealed_alpha() 說明——曲線形狀本身也是待驗證的變數，不是選定
#   Annealed 之後就沒有更多要調的東西。
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
    Relation-based KD loss（雙流版本，參考 I2SRF-TFCKD 的 time-flow / frequency-flow
    設計，比先前版本更完整）。

    ── 為什麼要雙流，不是只做時間流 ──
    先前版本只計算「時間流」自相似矩陣（哪些時間幀彼此關聯），這只回答了
    「跨時間的關聯結構」，但語音/生理訊號的頻率軸也有獨立的物理意義——
    哪些頻段彼此耦合、噪音與訊號在頻域上如何分佈——這是時間流矩陣看不到的。
    TFCKD 論文的實驗（Table III/IV）顯示，忽略頻率流會讓 relation loss
    只捕捉到一半資訊。這裡把它補齊。

    ── 做法 ──
    對每一層 feature map [B, C, T, F]：
      - 時間流：沿 batch 拆開，在 channel×frequency 維度 normalize，
                算 [B, T, T] 的自相似矩陣（哪些時間幀相關）
      - 頻率流：沿 time 拆開，在 channel×frequency 維度 normalize，
                算 [T, B, B] 或簡化為 [B, F, F]（哪些頻段相關）
    兩者都不需要 projector，對 teacher/student channel 數不同完全不敏感。

    簡化說明：這裡沒有實作 TFCKD 的 query-key calibration weight（可學習的
    語意相關性加權）、intra-set/inter-set 多層配對，只做最基本的雙流自相似
    矩陣比對——如果基本版本有效，再考慮要不要加 calibration weight 這層。
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

        # ── 頻率流：[B, F, F]（沿 channel×time 攤平後比較頻段間關聯）──
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


def compute_kd_loss(student, teacher, projector: FeatureProjector,
                     alpha: float = 0.5, w_feature: float = 0.3,
                     include_relation: bool = False, w_relation: float = 20.0,
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
# GT/KD 權重策略：fixed vs annealed（支援多種曲線形狀）
# ------------------------------------------------------------
# 之前這裡寫「回歸任務不需要 annealing」，是沒有實驗支撐的斷言，已修正，
# 且後續驗證顯示 Annealed（linear）在長訓練回合下優於 Fixed，已採用為
# 最終生產設定（見檔案開頭的歷史更正說明）。
#
# 曲線形狀本身也是可以調整的變數：
#   linear      : alpha 等速遞減，前後期依賴 teacher 的程度變化速率一致
#   cosine      : 頭尾變化慢、中段變化快——前期讓 student 穩定依賴 teacher
#                 較長時間，後期也有較長時間穩定收斂在 GT 主導，只在訓練
#                 中段快速切換
#   exponential : 前期快速下降，之後在低 alpha 值長時間停留——適合想讓
#                 student 儘早大量依賴自己的 GT loss，只在最早期短暫依賴 teacher
#
# 目前 linear 是唯一實測驗證過的曲線（60-epoch 生產訓練用的就是這個），
# cosine/exponential 是新加入、還未經消融驗證的選項——導入前應比照既有 SOP：
# dry-run → 至少一組完整長度（60 epoch）的消融比較 → 用結果決定要不要採用，
# 不要假設某條曲線「聽起來更合理」就直接換成生產設定。
# ══════════════════════════════════════════════════════════
def annealed_alpha(epoch: int, total_epochs: int,
                    alpha_start: float = 1.0, alpha_end: float = 0.0,
                    schedule: str = 'linear', exp_k: float = 5.0) -> float:
    """
    schedule: 'linear'（預設，向後相容）、'cosine'、'exponential'
    exp_k   : 只有 schedule='exponential' 時用到，控制下降速度
              （k 越大，前期下降越快，越早進入低 alpha 平原期）
    """
    frac = min(1.0, epoch / max(1, total_epochs))

    if schedule == 'linear':
        shape = frac
    elif schedule == 'cosine':
        # 0 -> 1 的 cosine 曲線：frac=0時 shape=0，frac=1時 shape=1，
        # 中段（frac=0.5附近）變化速率最快
        import math
        shape = 0.5 * (1 - math.cos(math.pi * frac))
    elif schedule == 'exponential':
        import math
        # 正規化到 [0,1]，讓 frac=1 時 shape 恰好等於 1（避免尾端沒完全降到底）
        raw = 1 - math.exp(-exp_k * frac)
        norm = 1 - math.exp(-exp_k)
        shape = raw / norm if norm > 1e-9 else frac
    else:
        raise ValueError(f"未知的 schedule 類型: {schedule}，可選 'linear'/'cosine'/'exponential'")

    return alpha_start + (alpha_end - alpha_start) * shape


def compute_kd_loss_annealed(student, teacher, epoch: int, total_epochs: int,
                              projector: FeatureProjector, w_feature: float = 0.3,
                              include_relation: bool = False, w_relation: float = 20.0,
                              resp_weights: dict = None, schedule: str = 'linear',
                              exp_k: float = 5.0):
    """跟 compute_kd_loss 邏輯相同，差別只在 alpha 隨 epoch 依指定曲線變化。"""
    kd_loss, _, info = compute_kd_loss(student, teacher, projector, alpha=0.0,
                                        w_feature=w_feature, include_relation=include_relation,
                                        w_relation=w_relation, resp_weights=resp_weights)
    alpha = annealed_alpha(epoch, total_epochs, schedule=schedule, exp_k=exp_k)
    info["alpha"] = alpha
    return kd_loss, alpha, info
