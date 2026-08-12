# ────────────────────────────────────────────────────
# param_count.py
# 比較 teacher (SSEMGNet, Mamba) vs student (StudentSSEMGNet, Conv) 的
# 參數量、模型大小（fp32）、FLOPs，計算壓縮比。
#
# 用法：
#   python param_count.py \
#       --teacher_config config/config_spectrogram_v19_tt_mask.yaml \
#       --student_config config/config_student_crossarch.yaml
#
# 注意：這支腳本會 import SSEMGNet（teacher），需要 mamba_ssm/CUDA 環境才能跑，
# 跟 StudentNet.py 本身不依賴 mamba_ssm 是兩回事——這裡只是要建構 teacher
# 模型物件來算參數量/FLOPs，不是要驗證 student 的可攜性。
# ────────────────────────────────────────────────────
import os, sys, argparse, math
import yaml
import torch

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.SSEMGNet import SSEMGNet          # noqa: E402  # 需要 mamba_ssm
from models.StudentNet import StudentSSEMGNet  # noqa: E402  # 不需要 mamba_ssm

# 嘗試匯入 Mamba，用於綁定 thop 的 custom_ops
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

def count_mamba_flops(m, x, y):
    """
    自訂 Mamba FLOPs 計算規則。
    解決 thop 無法追蹤 Selective Scan 自訂 CUDA Kernel 的問題。
    """
    input_tensor = x[0]
    # Mamba 輸入通常是 (Batch, Length, Dim)
    # 取 -1, -2, 0 確保不受維度順序變化影響
    B = input_tensor.shape[0]
    L = input_tensor.shape[-2]
    D = input_tensor.shape[-1]

    # 動態抓取超參數 (給予預設值以防萬一)
    d_state = getattr(m, 'd_state', 16)
    d_inner = getattr(m, 'd_inner', D * 2)
    dt_rank = getattr(m, 'dt_rank', math.ceil(D / 16))
    d_conv  = getattr(m, 'd_conv', 4)

    # 1. in_proj: D -> d_inner * 2
    flops_in_proj = B * L * D * (d_inner * 2)
    # 2. conv1d
    flops_conv = B * d_inner * L * d_conv
    # 3. x_proj
    flops_x_proj = B * L * d_inner * (dt_rank + d_state * 2)
    # 4. dt_proj
    flops_dt_proj = B * L * dt_rank * d_inner
    # 5. Selective Scan (每個時間步每個通道約 9 ops)
    flops_scan = B * L * d_inner * d_state * 9
    # 6. out_proj
    flops_out_proj = B * L * d_inner * D

    total_flops = flops_in_proj + flops_conv + flops_x_proj + flops_dt_proj + flops_scan + flops_out_proj
    
    # 將總合寫入 thop 規定的格式
    m.total_ops += torch.DoubleTensor([int(total_flops)])


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_size(n_params, bytes_per_param=4):
    """預設 fp32（4 bytes/param）；若最終要用 int8 量化部署，可傳 bytes_per_param=1 估算"""
    mb = n_params * bytes_per_param / (1024 ** 2)
    return mb


def count_flops(model, clean, noisy, model_name="", custom_ops=None):
    """
    用 thop 測 FLOPs。回傳 (macs, params_from_thop)。
    捕捉例外，避免某個模型算不出來就讓整支腳本掛掉。
    """
    try:
        from thop import profile
    except ImportError:
        print("[警告] 未安裝 thop，跳過 FLOPs 計算。可用 `pip install thop` 安裝。")
        return None, None

    model.eval()
    try:
        with torch.no_grad():
            # 引入 custom_ops 參數
            macs, params = profile(model, inputs=(clean, noisy), custom_ops=custom_ops, verbose=False)
        return macs, params
    except Exception as e:
        print(f"[警告] {model_name} 的 FLOPs 計算失敗：{e}")
        return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher_config', required=True)
    p.add_argument('--student_config', required=True)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--freq_bins', type=int, default=257)
    p.add_argument('--time_bins', type=int, default=79)
    p.add_argument('--skip_flops', action='store_true',
                    help="只算參數量/大小，跳過 FLOPs（FLOPs 計算在某些環境可能較慢或不穩定）")
    args = p.parse_args()

    with open(args.teacher_config) as f:
        teacher_cfg = yaml.safe_load(f)
    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    # ── Teacher 的 Mamba kernel 只能在 CUDA 上跑，即使只是要算參數量/FLOPs
    #    也要先把模型搬到 GPU（如果有的話），否則 thop profile 會直接崩潰。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("[警告] 沒有偵測到 GPU，Teacher 的 FLOPs 測量必然會失敗"
              "（Mamba kernel 需要 CUDA），將自動加上 --skip_flops 效果。")

    teacher = SSEMGNet(teacher_cfg).to(device)
    student = StudentSSEMGNet(student_cfg).to(device)

    t_total, t_train = count_params(teacher)
    s_total, s_train = count_params(student)

    t_mb_fp32 = fmt_size(t_total, 4)
    s_mb_fp32 = fmt_size(s_total, 4)
    t_mb_int8 = fmt_size(t_total, 1)
    s_mb_int8 = fmt_size(s_total, 1)

    print(f"{'':12}{'Params':>15}{'Size fp32(MB)':>16}{'Size int8(MB)':>16}")
    print(f"{'Teacher':12}{t_total:>15,}{t_mb_fp32:>16.2f}{t_mb_int8:>16.2f}")
    print(f"{'Student':12}{s_total:>15,}{s_mb_fp32:>16.2f}{s_mb_int8:>16.2f}")
    print()
    print(f"壓縮比（參數量）: {t_total / s_total:.2f}x")
    print(f"壓縮比（fp32 大小）: {t_mb_fp32 / s_mb_fp32:.2f}x")

    if not args.skip_flops:
        print()
        print("=" * 60)
        print("FLOPs（Teacher 包含 Mamba SSM 核心的理論估算值）")
        print("=" * 60)

        B, F_, T_ = args.batch_size, args.freq_bins, args.time_bins
        clean = torch.randn(B, 2, F_, T_, device=device)
        noisy = torch.randn(B, 2, F_, T_, device=device)

        # 綁定 custom_ops 字典給 Teacher 模型
        teacher_custom_ops = {Mamba: count_mamba_flops} if Mamba is not None else None
        
        t_macs, _ = count_flops(teacher, clean, noisy, "Teacher", custom_ops=teacher_custom_ops)
        s_macs, _ = count_flops(student, clean, noisy, "Student")

        if t_macs is not None and s_macs is not None:
            t_gflops = t_macs * 2 / 1e9   # MACs → FLOPs 慣例乘 2
            s_gflops = s_macs * 2 / 1e9
            print(f"{'':12}{'GFLOPs':>18}")
            print(f"{'Teacher':12}{t_gflops:>18.4f}  ← 已補足 Mamba 內部算力")
            print(f"{'Student':12}{s_gflops:>18.4f}  ← 純 CNN, 可信")
            print()
            print(f"運算量壓縮比: {t_gflops / s_gflops:.2f}x")
        else:
            print("[提示] 至少一個模型的 FLOPs 計算失敗，無法比較。"
                  "可加 --skip_flops 只看參數量/大小。")

    print()
    print("提醒：int8 大小是理論估算（假設每個參數從4 bytes壓到1 byte），")
    print("實際 QAT/PTQ 量化後的大小、以及量化對精度的實際影響，需要另外做量化實驗驗證。")


if __name__ == '__main__':
    main()
