# ────────────────────────────────────────────────────
# measure_memory.py
# 用 torchinfo 分析 StudentSSEMGNet（我們真正訓練出來的模型）的
# 逐層參數量、記憶體佔用、運算量，而不是簡化版的假模型。
#
# 用法：
#   python measure_memory.py
#   python measure_memory.py --student_weights model_weight/student_production_annealed_60.pth
# ────────────────────────────────────────────────────
import os, sys, argparse
import yaml
import torch
from torchinfo import summary

ROOT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT_DIR, 'MECG-E'))
from models.StudentNet import StudentSSEMGNet   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_config", default="config/config_student_crossarch.yaml")
    p.add_argument("--student_weights", default=None,
                    help="指定的話會用 strict=True 載入並驗證架構完全對得上，"
                         "不指定則用隨機初始化權重（不影響參數量/FLOPs 分析結果）")
    args = p.parse_args()

    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    model = StudentSSEMGNet(student_cfg)

    if args.student_weights:
        state = torch.load(args.student_weights, map_location='cpu')
        model.load_state_dict(state, strict=True)
        print(f"✅ 已載入並驗證 checkpoint：{args.student_weights}")

    B = 1
    C = student_cfg.get('input_channels', 2)
    Fbin = student_cfg.get('input_freq_bins', student_cfg['model']['n_fft'] // 2 + 1)
    T = student_cfg.get('input_time_bins', 79)

    print("\n" + "=" * 70)
    print("🧠 StudentSSEMGNet 模型資源佔用分析報告")
    print("=" * 70)

    # StudentSSEMGNet.forward 需要兩個輸入（clean, noisy），
    # torchinfo 的 input_data 用 list 傳多個輸入 tensor
    clean = torch.randn(B, C, Fbin, T)
    noisy = torch.randn(B, C, Fbin, T)

    summary(
        model,
        input_data=[clean, noisy],
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
    )


if __name__ == "__main__":
    main()