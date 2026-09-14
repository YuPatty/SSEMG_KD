# SSEMGNet（teacher）依賴 mamba_ssm / pesq，只有在這些套件存在時才載入，
# 避免 student-only（跨架構、無 GPU）環境因為缺依賴而整個 import 失敗。
try:
    from .SSEMGNet import SSEMGNet
    __all__ = ["SSEMGNet"]
except ImportError as e:
    SSEMGNet = None
    __all__ = []
    import warnings
    warnings.warn(f"SSEMGNet (teacher) 無法載入，可能缺少 mamba_ssm/pesq：{e}. "
                  f"若只需要 StudentNet，可忽略此警告。")

from .StudentNet import StudentSSEMGNet   # noqa: E402  # 不依賴 mamba_ssm/pesq
__all__.append("StudentSSEMGNet")
