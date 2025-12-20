"""
================================================================================
评估模块 (Evaluation Package)
================================================================================

模块化评估包，包含：
- core: 集成策略、CKA、MetricsCalculator、模型提取
- robustness: Corruption + 对抗攻击 + 域偏移
- ood: OOD 检测评估
- gradcam: GradCAM + Loss Landscape
- visualization: 图表可视化 + 报告生成
- checkpoint: Checkpoint 加载

使用方式:
    from ensemble.evaluation import ReportGenerator, evaluate_adversarial
"""

# 核心模块
# Checkpoint 加载
from .checkpoint import CheckpointLoader
from .core import (
    ENSEMBLE_STRATEGIES,
    EnsembleFn,
    MetricsCalculator,
    compute_ensemble_cka,
    extract_models,
    get_all_models_logits,
    get_ensemble_fn,
    linear_cka,
)

# GradCAM + Loss Landscape
from .gradcam import (
    GradCAM,
    GradCAMAnalyzer,
    LossLandscapeVisualizer,
    ModelListWrapper,
    get_target_layer,
)

# OOD 检测
from .ood import evaluate_ood

# 鲁棒性评估
from .robustness import (
    evaluate_adversarial,
    evaluate_corruption,
    evaluate_domain_shift,
    fgsm_attack,
    pgd_attack,
)

# 可视化与报告
from .visualization import ReportGenerator, ReportVisualizer, ResultsSaver

__all__ = [
    # Core
    "ENSEMBLE_STRATEGIES",
    "EnsembleFn",
    "get_ensemble_fn",
    "linear_cka",
    "compute_ensemble_cka",
    "MetricsCalculator",
    "extract_models",
    "get_all_models_logits",
    # Robustness
    "evaluate_corruption",
    "evaluate_domain_shift",
    "fgsm_attack",
    "pgd_attack",
    "evaluate_adversarial",
    # OOD
    "evaluate_ood",
    # GradCAM + Landscape
    "get_target_layer",
    "GradCAM",
    "GradCAMAnalyzer",
    "ModelListWrapper",
    "LossLandscapeVisualizer",
    # Checkpoint
    "CheckpointLoader",
    # Visualization & Reporting
    "ReportVisualizer",
    "ResultsSaver",
    "ReportGenerator",
]
