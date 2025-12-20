"""
================================================================================
评估模块 (Evaluation Package)
================================================================================

模块化评估包，包含：
- core: 集成策略、CKA、MetricsCalculator、模型提取、Checkpoint 加载
- robustness: Corruption + 域偏移评估
- adversarial: 对抗攻击 (FGSM/PGD) + 对抗鲁棒性评估
- ood: OOD 检测评估
- gradcam: GradCAM 热力图分析
- landscape: Loss Landscape 可视化
- visualizer: 图表可视化
- saver: 结果保存
- report: 报告生成

使用方式:
    from ensemble.evaluation import ReportGenerator, evaluate_adversarial
"""

# 核心模块 (含 Checkpoint 加载)
# 对抗鲁棒性评估
from .adversarial import (
    evaluate_adversarial,
    fgsm_attack,
    pgd_attack,
)
from .core import (
    ENSEMBLE_STRATEGIES,
    CheckpointLoader,
    EnsembleFn,
    MetricsCalculator,
    compute_ensemble_cka,
    extract_models,
    get_all_models_logits,
    get_ensemble_fn,
    linear_cka,
)

# GradCAM
from .gradcam import (
    GradCAM,
    GradCAMAnalyzer,
    ModelListWrapper,
    get_target_layer,
)

# Loss Landscape
from .landscape import LossLandscapeVisualizer

# OOD 检测
from .ood import evaluate_ood
from .report import ReportGenerator

# 鲁棒性评估 (Corruption + 域偏移)
from .robustness import (
    evaluate_corruption,
    evaluate_domain_shift,
)
from .saver import ResultsSaver

# 可视化与报告
from .visualizer import ReportVisualizer

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
    # Adversarial
    "fgsm_attack",
    "pgd_attack",
    "evaluate_adversarial",
    # OOD
    "evaluate_ood",
    # GradCAM
    "get_target_layer",
    "GradCAM",
    "GradCAMAnalyzer",
    "ModelListWrapper",
    # Landscape
    "LossLandscapeVisualizer",
    # Checkpoint
    "CheckpointLoader",
    # Visualization & Reporting
    "ReportVisualizer",
    "ResultsSaver",
    "ReportGenerator",
]
