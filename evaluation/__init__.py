"""
================================================================================
评估模块 (Evaluation Package)
================================================================================

模块化评估包，包含：
- strategies: 集成策略 (mean, voting)
- cka: CKA 相似度计算
- inference: 模型提取与推理
- metrics: MetricsCalculator 指标计算
- checkpoint: Checkpoint 加载器
- corruption_robustness: Corruption 鲁棒性评估
- domain_robustness: Domain Shift 鲁棒性评估
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

# 对抗鲁棒性评估
from .adversarial import (
    evaluate_adversarial,
    fgsm_attack,
    pgd_attack,
)

# Checkpoint 加载器
from .checkpoint import CheckpointLoader

# CKA 相似度
from .cka import compute_ensemble_cka, linear_cka

# Corruption 鲁棒性评估
from .corruption_robustness import evaluate_corruption

# Domain Shift 鲁棒性评估
from .domain_robustness import evaluate_domain_shift

# GradCAM
from .gradcam import (
    GradCAM,
    GradCAMAnalyzer,
    ModelListWrapper,
    get_target_layer,
)

# 模型推理
from .inference import get_all_models_logits, get_models_from_source

# Loss Landscape
from .landscape import LossLandscapeVisualizer

# 指标计算器
from .metrics import MetricsCalculator

# OOD 检测
from .ood import evaluate_ood

# 报告生成
from .report import ReportGenerator

# 结果保存
from .saver import ResultsSaver

# 集成策略
from .strategies import ENSEMBLE_STRATEGIES, EnsembleFn, get_ensemble_fn

# 可视化
from .visualizer import ReportVisualizer

__all__ = [
    # Strategies
    "ENSEMBLE_STRATEGIES",
    "EnsembleFn",
    "get_ensemble_fn",
    # CKA
    "linear_cka",
    "compute_ensemble_cka",
    # Inference
    "get_models_from_source",
    "get_all_models_logits",
    # Metrics
    "MetricsCalculator",
    # Checkpoint
    "CheckpointLoader",
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
    # Visualization & Reporting
    "ReportVisualizer",
    "ResultsSaver",
    "ReportGenerator",
]
