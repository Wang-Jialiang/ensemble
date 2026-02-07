"""
================================================================================
训练模块 (Training Package)
================================================================================

模块化训练包，包含：
- core: StagedEnsembleTrainer (三阶段集成训练器)、train_experiment
- worker: GPUWorker、HistorySaver
- augmentation: 数据增强方法
- scheduler: 优化器工厂、调度器工厂、EarlyStopping

使用方式:
    from ensemble.training import train_experiment, StagedEnsembleTrainer
    from ensemble.training import AUGMENTATION_REGISTRY, register_augmentation
"""

# 数据增强
from .augmentation import (
    AUGMENTATION_REGISTRY,
    AugmentationMethod,
    ClassAdaptiveAugmentation,
    CutoutAugmentation,
    GridMaskAugmentation,
    NoAugmentation,
    PerlinMaskAugmentation,
    PixelHaSAugmentation,
    register_augmentation,
)

# 校准追踪器
from .calibration_tracker import CalibrationTracker

# 核心训练器
from .core import (
    StagedEnsembleTrainer,
    train_experiment,
)

# 调度器与优化器
from .optimization import (
    EarlyStopping,
    create_optimizer,
    create_scheduler,
)

# GPU Worker
from .worker import (
    GPUWorker,
    HistorySaver,
)

__all__ = [
    # Core Trainers
    "StagedEnsembleTrainer",
    "train_experiment",
    # Worker
    "GPUWorker",
    "HistorySaver",
    # Calibration
    "CalibrationTracker",
    # Augmentation
    "AUGMENTATION_REGISTRY",
    "AugmentationMethod",
    "ClassAdaptiveAugmentation",
    "CutoutAugmentation",
    "GridMaskAugmentation",
    "PixelHaSAugmentation",
    "PerlinMaskAugmentation",
    "NoAugmentation",
    "register_augmentation",
    # Scheduler
    "create_optimizer",
    "create_scheduler",
    "EarlyStopping",
]
