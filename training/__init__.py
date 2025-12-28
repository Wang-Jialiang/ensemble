"""
================================================================================
训练模块 (Training Package)
================================================================================

模块化训练包，包含：
- base: BaseTrainer (训练器抽象基类)
- core: StagedEnsembleTrainer (三阶段集成训练器)、train_experiment
- worker: GPUWorker、HistorySaver
- augmentation: 数据增强方法、CloudMaskGenerator
- scheduler: 优化器工厂、调度器工厂、EarlyStopping

使用方式:
    from ensemble.training import train_experiment, StagedEnsembleTrainer
    from ensemble.training import BaseTrainer  # 抽象基类
    from ensemble.training import AUGMENTATION_REGISTRY, register_augmentation
"""

# 数据增强
from .augmentation import (
    AUGMENTATION_REGISTRY,
    AugmentationMethod,
    CloudMaskGenerator,
    CutMixAugmentation,
    CutoutAugmentation,
    DropoutAugmentation,
    MixupAugmentation,
    NoAugmentation,
    PerlinMaskAugmentation,
    register_augmentation,
)

# 训练器基类
from .base import BaseTrainer

# 核心训练器
from .core import (
    StagedEnsembleTrainer,
    train_experiment,
)

# 调度器与优化器
from .scheduler import (
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
    # Base
    "BaseTrainer",
    # Core Trainers
    "StagedEnsembleTrainer",
    "train_experiment",
    # Worker
    "GPUWorker",
    "HistorySaver",
    # Augmentation
    "AUGMENTATION_REGISTRY",
    "AugmentationMethod",
    "CloudMaskGenerator",
    "CutoutAugmentation",
    "MixupAugmentation",
    "CutMixAugmentation",
    "DropoutAugmentation",
    "PerlinMaskAugmentation",
    "NoAugmentation",
    "register_augmentation",
    # Scheduler
    "create_optimizer",
    "create_scheduler",
    "EarlyStopping",
]
