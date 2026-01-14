"""
================================================================================
数据集模块 (Datasets Package)
================================================================================

模块化数据集包，包含：
- base: BasePreloadedDataset 基类
- preloaded: PreloadedCIFAR10, PreloadedEuroSAT, DATASET_REGISTRY
- loader: load_dataset
- robustness/: 鲁棒性评估数据集子包 (corruption, ood, generate)

使用方式:
    from ensemble.datasets import load_dataset, DATASET_REGISTRY
    from ensemble.datasets.robustness import CorruptionDataset, OODDataset

数据生成:
    python -m ensemble.datasets.robustness.generate --type corruption --dataset eurosat
"""

# 基类
from .base import BasePreloadedDataset

# 数据加载器
from .loader import configure_dataset_params, load_dataset

# 预加载数据集
from .preloaded import (
    DATASET_REGISTRY,
    PreloadedCIFAR10,
    PreloadedEuroSAT,
    register_dataset,
)

__all__ = [
    # Base
    "BasePreloadedDataset",
    # Preloaded
    "DATASET_REGISTRY",
    "PreloadedCIFAR10",
    "PreloadedEuroSAT",
    "register_dataset",
    # Loader
    "load_dataset",
    "configure_dataset_params",
]
