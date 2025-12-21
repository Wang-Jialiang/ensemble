"""
================================================================================
数据集模块 (Datasets Package)
================================================================================

模块化数据集包，包含：
- base: BasePreloadedDataset 基类
- preloaded: PreloadedCIFAR10, PreloadedEuroSAT, DATASET_REGISTRY
- corruption: CorruptionDataset, CORRUPTIONS 常量
- ood: OODDataset, OOD_REGISTRY
- domain: DomainShiftDataset, DOMAIN_REGISTRY
- loader: load_dataset
- generate: 数据生成 CLI

使用方式:
    from ensemble.datasets import load_dataset, DATASET_REGISTRY
    from ensemble.datasets import CorruptionDataset, OODDataset
"""

# 基类
from .base import BasePreloadedDataset

# Corruption 数据集
from .corruption import CORRUPTIONS, CorruptionDataset

# Domain Shift 数据集
from .domain import (
    DOMAIN_REGISTRY,
    DomainShiftDataset,
    register_domain_dataset,
)

# 数据加载器
from .loader import load_dataset

# OOD 数据集
from .ood import (
    OOD_REGISTRY,
    OODDataset,
    register_ood_dataset,
)

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
    # Corruption
    "CORRUPTIONS",
    "CorruptionDataset",
    # Preloaded
    "DATASET_REGISTRY",
    "PreloadedCIFAR10",
    "PreloadedEuroSAT",
    "register_dataset",
    # OOD
    "OOD_REGISTRY",
    "OODDataset",
    "register_ood_dataset",
    # Domain
    "DOMAIN_REGISTRY",
    "DomainShiftDataset",
    "register_domain_dataset",
    # Loader
    "load_dataset",
]
