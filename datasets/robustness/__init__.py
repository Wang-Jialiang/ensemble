"""
================================================================================
Robustness 评估数据集子包
================================================================================

包含用于鲁棒性评估的数据集模块：
- corruption: Corruption 评估数据集 (CorruptionDataset, CORRUPTIONS, SEVERITIES)
- domain: Domain Shift 评估数据集 (DomainShiftDataset, DOMAIN_REGISTRY)
- ood: OOD 评估数据集 (OODDataset, OOD_REGISTRY)
- generate: 统一数据生成 CLI

使用方式:
    from ensemble.datasets.robustness import CorruptionDataset, OODDataset
    from ensemble.datasets.robustness import CORRUPTIONS, SEVERITIES

数据生成:
    python -m ensemble.datasets.robustness.generate --type corruption --dataset eurosat
"""

from .corruption import (
    CORRUPTION_CATEGORIES,
    CORRUPTIONS,
    SEVERITIES,
    CorruptionDataset,
)
from .domain import DomainShiftDataset
from .ood import OODDataset

__all__ = [
    # Corruption
    "CORRUPTION_CATEGORIES",
    "CORRUPTIONS",
    "SEVERITIES",
    "CorruptionDataset",
    # Domain
    "DomainShiftDataset",
    # OOD
    "OODDataset",
]
