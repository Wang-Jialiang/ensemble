"""
================================================================================
Corruption 数据集模块
================================================================================

包含: CorruptionDataset, CORRUPTIONS 常量
"""

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ...config.core import Config

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 全局常量定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 4种代表性Corruption类型 (简化版本，每类选一个代表)
# - Noise: gaussian_noise
# - Blur: motion_blur
# - Weather: fog
# - Digital: jpeg_compression
CORRUPTIONS = [
    "gaussian_noise",  # Noise 类代表
    "motion_blur",  # Blur 类代表
    "fog",  # Weather 类代表
    "jpeg_compression",  # Digital 类代表
]

# 3种严重程度 (轻/中/重)
SEVERITIES = [1, 3, 5]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption数据集                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CorruptionDataset:
    """Corruption 评估数据集 (仅支持预生成模式)

    从预生成的 .npy 文件加载 corruption 数据。
    使用 `python -m ensemble.datasets.robustness.generate` 预生成数据。

    使用示例:
        >>> dataset = CorruptionDataset.from_name("cifar10", "./data")
        >>> dataset = CorruptionDataset.from_name("eurosat", "./data")
    """

    # 引用模块级常量
    CORRUPTIONS = CORRUPTIONS
    SEVERITIES = SEVERITIES

    def __init__(self, name: str, data_dir: Path, mean: List[float], std: List[float]):
        """直接构造函数，推荐使用 from_name()"""
        labels_path = data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"未找到预生成数据: {labels_path}\n"
                f"请先运行: python -m ensemble.datasets.robustness.generate --type corruption --dataset <name>"
            )

        self.name = name
        self.data_dir = data_dir
        self.labels = torch.from_numpy(np.load(str(labels_path))).long()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @classmethod
    def from_name(cls, dataset_name: str, root: str = "./data") -> "CorruptionDataset":
        """从 DATASET_REGISTRY 自动派生配置"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        DatasetClass = DATASET_REGISTRY[dataset_name]
        data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        # CIFAR-10-C 特殊处理：官方下载

        return cls(
            name=f"{DatasetClass.NAME}-C",
            data_dir=data_dir,
            mean=DatasetClass.MEAN,
            std=DatasetClass.STD,
        )

    def get_loader(
        self,
        corruption_type: str,
        severity: int,
        config: "Config",
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器

        Args:
            config: 全局配置对象 (提供 batch_size, num_workers, pin_memory)
        """
        # 直接加载数据，不缓存 (避免 OOM)
        data_tensor = self._load_corruption(corruption_type, severity)

        dataset = TensorDataset(data_tensor, self.labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    def _load_corruption(self, corruption_type: str, severity: int) -> torch.Tensor:
        """从预生成文件加载

        注意: 文件中按 SEVERITIES 顺序存储 (索引 0, 1, 2 对应 severity 1, 3, 5)
        """
        if severity not in SEVERITIES:
            raise ValueError(f"Severity must be one of {SEVERITIES}, got {severity}")

        file_path = self.data_dir / f"{corruption_type}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 corruption 文件: {file_path}")

        data = np.load(str(file_path))
        n_samples = len(self.labels)

        # 根据 severity 在 SEVERITIES 中的索引计算偏移
        severity_idx = SEVERITIES.index(severity)
        images = data[severity_idx * n_samples : (severity_idx + 1) * n_samples]

        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return (images_tensor - self.mean) / self.std
