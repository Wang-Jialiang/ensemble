"""
================================================================================
Corruption 数据集模块
================================================================================

包含: CorruptionDataset
"""

import tarfile
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import DEFAULT_DATA_ROOT, ensure_dir, get_logger
from .base import CORRUPTIONS
from .preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption数据集                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CorruptionDataset:
    """Corruption 评估数据集 (仅支持预生成模式)

    从预生成的 .npy 文件加载 corruption 数据。
    使用 `python -m ensemble.datasets.generate` 预生成数据。

    使用示例:
        >>> dataset = CorruptionDataset.from_name("cifar10", "./data")
        >>> dataset = CorruptionDataset.from_name("eurosat", "./data")
    """

    # 引用模块级常量
    CORRUPTIONS = CORRUPTIONS

    def __init__(self, name: str, data_dir: Path, mean: List[float], std: List[float]):
        """直接构造函数，推荐使用 from_name()"""
        labels_path = data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"未找到预生成数据: {labels_path}\n"
                f"请先运行: python -m ensemble.datasets.generate --dataset <name>"
            )

        self.name = name
        self.data_dir = data_dir
        self.labels = torch.from_numpy(np.load(str(labels_path))).long()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self._cache = {}

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @classmethod
    def from_name(
        cls, dataset_name: str, root: str = DEFAULT_DATA_ROOT
    ) -> "CorruptionDataset":
        """从 DATASET_REGISTRY 自动派生配置"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        DatasetClass = DATASET_REGISTRY[dataset_name]
        data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        # CIFAR-10-C 特殊处理：官方下载
        if dataset_name == "cifar10" and not data_dir.exists():
            get_logger().info("📥 CIFAR-10-C 不存在，开始下载...")
            cls._download_cifar10c(root)

        return cls(
            name=f"{DatasetClass.NAME}-C",
            data_dir=data_dir,
            mean=DatasetClass.MEAN,
            std=DatasetClass.STD,
        )

    def get_loader(
        self,
        corruption_type: str,
        severity: int = 5,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器"""
        cache_key = (corruption_type, severity)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_corruption(corruption_type, severity)

        dataset = TensorDataset(self._cache[cache_key], self.labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _load_corruption(self, corruption_type: str, severity: int) -> torch.Tensor:
        """从预生成文件加载"""
        file_path = self.data_dir / f"{corruption_type}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 corruption 文件: {file_path}")

        data = np.load(str(file_path))
        n_samples = len(self.labels)
        images = data[(severity - 1) * n_samples : severity * n_samples]

        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return (images_tensor - self.mean) / self.std

    @staticmethod
    def _download_cifar10c(root: str):
        """下载 CIFAR-10-C 数据集"""
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        tar_path = Path(root) / "CIFAR-10-C.tar"
        ensure_dir(root)

        get_logger().info(f"📥 Downloading CIFAR-10-C from {url}...")
        urllib.request.urlretrieve(url, str(tar_path))

        get_logger().info(f"📦 Extracting to {root}...")
        with tarfile.open(str(tar_path), "r") as tar:
            tar.extractall(str(root))

        tar_path.unlink()
        get_logger().info("✅ CIFAR-10-C download complete!")
