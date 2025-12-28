"""
================================================================================
OOD 数据集模块
================================================================================

包含: OODDataset

"""

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ...config.core import Config

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ...utils import get_logger
from ..preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 数据集                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class OODDataset:
    """OOD (Out-of-Distribution) 评估数据集

    用于评估模型的 OOD 检测能力，支持多种 OOD 数据集。

    使用示例:
        >>> ood_dataset = OODDataset.from_name("svhn", id_dataset="cifar10", root="./data")
        >>> loader = ood_dataset.get_loader(batch_size=128)

    添加新数据集:
        >>> register_ood_dataset("lsun", "LSUN", lambda root: ...)
    """

    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        mean: List[float],
        std: List[float],
    ):
        """直接构造函数，推荐使用 from_name()"""
        self.name = name
        self.images = images  # [N, C, H, W], uint8
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)

    @property
    def num_samples(self) -> int:
        return len(self.images)

    @classmethod
    def from_generated(
        cls,
        id_dataset: str,
        root: str = "./data",
    ) -> "OODDataset":
        """从 generate.py 生成的 OOD 数据加载

        Args:
            id_dataset: ID 数据集名称 (用于确定标准化参数和路径)
            root: 数据根目录

        Returns:
            OODDataset 实例

        Example:
            >>> ood = OODDataset.from_generated("eurosat", root="./data")
        """
        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        id_class = DATASET_REGISTRY[id_dataset]
        data_dir = Path(root) / f"{id_class.NAME}-OOD-Generated"
        images_path = data_dir / "images.npy"

        if not images_path.exists():
            raise FileNotFoundError(
                f"未找到生成的 OOD 数据: {images_path}\n"
                f"请先运行: python -m ensemble.datasets.robustness.generate --type ood --dataset {id_dataset}"
            )

        get_logger().info(f"📥 加载生成的 OOD 数据: {images_path}...")

        images = np.load(str(images_path))  # [N, H, W, C]
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # [N, C, H, W]

        get_logger().info(f"✅ 加载了 {len(images_tensor)} 个 OOD 样本")

        return cls(
            name=f"{id_class.NAME}-OOD-Generated",
            images=images_tensor,
            mean=id_class.MEAN,
            std=id_class.STD,
        )

    def get_loader(
        self,
        config: "Config",
    ) -> DataLoader:
        """获取 OOD 数据加载器"""
        # 标准化
        images_float = self.images.float() / 255.0
        images_normalized = (images_float - self._mean) / self._std

        # 使用 -1 作为 OOD 标签
        labels = torch.full((len(self.images),), -1, dtype=torch.long)

        dataset = TensorDataset(images_normalized, labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
