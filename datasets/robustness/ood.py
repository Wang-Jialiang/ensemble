"""
================================================================================
OOD 数据集模块
================================================================================

包含: OODDataset

"""

from pathlib import Path
from typing import TYPE_CHECKING

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

    def __init__(self, id_dataset: str, root: str = "./data"):
        """OOD 数据集构造函数"""
        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        id_class = DATASET_REGISTRY[id_dataset]
        self._init_from_generated(id_class, root)

    def _init_from_generated(self, id_class, root):
        """执行实际的数据加载与初始化"""
        data_dir = Path(root) / f"{id_class.NAME}-OOD"
        imgs_path = data_dir / "images.npy"

        if not imgs_path.exists():
            raise FileNotFoundError(f"未找到预生成数据: {imgs_path}")

        get_logger().info(f"📥 加载生成的 OOD 数据: {imgs_path}...")

        # 1. 加载图像并转换维度
        self.name = f"{id_class.NAME}-OOD-Generated"
        self.images = self._load_numpy_images(imgs_path)

        # 2. 初始化统计信息
        self._setup_statistics(id_class)
        get_logger().info(f"✅ 加载了 {len(self.images)} 个 OOD 样本")

    def _load_numpy_images(self, path: Path):
        """读取并转换 numpy 全量数据"""
        data = np.load(str(path))  # [N, H, W, C]
        return torch.from_numpy(data).permute(0, 3, 1, 2)  # [N, C, H, W]

    def _setup_statistics(self, id_class):
        """根据 ID 类别设置标准化参数"""
        self._mean = torch.tensor(id_class.MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(id_class.STD).view(1, 3, 1, 1)

    def __len__(self):
        """返回数据集大小"""
        return len(self.images)

    def get_loader(self, config: "Config") -> DataLoader:
        """获取 OOD 数据加载器"""
        # 1. 标准化图像
        imgs_norm = self._normalize_images()

        # 2. 组装 DataLoader (-1 作为标签)
        return self._create_ood_dataloader(imgs_norm, config)

    def _normalize_images(self):
        """执行全量图像标准化对比"""
        images_float = self.images.float() / 255.0
        return (images_float - self._mean) / self._std

    def _create_ood_dataloader(self, data, config) -> DataLoader:
        """创建最终的 OOD 数据流"""
        labels = torch.full((len(data),), -1, dtype=torch.long)
        dataset = TensorDataset(data, labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
