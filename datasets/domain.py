"""
================================================================================
Domain Shift 数据集模块
================================================================================

包含: DomainShiftDataset, DOMAIN_REGISTRY, register_domain_dataset
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_logger
from .preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Domain Shift 数据集                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Domain Shift 数据集注册表 (可动态扩展)
DOMAIN_REGISTRY: Dict[str, dict] = {}


def register_domain_dataset(
    name: str,
    display_name: str,
    folder_path: str,
    compatible_with: List[str] = None,
):
    """注册 Domain Shift 数据集

    Args:
        name: 数据集标识符 (用于 from_name)
        display_name: 显示名称
        folder_path: 数据集文件夹路径
        compatible_with: 兼容的 ID 数据集列表

    Example:
        >>> register_domain_dataset(
        ...     "cifar10_sketch",
        ...     "CIFAR-10 Sketch",
        ...     "./data/cifar10_sketch",
        ...     compatible_with=["cifar10"]
        ... )
    """
    DOMAIN_REGISTRY[name] = {
        "name": display_name,
        "folder_path": folder_path,
        "compatible_with": compatible_with or [],
    }


class DomainShiftDataset:
    """Domain Shift (域偏移) 评估数据集

    用于评估模型在不同视觉域/风格上的泛化能力。
    与 OOD 不同的是，Domain Shift 数据集有相同的类别，只是风格不同。

    使用示例:
        # 从注册表加载
        >>> ds = DomainShiftDataset.from_name("cifar10_sketch", id_dataset="cifar10")

        # 从文件夹加载
        >>> ds = DomainShiftDataset.from_folder("./data/sketches", id_dataset="cifar10")

    添加新数据集:
        >>> register_domain_dataset("my_domain", "My Domain", "./data/my_domain")
    """

    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        labels: torch.Tensor,
        mean: List[float],
        std: List[float],
    ):
        """直接构造函数"""
        self.name = name
        self.images = images  # [N, C, H, W], uint8
        self.labels = labels  # [N], long
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)

    @property
    def num_samples(self) -> int:
        return len(self.images)

    @classmethod
    def from_name(
        cls,
        domain_name: str,
        id_dataset: str,
    ) -> "DomainShiftDataset":
        """从注册表加载 Domain Shift 数据集

        Args:
            domain_name: 已注册的域偏移数据集名称
            id_dataset: ID 数据集名称（用于确定标准化参数和图像尺寸）

        Returns:
            DomainShiftDataset 实例
        """
        if domain_name not in DOMAIN_REGISTRY:
            raise ValueError(
                f"未知 Domain 数据集: {domain_name}. 可用: {list(DOMAIN_REGISTRY.keys())}"
            )

        config = DOMAIN_REGISTRY[domain_name]
        return cls.from_folder(config["folder_path"], id_dataset)

    @classmethod
    def from_folder(
        cls,
        folder_path: str,
        id_dataset: str,
        class_names: List[str] = None,
    ) -> "DomainShiftDataset":
        """从文件夹加载 Domain Shift 数据集

        文件夹结构应为:
        folder_path/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img1.jpg
            ...

        Args:
            folder_path: 数据集文件夹路径
            id_dataset: ID 数据集名称（用于确定标准化参数和图像尺寸）
            class_names: 类别名称列表（可选，默认使用文件夹名）

        Returns:
            DomainShiftDataset 实例
        """
        from PIL import Image

        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        id_class = DATASET_REGISTRY[id_dataset]
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"未找到数据集文件夹: {folder_path}")

        # 获取类别
        class_folders = sorted([d for d in folder.iterdir() if d.is_dir()])
        if not class_folders:
            raise ValueError(f"文件夹中未找到子目录: {folder_path}")

        get_logger().info(f"📥 加载 Domain Shift 数据集: {folder.name}...")

        images_list = []
        labels_list = []
        target_size = id_class.IMAGE_SIZE

        for class_idx, class_folder in enumerate(class_folders):
            image_files = list(class_folder.glob("*.[jJ][pP][gG]")) + list(
                class_folder.glob("*.[pP][nN][gG]")
            )

            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(
                        (target_size, target_size), Image.Resampling.BILINEAR
                    )
                    img_np = np.array(img)
                    images_list.append(img_np)
                    labels_list.append(class_idx)
                except Exception as e:
                    get_logger().warning(f"跳过无效图像 {img_path}: {e}")

        if not images_list:
            raise ValueError(f"未找到有效图像: {folder_path}")

        images = np.stack(images_list, axis=0)
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        get_logger().info(
            f"✅ 加载了 {len(images_tensor)} 个样本, {len(class_folders)} 个类别"
        )

        return cls(
            name=folder.name,
            images=images_tensor,
            labels=labels_tensor,
            mean=id_class.MEAN,
            std=id_class.STD,
        )

    def get_loader(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> DataLoader:
        """获取数据加载器"""
        images_float = self.images.float() / 255.0
        images_normalized = (images_float - self._mean) / self._std

        dataset = TensorDataset(images_normalized, self.labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
