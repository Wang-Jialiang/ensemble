"""
================================================================================
OOD 数据集模块
================================================================================

包含: OODDataset, OOD_REGISTRY, register_ood_dataset
"""

from typing import Dict, List

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

from ..utils import DEFAULT_DATA_ROOT, get_logger
from .preloaded import DATASET_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 数据集                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# OOD 数据集注册表 (可动态扩展)
OOD_REGISTRY: Dict[str, dict] = {}


def register_ood_dataset(
    name: str, display_name: str, loader_fn, compatible_with: List[str] = None
):
    """注册 OOD 数据集

    Args:
        name: 数据集标识符 (用于 from_name)
        display_name: 显示名称
        loader_fn: 加载函数，接收 root 参数，返回 torchvision 兼容的数据集
        compatible_with: 兼容的 ID 数据集列表（可选，仅用于文档）

    Example:
        >>> register_ood_dataset(
        ...     "svhn",
        ...     "SVHN",
        ...     lambda root: torchvision.datasets.SVHN(root=root, split="test", download=True),
        ...     compatible_with=["cifar10"]
        ... )
    """
    OOD_REGISTRY[name] = {
        "name": display_name,
        "loader": loader_fn,
        "compatible_with": compatible_with or [],
    }


# 预注册常用 OOD 数据集
register_ood_dataset(
    "svhn",
    "SVHN",
    lambda root: torchvision.datasets.SVHN(root=root, split="test", download=True),
    compatible_with=["cifar10"],
)

register_ood_dataset(
    "textures",
    "Textures (DTD)",
    lambda root: torchvision.datasets.DTD(root=root, split="test", download=True),
    compatible_with=["cifar10", "eurosat"],
)


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
    def from_name(
        cls,
        ood_name: str,
        id_dataset: str,
        root: str = DEFAULT_DATA_ROOT,
    ) -> "OODDataset":
        """根据名称加载 OOD 数据集

        Args:
            ood_name: OOD 数据集名称 (svhn, textures 等)
            id_dataset: ID 数据集名称 (cifar10, eurosat)，用于确定标准化参数
            root: 数据根目录

        Returns:
            OODDataset 实例
        """
        if ood_name not in OOD_REGISTRY:
            raise ValueError(
                f"未知 OOD 数据集: {ood_name}. 可用: {list(OOD_REGISTRY.keys())}"
            )

        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        ood_config = OOD_REGISTRY[ood_name]
        id_class = DATASET_REGISTRY[id_dataset]

        get_logger().info(f"📥 加载 OOD 数据集: {ood_config['name']}...")

        # 加载 OOD 数据集
        try:
            ood_dataset = ood_config["loader"](root)
        except Exception as e:
            get_logger().error(f"❌ OOD 数据集加载失败: {e}")
            raise

        # 转换为张量
        images_list = []
        target_size = id_class.IMAGE_SIZE

        for i in range(len(ood_dataset)):
            img, _ = ood_dataset[i]

            img_np = np.array(img)

            # 确保是 RGB
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 4:
                img_np = img_np[:, :, :3]

            # Resize 到 ID 数据集的尺寸
            if img_np.shape[0] != target_size or img_np.shape[1] != target_size:
                from PIL import Image

                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize(
                    (target_size, target_size), Image.Resampling.BILINEAR
                )
                img_np = np.array(img_pil)

            images_list.append(img_np)

        images = np.stack(images_list, axis=0)  # [N, H, W, C]
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # [N, C, H, W]

        get_logger().info(
            f"✅ 加载了 {len(images_tensor)} 个 OOD 样本 (尺寸: {target_size}x{target_size})"
        )

        return cls(
            name=ood_config["name"],
            images=images_tensor,
            mean=id_class.MEAN,
            std=id_class.STD,
        )

    def get_loader(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
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
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
