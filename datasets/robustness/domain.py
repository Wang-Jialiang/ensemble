"""
================================================================================
Domain Shift 数据集模块
================================================================================

包含: DomainShiftDataset

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
# ║ Domain Shift 数据集                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


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
    def from_generated(
        cls,
        id_dataset: str,
        root: str = "./data",
        styles: List[str] = None,
    ) -> "DomainShiftDataset":
        """从 generate.py 生成的数据加载 Domain Shift 数据集

        Args:
            id_dataset: ID 数据集名称 (用于确定路径和标准化参数)
            root: 数据根目录
            styles: 要加载的风格列表 (默认加载所有)

        Returns:
            DomainShiftDataset 实例
        """
        from PIL import Image

        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        id_class = DATASET_REGISTRY[id_dataset]
        # matching generate.py output path
        folder_path = Path(root) / f"{id_class.NAME}-Domain"

        if not folder_path.exists():
            raise FileNotFoundError(
                f"未找到生成的 Domain 数据: {folder_path}\n"
                f"请先运行: python -m ensemble.datasets.robustness.generate --type domain --dataset {id_dataset}"
            )

        # 获取风格子目录
        available_styles = [d.name for d in folder_path.iterdir() if d.is_dir()]
        target_styles = styles or available_styles

        if not target_styles:
            raise ValueError(f"在 {folder_path} 中未找到任何风格目录")

        get_logger().info(f"📥 加载 Domain Shift 数据集: {folder_path.name}")
        get_logger().info(f"   风格: {target_styles}")

        images_list = []
        labels_list = []
        target_size = id_class.IMAGE_SIZE

        # 遍历选定的风格
        for style in target_styles:
            style_dir = folder_path / style
            if not style_dir.exists():
                get_logger().warning(f"跳过不存在的风格: {style}")
                continue

            # 遍历类别 (class_0, class_1, ...)
            class_folders = sorted([d for d in style_dir.iterdir() if d.is_dir()])

            for class_idx, class_folder in enumerate(class_folders):
                # 简单校验文件夹名是否匹配 class_{idx} 格式，或者直接信任排序
                # 这里的 class_idx 是相对于文件夹排序的，应与 ID 数据集一致

                image_files = list(class_folder.glob("*.[jJ][pP][gG]")) + list(
                    class_folder.glob("*.[pP][nN][gG]")
                )

                for img_path in image_files:
                    try:
                        img = Image.open(img_path).convert("RGB")

                        # 严格校验尺寸，不再 Resize
                        if img.size != (target_size, target_size):
                            raise ValueError(
                                f"尺寸不匹配: {img_path.name} is {img.size}, expected ({target_size}, {target_size})"
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
            f"✅ 加载了 {len(images_tensor)} 个样本, {len(target_styles)} 种风格"
        )

        return cls(
            name=f"{id_class.NAME}-Domain",
            images=images_tensor,
            labels=labels_tensor,
            mean=id_class.MEAN,
            std=id_class.STD,
        )

    def get_loader(
        self,
        config: "Config",
    ) -> DataLoader:
        """获取数据加载器"""
        images_float = self.images.float() / 255.0
        images_normalized = (images_float - self._mean) / self._std

        dataset = TensorDataset(images_normalized, self.labels)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
