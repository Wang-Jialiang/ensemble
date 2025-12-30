"""
================================================================================
Domain Shift 数据集模块
================================================================================

包含: DomainShiftDataset

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
# ║ 常量定义 (与 generate.py 保持同步)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DOMAIN_STYLES = ["sketch", "painting", "cartoon", "watercolor"]
DOMAIN_STRENGTHS = [0.3, 0.5, 0.7]  # 轻度、中度、重度


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Domain Shift 数据集                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class DomainShiftDataset:
    """Domain Shift (域偏移) 评估数据集

    用于评估模型在不同视觉域/风格上的泛化能力。
    与 OOD 不同的是，Domain Shift 数据集有相同的类别，只是风格不同。

    目录结构: {Dataset}-Domain/{style}/{strength}/class_X/img_Y.png

    使用示例:
        >>> ds = DomainShiftDataset("cifar10", "./data")
        >>> loader = ds.get_loader("sketch", 0.5, config)  # 获取素描风格、中等强度
    """

    # 引用模块级常量
    STYLES = DOMAIN_STYLES
    STRENGTHS = DOMAIN_STRENGTHS

    def __init__(
        self,
        id_dataset: str,
        root: str = "./data",
    ):
        """Domain Shift 数据集构造函数

        Args:
            id_dataset: ID 数据集名称 (用于确定路径和标准化参数)
            root: 数据根目录
        """
        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        self.id_class = DATASET_REGISTRY[id_dataset]
        self.folder_path = Path(root) / f"{self.id_class.NAME}-Domain"

        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"未找到生成的 Domain 数据: {self.folder_path}\n"
                f"请先运行: python -m ensemble.datasets.robustness.generate --type domain --dataset {id_dataset}"
            )

        self._mean = torch.tensor(self.id_class.MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(self.id_class.STD).view(1, 3, 1, 1)

        get_logger().info(f"📥 初始化 Domain Shift 数据集: {self.folder_path.name}")

    def get_loader(
        self,
        style: str,
        strength: float,
        config: "Config",
    ) -> DataLoader:
        """获取特定风格和强度的数据加载器

        Args:
            style: 风格名称 (sketch, painting, cartoon, watercolor)
            strength: 转换强度 (0.3, 0.5, 0.7)
            config: 全局配置对象
        """
        from PIL import Image

        if style not in self.STYLES:
            raise ValueError(f"Unknown style: {style}. Available: {self.STYLES}")

        if strength not in self.STRENGTHS:
            raise ValueError(
                f"Unknown strength: {strength}. Available: {self.STRENGTHS}"
            )

        strength_dir = self.folder_path / style / str(strength)
        if not strength_dir.exists():
            raise FileNotFoundError(
                f"未找到数据目录: {strength_dir}\n请确保已生成该风格和强度的数据"
            )

        images_list = []
        labels_list = []
        target_size = self.id_class.IMAGE_SIZE

        # 遍历类别 (class_0, class_1, ...)
        class_folders = sorted([d for d in strength_dir.iterdir() if d.is_dir()])

        for class_folder in class_folders:
            # 从文件夹名称解析真正的类别索引 (如 class_0005 -> 5)
            try:
                real_class_idx = int(class_folder.name.split("_")[1])
            except (IndexError, ValueError):
                get_logger().warning(f"跳过格式不正确的目录: {class_folder.name}")
                continue

            image_files = list(class_folder.glob("*.png"))

            for img_path in image_files:
                try:
                    img = Image.open(img_path)
                    if img.size != (target_size, target_size):
                        raise ValueError(
                            f"尺寸不匹配: {img_path.name} is {img.size}, expected ({target_size}, {target_size})"
                        )

                    img_np = np.array(img)
                    images_list.append(img_np)
                    labels_list.append(real_class_idx)
                except Exception as e:
                    get_logger().warning(f"跳过无效图像 {img_path}: {e}")

        if not images_list:
            raise ValueError(f"未找到有效图像: {strength_dir}")

        images = np.stack(images_list, axis=0)
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # 归一化
        images_float = images_tensor.float() / 255.0
        images_normalized = (images_float - self._mean) / self._std

        get_logger().info(
            f"✅ 加载了 {len(images_list)} 个样本 (style={style}, strength={strength})"
        )

        dataset = TensorDataset(images_normalized, labels_tensor)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
