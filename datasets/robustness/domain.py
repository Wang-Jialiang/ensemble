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

    def __init__(self, id_dataset: str, root: str = "./data"):
        """Domain Shift 数据集构造函数"""
        if id_dataset not in DATASET_REGISTRY:
            raise ValueError(
                f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        self.id_class = DATASET_REGISTRY[id_dataset]
        self.folder_path = Path(root) / f"{self.id_class.NAME}-Domain"

        self._verify_folder()
        self._init_statistics()
        get_logger().info(f"📥 初始化 Domain Shift: {self.folder_path.name}")

    def _verify_folder(self):
        """确保数据集文件夹存在"""
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"未找到产生的 Domain 数据: {self.folder_path}\n"
                f"请运行: python -m ensemble.datasets.robustness.generate --type domain --dataset {self.id_class.NAME}"
            )

    def _init_statistics(self):
        """初始化 ID 数据集的统计参数"""
        self._mean = torch.tensor(self.id_class.MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(self.id_class.STD).view(1, 3, 1, 1)

    def get_loader(self, style: str, strength: float, config: "Config") -> DataLoader:
        """获取特定风格和强度的数据加载器 (仅支持 .npy 格式)"""
        # 1. 验证参数
        if style not in self.STYLES:
            raise ValueError(f"未知风格: {style}")
        if strength not in self.STRENGTHS:
            raise ValueError(f"未知强度: {strength}")

        # 2. 定位并加载 .npy 数据
        npy_path = self.folder_path / style / f"{strength}.npy"
        label_npy_path = self.folder_path / "labels.npy"

        if not npy_path.exists() or not label_npy_path.exists():
            raise FileNotFoundError(
                f"未找到产生的 Domain 数据文件: {npy_path} 或 {label_npy_path}\n"
                f"请运行: python -m ensemble.datasets.robustness.generate --type domain --dataset {self.id_class.NAME}"
            )

        get_logger().info(f"💾 从 .npy 文件加载: {npy_path.name}")
        images_np = np.load(str(npy_path))
        labels_np = np.load(str(label_npy_path))

        # 3. 组装并标准化
        return self._create_dataloader(images_np, labels_np, config)

    def _create_dataloader(self, images_np, labels_np, config) -> DataLoader:
        """执行张量化、归一化并创建 Loader"""
        imgs = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
        imgs_norm = (imgs - self._mean) / self._std
        lbls = torch.from_numpy(labels_np).long()

        return DataLoader(
            TensorDataset(imgs_norm, lbls),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
