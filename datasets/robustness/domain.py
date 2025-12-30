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
            raise ValueError(f"未知 ID 数据集: {id_dataset}. 可用: {list(DATASET_REGISTRY.keys())}")

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
        """获取特定风格和强度的数据加载器"""
        # 1. 验证参数与路径
        strength_dir = self._verify_params(style, strength)

        # 2. 扫描并加载图像数据
        images_np, labels_np = self._scan_folder_for_samples(strength_dir)

        # 3. 组装并标准化
        return self._create_dataloader(images_np, labels_np, config)

    def _verify_params(self, style: str, strength: float) -> Path:
        """校验输入的风格和强度参数并定位目录"""
        if style not in self.STYLES:
            raise ValueError(f"未知风格: {style}")
        if strength not in self.STRENGTHS:
            raise ValueError(f"未知强度: {strength}")
            
        target_dir = self.folder_path / style / str(strength)
        if not target_dir.exists():
            raise FileNotFoundError(f"目录不存在: {target_dir}")
        return target_dir

    def _scan_folder_for_samples(self, target_dir: Path):
        """递归扫描文件夹并读取图像"""
        from PIL import Image
        images, labels = [], []
        target_size = self.id_class.IMAGE_SIZE

        # 遍历类别子目录
        for class_dir in sorted([d for d in target_dir.iterdir() if d.is_dir()]):
            class_idx = self._parse_class_idx(class_dir.name)
            if class_idx is None: continue

            # 读取该类别下所有图像
            for img_path in class_dir.glob("*.png"):
                img_data = self._read_single_image(img_path, target_size)
                if img_data is not None:
                    images.append(img_data)
                    labels.append(class_idx)

        if not images:
            raise ValueError(f"未在 {target_dir} 发现有效图像")
            
        return np.stack(images), np.array(labels)

    def _parse_class_idx(self, dir_name: str):
        """从文件夹名解析类别索引 (class_0005 -> 5)"""
        try:
            return int(dir_name.split("_")[1])
        except (IndexError, ValueError):
            return None

    def _read_single_image(self, path: Path, size: int):
        """读取单张图像并校验尺寸"""
        from PIL import Image
        try:
            img = Image.open(path)
            if img.size != (size, size): return None
            return np.array(img)
        except Exception:
            return None

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
