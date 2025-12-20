"""
================================================================================
数据集基类模块
================================================================================

包含: BasePreloadedDataset 基类、CORRUPTIONS 常量
"""

import os
import ssl

import torch
from torch.utils.data import Dataset

from ..utils import get_logger

# SSL 证书验证修复 (可通过环境变量 DISABLE_SSL_VERIFY=1 启用)
if os.environ.get("DISABLE_SSL_VERIFY", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 全局常量定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# 15种标准Corruption类型 (与ImageNet-C一致)
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据集基类                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class BasePreloadedDataset(Dataset):
    """内存预加载数据集的基类

    子类需要实现:
        - _load_data(): 加载数据到 self.images 和 self.targets
        - _get_dataset_name(): 返回数据集名称 (用于日志)

    子类应当覆盖以下类属性:
        - MEAN: 标准化均值
        - STD: 标准化标准差
        - IMAGE_SIZE: 图像尺寸
        - NUM_CLASSES: 类别数量
        - NAME: 数据集显示名称
    """

    # 默认元数据 (子类需覆盖)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224
    NUM_CLASSES = 1000
    NAME = "Base"

    def __init__(self, root: str, train: bool):
        """
        初始化数据集

        参数:
            root: 数据集根目录
            train: 是否为训练集
        """
        self.root = root
        self.train = train
        self.images: torch.Tensor = None
        self.targets: torch.Tensor = None

        # 预计算标准化参数
        self._mean = torch.tensor(self.MEAN).view(3, 1, 1)
        self._std = torch.tensor(self.STD).view(3, 1, 1)

        # 下载并加载数据
        self._load_data()

    def _load_data(self):
        """加载数据到 self.images 和 self.targets (子类实现)"""
        raise NotImplementedError("子类必须实现 _load_data 方法")

    def _get_dataset_name(self) -> str:
        """返回数据集名称"""
        return self.NAME

    def _log_loaded(self, elapsed: float):
        """打印加载完成日志"""
        mem_mb = self.images.numel() * self.images.element_size() / 1024 / 1024
        dataset_name = self._get_dataset_name()
        get_logger().info(
            f"✅ Loaded {len(self)} {dataset_name} samples ({mem_mb:.1f} MB) in {elapsed:.2f}s"
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """获取已标准化的图像和标签"""
        img = self.images[idx].float() / 255.0  # uint8 -> float [0-1]
        img = (img - self._mean) / self._std  # 标准化
        return img, self.targets[idx]
