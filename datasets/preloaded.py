"""
================================================================================
预加载数据集模块
================================================================================

包含: PreloadedCIFAR10, PreloadedEuroSAT, DATASET_REGISTRY
"""

from pathlib import Path
from typing import Dict, Type

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from ..utils import get_logger
from .base import BasePreloadedDataset

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据集注册表                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DATASET_REGISTRY: Dict[str, Type[BasePreloadedDataset]] = {}


def register_dataset(name: str):
    """装饰器：注册数据集到全局注册表

    使用方式:
        @register_dataset("my_dataset")
        class MyDataset(BasePreloadedDataset):
            ...
    """

    def decorator(cls: Type[BasePreloadedDataset]) -> Type[BasePreloadedDataset]:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 具体数据集实现                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@register_dataset("cifar10")
class PreloadedCIFAR10(BasePreloadedDataset):
    """内存预加载的CIFAR-10数据集"""

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2023, 0.1994, 0.2010]
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    NAME = "CIFAR-10"

    def _init_transforms(self):
        """CIFAR-10 数据增强: 保守策略"""
        if self.train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(self.IMAGE_SIZE, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                ]
            )
        else:
            self.transform = None

    def _load_data(self):
        """主加载流程"""
        source_ds = self._fetch_builtin_dataset()
        self._ingest_source_data(source_ds)
        self._log_loaded()

    def _fetch_builtin_dataset(self):
        """加载 torchvision CIFAR10 (假设已下载)"""
        return torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=False
        )

    def _ingest_source_data(self, source_ds):
        """将源数据集的 image/targets 转移到 Tensor 形式"""
        # (N, H, W, 3) -> (N, 3, H, W)
        self.images = torch.from_numpy(source_ds.data).permute(0, 3, 1, 2)
        self.targets = torch.tensor(source_ds.targets, dtype=torch.long)


@register_dataset("eurosat")
class PreloadedEuroSAT(BasePreloadedDataset):
    """内存预加载的EuroSAT遥感数据集"""

    MEAN = [0.3444, 0.3803, 0.4078]  # EuroSAT specific statistics
    STD = [0.2037, 0.1366, 0.1148]
    IMAGE_SIZE = 64
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    NAME = "EuroSAT"
    HAS_OFFICIAL_SPLIT = False  # 没有官方划分，需要手动划分

    def __init__(
        self,
        root: str,
        train: bool,
        test_split: float = 0.2,
        seed: int = 42,
    ):
        """
        初始化EuroSAT数据集

        参数:
            root: 数据集根目录
            train: 是否为训练集
            test_split: 训练/测试划分比例 (EuroSAT没有官方划分)
            seed: 随机种子
        """
        self.test_split = test_split
        self.seed = seed
        super().__init__(root, train)

    def _init_transforms(self):
        """EuroSAT 数据增强: 遥感图像适用更激进的策略"""
        if self.train:
            self.transform = transforms.Compose(
                [
                    # transforms.RandomCrop(self.IMAGE_SIZE, padding=4),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),  # 遥感图像可垂直翻转
                    # transforms.RandomRotation(degrees=90),  # 遥感图像可更大角度旋转
                ]
            )
        else:
            self.transform = None

    def _load_data(self):
        """主加载流程 (支持缓存加速)"""
        cache_path = Path(self.root) / f"eurosat_cache_seed{self.seed}.npz"

        if cache_path.exists():
            # 快速加载缓存
            get_logger().info(f"⚡ 从缓存加载 {self.NAME}: {cache_path}")
            data = np.load(cache_path)
            full_imgs, full_lbls = data["images"], data["targets"]
        else:
            # 首次加载并创建缓存
            get_logger().info(f"📡 首次加载 {self.NAME}，将创建缓存...")
            source_ds = self._fetch_builtin_dataset()
            full_imgs, full_lbls = self._extract_samples(source_ds)
            np.savez(cache_path, images=full_imgs, targets=full_lbls)
            get_logger().info(f"💾 缓存已保存: {cache_path}")

        self._apply_train_test_split(full_imgs, full_lbls)
        self._log_loaded()

    def _fetch_builtin_dataset(self):
        """加载 torchvision EuroSAT (假设已下载)"""
        return torchvision.datasets.EuroSAT(root=self.root, download=False)

    def _extract_samples(self, source_ds):
        """解析 PIL Image 序列为 NumPy 阵列 (带进度条)"""
        get_logger().info(f"📡 Parsing {self.NAME} samples...")
        imgs, lbls = [], []
        for img, target in tqdm(source_ds, desc=f"Loading {self.NAME}", unit="img"):
            imgs.append(np.array(img))
            lbls.append(target)
        return np.stack(imgs, axis=0), np.array(lbls)

    def _apply_train_test_split(self, all_images, all_targets):
        """对全量数据进行确定性随机划分"""
        total = len(all_images)
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(total)

        test_n = int(total * self.test_split)
        train_n = total - test_n

        indices = (
            shuffled_indices[:train_n] if self.train else shuffled_indices[train_n:]
        )

        # 转为 Tensor 并交换通道 (H,W,C) -> (C,H,W)
        self.images = torch.from_numpy(all_images[indices]).permute(0, 3, 1, 2)
        self.targets = torch.tensor(all_targets[indices], dtype=torch.long)
