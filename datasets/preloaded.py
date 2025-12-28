"""
================================================================================
预加载数据集模块
================================================================================

包含: PreloadedCIFAR10, PreloadedEuroSAT, DATASET_REGISTRY
"""

import time
from pathlib import Path
from typing import Dict, Type

import numpy as np
import torch
import torchvision
from tenacity import retry, stop_after_attempt, wait_fixed

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
    NAME = "CIFAR-10"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
    def _load_data(self):
        """加载数据 (带重试)"""
        try:
            # 检查数据集是否已存在，避免重复下载
            cifar_dir = Path(self.root) / "cifar-10-batches-py"
            should_download = not cifar_dir.exists()
            if should_download:
                get_logger().info("📥 CIFAR-10数据集不存在，开始下载...")
            else:
                get_logger().info("✅ CIFAR-10数据集已存在，跳过下载")

            base_dataset = torchvision.datasets.CIFAR10(
                root=self.root, train=self.train, download=should_download
            )
        except Exception as e:
            get_logger().error(f"❌ CIFAR-10加载失败: {e}")
            raise

        get_logger().info(
            f"📦 Preloading {'train' if self.train else 'test'} data to RAM..."
        )
        start = time.time()

        self.images = torch.from_numpy(base_dataset.data)
        self.images = self.images.permute(0, 3, 1, 2)
        self.targets = torch.tensor(base_dataset.targets, dtype=torch.long)

        self._log_loaded(time.time() - start)


@register_dataset("eurosat")
class PreloadedEuroSAT(BasePreloadedDataset):
    """内存预加载的EuroSAT遥感数据集"""

    MEAN = [0.485, 0.456, 0.406]  # ImageNet标准化
    STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 64
    NUM_CLASSES = 10
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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=True)
    def _load_data(self):
        """加载数据 (带重试)"""
        try:
            # 检查数据集是否已存在，避免重复下载
            eurosat_dir = Path(self.root) / "eurosat" / "2750"
            should_download = not eurosat_dir.exists()
            if should_download:
                get_logger().info("📥 EuroSAT数据集不存在，开始下载...")
            else:
                get_logger().info("✅ EuroSAT数据集已存在，跳过下载")

            full_dataset = torchvision.datasets.EuroSAT(
                root=self.root, download=should_download
            )
        except Exception as e:
            get_logger().error(f"❌ EuroSAT加载失败: {e}")
            raise

        get_logger().info(
            f"📡 Preloading {'train' if self.train else 'test'} data to RAM..."
        )
        start = time.time()

        # 获取所有数据
        all_images = []
        all_targets = []
        for img, target in full_dataset:
            # EuroSAT图像是PIL Image，转换为numpy再转tensor
            img_np = np.array(img)
            all_images.append(img_np)
            all_targets.append(target)

        all_images = np.stack(all_images, axis=0)  # (N, 64, 64, 3)
        all_targets = np.array(all_targets)

        # 划分训练/测试集: 使用隔离的 RNG 保证可重复性且不影响全局状态
        total_samples = len(all_images)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(total_samples)

        test_size = int(total_samples * self.test_split)
        train_size = total_samples - test_size

        if self.train:
            selected_indices = indices[:train_size]
        else:
            selected_indices = indices[train_size:]

        # 转换为tensor
        self.images = torch.from_numpy(all_images[selected_indices])
        self.images = self.images.permute(0, 3, 1, 2)  # (N, 3, 64, 64)
        self.targets = torch.tensor(all_targets[selected_indices], dtype=torch.long)

        self._log_loaded(time.time() - start)
