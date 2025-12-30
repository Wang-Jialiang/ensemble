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
        """主加载流程 (带重试保护)"""
        # 1. 准备原始数据集
        source_ds = self._fetch_builtin_dataset()
        
        # 2. 从源数据摄取到内存
        start_time = time.time()
        self._ingest_source_data(source_ds)
        
        # 3. 统计并完成
        self._log_loaded(time.time() - start_time)

    def _fetch_builtin_dataset(self):
        """检查并下载 torchvision CIFAR10"""
        cifar_dir = Path(self.root) / "cifar-10-batches-py"
        skip_download = cifar_dir.exists()
        
        log_msg = "✅ 数据集已存在，跳过下载" if skip_download else "📥 数据集不存在，开始下载..."
        get_logger().info(log_msg)
        
        return torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=not skip_download)

    def _ingest_source_data(self, source_ds):
        """将源数据集的 image/targets 转移到 Tensor 形式"""
        get_logger().info(f"📦 Preloading {self.NAME} {'train' if self.train else 'test'} to RAM...")
        # (N, H, W, 3) -> (N, 3, H, W)
        self.images = torch.from_numpy(source_ds.data).permute(0, 3, 1, 2)
        self.targets = torch.tensor(source_ds.targets, dtype=torch.long)


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
        """主加载流程 (由于 EuroSAT 无划分，包含本地采样逻辑)"""
        # 1. 准备源数据
        source_ds = self._fetch_builtin_dataset()
        
        # 2. 解析 PIL 数据
        start_time = time.time()
        full_imgs, full_lbls = self._extract_samples(source_ds)

        # 3. 划分数据集
        self._apply_train_test_split(full_imgs, full_lbls)
        
        # 4. 统计
        self._log_loaded(time.time() - start_time)

    def _fetch_builtin_dataset(self):
        """检查并下载 torchvision EuroSAT"""
        eurosat_dir = Path(self.root) / "eurosat" / "2750"
        skip_download = eurosat_dir.exists()
        
        log_msg = "✅ EuroSAT已存在" if skip_download else "📥 开始下载 EuroSAT..."
        get_logger().info(log_msg)
        return torchvision.datasets.EuroSAT(root=self.root, download=not skip_download)

    def _extract_samples(self, source_ds):
        """解析 PIL Image 序列为 NumPy 阵列"""
        get_logger().info(f"📡 Parsing {self.NAME} samples...")
        imgs, lbls = [], []
        for img, target in source_ds:
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

        indices = shuffled_indices[:train_n] if self.train else shuffled_indices[train_n:]
        
        # 转为 Tensor 并交换通道 (H,W,C) -> (C,H,W)
        self.images = torch.from_numpy(all_images[indices]).permute(0, 3, 1, 2)
        self.targets = torch.tensor(all_targets[indices], dtype=torch.long)
