"""
================================================================================
数据集模块
================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ SSL证书验证修复 (可通过环境变量 DISABLE_SSL_VERIFY=1 启用)
# ╚══════════════════════════════════════════════════════════════════════════════╝
import os
import ssl
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from tenacity import retry, stop_after_attempt, wait_fixed
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .utils import DEFAULT_DATA_ROOT, ensure_dir, get_logger

if os.environ.get("DISABLE_SSL_VERIFY", "0") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 全局常量定义
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
# ║ 数据集基类
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 具体数据集实现                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


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


DATASET_REGISTRY = {
    "cifar10": PreloadedCIFAR10,
    "eurosat": PreloadedEuroSAT,
}


def register_dataset(name: str, dataset_class: type):
    """动态注册新数据集"""
    DATASET_REGISTRY[name] = dataset_class


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption数据集
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CorruptionDataset:
    """Corruption 评估数据集 (仅支持预生成模式)

    从预生成的 .npy 文件加载 corruption 数据。
    使用 `python -m my.generate_corruption` 预生成数据。

    使用示例:
        >>> dataset = CorruptionDataset.from_name("cifar10", "./data")
        >>> dataset = CorruptionDataset.from_name("eurosat", "./data")
    """

    # 引用模块级常量
    CORRUPTIONS = CORRUPTIONS

    def __init__(self, name: str, data_dir: Path, mean: List[float], std: List[float]):
        """直接构造函数，推荐使用 from_name()"""
        labels_path = data_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"未找到预生成数据: {labels_path}\n"
                f"请先运行: python -m my.generate_corruption --dataset <name>"
            )

        self.name = name
        self.data_dir = data_dir
        self.labels = torch.from_numpy(np.load(str(labels_path))).long()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self._cache = {}

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @classmethod
    def from_name(
        cls, dataset_name: str, root: str = DEFAULT_DATA_ROOT
    ) -> "CorruptionDataset":
        """从 DATASET_REGISTRY 自动派生配置"""
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
            )

        DatasetClass = DATASET_REGISTRY[dataset_name]
        data_dir = Path(root) / f"{DatasetClass.NAME}-C"

        # CIFAR-10-C 特殊处理：官方下载
        if dataset_name == "cifar10" and not data_dir.exists():
            get_logger().info("📥 CIFAR-10-C 不存在，开始下载...")
            cls._download_cifar10c(root)

        return cls(
            name=f"{DatasetClass.NAME}-C",
            data_dir=data_dir,
            mean=DatasetClass.MEAN,
            std=DatasetClass.STD,
        )

    def get_loader(
        self,
        corruption_type: str,
        severity: int = 5,
        batch_size: int = 128,
        num_workers: int = 4,
    ) -> DataLoader:
        """获取特定损坏类型和严重程度的数据加载器"""
        cache_key = (corruption_type, severity)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._load_corruption(corruption_type, severity)

        dataset = TensorDataset(self._cache[cache_key], self.labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _load_corruption(self, corruption_type: str, severity: int) -> torch.Tensor:
        """从预生成文件加载"""
        file_path = self.data_dir / f"{corruption_type}.npy"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到 corruption 文件: {file_path}")

        data = np.load(str(file_path))
        n_samples = len(self.labels)
        images = data[(severity - 1) * n_samples : severity * n_samples]

        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return (images_tensor - self.mean) / self.std

    @staticmethod
    def _download_cifar10c(root: str):
        """下载 CIFAR-10-C 数据集"""
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        tar_path = Path(root) / "CIFAR-10-C.tar"
        ensure_dir(root)

        get_logger().info(f"📥 Downloading CIFAR-10-C from {url}...")
        urllib.request.urlretrieve(url, str(tar_path))

        get_logger().info(f"📦 Extracting to {root}...")
        with tarfile.open(str(tar_path), "r") as tar:
            tar.extractall(str(root))

        tar_path.unlink()
        get_logger().info("✅ CIFAR-10-C download complete!")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据集加载函数                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_dataset(cfg):
    """
    加载并预处理数据集

    参数:
        cfg: 配置对象

    返回:
        train_loader, val_loader, test_loader, corruption_dataset
    """
    dataset_name = cfg.dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"不支持的数据集: {dataset_name}. 支持: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]

    # 为没有官方划分的数据集传递额外参数
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["test_split"] = cfg.test_split

    # 创建完整训练集 (用于划分)
    train_full = DatasetClass(root=cfg.data_root, train=True, **extra_kwargs)
    test_dataset = DatasetClass(root=cfg.data_root, train=False, **extra_kwargs)

    # 划分训练集和验证集
    total_train = len(train_full)
    val_size = int(total_train * cfg.val_split)
    train_size = total_train - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(total_train, generator=generator)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    # 使用 PyTorch 内置 Subset
    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(train_full, val_indices)

    # 创建DataLoader
    common_loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "prefetch_factor": cfg.prefetch_factor if cfg.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True, **common_loader_kwargs
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size * 2, shuffle=False, **common_loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        **common_loader_kwargs,
    )

    get_logger().info(f"📊 数据集: {dataset_name.upper()}")
    get_logger().info(
        f"   训练集: {len(train_subset)} | 验证集: {len(val_subset)} | 测试集: {len(test_dataset)}"
    )

    # 加载Corruption数据集 (任何在 DATASET_REGISTRY 中的数据集都支持)
    corruption_dataset = None
    try:
        corruption_dataset = CorruptionDataset.from_name(dataset_name, cfg.data_root)
        get_logger().info(f"   Corruption数据集: {corruption_dataset.name}")
    except FileNotFoundError as e:
        get_logger().warning(f"   ⚠️ Corruption数据集未找到: {e}")

    return train_loader, val_loader, test_loader, corruption_dataset
