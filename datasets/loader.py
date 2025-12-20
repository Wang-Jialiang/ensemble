"""
================================================================================
数据集加载函数模块
================================================================================

包含: load_dataset
"""

import torch
from torch.utils.data import DataLoader, Subset

from ..utils import get_logger
from .corruption import CorruptionDataset
from .preloaded import DATASET_REGISTRY

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
