"""
================================================================================
数据集加载函数模块
================================================================================

包含: load_dataset
"""

import torch
from torch.utils.data import DataLoader, Subset

from ..utils import get_logger
from .preloaded import DATASET_REGISTRY
from .robustness import CorruptionDataset, OODDataset

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据集加载函数                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_dataset(cfg, mode: str = "all"):
    """
    按需加载数据集

    Args:
        cfg: 配置对象
        mode: 加载模式
            - "train": 仅返回 (train_loader, val_loader)
            - "eval": 仅返回 (test_loader, corruption_dataset, ood_dataset)
            - "all": 返回全部 (train_loader, val_loader, test_loader, c_ds, o_ds)
    """
    dataset_name = cfg.dataset_name.lower()
    DatasetClass = _get_dataset_class(dataset_name)

    if mode == "train":
        train_loader, val_loader = _prepare_train_loaders(cfg, DatasetClass)
        get_logger().info(f"📊 训练数据集加载完成: {dataset_name.upper()}")
        return train_loader, val_loader

    elif mode == "eval":
        test_loader = _prepare_test_loader(cfg, DatasetClass)
        robustness_suite = _init_robustness_group(cfg, dataset_name)
        get_logger().info(f"📊 评估数据集加载完成: {dataset_name.upper()}")
        return test_loader, *robustness_suite


def _get_dataset_class(name):
    """从注册表获取类，处理错误"""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"不支持的数据集: {name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


def _prepare_train_loaders(cfg, DatasetClass):
    """加载训练集和验证集"""
    extra = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra = {"test_split": cfg.test_split, "seed": cfg.seed}
    train_full = DatasetClass(root=cfg.data_root, train=True, **extra)

    # 训练/验证集划分
    v_size = int(len(train_full) * cfg.val_split)
    t_size = len(train_full) - v_size
    idx = torch.randperm(
        len(train_full), generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_sub = Subset(train_full, idx[:t_size].tolist())
    val_sub = Subset(train_full, idx[t_size:].tolist())

    kwargs = _get_loader_kwargs(cfg)
    train_loader = DataLoader(
        train_sub, batch_size=cfg.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_sub, batch_size=cfg.batch_size * 2, shuffle=False, **kwargs
    )
    return train_loader, val_loader


def _prepare_test_loader(cfg, DatasetClass):
    """仅加载测试集"""
    extra = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra = {"test_split": cfg.test_split, "seed": cfg.seed}
    test_ds = DatasetClass(root=cfg.data_root, train=False, **extra)

    kwargs = _get_loader_kwargs(cfg)
    return DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, **kwargs)


def _get_loader_kwargs(cfg):
    """获取 DataLoader 公共参数"""
    return {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
    }


def _init_robustness_group(cfg, name):
    """按需探索并加载鲁棒性数据集"""
    results = []

    # Corruption
    c_ds = None
    if cfg.corruption_dataset:
        try:
            c_ds = CorruptionDataset(name, cfg.data_root)
        except Exception as e:
            get_logger().warning(f"   ⚠️ Corruption 数据集不可用: {e}")
    results.append(c_ds)

    # OOD
    o_ds = None
    if cfg.ood_dataset:
        try:
            o_ds = OODDataset(id_dataset=name, root=cfg.data_root)
        except Exception as e:
            get_logger().warning(f"   ⚠️ OOD 数据集不可用: {e}")
    results.append(o_ds)

    return tuple(results)
