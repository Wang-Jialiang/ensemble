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
from .robustness import CorruptionDataset, DomainShiftDataset, OODDataset

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据集加载函数                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def load_dataset(cfg):
    """加载并预处理数据集 (主流程大纲)"""
    dataset_name = cfg.dataset_name.lower()
    DatasetClass = _get_dataset_class(dataset_name)

    # 1. 准备标准训练/验证/测试 Loader
    loaders = _prepare_standard_loaders(cfg, DatasetClass)
    
    # 2. 准备鲁棒性评估数据集 (基于 Cache)
    robustness_suite = _init_robustness_group(cfg, dataset_name)

    get_logger().info(f"📊 数据集初始化完成: {dataset_name.upper()}")
    return (*loaders, *robustness_suite)


def _get_dataset_class(name):
    """从注册表获取类，处理错误"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"不支持的数据集: {name}. 可用: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def _prepare_standard_loaders(cfg, DatasetClass):
    """执行数据集划分并创建标准 DataLoaders"""
    # 1. 实例化数据集 (处理 EuroSAT 等非官方划分情况)
    extra = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra = {"test_split": cfg.test_split, "seed": cfg.seed}
    train_full = DatasetClass(root=cfg.data_root, train=True, **extra)
    test_ds = DatasetClass(root=cfg.data_root, train=False, **extra)

    # 2. 训练/验证集划分
    v_size = int(len(train_full) * cfg.val_split)
    t_size = len(train_full) - v_size
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(cfg.seed))
    
    train_sub = Subset(train_full, idx[:t_size].tolist())
    val_sub = Subset(train_full, idx[t_size:].tolist())

    # 3. 构造 Loader
    kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
    }
    
    return (
        DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True, **kwargs),
        DataLoader(val_sub, batch_size=cfg.batch_size * 2, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, **kwargs)
    )


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

    # Domain
    d_ds = None
    if cfg.domain_dataset:
        try:
            d_ds = DomainShiftDataset(id_dataset=name, root=cfg.data_root)
        except Exception as e:
            get_logger().warning(f"   ⚠️ Domain Shift 数据集不可用: {e}")
    results.append(d_ds)

    return tuple(results)
