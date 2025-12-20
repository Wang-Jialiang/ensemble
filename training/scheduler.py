"""
================================================================================
调度器与优化器模块
================================================================================

优化器工厂、学习率调度器工厂、早停机制
"""

from typing import Optional

import torch.nn as nn
import torch.optim as optim

# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 常量定义                                                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

# 学习率调度器默认参数
STEP_LR_MILESTONES = (0.3, 0.6)  # 在这些位置降低学习率
STEP_LR_GAMMA = 0.1  # 学习率衰减因子
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 5


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 优化器与调度器工厂函数                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_optimizer(
    model: nn.Module, optimizer_name: str, lr: float, weight_decay: float
) -> optim.Optimizer:
    """
    创建优化器

    Args:
        model: 模型
        optimizer_name: 优化器名称 (adamw, sgd, adam, rmsprop)
        lr: 学习率
        weight_decay: 权重衰减

    Returns:
        optimizer: 优化器实例
    """
    optimizer_name = optimizer_name.lower()
    params = model.parameters()

    if optimizer_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"不支持的优化器: {optimizer_name}. 支持: adamw, sgd, adam, rmsprop"
        )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    total_epochs: int,
    steps_per_epoch: int = 0,
    max_lr_factor: float = 10.0,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 (cosine, step, plateau, onecycle, none)
        total_epochs: 总训练轮数
        steps_per_epoch: 每轮步数 (用于 OneCycleLR)
        max_lr_factor: OneCycleLR 最大学习率倍数 (默认10)

    Returns:
        scheduler: 调度器实例，none 时返回 None
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif scheduler_name == "step":
        milestones = [int(total_epochs * m) for m in STEP_LR_MILESTONES]
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=STEP_LR_GAMMA
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE
        )
    elif scheduler_name == "onecycle":
        if steps_per_epoch <= 0:
            raise ValueError("OneCycleLR 需要 steps_per_epoch > 0")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"] * max_lr_factor,
            total_steps=total_epochs * steps_per_epoch,
        )
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(
            f"不支持的调度器: {scheduler_name}. 支持: cosine, step, plateau, onecycle, none"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 早停机制                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class EarlyStopping:
    """早停机制

    用于在验证指标不再改善时提前停止训练，防止过拟合。

    Args:
        patience: 允许的最大无改善的epoch数
        min_delta: 最小改善阈值
        mode: 'min' 或 'max'，指定指标是越小越好还是越大越好
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """检查是否应该早停

        Returns:
            True 如果应该停止训练，否则 False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
