"""
================================================================================
调度器与优化器模块
================================================================================

优化器工厂、学习率调度器工厂、早停机制
"""

from typing import Optional

import torch.nn as nn
import torch.optim as optim

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 优化器与调度器工厂函数                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_optimizer(model, name, lr, weight_decay, sgd_momentum=0.9) -> optim.Optimizer:
    """创建优化器 (大纲化)"""
    name = name.lower()
    params = model.parameters()

    # 1. 路由至具体的优化器类
    if name == "adamw": return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam": return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd": return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=sgd_momentum)
    if name == "rmsprop": return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"不支持的优化器: {name}")


def create_scheduler(optimizer, name, total_epochs) -> Optional[optim.lr_scheduler.LRScheduler]:
    """创建学习率调度器 (大纲化)"""
    name = name.lower()
    
    if name == "cosine": return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    if name == "none": return None
    
    raise ValueError(f"不支持的调度器: {name}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 早停机制                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class EarlyStopping:
    """早停控制逻辑 (大纲化)"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        """检查并更新早停计数器"""
        if self.best_score is None:
            self.best_score = score
            return False

        # 判定是否有所改善
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            return False
        
        # 处理无改善情况
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def _is_improvement(self, score):
        """核心准则判定段"""
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)
