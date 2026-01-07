"""
================================================================================
调度器与优化器模块
================================================================================

优化器工厂、学习率调度器工厂、早停机制
"""

from typing import Optional

import torch.optim as optim

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 优化器与调度器工厂函数                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_optimizer(
    model, name, lr, weight_decay, sgd_momentum=0.9
) -> optim.Optimizer:
    """创建优化器 (大纲化)"""
    name = name.lower()
    params = model.parameters()

    # 1. 路由至具体的优化器类
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=sgd_momentum
        )
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"不支持的优化器: {name}")


def create_scheduler(
    optimizer, name, total_epochs
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """创建学习率调度器 (大纲化)"""
    name = name.lower()

    # 预设 optimizer step 计数，避免 PyTorch 1.1.0+ 的 lr_scheduler 警告
    # 警告: "Detected call of lr_scheduler.step() before optimizer.step()"
    if not hasattr(optimizer, "_step_count"):
        optimizer._step_count = 1

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    if name == "none":
        return None

    raise ValueError(f"不支持的调度器: {name}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 早停机制                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class EarlyStopping:
    """早停控制逻辑 (多指标监控版本)

    仅支持多指标配置:
    es = EarlyStopping(patience=10, metrics={"loss": "min", "acc": "max"}, criteria="any")

    Args:
        metrics: dict {指标名: 优化方向('min'/'max')}
        criteria: 'any' (任一指标改善即重置) 或 'all' (所有指标均无改善才计数)
        min_delta: 最小改善阈值
    """

    def __init__(
        self,
        metrics: dict,
        patience: int = 10,
        min_delta: float = 0.0,
        criteria: str = "any",
    ):
        self.metrics = metrics
        self.patience = patience
        self.min_delta = min_delta
        self.criteria = criteria

        self.counter = 0
        self.best_scores = {}  # {metric_name: best_value}
        self.early_stop = False

    def __call__(self, max_scores: dict, epoch: int = None) -> bool:
        """检查并更新早停计数器

        Args:
            max_scores: dict {metric_name: current_value}
        """
        # 1. 初始化首次运行
        if not self.best_scores:
            self.best_scores = max_scores
            return False

        # 2. 判定是否有所改善
        improvements = []
        for name, mode in self.metrics.items():
            if name not in max_scores:
                continue

            cur_val = max_scores[name]
            best_val = self.best_scores[name]

            if self._is_better(cur_val, best_val, mode):
                self.best_scores[name] = cur_val
                improvements.append(True)
            else:
                improvements.append(False)

        # 3. 根据 criteria 决定是否重置计数器
        if not improvements:
            is_reset = False
        elif self.criteria == "any":
            is_reset = any(improvements)
        elif self.criteria == "all":
            is_reset = all(improvements)
        else:
            is_reset = any(improvements)

        if is_reset:
            self.counter = 0
            return False

        # 4. 处理无改善情况
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def _is_better(self, current, best, mode):
        """判断单一指标是否变好"""
        if mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)
