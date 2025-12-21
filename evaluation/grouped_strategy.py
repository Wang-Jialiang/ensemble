"""
================================================================================
Grouped Ensemble 策略
================================================================================

Grouped Ensemble (组集成) 实现

核心思想: 将模型分组，先组内平均再组间平均，产生层次化集成效果。
- 可以增加多样性权重
- 支持不同的组间聚合策略

此模块通过扩展 ENSEMBLE_STRATEGIES 注册表实现，不修改原有代码。
"""

from typing import Optional

import torch

# 导入现有策略注册表
from .strategies import ENSEMBLE_STRATEGIES


def grouped_mean(
    all_logits: torch.Tensor,
    num_groups: int = 2,
) -> torch.Tensor:
    """
    分组平均聚合

    将 N 个模型分成 num_groups 组，每组内先平均，再组间平均。

    Args:
        all_logits: [num_models, batch_size, num_classes]
        num_groups: 分组数量

    Returns:
        [batch_size, num_classes] 集成 logits
    """
    num_models = all_logits.size(0)

    # 如果模型数少于分组数，直接平均
    if num_models <= num_groups:
        return all_logits.mean(dim=0)

    # 计算每组的模型数
    models_per_group = num_models // num_groups
    remainder = num_models % num_groups

    group_means = []
    start_idx = 0

    for g in range(num_groups):
        # 处理余数：前 remainder 个组多分一个模型
        group_size = models_per_group + (1 if g < remainder else 0)
        end_idx = start_idx + group_size

        # 组内平均
        group_logits = all_logits[start_idx:end_idx]
        group_mean = group_logits.mean(dim=0)
        group_means.append(group_mean)

        start_idx = end_idx

    # 组间平均
    return torch.stack(group_means).mean(dim=0)


def weighted_grouped_mean(
    all_logits: torch.Tensor,
    num_groups: int = 2,
    group_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    加权分组平均

    Args:
        all_logits: [num_models, batch_size, num_classes]
        num_groups: 分组数量
        group_weights: 可选的组间权重 [num_groups]

    Returns:
        [batch_size, num_classes]
    """
    num_models = all_logits.size(0)

    if num_models <= num_groups:
        return all_logits.mean(dim=0)

    models_per_group = num_models // num_groups
    remainder = num_models % num_groups

    group_means = []
    start_idx = 0

    for g in range(num_groups):
        group_size = models_per_group + (1 if g < remainder else 0)
        end_idx = start_idx + group_size

        group_logits = all_logits[start_idx:end_idx]
        group_mean = group_logits.mean(dim=0)
        group_means.append(group_mean)

        start_idx = end_idx

    stacked = torch.stack(group_means)  # [num_groups, batch_size, num_classes]

    if group_weights is not None:
        # 归一化权重
        weights = group_weights / group_weights.sum()
        weights = weights.view(-1, 1, 1).to(stacked.device)
        return (stacked * weights).sum(dim=0)
    else:
        return stacked.mean(dim=0)


def hierarchical_voting(
    all_logits: torch.Tensor,
    num_groups: int = 2,
) -> torch.Tensor:
    """
    层次化投票

    组内先投票得到组预测，再用组预测进行最终投票。

    Args:
        all_logits: [num_models, batch_size, num_classes]
        num_groups: 分组数量

    Returns:
        [batch_size, num_classes]
    """
    num_models = all_logits.size(0)
    batch_size = all_logits.size(1)
    num_classes = all_logits.size(2)

    if num_models <= num_groups:
        # 直接投票
        votes = all_logits.argmax(dim=2)  # [num_models, batch_size]
        vote_counts = torch.zeros(batch_size, num_classes, device=all_logits.device)
        for i in range(num_models):
            vote_counts.scatter_add_(
                1,
                votes[i].unsqueeze(1),
                torch.ones_like(votes[i].unsqueeze(1), dtype=vote_counts.dtype),
            )
        return vote_counts

    models_per_group = num_models // num_groups
    remainder = num_models % num_groups

    # 组内投票得到组预测
    group_predictions = []
    start_idx = 0

    for g in range(num_groups):
        group_size = models_per_group + (1 if g < remainder else 0)
        end_idx = start_idx + group_size

        # 组内投票
        group_logits = all_logits[
            start_idx:end_idx
        ]  # [group_size, batch_size, num_classes]
        group_votes = group_logits.argmax(dim=2)  # [group_size, batch_size]

        # 统计组内票数
        group_vote_counts = torch.zeros(
            batch_size, num_classes, device=all_logits.device
        )
        for i in range(group_size):
            group_vote_counts.scatter_add_(
                1,
                group_votes[i].unsqueeze(1),
                torch.ones_like(
                    group_votes[i].unsqueeze(1), dtype=group_vote_counts.dtype
                ),
            )

        # 组预测 = 组内最多票的类别
        group_pred = group_vote_counts.argmax(dim=1)  # [batch_size]
        group_predictions.append(group_pred)

        start_idx = end_idx

    # 组间投票
    final_vote_counts = torch.zeros(batch_size, num_classes, device=all_logits.device)
    for pred in group_predictions:
        final_vote_counts.scatter_add_(
            1,
            pred.unsqueeze(1),
            torch.ones_like(pred.unsqueeze(1), dtype=final_vote_counts.dtype),
        )

    return final_vote_counts


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 策略注册 (扩展现有注册表)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# 工厂函数，用于创建带参数的 grouped 策略
def create_grouped_fn(num_groups: int = 2):
    """创建指定分组数的 grouped_mean 函数"""

    def fn(all_logits: torch.Tensor) -> torch.Tensor:
        return grouped_mean(all_logits, num_groups=num_groups)

    return fn


def create_hierarchical_voting_fn(num_groups: int = 2):
    """创建指定分组数的 hierarchical_voting 函数"""

    def fn(all_logits: torch.Tensor) -> torch.Tensor:
        return hierarchical_voting(all_logits, num_groups=num_groups)

    return fn


# 注册默认策略 (2 组)
ENSEMBLE_STRATEGIES["grouped"] = create_grouped_fn(2)
ENSEMBLE_STRATEGIES["grouped_2"] = create_grouped_fn(2)
ENSEMBLE_STRATEGIES["grouped_3"] = create_grouped_fn(3)
ENSEMBLE_STRATEGIES["grouped_4"] = create_grouped_fn(4)

# 层次化投票
ENSEMBLE_STRATEGIES["hierarchical_voting"] = create_hierarchical_voting_fn(2)
ENSEMBLE_STRATEGIES["hierarchical_voting_2"] = create_hierarchical_voting_fn(2)
ENSEMBLE_STRATEGIES["hierarchical_voting_3"] = create_hierarchical_voting_fn(3)


__all__ = [
    "grouped_mean",
    "weighted_grouped_mean",
    "hierarchical_voting",
    "create_grouped_fn",
    "create_hierarchical_voting_fn",
]
