"""
================================================================================
集成策略模块
================================================================================

包含: ENSEMBLE_STRATEGIES, EnsembleFn, get_ensemble_fn
"""

from typing import TYPE_CHECKING, Callable, Dict

import torch

if TYPE_CHECKING:
    from ..config import Config

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 集成策略                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

EnsembleFn = Callable[[torch.Tensor], torch.Tensor]


def _voting_fn(all_logits: torch.Tensor) -> torch.Tensor:
    """多数投票 (将投票结果转换为 logits)"""
    preds = all_logits.argmax(dim=2)  # [N, Samples]
    num_classes = all_logits.shape[2]
    votes = torch.zeros(preds.shape[1], num_classes, device=all_logits.device)
    for i in range(preds.shape[0]):
        votes.scatter_add_(
            1,
            preds[i].unsqueeze(1),
            torch.ones_like(preds[i].unsqueeze(1), dtype=votes.dtype),
        )
    return votes


ENSEMBLE_STRATEGIES: Dict[str, EnsembleFn] = {
    "mean": lambda logits: logits.mean(dim=0),
    "voting": _voting_fn,
}


def get_ensemble_fn(cfg: "Config") -> EnsembleFn:
    """从配置获取集成函数"""
    strategy = getattr(cfg, "ensemble_strategy", "mean")
    return ENSEMBLE_STRATEGIES.get(strategy, ENSEMBLE_STRATEGIES["mean"])
