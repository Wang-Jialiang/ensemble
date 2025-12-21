"""
================================================================================
Domain Shift 鲁棒性评估模块
================================================================================

包含: evaluate_domain_shift 函数
"""

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from ..utils import get_logger
from .inference import get_all_models_logits, get_models_from_source
from .metrics import MetricsCalculator
from .strategies import ENSEMBLE_STRATEGIES

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Domain Shift (域偏移) 评估                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_domain_shift(
    trainer_or_models: Any,
    domain_loader: DataLoader,
    domain_name: str = "Domain",
    num_classes: int = 10,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Domain Shift (域偏移) 评估

    评估模型在不同视觉域/风格上的分类准确率。

    Args:
        trainer_or_models: StagedEnsembleTrainer 实例或 List[nn.Module]
        domain_loader: Domain Shift 数据加载器（需包含正确的标签）
        domain_name: 域名称（用于日志）
        num_classes: 类别数量
        logger: 日志记录器

    Returns:
        包含域偏移评估指标的字典
    """
    logger = logger or get_logger()
    logger.info(f"\n🌍 Running Domain Shift Evaluation ({domain_name})")

    models, device = get_models_from_source(trainer_or_models)
    calculator = MetricsCalculator(num_classes=num_classes)
    ensemble_fn = ENSEMBLE_STRATEGIES["mean"]

    # 获取所有模型的 logits
    all_logits, targets = get_all_models_logits(models, domain_loader, device)

    if len(all_logits) == 0:
        logger.warning("   ⚠️ 无数据可评估")
        return {"domain_name": domain_name, "error": "No data"}

    # 计算指标
    metrics = calculator.calculate_all_metrics(all_logits, targets, ensemble_fn)

    results = {
        "domain_name": domain_name,
        "num_samples": len(targets),
        "domain_acc": metrics["ensemble_acc"],
        "domain_balanced_acc": metrics["balanced_acc"],
        "domain_worst_class_acc": metrics["worst_class_acc"],
        "domain_avg_individual_acc": metrics["avg_individual_acc"],
        # 集成 vs 单模型的提升
        "domain_ensemble_gain": metrics["ensemble_acc"] - metrics["avg_individual_acc"],
    }

    logger.info(f"   ✅ Domain Shift Results ({domain_name}):")
    logger.info(f"      Ensemble Acc: {results['domain_acc']:.2f}%")
    logger.info(f"      Balanced Acc: {results['domain_balanced_acc']:.2f}%")
    logger.info(f"      Ensemble Gain: {results['domain_ensemble_gain']:+.2f}%")

    return results
