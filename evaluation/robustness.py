"""
================================================================================
鲁棒性评估模块
================================================================================

包含: Corruption 评估、域偏移评估
"""

from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader

from ..utils import get_logger
from .core import (
    ENSEMBLE_STRATEGIES,
    MetricsCalculator,
    extract_models,
    get_all_models_logits,
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption 鲁棒性评估                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_corruption(
    trainer_or_models: Any,
    corruption_dataset: "CorruptionDataset",
    batch_size: int = 128,
    num_workers: int = 4,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    通用 Corruption 鲁棒性评估
    """
    from tqdm import tqdm

    logger = logger or get_logger()
    dataset_name = corruption_dataset.name
    n_corruptions = len(corruption_dataset.CORRUPTIONS)
    total_evals = 5 * n_corruptions  # 5 severity × N corruptions

    logger.info(f"\n🧪 Running Corruption Evaluation on {dataset_name}")
    logger.info(
        f"   📊 {n_corruptions} corruptions × 5 severities = {total_evals} 次评估"
    )

    models, device = extract_models(trainer_or_models)
    results = {}
    overall_avg = 0.0

    # 创建总进度条
    pbar = tqdm(total=total_evals, desc="Corruption Eval", leave=False)

    for severity in range(1, 6):
        results[severity] = {}
        severity_accs = []

        for corruption in corruption_dataset.CORRUPTIONS:
            pbar.set_postfix({"severity": severity, "type": corruption[:10]})

            loader = corruption_dataset.get_loader(
                corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # 只需预测，不需要详细的 Calculator (只算 Acc)
            all_logits, targets = get_all_models_logits(models, loader, device)

            ensemble_logits = all_logits.mean(dim=0)
            ensemble_preds = ensemble_logits.argmax(dim=1)
            acc = 100.0 * (ensemble_preds == targets).float().mean().item()

            results[severity][corruption] = acc
            severity_accs.append(acc)
            pbar.update(1)

        avg_acc_sev = np.mean(severity_accs)
        results[severity]["avg"] = avg_acc_sev
        overall_avg += avg_acc_sev

    pbar.close()

    overall_avg /= 5.0
    logger.info(f"   ✅ 完成! Overall Avg: {overall_avg:.2f}%")

    results["severity_5_raw"] = results[5]
    results["overall_avg"] = overall_avg
    return results


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

    models, device = extract_models(trainer_or_models)
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
