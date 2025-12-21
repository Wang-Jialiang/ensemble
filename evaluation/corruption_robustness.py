"""
================================================================================
Corruption 鲁棒性评估模块
================================================================================

包含: evaluate_corruption 函数
"""

from typing import Any, Dict, Optional

import numpy as np

from ..utils import get_logger
from .inference import get_all_models_logits, get_models_from_source

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

    models, device = get_models_from_source(trainer_or_models)
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
