"""
================================================================================
Corruption 鲁棒性评估模块
================================================================================

包含: evaluate_corruption 函数
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..config.core import Config
    from ..datasets.robustness.corruption import CorruptionDataset

import numpy as np

from ..utils import get_logger
from .inference import get_all_models_logits, get_models_from_source

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Corruption 鲁棒性评估                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_corruption(
    trainer_or_models: Any,
    corruption_dataset: "CorruptionDataset",
    config: "Config",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    通用 Corruption 鲁棒性评估
    
    Returns:
        {
            "by_severity": {1: 85.2, 3: 72.1, 5: 58.3},  # 各强度平均 acc
            "by_category": {"noise": 70.5, "blur": 68.2, ...},  # 四大类平均 acc
            "overall_avg": 72.1
        }
    """
    from tqdm import tqdm

    logger = logger or get_logger()
    dataset_name = corruption_dataset.name
    n_corruptions = len(corruption_dataset.CORRUPTIONS)
    n_severities = len(corruption_dataset.SEVERITIES)
    total_evals = n_severities * n_corruptions

    logger.info(f"\n🧪 Running Corruption Evaluation on {dataset_name}")
    logger.info(
        f"   📊 {n_corruptions} corruptions × {n_severities} severities = {total_evals} 次评估"
    )

    models, device = get_models_from_source(trainer_or_models)
    
    # 存储详细结果用于汇总
    detail_results = {}  # {severity: {corruption: acc}}
    
    pbar = tqdm(total=total_evals, desc="Corruption Eval", leave=False)

    for severity in corruption_dataset.SEVERITIES:
        detail_results[severity] = {}

        for corruption in corruption_dataset.CORRUPTIONS:
            pbar.set_postfix({"severity": severity, "type": corruption[:10]})

            loader = corruption_dataset.get_loader(
                corruption,
                severity=severity,
                config=config,
            )

            all_logits, targets = get_all_models_logits(models, loader, device)
            ensemble_logits = all_logits.mean(dim=0)
            ensemble_preds = ensemble_logits.argmax(dim=1)
            acc = 100.0 * (ensemble_preds == targets).float().mean().item()

            detail_results[severity][corruption] = acc
            pbar.update(1)

    pbar.close()

    # ========== 汇总结果 ==========
    results = {}
    
    # 1. 按 severity 汇总
    by_severity = {}
    for sev in corruption_dataset.SEVERITIES:
        by_severity[sev] = np.mean(list(detail_results[sev].values()))
    results["by_severity"] = by_severity
    
    # 2. 按四大类汇总 (跨所有 severity 平均)
    by_category = {}
    for cat_name, corruptions in corruption_dataset.CATEGORIES.items():
        cat_accs = []
        for sev in corruption_dataset.SEVERITIES:
            for c in corruptions:
                if c in detail_results[sev]:
                    cat_accs.append(detail_results[sev][c])
        by_category[cat_name] = np.mean(cat_accs) if cat_accs else 0.0
    results["by_category"] = by_category
    
    # 3. 总体平均
    results["overall_avg"] = np.mean(list(by_severity.values()))
    
    logger.info(f"   ✅ 完成! Overall Avg: {results['overall_avg']:.2f}%")
    return results

