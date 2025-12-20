"""
================================================================================
OOD (Out-of-Distribution) 检测评估模块
================================================================================
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import get_logger
from .core import extract_models

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 检测评估                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_ood(
    trainer_or_models: Any,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    ood_name: str = "OOD",
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    OOD (Out-of-Distribution) 检测评估

    使用集成模型的置信度/熵来区分 ID 和 OOD 样本。

    Args:
        trainer_or_models: StagedEnsembleTrainer 实例或 List[nn.Module]
        id_loader: ID (In-Distribution) 测试数据加载器
        ood_loader: OOD 数据加载器
        ood_name: OOD 数据集名称（用于日志）
        logger: 日志记录器

    Returns:
        包含 OOD 检测指标的字典:
        - ood_auroc_msp: 基于 MSP 的 AUROC
        - ood_auroc_entropy: 基于熵的 AUROC
        - ood_fpr95_msp: 基于 MSP 的 FPR@95%TPR
        - ood_fpr95_entropy: 基于熵的 FPR@95%TPR
    """
    from sklearn.metrics import roc_auc_score

    logger = logger or get_logger()
    logger.info(f"\n🔍 Running OOD Detection Evaluation ({ood_name})")

    models, device = extract_models(trainer_or_models)

    def get_confidence_scores(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """获取集成模型的置信度分数"""
        all_msp = []
        all_entropy = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)

                # 收集所有模型的 logits
                batch_logits = []
                for model in models:
                    model.eval()
                    logits = model(inputs)
                    batch_logits.append(logits.unsqueeze(0))

                # 集成 logits: [num_models, batch_size, num_classes] -> [batch_size, num_classes]
                all_model_logits = torch.cat(batch_logits, dim=0)
                ensemble_logits = all_model_logits.mean(dim=0)

                # 计算概率
                probs = F.softmax(ensemble_logits, dim=1)

                # MSP (Maximum Softmax Probability)
                msp = probs.max(dim=1)[0].cpu().numpy()
                all_msp.extend(msp)

                # 熵 (Entropy): -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
                all_entropy.extend(entropy)

        return np.array(all_msp), np.array(all_entropy)

    # 获取 ID 和 OOD 的置信度分数
    logger.info("   📊 计算 ID 样本置信度...")
    id_msp, id_entropy = get_confidence_scores(id_loader)

    logger.info("   📊 计算 OOD 样本置信度...")
    ood_msp, ood_entropy = get_confidence_scores(ood_loader)

    # 创建标签: ID=1, OOD=0
    id_labels = np.ones(len(id_msp))
    ood_labels = np.zeros(len(ood_msp))

    all_labels = np.concatenate([id_labels, ood_labels])
    all_msp_combined = np.concatenate([id_msp, ood_msp])
    all_entropy_combined = np.concatenate([id_entropy, ood_entropy])

    # 边界检查：确保有足够的数据计算 AUROC
    if len(id_msp) == 0 or len(ood_msp) == 0:
        logger.warning("   ⚠️ ID 或 OOD 数据为空，无法计算 AUROC")
        return {
            "ood_dataset": ood_name,
            "id_samples": len(id_msp),
            "ood_samples": len(ood_msp),
            "error": "Empty data",
        }

    # 计算 AUROC
    # MSP: 高值表示 ID（所以直接用）
    auroc_msp = roc_auc_score(all_labels, all_msp_combined) * 100.0

    # Entropy: 低值表示 ID（所以用负值）
    auroc_entropy = roc_auc_score(all_labels, -all_entropy_combined) * 100.0

    # 计算 FPR@95%TPR
    def compute_fpr_at_tpr(
        scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95
    ) -> float:
        """计算给定 TPR 下的 FPR"""
        # 对于 ID 分数高的情况
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        # 找到使 TPR >= tpr_target 的阈值
        sorted_pos = np.sort(pos_scores)
        threshold_idx = int(len(sorted_pos) * (1 - tpr_target))
        threshold = (
            sorted_pos[threshold_idx]
            if threshold_idx < len(sorted_pos)
            else sorted_pos[0]
        )

        # 计算 FPR (避免除零)
        if len(neg_scores) == 0:
            return 0.0
        fpr = (neg_scores >= threshold).sum() / len(neg_scores)
        return fpr * 100.0

    fpr95_msp = compute_fpr_at_tpr(all_msp_combined, all_labels, 0.95)
    fpr95_entropy = compute_fpr_at_tpr(-all_entropy_combined, all_labels, 0.95)

    results = {
        "ood_dataset": ood_name,
        "id_samples": len(id_msp),
        "ood_samples": len(ood_msp),
        "ood_auroc_msp": auroc_msp,
        "ood_auroc_entropy": auroc_entropy,
        "ood_fpr95_msp": fpr95_msp,
        "ood_fpr95_entropy": fpr95_entropy,
        # 置信度统计
        "id_msp_mean": float(np.mean(id_msp)),
        "ood_msp_mean": float(np.mean(ood_msp)),
        "id_entropy_mean": float(np.mean(id_entropy)),
        "ood_entropy_mean": float(np.mean(ood_entropy)),
    }

    logger.info(f"   ✅ OOD Detection Results ({ood_name}):")
    logger.info(f"      AUROC (MSP): {auroc_msp:.2f}%")
    logger.info(f"      AUROC (Entropy): {auroc_entropy:.2f}%")
    logger.info(f"      FPR@95 (MSP): {fpr95_msp:.2f}%")
    logger.info(f"      FPR@95 (Entropy): {fpr95_entropy:.2f}%")

    return results
