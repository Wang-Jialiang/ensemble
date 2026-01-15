"""
================================================================================
OOD (Out-of-Distribution) 检测评估模块
================================================================================

支持的 OOD 检测方法:
- MSP (Maximum Softmax Probability): 基线方法
- Energy Score: Liu et al., NeurIPS 2020
- Mahalanobis Distance: Lee et al., NeurIPS 2018 (推荐用于 Near-OOD)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..utils import get_logger
from .inference import (
    _FeatureExtractor,
    get_all_models_logits,
    get_models_from_source,
)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 分数计算方法                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _compute_msp_scores(logits: torch.Tensor) -> np.ndarray:
    """计算 MSP (Maximum Softmax Probability) 分数

    Args:
        logits: [N, C] 或 [M, N, C] (ensemble 平均后为 [N, C])

    Returns:
        msp: [N] 每个样本的最大 softmax 概率 (ID 分数高)
    """
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1)[0].cpu().numpy()


def _compute_energy_scores(
    logits: torch.Tensor, temperature: float = 1.0
) -> np.ndarray:
    """计算 Energy Score (Liu et al., NeurIPS 2020)

    E(x) = -T * log(sum_c exp(f_c(x) / T))

    Args:
        logits: [N, C] 模型输出
        temperature: 温度参数 (默认 1.0)

    Returns:
        energy: [N] 能量分数 (ID 分数低/负值更小, OOD 分数高)
    """
    # 注意: logsumexp 值越大表示能量越低 (ID), 我们取负值使 ID 分数 > OOD 分数
    return -torch.logsumexp(logits / temperature, dim=-1).cpu().numpy()


def _compute_mahalanobis_scores(
    features: torch.Tensor,
    class_means: torch.Tensor,
    precision: torch.Tensor,
) -> np.ndarray:
    """计算 Mahalanobis Distance (Lee et al., NeurIPS 2018)

    Args:
        features: [N, D] 样本特征
        class_means: [C, D] 类条件均值
        precision: [D, D] 共享精度矩阵 (协方差逆)

    Returns:
        scores: [N] 负马氏距离 (ID 分数高, OOD 分数低)
    """
    num_classes = class_means.shape[0]
    scores = []

    for sample_feat in features:
        # 计算到每个类中心的马氏距离
        dists = []
        for c in range(num_classes):
            diff = sample_feat - class_means[c]  # [D]
            dist = torch.dot(diff, torch.mv(precision, diff))  # 标量
            dists.append(dist.item())
        # 取最小距离 (最近的类)
        scores.append(-min(dists))  # 取负使 ID 分数高

    return np.array(scores)


def _fit_gaussian(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """拟合类条件高斯分布

    Args:
        features: [N, D] ID 数据特征
        labels: [N] 标签
        num_classes: 类别数

    Returns:
        class_means: [C, D] 类均值
        precision: [D, D] 共享精度矩阵
    """
    device = features.device
    feat_dim = features.shape[1]

    # 计算类均值
    class_means = torch.zeros(num_classes, feat_dim, device=device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(dim=0)

    # 计算共享协方差
    centered = features - class_means[labels]  # [N, D]
    cov = (centered.T @ centered) / len(features)  # [D, D]

    # 添加正则化保证数值稳定
    cov += torch.eye(feat_dim, device=device) * 1e-6

    # 计算精度矩阵 (协方差逆)
    precision = torch.linalg.inv(cov)

    return class_means, precision


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 特征提取辅助函数                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _extract_ensemble_features(
    models: List[nn.Module], loader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取 ensemble 平均特征

    Args:
        models: 模型列表
        loader: 数据加载器
        device: 计算设备

    Returns:
        features: [N, D] 平均特征
        labels: [N] 标签
    """
    from tqdm import tqdm

    extractors = [_FeatureExtractor(m) for m in models]
    all_feats, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Feature Extraction", leave=False):
            batch_feats = []
            for ext in extractors:
                ext.model.eval()
                model_device = next(ext.model.parameters()).device
                feats = ext.extract(x.to(model_device))
                batch_feats.append(feats.cpu())

            # Ensemble 平均
            avg_feat = torch.stack(batch_feats).mean(dim=0)  # [B, D]
            all_feats.append(avg_feat)
            all_labels.append(y)

    # 清理 hooks
    for ext in extractors:
        ext.remove_hook()

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ OOD 检测评估主函数                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def evaluate_ood(
    trainer_or_models,
    id_loader,
    ood_loader,
    ood_name="OOD",
    num_classes=None,
    logger=None,
) -> Dict:
    """OOD 检测评估

    Args:
        trainer_or_models: Trainer 对象, Worker 列表, 或模型列表
        id_loader: ID (In-Distribution) 数据加载器
        ood_loader: OOD 数据加载器
        ood_name: OOD 数据集名称 (用于日志)
        num_classes: 类别数 (用于 Mahalanobis, 若为 None 则从 ID 标签推断)
        logger: 日志器

    Returns:
        包含以下指标的字典:
        - ood_auroc_msp: MSP 方法的 AUROC
        - ood_fpr95_msp: MSP 方法在 95% TPR 时的 FPR
        - ood_auroc_energy: Energy 方法的 AUROC
        - ood_fpr95_energy: Energy 方法在 95% TPR 时的 FPR
        - ood_auroc_mahalanobis: Mahalanobis 方法的 AUROC
        - ood_fpr95_mahalanobis: Mahalanobis 方法在 95% TPR 时的 FPR
    """
    log = logger or get_logger()
    log.info(f"🔍 OOD Eval ({ood_name})")

    models, device = get_models_from_source(trainer_or_models)

    # ==================== 1. 提取 Logits ====================
    log.info("  📊 Extracting logits...")
    id_logits, id_labels = get_all_models_logits(models, id_loader, device)
    ood_logits, _ = get_all_models_logits(models, ood_loader, device)

    if id_logits.numel() == 0 or ood_logits.numel() == 0:
        return {"error": "Empty data"}

    # Ensemble 平均 logits
    id_logits_avg = id_logits.mean(dim=0)  # [N_id, C]
    ood_logits_avg = ood_logits.mean(dim=0)  # [N_ood, C]

    # ==================== 2. 计算 MSP 和 Energy 分数 ====================
    id_msp = _compute_msp_scores(id_logits_avg)
    ood_msp = _compute_msp_scores(ood_logits_avg)

    id_energy = _compute_energy_scores(id_logits_avg)
    ood_energy = _compute_energy_scores(ood_logits_avg)

    # ==================== 3. 计算 Mahalanobis 分数 ====================
    log.info("  📊 Extracting features for Mahalanobis...")
    id_features, id_feat_labels = _extract_ensemble_features(models, id_loader, device)
    ood_features, _ = _extract_ensemble_features(models, ood_loader, device)

    # 推断类别数
    if num_classes is None:
        num_classes = int(id_feat_labels.max().item()) + 1

    # 在 ID 数据上拟合高斯分布
    class_means, precision = _fit_gaussian(id_features, id_feat_labels, num_classes)

    id_mahal = _compute_mahalanobis_scores(id_features, class_means, precision)
    ood_mahal = _compute_mahalanobis_scores(ood_features, class_means, precision)

    # ==================== 4. 计算评估指标 ====================
    y = np.concatenate([np.ones(len(id_msp)), np.zeros(len(ood_msp))])

    def _auroc(id_scores, ood_scores):
        scores = np.concatenate([id_scores, ood_scores])
        return roc_auc_score(y, scores) * 100

    res = {
        # MSP
        "ood_auroc_msp": _auroc(id_msp, ood_msp),
        "ood_fpr95_msp": _compute_fpr_at_95tpr(np.concatenate([id_msp, ood_msp]), y),
        # Energy
        "ood_auroc_energy": _auroc(id_energy, ood_energy),
        "ood_fpr95_energy": _compute_fpr_at_95tpr(
            np.concatenate([id_energy, ood_energy]), y
        ),
        # Mahalanobis
        "ood_auroc_mahalanobis": _auroc(id_mahal, ood_mahal),
        "ood_fpr95_mahalanobis": _compute_fpr_at_95tpr(
            np.concatenate([id_mahal, ood_mahal]), y
        ),
    }

    log.info(
        f"  ✅ MSP AUROC: {res['ood_auroc_msp']:.2f}%, Energy AUROC: {res['ood_auroc_energy']:.2f}%, Mahalanobis AUROC: {res['ood_auroc_mahalanobis']:.2f}%"
    )

    return res


def _compute_fpr_at_95tpr(scores, labels):
    """计算 95% TPR 下的 FPR"""
    id_scores, ood_scores = scores[labels == 1], scores[labels == 0]
    thresh = np.percentile(id_scores, 5)  # 5th percentile = 95% TPR
    return (ood_scores >= thresh).mean() * 100 if len(ood_scores) else 0.0
