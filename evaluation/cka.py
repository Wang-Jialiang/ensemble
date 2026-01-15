"""
================================================================================
CKA (Centered Kernel Alignment) 相似度模块
================================================================================

包含: _linear_cka (内部), compute_ensemble_cka
"""

from typing import Dict

import numpy as np
import torch

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ CKA (Centered Kernel Alignment) 相似度计算                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-10) -> float:
    """计算两个特征矩阵的 Linear CKA 相似度

    CKA (Centered Kernel Alignment) 用于衡量两个模型的表示相似度。
    相似度越高表示两个模型学到的表示越相似，多样性越低。

    Args:
        X: 第一个模型的特征矩阵，shape [n_samples, n_features]
        Y: 第二个模型的特征矩阵，shape [n_samples, n_features]
        eps: 数值稳定性阈值

    Returns:
        CKA 相似度，范围 [0, 1]，1 表示完全相同
    """
    # 中心化
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram 矩阵
    XX = X @ X.T
    YY = Y @ Y.T

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = (XX * YY).sum()
    hsic_xx = (XX * XX).sum()
    hsic_yy = (YY * YY).sum()

    # 数值稳定性检查：分别检查 XX 和 YY 的 HSIC
    # 如果其中一个模型输出接近常数，则返回 0
    if hsic_xx < eps or hsic_yy < eps:
        return 0.0

    # CKA
    cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    return float(cka.clamp(0.0, 1.0))  # 确保输出在有效范围内


def compute_ensemble_cka(all_features: torch.Tensor) -> Dict[str, float]:
    """计算集成模型中所有模型对的 CKA 相似度

    Args:
        all_features: [num_models, num_samples, feature_dim]
                      特征来自模型倒数第二层 (如 avgpool 输出)

    Returns:
        包含 CKA 统计信息的字典:
        - avg_cka: 平均 CKA 相似度 (上三角平均值)
        - min_cka: 最小 CKA 相似度 (多样性最高的模型对)
        - max_cka: 最大 CKA 相似度 (多样性最低的模型对)
        - cka_matrix: 完整的 CKA 相似度矩阵 (对角线为 1)
    """
    num_models = all_features.shape[0]
    if num_models < 2:
        return {"avg_cka": 1.0, "min_cka": 1.0, "max_cka": 1.0, "cka_matrix": [[1.0]]}

    # 构建完整的 CKA 矩阵
    cka_matrix = np.eye(num_models)
    cka_values = []

    for i in range(num_models):
        for j in range(i + 1, num_models):
            cka = _linear_cka(all_features[i], all_features[j])
            cka_matrix[i, j] = cka
            cka_matrix[j, i] = cka  # 对称矩阵
            cka_values.append(cka)

    return {
        "avg_cka": float(np.mean(cka_values)),
        "min_cka": float(np.min(cka_values)),
        "max_cka": float(np.max(cka_values)),
        "cka_matrix": cka_matrix.tolist(),
    }
