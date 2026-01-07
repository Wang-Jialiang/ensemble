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


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """计算两个特征矩阵的 Linear CKA 相似度

    CKA (Centered Kernel Alignment) 用于衡量两个模型的表示相似度。
    相似度越高表示两个模型学到的表示越相似，多样性越低。

    Args:
        X: 第一个模型的特征矩阵，shape [n_samples, n_features]
        Y: 第二个模型的特征矩阵，shape [n_samples, n_features]

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

    # CKA
    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    cka = hsic_xy / denominator
    return cka.item()


def compute_ensemble_cka(all_features: torch.Tensor) -> Dict[str, float]:
    """计算集成模型中所有模型对的 CKA 相似度

    Args:
        all_features: [num_models, num_samples, feature_dim]
                      特征来自模型倒数第二层 (如 avgpool 输出)

    Returns:
        包含 CKA 统计信息的字典:
        - avg_cka: 平均 CKA 相似度
        - min_cka: 最小 CKA 相似度
        - max_cka: 最大 CKA 相似度
        - cka_diversity: CKA 多样性 = 1 - avg_cka (越高表示越多样)
    """
    num_models = all_features.shape[0]
    if num_models < 2:
        return {
            "avg_cka": 1.0,
            "min_cka": 1.0,
            "max_cka": 1.0,
            "cka_diversity": 0.0,
        }

    cka_values = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            # 使用隐藏层特征作为表示
            X = all_features[i]  # [num_samples, feature_dim]
            Y = all_features[j]
            cka = _linear_cka(X, Y)
            cka_values.append(cka)

    avg_cka = np.mean(cka_values)
    return {
        "avg_cka": avg_cka,
        "min_cka": np.min(cka_values),
        "max_cka": np.max(cka_values),
        "cka_diversity": 1.0 - avg_cka,  # 多样性 = 1 - 相似度
    }
