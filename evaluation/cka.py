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
        - avg_cka: 平均 CKA 相似度 (上三角平均值)
    """
    num_models = all_features.shape[0]
    if num_models < 2:
        return {"avg_cka": 1.0}

    cka_values = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            X = all_features[i]
            Y = all_features[j]
            cka = _linear_cka(X, Y)
            cka_values.append(cka)

    return {"avg_cka": np.mean(cka_values)}
