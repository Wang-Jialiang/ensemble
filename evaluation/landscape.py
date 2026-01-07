"""
================================================================================
模型距离计算模块
================================================================================

包含: ModelDistanceCalculator - 计算模型参数空间距离
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型距离计算器                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ModelDistanceCalculator:
    """模型距离计算器

    计算集成模型中各成员在参数空间中的距离关系，
    用于评估 Ensemble 的多样性。
    """

    def __init__(self):
        self.logger = get_logger()

    def compute(self, models: List[nn.Module]) -> np.ndarray:
        """计算模型间的参数空间欧氏距离

        Args:
            models: 模型列表

        Returns:
            distance_matrix: [n_models, n_models] 距离矩阵
        """
        self.logger.info("📈 正在计算模型间参数距离...")

        n_models = len(models)
        distance_matrix = np.zeros((n_models, n_models))

        # 将所有模型参数展平
        flat_params = []
        for model in models:
            params = torch.cat(
                [p.data.view(-1).cpu() for p in model.parameters()]
            ).numpy()
            flat_params.append(params)

        # 计算成对距离
        for i in range(n_models):
            for j in range(i + 1, n_models):
                dist = np.linalg.norm(flat_params[i] - flat_params[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        self.logger.info(f"✅ 距离矩阵计算完成 ({n_models}x{n_models})")
        return distance_matrix
