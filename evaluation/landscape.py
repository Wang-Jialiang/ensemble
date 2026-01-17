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

    def _compute_distance_matrix(self, models: List[nn.Module]) -> np.ndarray:
        """计算模型间的参数空间余弦距离矩阵"""
        n_models = len(models)
        distance_matrix = np.zeros((n_models, n_models))

        # 将所有模型参数展平
        flat_params = []
        for model in models:
            params = torch.cat(
                [p.data.view(-1).cpu() for p in model.parameters()]
            ).numpy()
            flat_params.append(params)

        # 计算成对余弦距离
        for i in range(n_models):
            for j in range(i + 1, n_models):
                cos_sim = np.dot(flat_params[i], flat_params[j]) / (
                    np.linalg.norm(flat_params[i]) * np.linalg.norm(flat_params[j])
                )
                dist = 1 - cos_sim
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def compute(self, models: List[nn.Module]) -> dict:
        """计算模型间的参数空间余弦距离及衍生指标

        余弦距离 = 1 - 余弦相似度，对参数尺度不敏感，
        更适合高维参数空间的比较。

        Args:
            models: 模型列表

        Returns:
            dict: 包含 distance_matrix 和衍生指标的字典
                - distance_matrix: [n_models, n_models] 距离矩阵 (0~2)
                - avg_distance: 平均距离
                - std_distance: 距离标准差
                - direction_diversity: 方向多样性 (std/avg, 上限1.0)
        """
        import math

        self.logger.info("📈 正在计算模型间参数距离 (余弦距离)...")

        n_models = len(models)
        distance_matrix = self._compute_distance_matrix(models)

        self.logger.info(f"✅ 距离矩阵计算完成 ({n_models}x{n_models})")

        # 计算衍生指标
        result = {"distance_matrix": distance_matrix}

        if n_models > 1:
            distances = [
                distance_matrix[i][j]
                for i in range(n_models)
                for j in range(i + 1, n_models)
            ]
            count = len(distances)

            avg_dist = sum(distances) / count if count > 0 else 0
            result["avg_distance"] = avg_dist

            if count > 1:
                variance = sum((d - avg_dist) ** 2 for d in distances) / count
                std_dist = math.sqrt(variance)
            else:
                std_dist = 0
            result["std_distance"] = std_dist

            result["direction_diversity"] = (
                min(std_dist / avg_dist, 1.0) if avg_dist > 0 else 0
            )
        else:
            result["avg_distance"] = 0
            result["std_distance"] = 0
            result["direction_diversity"] = 0

        return result
