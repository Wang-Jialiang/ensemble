"""
================================================================================
Loss Landscape 可视化模块
================================================================================

包含: LossLandscapeVisualizer
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..utils import ensure_dir, get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Loss Landscape 可视化器                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class LossLandscapeVisualizer:
    """Loss Landscape 可视化器

    用于可视化集成模型中各成员在参数空间中的距离关系。
    通过距离热力图评估 Ensemble 的多样性。
    """

    def __init__(self, save_dir: str, dpi: int = 150):
        self.save_dir = Path(save_dir)
        self.dpi = dpi
        ensure_dir(self.save_dir)
        self.logger = get_logger()

    def compute_model_distances(self, models: List[nn.Module]) -> np.ndarray:
        """计算模型间的参数空间欧氏距离
        
        Args:
            models: 模型列表
            
        Returns:
            distance_matrix: [n_models, n_models] 距离矩阵
        """
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

        return distance_matrix

    def plot_model_distance_heatmap(
        self,
        models: List[nn.Module],
        filename: str = "model_distances.png",
    ) -> np.ndarray:
        """绘制模型间距离热力图
        
        Args:
            models: 模型列表
            filename: 输出文件名
            
        Returns:
            distance_matrix: 距离矩阵
        """
        import matplotlib.pyplot as plt

        self.logger.info("📈 正在计算模型间参数距离...")

        distance_matrix = self.compute_model_distances(models)
        n_models = len(models)

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(distance_matrix, cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="Euclidean Distance")

        # 设置标签
        labels = [f"M{i + 1}" for i in range(n_models)]
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # 在每个格子中显示数值
        for i in range(n_models):
            for j in range(n_models):
                ax.text(
                    j,
                    i,
                    f"{distance_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black"
                    if distance_matrix[i, j] < distance_matrix.max() / 2
                    else "white",
                    fontsize=8,
                )

        ax.set_title("Model Parameter Space Distances")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=self.dpi)
        plt.close()

        self.logger.info(f"📊 Saved: {filename}")
        return distance_matrix
