"""
================================================================================
Loss Landscape 可视化模块
================================================================================

包含: LossLandscapeVisualizer
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import ensure_dir, get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Loss Landscape 可视化器                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class LossLandscapeVisualizer:
    """Loss Landscape 可视化器

    用于可视化集成模型中各成员在损失地形上的位置分布。

    功能:
        - 1D 插值: 在两个模型之间线性插值，观察损失变化
        - 2D 平面: 围绕单个模型在随机方向上采样，生成等高线图
        - 模型间距离: 计算模型在参数空间中的欧氏距离

    依赖: pip install loss-landscapes
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)
        self.logger = get_logger()

    def _check_dependency(self):
        """检查 loss-landscapes 依赖"""
        import importlib.util

        if importlib.util.find_spec("loss_landscapes") is None:
            self.logger.warning(
                "⚠️ loss-landscapes 未安装，请运行: pip install loss-landscapes"
            )
            return False
        return True

    def _create_metric(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ):
        """创建损失评估器"""
        import loss_landscapes.metrics as metrics

        criterion = nn.CrossEntropyLoss()

        class LossMetric(metrics.Metric):
            """自定义损失评估器"""

            def __init__(self, criterion, dataloader, device):
                super().__init__()
                self.criterion = criterion
                self.dataloader = dataloader
                self.device = device

            def __call__(self, model):
                model.eval()
                total_loss = 0.0
                total_samples = 0
                with torch.no_grad():
                    for inputs, targets in self.dataloader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = model(inputs)
                        loss = self.criterion(outputs, targets)
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
                return total_loss / total_samples if total_samples > 0 else 0.0

        return LossMetric(criterion, dataloader, device)

    def plot_1d_interpolation(
        self,
        model1: nn.Module,
        model2: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        steps: int = 50,
        filename: str = "loss_landscape_1d.png",
        label1: str = "Model 1",
        label2: str = "Model 2",
    ) -> Optional[np.ndarray]:
        """绘制两个模型之间的1D损失插值曲线"""
        if not self._check_dependency():
            return None

        import loss_landscapes
        import matplotlib.pyplot as plt

        self.logger.info(f"📈 正在计算 1D Loss Landscape ({label1} → {label2})...")

        model1 = model1.to(device)
        model2 = model2.to(device)
        metric = self._create_metric(model1, dataloader, device)

        # 线性插值
        loss_data = loss_landscapes.linear_interpolation(
            model1, model2, metric, steps=steps
        )

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, 1, steps)
        ax.plot(x, loss_data, "b-", linewidth=2)
        ax.scatter([0, 1], [loss_data[0], loss_data[-1]], c="red", s=100, zorder=5)
        ax.annotate(
            label1,
            (0, loss_data[0]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )
        ax.annotate(
            label2,
            (1, loss_data[-1]),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )

        ax.set_xlabel("Interpolation (α)")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss Landscape: {label1} → {label2}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()

        self.logger.info(f"📊 Saved: {filename}")
        return loss_data

    def plot_2d_plane(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        distance: float = 1.0,
        steps: int = 40,
        filename: str = "loss_landscape_2d.png",
        model_name: str = "Model",
    ) -> Optional[np.ndarray]:
        """绘制模型周围的2D损失地形等高线图"""
        if not self._check_dependency():
            return None

        import loss_landscapes
        import matplotlib.pyplot as plt

        self.logger.info(f"📈 正在计算 2D Loss Landscape ({model_name})...")
        self.logger.info(
            f"   ⏳ 预计 {steps}×{steps}={steps * steps} 次前向传播，请耐心等待..."
        )

        model = model.to(device)
        metric = self._create_metric(model, dataloader, device)

        # 随机方向平面采样
        loss_data = loss_landscapes.random_plane(
            model, metric, distance=distance, steps=steps, normalization="filter"
        )
        self.logger.info("   ✅ 2D 采样完成")

        # 创建坐标网格
        x = np.linspace(-distance, distance, steps)
        y = np.linspace(-distance, distance, steps)
        X, Y = np.meshgrid(x, y)

        # 绘制等高线图
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, loss_data, levels=50, cmap="viridis")
        plt.colorbar(contour, ax=ax, label="Loss")
        ax.scatter([0], [0], c="red", s=100, marker="*", label=model_name, zorder=5)
        ax.legend()
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_title(f"2D Loss Landscape around {model_name}")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        self.logger.info(f"📊 Saved: {filename}")

        # 绘制 3D 表面图
        fig_3d = plt.figure(figsize=(12, 9))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        surf = ax_3d.plot_surface(
            X, Y, loss_data, cmap="viridis", edgecolor="none", alpha=0.9
        )
        fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, label="Loss")

        center_loss = loss_data[steps // 2, steps // 2]
        ax_3d.scatter(
            [0], [0], [center_loss], c="red", s=200, marker="*", label=model_name
        )

        ax_3d.set_xlabel("Direction 1")
        ax_3d.set_ylabel("Direction 2")
        ax_3d.set_zlabel("Loss")
        ax_3d.set_title(f"3D Loss Landscape around {model_name}")
        ax_3d.view_init(elev=30, azim=45)
        ax_3d.legend()

        filename_3d = filename.replace(".png", "_3d.png")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename_3d, dpi=150)
        plt.close()
        self.logger.info(f"📊 Saved: {filename_3d}")

        return loss_data

    def plot_ensemble_interpolations(
        self,
        models: List[nn.Module],
        dataloader: DataLoader,
        device: torch.device,
        steps: int = 50,
        filename: str = "ensemble_loss_landscape.png",
    ) -> Dict[str, np.ndarray]:
        """绘制集成中所有模型对之间的1D损失插值曲线"""
        if not self._check_dependency():
            return {}

        import loss_landscapes
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        n_models = len(models)
        if n_models < 2:
            self.logger.warning("⚠️ 需要至少 2 个模型来计算插值")
            return {}

        self.logger.info(
            f"📈 正在计算集成模型间的 Loss Landscape ({n_models} 个模型)..."
        )

        results = {}
        pairs = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]

        for idx, (i, j) in enumerate(
            tqdm(pairs, desc="Computing Loss Landscape", leave=False)
        ):
            model_i = models[i].to(device)
            model_j = models[j].to(device)
            metric = self._create_metric(model_i, dataloader, device)

            loss_data = loss_landscapes.linear_interpolation(
                model_i, model_j, metric, steps=steps
            )
            results[f"M{i + 1}-M{j + 1}"] = loss_data

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.linspace(0, 1, steps)
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for (pair_name, loss_data), color in zip(results.items(), colors):
            ax.plot(x, loss_data, label=pair_name, linewidth=1.5, color=color)

        ax.set_xlabel("Interpolation (α)")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Landscape: Pairwise Model Interpolations")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"📊 Saved: {filename}")
        return results

    def compute_model_distances(self, models: List[nn.Module]) -> np.ndarray:
        """计算模型间的参数空间欧氏距离"""
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
        """绘制模型间距离热力图"""
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
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()

        self.logger.info(f"📊 Saved: {filename}")
        return distance_matrix
