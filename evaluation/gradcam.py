"""
================================================================================
Grad-CAM 分析 + Loss Landscape 可视化模块
================================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import ensure_dir, get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Grad-CAM 目标层辅助函数                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """获取模型的目标层用于Grad-CAM

    根据模型架构自动确定适合用于Grad-CAM可视化的目标层。

    Args:
        model: PyTorch模型
        model_name: 模型名称 (resnet18, vgg16, efficientnet_b0等)

    Returns:
        目标层模块

    Raises:
        ValueError: 无法自动确定目标层时抛出
    """
    model_name = model_name.lower()
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "vgg" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    else:
        # 默认尝试layer4
        if hasattr(model, "layer4"):
            return model.layer4[-1]
        raise ValueError(f"无法自动确定 {model_name} 的目标层，请手动指定")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Grad-CAM 热力图生成器                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) - 基于 pytorch-grad-cam 库

    用于生成模型注意力热力图，可视化模型关注的图像区域。

    依赖: pip install grad-cam
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """初始化Grad-CAM

        Args:
            model: PyTorch模型
            target_layer: 用于生成CAM的目标层 (使用get_target_layer获取)
        """
        self.model = model
        self.target_layer = target_layer
        self._cam = None  # 延迟初始化

    def _get_cam(self):
        """延迟初始化 CAM 对象"""
        if self._cam is None:
            try:
                from pytorch_grad_cam import GradCAM as LibGradCAM

                self._cam = LibGradCAM(
                    model=self.model, target_layers=[self.target_layer]
                )
            except ImportError:
                raise ImportError("需要安装 grad-cam: pip install grad-cam")
        return self._cam

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """生成Grad-CAM热力图 (单张图像)

        Args:
            input_tensor: 输入张量，shape (1, C, H, W)
            target_class: 目标类别索引

        Returns:
            CAM 热力图，shape (H, W)，值在 [0, 1]
        """
        cams = self.generate_cam_batch(input_tensor, [target_class])
        return cams[0]

    def generate_cam_batch(
        self, input_tensor: torch.Tensor, target_classes: List[int]
    ) -> np.ndarray:
        """批量生成Grad-CAM热力图 (性能优化版)

        Args:
            input_tensor: 输入张量，shape (N, C, H, W)
            target_classes: 目标类别列表，长度为 N

        Returns:
            CAM 热力图数组，shape (N, H, W)，值在 [0, 1]
        """
        try:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            raise ImportError("需要安装 grad-cam: pip install grad-cam")

        cam_obj = self._get_cam()

        # 批量生成 CAM - 一次处理整个 batch
        targets = [ClassifierOutputTarget(cls) for cls in target_classes]
        grayscale_cams = cam_obj(input_tensor=input_tensor, targets=targets)

        # 返回 shape (N, H, W)
        return grayscale_cams

    def remove_hooks(self):
        """移除hooks，释放资源"""
        if self._cam is not None:
            del self._cam
            self._cam = None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Grad-CAM多样性分析器                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class GradCAMAnalyzer:
    """
    Grad-CAM多样性质量分析器

    用于分析集成模型中各成员的注意力分布多样性。

    分析指标:
        - avg_cam_entropy: CAM热力图的熵 (越高表示注意力越分散)
        - avg_cam_similarity: 模型间CAM相似度 (越低表示越多样)
        - avg_cam_overlap: CAM热点区域重叠度 (越低表示越多样)
        - pred_cam_correlation: 预测与CAM的相关性
    """

    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.model_name = cfg.model_name

    def analyze_ensemble_quality(
        self, workers: List, test_loader, num_samples: int = 50, image_size: int = 32
    ) -> Dict[str, Any]:
        """分析集成模型的Grad-CAM多样性

        Returns:
            metrics: 包含per_model和overall指标的字典
        """
        # 收集样本
        samples = []
        labels = []
        for inputs, targets in test_loader:
            for i in range(min(len(inputs), num_samples - len(samples))):
                samples.append(inputs[i : i + 1])
                labels.append(targets[i].item())
            if len(samples) >= num_samples:
                break

        if len(samples) == 0:
            get_logger().warning("⚠️ No samples for Grad-CAM analysis")
            return {}

        samples = torch.cat(samples, dim=0)

        # 收集所有模型的CAM
        all_cams = []
        all_preds = []

        for worker in workers:
            for model_idx, model in enumerate(worker.models):
                model.eval()
                device = next(model.parameters()).device
                target_layer = get_target_layer(model, self.model_name)
                gradcam = GradCAM(model, target_layer)

                model_cams = []
                model_preds = []

                samples_device = samples.to(device)
                with torch.no_grad():
                    logits = model(samples_device)
                    preds = logits.argmax(dim=1).cpu().tolist()

                # 批量生成CAM
                cams = gradcam.generate_cam_batch(samples_device, preds)
                model_cams = list(cams)
                model_preds = preds

                gradcam.remove_hooks()

                all_cams.append(model_cams)
                all_preds.append(model_preds)

        # 计算指标
        metrics = self._compute_diversity_metrics(all_cams, all_preds, labels)

        return metrics

    def _compute_diversity_metrics(
        self,
        all_cams: List[List[np.ndarray]],
        all_preds: List[List[int]],
        labels: List[int],
    ) -> Dict[str, Any]:
        """计算多样性指标"""
        num_models = len(all_cams)
        num_samples = len(all_cams[0]) if all_cams else 0

        if num_samples == 0:
            return {}

        per_model_metrics = []

        for model_idx in range(num_models):
            model_cams = all_cams[model_idx]
            model_preds = all_preds[model_idx]

            # 计算CAM熵
            entropies = []
            for cam in model_cams:
                cam_flat = cam.flatten()
                cam_flat = cam_flat / (cam_flat.sum() + 1e-8)
                entropy = -np.sum(cam_flat * np.log(cam_flat + 1e-8))
                entropies.append(entropy)

            # 预测准确率
            correct = sum(
                1 for pred, label in zip(model_preds, labels) if pred == label
            )
            accuracy = correct / len(labels)

            per_model_metrics.append(
                {
                    "avg_cam_entropy": np.mean(entropies),
                    "accuracy": accuracy * 100,
                }
            )

        # 计算模型间相似度
        similarities = []
        overlaps = []

        for i in range(num_models):
            for j in range(i + 1, num_models):
                for s in range(num_samples):
                    cam_i = all_cams[i][s].flatten()
                    cam_j = all_cams[j][s].flatten()

                    # 余弦相似度
                    sim = np.dot(cam_i, cam_j) / (
                        np.linalg.norm(cam_i) * np.linalg.norm(cam_j) + 1e-8
                    )
                    similarities.append(sim)

                    # 重叠度 (IoU)
                    threshold = 0.5
                    mask_i = cam_i > threshold * cam_i.max()
                    mask_j = cam_j > threshold * cam_j.max()
                    intersection = np.logical_and(mask_i, mask_j).sum()
                    union = np.logical_or(mask_i, mask_j).sum()
                    iou = intersection / (union + 1e-8)
                    overlaps.append(iou)

        overall_metrics = {
            "avg_cam_entropy": np.mean(
                [m["avg_cam_entropy"] for m in per_model_metrics]
            ),
            "avg_cam_similarity": np.mean(similarities) if similarities else 0,
            "avg_cam_overlap": np.mean(overlaps) if overlaps else 0,
            "std_cam_entropy": np.std(
                [m["avg_cam_entropy"] for m in per_model_metrics]
            ),
        }

        return {
            "per_model": per_model_metrics,
            "overall": overall_metrics,
        }


class ModelListWrapper:
    """模型列表的简易包装器，用于兼容 GradCAMAnalyzer 的 workers 接口"""

    def __init__(self, models: List[nn.Module]):
        self.models = models


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
