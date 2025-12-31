"""
================================================================================
Grad-CAM 分析模块
================================================================================

包含: GradCAM, GradCAMAnalyzer, ModelListWrapper
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Grad-CAM 目标层辅助函数                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
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
            target_layer: 用于生成CAM的目标层 (使用_get_target_layer获取)
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
                target_layer = _get_target_layer(model, self.model_name)
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
