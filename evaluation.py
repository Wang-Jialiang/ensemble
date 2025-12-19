"""
================================================================================
评估模块
================================================================================
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .datasets import CorruptionDataset
from .models import ModelFactory
from .utils import ensure_dir, format_duration, get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 集成策略                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

EnsembleFn = Callable[[torch.Tensor], torch.Tensor]


def _voting_fn(all_logits: torch.Tensor) -> torch.Tensor:
    """多数投票 (将投票结果转换为 logits)"""
    preds = all_logits.argmax(dim=2)  # [N, Samples]
    num_classes = all_logits.shape[2]
    votes = torch.zeros(preds.shape[1], num_classes, device=all_logits.device)
    for i in range(preds.shape[0]):
        votes.scatter_add_(
            1,
            preds[i].unsqueeze(1),
            torch.ones_like(preds[i].unsqueeze(1), dtype=votes.dtype),
        )
    return votes


ENSEMBLE_STRATEGIES: Dict[str, EnsembleFn] = {
    "mean": lambda logits: logits.mean(dim=0),
    "voting": _voting_fn,
}


def get_ensemble_fn(cfg: "Config") -> EnsembleFn:
    """从配置获取集成函数"""
    strategy = getattr(cfg, "ensemble_strategy", "mean")
    return ENSEMBLE_STRATEGIES.get(strategy, ENSEMBLE_STRATEGIES["mean"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 指标计算器                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class MetricsCalculator:
    """集成模型指标计算器

    计算各种评估指标：
    - 准确率：集成/个体/Oracle/Top-5
    - 校准：ECE、NLL
    - 多样性：分歧度、多样性
    - 公平性：平衡准确率、基尼系数
    """

    def __init__(self, num_classes: int, ece_n_bins: int = 15):
        from torchmetrics.classification import CalibrationError

        self.num_classes = num_classes
        self.ece_metric = CalibrationError(
            task="multiclass", num_classes=num_classes, n_bins=ece_n_bins, norm="l1"
        )

    def calculate_all_metrics(
        self,
        all_logits: torch.Tensor,
        targets: torch.Tensor,
        ensemble_fn: EnsembleFn = None,
    ) -> dict:
        """计算所有指标

        Args:
            all_logits: [num_models, num_samples, num_classes]
            targets: [num_samples]
            ensemble_fn: 集成函数，默认为等权平均
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            metrics = {}
            if ensemble_fn is None:
                ensemble_fn = ENSEMBLE_STRATEGIES["mean"]
            ensemble_logits = ensemble_fn(all_logits)
            ensemble_preds = ensemble_logits.argmax(dim=1)

            # 准确率和校准
            metrics["ensemble_acc"] = (
                100.0 * (ensemble_preds == targets).float().mean().item()
            )
            metrics["nll"] = F.cross_entropy(ensemble_logits, targets).item()
            metrics["ece"] = self.ece_metric(ensemble_logits, targets).item()

            # 个体模型准确率
            all_preds = all_logits.argmax(dim=2)
            correct_per_model = all_preds == targets.unsqueeze(0)
            individual_accs = correct_per_model.float().mean(dim=1) * 100.0
            metrics["avg_individual_acc"] = individual_accs.mean().item()
            metrics["min_individual_acc"] = individual_accs.min().item()
            metrics["max_individual_acc"] = individual_accs.max().item()
            metrics["std_individual_acc"] = individual_accs.std().item()

            # Oracle准确率
            metrics["oracle_acc"] = (
                100.0 * correct_per_model.any(dim=0).float().mean().item()
            )

            # 分歧度
            num_models = all_preds.shape[0]
            disagreement_sum = sum(
                (all_preds[i] != all_preds[j]).float().mean().item()
                for i in range(num_models)
                for j in range(i + 1, num_models)
            )
            pair_count = num_models * (num_models - 1) // 2
            metrics["disagreement"] = 100.0 * (
                disagreement_sum / pair_count if pair_count > 0 else 0.0
            )

            # 多样性
            probs = F.softmax(all_logits, dim=2)
            metrics["diversity"] = (
                ((probs - probs.mean(dim=0, keepdim=True)) ** 2).mean().item()
            )

            # JS散度 (软不一致性)
            js_sum = 0.0
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    p = probs[i]  # [num_samples, num_classes]
                    q = probs[j]
                    m = (p + q) / 2
                    # KL(P||M) + KL(Q||M), 使用 log2 使结果在 [0, 1]
                    kl_pm = (p * (torch.log2(p + 1e-10) - torch.log2(m + 1e-10))).sum(
                        dim=1
                    )
                    kl_qm = (q * (torch.log2(q + 1e-10) - torch.log2(m + 1e-10))).sum(
                        dim=1
                    )
                    js = 0.5 * (kl_pm + kl_qm)
                    js_sum += js.mean().item()
            metrics["js_divergence"] = js_sum / pair_count if pair_count > 0 else 0.0

            # Top-5准确率
            if self.num_classes >= 5:
                top5 = ensemble_logits.topk(5, dim=1)[1]
                metrics["top5_acc"] = (
                    100.0
                    * (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()
                )

            # 置信度
            ensemble_probs = F.softmax(ensemble_logits, dim=1)
            max_probs = ensemble_probs.max(dim=1)[0]
            metrics["avg_confidence"] = max_probs.mean().item()
            metrics["avg_correct_confidence"] = (
                max_probs[ensemble_preds == targets].mean().item()
            )
            incorrect_mask = ensemble_preds != targets
            metrics["avg_incorrect_confidence"] = (
                max_probs[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0
            )

            # 公平性指标 (内联计算)
            per_class_acc = []
            per_class_count = []
            for c in range(self.num_classes):
                mask = targets == c
                count = mask.sum().item()
                if count > 0:
                    acc = (
                        100.0
                        * ((ensemble_preds == targets) & mask).sum().item()
                        / count
                    )
                else:
                    acc = 0.0
                per_class_acc.append(acc)
                per_class_count.append(count)
                metrics[f"class_{c}_acc"] = acc

            valid_mask = torch.tensor(per_class_count) > 0
            valid_accs = torch.tensor(per_class_acc)[valid_mask]

            if len(valid_accs) > 0:
                metrics["balanced_acc"] = valid_accs.mean().item()
                metrics["acc_disparity"] = (valid_accs.max() - valid_accs.min()).item()
                metrics["worst_class_acc"] = valid_accs.min().item()
                metrics["best_class_acc"] = valid_accs.max().item()
                metrics["per_class_acc_std"] = (
                    valid_accs.std().item() if len(valid_accs) > 1 else 0.0
                )
                # 基尼系数计算 (内联)
                if len(valid_accs) <= 1:
                    gini = 0.0
                else:
                    sorted_vals = torch.sort(valid_accs)[0]
                    n = len(sorted_vals)
                    total = sorted_vals.sum()
                    if total == 0:
                        gini = 0.0
                    else:
                        indices = torch.arange(1, n + 1, dtype=torch.float32)
                        gini = max(
                            0.0,
                            (
                                (2.0 * (indices * sorted_vals).sum() / (n * total))
                                - (n + 1.0) / n
                            ).item(),
                        )
                metrics["acc_gini_coef"] = gini
                metrics["fairness_score"] = max(0.0, 100.0 - metrics["acc_disparity"])
            else:
                metrics.update(
                    {
                        "balanced_acc": 0.0,
                        "acc_disparity": 0.0,
                        "worst_class_acc": 0.0,
                        "best_class_acc": 0.0,
                        "per_class_acc_std": 0.0,
                        "acc_gini_coef": 0.0,
                        "fairness_score": 100.0,
                    }
                )

            return metrics


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 评估函数
# ╚══════════════════════════════════════════════════════════════════════════════╝


def extract_models(trainer_or_models: Any) -> Tuple[List[nn.Module], torch.device]:
    """
    从 Trainer 或模型列表中提取模型和设备

    Args:
        trainer_or_models: StagedEnsembleTrainer 实例或 List[nn.Module]

    Returns:
        (models, device): 模型列表和计算设备
    """
    if hasattr(trainer_or_models, "workers"):  # 是 Trainer
        models = [
            model for worker in trainer_or_models.workers for model in worker.models
        ]
        device = trainer_or_models.workers[0].device
    else:  # 是模型列表
        models = trainer_or_models
        try:
            device = next(models[0].parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return models, device


def get_all_models_logits(
    models: List[nn.Module], loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取所有模型在数据集上的 logits

    Args:
        models: 模型列表 List[nn.Module]
        loader: 数据加载器
        device: 计算设备

    Returns:
        all_logits: (num_models, num_samples, num_classes)
        targets: (num_samples,)
    """
    from tqdm import tqdm

    all_logits_list = []
    all_targets_list = []

    iterator = tqdm(loader, desc="Evaluating Models", leave=False)

    with torch.no_grad():
        for inputs, targets in iterator:
            inputs = inputs.to(device)
            batch_logits = []

            for model in models:
                model.eval()
                logits = model(inputs)  # (batch_size, num_classes)
                batch_logits.append(logits.unsqueeze(0).cpu())

            # combined: (num_models, batch_size, num_classes)
            if batch_logits:
                combined = torch.cat(batch_logits, dim=0)
                all_logits_list.append(combined)
                all_targets_list.append(targets.cpu())

    if not all_logits_list:
        return torch.tensor([]), torch.tensor([])

    # 沿着 batch 维度 (dim=1) 拼接
    all_logits = torch.cat(all_logits_list, dim=1)
    all_targets = torch.cat(all_targets_list)

    return all_logits, all_targets


def evaluate_corruption(
    trainer_or_models: Any,
    corruption_dataset: CorruptionDataset,
    batch_size: int = 128,
    num_workers: int = 4,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    通用 Corruption 鲁棒性评估
    """
    logger = logger or get_logger()
    dataset_name = corruption_dataset.name
    logger.info(f"\n🧪 Running Corruption Evaluation on {dataset_name}...")

    models, device = extract_models(trainer_or_models)
    results = {}
    overall_avg = 0.0

    for severity in range(1, 6):
        logger.info(f"   Severity {severity}:")
        results[severity] = {}
        severity_accs = []

        for corruption in corruption_dataset.CORRUPTIONS:
            loader = corruption_dataset.get_loader(
                corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # 只需预测，不需要详细的 Calculator (只算 Acc)
            all_logits, targets = get_all_models_logits(models, loader, device)

            ensemble_logits = all_logits.mean(dim=0)
            ensemble_preds = ensemble_logits.argmax(dim=1)
            acc = 100.0 * (ensemble_preds == targets).float().mean().item()

            results[severity][corruption] = acc
            severity_accs.append(acc)

        avg_acc_sev = np.mean(severity_accs)
        logger.info(f"     -> Avg: {avg_acc_sev:.2f}%")
        overall_avg += avg_acc_sev

    overall_avg /= 5.0
    logger.info(f"\n   📈 Overall Avg: {overall_avg:.2f}%")

    results["severity_5_raw"] = results[5]
    results["overall_avg"] = overall_avg
    return results


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
            correct = sum(1 for p, l in zip(model_preds, labels) if p == l)
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
        """绘制两个模型之间的1D损失插值曲线

        Args:
            model1: 起始模型
            model2: 终止模型
            dataloader: 数据加载器
            device: 计算设备
            steps: 插值步数
            filename: 保存文件名
            label1: 模型1标签
            label2: 模型2标签

        Returns:
            loss_data: 损失值数组，长度为 steps
        """
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
        """绘制模型周围的2D损失地形等高线图

        Args:
            model: 目标模型
            dataloader: 数据加载器
            device: 计算设备
            distance: 采样距离（参数空间中的范围）
            steps: 每个方向的采样步数
            filename: 保存文件名
            model_name: 模型名称

        Returns:
            loss_data: 2D损失值数组，shape (steps, steps)
        """
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

        # 绘制 3D 表面图 (裸眼3D效果)
        fig_3d = plt.figure(figsize=(12, 9))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # 绘制表面
        surf = ax_3d.plot_surface(
            X, Y, loss_data, cmap="viridis", edgecolor="none", alpha=0.9
        )
        fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, label="Loss")

        # 标记模型位置
        center_loss = loss_data[steps // 2, steps // 2]
        ax_3d.scatter(
            [0], [0], [center_loss], c="red", s=200, marker="*", label=model_name
        )

        ax_3d.set_xlabel("Direction 1")
        ax_3d.set_ylabel("Direction 2")
        ax_3d.set_zlabel("Loss")
        ax_3d.set_title(f"3D Loss Landscape around {model_name}")
        ax_3d.view_init(elev=30, azim=45)  # 设置视角
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
        """绘制集成中所有模型对之间的1D损失插值曲线

        Args:
            models: 模型列表
            dataloader: 数据加载器
            device: 计算设备
            steps: 插值步数
            filename: 保存文件名

        Returns:
            results: {(i,j): loss_data} 字典
        """
        if not self._check_dependency():
            return {}

        import loss_landscapes
        import matplotlib.pyplot as plt

        n_models = len(models)
        if n_models < 2:
            self.logger.warning("⚠️ 需要至少 2 个模型来计算插值")
            return {}

        self.logger.info(
            f"📈 正在计算集成模型间的 Loss Landscape ({n_models} 个模型)..."
        )

        results = {}
        pairs = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]

        # 计算所有模型对的插值 (带进度条)
        from tqdm import tqdm

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
        """计算模型间的参数空间欧氏距离

        Args:
            models: 模型列表

        Returns:
            distance_matrix: 距离矩阵，shape (n_models, n_models)
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
            filename: 保存文件名

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
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()

        self.logger.info(f"📊 Saved: {filename}")
        return distance_matrix


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Checkpoint 加载器                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointLoader:
    """从 checkpoint 加载模型进行评估

    完全独立于训练模块，只需 checkpoint 路径和配置即可加载模型。
    """

    @staticmethod
    def load(checkpoint_path: str, cfg: Config) -> Dict[str, Any]:
        """
        加载 checkpoint 并返回可评估的模型上下文

        Args:
            checkpoint_path: checkpoint 目录路径
            cfg: 配置对象

        Returns:
            context: {
                'name': 实验名称,
                'models': List[nn.Module],
                'training_time': float,
                'config': dict
            }
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

        # 推断实验名称
        experiment_name = checkpoint_dir.parent.name

        # 读取训练状态
        state_path = checkpoint_dir / "trainer_state.pth"
        training_time = 0.0
        train_config = {}

        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            training_time = state.get("total_training_time", 0.0)
            train_config = {
                "augmentation_method": state.get("augmentation_method", "unknown"),
                "use_curriculum": state.get("use_curriculum", False),
            }

        # 加载所有模型
        models = []
        model_files = sorted(checkpoint_dir.glob(f"{experiment_name}_*.pth"))

        for model_file in model_files:
            model = ModelFactory.create_model(
                cfg.model_name, num_classes=cfg.num_classes
            )
            state = torch.load(model_file, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            models.append(model)

        if not models:
            raise RuntimeError(f"未找到模型文件: {checkpoint_dir}")

        get_logger().info(f"✅ 加载 {experiment_name}: {len(models)} 个模型")

        return {
            "name": experiment_name,
            "models": models,
            "training_time": training_time,
            "config": train_config,
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 报告可视化器 (matplotlib)                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ReportVisualizer:
    """生成可视化图表 (matplotlib)"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)

    def plot_accuracy_comparison(
        self, results: Dict[str, Dict], filename: str = "accuracy_comparison.png"
    ):
        """准确率对比柱状图"""
        import matplotlib.pyplot as plt

        names = list(results.keys())
        ensemble_accs = [
            r.get("standard_metrics", {}).get("ensemble_acc", 0)
            for r in results.values()
        ]
        oracle_accs = [
            r.get("standard_metrics", {}).get("oracle_acc", 0) for r in results.values()
        ]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x - width / 2, ensemble_accs, width, label="Ensemble Acc", color="#2ecc71"
        )
        ax.bar(x + width / 2, oracle_accs, width, label="Oracle Acc", color="#3498db")

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def plot_calibration_comparison(
        self, results: Dict[str, Dict], filename: str = "calibration.png"
    ):
        """校准指标对比"""
        import matplotlib.pyplot as plt

        names = list(results.keys())
        ece = [
            r.get("standard_metrics", {}).get("ece", 0) * 100 for r in results.values()
        ]
        nll = [r.get("standard_metrics", {}).get("nll", 0) for r in results.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(names, ece, color="#e74c3c")
        ax1.set_ylabel("ECE (%)")
        ax1.set_title("Expected Calibration Error (↓ better)")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(names, nll, color="#9b59b6")
        ax2.set_ylabel("NLL")
        ax2.set_title("Negative Log Likelihood (↓ better)")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def plot_diversity_comparison(
        self, results: Dict[str, Dict], filename: str = "diversity.png"
    ):
        """多样性指标对比"""
        import matplotlib.pyplot as plt

        names = list(results.keys())
        disagreement = [
            r.get("standard_metrics", {}).get("disagreement", 0)
            for r in results.values()
        ]
        js_divergence = [
            r.get("standard_metrics", {}).get("js_divergence", 0)
            for r in results.values()
        ]
        diversity = [
            r.get("standard_metrics", {}).get("diversity", 0) * 1000
            for r in results.values()
        ]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.bar(names, disagreement, color="#f39c12")
        ax1.set_ylabel("Disagreement (%)")
        ax1.set_title("Hard Disagreement (↑ more diverse)")
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(names, js_divergence, color="#e74c3c")
        ax2.set_ylabel("JS Divergence")
        ax2.set_title("Soft Disagreement (↑ more diverse)")
        ax2.tick_params(axis="x", rotation=45)

        ax3.bar(names, diversity, color="#1abc9c")
        ax3.set_ylabel("Diversity (×1000)")
        ax3.set_title("Prediction Diversity (↑ more diverse)")
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def plot_robustness_heatmap(
        self, results: Dict[str, Dict], filename: str = "robustness.png"
    ):
        """鲁棒性热力图"""
        import matplotlib.pyplot as plt

        # 收集 corruption 结果
        exp_names = list(results.keys())
        first_exp = list(results.values())[0]
        corruption_results = first_exp.get("corruption_results", {})

        if not corruption_results:
            get_logger().info("⚠️ No corruption results to plot")
            return

        corruption_types = list(corruption_results.keys())

        data = []
        for exp_name in exp_names:
            row = []
            for ctype in corruption_types:
                acc = (
                    results[exp_name]
                    .get("corruption_results", {})
                    .get(ctype, {})
                    .get("ensemble_acc", 0)
                )
                row.append(acc)
            data.append(row)

        data = np.array(data)

        fig, ax = plt.subplots(figsize=(12, max(4, len(exp_names))))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(corruption_types)))
        ax.set_yticks(np.arange(len(exp_names)))
        ax.set_xticklabels(corruption_types, rotation=45, ha="right")
        ax.set_yticklabels(exp_names)

        # 添加数值标注
        for i in range(len(exp_names)):
            for j in range(len(corruption_types)):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)

        ax.set_title("Robustness to Corruptions (Accuracy %)")
        plt.colorbar(im, ax=ax, label="Accuracy (%)")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def plot_fairness_radar(
        self, results: Dict[str, Dict], filename: str = "fairness.png"
    ):
        """公平性雷达图"""
        import matplotlib.pyplot as plt

        names = list(results.keys())
        metrics = ["balanced_acc", "fairness_score", "worst_class_acc"]
        labels = ["Balanced Acc", "Fairness Score", "Worst Class Acc"]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(labels))
        width = 0.8 / len(names)

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

        for i, (name, result) in enumerate(results.items()):
            std_metrics = result.get("standard_metrics", {})
            values = [std_metrics.get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=name, color=colors[i])

        ax.set_ylabel("Score")
        ax.set_title("Fairness Metrics Comparison")
        ax.set_xticks(x + width * (len(names) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def plot_training_time(
        self, results: Dict[str, Dict], filename: str = "training_time.png"
    ):
        """训练时间对比"""
        import matplotlib.pyplot as plt

        names = list(results.keys())
        times = [
            r.get("training_time_seconds", 0) / 60 for r in results.values()
        ]  # 转换为分钟

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, times, color="#34495e")

        ax.set_ylabel("Training Time (minutes)")
        ax.set_title("Training Time Comparison")
        ax.tick_params(axis="x", rotation=45)

        # 添加数值标签
        for bar, t in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.1f}m",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=150)
        plt.close()
        get_logger().info(f"📊 Saved: {filename}")

    def generate_all(self, results: Dict[str, Dict]):
        """生成所有可视化图表"""
        self.plot_accuracy_comparison(results)
        self.plot_calibration_comparison(results)
        self.plot_diversity_comparison(results)
        self.plot_fairness_radar(results)
        self.plot_training_time(results)

        # 如果有corruption结果，生成热力图
        if any(r.get("corruption_results") for r in results.values()):
            self.plot_robustness_heatmap(results)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 评估结果保存器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ResultsSaver:
    """评估结果保存器

    支持将评估指标保存为 JSON 和 CSV 格式。
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics"):
        """保存单个实验的指标"""
        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        csv_path = self.save_dir / f"{filename}.csv"
        with open(csv_path, "w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])

        get_logger().info(f"💾 Metrics saved to: {json_path}")

    def save_comparison(self, results: Dict[str, Dict], filename: str = "comparison"):
        """保存多个实验的对比结果"""
        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
                default=lambda x: x.item() if hasattr(x, "item") else x,
            )

        csv_path = self.save_dir / f"{filename}.csv"
        with open(csv_path, "w", newline="") as f:
            import csv

            if results:
                all_metrics = set()
                for exp_results in results.values():
                    all_metrics.update(exp_results.keys())
                all_metrics = sorted(all_metrics)

                writer = csv.writer(f)
                writer.writerow(["Experiment"] + list(all_metrics))

                for exp_name, exp_metrics in results.items():
                    row = [exp_name] + [exp_metrics.get(m, "") for m in all_metrics]
                    writer.writerow(row)

        get_logger().info(f"💾 Comparison saved to: {json_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 报告生成器 (评估 + 报告)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ReportGenerator:
    """实验评估与报告生成器

    使用方式:
        ReportGenerator.evaluate_and_report(
            trainers=[trainer1, trainer2],
            test_loader=test_loader,
            cfg=cfg,
            save_dir=cfg.save_dir,
            corruption_dataset=corruption_ds,  # 可选
            run_gradcam=True,                  # 可选
        )
    """

    @staticmethod
    def _get_rank_marker(
        value: float, all_values: List[float], higher_is_better: bool
    ) -> str:
        """获取排名标记 (仅多实验时显示)"""
        if len(all_values) <= 1:
            return ""
        sorted_values = sorted(all_values, reverse=higher_is_better)
        if value == sorted_values[0]:
            return " 🥇"
        elif value == sorted_values[1]:
            return " 🥈"
        return ""

    @staticmethod
    def _evaluate_models(
        models: List[nn.Module],
        exp_name: str,
        test_loader: DataLoader,
        cfg: Config,
        device: torch.device,
        training_time: float = 0.0,
        corruption_dataset: Optional[CorruptionDataset] = None,
        run_gradcam: bool = False,
    ) -> Dict[str, Any]:
        """通用模型评估方法 - 核心评估逻辑

        Args:
            models: 模型列表
            exp_name: 实验名称
            test_loader: 测试数据加载器
            cfg: 配置对象 (包含 ensemble_strategy)
            device: 计算设备
            training_time: 训练时间（秒）
            corruption_dataset: Corruption 数据集 (可选)
            run_gradcam: 是否运行 Grad-CAM 分析

        Returns:
            评估结果字典
        """
        get_logger().info(f"\n📊 Evaluating: {exp_name}")

        # 获取集成策略
        ensemble_fn = get_ensemble_fn(cfg)

        # 标准评估
        get_logger().info("   🔍 Standard evaluation...")
        all_logits, all_targets = get_all_models_logits(models, test_loader, device)
        metrics_calc = MetricsCalculator(cfg.num_classes, cfg.ece_n_bins)
        standard_metrics = metrics_calc.calculate_all_metrics(
            all_logits, all_targets, ensemble_fn=ensemble_fn
        )

        get_logger().info(f"   Ensemble Acc:   {standard_metrics['ensemble_acc']:.2f}%")
        get_logger().info(f"   ECE:            {standard_metrics['ece']:.4f}")

        # Corruption 评估
        corruption_results = None
        if corruption_dataset is not None:
            get_logger().info("   🔍 Corruption evaluation...")
            corruption_results = evaluate_corruption(
                models, corruption_dataset, batch_size=cfg.batch_size
            )

        # Grad-CAM 分析
        gradcam_metrics = None
        if run_gradcam:
            get_logger().info("   🔍 Grad-CAM analysis...")
            workers = [ModelListWrapper(models)]
            gradcam_analyzer = GradCAMAnalyzer(cfg)
            gradcam_metrics = gradcam_analyzer.analyze_ensemble_quality(
                workers, test_loader, num_samples=50, image_size=cfg.image_size
            )

        return {
            "experiment_name": exp_name,
            "training_time_seconds": training_time,
            "standard_metrics": standard_metrics,
            "corruption_results": corruption_results,
            "gradcam_metrics": gradcam_metrics,
        }

    @staticmethod
    def _evaluate_trainer(
        trainer: Any,
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional[CorruptionDataset] = None,
        run_gradcam: bool = False,
    ) -> Dict[str, Any]:
        """评估单个 trainer 并返回结果字典"""
        # 从 trainer 提取模型
        models, device = extract_models(trainer)
        return ReportGenerator._evaluate_models(
            models=models,
            exp_name=trainer.name,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            training_time=getattr(trainer, "total_training_time", 0.0),
            corruption_dataset=corruption_dataset,
            run_gradcam=run_gradcam,
        )

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成报告字符串"""
        lines = []
        log = lambda s="": lines.append(str(s))

        exp_names = list(results.keys())
        is_single = len(exp_names) == 1

        # 标题
        log("=" * 115)
        if is_single:
            log(f"📊 EXPERIMENT RESULTS: {exp_names[0]}")
        else:
            log("📊 EXPERIMENTAL RESULTS COMPARISON")
            log(
                "   🥇 = Best, 🥈 = Second Best | ↑ = Higher is better, ↓ = Lower is better"
            )
        log("=" * 115)

        # 表格
        log("\n🎯 Performance Metrics")
        log("-" * 115)
        log(
            f"{'Experiment':<25} | {'EnsAcc↑':<10} | {'AvgInd↑':<10} | {'Oracle↑':<10} | {'ECE↓':<10} | {'NLL↓':<10} | {'Time':<12}"
        )
        log("-" * 115)

        acc_vals = [
            results[n].get("standard_metrics", {}).get("ensemble_acc", 0)
            for n in exp_names
        ]

        for name in exp_names:
            m = results[name].get("standard_metrics", {})
            t = format_duration(
                results[name].get(
                    "training_time_seconds", results[name].get("training_time", 0)
                )
            )
            acc = m.get("ensemble_acc", 0)
            mark = cls._get_rank_marker(acc, acc_vals, True)
            log(
                f"{name:<25} | {acc:<7.2f}{mark:<3} | {m.get('avg_individual_acc', 0):<10.2f} | "
                f"{m.get('oracle_acc', 0):<10.2f} | {m.get('ece', 0):<10.4f} | {m.get('nll', 0):<10.4f} | {t:<12}"
            )
        log("-" * 115)

        # 详细指标 (每个实验依次展示)
        log("\n📋 Detailed Metrics")
        log("=" * 115)

        for name in exp_names:
            m = results[name].get("standard_metrics", {})
            log(f"\n🔹 {name}")
            log("-" * 40)

            # Diversity
            log("   🔀 Diversity & Confidence")
            log(
                f"      Disagreement: {m.get('disagreement', 0):.2f}%  |  JS散度: {m.get('js_divergence', 0):.4f}  |  Diversity: {m.get('diversity', 0):.6f}"
            )
            log(
                f"      Confidence: avg={m.get('avg_confidence', 0):.4f}, correct={m.get('avg_correct_confidence', 0):.4f}, incorrect={m.get('avg_incorrect_confidence', 0):.4f}"
            )

            # Fairness
            log("\n   ⚖️ Fairness")
            log(
                f"      Balanced Acc: {m.get('balanced_acc', 0):.2f}%  |  Disparity: {m.get('acc_disparity', 0):.2f}%  |  Score: {m.get('fairness_score', 0):.2f}"
            )
            log("-" * 40)

        # Corruption
        has_corruption = any(results[n].get("corruption_results") for n in exp_names)
        if has_corruption:
            log("\n🧪 Corruption Robustness")
            log("-" * 60)
            overall_vals = [
                results[n].get("corruption_results", {}).get("overall_avg", 0)
                for n in exp_names
                if results[n].get("corruption_results")
            ]
            for name in exp_names:
                c = results[name].get("corruption_results", {})
                if c and "overall_avg" in c:
                    val = c["overall_avg"]
                    mark = cls._get_rank_marker(val, overall_vals, True)
                    log(f"   {name:<25} | Overall: {val:.2f}%{mark}")
            log("-" * 60)

        log("\n" + "=" * 115)
        return "\n".join(lines)

    @classmethod
    def _save_and_print(cls, results: Dict[str, Dict], save_dir: str):
        """保存并打印报告"""
        saver = ResultsSaver(save_dir)
        report_content = cls._generate_report(results)

        # 保存结果 (统一格式)
        saver.save_comparison(results, "comprehensive_results")

        # 保存报告到文件 (不打印到控制台)
        report_path = Path(save_dir) / "detailed_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        get_logger().info(f"\n✅ Detailed report saved to: {report_path}")
        get_logger().info(f"✅ All results saved to: {save_dir}")

    @classmethod
    def evaluate_and_report(
        cls,
        trainers: List["StagedEnsembleTrainer"],
        test_loader: DataLoader,
        cfg: Config,
        save_dir: str,
        corruption_dataset: Optional[CorruptionDataset] = None,
        run_gradcam: bool = False,
    ):
        """评估多个 trainer 并生成报告 (一步完成)

        Args:
            trainers: trainer 列表
            test_loader: 测试数据加载器
            cfg: 配置对象 (包含 ensemble_strategy)
            save_dir: 报告保存目录
            corruption_dataset: Corruption 数据集 (可选)
            run_gradcam: 是否运行 Grad-CAM 分析
        """
        get_logger().info(
            f"\n{'=' * 80}\n📊 EVALUATION MODE | Models: {len(trainers)}\n{'=' * 80}"
        )

        # 评估所有 trainers
        results = {}
        for idx, trainer in enumerate(trainers, 1):
            get_logger().info(f"\n[{idx}/{len(trainers)}] {trainer.name}")
            result = cls._evaluate_trainer(
                trainer, test_loader, cfg, corruption_dataset, run_gradcam
            )
            results[trainer.name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(save_dir)
        visualizer.generate_all(results)

        # 生成并保存报告
        cls._save_and_print(results, save_dir)

    @classmethod
    def generate_from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        test_loader: DataLoader,
        cfg: Config,
        output_dir: str,
        corruption_dataset: Optional[CorruptionDataset] = None,
        run_gradcam: bool = False,
        run_loss_landscape: bool = False,
    ):
        """
        从 checkpoint 直接评估并生成完整可视化报告

        这是 evaluation 模块的主入口，完全独立于 training 模块。

        Args:
            checkpoint_paths: checkpoint 目录路径列表
            test_loader: 测试数据加载器
            cfg: 配置对象 (包含 ensemble_strategy)
            output_dir: 输出目录
            corruption_dataset: Corruption 数据集 (可选)
            run_gradcam: 是否运行 Grad-CAM 分析
            run_loss_landscape: 是否运行 Loss Landscape 分析

        输出:
            output_dir/
            ├── detailed_report.txt      # 文本报告
            ├── accuracy_comparison.png  # 准确率对比
            ├── calibration.png          # 校准指标
            ├── diversity.png            # 多样性指标
            ├── fairness.png             # 公平性指标
            ├── training_time.png        # 训练时间
            ├── robustness.png           # 鲁棒性热力图 (如有)
            ├── model_distances.png      # 模型参数距离 (如有)
            ├── ensemble_loss_landscape.png  # Loss Landscape (如有)
            └── final_metrics.json       # 指标数据
        """
        get_logger().info(f"\n{'=' * 80}")
        get_logger().info(
            f"📊 EVALUATION FROM CHECKPOINTS | Count: {len(checkpoint_paths)}"
        )
        get_logger().info(f"{'=' * 80}")

        ensure_dir(output_dir)
        results = {}
        all_models = {}  # 收集所有实验的模型用于 Loss Landscape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for idx, ckpt_path in enumerate(checkpoint_paths, 1):
            get_logger().info(f"\n[{idx}/{len(checkpoint_paths)}] Loading: {ckpt_path}")

            # 加载模型
            ctx = CheckpointLoader.load(ckpt_path, cfg)
            exp_name = ctx["name"]
            models = [m.to(device) for m in ctx["models"]]
            all_models[exp_name] = models  # 保存用于后续分析

            # 使用通用评估方法
            result = cls._evaluate_models(
                models=models,
                exp_name=exp_name,
                test_loader=test_loader,
                cfg=cfg,
                device=device,
                training_time=ctx["training_time"],
                corruption_dataset=corruption_dataset,
                run_gradcam=run_gradcam,
            )
            result["train_config"] = ctx["config"]
            results[exp_name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(output_dir)
        visualizer.generate_all(results)

        # Loss Landscape 分析
        if run_loss_landscape and all_models:
            get_logger().info("\n🏔️ Generating Loss Landscape visualizations...")
            landscape_viz = LossLandscapeVisualizer(output_dir)

            for exp_name, models in all_models.items():
                # 模型参数距离热力图 (无需 loss-landscapes 依赖)
                landscape_viz.plot_model_distance_heatmap(
                    models, filename=f"{exp_name}_model_distances.png"
                )

                # Loss Landscape 插值 (需要 loss-landscapes)
                landscape_viz.plot_ensemble_interpolations(
                    models,
                    test_loader,
                    device,
                    filename=f"{exp_name}_loss_landscape.png",
                )

                # 2D/3D 表面图 - 为第一个模型生成 (计算量较大)
                if len(models) > 0:
                    landscape_viz.plot_2d_plane(
                        models[0],
                        test_loader,
                        device,
                        distance=1.0,
                        steps=20,  # 减少步数以加快计算
                        filename=f"{exp_name}_landscape_surface.png",
                        model_name=f"{exp_name}_M1",
                    )

        # 生成并保存文本报告
        cls._save_and_print(results, output_dir)

        get_logger().info(f"\n✅ Complete! All reports saved to: {output_dir}")
        return results
