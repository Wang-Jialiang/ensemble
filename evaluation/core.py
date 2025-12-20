"""
================================================================================
评估核心模块
================================================================================

集成策略、模型提取、logits 获取、MetricsCalculator、CKA 多样性、Checkpoint 加载
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models import ModelFactory
from ..utils import get_logger

if TYPE_CHECKING:
    from ..config import Config

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
# ║ CKA (Centered Kernel Alignment) 相似度计算                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
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


def compute_ensemble_cka(all_logits: torch.Tensor) -> Dict[str, float]:
    """计算集成模型中所有模型对的 CKA 相似度

    Args:
        all_logits: [num_models, num_samples, num_classes]

    Returns:
        包含 CKA 统计信息的字典:
        - avg_cka: 平均 CKA 相似度
        - min_cka: 最小 CKA 相似度
        - max_cka: 最大 CKA 相似度
        - cka_diversity: CKA 多样性 = 1 - avg_cka (越高表示越多样)
    """
    num_models = all_logits.shape[0]
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
            # 使用 logits 作为表示
            X = all_logits[i]  # [num_samples, num_classes]
            Y = all_logits[j]
            cka = linear_cka(X, Y)
            cka_values.append(cka)

    avg_cka = np.mean(cka_values)
    return {
        "avg_cka": avg_cka,
        "min_cka": np.min(cka_values),
        "max_cka": np.max(cka_values),
        "cka_diversity": 1.0 - avg_cka,  # 多样性 = 1 - 相似度
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型提取与推理                                                               ║
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
        from scipy.stats import spearmanr

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

            # 分歧度（churn）
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

            # CKA 多样性 (Centered Kernel Alignment)
            cka_metrics = compute_ensemble_cka(all_logits)
            metrics.update(cka_metrics)

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

            # 斯皮尔曼相关系数 (衡量预测排名一致性，越低表示多样性越高)
            spearman_sum = 0.0
            spearman_count = 0
            probs_np = probs.cpu().numpy()  # [num_models, num_samples, num_classes]
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    for s in range(probs_np.shape[1]):
                        corr, _ = spearmanr(probs_np[i, s], probs_np[j, s])
                        if not np.isnan(corr):
                            spearman_sum += corr
                            spearman_count += 1
            metrics["spearman_correlation"] = (
                spearman_sum / spearman_count if spearman_count > 0 else 1.0
            )

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

                # Bottom-k 类别准确率 (基于单模型平均表现找出最差类别)
                num_models = all_preds.shape[0]
                single_model_class_accs = []  # [num_models, num_classes]
                for m in range(num_models):
                    model_preds = all_preds[m]  # [num_samples]
                    model_class_acc = []
                    for c in range(self.num_classes):
                        mask = targets == c
                        count = mask.sum().item()
                        if count > 0:
                            acc = (
                                100.0
                                * ((model_preds == targets) & mask).sum().item()
                                / count
                            )
                        else:
                            acc = 0.0
                        model_class_acc.append(acc)
                    single_model_class_accs.append(model_class_acc)

                # 单模型平均类别准确率
                single_model_avg_class_acc = torch.tensor(single_model_class_accs).mean(
                    dim=0
                )  # [num_classes]

                # 找出单模型上表现最差的 k 个类别 (k=3)
                k_values = [3, 5] if self.num_classes >= 5 else [3]
                for k in k_values:
                    if k <= self.num_classes:
                        # 按单模型平均准确率排序，取最差的 k 个类别索引
                        _, bottom_k_indices = torch.topk(
                            single_model_avg_class_acc, k, largest=False
                        )
                        # 计算集成模型在这 k 个类别上的准确率
                        bottom_k_ensemble_accs = (
                            valid_accs[bottom_k_indices]
                            if len(valid_accs) == self.num_classes
                            else torch.tensor(
                                [per_class_acc[i] for i in bottom_k_indices.tolist()]
                            )
                        )
                        metrics[f"bottom_{k}_class_acc"] = (
                            bottom_k_ensemble_accs.mean().item()
                        )
                        # 同时记录单模型在这些类别上的平均准确率作为对比
                        metrics[f"single_model_bottom_{k}_class_acc"] = (
                            single_model_avg_class_acc[bottom_k_indices].mean().item()
                        )

                # 少数群体公平性指标
                class_counts = torch.tensor(per_class_count)
                valid_class_mask = class_counts > 0

                if valid_class_mask.sum() >= 2:  # 至少需要2个有效类别
                    valid_counts = class_counts[valid_class_mask]
                    valid_indices = torch.where(valid_class_mask)[0]

                    min_count_idx = valid_indices[valid_counts.argmin()].item()
                    max_count_idx = valid_indices[valid_counts.argmax()].item()

                    metrics["minority_class_idx"] = min_count_idx
                    metrics["majority_class_idx"] = max_count_idx
                    metrics["minority_class_count"] = per_class_count[min_count_idx]
                    metrics["majority_class_count"] = per_class_count[max_count_idx]

                    minority_recall_ensemble = per_class_acc[min_count_idx] / 100.0
                    majority_recall_ensemble = per_class_acc[max_count_idx] / 100.0

                    minority_recall_single = (
                        single_model_avg_class_acc[min_count_idx].item() / 100.0
                    )
                    majority_recall_single = (
                        single_model_avg_class_acc[max_count_idx].item() / 100.0
                    )

                    # RER (Relative Error Reduction)
                    minority_error_single = 1.0 - minority_recall_single
                    minority_error_ensemble = 1.0 - minority_recall_ensemble
                    majority_error_single = 1.0 - majority_recall_single
                    majority_error_ensemble = 1.0 - majority_recall_ensemble

                    if minority_error_single > 0:
                        metrics["minority_rer"] = (
                            (minority_error_single - minority_error_ensemble)
                            / minority_error_single
                        ) * 100.0
                    else:
                        metrics["minority_rer"] = 0.0

                    if majority_error_single > 0:
                        metrics["majority_rer"] = (
                            (majority_error_single - majority_error_ensemble)
                            / majority_error_single
                        ) * 100.0
                    else:
                        metrics["majority_rer"] = 0.0

                    metrics["rer_gap"] = (
                        metrics["minority_rer"] - metrics["majority_rer"]
                    )

                    # EOD (Equalized Odds Difference)
                    metrics["eod"] = (
                        abs(majority_recall_ensemble - minority_recall_ensemble) * 100.0
                    )
                    metrics["single_model_eod"] = (
                        abs(majority_recall_single - minority_recall_single) * 100.0
                    )
                    metrics["eod_improvement"] = (
                        metrics["single_model_eod"] - metrics["eod"]
                    )
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
# ║ Checkpoint 加载器                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointLoader:
    """从 checkpoint 加载模型进行评估

    完全独立于训练模块，只需 checkpoint 路径和配置即可加载模型。
    """

    @staticmethod
    def load(checkpoint_path: str, cfg: "Config") -> Dict[str, Any]:
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
