"""
================================================================================
指标计算器模块
================================================================================

包含: MetricsCalculator
"""

import numpy as np
import torch
import torch.nn.functional as F

from .cka import compute_ensemble_cka
from .strategies import ENSEMBLE_STRATEGIES, EnsembleFn

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

            # ════════════════════════════════════════════════════════════════
            # 准确率与校准指标
            # ════════════════════════════════════════════════════════════════
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

            # ════════════════════════════════════════════════════════════════
            # 多样性指标 (分歧度、JS散度、CKA、斯皮尔曼相关)
            # ════════════════════════════════════════════════════════════════
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

            # ════════════════════════════════════════════════════════════════
            # 置信度指标
            # ════════════════════════════════════════════════════════════════
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

            # ════════════════════════════════════════════════════════════════
            # 公平性指标 (类别准确率、基尼系数、EOD)
            # ════════════════════════════════════════════════════════════════
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
