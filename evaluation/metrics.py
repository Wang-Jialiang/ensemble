"""
================================================================================
指标计算器模块
================================================================================

包含: MetricsCalculator
"""

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
        all_features: torch.Tensor = None,
    ) -> dict:
        """核心指标调度中心 (大纲化)

        Args:
            all_logits: [num_models, num_samples, num_classes]
            targets: [num_samples]
            ensemble_fn: 集成策略函数
            all_features: [num_models, num_samples, feature_dim] 用于 CKA 计算
        """
        m = {}
        ens_logits = (ensemble_fn or ENSEMBLE_STRATEGIES["mean"])(all_logits)

        # 1. 基础性能与校准
        m.update(self._calc_base_performance(ens_logits, targets, all_logits))

        # 2. 模型多样性与核心不一致性
        m.update(self._calc_diversity(all_logits, all_features))

        # 3. 类别公平性与群体偏见 (最复杂段)
        m.update(self._calc_fairness(ens_logits.argmax(1), targets, all_logits))

        return m

    def _calc_base_performance(self, ens_logits, targets, all_logits):
        """计算集成/个体准确率与 NLL/ECE"""
        ens_preds = ens_logits.argmax(1)
        all_preds = all_logits.argmax(2)
        indiv_accs = (all_preds == targets).float().mean(1) * 100.0

        return {
            "ensemble_acc": 100.0 * (ens_preds == targets).float().mean().item(),
            "nll": F.cross_entropy(ens_logits, targets).item(),
            "ece": self.ece_metric(ens_logits, targets).item(),
            "avg_individual_acc": indiv_accs.mean().item(),
            "oracle_acc": 100.0 * (all_preds == targets).any(0).float().mean().item(),
        }

    def _calc_diversity(self, all_logits, all_features=None):
        """计算预测多样性: Disagreement + JS散度 + CKA

        Args:
            all_logits: [num_models, num_samples, num_classes]
            all_features: [num_models, num_samples, feature_dim] 用于 CKA，若为 None 则回退到 logits
        """
        num_m = all_logits.shape[0]
        all_preds = all_logits.argmax(2)
        all_probs = F.softmax(
            all_logits, dim=2
        )  # [num_models, num_samples, num_classes]

        # 1. Disagreement (硬不一致性 - 基于类别标签)
        pairs = [(i, j) for i in range(num_m) for j in range(i + 1, num_m)]
        dis_val = sum(
            (all_preds[p[0]] != all_preds[p[1]]).float().mean().item() for p in pairs
        )

        # 2. JS 散度 (软不一致性 - 基于概率分布)
        js_val = sum(
            self._js_divergence(all_probs[p[0]], all_probs[p[1]]) for p in pairs
        )

        # 3. CKA (表示层多样性)
        cka_input = all_features if all_features is not None else all_logits
        cka_m = compute_ensemble_cka(cka_input)

        return {
            "disagreement": 100.0 * (dis_val / len(pairs) if pairs else 0.0),
            "js_divergence": js_val / len(pairs) if pairs else 0.0,
            **cka_m,
        }

    def _js_divergence(
        self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
    ) -> float:
        """计算两个概率分布之间的 Jensen-Shannon 散度

        Args:
            p: [num_samples, num_classes] 概率分布
            q: [num_samples, num_classes] 概率分布
            eps: 数值稳定性常数

        Returns:
            平均 JS 散度 (范围: 0 到 ln(2) ≈ 0.693)
        """
        m = 0.5 * (p + q)
        kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
        kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)
        js = 0.5 * (kl_pm + kl_qm)
        return js.mean().item()

    def _calc_fairness(self, ens_preds, targets, all_logits):
        """计算公平性指标集（精简版）"""
        f = {}
        # 1. 逐类统计（内部使用，不输出）
        class_accs = self._get_per_class_stats(ens_preds, targets)

        # 2. 宏观公平性指标 (Balanced Acc, Gini)
        valid_accs = torch.tensor([a for a in class_accs if a >= 0])
        if len(valid_accs) > 0:
            f["balanced_acc"] = valid_accs.mean().item()
            f["acc_gini_coef"] = self._calc_gini(valid_accs)
            f["fairness_score"] = 100.0 - (valid_accs.max() - valid_accs.min()).item()

        # 3. 少数群体探测 (EOD, RER)
        f.update(self._calc_minority_bias(ens_preds, targets, all_logits, class_accs))
        return f

    def _get_per_class_stats(self, ens_preds, targets):
        """计算每个类别的准确率

        Returns:
            List[float]: 每个类别的准确率 (%)，不存在的类别返回 -1.0
        """
        class_accs = []
        for c in range(self.num_classes):
            mask = targets == c
            count = mask.sum().item()
            if count == 0:
                class_accs.append(-1.0)  # 标记不存在的类别
            else:
                correct = ((ens_preds == targets) & mask).sum().item()
                class_accs.append(100.0 * correct / count)
        return class_accs

    def _calc_gini(self, vals):
        """计算基尼系数"""
        if len(vals) <= 1 or vals.sum() == 0:
            return 0.0
        sorted_v = torch.sort(vals)[0]
        n = len(vals)
        indices = torch.arange(1, n + 1, dtype=torch.float32)
        return max(
            0.0,
            (
                (2.0 * (indices * sorted_v).sum() / (n * sorted_v.sum())) - (n + 1) / n
            ).item(),
        )

    def _calc_minority_bias(self, ens_preds, targets, all_logits, class_accs):
        """探测并量化针对少数类别的偏见（精简版）"""
        res = {}
        counts = torch.tensor(
            [(targets == c).sum().item() for c in range(self.num_classes)]
        )
        if (counts > 0).sum() < 2:
            return res

        # 1. 探测少数与多数群体
        valid_idx = torch.where(counts > 0)[0]
        mini_idx = valid_idx[counts[valid_idx].argmin()].item()
        majo_idx = valid_idx[counts[valid_idx].argmax()].item()

        # 2. 计算 EOD (Equalized Odds Difference)
        mini_rec = class_accs[mini_idx] / 100.0
        majo_rec = class_accs[majo_idx] / 100.0
        res["eod"] = abs(majo_rec - mini_rec) * 100.0

        # 3. Bottom-K 类别分析
        all_preds = all_logits.argmax(2)
        res.update(self._calc_bottom_k(all_preds, targets, class_accs))
        return res

    def _calc_bottom_k(self, all_preds, targets, class_accs):
        """计算单模型表现最差的 K 个类别的集成表现"""
        res = {}
        # 计算单模型在各类的平均准确率（用于识别 bottom-k 类别）
        sng_class_accs = torch.tensor(
            [
                (all_preds[:, targets == c] == c).float().mean().item() * 100.0
                if (targets == c).sum() > 0
                else 0.0
                for c in range(self.num_classes)
            ]
        )

        for k in [3, 5]:
            if k > self.num_classes:
                continue
            _, bottom_indices = torch.topk(sng_class_accs, k, largest=False)
            res[f"bottom_{k}_class_acc"] = (
                torch.tensor([class_accs[i] for i in bottom_indices]).mean().item()
            )

        return res
