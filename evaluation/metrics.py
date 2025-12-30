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

    def calculate_all_metrics(self, all_logits: torch.Tensor, targets: torch.Tensor, ensemble_fn: EnsembleFn = None) -> dict:
        """核心指标调度中心 (大纲化)"""
        m = {}
        ens_logits = (ensemble_fn or ENSEMBLE_STRATEGIES["mean"])(all_logits)
        probs = F.softmax(all_logits, dim=2)
        ens_probs = F.softmax(ens_logits, dim=1)

        # 1. 基础性能与校准
        m.update(self._calc_base_performance(ens_logits, targets, all_logits))
        
        # 2. 模型多样性与核心不一致性
        m.update(self._calc_diversity(all_logits, probs))
        
        # 3. 置信度分布特征
        m.update(self._calc_confidence(ens_probs, ens_logits.argmax(1), targets))
        
        # 4. 类别公平性与群体偏见 (最复杂段)
        m.update(self._calc_fairness(ens_logits.argmax(1), targets, all_logits))
        
        return m

    def _calc_base_performance(self, ens_logits, targets, all_logits):
        """计算集成/个体准确率与 NLL/ECE"""
        ens_preds = ens_logits.argmax(1)
        all_preds = all_logits.argmax(2)
        indiv_accs = (all_preds == targets).float().mean(1) * 100.
        
        return {
            "ensemble_acc": 100. * (ens_preds == targets).float().mean().item(),
            "nll": F.cross_entropy(ens_logits, targets).item(),
            "ece": self.ece_metric(ens_logits, targets).item(),
            "avg_individual_acc": indiv_accs.mean().item(),
            "oracle_acc": 100. * (all_preds == targets).any(0).float().mean().item()
        }

    def _calc_diversity(self, all_logits, probs):
        """计算预测多样性: Disagreement, JS, CKA, Spearman"""
        from scipy.stats import spearmanr
        num_m = all_logits.shape[0]
        all_preds = all_logits.argmax(2)
        
        # 1. Disagreement (硬不一致性)
        pairs = [(i, j) for i in range(num_m) for j in range(i + 1, num_m)]
        dis_val = sum((all_preds[p[0]] != all_preds[p[1]]).float().mean().item() for p in pairs)
        
        # 2. JS Divergence (软不一致性)
        js_val = self._calc_js_divergence(probs, pairs)
        
        # 3. Spearman & CKA
        corr_val = self._calc_spearman(probs.cpu().numpy(), pairs)
        cka_m = compute_ensemble_cka(all_logits)
        
        return {
            "disagreement": 100. * (dis_val / len(pairs) if pairs else 0.),
            "diversity": ((probs - probs.mean(0, keepdim=True))**2).mean().item(),
            "js_divergence": js_val / len(pairs) if pairs else 0.,
            "spearman_correlation": corr_val,
            **cka_m
        }

    def _calc_js_divergence(self, probs, pairs):
        """计算 JS 散度均值"""
        total_js = 0.0
        for i, j in pairs:
            p, q = probs[i], probs[j]
            m = (p + q) / 2
            kl_pm = (p * (torch.log2(p + 1e-10) - torch.log2(m + 1e-10))).sum(1)
            kl_qm = (q * (torch.log2(q + 1e-10) - torch.log2(m + 1e-10))).sum(1)
            total_js += 0.5 * (kl_pm + kl_qm).mean().item()
        return total_js

    def _calc_spearman(self, probs_np, pairs):
        """计算平均 Spearman 秩相关系数"""
        from scipy.stats import spearmanr
        total_corr, n = 0.0, 0
        for i, j in pairs:
            for s in range(probs_np.shape[1]):
                c, _ = spearmanr(probs_np[i, s], probs_np[j, s])
                if not np.isnan(c): total_corr += c; n += 1
        return total_corr / n if n > 0 else 1.0

    def _calc_confidence(self, ens_probs, ens_preds, targets):
        """计算结果置信度分布"""
        res = {}
        max_p = ens_probs.max(1)[0]
        res["avg_confidence"] = max_p.mean().item()
        res["avg_correct_conf"] = max_p[ens_preds == targets].mean().item()
        
        wrong_mask = ens_preds != targets
        res["avg_incorrect_conf"] = max_p[wrong_mask].mean().item() if wrong_mask.any() else 0.0
        return res

    def _calc_fairness(self, ens_preds, targets, all_logits):
        """计算这种复杂的公平性指标集"""
        f = {}
        # 1. 逐类统计
        class_accs = self._get_per_class_stats(ens_preds, targets)
        f.update({f"class_{i}_acc": v for i, v in enumerate(class_accs)})
        
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
        return [100.0 * ((ens_preds == targets) & (targets == c)).sum().item() / max(1, (targets == c).sum().item())
                for c in range(self.num_classes)]

    def _calc_gini(self, vals):
        """计算基尼系数"""
        if len(vals) <= 1 or vals.sum() == 0: return 0.0
        sorted_v = torch.sort(vals)[0]
        n = len(vals)
        indices = torch.arange(1, n + 1, dtype=torch.float32)
        return max(0.0, ((2.0 * (indices * sorted_v).sum() / (n * sorted_v.sum())) - (n+1)/n).item())

    def _calc_minority_bias(self, ens_preds, targets, all_logits, class_accs):
        """探测并量化针对少数类别的偏见 (完整实现)"""
        res = {}
        counts = torch.tensor([(targets == c).sum().item() for c in range(self.num_classes)])
        if (counts > 0).sum() < 2: return res
        
        # 1. 探测少数与多数群体
        valid_idx = torch.where(counts > 0)[0]
        mini_idx, majo_idx = valid_idx[counts[valid_idx].argmin()].item(), valid_idx[counts[valid_idx].argmax()].item()
        
        # 2. 计算 Recall 与 RER
        all_preds = all_logits.argmax(2)
        # single_accs = (all_preds == targets).float().mean(0) # This line was not used in the original snippet, and is not needed for the new logic.
        
        def get_group_stats(idx):
            mask = targets == idx
            ens_rec = class_accs[idx] / 100.0
            # Calculate single model average recall for this class
            sng_rec = (all_preds[:, mask] == idx).float().mean().item() if mask.sum() > 0 else 0.0
            err_s, err_e = 1.0 - sng_rec, 1.0 - ens_rec
            rer = ((err_s - err_e) / max(1e-6, err_s)) * 100.0
            return ens_rec, sng_rec, rer

        mini_stats = get_group_stats(mini_idx)
        majo_stats = get_group_stats(majo_idx)
        
        res.update({
            "minority_rer": mini_stats[2], "majority_rer": majo_stats[2],
            "eod": abs(majo_stats[0] - mini_stats[0]) * 100.0,
            "rer_gap": mini_stats[2] - majo_stats[2]
        })

        # 3. Bottom-K 类别分析
        res.update(self._calc_bottom_k(all_preds, targets, class_accs))
        return res

    def _calc_bottom_k(self, all_preds, targets, class_accs):
        """计算单模型表现最差的 K 个类别的集成表现"""
        res = {}
        # 计算单模型在各类的平均准确率
        sng_class_accs = torch.tensor([
            (all_preds[:, targets == c] == c).float().mean().item() * 100.0 if (targets == c).sum() > 0 else 0.0
            for c in range(self.num_classes)
        ])
        
        for k in [3, 5]:
            if k > self.num_classes: continue
            _, bottom_indices = torch.topk(sng_class_accs, k, largest=False)
            res[f"bottom_{k}_class_acc"] = torch.tensor([class_accs[i] for i in bottom_indices]).mean().item()
            res[f"single_model_bottom_{k}_class_acc"] = sng_class_accs[bottom_indices].mean().item()
        
        return res
```
