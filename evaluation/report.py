"""
================================================================================
报告生成器模块
================================================================================

包含: ReportGenerator - 实验评估与报告生成
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..datasets.robustness.corruption import CorruptionDataset
    from ..datasets.robustness.ood import OODDataset

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..utils import ensure_dir, get_logger
from .adversarial import evaluate_adversarial
from .checkpoint import CheckpointLoader
from .corruption_robustness import evaluate_corruption
from .gradcam import GradCAMAnalyzer, ModelListWrapper
from .inference import get_all_models_logits
from .landscape import ModelDistanceCalculator
from .metrics import MetricsCalculator
from .ood import evaluate_ood
from .saver import ResultsSaver
from .strategies import get_ensemble_fn

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 报告生成器 (评估 + 报告)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ReportGenerator:
    """实验评估与报告生成器

    主入口:
        ReportGenerator.evaluate_checkpoints(checkpoint_paths=[...], ...)
    """

    @staticmethod
    def _evaluate_models(
        models, exp_name, test_loader, cfg, device, **datasets
    ) -> Dict[str, Any]:
        """通用模型评估方法 - 生命周期钩子模式"""
        log = get_logger()
        log.info(f"\n┌{'─' * 60}")
        log.info(f"│ 📊 {exp_name}")
        log.info(f"└{'─' * 60}")
        res = {"experiment_name": exp_name}

        # 1. 标准标准指标 (Acc, ECE, NLL)
        res["standard_metrics"] = ReportGenerator._run_standard_eval(
            models, test_loader, cfg, device
        )

        # 2. 鲁棒性套件 (Corruption, OOD, Domain)
        res.update(
            ReportGenerator._run_robustness_eval(models, cfg, test_loader, **datasets)
        )

        # 3. 对抗性与可解释性分析
        res.update(
            ReportGenerator._run_analysis_eval(models, cfg, test_loader, **datasets)
        )

        return res

    @staticmethod
    def _run_standard_eval(models, loader, cfg, device):
        get_logger().info("  ├─ 🔍 Standard metrics")
        all_l, all_t = get_all_models_logits(models, loader, device)
        return MetricsCalculator(cfg.num_classes, cfg.ece_n_bins).calculate_all_metrics(
            all_l, all_t, get_ensemble_fn(cfg)
        )

    @staticmethod
    def _run_robustness_eval(models, cfg, loader, **ds):
        r = {"corruption_results": None, "ood_results": None}

        if ds.get("corruption_dataset"):
            get_logger().info("  ├─ 🌪️ Corruption robustness")
            r["corruption_results"] = evaluate_corruption(
                models, ds["corruption_dataset"], cfg
            )

        if ds.get("ood_dataset"):
            get_logger().info("  ├─ 🔮 OOD detection")
            r["ood_results"] = evaluate_ood(
                models,
                loader,
                ds["ood_dataset"].get_loader(cfg),
                ds["ood_dataset"].name,
            )

        return r

    @staticmethod
    def _run_analysis_eval(models, cfg, loader, **ds):
        a = {"adversarial_results": None, "gradcam_metrics": None}
        if ds.get("run_adversarial", True):
            get_logger().info("  ├─ ⚔️ Adversarial robustness")
            a["adversarial_results"] = evaluate_adversarial(
                models, loader, cfg=cfg, logger=get_logger()
            )

        if ds.get("run_gradcam", False):
            get_logger().info("  └─ 🔍 Grad-CAM analysis")
            a["gradcam_metrics"] = GradCAMAnalyzer(cfg).analyze_ensemble_quality(
                [ModelListWrapper(models)],
                loader,
                cfg.gradcam_num_samples,
                cfg.image_size,
            )
        return a

    @staticmethod
    def _get_rank_marker(
        value: float, all_values: List[float], higher_is_better: bool
    ) -> str:
        """获取排名标记 🥇🥈🥉"""
        if len(all_values) <= 1:
            return ""
        sorted_values = sorted(all_values, reverse=higher_is_better)
        if value == sorted_values[0]:
            return "🥇"
        elif len(sorted_values) > 1 and value == sorted_values[1]:
            return "🥈"
        elif len(sorted_values) > 2 and value == sorted_values[2]:
            return "🥉"
        return ""

    @classmethod
    def _format_val(
        cls,
        value: float,
        all_vals: List[float],
        higher_is_better: bool,
        fmt: str = ".4f",
    ) -> str:
        """格式化数值并添加排名标记"""
        mark = cls._get_rank_marker(value, all_vals, higher_is_better)
        return f"{value:{fmt}}{mark}"

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成文本报告 - 按9个维度组织，带排名标记和箭头指示"""
        lines = []
        exps = list(results.keys())

        # 辅助函数：获取指标值列表
        def get_std_vals(key):
            return [results[n].get("standard_metrics", {}).get(key, 0) for n in exps]

        def get_corr_vals(key):
            return [
                results[n].get("corruption_results", {}).get(key, 0)
                if results[n].get("corruption_results")
                else 0
                for n in exps
            ]

        def get_ood_vals(key):
            return [
                results[n].get("ood_results", {}).get(key, 0)
                if results[n].get("ood_results")
                else 0
                for n in exps
            ]

        def get_adv_vals(key):
            return [
                results[n].get("adversarial_results", {}).get(key, 0)
                if results[n].get("adversarial_results")
                else 0
                for n in exps
            ]

        def get_cam_vals(key):
            return [
                results[n].get("gradcam_metrics", {}).get(key, 0)
                if results[n].get("gradcam_metrics")
                else 0
                for n in exps
            ]

        def get_dist_vals(key):
            return [results[n].get(key, 0) for n in exps]

        lines.append("=" * 120)
        lines.append("📋 ENSEMBLE EVALUATION REPORT")
        lines.append("=" * 120)

        # 1. 准确率 (↑ 高好)
        lines.append("\n🎯 准确率 (Accuracy)")
        lines.append("-" * 100)
        lines.append(
            f"{'Experiment':<25} | {'ensemble_acc↑':<16} | {'avg_ind_acc↑':<16} | {'oracle_acc↑':<16}"
        )
        lines.append("-" * 100)
        ens_vals = get_std_vals("ensemble_acc")
        avg_vals = get_std_vals("avg_individual_acc")
        ora_vals = get_std_vals("oracle_acc")
        for n in exps:
            m = results[n].get("standard_metrics", {})
            lines.append(
                f"{n:<25} | {cls._format_val(m.get('ensemble_acc', 0), ens_vals, True):<16} | "
                f"{cls._format_val(m.get('avg_individual_acc', 0), avg_vals, True):<16} | "
                f"{cls._format_val(m.get('oracle_acc', 0), ora_vals, True):<16}"
            )

        # 2. 校准性 (↓ 低好)
        lines.append("\n📏 校准性 (Calibration)")
        lines.append("-" * 70)
        lines.append(f"{'Experiment':<25} | {'ece↓':<16} | {'nll↓':<16}")
        lines.append("-" * 70)
        ece_vals = get_std_vals("ece")
        nll_vals = get_std_vals("nll")
        for n in exps:
            m = results[n].get("standard_metrics", {})
            lines.append(
                f"{n:<25} | {cls._format_val(m.get('ece', 0), ece_vals, False, '.6f'):<16} | "
                f"{cls._format_val(m.get('nll', 0), nll_vals, False, '.6f'):<16}"
            )

        # 3. 多样性 (disagreement↑高好, js_div↑高好, avg_cka↓低好表示更多样)
        lines.append("\n🔀 多样性 (Diversity)")
        lines.append("-" * 90)
        lines.append(
            f"{'Experiment':<25} | {'disagreement↑':<16} | {'js_divergence↑':<16} | {'avg_cka↓':<16}"
        )
        lines.append("-" * 90)
        dis_vals = get_std_vals("disagreement")
        js_vals = get_std_vals("js_divergence")
        cka_vals = get_std_vals("avg_cka")
        for n in exps:
            m = results[n].get("standard_metrics", {})
            lines.append(
                f"{n:<25} | {cls._format_val(m.get('disagreement', 0), dis_vals, True):<16} | "
                f"{cls._format_val(m.get('js_divergence', 0), js_vals, True):<16} | "
                f"{cls._format_val(m.get('avg_cka', 0), cka_vals, False):<16}"
            )

        # 4. 公平性
        lines.append("\n⚖️ 公平性 (Fairness)")
        lines.append("-" * 140)
        lines.append(
            f"{'Experiment':<25} | {'balanced_acc↑':<14} | {'gini_coef↓':<14} | "
            f"{'fair_score↑':<14} | {'eod↓':<12} | {'bottom_3↑':<12} | {'bottom_5↑':<12}"
        )
        lines.append("-" * 140)
        bal_vals = get_std_vals("balanced_acc")
        gini_vals = get_std_vals("acc_gini_coef")
        fair_vals = get_std_vals("fairness_score")
        eod_vals = get_std_vals("eod")
        b3_vals = get_std_vals("bottom_3_class_acc")
        b5_vals = get_std_vals("bottom_5_class_acc")
        for n in exps:
            m = results[n].get("standard_metrics", {})
            lines.append(
                f"{n:<25} | {cls._format_val(m.get('balanced_acc', 0), bal_vals, True):<14} | "
                f"{cls._format_val(m.get('acc_gini_coef', 0), gini_vals, False):<14} | "
                f"{cls._format_val(m.get('fairness_score', 0), fair_vals, True):<14} | "
                f"{cls._format_val(m.get('eod', 0), eod_vals, False):<12} | "
                f"{cls._format_val(m.get('bottom_3_class_acc', 0), b3_vals, True):<12} | "
                f"{cls._format_val(m.get('bottom_5_class_acc', 0), b5_vals, True):<12}"
            )

        # 5. Corruption 鲁棒性 (↑ 高好)
        has_corr = any(results[n].get("corruption_results") for n in exps)
        if has_corr:
            lines.append("\n🌪️ Corruption鲁棒性 (Corruption Robustness)")
            lines.append("-" * 80)
            lines.append(f"{'Experiment':<25} | {'overall_avg↑':<16}")
            lines.append("-" * 50)
            overall_vals = get_corr_vals("overall_avg")
            for n in exps:
                corr = results[n].get("corruption_results") or {}
                lines.append(
                    f"{n:<25} | {cls._format_val(corr.get('overall_avg', 0), overall_vals, True):<16}"
                )

            # by_severity
            first_corr = next(
                (
                    results[n].get("corruption_results")
                    for n in exps
                    if results[n].get("corruption_results")
                ),
                {},
            )
            severities = sorted(
                list((first_corr.get("by_severity") or {}).keys()), key=lambda x: int(x)
            )
            if severities:
                lines.append(
                    f"\nBy Severity: {' | '.join([f'Sev_{s}↑' for s in severities])}"
                )
                for n in exps:
                    corr = results[n].get("corruption_results") or {}
                    by_sev = corr.get("by_severity") or {}
                    lines.append(
                        f"{n:<25} | "
                        + " | ".join([f"{by_sev.get(s, 0):.4f}" for s in severities])
                    )

            # by_category
            categories = list((first_corr.get("by_category") or {}).keys())
            if categories:
                lines.append(
                    f"\nBy Category: {' | '.join([f'{c}↑' for c in categories])}"
                )
                for n in exps:
                    corr = results[n].get("corruption_results") or {}
                    by_cat = corr.get("by_category") or {}
                    lines.append(
                        f"{n:<25} | "
                        + " | ".join([f"{by_cat.get(c, 0):.4f}" for c in categories])
                    )

        # 6. OOD 鲁棒性 (AUROC↑高好, FPR95↓低好)
        has_ood = any(results[n].get("ood_results") for n in exps)
        if has_ood:
            lines.append("\n🔮 OOD鲁棒性 (OOD Robustness)")
            lines.append("-" * 120)
            lines.append(
                f"{'Experiment':<25} | {'auroc_msp↑':<16} | {'auroc_entropy↑':<18} | "
                f"{'fpr95_msp↓':<16} | {'fpr95_entropy↓':<18}"
            )
            lines.append("-" * 120)
            auroc_msp = get_ood_vals("ood_auroc_msp")
            auroc_ent = get_ood_vals("ood_auroc_entropy")
            fpr_msp = get_ood_vals("ood_fpr95_msp")
            fpr_ent = get_ood_vals("ood_fpr95_entropy")
            for n in exps:
                ood = results[n].get("ood_results") or {}
                lines.append(
                    f"{n:<25} | {cls._format_val(ood.get('ood_auroc_msp', 0), auroc_msp, True):<16} | "
                    f"{cls._format_val(ood.get('ood_auroc_entropy', 0), auroc_ent, True):<18} | "
                    f"{cls._format_val(ood.get('ood_fpr95_msp', 0), fpr_msp, False):<16} | "
                    f"{cls._format_val(ood.get('ood_fpr95_entropy', 0), fpr_ent, False):<18}"
                )

        # 7. Adversarial 鲁棒性 (↑ 高好)
        has_adv = any(results[n].get("adversarial_results") for n in exps)
        if has_adv:
            lines.append("\n⚔️ Adversarial鲁棒性 (Adversarial Robustness)")
            lines.append("-" * 70)
            lines.append(f"{'Experiment':<25} | {'fgsm_acc↑':<16} | {'pgd_acc↑':<16}")
            lines.append("-" * 70)
            fgsm_vals = get_adv_vals("fgsm_acc")
            pgd_vals = get_adv_vals("pgd_acc")
            for n in exps:
                adv = results[n].get("adversarial_results") or {}
                lines.append(
                    f"{n:<25} | {cls._format_val(adv.get('fgsm_acc', 0), fgsm_vals, True):<16} | "
                    f"{cls._format_val(adv.get('pgd_acc', 0), pgd_vals, True):<16}"
                )

        # 8. GradCAM 多样性 (entropy↑高好, similarity/overlap↓低好)
        has_cam = any(results[n].get("gradcam_metrics") for n in exps)
        if has_cam:
            lines.append("\n🔍 GradCAM多样性 (GradCAM Diversity)")
            lines.append("-" * 100)
            lines.append(
                f"{'Experiment':<25} | {'cam_entropy↑':<16} | {'cam_similarity↓':<18} | {'cam_overlap↓':<16}"
            )
            lines.append("-" * 100)
            ent_vals = get_cam_vals("avg_cam_entropy")
            sim_vals = get_cam_vals("avg_cam_similarity")
            ovl_vals = get_cam_vals("avg_cam_overlap")
            for n in exps:
                g = results[n].get("gradcam_metrics") or {}
                lines.append(
                    f"{n:<25} | {cls._format_val(g.get('avg_cam_entropy', 0), ent_vals, True):<16} | "
                    f"{cls._format_val(g.get('avg_cam_similarity', 0), sim_vals, False):<18} | "
                    f"{cls._format_val(g.get('avg_cam_overlap', 0), ovl_vals, False):<16}"
                )

        # 9. 参数空间多样性 (avg_distance↑高好表示更多样, direction_div↑高好, std↑高好)
        has_dist = any(results[n].get("distance_matrix") is not None for n in exps)
        if has_dist:
            lines.append("\n📐 参数空间多样性 (Parameter Space Diversity)")
            lines.append("-" * 100)
            lines.append(
                f"{'Experiment':<25} | {'avg_distance↑':<16} | {'direction_div↑':<18} | {'std_distance↑':<16}"
            )
            lines.append("-" * 100)
            avg_d = get_dist_vals("avg_distance")
            dir_d = get_dist_vals("direction_diversity")
            std_d = get_dist_vals("std_distance")
            for n in exps:
                r = results[n]
                lines.append(
                    f"{n:<25} | {cls._format_val(r.get('avg_distance', 0), avg_d, True):<16} | "
                    f"{cls._format_val(r.get('direction_diversity', 0), dir_d, True):<18} | "
                    f"{cls._format_val(r.get('std_distance', 0), std_d, True):<16}"
                )

        lines.append("\n" + "=" * 120)
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
    def evaluate_checkpoints(
        cls,
        checkpoint_paths: List[str],
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        ood_dataset: Optional["OODDataset"] = None,
        run_gradcam: bool = False,
        run_loss_landscape: bool = False,
        run_adversarial: bool = True,
    ):
        """
        从磁盘加载 checkpoint 并评估

        适用场景: 评估已保存的模型，与训练解耦
        这是 evaluation 模块的主入口，完全独立于 training 模块。
        """

        output_dir = cfg.evaluation_dir
        ensure_dir(output_dir)
        results = {}
        all_models = {}  # 收集所有实验的模型用于 Loss Landscape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for idx, ckpt_path in enumerate(checkpoint_paths, 1):
            progress = f"[{idx:>2}/{len(checkpoint_paths)}]"
            get_logger().info(f"\n{'═' * 70}")
            get_logger().info(
                f"{progress} 📦 Loading: {Path(ckpt_path).parent.parent.name}"
            )

            # 加载模型
            ctx = CheckpointLoader.load(ckpt_path, cfg)
            exp_name = ctx["name"]
            # context 中的 models 已经被 CheckpointLoader 分配到了各个 GPU (如果可用)
            models = ctx["models"]
            all_models[exp_name] = models  # 保存用于后续分析

            # 使用通用评估方法
            result = cls._evaluate_models(
                models=models,
                exp_name=exp_name,
                test_loader=test_loader,
                cfg=cfg,
                device=device,
                corruption_dataset=corruption_dataset,
                ood_dataset=ood_dataset,
                run_gradcam=run_gradcam,
                run_adversarial=run_adversarial,
            )
            results[exp_name] = result

        # 模型距离计算
        if run_loss_landscape and all_models:
            get_logger().info("\n📐 Computing model distances...")
            distance_calc = ModelDistanceCalculator()

            for exp_name, models in all_models.items():
                dist_matrix = distance_calc.compute(models)
                results[exp_name]["distance_matrix"] = dist_matrix

                # 计算衍生指标并保存到 results
                n = len(dist_matrix)
                if n > 1:
                    import math

                    distances = [
                        dist_matrix[i][j] for i in range(n) for j in range(i + 1, n)
                    ]
                    count = len(distances)
                    avg_dist = sum(distances) / count if count > 0 else 0
                    results[exp_name]["avg_distance"] = avg_dist

                    if count > 1:
                        variance = sum((d - avg_dist) ** 2 for d in distances) / count
                        std_dist = math.sqrt(variance)
                    else:
                        std_dist = 0
                    results[exp_name]["std_distance"] = std_dist

                    if avg_dist > 0:
                        results[exp_name]["direction_diversity"] = min(
                            std_dist / avg_dist, 1.0
                        )
                    else:
                        results[exp_name]["direction_diversity"] = 0

        # 生成并保存文本报告
        cls._save_and_print(results, output_dir)

        get_logger().info(f"\n✅ Complete! All reports saved to: {output_dir}")
        return results
