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
from .scoring import DIMENSION_DISPLAY, DIMENSION_WEIGHTS, ScoreCalculator
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

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成增强版文本报告（含评分系统）"""
        lines = []
        exps = list(results.keys())

        # 0. 计算所有实验分数
        all_scores = {
            name: ScoreCalculator.calculate_all_scores(r) for name, r in results.items()
        }

        # 1. 综合评分卡 (NEW)
        lines.extend(cls._format_scorecard(results, exps, all_scores))

        # 2. 各维度详细分解 (NEW)
        lines.extend(cls._format_dimension_breakdown(results, exps, all_scores))

        # 3. 分隔线
        lines.append("\n" + "=" * 115)
        lines.append("📋 DETAILED METRICS")
        lines.append("=" * 115)

        # 4. 原有核心性能对比表 (保留)
        lines.extend(cls._format_perf_table(results, exps))

        # 5. 多样性/公平性/CAM 表格 (保留)
        lines.extend(cls._format_diversity_table(results, exps))

        # 6. CKA 详情 + EOD/Bottom-K (保留)
        lines.extend(cls._format_additional_metrics(results, exps))

        # 7. 鲁棒性专门板块 (保留)
        lines.extend(cls._format_robustness_sections(results, exps))

        return "\n".join(lines)

    @classmethod
    def _format_scorecard(cls, results, names, all_scores):
        """生成综合评分卡"""
        lines = []

        # Header
        lines.append("═" * 115)
        lines.append("                              🏆 ENSEMBLE EVALUATION SCORECARD")
        lines.append("═" * 115)
        lines.append("")

        # 排序实验
        sorted_exps = sorted(
            names, key=lambda n: all_scores[n]["total_score"], reverse=True
        )

        # 表头 - 动态构建维度列
        dim_order = [
            "accuracy",
            "calibration",
            "diversity",
            "fairness",
            "corruption",
            "ood",
            "adversarial",
            "interpretability",
        ]
        dim_headers = []
        for dim in dim_order:
            if dim in DIMENSION_DISPLAY:
                icon, _, short = DIMENSION_DISPLAY[dim]
                dim_headers.append(f"{short[:6]:^6}")

        header = (
            f"│ {'Experiment':<22} │ {'Score':^6} │ {'Grade':^5} │ "
            + " │ ".join(dim_headers)
            + " │"
        )
        sep_line = (
            "├"
            + "─" * 24
            + "┼"
            + "─" * 8
            + "┼"
            + "─" * 7
            + "┼"
            + ("─" * 8 + "┼") * len(dim_headers)
        )
        sep_line = sep_line[:-1] + "┤"

        lines.append(
            "┌"
            + "─" * 24
            + "┬"
            + "─" * 8
            + "┬"
            + "─" * 7
            + "┬"
            + ("─" * 8 + "┬") * len(dim_headers)
        )
        lines[-1] = lines[-1][:-1] + "┐"
        lines.append(header)
        lines.append(sep_line)

        # 数据行
        for rank, name in enumerate(sorted_exps):
            score_data = all_scores[name]
            medal = ScoreCalculator.get_medal(rank, len(sorted_exps))

            # 截断名称
            display_name = name[:18] if len(name) > 18 else name
            if medal:
                display_name = f"{medal} {display_name}"

            # 维度分数
            dim_scores = []
            for dim in dim_order:
                if dim in score_data["dimensions"]:
                    dim_score = score_data["dimensions"][dim]["score"]
                    dim_scores.append(f"{dim_score:^6.0f}")
                else:
                    dim_scores.append(f"{'N/A':^6}")

            row = (
                f"│ {display_name:<22} │ {score_data['total_score']:^6.1f} │ {score_data['grade']:^5} │ "
                + " │ ".join(dim_scores)
                + " │"
            )
            lines.append(row)

        lines.append(
            "└"
            + "─" * 24
            + "┴"
            + "─" * 8
            + "┴"
            + "─" * 7
            + "┴"
            + ("─" * 8 + "┴") * len(dim_headers)
        )
        lines[-1] = lines[-1][:-1] + "┘"
        lines.append("")

        # 图例
        lines.append(
            "📌 维度权重: "
            + " | ".join(
                [
                    f"{DIMENSION_DISPLAY[d][0]}{DIMENSION_DISPLAY[d][2]}({int(DIMENSION_WEIGHTS[d] * 100)}%)"
                    for d in dim_order
                    if d in DIMENSION_DISPLAY
                ]
            )
        )
        lines.append("📌 等级标准: S(≥90) | A(≥80) | B(≥70) | C(≥60) | D(<60)")
        lines.append("")

        return lines

    @classmethod
    def _format_dimension_breakdown(cls, results, names, all_scores):
        """生成各维度详细分解"""
        lines = []

        lines.append("─" * 115)
        lines.append("                              📊 DIMENSION BREAKDOWN")
        lines.append("─" * 115)

        # 排序实验
        sorted_exps = sorted(
            names, key=lambda n: all_scores[n]["total_score"], reverse=True
        )

        dim_order = [
            "accuracy",
            "calibration",
            "diversity",
            "fairness",
            "corruption",
            "ood",
            "adversarial",
            "interpretability",
        ]

        for dim in dim_order:
            # 检查是否有任何实验有此维度数据
            has_data = any(dim in all_scores[n]["dimensions"] for n in names)
            if not has_data:
                continue

            icon, cn_name, en_name = DIMENSION_DISPLAY.get(dim, ("", dim, dim))
            weight = DIMENSION_WEIGHTS.get(dim, 0)

            lines.append(f"\n{icon} {en_name} (Weight: {int(weight * 100)}%)")
            lines.append("-" * 80)

            # 收集所有该维度的指标
            all_metrics = set()
            for n in names:
                if dim in all_scores[n]["dimensions"]:
                    all_metrics.update(
                        all_scores[n]["dimensions"][dim]["metrics"].keys()
                    )
            all_metrics = sorted(all_metrics)[:5]  # 最多显示 5 个指标

            # 表头
            metric_headers = [f"{m[:10]:<10}" for m in all_metrics]
            header = (
                f"{'Experiment':<25} │ "
                + " │ ".join(metric_headers)
                + f" │ {'Score':>6}"
            )
            lines.append(header)
            lines.append("-" * 80)

            # 收集分数用于排名
            dim_scores = [
                (n, all_scores[n]["dimensions"].get(dim, {}).get("score", 0))
                for n in sorted_exps
            ]
            dim_scores_values = [s for _, s in dim_scores]

            for rank, name in enumerate(sorted_exps):
                if dim not in all_scores[name]["dimensions"]:
                    continue

                dim_data = all_scores[name]["dimensions"][dim]
                medal_str = ""
                if len(dim_scores_values) > 1:
                    sorted_scores = sorted(dim_scores_values, reverse=True)
                    if dim_data["score"] == sorted_scores[0]:
                        medal_str = "🥇"
                    elif (
                        len(sorted_scores) > 1 and dim_data["score"] == sorted_scores[1]
                    ):
                        medal_str = "🥈"
                    elif (
                        len(sorted_scores) > 2 and dim_data["score"] == sorted_scores[2]
                    ):
                        medal_str = "🥉"

                display_name = name[:22] if len(name) > 22 else name

                metric_vals = []
                for m in all_metrics:
                    val = dim_data["metrics"].get(m, None)
                    if val is not None:
                        metric_vals.append(f"{val:<10.1f}")
                    else:
                        metric_vals.append(f"{'N/A':<10}")

                row = (
                    f"{display_name:<25} │ "
                    + " │ ".join(metric_vals)
                    + f" │ {dim_data['score']:>5.1f} {medal_str}"
                )
                lines.append(row)

            lines.append("-" * 80)

        return lines

    @classmethod
    def _format_perf_table(cls, results, names):
        """核心性能对比表 - 带排名标记"""
        t = [
            "\n🎯 Performance Metrics",
            "-" * 100,
            f"{'Experiment':<25} | {'EnsAcc↑':<10} | {'AvgInd↑':<10} | {'Oracle↑':<10} | {'ECE↓':<10} | {'NLL↓':<10}",
            "-" * 100,
        ]

        # 收集所有指标值用于排名
        def get_vals(key):
            return [results[n].get("standard_metrics", {}).get(key, 0) for n in names]

        ens_accs = get_vals("ensemble_acc")
        avg_accs = get_vals("avg_individual_acc")
        oracle_accs = get_vals("oracle_acc")
        eces = get_vals("ece")
        nlls = get_vals("nll")

        for n in names:
            m = results[n].get("standard_metrics", {})

            # 获取每个指标的排名标记
            ens_mark = cls._get_rank_marker(m.get("ensemble_acc", 0), ens_accs, True)
            avg_mark = cls._get_rank_marker(
                m.get("avg_individual_acc", 0), avg_accs, True
            )
            ora_mark = cls._get_rank_marker(m.get("oracle_acc", 0), oracle_accs, True)
            ece_mark = cls._get_rank_marker(
                m.get("ece", 0), eces, False
            )  # ↓ lower is better
            nll_mark = cls._get_rank_marker(
                m.get("nll", 0), nlls, False
            )  # ↓ lower is better

            t.append(
                f"{n:<25} | {m.get('ensemble_acc', 0):<6.2f}{ens_mark:<4} | "
                f"{m.get('avg_individual_acc', 0):<6.2f}{avg_mark:<4} | "
                f"{m.get('oracle_acc', 0):<6.2f}{ora_mark:<4} | "
                f"{m.get('ece', 0):<6.4f}{ece_mark:<4} | "
                f"{m.get('nll', 0):<6.4f}{nll_mark:<4}"
            )
        t.append("-" * 100)
        return t

    @classmethod
    def _format_diversity_table(cls, results, names):
        """生成 Div/Fair/CAM 横向表格 - 带排名标记"""
        has_cam = any(results[n].get("gradcam_metrics") for n in names)

        header = f"{'Experiment':<25} | {'Dis↑':<10} | {'CKA_Div↑':<10} | {'BalAcc↑':<10} | {'Gini↓':<10} | {'Fair↑':<10}"
        if has_cam:
            header += f" | {'Entropy':<8} | {'Sim↓':<8} | {'Overlap↓':<8}"

        t = [
            "\n🔀 Diversity / Fairness / CAM Metrics",
            "-" * (115 if not has_cam else 145),
            header,
            "-" * (115 if not has_cam else 145),
        ]

        # 收集所有指标值
        def get_vals(key):
            return [results[n].get("standard_metrics", {}).get(key, 0) for n in names]

        def get_cam_vals(key):
            return [results[n].get("gradcam_metrics", {}).get(key, 0) for n in names]

        dis_vals = get_vals("disagreement")
        cka_div_vals = get_vals("cka_diversity")
        bal_vals = get_vals("balanced_acc")
        gini_vals = get_vals("acc_gini_coef")
        fair_vals = get_vals("fairness_score")
        sim_vals = get_cam_vals("avg_cam_similarity") if has_cam else []
        overlap_vals = get_cam_vals("avg_cam_overlap") if has_cam else []

        for n in names:
            m = results[n].get("standard_metrics", {})
            g = results[n].get("gradcam_metrics", {})

            # 获取排名标记
            dis_mark = cls._get_rank_marker(m.get("disagreement", 0), dis_vals, True)
            cka_mark = cls._get_rank_marker(
                m.get("cka_diversity", 0), cka_div_vals, True
            )
            bal_mark = cls._get_rank_marker(m.get("balanced_acc", 0), bal_vals, True)
            gini_mark = cls._get_rank_marker(
                m.get("acc_gini_coef", 0), gini_vals, False
            )  # ↓
            fair_mark = cls._get_rank_marker(
                m.get("fairness_score", 0), fair_vals, True
            )

            row = (
                f"{n:<25} | {m.get('disagreement', 0):<6.2f}{dis_mark:<4} | "
                f"{m.get('cka_diversity', 0):<6.4f}{cka_mark:<4} | "
                f"{m.get('balanced_acc', 0):<6.2f}{bal_mark:<4} | "
                f"{m.get('acc_gini_coef', 0):<6.4f}{gini_mark:<4} | "
                f"{m.get('fairness_score', 0):<6.2f}{fair_mark:<4}"
            )
            if has_cam:
                sim_mark = cls._get_rank_marker(
                    g.get("avg_cam_similarity", 0), sim_vals, False
                )
                ovl_mark = cls._get_rank_marker(
                    g.get("avg_cam_overlap", 0), overlap_vals, False
                )
                row += (
                    f" | {g.get('avg_cam_entropy', 0):<8.4f} | "
                    f"{g.get('avg_cam_similarity', 0):<4.4f}{sim_mark:<4} | "
                    f"{g.get('avg_cam_overlap', 0):<4.4f}{ovl_mark:<4}"
                )
            t.append(row)

        t.append("-" * (115 if not has_cam else 145))
        return t

    @classmethod
    def _format_additional_metrics(cls, results, names):
        """CKA 详情 + EOD/Bottom-K 表格"""
        t = []

        # ===== CKA 详情 =====
        t.append("\n📊 CKA Similarity Details")
        t.append("-" * 80)
        t.append(
            f"{'Experiment':<25} | {'Avg_CKA↓':<12} | {'Min_CKA':<12} | {'Max_CKA':<12} | {'CKA_Div↑':<12}"
        )
        t.append("-" * 80)

        for n in names:
            m = results[n].get("standard_metrics", {})
            t.append(
                f"{n:<25} | {m.get('avg_cka', 0):<12.4f} | "
                f"{m.get('min_cka', 0):<12.4f} | {m.get('max_cka', 0):<12.4f} | "
                f"{m.get('cka_diversity', 0):<12.4f}"
            )
        t.append("-" * 80)

        # ===== EOD + Bottom-K =====
        t.append("\n⚖️ Fairness Details (EOD + Bottom-K)")
        t.append("-" * 80)
        t.append(
            f"{'Experiment':<25} | {'EOD↓':<10} | {'Bottom3↑':<12} | {'Bottom5↑':<12}"
        )
        t.append("-" * 80)

        for n in names:
            m = results[n].get("standard_metrics", {})
            t.append(
                f"{n:<25} | {m.get('eod', 0):<10.2f} | "
                f"{m.get('bottom_3_class_acc', 0):<12.2f} | "
                f"{m.get('bottom_5_class_acc', 0):<12.2f}"
            )
        t.append("-" * 80)

        return t

    @classmethod
    def _format_robustness_sections(cls, results, names):
        """鲁棒性综合报告 - 包含 OOD/Adversarial/Corruption 全部指标"""
        s = []

        # ===== 1. 对抗鲁棒性 =====
        s.append("\n⚔️ Adversarial Robustness")
        s.append("-" * 100)
        s.append(
            f"{'Experiment':<25} | {'Clean↑':<10} | {'FGSM↑':<10} | {'PGD↑':<10} | {'ε':<8} | {'Steps':<6}"
        )
        s.append("-" * 100)

        for n in names:
            adv = results[n].get("adversarial_results") or {}
            if not adv:
                s.append(
                    f"{n:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<8} | {'N/A':<6}"
                )
            elif "pgd_acc" in adv:
                # 单 ε 模式
                s.append(
                    f"{n:<25} | {adv.get('clean_acc', 0):<10.2f} | "
                    f"{adv.get('fgsm_acc', 0):<10.2f} | {adv.get('pgd_acc', 0):<10.2f} | "
                    f"{adv.get('eps_255', 0):<8.1f} | {adv.get('pgd_steps', 0):<6}"
                )
            else:
                # 多 ε 模式: 显示每个 ε 的结果
                for eps_key, eps_data in adv.items():
                    s.append(
                        f"{n:<25} | {eps_data.get('clean_acc', 0):<10.2f} | "
                        f"{eps_data.get('fgsm_acc', 0):<10.2f} | {eps_data.get('pgd_acc', 0):<10.2f} | "
                        f"{eps_data.get('eps_255', 0):<8.1f} | {eps_data.get('pgd_steps', 0):<6}"
                    )
        s.append("-" * 100)

        # ===== 2. OOD 检测 =====
        has_ood = any(results[n].get("ood_results") for n in names)
        if has_ood:
            s.append("\n🔮 OOD Detection")
            s.append("-" * 100)
            s.append(
                f"{'Experiment':<25} | {'AUROC_MSP↑':<12} | {'AUROC_Ent↑':<12} | {'FPR95_MSP↓':<12} | {'FPR95_Ent↓':<12}"
            )
            s.append("-" * 100)

            for n in names:
                ood = results[n].get("ood_results") or {}
                if ood:
                    s.append(
                        f"{n:<25} | {ood.get('ood_auroc_msp', 0):<12.2f} | "
                        f"{ood.get('ood_auroc_entropy', 0):<12.2f} | "
                        f"{ood.get('ood_fpr95_msp', 0):<12.2f} | "
                        f"{ood.get('ood_fpr95_entropy', 0):<12.2f}"
                    )
                else:
                    s.append(
                        f"{n:<25} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}"
                    )
            s.append("-" * 100)

        # ===== 3. Corruption 鲁棒性 =====
        has_corr = any(results[n].get("corruption_results") for n in names)
        if has_corr:
            s.append("\n🌪️ Corruption Robustness")
            s.append("-" * 100)

            # 3.1 总体平均
            s.append(f"{'Experiment':<25} | {'Overall↑':<10}")
            s.append("-" * 50)
            for n in names:
                corr = results[n].get("corruption_results") or {}
                s.append(f"{n:<25} | {corr.get('overall_avg', 0):<10.2f}")
            s.append("")

            # 3.2 按严重程度展示
            first_corr = next(
                (
                    results[n].get("corruption_results")
                    for n in names
                    if results[n].get("corruption_results")
                ),
                {},
            )
            severities = sorted(
                list((first_corr.get("by_severity") or {}).keys()), key=lambda x: int(x)
            )

            if severities:
                s.append(
                    f"{'Experiment':<25} | "
                    + " | ".join([f"Sev {str(sev):<4}" for sev in severities])
                )
                s.append("-" * (28 + 10 * len(severities)))
                for n in names:
                    corr = results[n].get("corruption_results") or {}
                    by_sev = corr.get("by_severity") or {}
                    sev_vals = " | ".join(
                        [f"{by_sev.get(sev, 0):<8.2f}" for sev in severities]
                    )
                    s.append(f"{n:<25} | {sev_vals}")
            s.append("")

            # 3.3 按类别展示
            categories = list((first_corr.get("by_category") or {}).keys())
            if categories:
                s.append(
                    f"{'Experiment':<25} | "
                    + " | ".join([f"{c:<10}" for c in categories])
                )
                s.append("-" * (30 + 13 * len(categories)))
                for n in names:
                    corr = results[n].get("corruption_results") or {}
                    by_cat = corr.get("by_category") or {}
                    cat_vals = " | ".join(
                        [f"{by_cat.get(c, 0):<10.2f}" for c in categories]
                    )
                    s.append(f"{n:<25} | {cat_vals}")
            s.append("-" * 100)

        return s

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
        get_logger().info(f"\n{'=' * 80}")
        get_logger().info(
            f"📊 EVALUATION FROM CHECKPOINTS | Count: {len(checkpoint_paths)}"
        )
        get_logger().info(f"{'=' * 80}")

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

        # 生成并保存文本报告
        cls._save_and_print(results, output_dir)

        get_logger().info(f"\n✅ Complete! All reports saved to: {output_dir}")
        return results
