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
from ..utils import ensure_dir, format_duration, get_logger
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
        models, exp_name, test_loader, cfg, device, training_time=0.0, **datasets
    ) -> Dict[str, Any]:
        """通用模型评估方法 - 生命周期钩子模式"""
        get_logger().info(f"\n📊 Evaluating: {exp_name}")
        res = {"experiment_name": exp_name, "training_time_seconds": training_time}

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
        get_logger().info("   🔍 Standard evaluation...")
        all_l, all_t = get_all_models_logits(models, loader, device)
        return MetricsCalculator(cfg.num_classes, cfg.ece_n_bins).calculate_all_metrics(
            all_l, all_t, get_ensemble_fn(cfg)
        )

    @staticmethod
    def _run_robustness_eval(models, cfg, loader, **ds):
        r = {"corruption_results": None, "ood_results": None}

        if ds.get("corruption_dataset"):
            get_logger().info("   🔍 Corruption evaluation...")
            r["corruption_results"] = evaluate_corruption(
                models, ds["corruption_dataset"], cfg
            )

        if ds.get("ood_dataset"):
            get_logger().info("   🔍 OOD detection evaluation...")
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
            get_logger().info("   🔍 Adversarial evaluation...")
            a["adversarial_results"] = evaluate_adversarial(
                models, loader, cfg=cfg, logger=get_logger()
            )

        if ds.get("run_gradcam", False):
            get_logger().info("   🔍 Grad-CAM analysis...")
            a["gradcam_metrics"] = GradCAMAnalyzer(cfg).analyze_ensemble_quality(
                [ModelListWrapper(models)],
                loader,
                cfg.gradcam_num_samples,
                cfg.image_size,
            )
        return a

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成文本报告 (大纲化渲染)"""
        lines = []
        exps = list(results.keys())

        # 1. 绘制 Header
        lines.append("=" * 115)
        lines.append(
            "📊 EXPERIMENT COMPARISON" if len(exps) > 1 else f"📊 RESULTS: {exps[0]}"
        )
        lines.append("=" * 115)

        # 2. 核心性能对比表
        lines.extend(cls._format_perf_table(results, exps))

        # 3. 多样性/公平性/CAM 表格
        lines.extend(cls._format_diversity_table(results, exps))

        # 4. CKA 详情 + EOD/Bottom-K
        lines.extend(cls._format_additional_metrics(results, exps))

        # 5. 鲁棒性专门板块
        lines.extend(cls._format_robustness_sections(results, exps))

        return "\n".join(lines)

    @classmethod
    def _format_perf_table(cls, results, names):
        """核心性能对比表 - 带排名标记"""
        t = [
            "\n🎯 Performance Metrics",
            "-" * 115,
            f"{'Experiment':<25} | {'EnsAcc↑':<10} | {'AvgInd↑':<10} | {'Oracle↑':<10} | {'ECE↓':<10} | {'NLL↓':<10} | {'Time':<12}",
            "-" * 115,
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
            tm = format_duration(results[n].get("training_time_seconds", 0))

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
                f"{m.get('nll', 0):<6.4f}{nll_mark:<4} | {tm:<12}"
            )
        t.append("-" * 115)
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
            get_logger().info(f"\n[{idx}/{len(checkpoint_paths)}] Loading: {ckpt_path}")

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
                training_time=ctx["training_time"],
                corruption_dataset=corruption_dataset,
                ood_dataset=ood_dataset,
                run_gradcam=run_gradcam,
                run_adversarial=run_adversarial,
            )
            result["train_config"] = ctx["config"]
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
