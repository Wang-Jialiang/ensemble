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
    from ..datasets.robustness.domain import DomainShiftDataset
    from ..datasets.robustness.ood import OODDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..utils import ensure_dir, format_duration, get_logger
from .adversarial import evaluate_adversarial
from .checkpoint import CheckpointLoader
from .corruption_robustness import evaluate_corruption
from .domain_robustness import evaluate_domain_shift
from .gradcam import GradCAMAnalyzer, ModelListWrapper
from .inference import get_all_models_logits, get_models_from_source
from .landscape import LossLandscapeVisualizer
from .metrics import MetricsCalculator
from .ood import evaluate_ood
from .saver import ResultsSaver
from .strategies import get_ensemble_fn
from .visualizer import ReportVisualizer

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 报告生成器 (评估 + 报告)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ReportGenerator:
    """实验评估与报告生成器

    两种主要使用方式:
        1. 从内存评估 (训练后立即评估):
           ReportGenerator.evaluate_trainers(trainers=[...], ...)

        2. 从磁盘评估 (加载 checkpoint):
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
        models: List[nn.Module],
        exp_name: str,
        test_loader: DataLoader,
        cfg: Config,
        device: torch.device,
        training_time: float = 0.0,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        ood_dataset: Optional["OODDataset"] = None,
        domain_dataset: Optional["DomainShiftDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ) -> Dict[str, Any]:
        """通用模型评估方法 - 核心评估逻辑"""
        get_logger().info(f"\n📊 Evaluating: {exp_name}")
        ensemble_fn = get_ensemble_fn(cfg)

        # ════════════════════════════════════════════════════════════════
        # 标准评估 (Standard Metrics)
        # ════════════════════════════════════════════════════════════════
        get_logger().info("   🔍 Standard evaluation...")
        all_logits, all_targets = get_all_models_logits(models, test_loader, device)
        metrics_calc = MetricsCalculator(cfg.num_classes, cfg.ece_n_bins)
        standard_metrics = metrics_calc.calculate_all_metrics(
            all_logits, all_targets, ensemble_fn=ensemble_fn
        )

        get_logger().info(f"   Ensemble Acc:   {standard_metrics['ensemble_acc']:.2f}%")
        get_logger().info(f"   ECE:            {standard_metrics['ece']:.4f}")

        # ════════════════════════════════════════════════════════════════
        # 鲁棒性评估 (Robustness Evaluation)
        # ════════════════════════════════════════════════════════════════
        corruption_results = None
        if corruption_dataset is not None:
            get_logger().info("   🔍 Corruption evaluation...")
            corruption_results = evaluate_corruption(
                models, corruption_dataset, config=cfg
            )

        # OOD 评估
        ood_results = None
        if ood_dataset is not None:
            get_logger().info("   🔍 OOD detection evaluation...")
            ood_loader = ood_dataset.get_loader(config=cfg)
            ood_results = evaluate_ood(
                models, test_loader, ood_loader, ood_name=ood_dataset.name
            )

        # Domain Shift 评估 (遍历所有 style × strength 组合)
        domain_results = None
        if domain_dataset is not None:
            get_logger().info("   🔍 Domain shift evaluation...")
            domain_results = {"by_style_strength": {}, "overall_avg": 0.0}
            all_accs = []

            for style in domain_dataset.STYLES:
                for strength in domain_dataset.STRENGTHS:
                    try:
                        domain_loader = domain_dataset.get_loader(style, strength, cfg)
                        result = evaluate_domain_shift(
                            models,
                            domain_loader,
                            domain_name=f"{style}_{strength}",
                            num_classes=cfg.num_classes,
                        )
                        key = f"{style}_{strength}"
                        domain_results["by_style_strength"][key] = result
                        all_accs.append(result["domain_acc"])
                    except FileNotFoundError:
                        get_logger().warning(
                            f"      跳过未生成的组合: {style}/{strength}"
                        )

            if all_accs:
                domain_results["overall_avg"] = sum(all_accs) / len(all_accs)
                get_logger().info(
                    f"   ✅ Domain Overall Avg: {domain_results['overall_avg']:.2f}%"
                )

        # 对抗鲁棒性评估
        adversarial_results = None
        if run_adversarial:
            get_logger().info("   🔍 Adversarial evaluation...")
            adversarial_results = evaluate_adversarial(
                models,
                test_loader,
                eps=cfg.adv_eps,
                alpha=cfg.adv_alpha,
                pgd_steps=cfg.adv_pgd_steps,
                dataset_name=cfg.dataset_name,
            )

        # ════════════════════════════════════════════════════════════════
        # 可解释性分析 (Grad-CAM)
        # ════════════════════════════════════════════════════════════════
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
            "ood_results": ood_results,
            "domain_results": domain_results,
            "adversarial_results": adversarial_results,
            "gradcam_metrics": gradcam_metrics,
        }

    @staticmethod
    def _evaluate_trainer(
        trainer: Any,
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ) -> Dict[str, Any]:
        """评估单个 trainer 并返回结果字典"""
        models, device = get_models_from_source(trainer)
        return ReportGenerator._evaluate_models(
            models=models,
            exp_name=trainer.name,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            training_time=getattr(trainer, "total_training_time", 0.0),
            corruption_dataset=corruption_dataset,
            run_gradcam=run_gradcam,
            run_adversarial=run_adversarial,
        )

    @classmethod
    def _generate_report(cls, results: Dict[str, Any]) -> str:
        """生成报告字符串"""
        lines = []

        def log(s=""):
            lines.append(str(s))

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
                f"      Disagreement: {m.get('disagreement', 0):.2f}%  |  JS散度: {m.get('js_divergence', 0):.4f}  |  Spearman: {m.get('spearman_correlation', 1.0):.4f}"
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

        # Adversarial Robustness
        has_adversarial = any(results[n].get("adversarial_results") for n in exp_names)
        if has_adversarial:
            log("\n🗡️ Adversarial Robustness")
            log("-" * 80)
            log(
                f"   {'Experiment':<25} | {'Clean↑':<10} | {'FGSM↑':<10} | {'PGD↑':<10} | {'ε':<10}"
            )
            log("-" * 80)
            pgd_vals = [
                results[n].get("adversarial_results", {}).get("pgd_acc", 0)
                for n in exp_names
                if results[n].get("adversarial_results")
            ]
            for name in exp_names:
                adv = results[name].get("adversarial_results", {})
                if adv:
                    clean = adv.get("clean_acc", 0)
                    fgsm = adv.get("fgsm_acc", 0)
                    pgd = adv.get("pgd_acc", 0)
                    eps = adv.get("eps_255", 8)
                    mark = cls._get_rank_marker(pgd, pgd_vals, True)
                    log(
                        f"   {name:<25} | {clean:<10.2f} | {fgsm:<10.2f} | {pgd:<7.2f}{mark:<3} | {eps:.0f}/255"
                    )
            log("-" * 80)

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
    def evaluate_trainers(
        cls,
        trainers: List,  # List of StagedEnsembleTrainer instances
        test_loader: DataLoader,
        cfg: Config,
        save_dir: str,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ):
        """
        从内存评估多个 trainer 并生成报告

        适用场景: 训练刚完成，模型还在内存中
        """
        get_logger().info(
            f"\n{'=' * 80}\n📊 EVALUATION MODE | Models: {len(trainers)}\n{'=' * 80}"
        )

        # 评估所有 trainers
        results = {}
        for idx, trainer in enumerate(trainers, 1):
            get_logger().info(f"\n[{idx}/{len(trainers)}] {trainer.name}")
            result = cls._evaluate_trainer(
                trainer,
                test_loader,
                cfg,
                corruption_dataset,
                run_gradcam,
                run_adversarial,
            )
            results[trainer.name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(save_dir, dpi=cfg.plot_dpi)
        visualizer.generate_all(results)

        # 生成并保存报告
        cls._save_and_print(results, save_dir)

    @classmethod
    def evaluate_checkpoints(
        cls,
        checkpoint_paths: List[str],
        test_loader: DataLoader,
        cfg: Config,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        ood_dataset: Optional["OODDataset"] = None,
        domain_dataset: Optional["DomainShiftDataset"] = None,
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

        output_dir = cfg.save_dir
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
                ood_dataset=ood_dataset,
                domain_dataset=domain_dataset,
                run_gradcam=run_gradcam,
                run_adversarial=run_adversarial,
            )
            result["train_config"] = ctx["config"]
            results[exp_name] = result

        # 生成可视化图表
        get_logger().info("\n📊 Generating visualizations...")
        visualizer = ReportVisualizer(output_dir, dpi=cfg.plot_dpi)
        visualizer.generate_all(results)

        # Loss Landscape 分析
        if run_loss_landscape and all_models:
            get_logger().info("\n🏔️ Generating Loss Landscape visualizations...")
            landscape_viz = LossLandscapeVisualizer(output_dir, dpi=cfg.plot_dpi)

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
