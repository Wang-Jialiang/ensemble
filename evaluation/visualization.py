"""
================================================================================
可视化与报告生成模块
================================================================================

包含: ReportVisualizer, ResultsSaver, ReportGenerator
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..utils import ensure_dir, format_duration, get_logger
from .checkpoint import CheckpointLoader
from .core import (
    MetricsCalculator,
    extract_models,
    get_all_models_logits,
    get_ensemble_fn,
)
from .gradcam import GradCAMAnalyzer, LossLandscapeVisualizer, ModelListWrapper
from .robustness import evaluate_adversarial, evaluate_corruption

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
        import csv

        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        csv_path = self.save_dir / f"{filename}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])

        get_logger().info(f"💾 Metrics saved to: {json_path}")

    def save_comparison(self, results: Dict[str, Dict], filename: str = "comparison"):
        """保存多个实验的对比结果"""
        import csv

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
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ) -> Dict[str, Any]:
        """通用模型评估方法 - 核心评估逻辑"""
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

        # 对抗鲁棒性评估
        adversarial_results = None
        if run_adversarial:
            get_logger().info("   🔍 Adversarial evaluation...")
            adversarial_results = evaluate_adversarial(
                models,
                test_loader,
                eps=getattr(cfg, "adv_eps", 8 / 255),
                alpha=getattr(cfg, "adv_alpha", 2 / 255),
                pgd_steps=getattr(cfg, "adv_pgd_steps", 10),
                dataset_name=cfg.dataset_name,
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
    def evaluate_and_report(
        cls,
        trainers: List,  # List of StagedEnsembleTrainer instances
        test_loader: DataLoader,
        cfg: Config,
        save_dir: str,
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_adversarial: bool = True,
    ):
        """评估多个 trainer 并生成报告 (一步完成)"""
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
        corruption_dataset: Optional["CorruptionDataset"] = None,
        run_gradcam: bool = False,
        run_loss_landscape: bool = False,
        run_adversarial: bool = True,
    ):
        """
        从 checkpoint 直接评估并生成完整可视化报告

        这是 evaluation 模块的主入口，完全独立于 training 模块。
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
                run_adversarial=run_adversarial,
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
