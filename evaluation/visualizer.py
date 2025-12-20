"""
================================================================================
报告可视化器模块
================================================================================

包含: ReportVisualizer - 生成可视化图表 (matplotlib)
"""

from pathlib import Path
from typing import Dict

import numpy as np

from ..utils import ensure_dir, get_logger

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
