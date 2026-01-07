"""
================================================================================
评估结果保存器模块
================================================================================

包含: ResultsSaver - 保存评估结果为 JSON 和 CSV
"""

import json
from pathlib import Path
from typing import Any, Dict

from ..utils import ensure_dir, get_logger

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
                default=lambda x: x.tolist()
                if hasattr(x, "tolist")
                else (x.item() if hasattr(x, "item") else x),
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
