"""
================================================================================
评估结果保存器模块
================================================================================

包含: ResultsSaver - 保存评估结果为 JSON
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

    支持将评估指标保存为 JSON 格式。
    """

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics"):
        """保存单个实验的指标"""
        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        get_logger().info(f"💾 Metrics saved to: {json_path}")

    def save_comparison(self, results: Dict[str, Dict], filename: str = "comparison"):
        """保存多个实验的对比结果"""
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

        get_logger().info(f"💾 Comparison saved to: {json_path}")
