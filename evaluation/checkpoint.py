"""
================================================================================
Checkpoint 加载模块
================================================================================
"""

from pathlib import Path
from typing import Any, Dict

import torch

from ..models import ModelFactory
from ..utils import get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Checkpoint 加载器                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointLoader:
    """从 checkpoint 加载模型进行评估

    完全独立于训练模块，只需 checkpoint 路径和配置即可加载模型。
    """

    @staticmethod
    def load(checkpoint_path: str, cfg: "Config") -> Dict[str, Any]:
        """
        加载 checkpoint 并返回可评估的模型上下文

        Args:
            checkpoint_path: checkpoint 目录路径
            cfg: 配置对象

        Returns:
            context: {
                'name': 实验名称,
                'models': List[nn.Module],
                'training_time': float,
                'config': dict
            }
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

        # 推断实验名称
        experiment_name = checkpoint_dir.parent.name

        # 读取训练状态
        state_path = checkpoint_dir / "trainer_state.pth"
        training_time = 0.0
        train_config = {}

        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            training_time = state.get("total_training_time", 0.0)
            train_config = {
                "augmentation_method": state.get("augmentation_method", "unknown"),
                "use_curriculum": state.get("use_curriculum", False),
            }

        # 加载所有模型
        models = []
        model_files = sorted(checkpoint_dir.glob(f"{experiment_name}_*.pth"))

        for model_file in model_files:
            model = ModelFactory.create_model(
                cfg.model_name, num_classes=cfg.num_classes
            )
            state = torch.load(model_file, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            models.append(model)

        if not models:
            raise RuntimeError(f"未找到模型文件: {checkpoint_dir}")

        get_logger().info(f"✅ 加载 {experiment_name}: {len(models)} 个模型")

        return {
            "name": experiment_name,
            "models": models,
            "training_time": training_time,
            "config": train_config,
        }
