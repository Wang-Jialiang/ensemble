"""
================================================================================
Checkpoint 加载器模块
================================================================================

包含: CheckpointLoader
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn as nn

from ..models import ModelFactory
from ..utils import get_logger

if TYPE_CHECKING:
    from ..config import Config

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Checkpoint 加载器                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointLoader:
    """从 checkpoint 加载模型进行评估

    支持 StagedEnsembleTrainer 的 checkpoint 格式：
    - {name}_gpu{id}_model{i}.pth
    """

    # 支持的模型文件模式
    MODEL_PATTERNS = [
        "*_gpu*_model*.pth",  # StagedEnsembleTrainer
    ]

    @classmethod
    def _find_model_files(
        cls, checkpoint_dir: Path, experiment_name: str
    ) -> List[Path]:
        """
        查找 checkpoint 目录中的模型文件

        Returns:
            model_files: 排序后的模型文件路径列表
        """
        model_files = []

        # 首先尝试使用实验名称前缀匹配
        files = sorted(checkpoint_dir.glob(f"{experiment_name}_*.pth"))
        if files:
            return files

        # 依次尝试各种模式
        for pattern in cls.MODEL_PATTERNS:
            files = sorted(checkpoint_dir.glob(pattern))
            if files:
                return files

        return model_files  # 空列表

    @classmethod
    def _load_model_from_file(cls, model_file: Path, cfg: "Config") -> nn.Module:
        """
        从文件加载单个模型，自动检测文件格式
        """
        state = torch.load(model_file, weights_only=False)

        model = ModelFactory.create_model(cfg.model_name, num_classes=cfg.num_classes)

        # 支持两种格式: dict 包装 或直接 state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        model.eval()
        return model

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

        # 查找模型文件
        model_files = CheckpointLoader._find_model_files(
            checkpoint_dir, experiment_name
        )

        if not model_files:
            raise RuntimeError(f"未找到模型文件: {checkpoint_dir}")

        # 加载模型
        models = []
        for model_file in model_files:
            model = CheckpointLoader._load_model_from_file(model_file, cfg)
            models.append(model)

        get_logger().info(f"✅ 加载 {experiment_name}: {len(models)} 个模型")

        return {
            "name": experiment_name,
            "models": models,
            "training_time": training_time,
            "config": train_config,
        }
