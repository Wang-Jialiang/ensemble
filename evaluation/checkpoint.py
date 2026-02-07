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

    @staticmethod
    def _fix_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """修复 state_dict 键，移除 torch.compile 产生的 _orig_mod. 前缀"""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]  # len("_orig_mod.") == 10
            new_state_dict[k] = v
        return new_state_dict

    @classmethod
    def _load_model_from_file(
        cls, model_file: Path, cfg: "Config", device: str = None
    ) -> nn.Module:
        """
        从文件加载单个模型，自动检测文件格式

        Args:
            model_file: 模型文件路径
            cfg: 配置对象
            device: 目标设备 (如 "cuda:0", "cuda:1")
        """
        state_dict = torch.load(model_file, weights_only=False)

        model = ModelFactory.create_model(cfg.model_name, num_classes=cfg.num_classes)

        # 修复并加载
        state_dict = cls._fix_state_dict_keys(state_dict)
        model.load_state_dict(state_dict)

        model.eval()
        # 移到指定设备
        if device:
            model = model.to(device)
        elif torch.cuda.is_available():
            model = model.cuda()
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
                'models': List[nn.Module]
            }
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

        # 推断实验名称
        # 路径格式: {save_root}/training/{ts}/{exp_name}/checkpoints/best
        # checkpoint_dir = .../checkpoints/best
        # checkpoint_dir.parent = .../checkpoints
        # checkpoint_dir.parent.parent = .../{exp_name}
        experiment_name = checkpoint_dir.parent.parent.name

        # 查找模型文件
        model_files = CheckpointLoader._find_model_files(
            checkpoint_dir, experiment_name
        )

        if not model_files:
            raise RuntimeError(f"未找到模型文件: {checkpoint_dir}")

        # 限制模型数量 (eval_num_models)
        eval_num_models = getattr(cfg, "eval_num_models", None)
        if eval_num_models is not None and eval_num_models > 0:
            model_files = model_files[:eval_num_models]

        # 获取可用 GPU 列表
        gpu_ids = getattr(cfg, "gpu_ids", [0]) if cfg else [0]
        if not gpu_ids:
            gpu_ids = [0]
        n_gpus = len(gpu_ids)

        # 加载模型并分配到不同 GPU
        models = []
        for i, model_file in enumerate(model_files):
            # 循环分配到各 GPU
            gpu_idx = gpu_ids[i % n_gpus]
            model = CheckpointLoader._load_model_from_file(
                model_file, cfg, device=f"cuda:{gpu_idx}"
            )
            models.append(model)

        # 日志提示
        total_available = len(
            CheckpointLoader._find_model_files(checkpoint_dir, experiment_name)
        )
        if eval_num_models is not None and eval_num_models > 0:
            get_logger().info(
                f"✅ 加载 {experiment_name}: {len(models)}/{total_available} 个模型 "
                f"(eval_num_models={eval_num_models}, 分布在 {n_gpus} GPU)"
            )
        else:
            get_logger().info(
                f"✅ 加载 {experiment_name}: {len(models)} 个模型 (分布在 {n_gpus} GPU)"
            )

        return {
            "name": experiment_name,
            "models": models,
        }
