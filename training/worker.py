"""
================================================================================
GPU Worker 模块
================================================================================

GPUWorker (单GPU模型管理器)、HistorySaver (训练历史保存器)
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from ..config import Config
from ..models import ModelFactory
from ..utils import ensure_dir, get_logger
from .augmentation import AUGMENTATION_REGISTRY, AugmentationMethod
from .scheduler import create_optimizer, create_scheduler

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ GPU Worker                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class GPUWorker:
    """单GPU模型管理器 (支持多种数据增强方法)

    管理单个GPU上的多个模型实例，支持异步训练以最大化GPU利用率。
    """

    def __init__(
        self,
        gpu_id: int,
        num_models: int,
        cfg: Config,
        augmentation_method: str = "perlin",
    ):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.cfg = cfg
        self.num_models = num_models

        # 创建模型
        self.models: List[nn.Module] = []
        self.optimizers: List[optim.Optimizer] = []
        self.schedulers: List[Optional[optim.lr_scheduler.LRScheduler]] = []

        for _ in range(num_models):
            model = ModelFactory.create_model(
                cfg.model_name,
                num_classes=cfg.num_classes,
                init_method=cfg.init_method,
            )
            model = model.to(self.device)

            if cfg.compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)

            # 使用工厂函数创建优化器和调度器
            optimizer = create_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay)
            scheduler = create_scheduler(
                optimizer,
                cfg.scheduler,
                cfg.total_epochs,
                max_lr_factor=getattr(cfg, "onecycle_max_lr_factor", 10.0),
            )

            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

        # 创建增强方法
        self.augmentation_method = augmentation_method
        self.augmentation = self._create_augmentation(augmentation_method)

        # AMP
        self.scaler = GradScaler("cuda") if cfg.use_amp else None

        # Stream
        self.stream = torch.cuda.Stream(device=self.device)
        self._pending_loss = None

    def _create_augmentation(self, method: str) -> AugmentationMethod:
        """创建增强方法"""
        if method not in AUGMENTATION_REGISTRY:
            raise ValueError(
                f"不支持的增强方法: {method}. 支持: {list(AUGMENTATION_REGISTRY.keys())}"
            )
        return AUGMENTATION_REGISTRY[method](self.device, self.cfg)

    def precompute_masks(self, num_masks: int, target_ratio: float):
        """预计算mask（如果需要）"""
        if hasattr(self.augmentation, "precompute_masks"):
            self.augmentation.precompute_masks(target_ratio)

    def train_batch_async(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        mask_ratio: float,
        mask_prob: float,
        use_mask: bool,
    ):
        """异步训练一个batch"""
        with torch.cuda.stream(self.stream):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            total_loss = 0.0

            for model, optimizer in zip(self.models, self.optimizers):
                model.train()
                optimizer.zero_grad(set_to_none=True)

                # 应用增强
                if use_mask:
                    aug_inputs, aug_targets = self.augmentation.apply(
                        inputs, targets, mask_ratio, mask_prob
                    )
                else:
                    aug_inputs, aug_targets = inputs, targets

                # 前向传播
                if self.scaler:
                    with autocast("cuda"):
                        outputs = model(aug_inputs)
                        loss = criterion(outputs, aug_targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(aug_inputs)
                    loss = criterion(outputs, aug_targets)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
                    optimizer.step()

                total_loss += loss.item()

            self._pending_loss = total_loss / self.num_models

    def synchronize(self) -> float:
        """同步并返回平均loss"""
        self.stream.synchronize()
        return self._pending_loss if self._pending_loss else 0.0

    def step_schedulers(self, val_loss: Optional[float] = None):
        """更新学习率调度器

        Args:
            val_loss: 验证损失 (用于 ReduceLROnPlateau)
        """
        for scheduler in self.schedulers:
            if scheduler is None:
                continue
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    scheduler.step(val_loss)
            else:
                scheduler.step()

    def set_lr(self, lr: float):
        """设置所有模型的学习率

        Args:
            lr: 新的学习率
        """
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizers[0].param_groups[0]["lr"] if self.optimizers else 0.0

    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """批量预测"""
        inputs = inputs.to(self.device)
        all_logits = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
            all_logits.append(logits.unsqueeze(0))
        return torch.cat(all_logits, dim=0)

    def save_models(self, save_dir: str, prefix: str):
        """保存模型"""
        for i, (model, optimizer, scheduler) in enumerate(
            zip(self.models, self.optimizers, self.schedulers)
        ):
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }
            save_path = Path(save_dir) / f"{prefix}_gpu{self.gpu_id}_model{i}.pth"
            torch.save(state, save_path)

    def load_models(self, save_dir: str, prefix: str):
        """加载模型"""
        for i, (model, optimizer, scheduler) in enumerate(
            zip(self.models, self.optimizers, self.schedulers)
        ):
            load_path = Path(save_dir) / f"{prefix}_gpu{self.gpu_id}_model{i}.pth"
            if load_path.exists():
                state = torch.load(
                    load_path, map_location=self.device, weights_only=False
                )
                model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                if scheduler and state.get("scheduler_state_dict"):
                    scheduler.load_state_dict(state["scheduler_state_dict"])

    def broadcast_backbone_and_reinit_heads(self, backbone_state_dict: dict):
        """用共享 backbone 初始化所有模型，并重新初始化各模型的 classifier head

        Args:
            backbone_state_dict: 源模型的 backbone 权重 (不含 fc 层)
        """
        for model in self.models:
            # 加载 backbone 权重 (strict=False 因为不含 fc 层)
            model.load_state_dict(backbone_state_dict, strict=False)
            # 重新初始化 classifier head
            model.reinit_classifier()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 训练历史保存器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class HistorySaver:
    """训练历史保存器"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)

    def save(self, history: Dict[str, List], filename: str = "history"):
        """保存训练历史为JSON和CSV"""
        import csv
        import json

        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)

        csv_path = self.save_dir / f"{filename}.csv"
        with open(csv_path, "w", newline="") as f:
            if history:
                writer = csv.DictWriter(f, fieldnames=history.keys())
                writer.writeheader()
                for i in range(len(history[list(history.keys())[0]])):
                    row = {k: v[i] for k, v in history.items()}
                    writer.writerow(row)
        get_logger().info(f"💾 History saved to: {json_path}")
