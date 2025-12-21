"""
================================================================================
Batch Ensemble 训练器
================================================================================

独立的 Batch Ensemble 训练器，与 StagedEnsembleTrainer 完全解耦。

特点:
- 训练单个 BatchEnsembleResNet 模型 (内含多个隐式成员)
- 无需多 GPU 协调，单 GPU 即可运行 ensemble
- 支持三阶段课程学习 (可选)
"""

import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models.batch_ensemble import BatchEnsembleResNet
from ..utils import ensure_dir, get_logger
from .base import BaseTrainer
from .scheduler import create_optimizer, create_scheduler


class BatchEnsembleTrainer(BaseTrainer):
    """
    Batch Ensemble 训练器

    训练单个 BatchEnsembleResNet 模型，内部包含多个隐式集成成员。
    """

    def __init__(
        self,
        method_name: str,
        cfg: Config,
        num_members: int = 4,
        use_curriculum: bool = False,
    ):
        # 调用基类初始化
        super().__init__(method_name, cfg)

        self.num_members = num_members
        self.use_curriculum = use_curriculum

        # 设备
        self.device = torch.device(
            f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cuda:0"
        )

        # 性能优化
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        get_logger().info(f"\n🚀 Initializing {method_name} (Batch Ensemble)")
        get_logger().info(f"   Members: {num_members} (implicit)")
        get_logger().info(f"   Device: {self.device}")

        # 创建模型
        self.model = BatchEnsembleResNet(
            layers=[2, 2, 2, 2],  # ResNet-18 配置
            num_classes=cfg.num_classes,
            num_members=num_members,
        ).to(self.device)

        # 可选编译
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # 优化器和调度器
        self.optimizer = create_optimizer(
            self.model, cfg.optimizer, cfg.lr, cfg.weight_decay
        )
        self.scheduler = create_scheduler(
            self.optimizer, cfg.scheduler, cfg.total_epochs
        )

        # 设置日志
        self.setup_logging()

        # 保存配置
        cfg.save()

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

        total_loss = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.total_epochs}",
        )

        for inputs, targets in iterator:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 模型输出: [num_members, batch_size, num_classes]
            logits = self.model(inputs)

            # 计算每个成员的 loss 并平均
            # targets 需要扩展: [num_members, batch_size]
            targets_expanded = targets.unsqueeze(0).expand(self.num_members, -1)

            # 重塑 logits: [num_members * batch_size, num_classes]
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets_expanded.reshape(-1)

            loss = criterion(logits_flat, targets_flat)

            loss.backward()

            # 梯度裁剪
            if self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            iterator.set_postfix({"loss": loss.item()})

        # 更新调度器
        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 模型输出: [num_members, batch_size, num_classes]
            logits = self.model(inputs)

            # 集成预测: 平均所有成员的 logits
            ensemble_logits = logits.mean(dim=0)  # [batch_size, num_classes]

            loss = criterion(ensemble_logits, targets)
            total_loss += loss.item()

            preds = ensemble_logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """执行完整训练"""
        self.logger.info("=" * 70)
        self.logger.info(f"🎯 Batch Ensemble Training: {self.name}")
        self.logger.info(f"   Members: {self.num_members}")
        self.logger.info("=" * 70)

        training_start = time.time()

        try:
            for epoch in range(self.cfg.total_epochs):
                epoch_start = time.time()

                # 训练
                train_loss = self._train_epoch(train_loader, epoch)

                # 验证
                val_loss, val_acc = self._validate(val_loader)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                # 使用基类方法记录 epoch 信息
                self._log_epoch(
                    epoch, train_loss, val_loss, val_acc, current_lr, epoch_time
                )

                # 使用基类方法检查最佳并保存
                self._check_best_and_save(val_loss, epoch)

                # 早停
                if self.early_stopping(val_loss, epoch):
                    self.logger.info(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                    break

            # 使用基类方法完成训练
            self._finalize_training(training_start)

        except KeyboardInterrupt:
            self._handle_interrupt()
            raise

    def _save_checkpoint(self, tag: str):
        """保存 checkpoint"""
        ckpt_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        ensure_dir(ckpt_dir)

        # 保存模型
        torch.save(self.model.state_dict(), ckpt_dir / "model.pth")

        # 保存状态
        state = {
            "epoch": len(self.history["epoch"]),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "num_members": self.num_members,
            "total_training_time": self.total_training_time,
        }
        torch.save(state, ckpt_dir / "trainer_state.pth")
        self.logger.info(f"💾 Saved checkpoint: {tag}")

    def load_checkpoint(self, tag: str = "best") -> bool:
        """加载 checkpoint"""
        ckpt_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        if not ckpt_dir.exists():
            self.logger.warning(f"⚠️ Checkpoint not found: {ckpt_dir}")
            return False

        self.model.load_state_dict(
            torch.load(ckpt_dir / "model.pth", weights_only=True)
        )

        state_path = ckpt_dir / "trainer_state.pth"
        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            self.best_val_loss = state["best_val_loss"]
            self.best_epoch = state["best_epoch"]
            self.history = state["history"]
            self.total_training_time = state.get("total_training_time", 0.0)
            self.logger.info(f"✅ Loaded checkpoint: {tag}")
            return True
        return False

    def get_models(self) -> List[nn.Module]:
        """
        获取模型列表 (用于评估兼容)

        Batch Ensemble 只有一个物理模型，但返回列表以兼容现有评估代码
        """
        return [self.model]


def train_batch_ensemble(
    experiment_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_members: int = 4,
) -> Tuple[BatchEnsembleTrainer, float]:
    """
    Batch Ensemble 训练入口函数

    Args:
        experiment_name: 实验名称
        cfg: 配置
        train_loader, val_loader: 数据加载器
        num_members: 集成成员数

    Returns:
        (trainer, training_time)
    """
    trainer = BatchEnsembleTrainer(
        method_name=experiment_name,
        cfg=cfg,
        num_members=num_members,
    )

    trainer.train(train_loader, val_loader)
    training_time = trainer.total_training_time

    # 加载最佳模型
    trainer.load_checkpoint("best")
    trainer.total_training_time = training_time

    get_logger().info(f"\n✅ Batch Ensemble training completed: {experiment_name}")

    return trainer, training_time
