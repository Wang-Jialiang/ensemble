"""
================================================================================
Snapshot Ensemble 训练器
================================================================================

Snapshot Ensemble 实现 (Huang et al., 2017)

核心思想: 使用余弦退火 + 周期性热重启，在每个周期末尾保存模型快照。
- 单次训练获得多个模型
- 利用周期性学习率调度产生多样性
- 无额外训练成本

参考: https://arxiv.org/abs/1704.00109
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models import ModelFactory
from ..utils import ensure_dir, get_logger
from .base import BaseTrainer
from .scheduler import create_optimizer


class SnapshotEnsembleTrainer(BaseTrainer):
    """
    Snapshot Ensemble 训练器

    使用 CosineAnnealingWarmRestarts 调度器，在每个周期结束时保存快照。
    """

    def __init__(
        self,
        method_name: str,
        cfg: Config,
        num_cycles: int = 5,
    ):
        # 调用基类初始化
        super().__init__(method_name, cfg)

        self.num_cycles = num_cycles

        # 计算每个周期的 epoch 数
        self.epochs_per_cycle = cfg.total_epochs // num_cycles
        if self.epochs_per_cycle < 1:
            self.epochs_per_cycle = 1
            self.num_cycles = cfg.total_epochs

        # 设备
        self.device = torch.device(
            f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cuda:0"
        )

        # 性能优化
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        get_logger().info(f"\n🚀 Initializing {method_name} (Snapshot Ensemble)")
        get_logger().info(f"   Cycles: {self.num_cycles}")
        get_logger().info(f"   Epochs per cycle: {self.epochs_per_cycle}")
        get_logger().info(f"   Device: {self.device}")

        # 创建单个模型
        self.model = ModelFactory.create_model(
            cfg.model_name,
            num_classes=cfg.num_classes,
            init_method=cfg.init_method,
        ).to(self.device)

        # 可选编译
        if cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # 优化器 (使用公共工厂函数)
        self.optimizer = create_optimizer(
            self.model, cfg.optimizer, cfg.lr, cfg.weight_decay
        )

        # 使用 CosineAnnealingWarmRestarts 调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.epochs_per_cycle,
            T_mult=1,  # 每个周期长度相同
        )

        # 保存的快照
        self.snapshots: List[Dict] = []

        # 扩展 history 添加 cycle 字段
        self.history["cycle"] = []

        # 设置日志
        self.setup_logging()

        # 保存配置
        cfg.save()

    # 注: _create_optimizer 已移除，使用 scheduler.create_optimizer() 替代

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
            logits = self.model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            iterator.set_postfix({"loss": loss.item()})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证 (使用当前模型)"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    @torch.no_grad()
    def _validate_ensemble(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证 (使用所有快照集成)"""
        if not self.snapshots:
            return self._validate(val_loader)

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 收集所有快照的预测
            all_logits = []
            for snapshot in self.snapshots:
                self.model.load_state_dict(snapshot)
                self.model.eval()
                logits = self.model(inputs)
                all_logits.append(logits)

            # 集成预测
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            loss = criterion(ensemble_logits, targets)
            total_loss += loss.item()

            preds = ensemble_logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    def _save_snapshot(self, cycle: int):
        """保存当前模型快照"""
        snapshot = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.snapshots.append(snapshot)
        self.logger.info(f"📸 Saved snapshot {cycle + 1}/{self.num_cycles}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """执行完整训练"""
        self.logger.info("=" * 70)
        self.logger.info(f"🎯 Snapshot Ensemble Training: {self.name}")
        self.logger.info(
            f"   Cycles: {self.num_cycles}, Epochs/Cycle: {self.epochs_per_cycle}"
        )
        self.logger.info("=" * 70)

        training_start = time.time()
        current_cycle = 0

        try:
            for epoch in range(self.cfg.total_epochs):
                epoch_start = time.time()

                # 判断当前周期
                cycle = epoch // self.epochs_per_cycle
                is_cycle_end = (epoch + 1) % self.epochs_per_cycle == 0

                # 周期开始提示
                if cycle != current_cycle:
                    current_cycle = cycle
                    self.logger.info(
                        f"\n🔄 Starting Cycle {cycle + 1}/{self.num_cycles}"
                    )

                # 训练
                train_loss = self._train_epoch(train_loader, epoch)

                # 更新学习率
                self.scheduler.step()

                # 验证
                val_loss, val_acc = self._validate(val_loader)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                # 使用基类方法记录 epoch (同时记录 cycle)
                extra_info = f" [Cycle {cycle + 1}]"
                self._log_epoch(
                    epoch,
                    train_loss,
                    val_loss,
                    val_acc,
                    current_lr,
                    epoch_time,
                    extra_info,
                )
                self.history["cycle"].append(cycle + 1)

                # 周期结束时保存快照
                if is_cycle_end and cycle < self.num_cycles:
                    self._save_snapshot(cycle)
                    self._save_checkpoint(f"snapshot_{cycle + 1}")

                # 使用基类方法检查最佳
                self._check_best_and_save(val_loss, epoch)

            # 最终评估集成效果
            ens_loss, ens_acc = self._validate_ensemble(val_loader)
            self.logger.info(f"\n📊 Ensemble ({len(self.snapshots)} snapshots):")
            self.logger.info(f"   Val Loss: {ens_loss:.4f}, Val Acc: {ens_acc:.2f}%")

            # 使用基类方法完成训练
            self._finalize_training(training_start)

        except KeyboardInterrupt:
            # 使用基类方法处理中断
            self._handle_interrupt()
            raise

    def _save_checkpoint(self, tag: str):
        """保存 checkpoint"""
        ckpt_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        ensure_dir(ckpt_dir)

        # 保存当前模型
        torch.save(self.model.state_dict(), ckpt_dir / "model.pth")

        # 保存所有快照
        for i, snap in enumerate(self.snapshots):
            torch.save(snap, ckpt_dir / f"snapshot_{i}.pth")

        # 保存状态
        state = {
            "num_snapshots": len(self.snapshots),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "num_cycles": self.num_cycles,
            "total_training_time": self.total_training_time,
        }
        torch.save(state, ckpt_dir / "trainer_state.pth")
        self.logger.info(f"💾 Saved checkpoint: {tag}")

    def load_checkpoint(self, tag: str = "final") -> bool:
        """加载 checkpoint"""
        ckpt_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        if not ckpt_dir.exists():
            self.logger.warning(f"⚠️ Checkpoint not found: {ckpt_dir}")
            return False

        # 加载状态
        state_path = ckpt_dir / "trainer_state.pth"
        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            self.best_val_loss = state["best_val_loss"]
            self.history = state["history"]
            self.total_training_time = state.get("total_training_time", 0.0)

            # 加载快照
            self.snapshots = []
            for i in range(state["num_snapshots"]):
                snap_path = ckpt_dir / f"snapshot_{i}.pth"
                if snap_path.exists():
                    self.snapshots.append(torch.load(snap_path, weights_only=True))

            self.logger.info(f"✅ Loaded {len(self.snapshots)} snapshots from: {tag}")
            return True
        return False

    def get_models(self) -> List[nn.Module]:
        """
        获取所有快照模型 (用于评估)

        返回多个独立模型实例，每个加载不同快照
        """
        models = []
        for snapshot in self.snapshots:
            model = ModelFactory.create_model(
                self.cfg.model_name,
                num_classes=self.cfg.num_classes,
            ).to(self.device)
            model.load_state_dict(snapshot)
            models.append(model)
        return models


def train_snapshot_ensemble(
    experiment_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_cycles: int = 5,
) -> Tuple[SnapshotEnsembleTrainer, float]:
    """
    Snapshot Ensemble 训练入口函数

    Args:
        experiment_name: 实验名称
        cfg: 配置
        train_loader, val_loader: 数据加载器
        num_cycles: 周期数

    Returns:
        (trainer, training_time)
    """
    trainer = SnapshotEnsembleTrainer(
        method_name=experiment_name,
        cfg=cfg,
        num_cycles=num_cycles,
    )

    trainer.train(train_loader, val_loader)
    training_time = trainer.total_training_time

    # 加载最终快照
    trainer.load_checkpoint("final")
    trainer.total_training_time = training_time

    get_logger().info(f"\n✅ Snapshot Ensemble completed: {experiment_name}")

    return trainer, training_time
