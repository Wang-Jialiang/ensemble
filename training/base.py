"""
================================================================================
训练器基类模块
================================================================================

所有训练器共享的抽象基类，定义统一接口。
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..utils import ensure_dir, format_duration
from .scheduler import EarlyStopping
from .worker import HistorySaver


class BaseTrainer(ABC):
    """
    训练器抽象基类

    所有训练器必须实现的接口：
    - train(train_loader, val_loader): 执行完整训练
    - get_models() -> List[nn.Module]: 获取模型列表
    - load_checkpoint(tag) -> bool: 加载检查点
    - _save_checkpoint(tag): 保存检查点 (实现细节)

    共享属性：
    - name: 实验名称
    - cfg: 配置对象
    - total_training_time: 总训练时间
    - history: 训练历史
    - logger: 日志记录器
    """

    def __init__(self, method_name: str, cfg: Config):
        self.name = method_name
        self.cfg = cfg
        self.total_training_time = 0.0

        # 训练历史 (子类可以扩展)
        self.history: Dict[str, List] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

        # 早停
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience, mode="min"
        )
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # 历史保存器
        self.history_saver = HistorySaver(cfg.save_dir)

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if cfg.use_tensorboard:
            log_dir = Path(cfg.save_dir) / "tensorboard" / self.name
            ensure_dir(log_dir)
            self.writer = SummaryWriter(str(log_dir))

        # 日志 (延迟初始化)
        self.logger: Optional[logging.Logger] = None

    def setup_logging(self) -> None:
        """设置日志系统"""
        log_dir = Path(self.cfg.save_dir) / "logs"
        ensure_dir(log_dir)

        logger = logging.getLogger(self.name)
        logger.handlers.clear()
        logger.setLevel(getattr(logging, self.cfg.log_level))

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_dir / f"{self.name}_train.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger

    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """执行完整训练"""
        pass

    @abstractmethod
    def get_models(self) -> List[nn.Module]:
        """获取模型列表用于评估"""
        pass

    @abstractmethod
    def load_checkpoint(self, tag: str = "best") -> bool:
        """加载检查点"""
        pass

    @abstractmethod
    def _save_checkpoint(self, tag: str) -> None:
        """保存检查点"""
        pass

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
        extra_info: str = "",
    ) -> None:
        """记录 epoch 信息到历史和 TensorBoard"""
        # 记录历史
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)
        self.history["epoch_time"].append(epoch_time)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Hyperparameters/lr", lr, epoch)

        # 控制台日志
        if self.logger:
            self.logger.info(
                f"Epoch {epoch + 1:3d}/{self.cfg.total_epochs} | "
                f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"ValAcc: {val_acc:.2f}% | LR: {lr:.6f} | Time: {epoch_time:.1f}s"
                f"{extra_info}"
            )

    def _check_best_and_save(self, val_loss: float, epoch: int) -> bool:
        """检查是否为最佳模型并保存"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint("best")
            if self.logger:
                self.logger.info(f"   🏆 New best! Val Loss: {val_loss:.4f}")
            return True
        return False

    def _finalize_training(self, training_start_time: float) -> None:
        """训练结束后的清理工作"""
        self.total_training_time = time.time() - training_start_time

        if self.logger:
            self.logger.info(
                f"\n⏱️ Total Time: {format_duration(self.total_training_time)}"
            )

        self._save_checkpoint("final")
        self.history_saver.save(self.history)

        if self.logger:
            self.logger.info(f"\n✅ Training completed: {self.name}")

        if self.writer:
            self.writer.close()

    def _handle_interrupt(self) -> None:
        """处理用户中断"""
        if self.logger:
            self.logger.info("\n⚠️ Interrupted by user")
        self._save_checkpoint("interrupted")
        self.history_saver.save(self.history)
