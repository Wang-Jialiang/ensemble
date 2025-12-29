"""
================================================================================
训练核心模块
================================================================================

StagedEnsembleTrainer (三阶段集成训练器)、train_experiment (实验入口函数)
"""

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Config
from ..utils import ensure_dir, format_duration, get_logger
from .optimization import EarlyStopping
from .worker import GPUWorker, HistorySaver

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Checkpoint Mixin                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointMixin:
    """检查点管理 Mixin

    提供 checkpoint 保存、加载、清理功能。
    需要子类提供: cfg, name, workers, history, best_val_loss, best_epoch,
                  early_stopping, total_training_time, augmentation_method,
                  use_curriculum, fixed_ratio, fixed_prob, logger
    """

    def _save_checkpoint(self, tag: str):
        """保存checkpoint"""
        checkpoint_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        ensure_dir(checkpoint_dir)

        for worker in self.workers:
            worker.save_models(str(checkpoint_dir), self.name)

        state = {
            "epoch": len(self.history["epoch"]),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "early_stopping_counter": self.early_stopping.counter,
            "total_training_time": self.total_training_time,
            "augmentation_method": self.augmentation_method,
            "use_curriculum": self.use_curriculum,
            "fixed_ratio": self.fixed_ratio,
            "fixed_prob": self.fixed_prob,
        }
        torch.save(state, checkpoint_dir / "trainer_state.pth")
        self.logger.info(f"💾 Saved checkpoint: {tag}")

    def load_checkpoint(self, tag: str = "best") -> bool:
        """加载checkpoint"""
        checkpoint_dir = Path(self.cfg.save_dir) / "checkpoints" / self.name / tag
        if not checkpoint_dir.exists():
            self.logger.warning(f"⚠️ Checkpoint not found: {checkpoint_dir}")
            return False

        for worker in self.workers:
            worker.load_models(str(checkpoint_dir), self.name)

        state_path = checkpoint_dir / "trainer_state.pth"
        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            self.best_val_loss = state["best_val_loss"]
            self.best_epoch = state["best_epoch"]
            self.history = state["history"]
            self.early_stopping.counter = state.get("early_stopping_counter", 0)
            self.total_training_time = state.get("total_training_time", 0.0)
            self.augmentation_method = state.get(
                "augmentation_method", self.augmentation_method
            )
            self.use_curriculum = state.get("use_curriculum", self.use_curriculum)
            self.fixed_ratio = state.get("fixed_ratio", self.fixed_ratio)
            self.fixed_prob = state.get("fixed_prob", self.fixed_prob)
            self.logger.info(f"✅ Loaded checkpoint: {tag}")
            self.logger.info(
                f"   Augmentation: {self.augmentation_method}, Curriculum: {self.use_curriculum}"
            )
            return True
        return False

    def _cleanup_old_checkpoints(self):
        """清理旧checkpoint"""
        checkpoint_base = Path(self.cfg.save_dir) / "checkpoints" / self.name
        if not checkpoint_base.exists():
            return

        epoch_dirs = [
            d for d in checkpoint_base.iterdir() if d.name.startswith("epoch_")
        ]
        epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]))

        if len(epoch_dirs) > self.cfg.keep_last_n_checkpoints:
            for old_dir in epoch_dirs[: -self.cfg.keep_last_n_checkpoints]:
                shutil.rmtree(old_dir)
                self.logger.info(f"🗑️ Removed old checkpoint: {old_dir.name}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 三阶段集成训练器                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class StagedEnsembleTrainer(CheckpointMixin):
    """
    三阶段集成训练器 (支持多种数据增强方法)

    实现三阶段课程学习策略训练深度集成模型:

    阶段划分:
        1. Warmup阶段: 无遮挡热身训练，让模型学习基础特征
        2. Progressive阶段: 渐进式增加遮挡，培养模型关注不同区域
        3. Finetune阶段: 固定遮挡比例微调，稳定模型性能
    """

    def __init__(
        self,
        method_name: str,
        cfg: Config,
        augmentation_method: str = "perlin",
        use_curriculum: bool = True,
        fixed_ratio: float = 0.25,
        fixed_prob: float = 0.5,
        share_warmup_backbone: bool = False,
    ):
        self.name = method_name
        self.cfg = cfg
        self.total_training_time = 0.0

        # 增强配置
        self.augmentation_method = augmentation_method
        self.use_curriculum = use_curriculum
        self.fixed_ratio = fixed_ratio
        self.fixed_prob = fixed_prob
        self.share_warmup_backbone = share_warmup_backbone

        # 性能优化设置
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        get_logger().info(f"\n🚀 Initializing {method_name}")
        get_logger().info(f"   Augmentation: {augmentation_method}")
        get_logger().info(f"   Curriculum: {'Yes' if use_curriculum else 'No'}")
        get_logger().info(
            f"   Config: {cfg.total_models} {cfg.model_name} models across {len(cfg.gpu_ids)} GPUs"
        )

        # 创建Workers
        self.workers: List[GPUWorker] = []
        for gpu_id in cfg.gpu_ids:
            worker = GPUWorker(gpu_id, cfg.num_models_per_gpu, cfg, augmentation_method)
            self.workers.append(worker)

        # 日志系统
        self.setup_logging()

        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard:
            log_dir = Path(cfg.save_dir) / "tensorboard" / self.name
            self.writer = SummaryWriter(str(log_dir))
            get_logger().info(f"📊 TensorBoard logging to: {log_dir}")

        # 训练历史
        self.history = {
            "epoch": [],
            "stage": [],
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "mask_ratio": [],
            "mask_prob": [],
            "lr": [],
            "epoch_time": [],
        }

        # 早停
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience, mode="min"
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # 指标计算器和历史保存器
        self.history_saver = HistorySaver(cfg.save_dir)

        # 保存配置
        cfg.save()

    def get_models(self) -> List[nn.Module]:
        """获取所有模型列表 (与其他 Trainer 接口一致)"""
        return [model for worker in self.workers for model in worker.models]

    def setup_logging(self):
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

    def _get_stage_info(self, epoch: int) -> Tuple[int, str, float, float, bool]:
        """获取当前阶段信息

        Returns:
            Tuple: (stage_num, stage_name, mask_ratio, mask_prob, use_mask)
        """
        cfg = self.cfg

        # 模式1: 无增强 (Baseline)
        if self.augmentation_method == "none":
            return 1, "NoAug", 0.0, 0.0, False

        # 模式2: 固定参数模式
        if not self.use_curriculum:
            return 1, "Fixed", self.fixed_ratio, self.fixed_prob, True

        # 模式3: 课程学习模式 (三阶段)
        if epoch < cfg.warmup_epochs:
            return 1, "Warmup", 0.0, 0.0, False
        elif epoch < cfg.warmup_epochs + cfg.progressive_epochs:
            progress = (epoch - cfg.warmup_epochs) / cfg.progressive_epochs
            mask_ratio = (
                cfg.mask_start_ratio
                + (cfg.mask_end_ratio - cfg.mask_start_ratio) * progress
            )
            mask_prob = (
                cfg.mask_prob_start
                + (cfg.mask_prob_end - cfg.mask_prob_start) * progress
            )
            return 2, "Progressive", mask_ratio, mask_prob, True
        else:
            return 3, "Finetune", cfg.finetune_mask_ratio, cfg.finetune_mask_prob, True

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        stage_num, stage_name, mask_ratio, mask_prob, use_mask = self._get_stage_info(
            epoch
        )

        # 预计算mask（如果需要）
        for worker in self.workers:
            worker.precompute_masks(self.cfg.mask_pool_size, mask_ratio)

        total_loss = 0.0
        num_batches = 0
        current_lr = self.workers[0].get_lr()
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.total_epochs} [{stage_name}] lr={current_lr:.6f}",
        )

        for inputs, targets in iterator:
            # 异步训练
            for worker in self.workers:
                worker.train_batch_async(
                    inputs, targets, criterion, mask_ratio, mask_prob, use_mask
                )

            # 同步并累计loss
            batch_loss = 0.0
            for worker in self.workers:
                batch_loss += worker.synchronize()

            total_loss += batch_loss / len(self.workers)
            num_batches += 1

            iterator.set_postfix({"loss": total_loss / num_batches})

        # 更新学习率调度器（scheduler会基于缩放后的lr继续调整）
        for worker in self.workers:
            worker.step_schedulers()

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证"""
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用第一个worker的设备作为主设备进行计算
        primary_device = self.workers[0].device

        for inputs, targets in val_loader:
            all_logits = []
            for worker in self.workers:
                worker_logits = worker.predict_batch(inputs)
                all_logits.append(worker_logits.to(primary_device))

            all_logits = torch.cat(all_logits, dim=0)
            ensemble_logits = all_logits.mean(dim=0)

            # 确保targets也在主设备上
            targets = targets.to(primary_device)

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
        self.logger.info(f"🎓 Three-Stage Curriculum Learning: {self.name}")
        self.logger.info("=" * 70)

        current_stage = 0
        training_start_time = time.time()

        try:
            for epoch in range(self.cfg.total_epochs):
                epoch_start_time = time.time()
                stage_num, stage_name, mask_ratio, mask_prob, use_mask = (
                    self._get_stage_info(epoch)
                )

                # 阶段切换提示
                if stage_num != current_stage:
                    # 共享 backbone: 在从 Stage 1 切换到 Stage 2 时广播
                    if (
                        stage_num == 2
                        and current_stage == 1
                        and self.share_warmup_backbone
                    ):
                        self._broadcast_warmup_backbone()

                    current_stage = stage_num
                    self.logger.info("")
                    self.logger.info("=" * 70)
                    if stage_num == 1:
                        self.logger.info("🔥 STAGE 1: WARMUP (No Mask)")
                    elif stage_num == 2:
                        self.logger.info("🎭 STAGE 2: PROGRESSIVE MASKING")
                    else:
                        self.logger.info("🎯 STAGE 3: FINE-TUNING")
                    self.logger.info("=" * 70)

                # 训练
                try:
                    train_loss = self._train_epoch(train_loader, epoch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.error("❌ GPU Out of Memory!")
                        torch.cuda.empty_cache()
                        raise
                    else:
                        raise

                # 验证
                val_loss, val_acc = self._validate(val_loader)

                epoch_elapsed = time.time() - epoch_start_time
                current_lr = self.workers[0].optimizers[0].param_groups[0]["lr"]

                # 记录历史
                self.history["epoch"].append(epoch + 1)
                self.history["stage"].append(stage_num)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["mask_ratio"].append(mask_ratio)
                self.history["mask_prob"].append(mask_prob)
                self.history["lr"].append(current_lr)
                self.history["epoch_time"].append(epoch_elapsed)

                # TensorBoard
                if self.writer:
                    self.writer.add_scalar("Loss/train", train_loss, epoch)
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                    self.writer.add_scalar("Hyperparameters/lr", current_lr, epoch)
                    self.writer.add_scalar(
                        "Time/epoch_duration_sec", epoch_elapsed, epoch
                    )

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self._save_checkpoint("best")
                    self.logger.info(f"   🏆 New best model! Val Loss: {val_loss:.4f}")

                # 定期保存
                if (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch + 1}")
                    self._cleanup_old_checkpoints()

                # 日志
                mask_info = (
                    f"MaskR={mask_ratio:.1%}, MaskP={mask_prob:.1%}"
                    if use_mask
                    else "NoMask"
                )
                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{self.cfg.total_epochs} [{stage_name:11s}] | "
                    f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}% | "
                    f"{mask_info} | LR: {current_lr:.6f} | Time: {epoch_elapsed:.1f}s"
                )

                # 早停检查
                if self.early_stopping(val_loss, epoch):
                    self.logger.info(
                        f"\n⚠️ Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

            self.total_training_time = time.time() - training_start_time
            self.logger.info(
                f"\n⏱️ Total Training Time: {format_duration(self.total_training_time)}"
            )

            self._save_checkpoint("final")
            self.history_saver.save(self.history)
            self.logger.info(f"\n✅ Training completed: {self.name}")

        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Training interrupted by user")
            self.total_training_time = time.time() - training_start_time
            self._save_checkpoint("interrupted")
            self.history_saver.save(self.history)
            raise
        except Exception as e:
            self.logger.error(f"\n❌ Training failed with error: {e}")
            self._save_checkpoint("error")
            self.history_saver.save(self.history)
            raise
        finally:
            if self.writer:
                self.writer.close()

    def _broadcast_warmup_backbone(self):
        """从第一个模型获取 backbone，广播到所有子模型并重新初始化各自的 classifier head"""
        # 使用第一个 worker 的第一个模型作为源
        source_model = self.workers[0].models[0]
        backbone_state = source_model.get_backbone_state_dict()

        for worker in self.workers:
            worker.broadcast_backbone_and_reinit_heads(backbone_state)

        self.logger.info(
            "🔄 Shared warmup backbone to all models, re-initialized classifier heads"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 实验运行函数                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def train_experiment(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple["StagedEnsembleTrainer", float]:
    """
    仅训练实验 (不包含评估)

    所有增强参数从 cfg 读取:
    - cfg.experiment_name: 实验名称
    - cfg.augmentation_method: 增强方法
    - cfg.use_curriculum: 是否使用课程学习
    - cfg.fixed_ratio, cfg.fixed_prob: 固定遮挡参数
    - cfg.share_warmup_backbone: 是否共享 backbone

    参数:
        cfg: 配置对象 (包含所有实验参数)
        train_loader, val_loader: 数据加载器

    返回:
        (trainer, training_time)
    """
    trainer = StagedEnsembleTrainer(
        cfg.experiment_name,
        cfg,
        augmentation_method=cfg.augmentation_method,
        use_curriculum=cfg.use_curriculum,
        fixed_ratio=cfg.fixed_ratio,
        fixed_prob=cfg.fixed_prob,
        share_warmup_backbone=cfg.share_warmup_backbone,
    )

    # 训练
    trainer.train(train_loader, val_loader)
    training_time = trainer.total_training_time

    # 加载最佳模型
    trainer.load_checkpoint("best")
    trainer.total_training_time = training_time

    get_logger().info(f"\n✅ Training completed: {cfg.experiment_name}")
    get_logger().info(
        f"   Checkpoint saved to: {Path(cfg.save_dir) / 'checkpoints' / cfg.experiment_name}"
    )

    return trainer, training_time
