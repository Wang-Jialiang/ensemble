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

from ..config import Config
from ..utils import console, ensure_dir, format_duration, get_logger
from .optimization import EarlyStopping
from .worker import GPUWorker, HistorySaver

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Checkpoint Mixin                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CheckpointMixin:
    """检查点管理 Mixin (大纲化)"""

    def _save_checkpoint(self, tag: str):
        """保存模型权重"""
        path = self._get_checkpoint_dir(tag)
        ensure_dir(path)

        for worker in self.workers:
            worker.save_models(str(path), self.name)
        self.logger.info(f"💾 Checkpoint Saved: {tag}")

    def load_checkpoint(self, tag: str = "best") -> bool:
        """从指定 tag 加载模型权重"""
        path = self._get_checkpoint_dir(tag)
        if not path.exists():
            self.logger.warning(f"⚠️ Checkpoint 不存在: {path}")
            return False

        for worker in self.workers:
            worker.load_models(str(path), self.name)
        return True

    def _get_checkpoint_dir(self, tag: str) -> Path:
        """统一路径生成逻辑 (save_dir 已包含实验名)"""
        return Path(self.cfg.save_dir) / "checkpoints" / tag


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
        method_name,
        cfg,
        augmentation_method="perlin",
        use_curriculum=True,
        fixed_ratio=0.25,
        share_warmup_backbone=False,
    ):
        """三阶段集成训练器构造函数 (大纲化)"""
        self.name = method_name
        self.cfg = cfg
        self.total_training_time = 0.0

        # 1. 初始化属性与增强策略
        self.augmentation_method = augmentation_method
        self.use_curriculum = use_curriculum
        self.fixed_ratio = fixed_ratio
        self.share_warmup_backbone = share_warmup_backbone

        # 2. 硬件与日志初始化
        self._init_hardware_optimizations()
        self.setup_logging()
        self._init_monitoring_tools()

        # 3. 初始化工作节点 (Parallel Workers)
        self.workers: List[GPUWorker] = [
            GPUWorker(gid, cfg.num_models_per_gpu, cfg, augmentation_method)
            for gid in cfg.gpu_ids
        ]

        # 4. 初始化状态跟踪变量
        self._init_tracking_structures()

    def _init_hardware_optimizations(self):
        """配置 Cuda 后端加速选项"""
        if self.cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def _init_monitoring_tools(self):
        """初始化 wandb 观测工具"""

        # Weights & Biases
        self.wandb_run = None
        if getattr(self.cfg, "use_wandb", False):
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=getattr(self.cfg, "wandb_project", "ensemble"),
                    name=self.name,
                    config=self._get_wandb_config(),
                    mode="online",
                    reinit="finish_previous",
                    save_code=False,
                    settings=wandb.Settings(silent=True),
                )
            except ImportError:
                self.logger.warning("⚠️ wandb not installed, skipping")
            except Exception as e:
                self.logger.warning(f"⚠️ wandb init failed: {e}")

    def _get_wandb_config(self) -> dict:
        """提取配置用于 wandb"""
        return {
            "model": self.cfg.model_name,
            "dataset": self.cfg.dataset_name,
            "batch_size": self.cfg.batch_size,
            "lr": self.cfg.lr,
            "optimizer": self.cfg.optimizer,
            "augmentation": self.augmentation_method,
            "use_curriculum": self.use_curriculum,
            "total_epochs": self.cfg.total_epochs,
        }

    def _init_tracking_structures(self):
        """初始化训练历史、早停与记录器"""
        self.history = {
            k: []
            for k in [
                "epoch",
                "stage",
                "train_loss",
                "val_loss",
                "val_acc",
                "mask_ratio",
                "mask_prob",
                "lr",
                "time",
            ]
        }
        self.early_stopping = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            metrics={"val_loss": "min", "val_acc": "max"},
            criteria="any",
        )
        self._best_val_loss = float("inf")
        self.history_saver = HistorySaver(self.cfg.training_base_dir)

    def get_models(self) -> List[nn.Module]:
        """获取所有模型列表 (与其他 Trainer 接口一致)"""
        return [model for worker in self.workers for model in worker.models]

    def setup_logging(self):
        """设置日志系统"""
        logger = logging.getLogger(self.name)
        logger.handlers.clear()
        logger.setLevel(getattr(logging, self.cfg.log_level))

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # 文件输出 (放在时间戳目录下，文件名包含实验名)
        log_path = Path(self.cfg.training_base_dir) / f"{self.name}_train.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台输出 (可通过配置关闭)
        if getattr(self.cfg, "log_to_console", False):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self.logger = logger

    def _get_mask_prob_by_epoch(self, epoch: int) -> float:
        """计算给定 epoch 的遮罩概率 (基于三阶段)

        策略 (与 warmup/progressive/finetune_epochs 共用):
            - Warmup 阶段: mask_prob = 0 (不使用遮罩)
            - Progressive 阶段: mask_prob 从 mask_start_prob 线性增加到 mask_end_prob
            - Finetune 阶段: mask_prob = mask_end_prob (固定)
        """
        cfg = self.cfg
        start_prob = cfg.mask_start_prob  # 0.0
        end_prob = cfg.mask_end_prob  # 0.8

        if epoch < cfg.warmup_epochs:
            # Warmup 阶段: 概率为 0
            return 0.0
        elif epoch < cfg.warmup_epochs + cfg.progressive_epochs:
            # Progressive 阶段: 概率从 start_prob 线性增加到 end_prob
            prog_epoch = epoch - cfg.warmup_epochs
            progress = prog_epoch / max(cfg.progressive_epochs - 1, 1)
            return start_prob + (end_prob - start_prob) * progress
        else:
            # Finetune 阶段: 概率保持 end_prob
            return end_prob

    def _get_stage_info(self, epoch: int) -> Tuple[int, str, float, float, bool]:
        """获取当前阶段信息

        Returns:
            Tuple: (stage_num, stage_name, mask_ratio, mask_prob, use_mask)
        """
        cfg = self.cfg

        # 模式1: 无增强 (Baseline)
        if self.augmentation_method == "none":
            return 1, "NoAug", 0.0, 0.0, False

        # 计算统一概率 (所有模式共用)
        mask_prob = self._get_mask_prob_by_epoch(epoch)

        # 模式2: 固定参数模式
        if not self.use_curriculum:
            return 1, "Fixed", self.fixed_ratio, mask_prob, True

        # 模式3: 课程学习模式 (三阶段)
        if epoch < cfg.warmup_epochs:
            # Warmup 阶段: 不使用遮罩
            return 1, "Warmup", 0.0, 0.0, False
        elif epoch < cfg.warmup_epochs + cfg.progressive_epochs:
            # Progressive 阶段: ratio 线性增长，prob 使用统一策略
            progress = (epoch - cfg.warmup_epochs) / max(cfg.progressive_epochs - 1, 1)
            mask_ratio = (
                cfg.mask_start_ratio
                + (cfg.mask_end_ratio - cfg.mask_start_ratio) * progress
            )
            return 2, "Progressive", mask_ratio, mask_prob, True
        else:
            # Finetune 阶段: ratio 固定，prob 使用统一策略
            return 3, "Finetune", cfg.finetune_mask_ratio, mask_prob, True

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练单个 Epoch (大纲化)

        Returns:
            Tuple[float, float]: (train_loss, current_lr) - 本 epoch 的损失和使用的学习率
        """
        # 1. 准备当前阶段参数
        *_, m_ratio, m_prob, use_mask = self._get_stage_info(epoch)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

        # 2. 记录本 epoch 使用的 LR（在 step 之前）
        current_lr = self.workers[0].get_lr()

        # 3. 预热 Workers (如预计算 Mask 池)
        for w in self.workers:
            w.precompute_masks(m_ratio)

        # 4. 执行批次迭代
        train_loss = self._run_batch_iteration(
            train_loader, epoch, criterion, m_ratio, m_prob, use_mask
        )
        return train_loss, current_lr

    def _run_batch_iteration(self, loader, epoch, criterion, m_ratio, m_prob, use_mask):
        """具体执行张量流动与梯度更新 (使用 Rich Progress)"""
        total_loss, n = 0.0, 0
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        # 判断是否为 Warmup 单模型训练模式
        stage_num = self._get_stage_info(epoch)[0]
        is_warmup_single_model = self.share_warmup_backbone and stage_num == 1

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,  # 完成后自动消失
        ) as progress:
            desc = f"Epoch {epoch + 1:3d}"
            if is_warmup_single_model:
                desc += " [Warmup-SingleModel]"
            task_id = progress.add_task(desc, total=len(loader))

            for inputs, targets in loader:
                if is_warmup_single_model:
                    # Warmup 优化：仅训练第一个 Worker 的第一个模型
                    self.workers[0].train_batch_async(
                        inputs,
                        targets,
                        criterion,
                        m_ratio,
                        m_prob,
                        use_mask,
                        model_indices=[0],
                    )
                    batch_loss = self.workers[0].synchronize()
                else:
                    # 正常模式：所有 Worker 所有模型
                    for w in self.workers:
                        w.train_batch_async(
                            inputs, targets, criterion, m_ratio, m_prob, use_mask
                        )
                    batch_loss = sum(w.synchronize() for w in self.workers) / len(
                        self.workers
                    )

                total_loss += batch_loss
                n += 1
                progress.update(task_id, advance=1)

        # 步进调度器 (所有模型，保持 LR 同步)
        for w in self.workers:
            w.step_schedulers()
        return total_loss / n

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """集成验证过程"""
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        device = self.workers[0].device  # 主计算设备

        for inputs, targets in val_loader:
            # 1. 聚合所有 Worker 的预测 Logits
            ensemble_logits = self._collect_ensemble_logits(inputs, device)
            targets = targets.to(device)

            # 2. 计算指标
            total_loss += criterion(ensemble_logits, targets).item()
            correct += (ensemble_logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    def _collect_ensemble_logits(self, inputs, device):
        """从分布式 Workers 中收集并聚合预测结果"""
        from ..evaluation.strategies import get_ensemble_fn  # 延迟导入，避免循环依赖

        # 每个 worker 返回 [num_models, batch, classes]，concat 成 [total_models, batch, classes]
        logits_list = [w.predict_batch(inputs).to(device) for w in self.workers]
        stacked = torch.cat(logits_list, dim=0)  # [total_models, batch, classes]
        ensemble_fn = get_ensemble_fn(self.cfg)
        return ensemble_fn(stacked)  # [batch, classes]

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """执行全生命周期训练 (大纲化)"""
        self._log_training_start()
        self._train_start_time = time.time()  # 保存为实例变量供 checkpoint 使用
        current_stage = 0

        try:
            for epoch in range(self.cfg.total_epochs):
                # 1. 周期准备 (阶段切换)
                current_stage = self._handle_epoch_prep(epoch, current_stage)

                # 2. 执行训练与验证循环
                stats = self._run_epoch_cycle(train_loader, val_loader, epoch)
                stats["stage"] = current_stage

                # 3. 生命周期钩子: 记录、持久化、早停
                self._handle_epoch_post(epoch, stats)
                # 早停在 warmup + progressive 阶段后生效 (所有模式统一)
                finetune_start = self.cfg.warmup_epochs + self.cfg.progressive_epochs
                if epoch >= finetune_start and self.early_stopping(stats):
                    break

            self._finalize_training()

        except Exception as e:
            self._handle_training_error(e)
            raise
        finally:
            if self.wandb_run:
                import wandb

                wandb.finish()
                # 清理 wandb 本地缓存目录
                wandb_dir = Path.cwd() / "wandb"
                if wandb_dir.exists():
                    shutil.rmtree(wandb_dir, ignore_errors=False)

    def _handle_epoch_prep(self, epoch, current_stage):
        """处理 Epoch 开始前的预备动作 (如阶段切换)"""
        s_num, s_name, *_ = self._get_stage_info(epoch)

        # 统一 backbone 共享逻辑：在 warmup_epochs 结束后的第一个 epoch 触发
        # 无论是课程学习模式还是 Fixed 模式都适用
        if epoch == self.cfg.warmup_epochs and self.share_warmup_backbone:
            self._broadcast_warmup_backbone()

        # 阶段切换日志
        if s_num != current_stage:
            self._log_stage_header(s_num)
        return s_num

    def _run_epoch_cycle(self, train_loader, val_loader, epoch):
        """执行单个 Epoch 的计算循环并收集指标"""
        t0 = time.time()
        t_loss, current_lr = self._train_epoch(train_loader, epoch)
        v_loss, v_acc = self._validate(val_loader)

        # 获取当前元数据
        _, _, m_ratio, m_prob, _ = self._get_stage_info(epoch)
        return {
            "train_loss": t_loss,
            "val_loss": v_loss,
            "val_acc": v_acc,
            "mask_ratio": m_ratio,
            "mask_prob": m_prob,
            "lr": current_lr,  # 使用本 epoch 实际使用的 LR
            "time": time.time() - t0,
        }

    def _handle_epoch_post(self, epoch, stats):
        """处理 Epoch 结束后的辅助动作 (日志、快照)"""
        # 1. 记录历史
        self._record_metrics(epoch, stats)

        # 2. 保存最佳模型 (基于 loss)
        if stats["val_loss"] < self._best_val_loss:
            self._best_val_loss = stats["val_loss"]
            self._save_checkpoint("best")

        # 3. 打印汇总日志
        self._log_epoch_summary(epoch, stats)

    def _log_training_start(self):
        self.logger.info(
            "=" * 70 + f"\n🎓 Staged Training Start: {self.name}\n" + "=" * 70
        )

    def _log_stage_header(self, num):
        titles = {
            1: "STAGE 1: WARMUP",
            2: "STAGE 2: PROGRESSIVE",
            3: "STAGE 3: FINETUNE",
        }
        self.logger.info(f"\n{'=' * 20} {titles.get(num, 'UNKNOWN')} {'=' * 20}")

    def _record_metrics(self, epoch, stats):
        """同步历史记录与可视化工具"""
        for k, v in stats.items():
            self.history[k].append(v)
        self.history["epoch"].append(epoch + 1)

        # wandb
        if self.wandb_run:
            import wandb

            wandb.log(
                {
                    "train_loss": stats["train_loss"],
                    "val_loss": stats["val_loss"],
                    "val_acc": stats["val_acc"],
                    "epoch_time": stats["time"],
                },
                step=epoch + 1,  # x 轴从 1 开始
            )

    def _log_epoch_summary(self, epoch, stats):
        self.logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"T-Loss: {stats['train_loss']:.4f} | V-Loss: {stats['val_loss']:.4f} | "
            f"V-Acc: {stats['val_acc']:.2f}% | {stats['time']:.1f}s"
        )

    def _finalize_training(self):
        self.total_training_time = time.time() - self._train_start_time
        self.logger.info(f"\n⏱️ Total Time: {format_duration(self.total_training_time)}")
        self.history_saver.save(self.history, filename=f"{self.name}_history")

    def _handle_training_error(self, error):
        self.logger.error(f"\n❌ Training Failed: {error}")
        self.total_training_time = time.time() - self._train_start_time
        self._save_checkpoint("error")
        self.history_saver.save(self.history, filename=f"{self.name}_history")

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

    什所有增强参数从 cfg 读取:
    - cfg.experiment_name: 实验名称
    - cfg.augmentation_method: 增强方法
    - cfg.use_curriculum: 是否使用课程学习
    - cfg.fixed_ratio: 固定遮挡比例
    - cfg.mask_start_prob, cfg.mask_end_prob: 统一概率增长参数 (与三阶段共用)
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
        share_warmup_backbone=cfg.share_warmup_backbone,
    )

    # 训练
    trainer.train(train_loader, val_loader)

    # 加载最佳模型
    trainer.load_checkpoint("best")

    # 输出 best checkpoint 对应的 val acc
    if trainer.history["val_loss"]:
        best_idx = trainer.history["val_loss"].index(min(trainer.history["val_loss"]))
        best_val_acc = trainer.history["val_acc"][best_idx]
        best_epoch = trainer.history["epoch"][best_idx]
        get_logger().info(
            f"📊 Best Checkpoint (Epoch {best_epoch}): Val Acc = {best_val_acc:.2f}%"
        )

    get_logger().info(f"✅ Training completed: {cfg.experiment_name}")

    return trainer, trainer.total_training_time
