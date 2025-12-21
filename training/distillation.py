"""
================================================================================
知识蒸馏训练器
================================================================================

Knowledge Distillation Ensemble 实现 (Hinton et al., 2015)

核心思想: 用预训练的教师集成生成软标签，训练一个学生模型。
- 学生模型可以逼近教师集成的性能
- 单模型推理，但具有集成级别的知识

参考: https://arxiv.org/abs/1503.02531
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..models import ModelFactory
from ..utils import ensure_dir, get_logger
from .base import BaseTrainer
from .scheduler import create_optimizer, create_scheduler

if TYPE_CHECKING:
    from .core import StagedEnsembleTrainer


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失

    Loss = α * KL(student_soft || teacher_soft) * T^2 + (1 - α) * CE(student_hard, labels)

    其中:
    - T: 温度参数，控制软标签的平滑程度
    - α: 软标签损失权重
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: 学生模型输出 [batch_size, num_classes]
            teacher_logits: 教师模型输出 [batch_size, num_classes]
            targets: 硬标签 [batch_size]

        Returns:
            Combined distillation loss
        """
        # 软标签损失 (KL 散度)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)

        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        soft_loss = soft_loss * (self.temperature**2)

        # 硬标签损失 (交叉熵)
        hard_loss = self.ce_loss(student_logits, targets)

        # 组合损失
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class DistillationTrainer(BaseTrainer):
    """
    知识蒸馏训练器

    使用已训练的教师集成 (多个模型或 StagedEnsembleTrainer)
    训练一个学生模型。
    """

    def __init__(
        self,
        method_name: str,
        cfg: Config,
        teacher_models: Union[List[nn.Module], "StagedEnsembleTrainer"],
        temperature: float = 4.0,
        alpha: float = 0.7,
        student_model_name: Optional[str] = None,
    ):
        # 调用基类初始化
        super().__init__(method_name, cfg)

        self.temperature = temperature
        self.alpha = alpha

        # 设备
        self.device = torch.device(
            f"cuda:{cfg.gpu_ids[0]}" if cfg.gpu_ids else "cuda:0"
        )

        # 性能优化
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        get_logger().info(f"\n🚀 Initializing {method_name} (Knowledge Distillation)")
        get_logger().info(f"   Temperature: {temperature}, Alpha: {alpha}")
        get_logger().info(f"   Device: {self.device}")

        # 设置教师模型
        self.teacher_models = self._setup_teachers(teacher_models)
        get_logger().info(f"   Teachers: {len(self.teacher_models)} models")

        # 创建学生模型
        student_name = student_model_name or cfg.model_name
        self.student = ModelFactory.create_model(
            student_name,
            num_classes=cfg.num_classes,
            init_method=cfg.init_method,
        ).to(self.device)

        # 可选编译
        if cfg.compile_model and hasattr(torch, "compile"):
            self.student = torch.compile(self.student)

        # 蒸馏损失
        self.criterion = DistillationLoss(temperature, alpha)

        # 优化器和调度器 (使用公共工厂函数)
        self.optimizer = create_optimizer(
            self.student, cfg.optimizer, cfg.lr, cfg.weight_decay
        )
        self.scheduler = create_scheduler(self.optimizer, "cosine", cfg.total_epochs)

        # 设置日志
        self.setup_logging()

        # 保存配置
        cfg.save()

    def _setup_teachers(self, teachers) -> List[nn.Module]:
        """设置教师模型"""
        # 如果是 StagedEnsembleTrainer，提取其 workers 中的模型
        if hasattr(teachers, "workers"):
            models = []
            for worker in teachers.workers:
                for model in worker.models:
                    model.eval()
                    models.append(model)
            return models

        # 如果已经是模型列表
        elif isinstance(teachers, list):
            for m in teachers:
                m.eval()
                m.to(self.device)
            return teachers

        else:
            raise ValueError(f"Unknown teacher type: {type(teachers)}")

    # 注: _create_optimizer 和 _create_scheduler 已移除，使用 scheduler 模块的公共函数替代

    @torch.no_grad()
    def _get_teacher_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """获取教师集成的平均 logits"""
        all_logits = []

        for teacher in self.teacher_models:
            # 将输入移到教师模型所在设备
            teacher_device = next(teacher.parameters()).device
            inputs_t = inputs.to(teacher_device)
            logits = teacher(inputs_t).to(self.device)
            all_logits.append(logits)

        # 返回平均 logits
        return torch.stack(all_logits).mean(dim=0)

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个 epoch"""
        self.student.train()

        total_loss = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.total_epochs}",
        )

        for inputs, targets in iterator:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 获取教师软标签
            teacher_logits = self._get_teacher_logits(inputs)

            # 学生前向
            self.optimizer.zero_grad()
            student_logits = self.student(inputs)

            # 蒸馏损失
            loss = self.criterion(student_logits, teacher_logits, targets)

            loss.backward()

            if self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.cfg.max_grad_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            iterator.set_postfix({"loss": loss.item()})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证学生模型"""
        self.student.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.student(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    @torch.no_grad()
    def _validate_teacher(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证教师集成 (用于对比)"""
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            ensemble_logits = self._get_teacher_logits(inputs)
            loss = criterion(ensemble_logits, targets)
            total_loss += loss.item()

            preds = ensemble_logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """执行完整训练"""
        self.logger.info("=" * 70)
        self.logger.info(f"🎯 Knowledge Distillation: {self.name}")
        self.logger.info(
            f"   Teachers: {len(self.teacher_models)}, T={self.temperature}, α={self.alpha}"
        )
        self.logger.info("=" * 70)

        # 先评估教师性能
        teacher_loss, teacher_acc = self._validate_teacher(val_loader)
        self.logger.info(
            f"📚 Teacher ensemble: Loss={teacher_loss:.4f}, Acc={teacher_acc:.2f}%"
        )

        training_start = time.time()

        try:
            for epoch in range(self.cfg.total_epochs):
                epoch_start = time.time()

                # 训练
                train_loss = self._train_epoch(train_loader, epoch)

                # 更新调度器
                if self.scheduler:
                    self.scheduler.step()

                # 验证
                val_loss, val_acc = self._validate(val_loader)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                # 使用基类方法记录 epoch
                self._log_epoch(
                    epoch, train_loss, val_loss, val_acc, current_lr, epoch_time
                )

                # 使用基类方法检查最佳并保存
                self._check_best_and_save(val_loss, epoch)

                # 早停
                if self.early_stopping(val_loss, epoch):
                    self.logger.info(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                    break

            # 最终对比
            final_loss, final_acc = self._validate(val_loader)
            self.logger.info("\n📊 Final Comparison:")
            self.logger.info(
                f"   Teacher: Loss={teacher_loss:.4f}, Acc={teacher_acc:.2f}%"
            )
            self.logger.info(f"   Student: Loss={final_loss:.4f}, Acc={final_acc:.2f}%")

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

        torch.save(self.student.state_dict(), ckpt_dir / "student.pth")

        state = {
            "epoch": len(self.history["epoch"]),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "temperature": self.temperature,
            "alpha": self.alpha,
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

        self.student.load_state_dict(
            torch.load(ckpt_dir / "student.pth", weights_only=True)
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
        """获取模型列表 (返回学生模型)"""
        return [self.student]


def train_distillation(
    experiment_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher_models: Union[List[nn.Module], "StagedEnsembleTrainer"],
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> Tuple[DistillationTrainer, float]:
    """
    知识蒸馏训练入口函数

    Args:
        experiment_name: 实验名称
        cfg: 配置
        train_loader, val_loader: 数据加载器
        teacher_models: 教师模型列表或 StagedEnsembleTrainer
        temperature: 蒸馏温度
        alpha: 软标签权重

    Returns:
        (trainer, training_time)
    """
    trainer = DistillationTrainer(
        method_name=experiment_name,
        cfg=cfg,
        teacher_models=teacher_models,
        temperature=temperature,
        alpha=alpha,
    )

    trainer.train(train_loader, val_loader)
    training_time = trainer.total_training_time

    # 加载最佳模型
    trainer.load_checkpoint("best")
    trainer.total_training_time = training_time

    get_logger().info(f"\n✅ Distillation completed: {experiment_name}")

    return trainer, training_time
