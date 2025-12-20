"""
================================================================================
训练模块 - 数据增强、GPUWorker、三阶段集成训练器
================================================================================
"""

import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .models import ModelFactory
from .utils import ensure_dir, format_duration, get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 优化器与调度器工厂                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_optimizer(
    model: nn.Module, optimizer_name: str, lr: float, weight_decay: float
) -> optim.Optimizer:
    """
    创建优化器

    Args:
        model: 模型
        optimizer_name: 优化器名称 (adamw, sgd, adam, rmsprop)
        lr: 学习率
        weight_decay: 权重衰减

    Returns:
        optimizer: 优化器实例
    """
    optimizer_name = optimizer_name.lower()
    params = model.parameters()

    if optimizer_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"不支持的优化器: {optimizer_name}. 支持: adamw, sgd, adam, rmsprop"
        )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    total_epochs: int,
    steps_per_epoch: int = 0,
) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 (cosine, step, plateau, onecycle, none)
        total_epochs: 总训练轮数
        steps_per_epoch: 每轮步数 (用于 OneCycleLR)

    Returns:
        scheduler: 调度器实例，none 时返回 None
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif scheduler_name == "step":
        # 每 30% 和 60% 的 epoch 时降低学习率
        milestones = [int(total_epochs * 0.3), int(total_epochs * 0.6)]
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif scheduler_name == "onecycle":
        if steps_per_epoch <= 0:
            raise ValueError("OneCycleLR 需要 steps_per_epoch > 0")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"] * 10,
            total_steps=total_epochs * steps_per_epoch,
        )
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(
            f"不支持的调度器: {scheduler_name}. 支持: cosine, step, plateau, onecycle, none"
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 早停机制                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class EarlyStopping:
    """早停机制

    用于在验证指标不再改善时提前停止训练，防止过拟合。

    Args:
        patience: 允许的最大无改善的epoch数
        min_delta: 最小改善阈值
        mode: 'min' 或 'max'，指定指标是越小越好还是越大越好
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """检查是否应该早停

        Returns:
            True 如果应该停止训练，否则 False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 云状Mask生成器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CloudMaskGenerator:
    """GPU加速的云状Mask生成器"""

    def __init__(self, height: int, width: int, device: torch.device):
        self.h = height
        self.w = width
        self.device = device
        # base_scale 随图像尺寸动态调整: 32x32 -> 16, 64x64 -> 32
        self.base_scale = min(height, width) / 2.0

    def generate_batch(
        self, num_masks: int, target_ratio: float = 0.3
    ) -> List[torch.Tensor]:
        """批量生成Perlin噪声Mask"""
        masks = []
        for _ in range(num_masks):
            # 动态调整 octaves 参数
            scale = self.base_scale * random.uniform(0.8, 1.2)
            octaves = 4 if self.h >= 64 else 3
            persistence = 0.5

            noise = self._generate_perlin_noise(scale, octaves, persistence)
            # 使用target_ratio作为阈值
            threshold = torch.quantile(noise, 1.0 - target_ratio)
            mask = (noise < threshold).float()
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            masks.append(mask)

        return masks

    def _generate_perlin_noise(
        self, scale: float, octaves: int = 4, persistence: float = 0.5
    ) -> torch.Tensor:
        """生成Perlin噪声"""
        noise = torch.zeros(self.h, self.w, device=self.device)
        amplitude = 1.0
        max_val = 0.0

        for i in range(octaves):
            freq = 2**i
            # 确保频率不会太高导致尺寸为0
            grid_h = max(2, int(self.h / (scale / freq)))
            grid_w = max(2, int(self.w / (scale / freq)))

            rand_grid = torch.rand(grid_h + 1, grid_w + 1, device=self.device)

            # 双线性插值
            upsampled = F.interpolate(
                rand_grid.unsqueeze(0).unsqueeze(0),
                size=(self.h, self.w),
                mode="bilinear",
                align_corners=True,
            ).squeeze()

            noise += upsampled * amplitude
            max_val += amplitude
            amplitude *= persistence

        return noise / max_val


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据增强方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class AugmentationMethod:
    """数据增强方法基类"""

    def __init__(self, device: torch.device):
        self.device = device

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强方法"""
        raise NotImplementedError


class CutoutAugmentation(AugmentationMethod):
    """Cutout硬遮挡"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape
        mask_size = int(H * np.sqrt(ratio))

        augmented = images.clone()
        for i in range(B):
            y = random.randint(0, max(0, H - mask_size))
            x = random.randint(0, max(0, W - mask_size))
            augmented[i, :, y : y + mask_size, x : x + mask_size] = 0.5

        return augmented, targets


class MixupAugmentation(AugmentationMethod):
    """Mixup混合增强"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        # 边界检查防止 beta 分布参数无效
        ratio = np.clip(ratio, 0.01, 0.99)
        lam = np.random.beta(ratio * 10, (1 - ratio) * 10)
        lam = max(lam, 1 - lam)

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, targets


class CutMixAugmentation(AugmentationMethod):
    """CutMix剪切混合"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape

        lam = np.random.beta(1.0, 1.0)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        index = torch.randperm(B).to(self.device)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            index, :, bby1:bby2, bbx1:bbx2
        ]

        return mixed_images, targets


class DropoutAugmentation(AugmentationMethod):
    """特征级Dropout"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        mask = torch.rand_like(images) > ratio
        augmented = images * mask.float()
        return augmented, targets


class PerlinMaskAugmentation(AugmentationMethod):
    """Perlin噪声遮挡（原方法）"""

    def __init__(
        self, device: torch.device, height: int, width: int, pool_size: int = 100
    ):
        super().__init__(device)
        self.mask_generator = CloudMaskGenerator(height, width, device)
        self.masks = []
        self.pool_size = pool_size

    def precompute_masks(self, target_ratio: float):
        """预计算mask池"""
        self.masks = self.mask_generator.generate_batch(self.pool_size, target_ratio)

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob or not self.masks:
            return images, targets

        mask = self.masks[random.randint(0, len(self.masks) - 1)]
        if mask.shape[1] == 1:
            mask = mask.expand(1, 3, -1, -1)

        augmented = images * mask
        return augmented, targets


class NoAugmentation(AugmentationMethod):
    """无增强（Baseline）"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return images, targets


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 增强方法注册表                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

AUGMENTATION_REGISTRY = {
    "cutout": lambda device, cfg: CutoutAugmentation(device),
    "mixup": lambda device, cfg: MixupAugmentation(device),
    "cutmix": lambda device, cfg: CutMixAugmentation(device),
    "dropout": lambda device, cfg: DropoutAugmentation(device),
    "perlin": lambda device, cfg: PerlinMaskAugmentation(
        device, cfg.image_size, cfg.image_size, cfg.mask_pool_size
    ),
    "none": lambda device, cfg: NoAugmentation(device),
}


def register_augmentation(name: str, builder: Callable):
    """动态注册增强方法"""
    AUGMENTATION_REGISTRY[name] = builder


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
        self.schedulers: List[optim.lr_scheduler.LRScheduler] = []

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
            scheduler = create_scheduler(optimizer, cfg.scheduler, cfg.total_epochs)

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
                "scheduler_state_dict": scheduler.state_dict(),
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
                scheduler.load_state_dict(state["scheduler_state_dict"])


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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 三阶段集成训练器                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class StagedEnsembleTrainer:
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
    ):
        self.name = method_name
        self.cfg = cfg
        self.total_training_time = 0.0

        # 增强配置
        self.augmentation_method = augmentation_method
        self.use_curriculum = use_curriculum
        self.fixed_ratio = fixed_ratio
        self.fixed_prob = fixed_prob

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

    def _get_stage_info(self, epoch: int) -> Tuple[int, str, float, float, bool, float]:
        """获取当前阶段信息

        Returns:
            Tuple: (stage_num, stage_name, mask_ratio, mask_prob, use_mask, lr_scale)
        """
        cfg = self.cfg

        # 模式1: 无增强 (Baseline)
        if self.augmentation_method == "none":
            return 1, "NoAug", 0.0, 0.0, False, 1.0

        # 模式2: 固定参数模式
        if not self.use_curriculum:
            return 1, "Fixed", self.fixed_ratio, self.fixed_prob, True, 1.0

        # 模式3: 课程学习模式 (三阶段)
        if epoch < cfg.warmup_epochs:
            return 1, "Warmup", 0.0, 0.0, False, cfg.warmup_lr_scale
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
            return (
                2,
                "Progressive",
                mask_ratio,
                mask_prob,
                True,
                cfg.progressive_lr_scale,
            )
        else:
            return (
                3,
                "Finetune",
                cfg.finetune_mask_ratio,
                cfg.finetune_mask_prob,
                True,
                cfg.finetune_lr_scale,
            )

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """训练一个epoch"""
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        stage_num, stage_name, mask_ratio, mask_prob, use_mask, lr_scale = (
            self._get_stage_info(epoch)
        )

        # 应用阶段学习率缩放
        stage_lr = self.cfg.lr * lr_scale
        for worker in self.workers:
            worker.set_lr(stage_lr)

        # 预计算mask（如果需要）
        for worker in self.workers:
            worker.precompute_masks(self.cfg.mask_pool_size, mask_ratio)

        total_loss = 0.0
        num_batches = 0
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.total_epochs} [{stage_name}] lr={stage_lr:.6f}",
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
                stage_num, stage_name, mask_ratio, mask_prob, use_mask, lr_scale = (
                    self._get_stage_info(epoch)
                )

                # 阶段切换提示
                if stage_num != current_stage:
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
# ║ 实验运行函数 (从 evaluation.py 移动过来)                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def train_experiment(
    experiment_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    augmentation_method: Optional[str] = None,
    use_curriculum: Optional[bool] = None,
    fixed_ratio: Optional[float] = None,
    fixed_prob: Optional[float] = None,
    resume: Optional[str] = None,
) -> Tuple["StagedEnsembleTrainer", float]:
    """
    仅训练实验 (不包含评估)

    参数:
        experiment_name: 实验名称
        cfg: 配置对象
        train_loader, val_loader: 数据加载器
        augmentation_method: 增强方法 (None=使用cfg默认)
        use_curriculum: 是否使用课程学习 (None=使用cfg默认)
        fixed_ratio: 固定遮挡比例 (仅在use_curriculum=False时生效)
        fixed_prob: 固定遮挡概率 (仅在use_curriculum=False时生效)
        resume: 恢复checkpoint的路径

    返回:
        (trainer, training_time)
    """
    aug_method = augmentation_method or ("perlin" if cfg.use_perlin_mask else "none")
    curriculum = use_curriculum if use_curriculum is not None else True
    f_ratio = fixed_ratio if fixed_ratio is not None else 0.25
    f_prob = fixed_prob if fixed_prob is not None else 0.5

    trainer = StagedEnsembleTrainer(
        experiment_name,
        cfg,
        augmentation_method=aug_method,
        use_curriculum=curriculum,
        fixed_ratio=f_ratio,
        fixed_prob=f_prob,
    )

    # 恢复训练
    if resume:
        trainer.load_checkpoint(resume)

    # 训练
    trainer.train(train_loader, val_loader)
    training_time = trainer.total_training_time

    # 加载最佳模型
    trainer.load_checkpoint("best")
    trainer.total_training_time = training_time

    get_logger().info(f"\n✅ Training completed: {experiment_name}")
    get_logger().info(
        f"   Checkpoint saved to: {Path(cfg.save_dir) / 'checkpoints' / experiment_name}"
    )

    return trainer, training_time
