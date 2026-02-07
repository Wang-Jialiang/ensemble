"""
================================================================================
GPU Worker 模块
================================================================================

GPUWorker (单GPU模型管理器)、HistorySaver (训练历史保存器)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast

from ..config import Config
from ..models import ModelFactory
from ..utils import ensure_dir
from .augmentation import AUGMENTATION_REGISTRY, ClassAdaptiveAugmentation
from .optimization import create_optimizer, create_scheduler

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
        """GPU Worker 构造函数 (大纲化)"""
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.cfg, self.num_models = cfg, num_models

        # 1. 初始化深度学习组件 (模型, 优化器, 调度器)
        self.models, self.optimizers, self.schedulers = self._setup_models_and_optim()

        # 2. 初始化数据增强引擎
        self._init_augmentation(augmentation_method)

        # 3. 初始化异步执行流水线 (Stream)
        self.stream = torch.cuda.Stream(device=self.device)
        self._pending_loss = None

    def _setup_models_and_optim(self) -> Tuple[list, list, list]:
        """批量创建模型及配套优化工具"""
        ms, os, ss = [], [], []
        for _ in range(self.num_models):
            m = ModelFactory.create_model(
                self.cfg.model_name, self.cfg.num_classes, self.cfg.init_method
            ).to(self.device)
            if self.cfg.compile_model and hasattr(torch, "compile"):
                m = torch.compile(m)

            opt = create_optimizer(
                m,
                self.cfg.optimizer,
                self.cfg.lr,
                self.cfg.weight_decay,
                sgd_momentum=self.cfg.sgd_momentum,
            )
            sch = create_scheduler(
                opt, self.cfg.scheduler, self.cfg.total_epochs, self.cfg.min_lr
            )

            ms.append(m)
            os.append(opt)
            ss.append(sch)
        return ms, os, ss

    def _init_augmentation(self, method):
        """配置增强实例及其固定种子池"""
        if method not in AUGMENTATION_REGISTRY:
            raise ValueError(f"不支持的增强方法: {method}")

        # 创建基础增强方法
        base_augmentation = AUGMENTATION_REGISTRY[method](self.device, self.cfg)

        # 如果启用 CADA，使用 ClassAdaptiveAugmentation 包装
        if getattr(self.cfg, "cada_enabled", False) and method != "none":
            self.augmentation = ClassAdaptiveAugmentation(
                base_method=base_augmentation,
                num_classes=self.cfg.num_classes,
                base_prob=self.cfg.mask_end_prob,
            )
        else:
            self.augmentation = base_augmentation

    def precompute_masks(self, target_ratio: float):
        """预计算 mask 池

        每个 epoch 用当前 ratio 预计算共享 mask 池。
        """
        if hasattr(self.augmentation, "precompute_masks"):
            self.augmentation.precompute_masks(target_ratio)

    def train_batch_async(
        self, inputs, targets, criterion, m_ratio, m_prob, use_mask, model_indices=None
    ):
        """执行异步批次训练 (大纲化)

        Args:
            model_indices: 可选，指定要训练的模型索引列表。None 表示训练全部模型。
        """
        with torch.cuda.stream(self.stream):
            # 1. 搬运数据至显存
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 2. 确定要训练的模型
            indices = (
                list(model_indices)
                if model_indices is not None
                else list(range(self.num_models))
            )

            # 3. 迭代指定的模型
            total_loss = 0.0
            for i in indices:
                m, opt = self.models[i], self.optimizers[i]
                total_loss += self._step_model(
                    i, m, opt, inputs, targets, criterion, m_ratio, m_prob, use_mask
                )

            self._pending_loss = total_loss / len(indices) if len(indices) > 0 else 0.0

    def _step_model(
        self,
        idx,
        model,
        optimizer,
        inputs,
        targets,
        criterion,
        m_ratio,
        m_prob,
        use_mask,
    ):
        """执行单个模型的梯度更新步"""
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # 1. 准备增强数据
        x, y = self._prepare_training_data(
            idx, inputs, targets, m_ratio, m_prob, use_mask
        )

        # 2. 执行前向与反向传播
        loss = self._forward_backward(model, x, y, criterion)

        # 3. 梯度裁剪与参数更新
        nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
        optimizer.step()

        return loss.item()

    def _prepare_training_data(self, idx, x, y, ratio, prob, use_mask):
        """根据策略应用数据增强"""
        if not use_mask:
            return x, y

        return self.augmentation.apply(x, y, ratio, prob)

    def _forward_backward(self, model, x, y, criterion):
        """内部执行计算链路"""
        if self.cfg.use_amp:
            with autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
        else:
            loss = criterion(model(x), y)

        loss.backward()
        return loss

    def synchronize(self) -> float:
        """同步并返回平均loss"""
        self.stream.synchronize()
        return self._pending_loss if self._pending_loss else 0.0

    def step_schedulers(self):
        """更新学习率调度器"""
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()

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
        """保存模型权重"""
        for i, model in enumerate(self.models):
            save_path = Path(save_dir) / f"{prefix}_gpu{self.gpu_id}_model{i}.pth"
            torch.save(model.state_dict(), save_path)

    def load_models(self, save_dir: str, prefix: str):
        """加载模型权重"""
        for i, model in enumerate(self.models):
            load_path = Path(save_dir) / f"{prefix}_gpu{self.gpu_id}_model{i}.pth"
            if load_path.exists():
                state_dict = torch.load(
                    load_path, map_location=self.device, weights_only=False
                )
                model.load_state_dict(state_dict)

    def broadcast_backbone_and_reinit_heads(self, backbone_state_dict: dict):
        """用共享 backbone 初始化所有模型，并重新初始化各模型的 classifier head

        Args:
            backbone_state_dict: 源模型的 backbone 权重 (不含 fc 层)
        """
        for model in self.models:
            # 加载 backbone 权重 (strict=False 因为不含 fc 层)
            model.load_state_dict(backbone_state_dict, strict=False)
            # 重新初始化 classifier head
            model.reinit_classifier(init_method=self.cfg.init_method)

    def update_adaptive_probs(self, class_probs: list):
        """更新类别自适应增强概率

        Args:
            class_probs: 每个类别的触发概率
        """
        if hasattr(self.augmentation, "update_adaptive_probs"):
            self.augmentation.update_adaptive_probs(class_probs)
            self.augmentation.enable()  # 启用自适应模式


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 训练历史保存器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class HistorySaver:
    """训练历史 CSV 保存器 (大纲化)"""

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        ensure_dir(self.save_dir)

    def save(self, history: Dict[str, List], filename: str = "history"):
        """将历史字典导出至 CSV 文件"""
        import csv

        path = self.save_dir / f"{filename}.csv"

        if not history:
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history.keys())
            writer.writeheader()
            self._write_rows(writer, history)

    def _write_rows(self, writer, history):
        """遍历并写入行数据"""
        num_entries = len(next(iter(history.values())))
        for i in range(num_entries):
            writer.writerow({k: v[i] for k, v in history.items()})
