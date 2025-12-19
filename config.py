"""
================================================================================
配置模块
================================================================================
"""

import datetime
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import torch

from .utils import DEFAULT_SAVE_ROOT, ensure_dir, get_logger


@dataclass
class Experiment:
    """实验配置"""

    name: str
    desc: str = ""
    augmentation: str = "perlin"
    use_curriculum: bool = True
    fixed_ratio: Optional[float] = None
    fixed_prob: Optional[float] = None
    # 覆盖参数
    warmup_epochs: Optional[int] = None
    progressive_epochs: Optional[int] = None
    finetune_epochs: Optional[int] = None
    mask_start_ratio: Optional[float] = None
    mask_end_ratio: Optional[float] = None
    mask_prob_start: Optional[float] = None
    mask_prob_end: Optional[float] = None
    finetune_mask_ratio: Optional[float] = None
    finetune_mask_prob: Optional[float] = None

    def get_overrides(self) -> dict:
        """获取有效的覆盖参数"""
        exclude = {
            "name",
            "desc",
            "augmentation",
            "use_curriculum",
            "fixed_ratio",
            "fixed_prob",
        }
        return {
            k: v for k, v in asdict(self).items() if v is not None and k not in exclude
        }


@dataclass
class Config:
    """三阶段课程学习集成训练配置"""

    # 数据与模型
    data_root: str
    dataset_name: str
    val_split: float
    test_split: float  # 用于没有官方划分的数据集
    model_name: str
    num_models_per_gpu: int
    compile_model: bool

    # 训练超参数
    batch_size: int
    lr: float
    weight_decay: float
    max_grad_norm: float
    seed: int

    # 三阶段与 Mask
    warmup_epochs: int
    progressive_epochs: int
    finetune_epochs: int
    mask_pool_size: int
    mask_start_ratio: float
    mask_end_ratio: float
    mask_prob_start: float
    mask_prob_end: float
    finetune_mask_ratio: float
    finetune_mask_prob: float

    # 加载与优化
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    use_amp: bool
    use_tf32: bool
    early_stopping_patience: int

    # 保存与日志
    save_dir: str
    save_every_n_epochs: int
    keep_last_n_checkpoints: int
    use_tensorboard: bool
    log_level: str
    ece_n_bins: int
    ensemble_strategy: str  # 集成策略: "mean", "voting", "weighted"

    # 运行控制
    quick_test: bool
    resume_from: str

    # 自动计算字段
    num_classes: int = 0
    image_size: int = 0
    gpu_ids: List[int] = field(default_factory=list)
    experiment_name: str = ""

    @property
    def total_models(self) -> int:
        return len(self.gpu_ids) * self.num_models_per_gpu

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.progressive_epochs + self.finetune_epochs

    def copy(self, **kwargs) -> "Config":
        """克隆配置并可选地覆盖参数"""
        return replace(self, **kwargs)

    def apply_quick_test(self) -> "Config":
        """应用快速测试模式"""
        return replace(
            self,
            warmup_epochs=1,
            progressive_epochs=2,
            finetune_epochs=1,
            num_models_per_gpu=1,
        )

    def __post_init__(self) -> None:
        """初始化验证与自动配置"""
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("❌ 未检测到可用GPU")

        self.gpu_ids = self.gpu_ids or list(range(available_gpus))
        self.gpu_ids = [i for i in self.gpu_ids if i < available_gpus] or [0]

        self._auto_configure_for_dataset()

        if not self.save_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = str(
                Path(DEFAULT_SAVE_ROOT) / f"{self.experiment_name or 'exp'}_{timestamp}"
            )
        ensure_dir(self.save_dir)

    def _auto_configure_for_dataset(self) -> None:
        """根据数据集自动配置 num_classes 和 image_size"""
        from .datasets import DATASET_REGISTRY

        dataset_name = self.dataset_name.lower()

        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"❌ 不支持的数据集: {self.dataset_name}")

        DatasetClass = DATASET_REGISTRY[dataset_name]
        self.num_classes = self.num_classes or getattr(DatasetClass, "NUM_CLASSES", 10)
        self.image_size = self.image_size or getattr(DatasetClass, "IMAGE_SIZE", 32)

        # 如果需要 config_overrides，可以在 DatasetClass 中定义它
        if hasattr(DatasetClass, "CONFIG_OVERRIDES"):
            for k, v in DatasetClass.CONFIG_OVERRIDES.items():
                setattr(self, k, v)

        # 检查模型兼容性
        from .models import ModelFactory

        warnings = ModelFactory.check_compatibility(self.model_name, self.dataset_name)
        for w in warnings:
            get_logger().warning(w)

    def save(self, path: Optional[str] = None) -> None:
        """保存配置到 JSON 文件"""
        save_path = Path(path) if path else Path(self.save_dir) / "config.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        get_logger().info(f"💾 Config saved to: {save_path}")

    @classmethod
    def load_yaml(cls, yaml_path: str) -> tuple["Config", list[Experiment], list]:
        """从 YAML 加载完整任务配置 (Config, experiments, eval_checkpoints)"""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        base_cfg = cls(**data.get("base", {}))
        exps = [Experiment(**exp) for exp in data.get("experiments", [])]
        ckpts = data.get("eval_checkpoints", [])  # 保持简单列表或按需包装

        return base_cfg, exps, ckpts
