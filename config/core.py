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

from ..utils import ensure_dir, get_logger


@dataclass
class Config:
    """三阶段课程学习集成训练配置"""

    # ==========================================================================
    # [全局] 数据配置 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    data_root: str  # 数据集根目录路径
    save_root: str  # 检查点/输出保存根目录
    dataset_name: str  # 数据集名称: "cifar10", "cifar100", "eurosat" 等
    val_split: float  # 验证集划分比例 (0.0-1.0)
    test_split: float  # 测试集划分比例，用于无官方划分的数据集

    # ==========================================================================
    # [全局] 模型配置 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    model_name: str  # 模型名称: "resnet18", "resnet50", "vgg16" 等
    num_models_per_gpu: int  # 每个 GPU 上的模型数量
    compile_model: bool  # 是否启用 PyTorch 2.0+ 编译优化 (可提升10-50%速度)

    # ==========================================================================
    # [全局] 训练超参数 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    batch_size: int  # 批次大小
    lr: float  # 基础学习率
    weight_decay: float  # 权重衰减 (L2 正则化系数)
    max_grad_norm: float  # 梯度裁剪阈值
    seed: int  # 随机种子
    optimizer: str  # 优化器: "adamw", "sgd", "adam", "rmsprop"
    scheduler: str  # 调度器: "cosine", "step", "plateau", "none"
    label_smoothing: float  # 标签平滑系数 (0.0=不使用, 0.1=常用值)

    # ==========================================================================
    # [阶段训练专用] 三阶段与 Mask - 仅 StagedEnsembleTrainer 使用
    # ==========================================================================
    warmup_epochs: int  # Warmup 阶段轮数
    progressive_epochs: int  # Progressive 阶段轮数
    finetune_epochs: int  # Finetune 阶段轮数
    mask_pool_size: int  # 预生成的 Mask 池大小
    mask_start_ratio: float  # Progressive 阶段起始遮罩比例
    mask_end_ratio: float  # Progressive 阶段结束遮罩比例
    mask_prob_start: float  # Progressive 阶段起始应用概率
    mask_prob_end: float  # Progressive 阶段结束应用概率
    finetune_mask_ratio: float  # Finetune 阶段固定遮罩比例
    finetune_mask_prob: float  # Finetune 阶段固定应用概率

    # ==========================================================================
    # [阶段训练专用] 阶段学习率缩放 - 仅 StagedEnsembleTrainer 使用
    # ==========================================================================
    warmup_lr_scale: float  # Warmup 阶段学习率缩放因子 (lr * scale)
    progressive_lr_scale: float  # Progressive 阶段学习率缩放因子
    finetune_lr_scale: float  # Finetune 阶段学习率缩放因子

    # ==========================================================================
    # [全局] 数据加载配置 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    num_workers: int  # DataLoader 工作进程数
    pin_memory: bool  # 是否使用锁页内存加速 GPU 传输
    persistent_workers: bool  # 是否保持工作进程存活
    prefetch_factor: int  # 每个 worker 预取的批次数

    # ==========================================================================
    # [全局] 训练优化配置 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    use_amp: bool  # 是否启用自动混合精度 (AMP)
    use_tf32: bool  # 是否启用 TF32 加速 (仅 Ampere+ GPU)
    early_stopping_patience: int  # 早停耐心值 (验证集无改善的轮数)

    # ==========================================================================
    # [全局] 保存与日志配置 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    save_every_n_epochs: int  # 每 N 轮保存一次检查点
    keep_last_n_checkpoints: int  # 保留最近 N 个检查点
    use_tensorboard: bool  # 是否启用 TensorBoard 日志
    log_level: str  # 日志级别: "DEBUG", "INFO", "WARNING", "ERROR"

    # ==========================================================================
    # [评估专用] 评估配置 - 仅评估模块使用
    # ==========================================================================
    ece_n_bins: int  # 校准度 (ECE) 计算的分箱数量
    ensemble_strategy: str  # 集成策略: "mean" (等权平均), "voting" (多数投票)
    corruption_dataset: bool  # 是否加载 Corruption 数据集进行评估
    ood_dataset: bool  # 是否加载 OOD 数据集进行评估
    domain_dataset: bool  # 是否加载 Domain Shift 数据集进行评估

    # ==========================================================================
    # [评估专用] 对抗鲁棒性评估参数 - 仅评估模块使用
    # ==========================================================================
    adv_eps: float  # FGSM/PGD 扰动强度 ε (常用值: 8/255 ≈ 0.031)
    adv_alpha: float  # PGD 步长 α (常用值: 2/255 ≈ 0.008)
    adv_pgd_steps: int  # PGD 迭代步数 (常用值: 10, 20)

    # ==========================================================================
    # [全局] 优化器高级参数 - SGD 专用
    # ==========================================================================
    sgd_momentum: float  # SGD 动量 (默认 0.9)

    # ==========================================================================
    # [增强专用] 数据增强参数 - Perlin/Cutout 使用
    # ==========================================================================
    cutout_fill_value: float  # Cutout 填充值 (默认 0.5)
    perlin_persistence: float  # Perlin 噪声持久度 (默认 0.5)

    # ==========================================================================
    # [评估专用] 可视化参数
    # ==========================================================================
    plot_dpi: int  # 图表保存 DPI (默认 150)

    # ==========================================================================
    # [全局] 模型初始化 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    init_method: str  # 初始化方法: "kaiming", "xavier", "orthogonal", "default"

    # ==========================================================================
    # [全局] 运行控制 - 被 BaseTrainer 及所有子类使用
    # ==========================================================================
    quick_test: bool  # 快速测试模式 (减少轮数/模型数)

    # ==========================================================================
    # [实验级别] 增强与课程学习参数 - 每个实验可覆盖
    # ==========================================================================
    augmentation_method: str  # 增强方法: "perlin", "cutout", "none" 等
    use_curriculum: bool  # 是否使用课程学习
    fixed_ratio: float  # 固定遮挡比例 (仅 use_curriculum=False 时生效)
    fixed_prob: float  # 固定遮挡概率 (仅 use_curriculum=False 时生效)
    share_warmup_backbone: bool  # 是否在 warmup 后共享 backbone

    # 自动计算/生成字段 (有默认值)
    save_dir: str = ""  # 检查点保存目录 (由 __post_init__ 自动生成)
    num_classes: int = 0
    image_size: int = 0
    gpu_ids: List[int] = field(
        default_factory=list, init=False
    )  # 由 __post_init__ 自动设置
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

        self.gpu_ids = list(range(available_gpus))  # 使用所有可用 GPU

        self._auto_configure_for_dataset()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = str(
            Path(self.save_root) / f"{self.experiment_name or 'exp'}_{timestamp}"
        )
        ensure_dir(self.save_dir)

    def _auto_configure_for_dataset(self) -> None:
        """根据数据集自动配置 num_classes 和 image_size"""
        from ..datasets import DATASET_REGISTRY

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

    def save(self, path: Optional[str] = None) -> None:
        """保存配置到 JSON 文件"""
        save_path = Path(path) if path else Path(self.save_dir) / "config.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        get_logger().info(f"💾 Config saved to: {save_path}")

    @classmethod
    def load_yaml(cls, yaml_path: str) -> tuple["Config", list["Experiment"], list]:
        """从 YAML 加载完整任务配置 (Config, experiments, eval_checkpoints)

        配置合并顺序: constants (业界标准) -> base (用户自定义)
        """
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 合并 constants 和 base，base 覆盖 constants
        merged_cfg = {**data.get("constants", {}), **data.get("base", {})}
        base_cfg = cls(**merged_cfg)
        exps = [Experiment(**exp) for exp in data.get("experiments", [])]
        ckpts = data.get("eval_checkpoints", [])  # 保持简单列表或按需包装

        return base_cfg, exps, ckpts


@dataclass
class Experiment:
    """实验配置

    字段名与 Config 保持一致，方便直接 copy 覆盖
    """

    name: str
    desc: str = ""
    # 与 Config 同名的字段，可直接覆盖
    augmentation_method: str = "perlin"
    use_curriculum: bool = True
    fixed_ratio: Optional[float] = None
    fixed_prob: Optional[float] = None

    def get_config_overrides(self) -> dict:
        """获取所有需要覆盖的参数 (过滤 name/desc 和 None 值)"""
        exclude = {"name", "desc"}
        return {
            k: v for k, v in asdict(self).items() if k not in exclude and v is not None
        }
