"""
================================================================================
配置模块
================================================================================
"""

import datetime
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import torch


@dataclass
class GenerationConfig:
    """数据生成配置 (Corruption / OOD) - SDXL Lightning"""

    base_model: str = ""  # SDXL 基础模型 (由 yaml 填充)
    lightning_repo: str = ""  # Lightning UNet 仓库
    lightning_ckpt: str = ""  # 4-step 检查点
    num_steps: int = 4  # 推理步数 (2/4/8)
    batch_size: int = 12  # 与 yaml 同步
    samples_per_group: int = 500  # 与 yaml 同步
    visualize: bool = True
    num_vis: int = 5  # 与 yaml 同步

    # SDXL 生成参数
    sdxl_height: int = 1024  # Text2Img 原始输出高度
    sdxl_width: int = 1024  # Text2Img 原始输出宽度
    guidance_scale_text2img: float = 0.0  # Text2Img CFG

    ood_prompts: List[str] = field(default_factory=list)  # 由 default.yaml 填充
    vis_corruptions: List[str] = field(default_factory=list)  # 由 default.yaml 填充


@dataclass
class Config:
    """三阶段课程学习集成训练配置"""

    # ==========================================================================
    # [全局] 数据配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    data_root: str  # 数据集根目录路径
    save_root: str  # 检查点/输出保存根目录
    dataset_name: str  # 数据集名称: "cifar10", "cifar100", "eurosat" 等
    val_split: float  # 验证集划分比例 (0.0-1.0)
    test_split: float  # 测试集划分比例，用于无官方划分的数据集

    # ==========================================================================
    # [全局] 模型配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    model_name: str  # 模型名称: "resnet18", "resnet50", "vgg16" 等
    num_models_per_gpu: int  # 每个 GPU 上的模型数量
    compile_model: bool  # 是否启用 PyTorch 2.0+ 编译优化 (可提升10-50%速度)

    # ==========================================================================
    # [全局] 训练超参数 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    batch_size: int  # 批次大小
    lr: float  # 基础学习率
    weight_decay: float  # 权重衰减 (L2 正则化系数)
    max_grad_norm: float  # 梯度裁剪阈值
    seed: int  # 随机种子
    optimizer: str  # 优化器: "adamw", "sgd", "adam", "rmsprop"
    scheduler: str  # 调度器: "cosine", "step", "plateau", "none"
    min_lr: float  # 学习率调度器最小值 (cosine 衰减终点)
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
    # [全局] 数据加载配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    num_workers: int  # DataLoader 工作进程数
    pin_memory: bool  # 是否使用锁页内存加速 GPU 传输
    persistent_workers: bool  # 是否保持工作进程存活
    prefetch_factor: int  # 每个 worker 预取的批次数

    # ==========================================================================
    # [全局] 训练优化配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    use_amp: bool  # 是否启用自动混合精度 (AMP, 默认使用 BFloat16)
    use_tf32: bool  # 是否启用 TF32 加速 (仅 Ampere+ GPU)
    early_stopping_patience: int  # 早停耐心值 (验证集无改善的轮数)

    # ==========================================================================
    # [全局] 保存与日志配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    use_wandb: bool  # 是否启用 Weights & Biases 日志
    wandb_project: str  # wandb 项目名称
    log_level: str  # 日志级别: "DEBUG", "INFO", "WARNING", "ERROR"
    log_to_console: bool  # 训练日志是否同时输出到控制台 (默认 True)

    # ==========================================================================
    # [评估专用] 评估配置 - 仅评估模块使用
    # ==========================================================================
    ece_n_bins: int  # 校准度 (ECE) 计算的分箱数量
    ensemble_strategy: str  # 集成策略: "mean" (等权平均), "voting" (多数投票)
    corruption_dataset: bool  # 是否加载 Corruption 数据集进行评估
    ood_dataset: bool  # 是否加载 OOD 数据集进行评估
    eval_run_gradcam: bool  # 是否在评估时运行 Grad-CAM 分析
    eval_run_landscape: bool  # 是否在评估时运行 Loss Landscape 分析

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
    # [增强专用] 数据增强参数 - Perlin/Cutout/GridMask 使用
    # ==========================================================================

    perlin_persistence: float  # Perlin 噪声持久度 (默认 0.5)
    perlin_scale_ratio: float  # Perlin 噪声尺度比例 (默认 0.3)
    gridmask_d_ratio_min: float  # GridMask 网格单元最小尺寸比例 (默认 0.2)
    gridmask_d_ratio_max: float  # GridMask 网格单元最大尺寸比例 (默认 0.4)
    perlin_octaves_large: int  # Perlin octaves (图像 >= 64 时, 默认 4)
    perlin_octaves_small: int  # Perlin octaves (图像 < 64 时, 默认 3)
    model_level_augmentation: bool  # 是否启用模型级固定 seed (每个模型固定视角)

    # ==========================================================================
    # [评估专用] 可视化参数
    # ==========================================================================
    plot_dpi: int  # 图表保存 DPI (默认 300)
    gradcam_num_samples: int  # Grad-CAM 分析样本数

    # ==========================================================================
    # [全局] 模型初始化 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    init_method: str  # 初始化方法: "kaiming", "xavier", "orthogonal", "default"

    # ==========================================================================
    # [全局] 运行控制 - 被 StagedEnsembleTrainer 使用
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

    # ==========================================================================
    # 有默认值的字段 (必须放在最后)
    # ==========================================================================
    adv_eps_list: Optional[List[float]] = None  # 多 ε 评估列表 (可选)
    adv_targeted: bool = False  # 是否为针对性攻击
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # 自动计算/生成字段 (有默认值, 禁止人工初始化)
    save_dir: str = field(
        default="", init=False
    )  # 训练产物目录: output/training/{exp_name}/
    evaluation_dir: str = field(
        default="", init=False
    )  # 评估产物目录: output/evaluation/{ts}/
    num_classes: int = field(default=0, init=False)
    image_size: int = field(default=0, init=False)
    dataset_mean: List[float] = field(
        default_factory=list, init=False
    )  # 数据集均值 (Post-init 填充)
    dataset_std: List[float] = field(
        default_factory=list, init=False
    )  # 数据集方差 (Post-init 填充)
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

    def save(self, filename: str = "config.yaml"):
        """保存当前配置到文件"""
        import yaml

        path = Path(self.save_dir) / filename
        # 转换为字典并过滤掉不可序列化的字段
        data = {
            k: v
            for k, v in asdict(self).items()
            if not callable(v) and k not in ("_experiments", "_eval_ckpts")
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, yaml_path: str) -> tuple["Config", List["Experiment"], list]:
        """加载层级配置: constants -> base -> generation"""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}

        # 核心合并逻辑: Base 覆盖 Constants
        merged = {**d.get("constants", {}), **d.get("base", {})}
        cfg = cls(**merged)

        # 处理子配置与嵌套列表
        if "generation" in d:
            cfg.generation = GenerationConfig(**d["generation"])
        exps = [Experiment(**e) for e in d.get("experiments", [])]

        # 处理 eval_checkpoints: 支持简化格式 (仅实验名列表)
        raw_ckpts = d.get("eval_checkpoints", [])
        ts = d.get("training_timestamp", "")
        save_root = merged.get("save_root", "./output")

        eval_ckpts = []
        for item in raw_ckpts:
            if isinstance(item, str):
                # 简化格式: 仅实验名，自动拼接路径
                exp_name = item
                path = f"{save_root}/training/{ts}/{exp_name}/checkpoints/best"
                eval_ckpts.append({"name": exp_name, "path": path})
            else:
                # 原始格式: {name, path} 字典，保持不变
                eval_ckpts.append(item)

        return cfg, exps, eval_ckpts

    def __post_init__(self) -> None:
        """配置校验与派生字段注入"""
        self._setup_hardware()
        self._auto_configure_for_dataset()

        # 自动生成实验目录路径 (按阶段分离，但不立即创建)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.experiment_name or "exp"

        # 主目录结构: output/training/{ts}/{exp_name}/ 和 output/evaluation/{ts}/
        self.save_dir = str(Path(self.save_root) / "training" / ts / exp_name)
        self.evaluation_dir = str(Path(self.save_root) / "evaluation" / ts)
        # 注意: 目录创建由调用方负责 (main.py 中的 ensure_dir)

    def _setup_hardware(self):
        """硬件资源探测"""
        self.gpu_ids = list(range(torch.cuda.device_count()))
        if not self.gpu_ids:
            raise RuntimeError("❌ No GPU found")

    def _auto_configure_for_dataset(self) -> None:
        """数据集注入 (耦合隔离版)"""
        from ..datasets import DATASET_REGISTRY

        name = self.dataset_name.lower()
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unsupported: {name}")

        ds = DATASET_REGISTRY[name]
        self.num_classes, self.image_size = ds.NUM_CLASSES, ds.IMAGE_SIZE
        self.dataset_mean, self.dataset_std = ds.MEAN, ds.STD

        # 按需覆盖数据集定义的特殊参数
        for k, v in getattr(ds, "CONFIG_OVERRIDES", {}).items():
            if hasattr(self, k):
                setattr(self, k, v)


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
