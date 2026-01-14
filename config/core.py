"""
================================================================================
配置模块
================================================================================
"""

import datetime
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import torch


@dataclass
class GenerationConfig:
    """数据生成配置 (Corruption / OOD) - SDXL Lightning"""

    base_model: Optional[str] = None  # SDXL 基础模型 (由 yaml 填充)
    lightning_repo: Optional[str] = None  # Lightning UNet 仓库
    lightning_ckpt: Optional[str] = None  # 4-step 检查点
    num_steps: Optional[int] = None  # 推理步数 (2/4/8)
    batch_size: Optional[int] = None  # 与 yaml 同步
    samples_per_group: Optional[int] = None  # OOD 生成总图片数
    visualize: Optional[bool] = None
    num_vis: Optional[int] = None  # 与 yaml 同步

    # SDXL 生成参数
    sdxl_height: Optional[int] = None  # Text2Img 原始输出高度
    sdxl_width: Optional[int] = None  # Text2Img 原始输出宽度
    guidance_scale_text2img: Optional[float] = None  # Text2Img CFG

    ood_prompts: Optional[list[str]] = None  # 由 default.yaml 填充
    vis_corruptions: Optional[list[str]] = None  # 由 default.yaml 填充


@dataclass
class Config:
    """三阶段课程学习集成训练配置"""

    # ==========================================================================
    # [全局] 数据配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    data_root: Optional[str] = None  # 数据集根目录路径
    save_root: Optional[str] = None  # 检查点/输出保存根目录
    dataset_name: Optional[str] = (
        None  # 数据集名称: "cifar10", "cifar100", "eurosat" 等
    )
    val_split: Optional[float] = None  # 验证集划分比例 (0.0-1.0)
    test_split: Optional[float] = None  # 测试集划分比例，用于无官方划分的数据集

    # ==========================================================================
    # [全局] 模型配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    model_name: Optional[str] = None  # 模型名称: "resnet18", "resnet50", "vgg16" 等
    num_models_per_gpu: Optional[int] = None  # 每个 GPU 上的模型数量
    compile_model: Optional[bool] = (
        None  # 是否启用 PyTorch 2.0+ 编译优化 (可提升10-50%速度)
    )

    # ==========================================================================
    # [全局] 训练超参数 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    batch_size: Optional[int] = None  # 批次大小
    lr: Optional[float] = None  # 基础学习率
    weight_decay: Optional[float] = None  # 权重衰减 (L2 正则化系数)
    max_grad_norm: Optional[float] = None  # 梯度裁剪阈值
    seed: Optional[int] = None  # 随机种子
    optimizer: Optional[str] = None  # 优化器: "adamw", "sgd", "adam", "rmsprop"
    scheduler: Optional[str] = None  # 调度器: "cosine", "step", "plateau", "none"
    min_lr: Optional[float] = None  # 学习率调度器最小值 (cosine 衰减终点)
    label_smoothing: Optional[float] = None  # 标签平滑系数 (0.0=不使用, 0.1=常用值)

    # ==========================================================================
    # [阶段训练专用] 三阶段与 Mask - 仅 StagedEnsembleTrainer 使用
    # ==========================================================================
    warmup_epochs: Optional[int] = None  # Warmup 阶段轮数
    progressive_epochs: Optional[int] = None  # Progressive 阶段轮数
    finetune_epochs: Optional[int] = None  # Finetune 阶段轮数
    mask_pool_size: Optional[int] = None  # 预生成的 Mask 池大小
    mask_start_ratio: Optional[float] = None  # Progressive 阶段起始遮罩比例
    mask_end_ratio: Optional[float] = None  # Progressive 阶段结束遮罩比例
    mask_prob_start: Optional[float] = None  # Progressive 阶段起始应用概率
    mask_prob_end: Optional[float] = None  # Progressive 阶段结束应用概率
    finetune_mask_ratio: Optional[float] = None  # Finetune 阶段固定遮罩比例
    finetune_mask_prob: Optional[float] = None  # Finetune 阶段固定应用概率

    # ==========================================================================
    # [全局] 数据加载配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    num_workers: Optional[int] = None  # DataLoader 工作进程数
    pin_memory: Optional[bool] = None  # 是否使用锁页内存加速 GPU 传输
    persistent_workers: Optional[bool] = None  # 是否保持工作进程存活
    prefetch_factor: Optional[int] = None  # 每个 worker 预取的批次数

    # ==========================================================================
    # [全局] 训练优化配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    use_amp: Optional[bool] = None  # 是否启用自动混合精度 (AMP, 默认使用 BFloat16)
    use_tf32: Optional[bool] = None  # 是否启用 TF32 加速 (仅 Ampere+ GPU)
    early_stopping_patience: Optional[int] = None  # 早停耐心值 (验证集无改善的轮数)

    # ==========================================================================
    # [全局] 保存与日志配置 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    use_wandb: Optional[bool] = None  # 是否启用 Weights & Biases 日志
    wandb_project: Optional[str] = None  # wandb 项目名称
    log_level: Optional[str] = None  # 日志级别: "DEBUG", "INFO", "WARNING", "ERROR"
    log_to_console: Optional[bool] = None  # 训练日志是否同时输出到控制台 (默认 True)

    # ==========================================================================
    # [评估专用] 评估配置 - 仅评估模块使用
    # ==========================================================================
    ece_n_bins: Optional[int] = None  # 校准度 (ECE) 计算的分箱数量
    ensemble_strategy: Optional[str] = (
        None  # 集成策略: "mean" (等权平均), "voting" (多数投票)
    )
    corruption_dataset: Optional[bool] = None  # 是否加载 Corruption 数据集进行评估
    ood_dataset: Optional[bool] = None  # 是否加载 OOD 数据集进行评估
    eval_run_gradcam: Optional[bool] = None  # 是否在评估时运行 Grad-CAM 分析
    eval_run_landscape: Optional[bool] = None  # 是否在评估时运行 Loss Landscape 分析

    # ==========================================================================
    # [评估专用] 对抗鲁棒性评估参数 - 仅评估模块使用
    # ==========================================================================
    adv_eps: Optional[float] = None  # FGSM/PGD 扰动强度 ε (常用值: 8/255 ≈ 0.031)
    adv_alpha: Optional[float] = None  # PGD 步长 α (常用值: 2/255 ≈ 0.008)
    adv_pgd_steps: Optional[int] = None  # PGD 迭代步数 (常用值: 10, 20)
    adv_eps_list: Optional[list[float]] = None  # 多 ε 评估列表 (可选)
    adv_targeted: Optional[bool] = None  # 是否为针对性攻击

    # ==========================================================================
    # [全局] 优化器高级参数 - SGD 专用
    # ==========================================================================
    sgd_momentum: Optional[float] = None  # SGD 动量 (默认 0.9)

    # ==========================================================================
    # [增强专用] 数据增强参数 - Perlin/Cutout/GridMask 使用
    # ==========================================================================
    perlin_persistence: Optional[float] = None  # Perlin 噪声持久度 (默认 0.5)
    perlin_scale_ratio: Optional[float] = None  # Perlin 噪声尺度比例 (默认 0.3)
    gridmask_d_ratio_min: Optional[float] = (
        None  # GridMask 网格单元最小尺寸比例 (默认 0.2)
    )
    gridmask_d_ratio_max: Optional[float] = (
        None  # GridMask 网格单元最大尺寸比例 (默认 0.4)
    )
    perlin_octaves: Optional[int] = None  # Perlin octaves 数量 (默认 4)
    augmentation_use_mean_fill: Optional[bool] = None  # 遮挡填充: False=黑色, True=均值

    # ==========================================================================
    # [评估专用] 可视化参数
    # ==========================================================================
    plot_dpi: Optional[int] = None  # 图表保存 DPI (默认 300)
    gradcam_num_samples: Optional[int] = None  # Grad-CAM 分析样本数

    # ==========================================================================
    # [全局] 模型初始化 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    init_method: Optional[str] = (
        None  # 初始化方法: "kaiming", "xavier", "orthogonal", "default"
    )

    # ==========================================================================
    # [全局] 运行控制 - 被 StagedEnsembleTrainer 使用
    # ==========================================================================
    quick_test: Optional[bool] = None  # 快速测试模式 (减少轮数/模型数)

    # ==========================================================================
    # [实验级别] 增强与课程学习参数 - 每个实验可覆盖
    # ==========================================================================
    augmentation_method: Optional[str] = None  # 增强方法: "perlin", "cutout", "none" 等
    use_curriculum: Optional[bool] = None  # 是否使用课程学习
    fixed_ratio: Optional[float] = None  # 固定遮挡比例 (仅 use_curriculum=False 时生效)
    fixed_prob: Optional[float] = None  # 固定遮挡概率 (仅 use_curriculum=False 时生效)
    share_warmup_backbone: Optional[bool] = None  # 是否在 warmup 后共享 backbone

    # ==========================================================================
    # [数据生成] SDXL Lightning 生成配置
    # ==========================================================================
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # 自动计算/生成字段 (有默认值, 禁止人工初始化)
    save_dir: str = field(
        default="", init=False
    )  # 训练产物目录: output/training/{ts}/{exp_name}/
    training_base_dir: str = field(
        default="", init=False
    )  # 时间戳目录: output/training/{ts}/ (日志/历史文件)
    evaluation_dir: str = field(
        default="", init=False
    )  # 评估产物目录: output/evaluation/{ts}/
    num_classes: int = field(default=0, init=False)
    image_size: int = field(default=0, init=False)
    num_channels: int = field(default=3, init=False)  # 图像通道数 (RGB=3)
    dataset_mean: list[float] = field(
        default_factory=list, init=False
    )  # 数据集均值 (Post-init 填充)
    dataset_std: list[float] = field(
        default_factory=list, init=False
    )  # 数据集方差 (Post-init 填充)
    gpu_ids: list[int] = field(
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
            progressive_epochs=1,
            finetune_epochs=1,
            num_models_per_gpu=1,
        )

    @classmethod
    def load_yaml(cls, yaml_path: str) -> tuple["Config", list["Experiment"], list]:
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

        # 处理 eval_checkpoints: 仅支持简化格式 (仅实验名列表)
        raw_ckpts = d.get("eval_checkpoints", [])
        ts = d.get("training_timestamp", "")
        save_root = merged.get("save_root", "./output")

        eval_ckpts = []
        for exp_name in raw_ckpts:
            if not isinstance(exp_name, str):
                continue
            path = f"{save_root}/training/{ts}/{exp_name}/checkpoints/best"
            eval_ckpts.append({"name": exp_name, "path": path})

        return cfg, exps, eval_ckpts

    def __post_init__(self) -> None:
        """配置校验与派生字段注入"""
        self._setup_hardware()

        # 自动生成实验目录路径 (按阶段分离，但不立即创建)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.experiment_name or "exp"

        # 主目录结构: output/training/{ts}/{exp_name}/ 和 output/evaluation/{ts}/
        self.training_base_dir = str(Path(self.save_root) / "training" / ts)
        self.save_dir = str(Path(self.training_base_dir) / exp_name)
        self.evaluation_dir = str(Path(self.save_root) / "evaluation" / ts)
        # 注意: 目录创建由调用方负责 (main.py 中的 ensure_dir)

    def _setup_hardware(self):
        """硬件资源探测"""
        self.gpu_ids = list(range(torch.cuda.device_count()))
        if not self.gpu_ids:
            raise RuntimeError("❌ No GPU found")


@dataclass
class Experiment:
    """实验配置

    字段名与 Config 保持一致，方便直接 copy 覆盖
    """

    name: str

    # 与 Config 同名的字段，可直接覆盖
    augmentation_method: str = "perlin"
    use_curriculum: bool = True
    fixed_ratio: Optional[float] = None
    fixed_prob: Optional[float] = None

    def get_config_overrides(self) -> dict:
        """获取所有需要覆盖的参数 (过滤 name 和 None 值)"""
        exclude = {"name"}
        return {
            k: v for k, v in asdict(self).items() if k not in exclude and v is not None
        }
