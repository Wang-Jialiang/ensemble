"""
================================================================================
统一数据生成脚本 (SDXL Lightning 版)
================================================================================

支持数据类型的生成:
- Corruption: 使用 imagecorruptions 库生成损坏数据
- OOD: 使用 SDXL Lightning Text2Img 生成分布外数据

使用示例:
    python -m ensemble.datasets.robustness.generate --type corruption --dataset cifar10
    python -m ensemble.datasets.robustness.generate --type ood --dataset cifar10
"""

import argparse
import multiprocessing
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import skimage.filters
import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from PIL import Image
from safetensors.torch import load_file

from ...config import Config
from ...utils import console, ensure_dir, get_logger
from ..preloaded import DATASET_REGISTRY
from .corruption import CORRUPTIONS, SEVERITIES

# 注意: 为解决 CUDA 在 fork 子进程中无法重新初始化的问题，
# 使用 multiprocessing.get_context("spawn") 创建局部上下文
# 而非全局 set_start_method，避免影响其他模块


# =============================================================================
# Monkey-patch dependencies for imagecorruptions compatibility
# =============================================================================

original_gaussian = skimage.filters.gaussian


def patched_gaussian(*args, **kwargs):
    if "multichannel" in kwargs:
        multichannel = kwargs.pop("multichannel")
        if multichannel and "channel_axis" not in kwargs:
            kwargs["channel_axis"] = -1
    return original_gaussian(*args, **kwargs)


skimage.filters.gaussian = patched_gaussian

if not hasattr(np, "float_"):
    np.float_ = np.float64


def _suppress_known_warnings():
    """集中管理已知的无害警告"""
    # 依赖库兼容性警告
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", category=UserWarning, module="imagecorruptions")
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="imagecorruptions"
    )
    # 数值计算警告 (Corruption 生成中的边界情况)
    warnings.filterwarnings("ignore", "invalid value encountered in divide")
    warnings.filterwarnings("ignore", "invalid value encountered in cast")
    # Diffusers 版本警告
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")


_suppress_known_warnings()


# =============================================================================
# 可视化工具
# =============================================================================


def save_visual_comparison(
    original_imgs: np.ndarray,
    processed_imgs: np.ndarray,
    output_path: Path,
    title: str,
    num_samples: int = 8,
):
    """保存原始图像与处理后图像的对比网格"""
    n = min(len(original_imgs), num_samples)
    if n == 0:
        return

    indices = np.linspace(0, len(original_imgs) - 1, n, dtype=int)
    orig = original_imgs[indices]
    proc = processed_imgs[indices]

    h, w = orig.shape[1:3]
    grid = Image.new("RGB", (w * n, h * 2))

    for i, (o, p) in enumerate(zip(orig, proc)):
        grid.paste(Image.fromarray(o.astype(np.uint8)), (i * w, 0))
        grid.paste(Image.fromarray(p.astype(np.uint8)), (i * w, h))

    ensure_dir(output_path.parent)
    grid.save(str(output_path))
    get_logger().info(f"📊 可视化保存: {output_path}")


# =============================================================================
# 图像处理工具
# =============================================================================


def _convert_to_numpy_batch(images_pil: list, target_size: tuple):
    """将 PIL 批量图像恢复到目标尺寸并转回 numpy 格式"""
    return [np.array(img.resize(target_size, Image.LANCZOS)) for img in images_pil]


def _get_gpu_id(device: str):
    """从设备字符串中提取 GPU ID"""
    try:
        return int(device.split(":")[-1])
    except (ValueError, IndexError):
        return 0


def _check_existing_dataset(output_dir: Path, force: bool):
    """检查数据集是否已存在"""
    if output_dir.exists() and not force:
        get_logger().info(f"⏭️ 数据集已存在: {output_dir}，跳过生成")
        return True
    ensure_dir(output_dir)
    return False


def _load_test_set_numpy(DatasetClass, root, seed=42):
    """加载测试集并转换为 numpy 格式"""
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    dataset = DatasetClass(root=root, train=False, **extra_kwargs)
    images_np = dataset.images.permute(0, 2, 3, 1).numpy()
    labels_np = dataset.targets.numpy()
    return images_np, labels_np


# =============================================================================
# Corruption 生成器
# =============================================================================


class CorruptionGenerator:
    """Corruption 生成器 - 基于 imagecorruptions 库"""

    CORRUPTIONS = CORRUPTIONS
    SEVERITIES = SEVERITIES

    @staticmethod
    def apply(img: np.ndarray, corruption_type: str, severity: int = 5) -> np.ndarray:
        """对单张图像应用 corruption"""
        try:
            from imagecorruptions import corrupt
        except ImportError:
            raise ImportError("需要安装 imagecorruptions: pip install imagecorruptions")

        if corruption_type not in CorruptionGenerator.CORRUPTIONS:
            raise ValueError(f"Unknown corruption: {corruption_type}")

        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        corrupted = corrupt(
            img_uint8, corruption_name=corruption_type, severity=severity
        )
        return corrupted.astype(np.float32)

    @staticmethod
    def apply_batch(
        images: np.ndarray,
        corruption_type: str,
        severity: int = 5,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """批量应用 corruption"""
        if seed is not None:
            np.random.seed(seed)
        return np.stack(
            [
                CorruptionGenerator.apply(img, corruption_type, severity)
                for img in images
            ]
        )


# =============================================================================
# SDXL Lightning Pipeline 加载器
# =============================================================================


class LightningPipelineLoader:
    """SDXL Lightning Pipeline 加载器 (单例模式)"""

    _text2img_cache = {}

    @classmethod
    def get_text2img(
        cls, device: str, base_model: str, repo: str, ckpt: str
    ) -> "StableDiffusionXLPipeline":
        """获取 Text2Img Pipeline (强制本地单文件加载)"""
        return cls._get_pipeline(
            device,
            base_model,
            repo,
            ckpt,
            StableDiffusionXLPipeline,
            cls._text2img_cache,
            "Text2Img",
        )

    @classmethod
    def _get_pipeline(
        cls,
        device: str,
        base_model: str,
        repo: str,
        ckpt: str,
        pipe_cls,
        cache,
        name: str,
    ):
        """通用 Pipeline 加载逻辑 (支持全离线 YAML 配置)"""
        if device not in cache:
            get_logger().info(
                f"📥 [{device}] 正在从本地文件加载 SDXL Lightning {name}..."
            )

            if not os.path.isfile(base_model):
                raise FileNotFoundError(f"基础权重文件未找到: {base_model}")

            # 寻找配套的 CLIP 字典配置目录 (用于离线 Tokenizer/TextEncoder)
            # 优先寻找与模型同目录下的 'config' 文件夹
            config_dir = os.path.join(os.path.dirname(base_model), "config")
            local_config = None
            if os.path.isdir(config_dir):
                get_logger().info(f"📚 发现本地组件配置目录: {config_dir}")
                local_config = config_dir

            # 强制本地单文件加载 (如果提供了 original_config 和 local_config，则可全离线运行)
            pipe = pipe_cls.from_single_file(
                base_model,
                config=local_config,
                torch_dtype=torch.float16,
                local_files_only=True,
            ).to(device)

            # 注入 Lightning 权重
            cls._apply_lightning_to_pipe(pipe, device, repo, ckpt)

            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            pipe.set_progress_bar_config(disable=True)
            cls._try_enable_optimizations(pipe)
            cache[device] = pipe
        return cache[device]

    @classmethod
    def _apply_lightning_to_pipe(cls, pipe, device: str, repo: str, ckpt: str):
        """将 Lightning 权重注入到现有 Pipeline 的 UNet 中"""
        local_path = Path(repo) / ckpt
        if not local_path.exists():
            raise FileNotFoundError(f"Lightning 权重文件未找到: {local_path}")

        get_logger().info(f"🚀 加载本地 Lightning 权重: {local_path}")
        state_dict = load_file(str(local_path), device=device)
        pipe.unet.load_state_dict(state_dict)

    @staticmethod
    def _try_enable_optimizations(pipe):
        """尝试启用显存优化和加速"""

        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_xformers_memory_efficient_attention()
        get_logger().info("   ⚡ 已启用 xformers 加速")


# torch.compile: 已禁用
# 原因: SDXL UNet 首次编译需要 10-30 分钟，对于 Lightning 4-step 推理收益很小
# 如果需要大量生成，可以考虑启用，但需要等待首次编译完成
# try:
#     import torch
#     if hasattr(torch, "compile") and torch.cuda.is_available():
#         pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
#         get_logger().info("   ⚡ 已启用 torch.compile 加速")
# except Exception:
#     pass


# =============================================================================
# OOD 生成器 (SDXL Lightning)
# =============================================================================


class OODGenerator:
    """OOD 生成器 - 基于 SDXL Lightning Text2Img"""

    def __init__(
        self,
        device: str = "cuda",
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lightning_repo: str = "ByteDance/SDXL-Lightning",
        lightning_ckpt: str = "sdxl_lightning_4step_unet.safetensors",
        prompts: Optional[list] = None,
        num_steps: int = 4,
        sdxl_height: int = 1024,
        sdxl_width: int = 1024,
        guidance_scale: float = 0.0,
    ):
        self.device = device
        self.base_model = base_model
        self.lightning_repo = lightning_repo
        self.lightning_ckpt = lightning_ckpt
        if prompts is None:
            raise ValueError(
                "❌ 必须提供 prompts 参数 (从 yaml 按数据集获取对应的 ood_prompts)"
            )
        self.prompts = prompts
        self.num_steps = num_steps
        self.sdxl_height = sdxl_height
        self.sdxl_width = sdxl_width
        self.guidance_scale = guidance_scale
        self._pipe = None

    def _get_pipe(self):
        """获取 Text2Img Pipeline"""
        if self._pipe is None:
            self._pipe = LightningPipelineLoader.get_text2img(
                self.device, self.base_model, self.lightning_repo, self.lightning_ckpt
            )
        return self._pipe

    def generate_batch(
        self,
        num_samples: int,
        target_size: int = 64,
        batch_size: int = 24,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """批量生成 OOD 图像 (resize 后的小尺寸)"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        results = []

        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"      [{self.device}] OOD 生成", total=num_samples
            )

            for i in range(0, num_samples, batch_size):
                current_bs = min(batch_size, num_samples - i)
                prompts = [random.choice(self.prompts) for _ in range(current_bs)]

                outputs = pipe(
                    prompt=prompts,
                    height=self.sdxl_height,
                    width=self.sdxl_width,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_steps,
                ).images

                results.extend(
                    _convert_to_numpy_batch(outputs, (target_size, target_size))
                )
                progress.update(task_id, advance=current_bs)

        return np.stack(results)

    def generate_hires_samples(
        self,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """生成少量高分辨率原图 (仅用于可视化)"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        results = []

        for _ in range(num_samples):
            prompt = random.choice(self.prompts)
            output = pipe(
                prompt=prompt,
                height=self.sdxl_height,
                width=self.sdxl_width,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_steps,
            ).images[0]
            results.append(np.array(output))

        return np.stack(results)


# =============================================================================
# 并行处理助手
# =============================================================================


def _process_single_corruption(args):
    """单种 corruption 处理函数 (用于 multiprocessing)"""
    corruption, images_np, severities, output_dir, seed = args

    # 全量处理: 不再支持切片
    images_to_process = images_np

    all_severities = []
    for severity in severities:
        corrupted = CorruptionGenerator.apply_batch(
            images_to_process, corruption, severity, seed=seed
        )
        all_severities.append(corrupted.astype(np.uint8))

    stacked = np.concatenate(all_severities, axis=0)
    np.save(str(output_dir / f"{corruption}.npy"), stacked)
    return corruption


def _worker_ood_gpu(
    gpu_id,
    n,
    target_size,
    bs,
    seed,
    q,
    base_model,
    lightning_repo,
    lightning_ckpt,
    prompts,
    num_steps,
):
    """OOD 工作者线程 (用于 GPU 并行)"""
    generator = OODGenerator(
        device=f"cuda:{gpu_id}",
        base_model=base_model,
        lightning_repo=lightning_repo,
        lightning_ckpt=lightning_ckpt,
        prompts=prompts,
        num_steps=num_steps,
    )
    imgs_resized = generator.generate_batch(
        num_samples=n, target_size=target_size, batch_size=bs, seed=seed + gpu_id
    )
    q.put(imgs_resized)


# =============================================================================
# 生成函数
# =============================================================================


def generate_corruption_dataset(
    dataset_name: str,
    root: str = "./data",
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 corruption 数据集 (使用 CPU 多进程加速) - 无论如何都生成全量数据 (Full Coverage)"""
    import multiprocessing
    import os

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    get_logger().info(
        f"🔧 生成 Corruption: {DatasetClass.NAME}-C (Full Coverage Strategy)..."
    )

    images_np, labels_np = _load_test_set_numpy(DatasetClass, root, seed)

    # 全量模式: 仅仅传入 full data
    # 不再由于类别均衡切分。
    # 所有的 Corruptions 都应用在所有 Images 上

    tasks = []

    for corruption in CORRUPTIONS:
        # Full Mode: 传递 None 作为 slice_obj，表示处理全量
        tasks.append((corruption, images_np, SEVERITIES, output_dir, seed))

    # ===============================================

    num_cpus = os.cpu_count()

    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("   Corruption 总进度", total=len(tasks))

        # 使用 spawn 上下文创建进程池 (不污染全局设置)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(len(tasks), num_cpus)) as pool:
            for _ in pool.imap_unordered(_process_single_corruption, tasks):
                progress.update(task_id, advance=1)

    np.save(str(output_dir / "labels.npy"), labels_np)

    get_logger().info(
        f"✅ {DatasetClass.NAME}-C 生成完成: {len(tasks)} corruptions (Full Coverage)"
    )
    return output_dir


def generate_ood_dataset(
    dataset_name: str,
    root: str = "./data",
    num_samples: int = 1000,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 24,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lightning_repo: str = "ByteDance/SDXL-Lightning",
    lightning_ckpt: str = "sdxl_lightning_4step_unet.safetensors",
    prompts: Optional[list] = None,
    num_steps: int = 4,
    ood_type: str = "near",  # "near" 或 "far"
) -> Path:
    """预生成 OOD 数据集 (仅保存 resize 后的小图)

    Args:
        ood_type: "near" = Near-OOD, "far" = Far-OOD
    """
    import time

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    # 根据 ood_type 区分输出目录
    ood_suffix = "Near-OOD" if ood_type == "near" else "Far-OOD"
    output_dir = Path(root) / f"{DatasetClass.NAME}-{ood_suffix}"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    get_logger().info(
        f"🔧 生成 {ood_suffix}: {DatasetClass.NAME} ({num_samples} 张, SDXL Lightning 4-step)"
    )

    start_time = time.time()
    num_gpus = torch.cuda.device_count()

    # 使用 spawn 上下文 (CUDA 要求)
    ctx = multiprocessing.get_context("spawn")

    samples_per_gpu = num_samples // num_gpus
    q = ctx.Queue()
    processes = []

    for i in range(num_gpus):
        gpu_n = samples_per_gpu + (num_samples % num_gpus if i == num_gpus - 1 else 0)
        p = ctx.Process(
            target=_worker_ood_gpu,
            args=(
                i,
                gpu_n,
                DatasetClass.IMAGE_SIZE,
                batch_size,
                seed,
                q,
                base_model,
                lightning_repo,
                lightning_ckpt,
                prompts,
                num_steps,
            ),
        )
        p.start()
        processes.append(p)

    all_imgs = [q.get() for _ in range(num_gpus)]
    for p in processes:
        p.join()
    np.save(str(output_dir / "images.npy"), np.concatenate(all_imgs, axis=0))

    elapsed = time.time() - start_time
    get_logger().info(
        f"✅ {DatasetClass.NAME}-{ood_suffix} 生成完成! {num_samples} 张 ⏱️ 耗时: {elapsed:.1f}s ({elapsed / 60:.1f}分钟)"
    )
    return output_dir


def visualize_corruption(
    dataset_name: str,
    root: str = "./data",
    num_vis: int = 8,
    gen_cfg=None,
    seed: int = 42,
):
    """为 Corruption 生成可视化对比图"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    test_dataset = DatasetClass(root=root, train=False, **extra_kwargs)
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()

    get_logger().info("🎨 正在生成 Corruption 可视化对比图...")

    vis_corruptions = (
        gen_cfg.vis_corruptions if gen_cfg else ["gaussian_noise", "fog", "glass_blur"]
    )

    for c in vis_corruptions:
        for s in SEVERITIES:
            corrupted = CorruptionGenerator.apply_batch(
                images_np[:num_vis], c, s, seed=seed
            )
            save_visual_comparison(
                images_np[:num_vis],
                corrupted,
                vis_dir / f"{c}_s{s}.png",
                f"{c} (severity={s})",
                num_samples=num_vis,
            )


# =============================================================================
# CLI 入口
# =============================================================================


def main():
    """CLI 入口"""
    args = _parse_args()
    config = _load_config()
    _execute_generation(args, config)

    if config.generation.visualize:
        _execute_visualization(args, config)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="项目鲁棒性数据生成器 (SDXL Lightning)"
    )
    parser.add_argument(
        "--type", type=str, required=True, choices=["corruption", "ood"]
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys())
    )
    parser.add_argument(
        "--ood-type",
        type=str,
        default="both",
        choices=["near", "far", "both"],
        help="OOD 类型: near=Near-OOD, far=Far-OOD, both=两者都生成 (默认)",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存强制生成")
    return parser.parse_args()


def _load_config():
    cfg_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config, _, _ = Config.load_yaml(str(cfg_path))
    return config


def _execute_generation(args, config):
    gen_cfg = config.generation
    if args.type == "corruption":
        generate_corruption_dataset(
            args.dataset, config.data_root, config.seed, args.force
        )
    elif args.type == "ood":
        ood_types = ["near", "far"] if args.ood_type == "both" else [args.ood_type]

        for ood_type in ood_types:
            # 根据 OOD 类型选择对应的 prompts
            prompts_dict = (
                gen_cfg.near_ood_prompts
                if ood_type == "near"
                else gen_cfg.far_ood_prompts
            )
            ood_prompts = prompts_dict.get(args.dataset) if prompts_dict else None

            if ood_prompts is None:
                raise ValueError(
                    f"❌ 未找到数据集 '{args.dataset}' 的 {ood_type}_ood_prompts，请在 default.yaml 中配置"
                )

            generate_ood_dataset(
                args.dataset,
                config.data_root,
                gen_cfg.samples_per_group,
                config.seed,
                args.force,
                gen_cfg.batch_size,
                gen_cfg.base_model,
                gen_cfg.lightning_repo,
                gen_cfg.lightning_ckpt,
                ood_prompts,
                gen_cfg.num_steps,
                ood_type=ood_type,  # 传递 OOD 类型
            )


def _execute_visualization(args, config):
    if args.type == "corruption":
        visualize_corruption(
            args.dataset,
            config.data_root,
            config.generation.num_vis,
            config.generation,
            config.seed,
        )
    elif args.type == "ood":
        ood_types = ["near", "far"] if args.ood_type == "both" else [args.ood_type]
        for ood_type in ood_types:
            visualize_ood(
                args.dataset,
                config.data_root,
                config.generation.num_vis,
                config.generation,
                ood_type=ood_type,
            )


def save_visual_grid(
    images: np.ndarray,
    output_path: Path,
    title: str,
    num_samples: int = 8,
    nrow: int = 4,
):
    """保存单组图像的网格"""
    n = min(len(images), num_samples)
    if n == 0:
        return

    # Randomly select n images if we have more than n
    if len(images) > n:
        indices = np.linspace(0, len(images) - 1, n, dtype=int)
        imgs = images[indices]
    else:
        imgs = images

    h, w = imgs.shape[1:3]
    # 动态调整列数：实际列数不超过样本数
    actual_cols = min(nrow, n)
    actual_rows = (n + actual_cols - 1) // actual_cols

    grid = Image.new("RGB", (w * actual_cols, h * actual_rows))

    for i, img in enumerate(imgs):
        r = i // actual_cols
        c = i % actual_cols
        grid.paste(Image.fromarray(img.astype(np.uint8)), (c * w, r * h))

    ensure_dir(output_path.parent)
    grid.save(str(output_path))
    get_logger().info(f"📊 可视化保存: {output_path}")


def visualize_ood(
    dataset_name: str,
    root: str = "./data",
    num_vis: int = 8,
    gen_cfg=None,
    ood_type: str = "near",  # "near" 或 "far"
):
    """为 OOD 生成可视化网格

    Args:
        ood_type: "near" = Near-OOD, "far" = Far-OOD

    1. 展示 resize 后的小图 (从 images.npy)
    2. 实时生成 num_vis 个高分辨率原图并展示
    """
    DatasetClass = DATASET_REGISTRY[dataset_name]
    ood_suffix = "Near-OOD" if ood_type == "near" else "Far-OOD"
    output_dir = Path(root) / f"{DatasetClass.NAME}-{ood_suffix}"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    images_path = output_dir / "images.npy"

    if not images_path.exists():
        get_logger().warning(f"⚠️ {ood_suffix} 数据未找到: {images_path}")
        return

    get_logger().info(f"🎨 正在生成 {ood_suffix} 可视化...")

    # 1. 加载并展示 resize 后的小图
    images = np.load(str(images_path), mmap_mode="r")
    total_images = len(images)
    indices = np.linspace(0, total_images - 1, min(total_images, num_vis), dtype=int)
    vis_images = images[indices]

    save_visual_grid(
        vis_images,
        vis_dir / f"{ood_type}_ood_samples_resized.png",
        f"{ood_suffix} Samples (Resized)",
        num_samples=num_vis,
        nrow=4,
    )

    # 2. 实时生成 num_vis 个高分辨率原图
    if gen_cfg is not None:
        # 根据 ood_type 选择对应的 prompts
        prompts_dict = (
            gen_cfg.near_ood_prompts if ood_type == "near" else gen_cfg.far_ood_prompts
        )
        ood_prompts = prompts_dict.get(dataset_name) if prompts_dict else None

        if ood_prompts is None:
            get_logger().warning(
                f"⚠️ 未找到数据集 '{dataset_name}' 的 {ood_type}_ood_prompts，跳过高分辨率可视化"
            )
            return
        get_logger().info(f"   📷 生成 {num_vis} 张高分辨率原图用于可视化...")
        generator = OODGenerator(
            device="cuda" if torch.cuda.is_available() else "cpu",
            base_model=gen_cfg.base_model,
            lightning_repo=gen_cfg.lightning_repo,
            lightning_ckpt=gen_cfg.lightning_ckpt,
            prompts=ood_prompts,
            num_steps=gen_cfg.num_steps,
        )
        hires_samples = generator.generate_hires_samples(num_vis, seed=42)

        save_visual_grid(
            hires_samples,
            vis_dir / f"{ood_type}_ood_samples_hires.png",
            f"{ood_suffix} Samples (High-Resolution 1024x1024)",
            num_samples=num_vis,
            nrow=4,
        )
        get_logger().info(
            f"   ✅ 高分辨率原图: {vis_dir / f'{ood_type}_ood_samples_hires.png'}"
        )


if __name__ == "__main__":
    main()
