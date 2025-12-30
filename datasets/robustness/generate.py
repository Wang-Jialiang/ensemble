"""
================================================================================
统一数据生成脚本
================================================================================

支持三种数据类型的生成:
- Corruption: 使用 imagecorruptions 库生成损坏数据
- Domain Shift: 使用 Stable Diffusion Img2Img 生成风格迁移数据
- OOD: 使用 Stable Diffusion Text2Img 生成分布外数据

使用示例:
    # Corruption
    python -m ensemble.datasets.robustness.generate --type corruption --dataset eurosat

    # Domain Shift
    python -m ensemble.datasets.robustness.generate --type domain --dataset eurosat --styles sketch

    # OOD
    python -m ensemble.datasets.robustness.generate --type ood --dataset eurosat --num_samples 100
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def patch_dependencies():
    """Monkey-patch dependencies for imagecorruptions compatibility.
    1. imagecorruptions uses multichannel=True in skimage.filters.gaussian,
       which was replaced by channel_axis=-1 in scikit-image >= 0.19.0.
    2. imagecorruptions uses np.float_, which was removed in NumPy 2.0.
    """
    # Patch scikit-image
    try:
        import skimage.filters

        original_gaussian = skimage.filters.gaussian

        def patched_gaussian(*args, **kwargs):
            if "multichannel" in kwargs:
                multichannel = kwargs.pop("multichannel")
                if multichannel and "channel_axis" not in kwargs:
                    kwargs["channel_axis"] = -1
            return original_gaussian(*args, **kwargs)

        skimage.filters.gaussian = patched_gaussian
    except (ImportError, AttributeError):
        pass

    # Patch NumPy
    try:
        import numpy as np

        if not hasattr(np, "float_"):
            np.float_ = np.float64
    except (ImportError, AttributeError):
        pass


import warnings

patch_dependencies()

import torch

# 过滤 imagecorruptions/pkg_resources 的过时警告
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="imagecorruptions")

try:
    from diffusers import (
        AutoPipelineForImage2Image,
        AutoPipelineForText2Image,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
    )
except ImportError:
    pass

from ...config import Config
from ...utils import ensure_dir, get_logger
from ..preloaded import DATASET_REGISTRY
from .corruption import CORRUPTIONS, SEVERITIES

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
    """保存原始图像与处理后图像的对比网格
    args:
        original_imgs: (N, H, W, C) numpy array
        processed_imgs: (N, H, W, C) numpy array
        output_path: 保存路径
        title: 标题描述
        num_samples: 展示的样本数量
    """
    n = min(len(original_imgs), num_samples)
    if n == 0:
        return

    # 选取样本
    indices = np.linspace(0, len(original_imgs) - 1, n, dtype=int)
    orig = original_imgs[indices]
    proc = processed_imgs[indices]

    h, w = orig.shape[1:3]
    # 创建网格 (2行, n列)
    grid = Image.new("RGB", (w * n, h * 2))

    for i in range(n):
        # 第一行: 原图
        grid.paste(Image.fromarray(orig[i].astype(np.uint8)), (i * w, 0))
        # 第二行: 处理图
        grid.paste(Image.fromarray(proc[i].astype(np.uint8)), (i * w, h))

    grid.save(output_path)
    get_logger().info(
        f"📊 可视化结果已保存至: {output_path} (第一行: 原图, 第二行: {title})"
    )


# =============================================================================
# 图像处理与硬件助手 (内部使用)
# =============================================================================


def _prepare_pil_batch(images_np: np.ndarray, target_size: int = 512) -> list:
    """将 numpy 批量图像转换为 PIL 格式并统一缩放"""
    return [
        Image.fromarray(img.astype(np.uint8)).resize(
            (target_size, target_size), Image.Resampling.LANCZOS
        )
        for img in images_np
    ]


def _convert_to_numpy_batch(images_pil: list, target_size: tuple) -> list:
    """将 PIL 批量图像恢复到目标尺寸并转回 numpy 格式"""
    return [
        np.array(img.resize(target_size, Image.Resampling.LANCZOS))
        for img in images_pil
    ]


def _get_gpu_id(device: str) -> int:
    """从设备字符串中提取 GPU ID，用于 tqdm 位置控制"""
    if "cuda:" in device:
        try:
            return int(device.split(":")[-1])
        except (ValueError, IndexError):
            pass
    return 0


def _check_existing_dataset(output_dir: Path, force: bool) -> bool:
    """检查数据集是否已存在，返回 True 表示可以跳过生成"""
    if output_dir.exists() and not force:
        get_logger().info(f"✅ {output_dir} 已存在，跳过生成 (使用 --force 强制重新生成)")
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


def _sample_dataset(images_np, labels_np, n, seed=42):
    """对数据集进行随机抽样"""
    total = len(labels_np)
    target_n = min(n or total, total)
    np.random.seed(seed)
    indices = np.random.permutation(total)[:target_n]
    return images_np[indices], labels_np[indices]


# =============================================================================
# Corruption 生成器
# =============================================================================


class CorruptionGenerator:
    """Corruption 生成器 - 基于 imagecorruptions 库

    使用 imagecorruptions 库实现与 ImageNet-C 相同的 corruption 类型。
    依赖: pip install imagecorruptions
    """

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

        if severity not in CorruptionGenerator.SEVERITIES:
            if not 1 <= severity <= 5:
                raise ValueError(f"Severity must be 1-5, got {severity}")

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

        corrupted = []
        for img in images:
            c_img = CorruptionGenerator.apply(img, corruption_type, severity)
            corrupted.append(c_img)

        return np.stack(corrupted)


# =============================================================================
# Domain Shift 生成器
# =============================================================================


class DomainGenerator:
    """Domain Shift 生成器 - 基于 Stable Diffusion Img2Img"""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        styles: Optional[dict] = None,
        strengths: Optional[list] = None,
    ):
        self.device = device
        self.model_path = model_path
        self.styles = styles or {}
        self._pipe = None

    def _get_pipe(self):
        """延迟加载标准 SD pipeline"""
        if self._pipe is None:
            get_logger().info(f"📥 加载 SD Img2Img: {self.model_path} ({self.device})")
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            self._enable_optimizations()
        return self._pipe

    def _enable_optimizations(self):
        """启用显存优化"""
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def apply_batch(
        self, images: np.ndarray, style: str, strength: float, batch_size: int = 16
    ) -> np.ndarray:
        """批量风格转换 (主干逻辑)"""
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")

        pipe = self._get_pipe()
        prompt = self.styles[style]
        results = []

        pbar = tqdm(
            range(0, len(images), batch_size),
            desc=f"      [{self.device}] {style}/{strength}",
            position=_get_gpu_id(self.device),
            leave=False,
            mininterval=1.0,
        )

        for i in pbar:
            batch = images[i : i + batch_size]
            orig_h, orig_w = batch.shape[1], batch.shape[2]

            # 1. 预处理
            pils = _prepare_pil_batch(batch, target_size=512)

            # 2. 推理
            outputs = pipe(
                prompt=[prompt] * len(pils),
                image=pils,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images

            # 3. 后处理
            results.extend(_convert_to_numpy_batch(outputs, (orig_w, orig_h)))

        return np.stack(results)

    def apply(self, img: np.ndarray, style: str, strength: float) -> np.ndarray:
        return self.apply_batch(np.expand_dims(img, 0), style, strength, batch_size=1)[
            0
        ]


# =============================================================================
# Domain Shift 生成器 (SDXL Turbo 极速版)
# =============================================================================


class TurboDomainGenerator:
    """Domain Shift 生成器 - 基于 SDXL Turbo (极速版)"""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        styles: Optional[dict] = None,
        strengths: Optional[list] = None,
        num_steps: int = 1,
    ):
        self.device = device
        self.model_path = model_path or "stabilityai/sdxl-turbo"
        self.styles = styles or {}
        self.num_steps = num_steps
        self._pipe = None

    def _get_pipe(self):
        """延迟加载 SDXL Turbo pipeline"""
        if self._pipe is None:
            get_logger().info(f"� 加载 SDXL Turbo Img2Img: {self.model_path}")
            self._pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in self.device else None,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            self._enable_optimizations()
        return self._pipe

    def _enable_optimizations(self):
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def apply_batch(
        self, images: np.ndarray, style: str, strength: float, batch_size: int = 16
    ) -> np.ndarray:
        """批量风格转换 (Turbo 极速版)"""
        if style not in self.styles:
            raise ValueError(f"Unknown style: {style}")

        pipe = self._get_pipe()
        prompt = self.styles[style]
        results = []

        pbar = tqdm(
            range(0, len(images), batch_size),
            desc=f"      [{self.device}] {style}/{strength} (Turbo)",
            position=_get_gpu_id(self.device),
            leave=False,
            mininterval=1.0,
        )

        for i in pbar:
            batch = images[i : i + batch_size]
            orig_h, orig_w = batch.shape[1], batch.shape[2]

            # 1. 预处理 (SDXL Turbo 512x512)
            pils = _prepare_pil_batch(batch, target_size=512)

            # 2. 推理
            outputs = pipe(
                prompt=[prompt] * len(pils),
                image=pils,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=max(int(1 / strength), self.num_steps),
            ).images

            # 3. 后处理
            results.extend(_convert_to_numpy_batch(outputs, (orig_w, orig_h)))

        return np.stack(results)

    def apply(self, img: np.ndarray, style: str, strength: float) -> np.ndarray:
        return self.apply_batch(np.expand_dims(img, 0), style, strength, batch_size=1)[0]


# =============================================================================
# OOD 生成器
# =============================================================================


class OODGenerator:
    """OOD 生成器 - 基于 Stable Diffusion Text2Img"""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        prompts: Optional[list] = None,
    ):
        self.device = device
        self.model_path = model_path
        self.prompts = prompts or []
        self._pipe = None

    def _get_pipe(self):
        """延迟加载标准 SD pipeline"""
        if self._pipe is None:
            get_logger().info(f"📥 加载 SD Text2Img: {self.model_path} ({self.device})")
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            self._enable_optimizations()
        return self._pipe

    def _enable_optimizations(self):
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def generate_batch(
        self,
        num_samples: int,
        target_size: int = 64,
        batch_size: int = 16,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """批量生成 OOD 图像 (标准版)"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        results = []

        pbar = tqdm(
            range(0, num_samples, batch_size),
            desc=f"      [{self.device}] OOD 生成",
            position=_get_gpu_id(self.device),
            leave=False,
            mininterval=1.0,
        )

        for i in pbar:
            current_bs = min(batch_size, num_samples - i)
            prompts = [random.choice(self.prompts) for _ in range(current_bs)]

            # 1. 生成 (Text2Img 不需要预处理图像)
            outputs = pipe(
                prompt=prompts,
                height=512,
                width=512,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images

            # 2. 后处理
            results.extend(_convert_to_numpy_batch(outputs, (target_size, target_size)))

        return np.stack(results)

    def generate(self, target_size: int = 64, seed: Optional[int] = None) -> np.ndarray:
        return self.generate_batch(1, target_size, batch_size=1, seed=seed)[0]


# =============================================================================
# OOD 生成器 (SDXL Turbo 极速版)
# =============================================================================


class TurboOODGenerator:
    """OOD 生成器 - 基于 SDXL Turbo (极速版)"""

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        prompts: Optional[list] = None,
        num_steps: int = 1,
    ):
        self.device = device
        self.model_path = model_path or "stabilityai/sdxl-turbo"
        self.prompts = prompts or []
        self.num_steps = num_steps
        self._pipe = None

    def _get_pipe(self):
        """延迟加载 SDXL Turbo pipeline"""
        if self._pipe is None:
            get_logger().info(f"� 加载 SDXL Turbo Text2Img: {self.model_path}")
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                variant="fp16" if "cuda" in self.device else None,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
            self._enable_optimizations()
        return self._pipe

    def _enable_optimizations(self):
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def generate_batch(
        self,
        num_samples: int,
        target_size: int = 64,
        batch_size: int = 16,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """批量生成 OOD 图像 (Turbo 极速版)"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        results = []

        pbar = tqdm(
            range(0, num_samples, batch_size),
            desc=f"      [{self.device}] OOD 生成 (Turbo)",
            position=_get_gpu_id(self.device),
            leave=False,
            mininterval=1.0,
        )

        for i in pbar:
            current_bs = min(batch_size, num_samples - i)
            prompts = [random.choice(self.prompts) for _ in range(current_bs)]

            # 1. 推理
            outputs = pipe(
                prompt=prompts,
                height=512,
                width=512,
                guidance_scale=0.0,
                num_inference_steps=self.num_steps,
            ).images

            # 2. 后处理
            results.extend(_convert_to_numpy_batch(outputs, (target_size, target_size)))

        return np.stack(results)

    def generate(self, target_size: int = 64, seed: Optional[int] = None) -> np.ndarray:
        return self.generate_batch(1, target_size, batch_size=1, seed=seed)[0]


# =============================================================================
# 并行处理助手
# =============================================================================


def _process_single_corruption(args):
    """单种 corruption 处理函数 (用于 multiprocessing)"""
    corruption, images_np, severities, output_dir, seed = args
    all_severities = []
    for severity in severities:
        corrupted = CorruptionGenerator.apply_batch(
            images_np, corruption, severity, seed=seed
        )
        all_severities.append(corrupted.astype(np.uint8))

    stacked = np.concatenate(all_severities, axis=0)
    np.save(str(output_dir / f"{corruption}.npy"), stacked)
    return corruption


def _worker_domain(
    device,
    styles,
    strengths,
    images_np,
    labels_np,
    output_dir,
    dataset_name,
    batch_size,
    model_path=None,
    full_styles_dict=None,
    use_turbo=False,
    turbo_steps=1,
):
    """Domain 工作者线程 (用于 GPU 并行)"""
    if use_turbo:
        generator = TurboDomainGenerator(
            device=device,
            model_path=model_path,
            styles=full_styles_dict,
            strengths=strengths,
            num_steps=turbo_steps,
        )
    else:
        generator = DomainGenerator(
            device=device,
            model_path=model_path,
            styles=full_styles_dict,
            strengths=strengths,
        )
    DatasetClass = DATASET_REGISTRY[dataset_name]

    for style in styles:
        for strength in strengths:
            mode_str = "Turbo" if use_turbo else "SD"
            get_logger().info(
                f"   [{device}] ({mode_str}) 生成: {style} (strength={strength})..."
            )
            strength_dir = output_dir / style / str(strength)

            for class_idx in range(DatasetClass.NUM_CLASSES):
                ensure_dir(strength_dir / f"class_{class_idx:04d}")

            # 使用包装好的 apply_batch
            styled_images = generator.apply_batch(
                images_np, style, strength, batch_size=batch_size
            )

            # 保存
            for i, (img, label) in enumerate(zip(styled_images, labels_np)):
                img_path = strength_dir / f"class_{label:04d}" / f"img_{i}.png"
                Image.fromarray(img).save(str(img_path))


def _worker_ood_gpu(
    gpu_id,
    n,
    target_size,
    bs,
    s,
    q,
    model_path=None,
    prompts=None,
    use_turbo=False,
    turbo_steps=1,
):
    """OOD 工作者线程 (用于 GPU 并行)"""
    if use_turbo:
        generator = TurboOODGenerator(
            device=f"cuda:{gpu_id}",
            model_path=model_path,
            prompts=prompts,
            num_steps=turbo_steps,
        )
    else:
        generator = OODGenerator(
            device=f"cuda:{gpu_id}", model_path=model_path, prompts=prompts
        )
    imgs = generator.generate_batch(
        num_samples=n, target_size=target_size, batch_size=bs, seed=s + gpu_id
    )
    q.put(imgs)


# =============================================================================
# 生成函数
# =============================================================================


def generate_corruption_dataset(
    dataset_name: str,
    root: str = "./data",
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 corruption 数据集 (使用 CPU 多进程加速)"""
    import multiprocessing
    import os

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    get_logger().info(f"🔧 生成 Corruption: {DatasetClass.NAME}-C (EPYC 并行)...")

    # 1. 加载并转换
    images_np, labels_np = _load_test_set_numpy(DatasetClass, root, seed)
    total_samples = len(labels_np)

    # 2. 准备并行任务
    tasks = [(c, images_np, SEVERITIES, output_dir, seed) for c in CORRUPTIONS]
    num_cpus = os.cpu_count()

    with multiprocessing.Pool(processes=min(len(CORRUPTIONS), num_cpus)) as pool:
        list(tqdm(pool.imap_unordered(_process_single_corruption, tasks), total=len(tasks), desc="   Corruption 总进度"))

    # 3. 保存标签
    np.save(str(output_dir / "labels.npy"), labels_np)

    get_logger().info(f"✅ {DatasetClass.NAME}-C 生成完成: {len(CORRUPTIONS)} corruptions × {total_samples} samples")
    return output_dir


def visualize_corruption(
    dataset_name: str,
    root: str = "./data",
    num_vis: int = 8,
):
    """为 Corruption 生成可视化对比图"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    # 加载测试集
    test_dataset = DatasetClass(root=root, train=False)
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()

    get_logger().info("🎨 正在生成 Corruption 可视化对比图...")

    # 随机选几种 corruption 和 severity
    sample_corruptions = ["gaussian_noise", "shot_noise", "fog", "snow", "glass_blur"]
    sample_corruptions = [c for c in sample_corruptions if c in CORRUPTIONS]

    for c in sample_corruptions:
        for s in [3, 5]:  # 只对比中等和最高强度
            corrupted = CorruptionGenerator.apply_batch(
                images_np[:num_vis], c, s, seed=42
            )
            save_visual_comparison(
                images_np[:num_vis],
                corrupted,
                vis_dir / f"{c}_s{s}.png",
                f"{c} (severity={s})",
                num_samples=num_vis,
            )


def generate_domain_dataset(
    dataset_name: str,
    root: str = "./data",
    samples_per_group: Optional[int] = 1000,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 16,
    model_path: Optional[str] = None,
    styles: Optional[dict] = None,
    strengths: Optional[list] = None,
    use_turbo: bool = False,
    turbo_steps: int = 1,
) -> Path:
    """预生成 domain shift 数据集 (外层接口)"""
    import multiprocessing

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    # 1. 加载并抽样
    images_np, labels_np = _load_test_set_numpy(DatasetClass, root, seed)
    images_np, labels_np = _sample_dataset(images_np, labels_np, samples_per_group, seed)

    # 2. 选择模式与 GPU 分发
    num_gpus = torch.cuda.device_count()
    mode_str = "SDXL Turbo" if use_turbo else "SD 2.1"
    get_logger().info(f"🔧 生成 Domain: {DatasetClass.NAME} ({mode_str}, GPU={num_gpus})")

    styles_list = list(styles.keys()) if styles else []

    if num_gpus == 0:
        _run_domain_serial(
            output_dir, images_np, labels_np, styles, strengths, model_path,
            batch_size, use_turbo, turbo_steps, DatasetClass
        )
    else:
        _run_domain_parallel(
            output_dir, images_np, labels_np, styles_list, strengths, model_path,
            batch_size, use_turbo, turbo_steps, dataset_name, styles
        )

    get_logger().info(f"✅ {DatasetClass.NAME}-Domain 生成完成!")
    return output_dir


def _run_domain_serial(output_dir, images, labels, styles_dict, strengths, model_path, bs, turbo, steps, DatasetClass):
    """单进程回退模式 (CPU 或单 GPU 强制串行)"""
    get_logger().warning("使用串行模式生成，速度较慢...")
    if turbo:
        generator = TurboDomainGenerator(device="cpu", model_path=model_path, styles=styles_dict, strengths=strengths, num_steps=steps)
    else:
        generator = DomainGenerator(device="cpu", model_path=model_path, styles=styles_dict, strengths=strengths)

    for style, prompt in styles_dict.items():
        for str_val in strengths:
            strength_dir = output_dir / style / str(str_val)
            for c in range(DatasetClass.NUM_CLASSES): ensure_dir(strength_dir / f"class_{c:04d}")

            styled = generator.apply_batch(images, style, str_val, batch_size=bs)
            for i, (img, lbl) in enumerate(zip(styled, labels)):
                Path(strength_dir / f"class_{lbl:04d}" / f"img_{i}.png").parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(str(strength_dir / f"class_{lbl:04d}" / f"img_{i}.png"))


def _run_domain_parallel(output_dir, images, labels, styles_list, strengths, model_path, bs, turbo, steps, dataset_name, styles_dict):
    """多 GPU 并行分发"""
    import multiprocessing
    num_gpus = torch.cuda.device_count()
    processes = []
    for i in range(num_gpus):
        gpu_styles = styles_list[i::num_gpus]
        if not gpu_styles: continue
        p = multiprocessing.Process(
            target=_worker_domain,
            args=(f"cuda:{i}", gpu_styles, strengths, images, labels, output_dir, dataset_name, bs, model_path, styles_dict, turbo, steps)
        )
        p.start()
        processes.append(p)
    for p in processes: p.join()


def visualize_domain(
    dataset_name: str,
    root: str = "./data",
    num_vis: int = 8,
    seed: int = 42,
    model_path: Optional[str] = None,
    styles: Optional[dict] = None,
):
    """为 Domain Shift 生成可视化对比图"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    # 加载样本
    test_dataset = DatasetClass(root=root, train=False)
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()[:num_vis]

    get_logger().info("🎨 正在生成 Domain 可视化对比图...")

    generator = DomainGenerator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=model_path,
        styles=styles,
    )

    style_names = list(styles.keys()) if styles else []
    for style in style_names:
        for strength in [0.3, 0.7]:
            styled = generator.apply_batch(
                images_np, style, strength, batch_size=num_vis
            )
            save_visual_comparison(
                images_np,
                styled,
                vis_dir / f"{style}_st{strength}.png",
                f"{style} (strength={strength})",
                num_samples=num_vis,
            )


def generate_ood_dataset(
    dataset_name: str,
    root: str = "./data",
    num_samples: int = 1000,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 16,
    model_path: Optional[str] = None,
    prompts: Optional[list] = None,
    use_turbo: bool = False,
    turbo_steps: int = 1,
) -> Path:
    """预生成 OOD 数据集 (外层接口)"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-OOD"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    mode_str = "SDXL Turbo" if use_turbo else "SD 2.1"
    get_logger().info(f"🔧 生成 OOD: {DatasetClass.NAME} ({num_samples} 张, {mode_str})")

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        _run_ood_serial(
            output_dir, num_samples, DatasetClass.IMAGE_SIZE, batch_size,
            seed, model_path, prompts, use_turbo, turbo_steps, num_gpus
        )
    else:
        _run_ood_parallel(
            output_dir, num_samples, DatasetClass.IMAGE_SIZE, batch_size,
            seed, model_path, prompts, use_turbo, turbo_steps
        )

    get_logger().info(f"✅ {DatasetClass.NAME}-OOD 生成完成!")
    return output_dir


def _run_ood_serial(output_dir, n, size, bs, seed, model_path, prompts, turbo, steps, num_gpus):
    """单卡或 CPU OOD 生成"""
    device = "cuda:0" if num_gpus == 1 else "cpu"
    if turbo:
        generator = TurboOODGenerator(device=device, model_path=model_path, prompts=prompts, num_steps=steps)
    else:
        generator = OODGenerator(device=device, model_path=model_path, prompts=prompts)

    imgs = generator.generate_batch(num_samples=n, target_size=size, batch_size=bs, seed=seed)
    np.save(str(output_dir / "images.npy"), imgs)


def _run_ood_parallel(output_dir, n, size, bs, seed, model_path, prompts, turbo, steps):
    """多卡并行 OOD 生成"""
    import multiprocessing
    num_gpus = torch.cuda.device_count()
    samples_per_gpu = n // num_gpus
    q = multiprocessing.Queue()
    processes = []

    for i in range(num_gpus):
        gpu_n = samples_per_gpu + (n % num_gpus if i == num_gpus - 1 else 0)
        p = multiprocessing.Process(
            target=_worker_ood_gpu,
            args=(i, gpu_n, size, bs, seed, q, model_path, prompts, turbo, steps)
        )
        p.start()
        processes.append(p)

    all_imgs = [q.get() for _ in range(num_gpus)]
    for p in processes: p.join()
    np.save(str(output_dir / "images.npy"), np.concatenate(all_imgs, axis=0))

    # =============================================================================
    # CLI 入口
    # =============================================================================


def main():
    """CLI 入口 - 重构为大纲结构"""
    args = _parse_args()
    config = _load_config()
    params = _get_gen_params(args, config)

    # 1. 执行生成
    _execute_generation(args, config, params)

    # 2. 统一可视化 (如果启用)
    if config.generation.visualize:
        _execute_visualization(args, config, params)


def _parse_args():
    parser = argparse.ArgumentParser(description="项目鲁棒性数据生成器")
    parser.add_argument("--type", type=str, required=True, choices=["corruption", "domain", "ood"])
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--model", type=str, choices=["sd", "turbo"], help="sd (SD 2.1) | turbo (SDXL Turbo)")
    parser.add_argument("--force", action="store_true", help="忽略缓存强制生成")
    return parser.parse_args()


def _load_config():
    cfg_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config, _, _ = Config.load_yaml(str(cfg_path))
    return config


def _get_gen_params(args, config):
    gen_cfg = config.generation
    use_turbo = (args.model == "turbo") if args.model else gen_cfg.use_turbo
    
    return {
        "use_turbo": use_turbo,
        "model_path": gen_cfg.turbo_model_path if use_turbo else gen_cfg.model_path,
        "turbo_steps": gen_cfg.turbo_steps if use_turbo else 1,
        "batch_size": gen_cfg.batch_size,
        "samples_per_group": gen_cfg.samples_per_group,
        "seed": config.seed,
        "root": config.data_root
    }


def _execute_generation(args, config, p):
    gen_cfg = config.generation
    if args.type == "corruption":
        generate_corruption_dataset(args.dataset, p["root"], p["seed"], args.force)
    elif args.type == "domain":
        generate_domain_dataset(
            args.dataset, p["root"], p["samples_per_group"], p["seed"], args.force,
            p["batch_size"], p["model_path"], gen_cfg.styles, gen_cfg.strengths,
            p["use_turbo"], p["turbo_steps"]
        )
    elif args.type == "ood":
        generate_ood_dataset(
            args.dataset, p["root"], p["samples_per_group"] * 2, p["seed"], args.force,
            p["batch_size"], p["model_path"], gen_cfg.ood_prompts,
            p["use_turbo"], p["turbo_steps"]
        )


def _execute_visualization(args, config, p):
    if args.type == "corruption":
        visualize_corruption(args.dataset, p["root"], config.generation.num_vis)
    elif args.type == "domain":
        visualize_domain(args.dataset, p["root"], config.generation.num_vis, p["seed"], p["model_path"], config.generation.styles)
    elif args.type == "ood":
        get_logger().info("ℹ️ OOD 模式暂不支持原图对比")

if __name__ == "__main__":
    main()
