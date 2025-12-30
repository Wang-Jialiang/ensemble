"""
================================================================================
统一数据生成脚本 (SDXL Lightning 版)
================================================================================

支持三种数据类型的生成:
- Corruption: 使用 imagecorruptions 库生成损坏数据
- Domain Shift: 使用 SDXL Lightning Img2Img 生成风格迁移数据
- OOD: 使用 SDXL Lightning Text2Img 生成分布外数据

使用示例:
    python -m ensemble.datasets.robustness.generate --type corruption --dataset cifar10
    python -m ensemble.datasets.robustness.generate --type domain --dataset cifar10
    python -m ensemble.datasets.robustness.generate --type ood --dataset cifar10
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def patch_dependencies():
    """Monkey-patch dependencies for imagecorruptions compatibility."""
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
    try:
        import numpy as np

        if not hasattr(np, "float_"):
            np.float_ = np.float64
    except (ImportError, AttributeError):
        pass


patch_dependencies()

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="imagecorruptions")

try:
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
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


def _prepare_pil_batch(images_np: np.ndarray, target_size: int = 1024):
    """将 numpy 批量图像转换为 PIL 格式并统一缩放"""
    return [
        Image.fromarray(img.astype(np.uint8)).resize(
            (target_size, target_size), Image.LANCZOS
        )
        for img in images_np
    ]


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


def _sample_dataset(images_np, labels_np, n, seed=42):
    """对数据集进行随机抽样"""
    if n is None or n >= len(images_np):
        return images_np, labels_np
    np.random.seed(seed)
    indices = np.random.choice(len(images_np), size=n, replace=False)
    return images_np[indices], labels_np[indices]


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
    _img2img_cache = {}

    @classmethod
    def get_text2img(
        cls, device: str, base_model: str, repo: str, ckpt: str
    ) -> "StableDiffusionXLPipeline":
        """获取 Text2Img Pipeline (带缓存)"""
        if device not in cls._text2img_cache:
            get_logger().info(f"📥 [{device}] 加载 SDXL Lightning Text2Img...")
            unet = cls._load_unet(device, base_model, repo, ckpt)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                base_model, unet=unet, torch_dtype=torch.float16, variant="fp16"
            ).to(device)
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            pipe.set_progress_bar_config(disable=True)
            cls._try_enable_optimizations(pipe)
            cls._text2img_cache[device] = pipe
        return cls._text2img_cache[device]

    @classmethod
    def get_img2img(
        cls, device: str, base_model: str, repo: str, ckpt: str
    ) -> "StableDiffusionXLImg2ImgPipeline":
        """获取 Img2Img Pipeline (带缓存)"""
        if device not in cls._img2img_cache:
            get_logger().info(f"📥 [{device}] 加载 SDXL Lightning Img2Img...")
            unet = cls._load_unet(device, base_model, repo, ckpt)
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model, unet=unet, torch_dtype=torch.float16, variant="fp16"
            ).to(device)
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing="trailing"
            )
            pipe.set_progress_bar_config(disable=True)
            cls._try_enable_optimizations(pipe)
            cls._img2img_cache[device] = pipe
        return cls._img2img_cache[device]

    @staticmethod
    def _load_unet(device: str, base_model: str, repo: str, ckpt: str):
        """加载 Lightning UNet (支持本地路径)"""
        unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to(
            device, torch.float16
        )

        local_path = Path(repo) / ckpt
        if local_path.exists():
            get_logger().info(f"🚀 加载本地 Lightning 权重: {local_path}")
            state_dict = load_file(str(local_path), device=device)
        else:
            get_logger().info(f"🌐 从 Hugging Face 下载权重: {repo}/{ckpt}")
            state_dict = load_file(hf_hub_download(repo, ckpt), device=device)

        unet.load_state_dict(state_dict)
        return unet

    @staticmethod
    def _try_enable_optimizations(pipe):
        """尝试启用显存优化"""
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass


# =============================================================================
# Domain Shift 生成器 (SDXL Lightning)
# =============================================================================


class DomainGenerator:
    """Domain Shift 生成器 - 基于 SDXL Lightning Img2Img"""

    def __init__(
        self,
        device: str = "cuda",
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lightning_repo: str = "ByteDance/SDXL-Lightning",
        lightning_ckpt: str = "sdxl_lightning_4step_unet.safetensors",
        styles: Optional[dict] = None,
        num_steps: int = 4,
    ):
        self.device = device
        self.base_model = base_model
        self.lightning_repo = lightning_repo
        self.lightning_ckpt = lightning_ckpt
        self.styles = styles or {}
        self.num_steps = num_steps
        self._pipe = None

    def _get_pipe(self):
        """获取 Img2Img Pipeline"""
        if self._pipe is None:
            self._pipe = LightningPipelineLoader.get_img2img(
                self.device, self.base_model, self.lightning_repo, self.lightning_ckpt
            )
        return self._pipe

    def apply_batch(
        self, images: np.ndarray, style: str, strength: float, batch_size: int = 24
    ) -> np.ndarray:
        """批量风格转换"""
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

            pils = _prepare_pil_batch(batch, target_size=1024)
            outputs = pipe(
                prompt=[prompt] * len(pils),
                image=pils,
                strength=strength,
                guidance_scale=0.0,
                num_inference_steps=self.num_steps,
            ).images

            results.extend(_convert_to_numpy_batch(outputs, (orig_w, orig_h)))

        return np.stack(results)


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
    ):
        self.device = device
        self.base_model = base_model
        self.lightning_repo = lightning_repo
        self.lightning_ckpt = lightning_ckpt
        self.prompts = prompts or []
        self.num_steps = num_steps
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
        """批量生成 OOD 图像"""
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

            outputs = pipe(
                prompt=prompts,
                height=1024,
                width=1024,
                guidance_scale=0.0,
                num_inference_steps=self.num_steps,
            ).images

            results.extend(_convert_to_numpy_batch(outputs, (target_size, target_size)))

        return np.stack(results)


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
    base_model,
    lightning_repo,
    lightning_ckpt,
    full_styles_dict,
    num_steps,
):
    """Domain 工作者线程 (用于 GPU 并行)"""
    generator = DomainGenerator(
        device=device,
        base_model=base_model,
        lightning_repo=lightning_repo,
        lightning_ckpt=lightning_ckpt,
        styles=full_styles_dict,
        num_steps=num_steps,
    )
    DatasetClass = DATASET_REGISTRY[dataset_name]

    for style in styles:
        for strength in strengths:
            get_logger().info(f"   [{device}] 生成: {style} (strength={strength})...")
            strength_dir = output_dir / style / str(strength)

            for class_idx in range(DatasetClass.NUM_CLASSES):
                ensure_dir(strength_dir / f"class_{class_idx:04d}")

            styled_images = generator.apply_batch(
                images_np, style, strength, batch_size=batch_size
            )

            for i, (img, label) in enumerate(zip(styled_images, labels_np)):
                img_path = strength_dir / f"class_{label:04d}" / f"img_{i}.png"
                Image.fromarray(img).save(str(img_path))


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
    imgs = generator.generate_batch(
        num_samples=n, target_size=target_size, batch_size=bs, seed=seed + gpu_id
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

    get_logger().info(f"🔧 生成 Corruption: {DatasetClass.NAME}-C...")

    images_np, labels_np = _load_test_set_numpy(DatasetClass, root, seed)
    total_samples = len(labels_np)

    tasks = [(c, images_np, SEVERITIES, output_dir, seed) for c in CORRUPTIONS]
    num_cpus = os.cpu_count()

    with multiprocessing.Pool(processes=min(len(CORRUPTIONS), num_cpus)) as pool:
        list(
            tqdm(
                pool.imap_unordered(_process_single_corruption, tasks),
                total=len(tasks),
                desc="   Corruption 总进度",
            )
        )

    np.save(str(output_dir / "labels.npy"), labels_np)

    get_logger().info(
        f"✅ {DatasetClass.NAME}-C 生成完成: {len(CORRUPTIONS)} corruptions × {total_samples} samples"
    )
    return output_dir


def generate_domain_dataset(
    dataset_name: str,
    root: str = "./data",
    samples_per_group: Optional[int] = 1000,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 24,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lightning_repo: str = "ByteDance/SDXL-Lightning",
    lightning_ckpt: str = "sdxl_lightning_4step_unet.safetensors",
    styles: Optional[dict] = None,
    strengths: Optional[list] = None,
    num_steps: int = 4,
) -> Path:
    """预生成 domain shift 数据集"""
    import multiprocessing

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    images_np, labels_np = _load_test_set_numpy(DatasetClass, root, seed)
    images_np, labels_np = _sample_dataset(
        images_np, labels_np, samples_per_group, seed
    )

    num_gpus = torch.cuda.device_count()
    get_logger().info(
        f"🔧 生成 Domain: {DatasetClass.NAME} (SDXL Lightning 4-step, GPU={num_gpus})"
    )

    styles_list = list(styles.keys()) if styles else []

    if num_gpus == 0:
        # CPU 串行模式
        generator = DomainGenerator(
            device="cpu",
            base_model=base_model,
            lightning_repo=lightning_repo,
            lightning_ckpt=lightning_ckpt,
            styles=styles,
            num_steps=num_steps,
        )
        for style in styles_list:
            for str_val in strengths:
                strength_dir = output_dir / style / str(str_val)
                for c in range(DatasetClass.NUM_CLASSES):
                    ensure_dir(strength_dir / f"class_{c:04d}")
                styled = generator.apply_batch(
                    images_np, style, str_val, batch_size=batch_size
                )
                for i, (img, lbl) in enumerate(zip(styled, labels_np)):
                    Image.fromarray(img).save(
                        str(strength_dir / f"class_{lbl:04d}" / f"img_{i}.png")
                    )
    else:
        # GPU 并行模式
        processes = []
        for i in range(num_gpus):
            gpu_styles = styles_list[i::num_gpus]
            if not gpu_styles:
                continue
            p = multiprocessing.Process(
                target=_worker_domain,
                args=(
                    f"cuda:{i}",
                    gpu_styles,
                    strengths,
                    images_np,
                    labels_np,
                    output_dir,
                    dataset_name,
                    batch_size,
                    base_model,
                    lightning_repo,
                    lightning_ckpt,
                    styles,
                    num_steps,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    get_logger().info(f"✅ {DatasetClass.NAME}-Domain 生成完成!")
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
) -> Path:
    """预生成 OOD 数据集"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-OOD"

    if _check_existing_dataset(output_dir, force):
        return output_dir

    get_logger().info(
        f"🔧 生成 OOD: {DatasetClass.NAME} ({num_samples} 张, SDXL Lightning 4-step)"
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        device = "cuda:0" if num_gpus == 1 else "cpu"
        generator = OODGenerator(
            device=device,
            base_model=base_model,
            lightning_repo=lightning_repo,
            lightning_ckpt=lightning_ckpt,
            prompts=prompts,
            num_steps=num_steps,
        )
        imgs = generator.generate_batch(
            num_samples=num_samples,
            target_size=DatasetClass.IMAGE_SIZE,
            batch_size=batch_size,
            seed=seed,
        )
        np.save(str(output_dir / "images.npy"), imgs)
    else:
        import multiprocessing

        samples_per_gpu = num_samples // num_gpus
        q = multiprocessing.Queue()
        processes = []

        for i in range(num_gpus):
            gpu_n = samples_per_gpu + (
                num_samples % num_gpus if i == num_gpus - 1 else 0
            )
            p = multiprocessing.Process(
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

    get_logger().info(f"✅ {DatasetClass.NAME}-OOD 生成完成!")
    return output_dir


def visualize_corruption(dataset_name: str, root: str = "./data", num_vis: int = 8):
    """为 Corruption 生成可视化对比图"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    test_dataset = DatasetClass(root=root, train=False)
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()

    get_logger().info("🎨 正在生成 Corruption 可视化对比图...")

    for c in ["gaussian_noise", "fog", "glass_blur"]:
        for s in [3, 5]:
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


def visualize_domain(
    dataset_name: str, root: str = "./data", num_vis: int = 8, gen_cfg=None
):
    """为 Domain Shift 生成可视化对比图"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    test_dataset = DatasetClass(root=root, train=False)
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()[:num_vis]

    get_logger().info("🎨 正在生成 Domain 可视化对比图...")

    generator = DomainGenerator(
        device="cuda" if torch.cuda.is_available() else "cpu",
        base_model=gen_cfg.base_model,
        lightning_repo=gen_cfg.lightning_repo,
        lightning_ckpt=gen_cfg.lightning_ckpt,
        styles=gen_cfg.styles,
        num_steps=gen_cfg.num_steps,
    )

    for style in list(gen_cfg.styles.keys()):
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
        "--type", type=str, required=True, choices=["corruption", "domain", "ood"]
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_REGISTRY.keys())
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
    elif args.type == "domain":
        generate_domain_dataset(
            args.dataset,
            config.data_root,
            gen_cfg.samples_per_group,
            config.seed,
            args.force,
            gen_cfg.batch_size,
            gen_cfg.base_model,
            gen_cfg.lightning_repo,
            gen_cfg.lightning_ckpt,
            gen_cfg.styles,
            gen_cfg.strengths,
            gen_cfg.num_steps,
        )
    elif args.type == "ood":
        generate_ood_dataset(
            args.dataset,
            config.data_root,
            gen_cfg.samples_per_group * 2,
            config.seed,
            args.force,
            gen_cfg.batch_size,
            gen_cfg.base_model,
            gen_cfg.lightning_repo,
            gen_cfg.lightning_ckpt,
            gen_cfg.ood_prompts,
            gen_cfg.num_steps,
        )


def _execute_visualization(args, config):
    if args.type == "corruption":
        visualize_corruption(args.dataset, config.data_root, config.generation.num_vis)
    elif args.type == "domain":
        visualize_domain(
            args.dataset, config.data_root, config.generation.num_vis, config.generation
        )
    elif args.type == "ood":
        visualize_ood(args.dataset, config.data_root, config.generation.num_vis)


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
    ncols = (n + nrow - 1) // nrow

    grid = Image.new("RGB", (w * nrow, h * ncols))

    for i, img in enumerate(imgs):
        r = i // nrow
        c = i % nrow
        grid.paste(Image.fromarray(img.astype(np.uint8)), (c * w, r * h))

    ensure_dir(output_path.parent)
    grid.save(str(output_path))
    get_logger().info(f"📊 可视化保存: {output_path}")


def visualize_ood(dataset_name: str, root: str = "./data", num_vis: int = 8):
    """为 OOD 生成可视化网格"""
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-OOD"
    vis_dir = output_dir / "visuals"
    ensure_dir(vis_dir)

    images_path = output_dir / "images.npy"
    if not images_path.exists():
        get_logger().warning(f"⚠️ OOD 数据未找到: {images_path}")
        return

    get_logger().info("🎨 正在生成 OOD 可视化...")

    # Load images (using mmap to avoid loading everything if large)
    images = np.load(str(images_path), mmap_mode="r")

    # Take a subset for visualization
    total_images = len(images)
    indices = np.linspace(0, total_images - 1, min(total_images, num_vis), dtype=int)
    vis_images = images[indices]

    save_visual_grid(
        vis_images,
        vis_dir / "ood_samples.png",
        "OOD Samples",
        num_samples=num_vis,
        nrow=4,
    )


if __name__ == "__main__":
    main()
