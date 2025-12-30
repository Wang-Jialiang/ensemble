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

from ...utils import ensure_dir, get_logger
from ..preloaded import DATASET_REGISTRY
from .corruption import CORRUPTIONS, SEVERITIES

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
    """Domain Shift 生成器 - 基于 Stable Diffusion Img2Img

    使用 Stable Diffusion 将原始图像转换为不同风格。
    依赖: pip install diffusers transformers accelerate
    """

    # 4 种风格 (prompt 用于引导 Stable Diffusion)
    STYLES = {
        "sketch": "pencil sketch drawing",
        "painting": "oil painting artwork",
        "cartoon": "cartoon illustration style",
        "watercolor": "watercolor painting art",
    }

    # 3 种强度等级 (类似 Corruption 的 severity)
    STRENGTHS = [0.3, 0.5, 0.7]  # 轻度、中度、重度

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        """延迟加载 pipeline"""
        if self._pipe is None:
            try:
                import torch
                from diffusers import StableDiffusionImg2ImgPipeline
            except ImportError:
                raise ImportError(
                    "需要安装 diffusers: pip install diffusers transformers accelerate"
                )

            get_logger().info(
                f"📥 加载 Stable Diffusion Img2Img 模型 (设备: {self.device})..."
            )
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)

            # 尝试启用 xformers
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        return self._pipe

    def apply_batch(
        self, images: np.ndarray, style: str, strength: float, batch_size: int = 16
    ) -> np.ndarray:
        """批量风格转换"""
        if style not in self.STYLES:
            raise ValueError(f"Unknown style: {style}")

        prompt = self.STYLES[style]
        pipe = self._get_pipe()
        all_results = []

        # 获取 GPU ID 以便进度条不重叠
        gpu_id = 0
        if "cuda:" in self.device:
            gpu_id = int(self.device.split(":")[-1])

        pbar = tqdm(
            range(0, len(images), batch_size),
            desc=f"      [{self.device}] {style}/{strength}",
            position=gpu_id,
            leave=False,
            mininterval=1.0,  # 避免频繁刷新
        )

        for i in pbar:
            batch = images[i : i + batch_size]
            original_size = (batch.shape[2], batch.shape[1])  # (W, H)

            # 转换为 PIL 并调整大小为 512
            pils = [
                Image.fromarray(img.astype(np.uint8)).resize(
                    (512, 512), Image.Resampling.LANCZOS
                )
                for img in batch
            ]

            # 批量生成
            outputs = pipe(
                prompt=[prompt] * len(pils),
                image=pils,
                strength=strength,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images

            # 恢复尺寸并转回 numpy
            transformed = [
                np.array(img.resize(original_size, Image.Resampling.LANCZOS))
                for img in outputs
            ]
            all_results.extend(transformed)

        return np.stack(all_results)

    def apply(self, img: np.ndarray, style: str, strength: float) -> np.ndarray:
        """对单张图像应用风格迁移 (封装 apply_batch)"""
        return self.apply_batch(np.expand_dims(img, 0), style, strength, batch_size=1)[
            0
        ]


# =============================================================================
# OOD 生成器
# =============================================================================


class OODGenerator:
    """OOD 生成器 - 基于 Stable Diffusion Text2Img

    使用 Stable Diffusion 生成与原数据集无关的图像。
    依赖: pip install diffusers transformers accelerate
    """

    # 预设 OOD prompts (与任何常见图像分类数据集无关)
    OOD_PROMPTS = [
        "abstract colorful geometric patterns",
        "underwater coral reef with tropical fish",
        "close-up of delicious food dishes",
        "city street at night with neon lights",
        "cartoon character illustration",
        "ancient stone ruins in jungle",
        "microscopic view of cells",
        "aurora borealis in night sky",
        "vintage book pages with text",
        "crystal formations in cave",
    ]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None

    def _get_pipe(self):
        """延迟加载 pipeline"""
        if self._pipe is None:
            try:
                import torch
                from diffusers import StableDiffusionPipeline
            except ImportError:
                raise ImportError(
                    "需要安装 diffusers: pip install diffusers transformers accelerate"
                )

            get_logger().info(
                f"📥 加载 Stable Diffusion Text2Img 模型 (设备: {self.device})..."
            )
            self._pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)

            # 尝试启用 xformers
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        return self._pipe

    def generate_batch(
        self,
        num_samples: int,
        target_size: int = 64,
        batch_size: int = 16,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """批量生成 OOD 图像"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        all_results = []

        # 获取 GPU ID 以便进度条不重叠
        gpu_id = 0
        if "cuda:" in self.device:
            gpu_id = int(self.device.split(":")[-1])

        pbar = tqdm(
            range(0, num_samples, batch_size),
            desc=f"      [{self.device}] OOD 生成",
            position=gpu_id,
            leave=False,
            mininterval=1.0,
        )

        for i in pbar:
            current_bs = min(batch_size, num_samples - i)
            prompts = [random.choice(self.OOD_PROMPTS) for _ in range(current_bs)]

            # 批量生成
            outputs = pipe(
                prompt=prompts,
                height=512,
                width=512,
                guidance_scale=7.5,
                num_inference_steps=30,
            ).images

            # 调整尺寸并转回 numpy
            transformed = [
                np.array(
                    img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                )
                for img in outputs
            ]
            all_results.extend(transformed)

        return np.stack(all_results)

    def generate(self, target_size: int = 64, seed: Optional[int] = None) -> np.ndarray:
        """生成单张 OOD 图像 (封装 generate_batch)"""
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
):
    """Domain 工作者线程 (用于 GPU 并行)"""
    generator = DomainGenerator(device=device)
    DatasetClass = DATASET_REGISTRY[dataset_name]

    for style in styles:
        for strength in strengths:
            get_logger().info(f"   [{device}] 生成: {style} (strength={strength})...")
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


def _worker_ood_gpu(gpu_id, n, target_size, bs, s, q):
    """OOD 工作者线程 (用于 GPU 并行)"""
    generator = OODGenerator(device=f"cuda:{gpu_id}")
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
    """预生成 corruption 数据集（使用 CPU 多进程加速）"""
    import multiprocessing
    import os

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"
    ensure_dir(output_dir)

    labels_path = output_dir / "labels.npy"
    if labels_path.exists() and not force:
        get_logger().info(
            f"✅ {output_dir} 已存在，跳过生成 (使用 --force 强制重新生成)"
        )
        return output_dir

    get_logger().info(f"🔧 开始生成 {DatasetClass.NAME}-C (EPYC 并行模式)...")

    # 加载测试集
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    test_dataset = DatasetClass(root=root, train=False, **extra_kwargs)

    # 转换为 numpy (H, W, C) 格式
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()
    labels_np = test_dataset.targets.numpy()
    total_samples = len(labels_np)

    # 准备并行参数
    tasks = []
    for corruption in CORRUPTIONS:
        tasks.append((corruption, images_np, SEVERITIES, output_dir, seed))

    # 使用所有可用的 CPU 核心
    num_cpus = os.cpu_count()
    get_logger().info(f"   使用 {num_cpus} 个进程并行生成...")

    with multiprocessing.Pool(processes=min(len(CORRUPTIONS), num_cpus)) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_single_corruption, tasks),
            total=len(tasks),
            desc="   Corruption 总进度",
        ):
            pass

    # 保存标签
    np.save(str(labels_path), labels_np)

    # 统计信息
    msg = f"✅ {DatasetClass.NAME}-C 生成完成: {len(CORRUPTIONS)} corruptions × {total_samples} samples × {len(SEVERITIES)} severities"
    get_logger().info(msg)
    return output_dir


def generate_domain_dataset(
    dataset_name: str,
    root: str = "./data",
    samples_per_group: Optional[int] = 1000,
    seed: int = 42,
    force: bool = False,
    batch_size: int = 16,
) -> Path:
    """预生成 domain shift 数据集（双 GPU + 批量推理加速）"""
    import multiprocessing

    import torch

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset_name}")

    styles = list(DomainGenerator.STYLES.keys())
    strengths = DomainGenerator.STRENGTHS
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"

    if output_dir.exists() and not force:
        get_logger().info(f"✅ {output_dir} 已存在，跳过生成")
        return output_dir

    get_logger().info(
        f"🔧 开始生成 {DatasetClass.NAME}-Domain (GPU 并行 + Batch Size {batch_size})..."
    )

    # 加载并抽样
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    test_dataset = DatasetClass(root=root, train=False, **extra_kwargs)

    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()
    labels_np = test_dataset.targets.numpy()
    total_available = len(labels_np)
    target_n = min(samples_per_group or total_available, total_available)

    np.random.seed(seed)
    indices = np.random.permutation(total_available)[:target_n]
    images_np = images_np[indices]
    labels_np = labels_np[indices]

    # 检测 GPU 并分发任务
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        get_logger().warning("未检测到 GPU，将回退到 CPU (单进程)，速度可能会非常慢。")
        generator = DomainGenerator(device="cpu")
        for style in styles:
            for strength in strengths:
                get_logger().info(f"   生成: {style} (strength={strength})...")
                strength_dir = output_dir / style / str(strength)
                for class_idx in range(DatasetClass.NUM_CLASSES):
                    ensure_dir(strength_dir / f"class_{class_idx:04d}")

                styled_images = generator.apply_batch(
                    images_np, style, strength, batch_size=batch_size
                )
                for i, (img, label) in enumerate(zip(styled_images, labels_np)):
                    img_path = strength_dir / f"class_{label:04d}" / f"img_{i}.png"
                    Image.fromarray(img).save(str(img_path))
        get_logger().info(f"✅ {DatasetClass.NAME}-Domain 生成完成!")
        return output_dir

    get_logger().info(f"   检测到 {num_gpus} 个 GPU, 开始分发任务...")

    processes = []
    # 将 styles 平分
    for i in range(num_gpus):
        gpu_styles = styles[i::num_gpus]
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
    batch_size: int = 16,
) -> Path:
    """预生成 OOD 数据集（双 GPU + 批量生成加速）"""
    import multiprocessing

    import torch

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-OOD"
    ensure_dir(output_dir)

    images_path = output_dir / "images.npy"
    if images_path.exists() and not force:
        get_logger().info(
            f"✅ {output_dir} 已存在，跳过生成 (使用 --force 强制重新生成)"
        )
        return output_dir

    get_logger().info(
        f"🔧 开始生成 {DatasetClass.NAME}-OOD ({num_samples} 张, GPU 并行模式)..."
    )

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        get_logger().warning("未检测到 GPU，将使用 CPU 进行 OOD 生成，速度会非常慢。")
        # Fallback to CPU if no GPU is found
        generator = OODGenerator(device="cpu")
        images_array = generator.generate_batch(
            num_samples=num_samples,
            target_size=DatasetClass.IMAGE_SIZE,
            batch_size=batch_size,
            seed=seed,
        )
        np.save(str(images_path), images_array)
    elif num_gpus == 1:
        # 单 GPU 模式
        get_logger().info("   检测到 1 个 GPU (cuda:0)，使用单 GPU 模式生成...")
        generator = OODGenerator(device="cuda:0")
        images_array = generator.generate_batch(
            num_samples=num_samples,
            target_size=DatasetClass.IMAGE_SIZE,
            batch_size=batch_size,
            seed=seed,
        )
        np.save(str(images_path), images_array)
    else:
        # 多 GPU 并行
        get_logger().info(f"   检测到 {num_gpus} 个 GPU，使用多 GPU 并行模式生成...")
        samples_per_gpu = num_samples // num_gpus
        results_queue = multiprocessing.Queue()

        processes = []
        for i in range(num_gpus):
            # 分配样本，确保总数正确
            n = samples_per_gpu + (num_samples % num_gpus if i == num_gpus - 1 else 0)
            p = multiprocessing.Process(
                target=_worker_ood_gpu,
                args=(i, n, DatasetClass.IMAGE_SIZE, batch_size, seed, results_queue),
            )
            p.start()
            processes.append(p)

        all_imgs = []
        for _ in range(num_gpus):
            all_imgs.append(results_queue.get())
        for p in processes:
            p.join()

        final_array = np.concatenate(all_imgs, axis=0)
        np.save(str(images_path), final_array)

    get_logger().info(f"✅ {DatasetClass.NAME}-OOD 生成完成: {num_samples} 张")
    return output_dir


# =============================================================================
# CLI 入口
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="统一数据生成脚本 (Corruption / Domain Shift / OOD)"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["corruption", "domain", "ood"],
        help="生成类型: corruption, domain, ood",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="数据集名称",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="数据根目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=1000,
        help="每组样本数（仅 Domain/OOD）。Domain: 每风格×强度; OOD: 总数。Corruption 始终使用全量测试集",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="渲染生成时的 Batch Size (Stable Diffusion 优化)",
    )
    args = parser.parse_args()

    if args.type == "corruption":
        # Corruption 使用全量测试集（CPU 操作，速度快）
        generate_corruption_dataset(
            dataset_name=args.dataset,
            root=args.root,
            seed=args.seed,
            force=args.force,
        )
    elif args.type == "domain":
        generate_domain_dataset(
            dataset_name=args.dataset,
            root=args.root,
            samples_per_group=args.samples_per_group,
            seed=args.seed,
            force=args.force,
            batch_size=args.batch_size,
        )
    elif args.type == "ood":
        # OOD 用 samples_per_group × 2 (补偿 Text2Img 更慢)
        ood_samples = args.samples_per_group * 2
        generate_ood_dataset(
            dataset_name=args.dataset,
            root=args.root,
            num_samples=ood_samples,
            seed=args.seed,
            force=args.force,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
