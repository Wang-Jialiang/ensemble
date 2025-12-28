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
from typing import List, Optional

import numpy as np
from PIL import Image

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

    # 预设风格配置
    STYLES = {
        "sketch": {"prompt": "pencil sketch drawing", "strength": 0.5},
        "painting": {"prompt": "oil painting artwork", "strength": 0.6},
        "cartoon": {"prompt": "cartoon illustration style", "strength": 0.5},
    }

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

            get_logger().info("📥 加载 Stable Diffusion Img2Img 模型...")
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def apply(self, img: np.ndarray, style: str) -> np.ndarray:
        """对单张图像应用风格迁移"""
        if style not in self.STYLES:
            raise ValueError(
                f"Unknown style: {style}. Available: {list(self.STYLES.keys())}"
            )

        config = self.STYLES[style]
        pipe = self._get_pipe()

        # 转换为 PIL
        img_pil = Image.fromarray(img.astype(np.uint8))

        # 确保是正确的尺寸 (512x512 for SD)
        original_size = img_pil.size
        img_pil = img_pil.resize((512, 512), Image.Resampling.BILINEAR)

        # 生成
        result = pipe(
            prompt=config["prompt"],
            image=img_pil,
            strength=config["strength"],
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]

        # 恢复原始尺寸
        result = result.resize(original_size, Image.Resampling.BILINEAR)
        return np.array(result)


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

            get_logger().info("📥 加载 Stable Diffusion Text2Img 模型...")
            self._pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def generate(self, target_size: int = 64, seed: Optional[int] = None) -> np.ndarray:
        """生成单张 OOD 图像"""
        import random

        if seed is not None:
            random.seed(seed)

        pipe = self._get_pipe()
        prompt = random.choice(self.OOD_PROMPTS)

        result = pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]

        # 调整到目标尺寸
        result = result.resize((target_size, target_size), Image.Resampling.BILINEAR)
        return np.array(result)


# =============================================================================
# 生成函数
# =============================================================================


def generate_corruption_dataset(
    dataset_name: str,
    root: str = "./data",
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 corruption 数据集"""
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

    get_logger().info(f"🔧 开始生成 {DatasetClass.NAME}-C...")

    # 加载测试集
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    test_dataset = DatasetClass(root=root, train=False, **extra_kwargs)

    # 转换为 numpy (H, W, C) 格式
    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()
    labels_np = test_dataset.targets.numpy()
    n_samples = len(labels_np)

    # 生成每种 corruption (简化版: 4 类 × 3 级)
    for corruption in CORRUPTIONS:
        get_logger().info(f"   生成 {corruption}...")
        all_severities = []
        for severity in SEVERITIES:
            corrupted = CorruptionGenerator.apply_batch(
                images_np, corruption, severity, seed=seed
            )
            all_severities.append(corrupted.astype(np.uint8))

        # 保存: shape = (N*3, H, W, 3)
        stacked = np.concatenate(all_severities, axis=0)
        np.save(str(output_dir / f"{corruption}.npy"), stacked)

    # 保存标签
    np.save(str(labels_path), labels_np)

    get_logger().info(
        f"✅ {DatasetClass.NAME}-C 生成完成: "
        f"{n_samples} samples × {len(CORRUPTIONS)} corruptions × {len(SEVERITIES)} severities"
    )
    return output_dir


def generate_domain_dataset(
    dataset_name: str,
    root: str = "./data",
    styles: List[str] = None,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 domain shift 数据集"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )

    styles = styles or list(DomainGenerator.STYLES.keys())
    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-Domain"

    # 检查是否已存在
    if output_dir.exists() and not force:
        get_logger().info(
            f"✅ {output_dir} 已存在，跳过生成 (使用 --force 强制重新生成)"
        )
        return output_dir

    get_logger().info(f"🔧 开始生成 {DatasetClass.NAME}-Domain...")

    # 加载测试集
    extra_kwargs = {}
    if not getattr(DatasetClass, "HAS_OFFICIAL_SPLIT", True):
        extra_kwargs["seed"] = seed
    test_dataset = DatasetClass(root=root, train=False, **extra_kwargs)

    images_np = test_dataset.images.permute(0, 2, 3, 1).numpy()
    labels_np = test_dataset.targets.numpy()

    generator = DomainGenerator()

    for style in styles:
        get_logger().info(f"   生成风格: {style}...")
        style_dir = output_dir / style

        # 按类别组织
        for class_idx in range(DatasetClass.NUM_CLASSES):
            class_dir = style_dir / f"class_{class_idx}"
            ensure_dir(class_dir)

        # 处理每张图片
        for i, (img, label) in enumerate(zip(images_np, labels_np)):
            styled_img = generator.apply(img, style)
            img_path = style_dir / f"class_{label}" / f"img_{i}.png"
            Image.fromarray(styled_img.astype(np.uint8)).save(str(img_path))

            if (i + 1) % 100 == 0:
                get_logger().info(f"      已处理 {i + 1}/{len(images_np)} 张")

    get_logger().info(f"✅ {DatasetClass.NAME}-Domain 生成完成!")
    return output_dir


def generate_ood_dataset(
    dataset_name: str,
    root: str = "./data",
    num_samples: int = 100,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 OOD 数据集"""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-OOD-Generated"
    ensure_dir(output_dir)

    images_path = output_dir / "images.npy"
    if images_path.exists() and not force:
        get_logger().info(
            f"✅ {output_dir} 已存在，跳过生成 (使用 --force 强制重新生成)"
        )
        return output_dir

    get_logger().info(f"🔧 开始生成 {DatasetClass.NAME}-OOD ({num_samples} 张)...")

    generator = OODGenerator()
    images = []

    for i in range(num_samples):
        img = generator.generate(target_size=DatasetClass.IMAGE_SIZE, seed=seed + i)
        images.append(img)

        if (i + 1) % 10 == 0:
            get_logger().info(f"   已生成 {i + 1}/{num_samples} 张")

    images_array = np.stack(images, axis=0)
    np.save(str(images_path), images_array)

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
    # Domain 专用参数
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=None,
        help="Domain: 要生成的风格列表 (sketch, painting, cartoon)",
    )
    # OOD 专用参数
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="OOD: 生成的样本数量",
    )
    args = parser.parse_args()

    if args.type == "corruption":
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
            styles=args.styles,
            seed=args.seed,
            force=args.force,
        )
    elif args.type == "ood":
        generate_ood_dataset(
            dataset_name=args.dataset,
            root=args.root,
            num_samples=args.num_samples,
            seed=args.seed,
            force=args.force,
        )


if __name__ == "__main__":
    main()
