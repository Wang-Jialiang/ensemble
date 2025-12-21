"""
================================================================================
Corruption 数据集生成脚本
================================================================================

生成 corruption 数据集 (CIFAR-10-C 格式):
    python -m ensemble.datasets.generate --dataset eurosat --root ./data

生成的文件结构:
    {root}/{DatasetName}-C/
        gaussian_noise.npy   # shape: (N*5, H, W, 3)
        shot_noise.npy
        ...
        labels.npy           # shape: (N,)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from ..utils import DEFAULT_DATA_ROOT, ensure_dir, get_logger
from .corruption import CORRUPTIONS
from .preloaded import DATASET_REGISTRY

# =============================================================================
# Corruption 生成器
# =============================================================================


class CorruptionGenerator:
    """Corruption 生成器 - 基于 imagecorruptions 库

    使用 imagecorruptions 库实现与 ImageNet-C / CIFAR-10-C 相同的 15 种 corruption 类型。
    依赖: pip install imagecorruptions
    """

    # 引用模块级常量
    CORRUPTIONS = CORRUPTIONS

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


def generate_corruption_dataset(
    dataset_name: str,
    root: str = DEFAULT_DATA_ROOT,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """预生成 corruption 数据集 (CIFAR-10-C 格式)

    Args:
        dataset_name: 数据集名称 (必须在 DATASET_REGISTRY 中注册)
        root: 数据根目录
        seed: 随机种子
        force: 是否强制重新生成

    Returns:
        生成的数据集目录路径
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"未知数据集: {dataset_name}. 可用: {list(DATASET_REGISTRY.keys())}"
        )

    DatasetClass = DATASET_REGISTRY[dataset_name]
    output_dir = Path(root) / f"{DatasetClass.NAME}-C"
    ensure_dir(output_dir)

    # 检查是否已存在
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

    # 生成每种 corruption
    for corruption in CorruptionGenerator.CORRUPTIONS:
        get_logger().info(f"   生成 {corruption}...")
        all_severities = []
        for severity in range(1, 6):
            corrupted = CorruptionGenerator.apply_batch(
                images_np, corruption, severity, seed=seed
            )
            all_severities.append(corrupted.astype(np.uint8))

        # 保存: shape = (N*5, H, W, 3)
        stacked = np.concatenate(all_severities, axis=0)
        np.save(str(output_dir / f"{corruption}.npy"), stacked)

    # 保存标签
    np.save(str(labels_path), labels_np)

    get_logger().info(
        f"✅ {DatasetClass.NAME}-C 生成完成: {n_samples} samples × 15 corruptions × 5 severities"
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="生成 Corruption 数据集")
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
        default=DEFAULT_DATA_ROOT,
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
    args = parser.parse_args()

    generate_corruption_dataset(
        dataset_name=args.dataset,
        root=args.root,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
