"""
================================================================================
TTA (测试时数据增强) 模块
================================================================================

包含: TTAStrategy, TTAAugmentor, get_all_models_logits_with_tta

核心原理:
    1. 对每张测试图像生成多个增强副本 (翻转、裁剪、旋转等)
    2. 分别对所有副本进行模型推理
    3. 聚合所有预测的概率 (取平均) 作为最终预测

使用示例:
    >>> from evaluation.tta import TTAAugmentor, get_all_models_logits_with_tta
    >>> augmentor = TTAAugmentor.from_config(cfg)
    >>> logits, targets = get_all_models_logits_with_tta(models, loader, augmentor)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ..utils import get_logger

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ TTA 策略定义                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class TTAStrategyType(Enum):
    """TTA 策略类型枚举"""

    NONE = "none"  # 无增强 (用于对照)
    LIGHT = "light"  # 轻量: 仅水平翻转 (2x)
    STANDARD = "standard"  # 标准: 翻转 + 5裁剪 (~10x)
    HEAVY = "heavy"  # 重量: 翻转 + 裁剪 + 旋转 (~32x)
    GEOSPATIAL = "geospatial"  # 遥感专用: 翻转 + 90°旋转 (8x)


@dataclass
class TTAConfig:
    """TTA 配置数据类"""

    enabled: bool = False
    strategy: TTAStrategyType = TTAStrategyType.STANDARD
    crop_scales: List[float] = field(default_factory=lambda: [0.875, 0.9])
    num_crops: int = 5  # 4角 + 中心

    @classmethod
    def from_dict(cls, cfg: dict) -> "TTAConfig":
        """从配置字典创建 TTAConfig"""
        strategy_str = cfg.get("tta_strategy", "standard").lower()
        try:
            strategy = TTAStrategyType(strategy_str)
        except ValueError:
            get_logger().warning(
                f"⚠️ 未知 TTA 策略 '{strategy_str}'，使用默认 'standard'"
            )
            strategy = TTAStrategyType.STANDARD

        return cls(
            enabled=cfg.get("tta_enabled", False),
            strategy=strategy,
            crop_scales=cfg.get("tta_crop_scales", [0.875, 0.9]),
            num_crops=cfg.get("tta_num_crops", 5),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ TTA 增强器                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class TTAAugmentor:
    """TTA 增强器 - 生成图像的多个增强副本

    支持的增强变换:
        - 水平翻转 (hflip)
        - 垂直翻转 (vflip)
        - 五点裁剪 (5-crop: 4角 + 中心)
        - 旋转 (90°, 180°, 270°)

    Args:
        config: TTA 配置
        image_size: 目标图像尺寸 (用于裁剪后 resize)
    """

    def __init__(self, config: TTAConfig, image_size: int = 32):
        self.config = config
        self.image_size = image_size
        self._setup_transforms()

    @classmethod
    def from_config(cls, cfg: dict, image_size: int = 32) -> "TTAAugmentor":
        """从配置字典创建增强器"""
        tta_config = TTAConfig.from_dict(cfg)
        return cls(tta_config, image_size)

    def _setup_transforms(self):
        """根据策略设置增强变换"""
        self.transforms_list = []

        strategy = self.config.strategy

        if strategy == TTAStrategyType.NONE:
            # 无增强，只保留原图
            self.transforms_list = [lambda x: x]

        elif strategy == TTAStrategyType.LIGHT:
            # 轻量: 原图 + 水平翻转
            self.transforms_list = [
                lambda x: x,  # 原图
                transforms.functional.hflip,  # 水平翻转
            ]

        elif strategy == TTAStrategyType.STANDARD:
            # 标准: 原图 + 水平翻转 + 5裁剪
            self.transforms_list = self._build_standard_transforms()

        elif strategy == TTAStrategyType.HEAVY:
            # 重量: 标准 + 旋转
            self.transforms_list = self._build_heavy_transforms()

        elif strategy == TTAStrategyType.GEOSPATIAL:
            # 遥感专用: 翻转 + 90°旋转组合
            self.transforms_list = self._build_geospatial_transforms()

        get_logger().info(
            f"📸 TTA 增强器初始化: 策略={strategy.value}, 增强倍数={len(self.transforms_list)}x"
        )

    def _build_standard_transforms(self) -> List:
        """构建标准策略的变换列表"""
        tfms = [
            lambda x: x,  # 原图
            transforms.functional.hflip,  # 水平翻转
        ]

        # 添加 5-crop 裁剪 (在 generate 时动态处理)
        # 这里只标记需要裁剪
        for scale in self.config.crop_scales:
            crop_size = int(self.image_size * scale)
            # 4角裁剪
            tfms.extend(
                [
                    lambda x, s=crop_size: self._crop_and_resize(x, "top_left", s),
                    lambda x, s=crop_size: self._crop_and_resize(x, "top_right", s),
                    lambda x, s=crop_size: self._crop_and_resize(x, "bottom_left", s),
                    lambda x, s=crop_size: self._crop_and_resize(x, "bottom_right", s),
                    lambda x, s=crop_size: self._crop_and_resize(x, "center", s),
                ]
            )

        return tfms

    def _build_heavy_transforms(self) -> List:
        """构建重量级策略的变换列表"""
        base_tfms = self._build_standard_transforms()

        # 添加旋转
        rotation_tfms = []
        for angle in [90, 180, 270]:
            rotation_tfms.append(lambda x, a=angle: transforms.functional.rotate(x, a))

        return base_tfms + rotation_tfms

    def _build_geospatial_transforms(self) -> List:
        """构建遥感专用策略 - 遥感图像无方向性，可使用所有翻转和 90° 旋转"""
        tfms = []

        # 8 种组合: 原图/水平翻转 × 0°/90°/180°/270° 旋转
        for hflip in [False, True]:
            for angle in [0, 90, 180, 270]:

                def make_transform(h=hflip, a=angle):
                    def transform(x):
                        if h:
                            x = transforms.functional.hflip(x)
                        if a != 0:
                            x = transforms.functional.rotate(x, a)
                        return x

                    return transform

                tfms.append(make_transform())

        return tfms

    def _crop_and_resize(
        self, img: torch.Tensor, position: str, crop_size: int
    ) -> torch.Tensor:
        """裁剪并 resize 回原始尺寸

        Args:
            img: [C, H, W] 图像张量
            position: 裁剪位置 ("top_left", "top_right", "bottom_left", "bottom_right", "center")
            crop_size: 裁剪尺寸
        """
        _, h, w = img.shape

        # 计算裁剪坐标
        if position == "top_left":
            top, left = 0, 0
        elif position == "top_right":
            top, left = 0, w - crop_size
        elif position == "bottom_left":
            top, left = h - crop_size, 0
        elif position == "bottom_right":
            top, left = h - crop_size, w - crop_size
        else:  # center
            top, left = (h - crop_size) // 2, (w - crop_size) // 2

        # 裁剪
        cropped = img[:, top : top + crop_size, left : left + crop_size]

        # Resize 回原始尺寸
        cropped = cropped.unsqueeze(0)  # [1, C, H, W]
        resized = F.interpolate(
            cropped,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)  # [C, H, W]

    def generate_augmented_batch(self, images: torch.Tensor) -> torch.Tensor:
        """生成批量图像的所有增强副本

        Args:
            images: [B, C, H, W] 输入图像批次

        Returns:
            [B * num_augmentations, C, H, W] 增强后的图像
        """
        batch_size = images.shape[0]
        num_augs = len(self.transforms_list)

        augmented = []
        for tfm in self.transforms_list:
            # 对整个批次应用变换
            aug_batch = torch.stack([tfm(img) for img in images])
            augmented.append(aug_batch)

        # [num_augs, B, C, H, W] -> [B * num_augs, C, H, W]
        result = torch.stack(augmented, dim=1)  # [B, num_augs, C, H, W]
        return result.view(batch_size * num_augs, *images.shape[1:])

    @property
    def num_augmentations(self) -> int:
        """返回增强变换数量"""
        return len(self.transforms_list)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ TTA 推理函数                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_all_models_logits_with_tta(
    models: List[nn.Module],
    loader: DataLoader,
    augmentor: TTAAugmentor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """支持 TTA 的模型推理

    对每个样本生成多个增强副本，分别推理后取平均概率。

    Args:
        models: 模型列表
        loader: 数据加载器
        augmentor: TTA 增强器
        device: 计算设备 (已弃用，使用模型自身设备)

    Returns:
        logits: [num_models, num_samples, num_classes]
        targets: [num_samples]
    """
    num_augs = augmentor.num_augmentations
    all_logits, all_targets = [], []

    get_logger().info(f"🔄 TTA 推理: 每样本生成 {num_augs} 个增强副本")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="TTA Inference", leave=False):
            batch_size = x.shape[0]

            # 1. 生成增强批次 [B * num_augs, C, H, W]
            x_aug = augmentor.generate_augmented_batch(x)

            # 2. 多模型推理 (支持多 GPU)
            batch_logits = _infer_models_on_batch_multi_gpu_tta(models, x_aug)
            # batch_logits: [num_models, B * num_augs, num_classes]

            # 3. 重塑并聚合: [num_models, B, num_augs, num_classes]
            num_models = len(models)
            num_classes = batch_logits.shape[-1]
            batch_logits = batch_logits.view(
                num_models, batch_size, num_augs, num_classes
            )

            # 4. 概率平均聚合
            # 先转换为概率，再平均，最后转回 logits
            probs = F.softmax(batch_logits, dim=-1)  # [M, B, A, C]
            avg_probs = probs.mean(dim=2)  # [M, B, C]
            # 转回 logits (log 概率)
            eps = 1e-8
            avg_logits = torch.log(avg_probs + eps)  # [M, B, C]

            all_logits.append(avg_logits)
            all_targets.append(y)

    if not all_logits:
        return torch.tensor([]), torch.tensor([])

    return torch.cat(all_logits, dim=1), torch.cat(all_targets)


def _infer_models_on_batch_multi_gpu_tta(
    models: List[nn.Module], x: torch.Tensor
) -> torch.Tensor:
    """多 GPU 并行推理 (TTA 版本)

    Args:
        models: 模型列表
        x: [B * num_augs, C, H, W] 增强后的输入

    Returns:
        [num_models, B * num_augs, num_classes]
    """
    batch_res = []
    for m in models:
        m.eval()
        model_device = next(m.parameters()).device
        x_dev = x.to(model_device)
        out = m(x_dev).unsqueeze(0).cpu()
        batch_res.append(out)
    return torch.cat(batch_res, dim=0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 便捷函数                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def create_tta_augmentor_for_dataset(
    dataset_name: str, cfg: dict, image_size: int
) -> TTAAugmentor:
    """根据数据集类型创建合适的 TTA 增强器

    自动为特定数据集选择最优策略:
        - CIFAR-10/100: light (小图像增强空间有限)
        - EuroSAT: geospatial (遥感专用)
        - ImageNet/FGVC-Aircraft: standard/heavy (大图像效果显著)

    Args:
        dataset_name: 数据集名称
        cfg: 配置字典
        image_size: 图像尺寸

    Returns:
        配置好的 TTAAugmentor
    """
    # 如果用户明确指定了策略，使用用户配置
    if cfg.get("tta_strategy"):
        return TTAAugmentor.from_config(cfg, image_size)

    # 自动推断最优策略
    auto_strategy = "standard"

    if "cifar" in dataset_name.lower():
        auto_strategy = "light"
        get_logger().info(f"📊 {dataset_name}: 自动选择 'light' TTA 策略 (小图像)")
    elif "eurosat" in dataset_name.lower():
        auto_strategy = "geospatial"
        get_logger().info(f"🛰️ {dataset_name}: 自动选择 'geospatial' TTA 策略 (遥感)")
    elif "fgvc" in dataset_name.lower() or "aircraft" in dataset_name.lower():
        auto_strategy = "heavy"
        get_logger().info(f"✈️ {dataset_name}: 自动选择 'heavy' TTA 策略 (细粒度)")
    elif image_size >= 224:
        auto_strategy = "standard"
        get_logger().info(f"🖼️ {dataset_name}: 自动选择 'standard' TTA 策略 (大图像)")

    # 覆盖配置并创建增强器
    cfg_copy = cfg.copy()
    cfg_copy["tta_strategy"] = auto_strategy
    cfg_copy["tta_enabled"] = True

    return TTAAugmentor.from_config(cfg_copy, image_size)
