"""
================================================================================
数据增强模块
================================================================================

云状Mask生成器、各种数据增强方法、增强方法注册表
"""

import random
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ 常量定义                                                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

# Perlin 噪声生成参数
PERLIN_OCTAVES_LARGE = 4  # 图像尺寸 >= 64 时的 octaves
PERLIN_OCTAVES_SMALL = 3  # 图像尺寸 < 64 时的 octaves

# Cutout 增强的填充值
CUTOUT_FILL_VALUE = 0.5


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 云状Mask生成器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CloudMaskGenerator:
    """GPU加速的云状Mask生成器"""

    def __init__(self, height: int, width: int, device: torch.device):
        self.h = height
        self.w = width
        self.device = device
        # base_scale 随图像尺寸动态调整: 32x32 -> 16, 64x64 -> 32
        self.base_scale = min(height, width) / 2.0

    def generate_batch(
        self, num_masks: int, target_ratio: float = 0.3
    ) -> List[torch.Tensor]:
        """批量生成Perlin噪声Mask"""
        masks = []
        for _ in range(num_masks):
            # 动态调整 octaves 参数
            scale = self.base_scale * random.uniform(0.8, 1.2)
            octaves = PERLIN_OCTAVES_LARGE if self.h >= 64 else PERLIN_OCTAVES_SMALL
            persistence = 0.5

            noise = self._generate_perlin_noise(scale, octaves, persistence)
            # 使用target_ratio作为阈值
            threshold = torch.quantile(noise, 1.0 - target_ratio)
            mask = (noise < threshold).float()
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            masks.append(mask)

        return masks

    def _generate_perlin_noise(
        self, scale: float, octaves: int = 4, persistence: float = 0.5
    ) -> torch.Tensor:
        """生成Perlin噪声"""
        noise = torch.zeros(self.h, self.w, device=self.device)
        amplitude = 1.0
        max_val = 0.0

        for i in range(octaves):
            freq = 2**i
            # 确保频率不会太高导致尺寸为0
            grid_h = max(2, int(self.h / (scale / freq)))
            grid_w = max(2, int(self.w / (scale / freq)))

            rand_grid = torch.rand(grid_h + 1, grid_w + 1, device=self.device)

            # 双线性插值
            upsampled = F.interpolate(
                rand_grid.unsqueeze(0).unsqueeze(0),
                size=(self.h, self.w),
                mode="bilinear",
                align_corners=True,
            ).squeeze()

            noise += upsampled * amplitude
            max_val += amplitude
            amplitude *= persistence

        return noise / max_val


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据增强方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class AugmentationMethod:
    """数据增强方法基类"""

    def __init__(self, device: torch.device):
        self.device = device

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强方法"""
        raise NotImplementedError


class CutoutAugmentation(AugmentationMethod):
    """Cutout硬遮挡"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape
        mask_size = int(H * np.sqrt(ratio))

        augmented = images.clone()
        for i in range(B):
            y = random.randint(0, max(0, H - mask_size))
            x = random.randint(0, max(0, W - mask_size))
            augmented[i, :, y : y + mask_size, x : x + mask_size] = CUTOUT_FILL_VALUE

        return augmented, targets


class MixupAugmentation(AugmentationMethod):
    """Mixup混合增强"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        # 边界检查防止 beta 分布参数无效
        ratio = np.clip(ratio, 0.01, 0.99)
        lam = np.random.beta(ratio * 10, (1 - ratio) * 10)
        lam = max(lam, 1 - lam)

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, targets


class CutMixAugmentation(AugmentationMethod):
    """CutMix剪切混合"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape

        lam = np.random.beta(1.0, 1.0)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        index = torch.randperm(B).to(self.device)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            index, :, bby1:bby2, bbx1:bbx2
        ]

        return mixed_images, targets


class DropoutAugmentation(AugmentationMethod):
    """特征级Dropout"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        mask = torch.rand_like(images) > ratio
        augmented = images * mask.float()
        return augmented, targets


class PerlinMaskAugmentation(AugmentationMethod):
    """Perlin噪声遮挡（原方法）"""

    def __init__(
        self, device: torch.device, height: int, width: int, pool_size: int = 100
    ):
        super().__init__(device)
        self.mask_generator = CloudMaskGenerator(height, width, device)
        self.masks = []
        self.pool_size = pool_size

    def precompute_masks(self, target_ratio: float):
        """预计算mask池"""
        self.masks = self.mask_generator.generate_batch(self.pool_size, target_ratio)

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob or not self.masks:
            return images, targets

        mask = self.masks[random.randint(0, len(self.masks) - 1)]
        if mask.shape[1] == 1:
            mask = mask.expand(1, 3, -1, -1)

        augmented = images * mask
        return augmented, targets


class NoAugmentation(AugmentationMethod):
    """无增强（Baseline）"""

    def apply(
        self, images: torch.Tensor, targets: torch.Tensor, ratio: float, prob: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return images, targets


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 增强方法注册表                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

AUGMENTATION_REGISTRY = {
    "cutout": lambda device, cfg: CutoutAugmentation(device),
    "mixup": lambda device, cfg: MixupAugmentation(device),
    "cutmix": lambda device, cfg: CutMixAugmentation(device),
    "dropout": lambda device, cfg: DropoutAugmentation(device),
    "perlin": lambda device, cfg: PerlinMaskAugmentation(
        device, cfg.image_size, cfg.image_size, cfg.mask_pool_size
    ),
    "none": lambda device, cfg: NoAugmentation(device),
}


def register_augmentation(name: str, builder: Callable):
    """动态注册增强方法"""
    AUGMENTATION_REGISTRY[name] = builder
