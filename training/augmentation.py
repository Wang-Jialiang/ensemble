"""
================================================================================
数据增强模块
================================================================================

_CloudMaskGenerator (内部)、各种数据增强方法、增强方法注册表
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 云状Mask生成器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class _CloudMaskGenerator:
    """GPU加速的云状Mask生成器"""

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        persistence: float,
        octaves: int,
        scale_ratio: float,
    ):
        self.h = height
        self.w = width
        self.device = device
        self.persistence = persistence
        self.octaves = octaves
        # base_scale 随图像尺寸动态调整，使用 scale_ratio 控制碎裂程度
        self.base_scale = min(height, width) * scale_ratio

    def generate_batch(
        self, num_masks: int, target_ratio: float = 0.3
    ) -> List[torch.Tensor]:
        """批量产生 Perlin 噪声掩码包"""
        masks = []
        for _ in range(num_masks):
            # 1. 执行具体的噪声生成
            noise = self._get_noise_map()

            # 2. 基于分位数执行掩码量化
            masks.append(self._threshold_noise(noise, target_ratio))
        return masks

    def _get_noise_map(self):
        """生成一张随机缩放的底图"""
        scale = self.base_scale * random.uniform(0.8, 1.2)
        return self._generate_perlin_noise(scale, self.octaves, self.persistence)

    def _threshold_noise(self, noise, ratio):
        """执行二进制量子化"""
        threshold = torch.quantile(noise, 1.0 - ratio)
        mask = (noise < threshold).float()
        return mask.view(1, 1, self.h, self.w)  # [1, 1, H, W]

    def _generate_perlin_noise(
        self, scale: float, octaves: int = 4, persistence: float = 0.5
    ) -> torch.Tensor:
        """生成Perlin噪声"""
        noise = torch.zeros(self.h, self.w, device=self.device)
        amplitude = 1.0
        max_val = 0.0

        for i in range(octaves):
            freq = 2**i
            grid_h = max(2, int(self.h / (scale / freq)))
            grid_w = max(2, int(self.w / (scale / freq)))

            # 1. 使用更大一点的网格并改用正态分布 (randn)，形态更自然
            rand_grid = torch.randn(grid_h + 2, grid_w + 2, device=self.device)

            # 2. 随机空间偏移：让每一层细节在空间上错开，消除格点感
            off_h = random.randint(0, 1)
            off_w = random.randint(0, 1)
            sub_grid = rand_grid[off_h : off_h + grid_h + 1, off_w : off_w + grid_w + 1]

            # 3. 双三次插值 (更平滑)
            upsampled = F.interpolate(
                sub_grid.unsqueeze(0).unsqueeze(0),
                size=(self.h, self.w),
                mode="bicubic",
                align_corners=True,
            ).squeeze()

            noise += upsampled * amplitude
            max_val += amplitude
            amplitude *= self.persistence

        # 4. 归一化并截断 (防止 bicubic 造成的数值溢出)
        return (noise / max_val).clamp_(0, 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 增强方法注册表                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

AUGMENTATION_REGISTRY: dict = {}


def register_augmentation(name: str):
    """类装饰器：注册增强方法

    被装饰的类必须实现 from_config(cls, device, cfg) 类方法。

    用法:
        @register_augmentation("method_name")
        class MethodAugmentation(AugmentationMethod):
            @classmethod
            def from_config(cls, device, cfg):
                return cls(device, ...)
    """

    def decorator(cls):
        if not hasattr(cls, "from_config"):
            raise AttributeError(
                f"{cls.__name__} must implement from_config classmethod"
            )
        AUGMENTATION_REGISTRY[name] = cls.from_config
        return cls

    return decorator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 数据增强方法                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class AugmentationMethod:
    """数据增强方法基类

    所有增强方法支持预计算 mask 池，训练时从池中随机选择。
    """

    def __init__(self, device: torch.device, pool_size: int = 1024):
        self.device = device
        self._pool_size = pool_size
        self._masks: List[torch.Tensor] = []

    @staticmethod
    def _compute_fill_value(cfg) -> float:
        """计算归一化后的填充值（黑色）"""
        mean, std = np.mean(cfg.dataset_mean), np.mean(cfg.dataset_std)
        return (0.0 - mean) / std

    def precompute_masks(self, target_ratio: float):
        """预计算 mask 池，子类必须实现"""
        raise NotImplementedError

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强方法，子类必须实现"""
        raise NotImplementedError


@register_augmentation("cutout")
class CutoutAugmentation(AugmentationMethod):
    """Cutout 硬遮挡

    预计算多个随机位置的方形遮挡 mask，训练时从池中随机抽取。
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        pool_size: int = 1024,
        fill_value: float = 0.0,
    ):
        super().__init__(device, pool_size)
        self.height, self.width = height, width
        self._fill_value = fill_value

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 Cutout mask 池"""
        H, W = self.height, self.width
        size = int(H * np.sqrt(target_ratio))
        self._masks = []
        for _ in range(self._pool_size):
            mask = torch.ones(1, 1, H, W, device=self.device)
            if size > 0:
                y = random.randint(0, max(0, H - size))
                x = random.randint(0, max(0, W - size))
                mask[:, :, y : y + size, x : x + size] = 0
            self._masks.append(mask)

    def apply(self, images, targets, ratio, prob) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Cutout 增强"""
        if random.random() > prob:
            return images, targets
        if not self._masks:
            raise RuntimeError("Cutout: call precompute_masks() first")

        B, C, H, W = images.shape
        augmented = images.clone()
        for b in range(B):
            m = random.choice(self._masks).expand(-1, C, -1, -1).squeeze(0)
            augmented[b] = images[b] * m + (1 - m) * self._fill_value
        return augmented, targets


@register_augmentation("pixel_has")
class PixelHaSAugmentation(AugmentationMethod):
    """像素级 Hide-and-Seek (1×1 HaS)

    预计算多个随机像素 mask，训练时从池中随机抽取。
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        channels: int = 3,
        pool_size: int = 1024,
        fill_value: float = 0.0,
    ):
        super().__init__(device, pool_size)
        self.height, self.width, self.channels = height, width, channels
        self._fill_value = fill_value

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.num_channels,
            cfg.mask_pool_size,
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 PixelHaS mask 池"""
        H, W, C = self.height, self.width, self.channels
        self._masks = []
        for _ in range(self._pool_size):
            mask = (torch.rand((1, C, H, W), device=self.device) > target_ratio).float()
            self._masks.append(mask)

    def apply(self, images, targets, ratio, prob) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用像素级 HaS 增强"""
        if random.random() > prob:
            return images, targets
        if not self._masks:
            raise RuntimeError("PixelHaS: call precompute_masks() first")

        B, C, H, W = images.shape
        augmented = images.clone()
        for b in range(B):
            m = random.choice(self._masks).squeeze(0)  # [C, H, W]
            augmented[b] = images[b] * m + (1 - m) * self._fill_value
        return augmented, targets


@register_augmentation("gridmask")
class GridMaskAugmentation(AugmentationMethod):
    """GridMask 网格遮挡

    预计算多个网格 mask，训练时从池中随机抽取。
    参考: GridMask Data Augmentation (Chen et al., 2020)
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        pool_size: int = 1024,
        d_ratio_min: float = 0.1,
        d_ratio_max: float = 0.3,
        fill_value: float = 0.0,
    ):
        super().__init__(device, pool_size)
        self.height, self.width = height, width
        self.d_ratio_min = d_ratio_min
        self.d_ratio_max = d_ratio_max
        self._fill_value = fill_value

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cfg.gridmask_d_ratio_min,
            cfg.gridmask_d_ratio_max,
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 GridMask mask 池"""
        H, W = self.height, self.width
        self._masks = []
        for _ in range(self._pool_size):
            self._masks.append(self._generate_single_mask(H, W, target_ratio))

    def _generate_single_mask(self, H, W, ratio):
        """生成单张网格 Mask"""
        min_dim = min(H, W)
        d = random.randint(
            max(4, int(min_dim * self.d_ratio_min)),
            max(8, int(min_dim * self.d_ratio_max)),
        )
        off_x, off_y = random.randint(0, d - 1), random.randint(0, d - 1)
        block_len = int(d * np.sqrt(ratio))

        mask = torch.ones(1, 1, H, W, device=self.device)
        for i in range(-1, H // d + 1):
            for j in range(-1, W // d + 1):
                y1, y2 = max(0, i * d + off_y), min(H, i * d + off_y + block_len)
                x1, x2 = max(0, j * d + off_x), min(W, j * d + off_x + block_len)
                if y2 > y1 and x2 > x1:
                    mask[:, :, y1:y2, x1:x2] = 0
        return mask

    def apply(self, images, targets, ratio, prob) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 GridMask 增强"""
        if random.random() > prob:
            return images, targets
        if not self._masks:
            raise RuntimeError("GridMask: call precompute_masks() first")

        B, C, H, W = images.shape
        augmented = images.clone()
        for b in range(B):
            m = random.choice(self._masks).expand(-1, C, -1, -1).squeeze(0)
            augmented[b] = images[b] * m + (1 - m) * self._fill_value
        return augmented, targets


@register_augmentation("perlin")
class PerlinMaskAugmentation(AugmentationMethod):
    """Perlin 噪声遮挡

    预计算多个云状 mask，训练时从池中随机抽取。
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        pool_size: int = 1024,
        persistence: float = 0.5,
        octaves: int = 4,
        scale_ratio: float = 0.3,
        fill_value: float = 0.0,
    ):
        super().__init__(device, pool_size)
        self.height, self.width = height, width
        self.mask_generator = _CloudMaskGenerator(
            height,
            width,
            device,
            persistence,
            octaves,
            scale_ratio,
        )
        self._fill_value = fill_value

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cfg.perlin_persistence,
            cfg.perlin_octaves,
            cfg.perlin_scale_ratio,
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 Perlin mask 池"""
        self._masks = self.mask_generator.generate_batch(self._pool_size, target_ratio)

    def apply(self, images, targets, ratio, prob) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Perlin 噪声增强"""
        if random.random() > prob:
            return images, targets
        if not self._masks:
            raise RuntimeError("Perlin: call precompute_masks() first")

        B, C, H, W = images.shape
        augmented = images.clone()
        for b in range(B):
            m = random.choice(self._masks).expand(-1, C, -1, -1).squeeze(0)
            augmented[b] = images[b] * m + (1 - m) * self._fill_value
        return augmented, targets


@register_augmentation("none")
class NoAugmentation(AugmentationMethod):
    """无增强（Baseline）"""

    @classmethod
    def from_config(cls, device, cfg):
        return cls(device)

    def precompute_masks(self, target_ratio: float):
        """无操作"""
        pass

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return images, targets
