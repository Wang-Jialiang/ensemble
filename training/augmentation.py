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
        scale_ratio_min: float,
        scale_ratio_max: float,
    ):
        self.h = height
        self.w = width
        self.device = device
        self.persistence = persistence
        self.octaves = octaves
        # scale_ratio 范围，每次生成时随机选择
        self.scale_ratio_min = scale_ratio_min
        self.scale_ratio_max = scale_ratio_max

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
        # 在 [scale_ratio_min, scale_ratio_max] 范围随机选择尺度比例
        scale_ratio = random.uniform(self.scale_ratio_min, self.scale_ratio_max)
        scale = min(self.h, self.w) * scale_ratio
        return self._generate_perlin_noise(scale)

    def _threshold_noise(self, noise, ratio):
        """执行二进制量子化"""
        threshold = torch.quantile(noise, 1.0 - ratio)
        mask = (noise < threshold).float()
        return mask.view(1, 1, self.h, self.w)  # [1, 1, H, W]

    def _generate_perlin_noise(self, scale: float) -> torch.Tensor:
        """生成Perlin噪声"""
        # 防止 scale 为 0 导致除零错误
        scale = max(scale, 1.0)

        noise = torch.zeros(self.h, self.w, device=self.device)
        amplitude = 1.0
        max_val = 0.0

        for i in range(self.octaves):
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

        # 4. 归一化到 [0, 1] (使用 min-max 归一化，避免 randn 产生的负值导致 mask 全黑)
        noise = noise / max_val
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


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
    """数据增强方法基类。

    所有增强方法支持预计算 mask 池，训练时从池中随机选择。

    Attributes:
        device: 计算设备 (CPU/GPU)。
        _pool_size: mask 池大小。
        _masks: 预计算的 mask 列表。
        _fill_value: 遮罩区域的填充值。
    """

    def __init__(
        self,
        device: torch.device,
        pool_size: int = 1024,
        fill_value: torch.Tensor = None,
    ):
        self.device = device
        self._pool_size = pool_size
        self._masks: List[torch.Tensor] = []
        self._fill_value = fill_value.to(device) if fill_value is not None else None

    @staticmethod
    def _compute_fill_value(cfg, use_mean: bool = False) -> torch.Tensor:
        """计算归一化后的填充值。

        Args:
            cfg: 配置对象，包含 dataset_mean 和 dataset_std。
            use_mean: 如果为 True，填充均值（归一化后为 0）；否则填充黑色。

        Returns:
            torch.Tensor: 形状为 [C, 1, 1] 的填充值张量。
        """
        if use_mean:
            # 均值在归一化后为 0
            num_channels = len(cfg.dataset_mean)
            return torch.zeros(num_channels, 1, 1, dtype=torch.float32)
        else:
            # 黑色 (0, 0, 0) 归一化后的值
            mean = np.array(cfg.dataset_mean)
            std = np.array(cfg.dataset_std)
            fill = (0.0 - mean) / std  # [C]
            return torch.tensor(fill, dtype=torch.float32).view(-1, 1, 1)  # [C, 1, 1]

    def _clear_masks(self):
        """清理旧的 mask 池并释放显存。"""
        if self._masks:
            del self._masks
            self._masks = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def precompute_masks(self, target_ratio: float):
        """预计算 mask 池。

        Args:
            target_ratio: 目标遮罩比例 (0.0-1.0)。

        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        raise NotImplementedError

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强方法。

        默认实现：单通道 mask + 分通道填充。子类如有特殊 mask 形状需覆盖此方法。

        Args:
            images: 输入图像，形状为 [B, C, H, W]。
            targets: 目标标签。
            ratio: 当前遮罩比例（未使用，由 precompute_masks 决定）。
            prob: 应用增强的概率。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 增强后的图像和标签。

        Raises:
            RuntimeError: 如果未调用 precompute_masks()。
        """
        if random.random() > prob:
            return images, targets
        if not self._masks:
            raise RuntimeError(
                f"{self.__class__.__name__}: call precompute_masks() first"
            )

        B, C, H, W = images.shape
        augmented = images.clone()
        for b in range(B):
            m = random.choice(self._masks).squeeze(0)  # [1, H, W]
            augmented[b] = (
                images[b] * m + (1 - m) * self._fill_value
            )  # [C, 1, 1] broadcasts
        return augmented, targets


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
        fill_value: torch.Tensor = None,
    ):
        super().__init__(device, pool_size, fill_value)
        self.height, self.width = height, width

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cls._compute_fill_value(cfg, cfg.augmentation_use_mean_fill),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 Cutout mask 池"""
        self._clear_masks()  # 清理旧 mask，释放显存
        H, W = self.height, self.width
        size = int(H * np.sqrt(target_ratio))
        for _ in range(self._pool_size):
            mask = torch.ones(1, 1, H, W, device=self.device)
            if size > 0:
                y = random.randint(0, max(0, H - size))
                x = random.randint(0, max(0, W - size))
                mask[:, :, y : y + size, x : x + size] = 0
            self._masks.append(mask)


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
        pool_size: int = 1024,
        fill_value: torch.Tensor = None,
    ):
        super().__init__(device, pool_size, fill_value)
        self.height, self.width = height, width

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cls._compute_fill_value(cfg, cfg.augmentation_use_mean_fill),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 PixelHaS mask 池（单通道，每像素独立随机）"""
        self._clear_masks()  # 清理旧 mask，释放显存
        H, W = self.height, self.width
        for _ in range(self._pool_size):
            mask = (torch.rand((1, 1, H, W), device=self.device) > target_ratio).float()
            self._masks.append(mask)


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
        d_ratio_min: float = 0.4,
        d_ratio_max: float = 0.6,
        fill_value: torch.Tensor = None,
    ):
        super().__init__(device, pool_size, fill_value)
        self.height, self.width = height, width
        self.d_ratio_min = d_ratio_min
        self.d_ratio_max = d_ratio_max

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
        self._clear_masks()  # 清理旧 mask，释放显存
        H, W = self.height, self.width
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
        scale_ratio_min: float = 0.20,
        scale_ratio_max: float = 0.30,
        fill_value: torch.Tensor = None,
    ):
        super().__init__(device, pool_size, fill_value)
        self.height, self.width = height, width
        self.mask_generator = _CloudMaskGenerator(
            height,
            width,
            device,
            persistence,
            octaves,
            scale_ratio_min,
            scale_ratio_max,
        )

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cfg.perlin_persistence,
            cfg.perlin_octaves,
            cfg.perlin_scale_ratio_min,
            cfg.perlin_scale_ratio_max,
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算 Perlin mask 池"""
        self._clear_masks()  # 清理旧 mask，释放显存
        self._masks = self.mask_generator.generate_batch(self._pool_size, target_ratio)


@register_augmentation("gridded_perlin")
class GriddedPerlinAugmentation(AugmentationMethod):
    """网格化 Perlin 噪声遮挡

    将 Perlin 云状遮挡分布在规则网格位置上，形成分散的柔和云块。
    结合了 GridMask 的规则分布和 Perlin 噪声的自然边缘。
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        pool_size: int = 1024,
        persistence: float = 0.5,
        octaves: int = 4,
        grid_size: int = 32,
        cloud_ratio: float = 0.6,
        fill_value: torch.Tensor = None,
    ):
        super().__init__(device, pool_size, fill_value)
        self.height, self.width = height, width
        self.persistence = persistence
        self.octaves = octaves
        self.grid_size = grid_size
        self.cloud_ratio = cloud_ratio  # 每个网格内云块占比

    @classmethod
    def from_config(cls, device, cfg):
        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            cfg.perlin_persistence,
            cfg.perlin_octaves,
            getattr(cfg, "gridded_perlin_grid_size", 32),
            getattr(cfg, "gridded_perlin_cloud_ratio", 0.6),
            cls._compute_fill_value(cfg),
        )

    def precompute_masks(self, target_ratio: float):
        """预计算网格化 Perlin mask 池"""
        self._clear_masks()
        for _ in range(self._pool_size):
            self._masks.append(self._generate_gridded_perlin_mask(target_ratio))

    def _generate_gridded_perlin_mask(self, target_ratio: float) -> torch.Tensor:
        """生成单张网格化 Perlin mask"""
        H, W = self.height, self.width
        grid_size = self.grid_size

        # 1. 生成基础 Perlin 噪声
        noise = self._generate_perlin_noise()

        # 2. 创建网格蒙版 (每个格子中心放一个圆形柔和区域)
        grid_mask = torch.zeros(H, W, device=self.device)
        radius = int(grid_size * self.cloud_ratio / 2)

        # 随机偏移，增加多样性
        off_y = random.randint(0, grid_size - 1)
        off_x = random.randint(0, grid_size - 1)

        for i in range(-1, H // grid_size + 2):
            for j in range(-1, W // grid_size + 2):
                # 格子中心
                cy = i * grid_size + grid_size // 2 + off_y
                cx = j * grid_size + grid_size // 2 + off_x

                # 创建柔和的圆形区域
                y_coords = torch.arange(H, device=self.device).float()
                x_coords = torch.arange(W, device=self.device).float()
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

                dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                # 柔和边缘：从中心到边缘渐变
                soft_circle = torch.clamp(1.0 - dist / radius, 0.0, 1.0)
                grid_mask = torch.maximum(grid_mask, soft_circle)

        # 3. 将 Perlin 噪声与网格蒙版相乘
        combined = noise * grid_mask

        # 4. 基于分位数阈值化
        threshold = torch.quantile(combined[combined > 0], 1.0 - target_ratio)
        mask = (combined < threshold).float()

        return mask.view(1, 1, H, W)

    def _generate_perlin_noise(self) -> torch.Tensor:
        """生成 Perlin 噪声底图"""
        H, W = self.height, self.width
        scale = min(H, W) * 0.15  # 固定较小尺度，云块更紧凑

        noise = torch.zeros(H, W, device=self.device)
        amplitude = 1.0
        max_val = 0.0

        for i in range(self.octaves):
            freq = 2**i
            grid_h = max(2, int(H / (scale / freq)))
            grid_w = max(2, int(W / (scale / freq)))

            rand_grid = torch.randn(grid_h + 2, grid_w + 2, device=self.device)
            off_h = random.randint(0, 1)
            off_w = random.randint(0, 1)
            sub_grid = rand_grid[off_h : off_h + grid_h + 1, off_w : off_w + grid_w + 1]

            upsampled = F.interpolate(
                sub_grid.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bicubic",
                align_corners=True,
            ).squeeze()

            noise += upsampled * amplitude
            max_val += amplitude
            amplitude *= self.persistence

        noise = noise / max_val
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)


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
