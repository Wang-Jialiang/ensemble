"""
================================================================================
数据增强模块
================================================================================

云状Mask生成器、各种数据增强方法、增强方法注册表
"""

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 云状Mask生成器                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class CloudMaskGenerator:
    """GPU加速的云状Mask生成器"""

    def __init__(
        self,
        height: int,
        width: int,
        device: torch.device,
        persistence: float,
        octaves_large: int,
        octaves_small: int,
        scale_ratio: float = 0.3,
    ):
        self.h = height
        self.w = width
        self.device = device
        self.persistence = persistence
        self.octaves_large = octaves_large
        self.octaves_small = octaves_small
        # base_scale 随图像尺寸动态调整，使用 scale_ratio 控制碎裂程度
        self.base_scale = min(height, width) * scale_ratio

    def generate_batch(
        self, num_masks: int, target_ratio: float = 0.3
    ) -> List[torch.Tensor]:
        """批量生成Perlin噪声Mask"""
        masks = []
        for _ in range(num_masks):
            # 动态调整 octaves 参数
            scale = self.base_scale * random.uniform(0.8, 1.2)
            octaves = self.octaves_large if self.h >= 64 else self.octaves_small
            persistence = self.persistence

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

    支持两种模式:
    - 样本级 (model_index=None): 每个样本随机不同的 mask
    - 模型级 (model_index=int): 每个模型固定 seed，动态生成 mask (ratio 可变)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model_pools_initialized = False

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed（子类实现）

        每个模型分配唯一 seed，用于动态生成 mask。
        同一 seed 生成的噪声底图一致，ratio 参数控制遮挡面积。

        Args:
            num_models: 模型数量
            base_seed: 基础 seed，None 则随机生成
        """
        pass

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用增强方法

        Args:
            images: 输入图像 [B, C, H, W]
            targets: 标签
            ratio: 遮挡比例
            prob: 应用概率
            model_index: 模型索引，None 表示样本级随机，int 表示使用模型级固定池
        """
        raise NotImplementedError


@register_augmentation("cutout")
class CutoutAugmentation(AugmentationMethod):
    """Cutout硬遮挡

    支持两种模式:
    - 样本级: 每个样本随机位置
    - 模型级: 每个模型固定 seed，动态生成位置
    """

    def __init__(self, device: torch.device, fill_value: float = 0.0):
        super().__init__(device)
        self.fill_value = fill_value
        self.model_seeds = {}  # {model_idx: seed}

    @classmethod
    def from_config(cls, device, cfg):
        # 强制使用 Black 填充: (0 - mean) / std
        # 计算数据集归一化后的"绝对黑色"值
        mean = np.mean(cfg.dataset_mean)
        std = np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std

        return cls(device, fill_value=fill_value)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self.model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape
        mask_size = int(H * np.sqrt(ratio))
        augmented = images.clone()

        if model_index is not None and self.model_pools_initialized:
            # 模型级: 使用固定 seed 生成位置
            seed = self.model_seeds[model_index]
            rng = random.Random(seed)
            y = rng.randint(0, max(0, H - mask_size))
            x = rng.randint(0, max(0, W - mask_size))
            # 所有样本使用相同位置
            for i in range(B):
                augmented[i, :, y : y + mask_size, x : x + mask_size] = self.fill_value
        else:
            # 样本级: 每个样本随机位置
            for i in range(B):
                y = random.randint(0, max(0, H - mask_size))
                x = random.randint(0, max(0, W - mask_size))
                augmented[i, :, y : y + mask_size, x : x + mask_size] = self.fill_value

        return augmented, targets


@register_augmentation("pixel_has")
class PixelHaSAugmentation(AugmentationMethod):
    """像素级 Hide-and-Seek (1×1 HaS)

    每个像素独立以概率 ratio 被隐藏 (置零)。
    等价于 block_size=1 的 Hide-and-Seek。

    支持两种模式:
    - 样本级: 每个样本随机 mask
    - 模型级: 每个模型固定 seed
    """

    def __init__(self, device: torch.device, fill_value: float = 0.0):
        super().__init__(device)
        self.fill_value = fill_value
        self.model_seeds = {}  # {model_idx: seed}

    @classmethod
    def from_config(cls, device, cfg):
        # 强制使用 Black 填充
        mean = np.mean(cfg.dataset_mean)
        std = np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std
        return cls(device, fill_value=fill_value)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self.model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        if model_index is not None and self.model_pools_initialized:
            # 模型级: 使用固定 seed 生成 mask
            seed = self.model_seeds[model_index]
            generator = torch.Generator(device=self.device).manual_seed(seed)
            mask = (
                torch.rand(images.shape, device=self.device, generator=generator)
                > ratio
            )
        else:
            # 样本级: 随机 mask
            mask = torch.rand_like(images) > ratio

        augmented = images * mask.float() + (1 - mask.float()) * self.fill_value
        return augmented, targets


@register_augmentation("gridmask")
class GridMaskAugmentation(AugmentationMethod):
    """GridMask 网格遮挡

    使用规则的网格模式遮挡图像，遮挡区域呈均匀分布的正方形网格。
    作为"人造规则形状"的代表，与 Perlin 的自然形状形成对比。
    网格大小会根据图像尺寸自适应缩放。

    支持两种模式:
    - 样本级: 每个样本随机参数
    - 模型级: 每个模型固定 seed，动态生成参数

    参考: GridMask Data Augmentation (Chen et al., 2020)
    """

    def __init__(
        self,
        device: torch.device,
        d_ratio_min: float = 0.1,
        d_ratio_max: float = 0.3,
        fill_value: float = 0.0,
    ):
        """
        Args:
            device: 计算设备
            d_ratio_min: 网格单元最小尺寸占图像尺寸的比例 (默认 0.1 = 10%)
            d_ratio_max: 网格单元最大尺寸占图像尺寸的比例 (默认 0.3 = 30%)
            fill_value: 遮挡填充值
        """
        super().__init__(device)
        self.d_ratio_min = d_ratio_min
        self.d_ratio_max = d_ratio_max
        self.fill_value = fill_value
        self.model_seeds = {}  # {model_idx: seed}

    @classmethod
    def from_config(cls, device, cfg):
        # 强制使用 Black 填充
        mean = np.mean(cfg.dataset_mean)
        std = np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std

        return cls(
            device,
            d_ratio_min=cfg.gridmask_d_ratio_min,
            d_ratio_max=cfg.gridmask_d_ratio_max,
            fill_value=fill_value,
        )

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self.model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B, C, H, W = images.shape
        min_size = min(H, W)
        augmented = images.clone()

        if model_index is not None and self.model_pools_initialized:
            # 模型级: 使用固定 seed 生成参数
            seed = self.model_seeds[model_index]
            rng = random.Random(seed)
            d_min = max(4, int(min_size * self.d_ratio_min))
            d_max = max(8, int(min_size * self.d_ratio_max))
            d = rng.randint(d_min, d_max)
            offset_x = rng.randint(0, d - 1)
            offset_y = rng.randint(0, d - 1)

            block_len = int(d * np.sqrt(ratio))  # ratio 表示面积比例

            # 生成网格 mask
            mask = torch.ones(H, W, device=self.device)
            for i in range(-1, H // d + 1):
                for j in range(-1, W // d + 1):
                    y1 = i * d + offset_y
                    y2 = y1 + block_len
                    x1 = j * d + offset_x
                    x2 = x1 + block_len
                    y1, y2 = max(0, y1), min(H, y2)
                    x1, x2 = max(0, x1), min(W, x2)
                    if y2 > y1 and x2 > x1:
                        mask[y1:y2, x1:x2] = 0

            # 所有样本使用相同的 mask
            # mask: 1=keep, 0=drop
            mask_expanded = mask.unsqueeze(0)
            for b in range(B):
                augmented[b] = (
                    images[b] * mask_expanded + (1 - mask_expanded) * self.fill_value
                )
        else:
            # 样本级: 每个样本随机参数
            for b in range(B):
                d_min = max(4, int(min_size * self.d_ratio_min))
                d_max = max(8, int(min_size * self.d_ratio_max))
                d = random.randint(d_min, d_max)
                offset_x = random.randint(0, d - 1)
                offset_y = random.randint(0, d - 1)

                block_len = int(d * np.sqrt(ratio))

                mask = torch.ones(H, W, device=self.device)
                for i in range(-1, H // d + 1):
                    for j in range(-1, W // d + 1):
                        y1 = i * d + offset_y
                        y2 = y1 + block_len
                        x1 = j * d + offset_x
                        x2 = x1 + block_len
                        y1, y2 = max(0, y1), min(H, y2)
                        x1, x2 = max(0, x1), min(W, x2)
                        if y2 > y1 and x2 > x1:
                            mask[y1:y2, x1:x2] = 0

                augmented[b] = (
                    images[b] * mask.unsqueeze(0)
                    + (1 - mask.unsqueeze(0)) * self.fill_value
                )

        return augmented, targets


@register_augmentation("perlin")
class PerlinMaskAugmentation(AugmentationMethod):
    """Perlin噪声遮挡

    支持两种模式:
    - 样本级: 每个样本从共享池随机选择 mask
    - 模型级: 每个模型拥有固定 seed，动态生成 mask（ratio 可变）
    """

    def __init__(
        self,
        device: torch.device,
        height: int,
        width: int,
        pool_size: int = 100,
        persistence: float = 0.5,
        octaves_large: int = 4,
        octaves_small: int = 3,
        scale_ratio: float = 0.3,
        fill_value: float = 0.0,
    ):
        super().__init__(device)
        self.height = height
        self.width = width
        self.mask_generator = CloudMaskGenerator(
            height,
            width,
            device,
            persistence,
            octaves_large,
            octaves_small,
            scale_ratio,
        )
        self.fill_value = fill_value
        self.masks = []  # 样本级共享池
        self.pool_size = pool_size
        self.model_seeds = {}  # {model_idx: seed} 模型级固定 seed

    @classmethod
    def from_config(cls, device, cfg):
        # 强制使用 Black 填充
        mean = np.mean(cfg.dataset_mean)
        std = np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std

        return cls(
            device,
            cfg.image_size,
            cfg.image_size,
            cfg.mask_pool_size,
            persistence=cfg.perlin_persistence,
            octaves_large=cfg.perlin_octaves_large,
            octaves_small=cfg.perlin_octaves_small,
            scale_ratio=cfg.perlin_scale_ratio,
            fill_value=fill_value,
        )

    def precompute_masks(self, target_ratio: float):
        """预计算样本级共享 mask 池"""
        self.masks = self.mask_generator.generate_batch(self.pool_size, target_ratio)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed

        每个模型分配唯一 seed，用于动态生成 mask。
        同一 seed 生成的 Perlin 噪声底图一致，ratio 控制遮挡面积。

        Args:
            num_models: 模型数量
            base_seed: 基础 seed，None 则随机生成
        """
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self.model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def _generate_mask_with_seed(self, seed: int, ratio: float) -> torch.Tensor:
        """使用固定 seed 生成 Perlin mask

        Args:
            seed: 随机种子，决定噪声底图
            ratio: 遮挡比例，决定遮挡面积

        Returns:
            mask: [1, 1, H, W] 的 mask tensor
        """
        # 保存当前随机状态
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        # 设置固定 seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 生成 mask
        masks = self.mask_generator.generate_batch(1, ratio)
        mask = masks[0]

        # 恢复随机状态
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

        return mask

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > prob:
            return images, targets

        B = images.shape[0]
        C = images.shape[1]
        augmented = images.clone()

        if model_index is not None and self.model_pools_initialized:
            # 模型级: 使用固定 seed 动态生成 mask (ratio 生效)
            seed = self.model_seeds[model_index]
            mask = self._generate_mask_with_seed(seed, ratio)
            if mask.shape[1] == 1:
                mask = mask.expand(1, C, -1, -1)
            # 所有样本使用相同的 mask
            # 所有样本使用相同的 mask
            mask_expanded = mask.squeeze(0)  # [C, H, W]
            for b in range(B):
                augmented[b] = (
                    images[b] * mask_expanded + (1 - mask_expanded) * self.fill_value
                )
        else:
            # 样本级: 从共享池随机选择
            if not self.masks:
                raise RuntimeError(
                    "PerlinMaskAugmentation: must call precompute_masks() before apply()"
                )
            for b in range(B):
                mask = random.choice(self.masks)
                if mask.shape[1] == 1:
                    mask = mask.expand(1, C, -1, -1)

                mask_expanded = mask.squeeze(0)
                augmented[b] = (
                    images[b] * mask_expanded + (1 - mask_expanded) * self.fill_value
                )

        return augmented, targets


@register_augmentation("none")
class NoAugmentation(AugmentationMethod):
    """无增强（Baseline）"""

    @classmethod
    def from_config(cls, device, cfg):
        return cls(device)

    def apply(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        ratio: float,
        prob: float,
        model_index: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return images, targets
