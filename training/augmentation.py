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
        scale_ratio: float,
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
        octaves = self.octaves_large if self.h >= 64 else self.octaves_small
        return self._generate_perlin_noise(scale, octaves, self.persistence)

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
        self._fill_value = fill_value
        self._model_seeds = {}  # {model_idx: seed}

    @classmethod
    def from_config(cls, device, cfg):
        # 1. 计算数据集归一化后的"绝对黑色"值
        mean, std = np.mean(cfg.dataset_mean), np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std
        return cls(device, fill_value=fill_value)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self._model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self, images, targets, ratio, prob, model_index=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Cutout 增强 (主流程)"""
        if random.random() > prob:
            return images, targets

        # 1. 定位遮挡区域 (y, x, size)
        y, x, size = self._get_region_params(images, ratio, model_index)

        # 2. 执行填充
        augmented = self._fill_cutout(images, y, x, size)
        return augmented, targets

    def _get_region_params(self, images, ratio, model_idx):
        """确定遮挡的具体坐标"""
        B, _, H, W = images.shape
        size = int(H * np.sqrt(ratio))

        # 确定随机数生成器
        if model_idx is not None and self.model_pools_initialized:
            rng = random.Random(self._model_seeds[model_idx])
            y, x = rng.randint(0, max(0, H - size)), rng.randint(0, max(0, W - size))
        else:
            y, x = (
                random.randint(0, max(0, H - size)),
                random.randint(0, max(0, W - size)),
            )

        return y, x, size

    def _fill_cutout(self, images, y, x, size):
        """执行实际的张量填充"""
        augmented = images.clone()
        if size > 0:
            augmented[:, :, y : y + size, x : x + size] = self._fill_value
        return augmented


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
        self._fill_value = fill_value
        self._model_seeds = {}

    @classmethod
    def from_config(cls, device, cfg):
        mean, std = np.mean(cfg.dataset_mean), np.mean(cfg.dataset_std)
        fill_value = (0.0 - mean) / std
        return cls(device, fill_value=fill_value)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self._model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self, images, targets, ratio, prob, model_index=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用像素级 HaS 增强"""
        if random.random() > prob:
            return images, targets

        # 1. 生成掩码
        mask = self._generate_selection_mask(images.shape, ratio, model_index)

        # 2. 混合图像
        augmented = images * mask.float() + (1 - mask.float()) * self._fill_value
        return augmented, targets

    def _generate_selection_mask(self, shape, ratio, model_idx):
        """生成 0-1 掩码张量"""
        if model_idx is not None and self.model_pools_initialized:
            gen = torch.Generator(device=self.device).manual_seed(
                self._model_seeds[model_idx]
            )
            return torch.rand(shape, device=self.device, generator=gen) > ratio

        return torch.rand(shape, device=self.device) > ratio


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
        super().__init__(device)
        self.d_ratio_min = d_ratio_min
        self.d_ratio_max = d_ratio_max
        self._fill_value = fill_value
        self._model_seeds = {}

    @classmethod
    def from_config(cls, device, cfg):
        mean, std = np.mean(cfg.dataset_mean), np.mean(cfg.dataset_std)
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
            self._model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self, images, targets, ratio, prob, model_index=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 GridMask 增强 (主流程)"""
        if random.random() > prob:
            return images, targets

        # 1. 执行网格遮挡逻辑
        augmented = self._apply_grid_patterns(images, ratio, model_index)
        return augmented, targets

    def _apply_grid_patterns(self, images, ratio, model_idx):
        """处理批次图像的网格遮挡"""
        B, C, H, W = images.shape
        augmented = images.clone()

        # 模式判定
        is_model_level = model_idx is not None and self.model_pools_initialized

        if is_model_level:
            # 模型级: 整个 Batch 使用同一个固定网格
            mask = self._get_single_mask(H, W, ratio, self._model_seeds[model_idx])
            mask_v = mask.view(1, 1, H, W)
            augmented = images * mask_v + (1 - mask_v) * self._fill_value
        else:
            # 样本级: 每个样本随机网格
            for b in range(B):
                mask = self._get_single_mask(H, W, ratio, seed=None)
                augmented[b] = images[b] * mask + (1 - mask) * self._fill_value

        return augmented

    def _get_single_mask(self, H, W, ratio, seed=None):
        """生成单张网格 Mask"""
        rng = random.Random(seed) if seed is not None else random

        # 1. 计算网格参数
        min_dim = min(H, W)
        d = rng.randint(
            max(4, int(min_dim * self.d_ratio_min)),
            max(8, int(min_dim * self.d_ratio_max)),
        )
        off_x, off_y = rng.randint(0, d - 1), rng.randint(0, d - 1)
        block_len = int(d * np.sqrt(ratio))

        # 2. 绘制掩码
        mask = torch.ones(H, W, device=self.device)
        for i in range(-1, H // d + 1):
            for j in range(-1, W // d + 1):
                y1, y2 = max(0, i * d + off_y), min(H, i * d + off_y + block_len)
                x1, x2 = max(0, j * d + off_x), min(W, j * d + off_x + block_len)
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = 0
        return mask.unsqueeze(0)  # [1, H, W]


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
        self.height, self.width = height, width
        self.mask_generator = CloudMaskGenerator(
            height,
            width,
            device,
            persistence,
            octaves_large,
            octaves_small,
            scale_ratio,
        )
        self._fill_value = fill_value
        self._masks = []  # 样本级共享池
        self._pool_size = pool_size
        self._model_seeds = {}

    @classmethod
    def from_config(cls, device, cfg):
        mean, std = np.mean(cfg.dataset_mean), np.mean(cfg.dataset_std)
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
        self._masks = self.mask_generator.generate_batch(self._pool_size, target_ratio)

    def init_model_seeds(self, num_models: int, base_seed: int = None):
        """初始化模型级固定 seed"""
        base = base_seed if base_seed is not None else random.randint(0, 1000000)
        for model_idx in range(num_models):
            self._model_seeds[model_idx] = base + model_idx * 10000
        self.model_pools_initialized = True

    def apply(
        self, images, targets, ratio, prob, model_index=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Perlin 噪声增强 (主流程)"""
        if random.random() > prob:
            return images, targets

        # 1. 确定工作模式并执行增强
        augmented = self._apply_perlin_mask(images, ratio, model_index)
        return augmented, targets

    def _apply_perlin_mask(self, images, ratio, model_idx):
        """执行具体的 Perlin 掩码应用"""
        B, C, H, W = images.shape
        augmented = images.clone()

        if model_idx is not None and self.model_pools_initialized:
            # 模型级: 动态生成单个固定 Mask
            mask = self._generate_managed_mask(self._model_seeds[model_idx], ratio)
            mask_v = mask.expand(-1, C, -1, -1).squeeze(0)  # [C, H, W]
            for b in range(B):
                augmented[b] = images[b] * mask_v + (1 - mask_v) * self._fill_value
        else:
            # 样本级: 从池中抽样
            if not self._masks:
                raise RuntimeError("PerlinMask: call precompute_masks() first")
            for b in range(B):
                m = random.choice(self._masks).expand(-1, C, -1, -1).squeeze(0)
                augmented[b] = images[b] * m + (1 - m) * self._fill_value

        return augmented

    def _generate_managed_mask(self, seed: int, ratio: float) -> torch.Tensor:
        """安全地使用特定 seed 生成 Mask 并恢复状态"""
        t_state, n_state, p_state = (
            torch.get_rng_state(),
            np.random.get_state(),
            random.getstate(),
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        mask = self.mask_generator.generate_batch(1, ratio)[0]

        torch.set_rng_state(t_state)
        np.random.set_state(n_state)
        random.setstate(p_state)
        return mask


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
