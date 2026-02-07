"""
================================================================================
模型推理模块
================================================================================

包含: get_models_from_source, get_all_models_logits, _FeatureExtractor (内部), _get_all_models_features (内部)
"""

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型提取与推理                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def get_models_from_source(source: Any) -> Tuple[List[nn.Module], torch.device]:
    """多态探测: 从 Trainer, Worker 或 List[Module] 中提取资源"""
    # 1. 解析模型列表
    if hasattr(source, "get_models"):
        models = source.get_models()
    elif hasattr(source, "workers"):
        models = [m for w in source.workers for m in w.models]
    else:
        models = source  # 假定为 List[nn.Module]

    # 2. 解析计算设备
    def _probe_device(objs):
        for o in objs:
            if hasattr(o, "device"):
                return o.device
            p = next(o.parameters(), None)
            if p is not None:
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return models, _probe_device(
        source.workers if hasattr(source, "workers") else models
    )


def get_all_models_logits(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device = None,
    tta_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """全量数据集推理门面 - 支持多 GPU 并行推理和可选 TTA

    Args:
        models: 模型列表（可能分布在不同 GPU 上）
        loader: 数据加载器
        device: 已弃用，保留兼容性（实际使用模型自身的设备）
        tta_config: TTA 配置字典 (可选)，包含:
            - tta_enabled: bool
            - tta_strategy: str ("light", "standard", "heavy", "geospatial")
            - tta_crop_scales: List[float]
            - tta_num_crops: int

    Returns:
        logits: [num_models, num_samples, num_classes]
        targets: [num_samples]
    """
    # 如果启用 TTA，使用 TTA 推理路径
    if tta_config and tta_config.get("tta_enabled", False):
        from .tta import TTAAugmentor, get_all_models_logits_with_tta

        # 获取图像尺寸 (从第一个 batch 推断)
        sample_batch = next(iter(loader))
        image_size = sample_batch[0].shape[-1]  # 假设方形图像

        augmentor = TTAAugmentor.from_config(tta_config, image_size)
        return get_all_models_logits_with_tta(models, loader, augmentor, device)

    # 标准推理路径
    from tqdm import tqdm

    all_l, all_t = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Inference", leave=False):
            # 多 GPU 并行推理
            batch_l = _infer_models_on_batch_multi_gpu(models, x)
            all_l.append(batch_l)
            all_t.append(y)

    if not all_l:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(all_l, dim=1), torch.cat(all_t)


def _infer_models_on_batch_multi_gpu(
    models: List[nn.Module], x: torch.Tensor
) -> torch.Tensor:
    """多 GPU 并行推理 - 每个模型在其所在设备上推理

    Args:
        models: 模型列表（可能在不同 GPU 上）
        x: 输入张量 (CPU)

    Returns:
        [num_models, batch_size, num_classes] 张量 (CPU)
    """
    batch_res = []
    for m in models:
        m.eval()
        # 获取模型所在设备
        model_device = next(m.parameters()).device
        # 将数据移到模型所在设备
        x_dev = x.to(model_device)
        # 推理并移回 CPU
        out = m(x_dev).unsqueeze(0).cpu()
        batch_res.append(out)
    return torch.cat(batch_res, dim=0)  # [num_models, batch_size, num_classes]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 特征提取 (用于 CKA 计算)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class _FeatureExtractor:
    """通用特征提取器 - 使用 Hook 机制，支持任意模型架构

    自动定位模型倒数第二层（分类器前的特征层），提取隐藏层表示。
    支持 ResNet、VGG、EfficientNet、MobileNet 等常见架构。

    Args:
        model: 目标模型
        layer_name: 可选，手动指定要提取特征的层名（如 'avgpool', 'features.28'）
    """

    # 支持的池化层类型（按优先级排序）
    _POOL_TYPES = (
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveMaxPool2d,
        nn.AvgPool2d,
        nn.MaxPool2d,
    )

    # 应该跳过的层类型（不适合作为特征层）
    _SKIP_TYPES = (
        nn.Linear,
        nn.Dropout,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.LeakyReLU,
        nn.GELU,
        nn.Sigmoid,
        nn.Softmax,
        nn.Flatten,
    )

    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        self.model = model
        self.layer_name = layer_name
        self.features: Optional[torch.Tensor] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hook()

    def _find_penultimate_layer(self) -> nn.Module:
        """自动定位倒数第二层 (分类器前的特征层)

        策略优先级：
        1. 用户手动指定的层名
        2. 池化层（AdaptiveAvgPool2d > AdaptiveMaxPool2d > AvgPool2d > MaxPool2d）
        3. 最后一个卷积层（Conv2d）
        4. 最后一个有意义的非分类器模块
        """
        # 策略 1: 用户手动指定
        if self.layer_name is not None:
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    return module
            raise ValueError(f"未找到指定的层: {self.layer_name}")

        last_pool = None
        last_conv = None
        last_valid = None

        for name, module in self.model.named_modules():
            # 策略 2: 查找池化层
            if isinstance(module, self._POOL_TYPES):
                last_pool = module

            # 策略 3: 查找卷积层
            if isinstance(module, nn.Conv2d):
                last_conv = module

            # 策略 4: 查找任意有效层（排除分类器相关层）
            if not isinstance(module, self._SKIP_TYPES) and name != "":
                # 排除顶层容器
                if not isinstance(
                    module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)
                ):
                    last_valid = module

        # 按优先级返回
        if last_pool is not None:
            return last_pool
        if last_conv is not None:
            return last_conv
        return last_valid

    def _register_hook(self):
        """注册前向 Hook"""
        target = self._find_penultimate_layer()
        if target is None:
            raise RuntimeError("无法定位特征提取层")

        def hook_fn(module, input, output):
            # 确保输出是扁平化的 [B, D]
            if output.dim() > 2:
                self.features = output.flatten(1)
            else:
                self.features = output

        self._hook = target.register_forward_hook(hook_fn)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播并返回中间特征"""
        _ = self.model(x)  # 触发 hook
        return self.features

    def remove_hook(self):
        """移除 Hook，避免内存泄漏"""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def _get_all_models_features(
    models: List[nn.Module], loader: DataLoader, device: torch.device
) -> torch.Tensor:
    """提取集成模型所有子模型的隐藏层特征

    Args:
        models: 模型列表
        loader: 数据加载器
        device: 计算设备

    Returns:
        all_features: [num_models, num_samples, feature_dim]
    """
    from tqdm import tqdm

    extractors = [_FeatureExtractor(m) for m in models]
    all_features = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Feature Extraction", leave=False):
            batch_feats = []
            for ext in extractors:
                ext.model.eval()
                feats = ext.extract(x.to(device))
                batch_feats.append(feats.cpu().unsqueeze(0))
            all_features.append(torch.cat(batch_feats, dim=0))

    # 清理 hooks
    for ext in extractors:
        ext.remove_hook()

    if not all_features:
        return torch.tensor([])

    return torch.cat(all_features, dim=1)  # [num_models, num_samples, feature_dim]
