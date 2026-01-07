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
    models: List[nn.Module], loader: DataLoader, device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """全量数据集推理门面 - 支持多 GPU 并行推理

    Args:
        models: 模型列表（可能分布在不同 GPU 上）
        loader: 数据加载器
        device: 已弃用，保留兼容性（实际使用模型自身的设备）
    """
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


def _infer_models_on_batch(models, x):
    """单批次多模型推理核心 (保留兼容性)"""
    return _infer_models_on_batch_multi_gpu(models, x.cpu())


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 特征提取 (用于 CKA 计算)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class _FeatureExtractor:
    """通用特征提取器 - 使用 Hook 机制，支持任意模型架构

    自动定位模型倒数第二层（分类器前的特征层），提取隐藏层表示。
    支持 ResNet、VGG、EfficientNet 等常见架构。
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.features: Optional[torch.Tensor] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._register_hook()

    def _find_penultimate_layer(self) -> nn.Module:
        """自动定位倒数第二层 (分类器前的特征层)

        策略：
        1. 优先查找 AdaptiveAvgPool2d (ResNet/EfficientNet 风格)
        2. 若未找到，回退到最后一个非 Linear 模块
        """
        last_pool = None
        last_non_linear = None

        for name, module in self.model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                last_pool = module
            if not isinstance(module, (nn.Linear, nn.Dropout)):
                last_non_linear = module

        return last_pool if last_pool is not None else last_non_linear

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
