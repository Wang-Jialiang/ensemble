"""
================================================================================
模型工厂模块
================================================================================

包含: 模型注册表、ModelFactory
"""

from typing import List, Optional

import torch.nn as nn

from .init import apply_init  # 只导入 factory 实际使用的
from .resnet import BasicBlock, Bottleneck, ResNet

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型注册表                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODEL_REGISTRY = {}


def register_model(name: str):
    """模型注册装饰器。

    将被装饰的函数注册到 MODEL_REGISTRY，便于通过名称创建模型。

    Args:
        name: 注册的模型名称。

    Returns:
        Callable: 装饰器函数。

    Example:
        >>> @register_model('resnet18')
        ... def resnet18(num_classes):
        ...     return ResNet(...)
    """

    def decorator(builder_fn):
        MODEL_REGISTRY[name] = builder_fn
        return builder_fn

    return decorator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 具体模型定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@register_model("resnet18")
def resnet18(num_classes: int):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


@register_model("resnet34")
def resnet34(num_classes: int):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


@register_model("resnet50")
def resnet50(num_classes: int):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


@register_model("vgg16")
def vgg16(num_classes: int):
    from .wrappers import VGG16Wrapper

    return VGG16Wrapper(num_classes=num_classes)


@register_model("efficientnet_b0")
def efficientnet_b0(num_classes: int):
    from .wrappers import EfficientNetB0Wrapper

    return EfficientNetB0Wrapper(num_classes=num_classes)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型工厂                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ModelFactory:
    """模型生产工厂。

    提供统一的模型创建接口，支持通过名称创建已注册的模型。
    """

    @staticmethod
    def create_model(
        name: str, num_classes: int = 10, init: Optional[str] = None
    ) -> nn.Module:
        """创建并初始化模型。

        Args:
            name: 模型名称，如 'resnet18', 'vgg16' 等。
            num_classes: 分类类别数，默认为 10。
            init: 初始化方法，如 'kaiming', 'xavier'，可选。

        Returns:
            nn.Module: 初始化后的模型实例。

        Raises:
            ValueError: 如果模型名称未注册。
        """
        name = name.lower()

        # 1. 解析构建器
        if name not in MODEL_REGISTRY:
            raise ValueError(f"模型未注册: {name}. 支持: {list(MODEL_REGISTRY.keys())}")

        # 2. 实例化模型
        model = MODEL_REGISTRY[name](num_classes)

        # 3. 策略初始化 (可选)
        if init:
            apply_init(model, init)

        return model

    @staticmethod
    def get_supported_models() -> List[str]:
        """获取所有已注册的模型名称列表。

        Returns:
            List[str]: 支持的模型名称列表。
        """
        return list(MODEL_REGISTRY.keys())
