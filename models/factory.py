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
    """
    模型注册装饰器

    使用方式:
    @register_model('resnet18')
    def resnet18(num_classes):
        ...
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
    """模型生产工厂 (大纲化)"""

    @staticmethod
    def create_model(
        name: str, num_classes: int = 10, init: Optional[str] = None
    ) -> nn.Module:
        """核心生产线: 路由 -> 实例化 -> 初始化"""
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
        return list(MODEL_REGISTRY.keys())
