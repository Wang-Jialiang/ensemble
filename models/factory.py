"""
================================================================================
模型工厂模块
================================================================================

包含: 初始化策略、模型注册表、ModelFactory
"""

from typing import Callable, Dict, List, Optional

import torch.nn as nn

from .resnet import BasicBlock, Bottleneck, ResNet

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 初始化策略                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

INIT_REGISTRY: Dict[str, Callable[[nn.Module], None]] = {}


def register_init(name: str):
    """初始化策略注册装饰器"""

    def decorator(fn):
        INIT_REGISTRY[name] = fn
        return fn

    return decorator


@register_init("kaiming")
def init_kaiming(model: nn.Module) -> None:
    """Kaiming (He) 初始化 - 适合ReLU激活"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


@register_init("xavier")
def init_xavier(model: nn.Module) -> None:
    """Xavier (Glorot) 初始化 - 适合tanh/sigmoid激活"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


@register_init("orthogonal")
def init_orthogonal(model: nn.Module) -> None:
    """正交初始化 - 保持梯度范数稳定"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


@register_init("default")
def init_default(model: nn.Module) -> None:
    """PyTorch 默认初始化 (不做任何修改)"""
    pass


def apply_init(model: nn.Module, init_method: str = "kaiming") -> nn.Module:
    """
    应用初始化策略

    Args:
        model: PyTorch模型
        init_method: 初始化方法名称 (kaiming, xavier, orthogonal, default)

    Returns:
        初始化后的模型
    """
    init_method = init_method.lower()
    if init_method not in INIT_REGISTRY:
        raise ValueError(
            f"不支持的初始化方法: {init_method}. 支持: {list(INIT_REGISTRY.keys())}"
        )
    INIT_REGISTRY[init_method](model)
    return model


def get_supported_inits() -> List[str]:
    """获取支持的初始化方法列表"""
    return list(INIT_REGISTRY.keys())


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
    from torchvision.models import vgg16 as tv_vgg16

    return tv_vgg16(num_classes=num_classes)


@register_model("efficientnet_b0")
def efficientnet_b0(num_classes: int):
    from torchvision.models import efficientnet_b0 as tv_effnet

    return tv_effnet(num_classes=num_classes)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型工厂                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int = 10,
        init_method: Optional[str] = None,
        **kwargs,
    ) -> nn.Module:
        """
        创建模型

        参数:
            model_name: 模型名称 (resnet18, resnet34, resnet50, vgg16, efficientnet_b0)
            num_classes: 类别数
            init_method: 初始化方法 (kaiming, xavier, orthogonal, default, None)
                         None 表示使用模型内置初始化
            **kwargs: 其他传递给模型构建器的参数

        返回:
            model: 创建的模型
        """
        model_name = model_name.lower()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"不支持的模型: {model_name}. 支持: {list(MODEL_REGISTRY.keys())}"
            )

        builder = MODEL_REGISTRY[model_name]
        model = builder(num_classes, **kwargs)

        # 应用自定义初始化 (如果指定)
        if init_method is not None:
            apply_init(model, init_method)

        return model

    @staticmethod
    def get_supported_models() -> List[str]:
        """获取支持的模型列表"""
        return list(MODEL_REGISTRY.keys())
