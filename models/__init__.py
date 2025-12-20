"""
================================================================================
模型模块 (Models Package)
================================================================================

模块化模型包，包含：
- resnet: ResNet 架构 (BasicBlock, Bottleneck, ResNet - CIFAR适配)
- factory: 初始化策略、模型注册表、ModelFactory

使用方式:
    from ensemble.models import ModelFactory, ResNet
    from ensemble.models import apply_init, INIT_REGISTRY
"""

# ResNet 架构
# 工厂与初始化
from .factory import (
    INIT_REGISTRY,
    MODEL_REGISTRY,
    ModelFactory,
    apply_init,
    get_supported_inits,
    init_default,
    init_kaiming,
    init_orthogonal,
    init_xavier,
    register_init,
    register_model,
)
from .resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    conv1x1,
    conv3x3,
)

__all__ = [
    # ResNet
    "conv3x3",
    "conv1x1",
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    # Init
    "INIT_REGISTRY",
    "register_init",
    "init_kaiming",
    "init_xavier",
    "init_orthogonal",
    "init_default",
    "apply_init",
    "get_supported_inits",
    # Registry & Factory
    "MODEL_REGISTRY",
    "register_model",
    "ModelFactory",
]
