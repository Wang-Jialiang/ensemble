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

# 初始化策略 (从 init.py 直接导入)
# 模型工厂 (从 factory.py 导入)
from .factory import (
    MODEL_REGISTRY,
    ModelFactory,
    register_model,
)
from .init import (
    INIT_REGISTRY,
    apply_init,
    get_supported_inits,
    init_kaiming,
    init_orthogonal,
    init_xavier,
    register_init,
)

# ResNet 架构
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
    "apply_init",
    "get_supported_inits",
    # Registry & Factory
    "MODEL_REGISTRY",
    "register_model",
    "ModelFactory",
]
