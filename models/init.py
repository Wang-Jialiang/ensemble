"""
================================================================================
模型初始化策略模块
================================================================================

包含: 初始化注册表、Kaiming/Xavier/Orthogonal/Default 初始化
"""

from typing import Callable, Dict, List

import torch.nn as nn

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 初始化注册表                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

INIT_REGISTRY: Dict[str, Callable[[nn.Module], None]] = {}


def register_init(name: str):
    """初始化策略注册装饰器"""

    def decorator(fn):
        INIT_REGISTRY[name] = fn
        return fn

    return decorator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 具体初始化策略                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@register_init("kaiming")
def init_kaiming(model: nn.Module) -> None:
    """Kaiming (He) 初始化 - 适用于 ReLU 激活函数"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


@register_init("xavier")
def init_xavier(model: nn.Module) -> None:
    """Xavier (Glorot) 初始化 - 适用于 Sigmoid/Tanh 激活函数"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
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
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 公开 API                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def apply_init(model: nn.Module, method: str = "kaiming") -> nn.Module:
    """解析并应用初始化策略"""
    method = method.lower()
    if method not in INIT_REGISTRY:
        raise ValueError(
            f"不支持的初始化: {method}. 已注册: {list(INIT_REGISTRY.keys())}"
        )

    INIT_REGISTRY[method](model)
    return model


def get_supported_inits() -> List[str]:
    """获取支持的初始化方法列表"""
    return list(INIT_REGISTRY.keys())
