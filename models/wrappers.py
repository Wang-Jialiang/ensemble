"""
================================================================================
外部模型包装模块
================================================================================

包含: VGG16Wrapper, EfficientNetB0Wrapper
为 torchvision 模型添加与自定义 ResNet 兼容的接口
"""

import torch.nn as nn
from torchvision.models import vgg16, efficientnet_b0

from .init import apply_init


class VGG16Wrapper(nn.Module):
    """VGG16 包装类 - 添加与 ResNet 兼容的接口"""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.model = vgg16(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def get_backbone_state_dict(self) -> dict:
        """获取 backbone 权重 (不含分类器层)

        VGG16 的分类器是 model.classifier
        """
        return {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("model.classifier")
        }

    def reinit_classifier(self, init_method: str = "kaiming") -> None:
        """重新初始化分类器"""
        apply_init(self.model.classifier, init_method)


class EfficientNetB0Wrapper(nn.Module):
    """EfficientNet-B0 包装类 - 添加与 ResNet 兼容的接口"""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.model = efficientnet_b0(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def get_backbone_state_dict(self) -> dict:
        """获取 backbone 权重 (不含分类器层)

        EfficientNet 的分类器是 model.classifier
        """
        return {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("model.classifier")
        }

    def reinit_classifier(self, init_method: str = "kaiming") -> None:
        """重新初始化分类器"""
        apply_init(self.model.classifier, init_method)
