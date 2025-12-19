"""
================================================================================
模型模块
================================================================================
"""

from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ResNet基础模块                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3卷积"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """自定义ResNet (适配CIFAR)"""

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # CIFAR适配: 3x3 conv, 无maxpool
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型注册表                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

MODEL_REGISTRY = {}


def register_model(name: str, supports_cifar: bool = False):
    """
    模型注册装饰器

    使用方式:
    @register_model('resnet18', supports_cifar=True)
    def resnet18(num_classes, pretrained):
        ...
    """

    def decorator(builder_fn):
        MODEL_REGISTRY[name] = {"builder": builder_fn, "supports_cifar": supports_cifar}
        return builder_fn

    return decorator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 具体模型定义                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@register_model("resnet18", supports_cifar=True)
def resnet18(num_classes: int):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


@register_model("resnet34", supports_cifar=True)
def resnet34(num_classes: int):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


@register_model("resnet50", supports_cifar=True)
def resnet50(num_classes: int):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


@register_model("vgg16", supports_cifar=False)
def vgg16(num_classes: int):
    from torchvision.models import vgg16 as tv_vgg16

    return tv_vgg16(num_classes=num_classes)


@register_model("efficientnet_b0", supports_cifar=False)
def efficientnet_b0(num_classes: int):
    from torchvision.models import efficientnet_b0 as tv_effnet

    return tv_effnet(num_classes=num_classes)


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
        """
        创建模型

        参数:
            model_name: 模型名称 (resnet18, resnet34, resnet50, vgg16, efficientnet_b0)
            num_classes: 类别数
            **kwargs: 其他传递给模型构建器的参数

        返回:
            model: 创建的模型
        """
        model_name = model_name.lower()

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"不支持的模型: {model_name}. 支持: {list(MODEL_REGISTRY.keys())}"
            )

        config = MODEL_REGISTRY[model_name]
        return config["builder"](num_classes, **kwargs)

    @staticmethod
    def get_supported_models() -> List[str]:
        """获取支持的模型列表"""
        return list(MODEL_REGISTRY.keys())

    @staticmethod
    def check_compatibility(model_name: str, dataset_name: str) -> List[str]:
        """
        检查模型与数据集的兼容性

        Args:
            model_name: 模型名称
            dataset_name: 数据集名称

        Returns:
            warnings: 警告信息列表
        """
        warnings = []
        model_name = model_name.lower()
        dataset_name = dataset_name.lower()

        if model_name not in MODEL_REGISTRY:
            return warnings  # 让create_model处理未知模型错误

        model_info = MODEL_REGISTRY[model_name]
        is_small_image = "cifar" in dataset_name or "eurosat" in dataset_name

        # 检查 supports_cifar 标记
        if is_small_image and not model_info["supports_cifar"]:
            warnings.append(
                f"⚠️ Model '{model_name}' (supports_cifar=False) may not work well with "
                f"small image dataset '{dataset_name}' (32x32 or 64x64). "
                f"Consider using a ResNet model."
            )

        return warnings
