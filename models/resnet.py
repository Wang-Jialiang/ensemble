"""
================================================================================
ResNet 架构模块
================================================================================

包含: BasicBlock, Bottleneck, ResNet (CIFAR适配版)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        norm = norm_layer or nn.BatchNorm2d
        self.conv1, self.bn1 = conv3x3(inplanes, planes, stride), norm(planes)
        self.conv2, self.bn2 = conv3x3(planes, planes), norm(planes)
        self.relu, self.downsample = nn.ReLU(inplace=True), downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm = norm_layer or nn.BatchNorm2d
        w = int(planes * (base_width / 64.0)) * groups
        self.conv1, self.bn1 = conv1x1(inplanes, w), norm(w)
        self.conv2, self.bn2 = conv3x3(w, w, stride, groups, dilation), norm(w)
        self.conv3, self.bn3 = conv1x1(w, planes * self.expansion), norm(planes * self.expansion)
        self.relu, self.downsample = nn.ReLU(inplace=True), downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    """自定义ResNet (适配CIFAR)"""

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        """ResNet 构造函数 (大纲化)"""
        super().__init__()
        self._norm_layer = norm_layer or nn.BatchNorm2d
        self.inplanes, self.dilation = 64, 1
        self.groups, self.base_width = groups, width_per_group

        # 1. 构建 Backbone (适配 CIFAR: 3x3 stem, 无 maxpool)
        self._build_backbone(block, layers, replace_stride_with_dilation)

        # 2. 构建分类器
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 3. 参数初始化
        self._init_params(zero_init_residual)

    def _build_backbone(self, block, layers, replace_stride_with_dilation):
        """纵向搭建特征提取网络"""
        res = replace_stride_with_dilation or [False, False, False]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=res[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=res[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=res[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _init_params(self, zero_init_residual):
        """应用默认 Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            self._zero_init_residual_weights()

    def _zero_init_residual_weights(self):
        """Residual 路径权值归零 (针对最后一层 BN)"""
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

    def get_backbone_state_dict(self) -> dict:
        """获取 backbone 权重 (不含 fc 层)

        Returns:
            dict: 不包含 'fc.' 开头键的 state_dict
        """
        return {k: v for k, v in self.state_dict().items() if not k.startswith("fc.")}

    def reinit_classifier(self, init_method: str = "kaiming") -> None:
        """重新初始化分类器 (fc 层)"""
        from .factory import apply_init
        apply_init(self.fc, init_method)
