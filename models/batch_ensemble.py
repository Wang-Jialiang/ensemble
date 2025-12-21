"""
================================================================================
Batch Ensemble 模型模块
================================================================================

Batch Ensemble 实现 (Wen et al., 2020)

核心思想: 共享主干网络权重，每个集成成员只维护独立的 rank-1 乘性因子。
- 计算效率: 参数量仅增加 ~1-2%，但获得 ensemble 效果
- 训练方式: 输入复制 M 份，每份用不同的 rank-1 因子变换

参考: https://arxiv.org/abs/2002.06715
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .factory import MODEL_REGISTRY

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Batch Ensemble 基础层                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class BatchEnsembleLinear(nn.Module):
    """
    Batch Ensemble 线性层

    每个成员 i 的输出: y_i = (W ⊙ (r_i ⊗ s_i)) @ x + b
    其中 r_i, s_i 是 rank-1 因子
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_members: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_members = num_members

        # 共享权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # 每个成员的 rank-1 因子
        self.r = nn.Parameter(torch.empty(num_members, in_features))  # 输入调制
        self.s = nn.Parameter(torch.empty(num_members, out_features))  # 输出调制

        if bias:
            self.bias = nn.Parameter(torch.empty(num_members, out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        # 共享权重使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # rank-1 因子初始化为 1 (初始时等效于标准网络)
        nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size * num_members, in_features]
               输入应该已经被复制 num_members 次

        Returns:
            [batch_size * num_members, out_features]
        """
        batch_size = x.size(0) // self.num_members

        # 重塑为 [num_members, batch_size, in_features]
        x = x.view(self.num_members, batch_size, self.in_features)

        # 应用输入调制 r: [num_members, 1, in_features] * [num_members, batch_size, in_features]
        x_scaled = x * self.r.unsqueeze(1)

        # 线性变换: [num_members, batch_size, out_features]
        out = torch.einsum("mbi,oi->mbo", x_scaled, self.weight)

        # 应用输出调制 s
        out = out * self.s.unsqueeze(1)

        # 添加 bias
        if self.bias is not None:
            out = out + self.bias.unsqueeze(1)

        # 重塑回 [batch_size * num_members, out_features]
        return out.view(-1, self.out_features)


class BatchEnsembleConv2d(nn.Module):
    """
    Batch Ensemble 卷积层

    每个成员 i 的卷积核: W_i = W ⊙ (r_i ⊗ s_i)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_members: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_members = num_members
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 共享卷积权重
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        # rank-1 因子
        self.r = nn.Parameter(torch.empty(num_members, in_channels))
        self.s = nn.Parameter(torch.empty(num_members, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(num_members, out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size * num_members, C, H, W]

        Returns:
            [batch_size * num_members, C_out, H_out, W_out]
        """
        batch_size = x.size(0) // self.num_members

        # 重塑: [num_members, batch_size, C, H, W]
        x = x.view(self.num_members, batch_size, self.in_channels, x.size(2), x.size(3))

        # 输入调制: 逐通道缩放
        r_scaled = self.r.view(self.num_members, 1, self.in_channels, 1, 1)
        x = x * r_scaled

        # 合并回 [N*M, C, H, W] 进行卷积
        x = x.view(-1, self.in_channels, x.size(3), x.size(4))

        # 标准卷积
        out = F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # 输出调制
        out_h, out_w = out.size(2), out.size(3)
        out = out.view(self.num_members, batch_size, self.out_channels, out_h, out_w)
        s_scaled = self.s.view(self.num_members, 1, self.out_channels, 1, 1)
        out = out * s_scaled

        # 添加 bias
        if self.bias is not None:
            out = out + self.bias.view(self.num_members, 1, self.out_channels, 1, 1)

        return out.view(-1, self.out_channels, out_h, out_w)


class BatchEnsembleBatchNorm2d(nn.Module):
    """
    Batch Ensemble 专用 BatchNorm

    每个成员共享统计量，但有独立的 affine 参数
    """

    def __init__(
        self,
        num_features: int,
        num_members: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_members = num_members
        self.eps = eps
        self.momentum = momentum

        # 共享 running stats
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # 每个成员独立的 affine 参数
        self.weight = nn.Parameter(torch.ones(num_members, num_features))
        self.bias = nn.Parameter(torch.zeros(num_members, num_features))

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0) // self.num_members
        _, C, H, W = x.shape

        if self.training:
            # 计算全局统计量
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            # 更新 running stats
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_norm = (x - mean.view(1, C, 1, 1)) / torch.sqrt(
            var.view(1, C, 1, 1) + self.eps
        )

        # 每个成员独立的 affine
        x_norm = x_norm.view(self.num_members, batch_size, C, H, W)
        out = x_norm * self.weight.view(self.num_members, 1, C, 1, 1)
        out = out + self.bias.view(self.num_members, 1, C, 1, 1)

        return out.view(-1, C, H, W)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ Batch Ensemble ResNet 构建模块                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


class BatchEnsembleBasicBlock(nn.Module):
    """Batch Ensemble 版本 BasicBlock"""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        num_members: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = BatchEnsembleConv2d(
            inplanes, planes, 3, num_members, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchEnsembleBatchNorm2d(planes, num_members)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BatchEnsembleConv2d(
            planes, planes, 3, num_members, stride=1, padding=1, bias=False
        )
        self.bn2 = BatchEnsembleBatchNorm2d(planes, num_members)
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


class BatchEnsembleResNet(nn.Module):
    """
    Batch Ensemble 版本 ResNet (CIFAR 适配)

    与标准 ResNet 不同:
    - 所有 Conv2d/Linear 替换为 BatchEnsemble 版本
    - 输入会被复制 num_members 次
    - 输出是 [batch_size * num_members, num_classes]
    """

    def __init__(
        self,
        layers: List[int],
        num_classes: int = 10,
        num_members: int = 4,
    ):
        super().__init__()
        self.num_members = num_members
        self.inplanes = 64

        # 第一层卷积 (CIFAR 适配: 3x3, 无 maxpool)
        self.conv1 = BatchEnsembleConv2d(
            3, 64, 3, num_members, stride=1, padding=1, bias=False
        )
        self.bn1 = BatchEnsembleBatchNorm2d(64, num_members)
        self.relu = nn.ReLU(inplace=True)

        # ResNet 主体
        self.layer1 = self._make_layer(64, layers[0], num_members, stride=1)
        self.layer2 = self._make_layer(128, layers[1], num_members, stride=2)
        self.layer3 = self._make_layer(256, layers[2], num_members, stride=2)
        self.layer4 = self._make_layer(512, layers[3], num_members, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BatchEnsembleLinear(512, num_classes, num_members)

    def _make_layer(
        self, planes: int, blocks: int, num_members: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                BatchEnsembleConv2d(
                    self.inplanes, planes, 1, num_members, stride=stride, bias=False
                ),
                BatchEnsembleBatchNorm2d(planes, num_members),
            )

        layers = []
        layers.append(
            BatchEnsembleBasicBlock(
                self.inplanes, planes, num_members, stride, downsample
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BatchEnsembleBasicBlock(self.inplanes, planes, num_members))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch_size, 3, H, W] - 原始输入

        Returns:
            [num_members, batch_size, num_classes] - 每个成员的 logits
        """
        batch_size = x.size(0)

        # 复制输入 num_members 次: [num_members * batch_size, 3, H, W]
        x = x.repeat(self.num_members, 1, 1, 1)

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

        # 重塑为 [num_members, batch_size, num_classes]
        return x.view(self.num_members, batch_size, -1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ 模型注册                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


def _register_batch_ensemble_models():
    """注册 Batch Ensemble 模型到全局 MODEL_REGISTRY"""

    def batch_resnet18(num_classes: int, num_members: int = 4):
        return BatchEnsembleResNet([2, 2, 2, 2], num_classes, num_members)

    def batch_resnet34(num_classes: int, num_members: int = 4):
        return BatchEnsembleResNet([3, 4, 6, 3], num_classes, num_members)

    # 注册到 MODEL_REGISTRY
    MODEL_REGISTRY["batch_resnet18"] = {
        "builder": batch_resnet18,
        "supports_cifar": True,
    }
    MODEL_REGISTRY["batch_resnet34"] = {
        "builder": batch_resnet34,
        "supports_cifar": True,
    }


# 模块加载时自动注册
_register_batch_ensemble_models()


__all__ = [
    "BatchEnsembleLinear",
    "BatchEnsembleConv2d",
    "BatchEnsembleBatchNorm2d",
    "BatchEnsembleBasicBlock",
    "BatchEnsembleResNet",
]
