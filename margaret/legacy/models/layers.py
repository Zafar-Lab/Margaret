import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    """Extends the standard pytorch convolutional layer to use weight standardization"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        # Apply weight standardization
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class SeparableConvolution(nn.Module):
    """Implements a Depthwise separable convolution module"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        dilation=1,
        norm="bn",
    ):
        super(SeparableConvolution, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=bias,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        x = self.squeeze(x)
        x = self.excite(x)
        return identity * x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        grid_factor=1,
    ):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            3,
            stride=stride,
            padding=dilation * grid_factor,
            dilation=dilation * grid_factor,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
