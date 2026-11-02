from dataclasses import dataclass
import torch
from torch import nn
from src.architectures.base import ConvBnRelu

@dataclass(frozen=True)
class Cnn1dConfig:
    channels: tuple = (32, 64, 128, 256)
    kernel_sizes: tuple = (7, 5, 5, 3)

class Cnn1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        widths = (in_channels,) + config.channels
        self.layers = nn.Sequential(*[
            nn.Sequential(ConvBnRelu(widths[index], widths[index + 1], kernel_size), nn.MaxPool1d(2))
            for index, kernel_size in enumerate(config.kernel_sizes)
        ])
        self.feature_channels = config.channels[-1]

    def forward(self, signals):
        return self.layers(signals)

@dataclass(frozen=True)
class FcnConfig:
    channels: tuple = (128, 256, 128)
    kernel_sizes: tuple = (8, 5, 3)

class Fcn(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        widths = (in_channels,) + config.channels
        self.layers = nn.Sequential(*[ConvBnRelu(widths[index], widths[index + 1], kernel_size) for index, kernel_size in enumerate(config.kernel_sizes)])
        self.feature_channels = config.channels[-1]

    def forward(self, signals):
        return self.layers(signals)

class WangResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        widths = (in_channels,) + (out_channels,) * len(kernel_sizes)
        self.convolutions = nn.Sequential(*[
            ConvBnRelu(widths[index], widths[index + 1], kernel_size, activation=index < len(kernel_sizes) - 1)
            for index, kernel_size in enumerate(kernel_sizes)
        ])
        self.shortcut = ConvBnRelu(in_channels, out_channels, 1, activation=False) if in_channels != out_channels else nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activation(self.convolutions(features) + self.shortcut(features))

@dataclass(frozen=True)
class ResNet1dConfig:
    channels: tuple = (64, 128, 128)
    kernel_sizes: tuple = (8, 5, 3)

class ResNet1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        widths = (in_channels,) + config.channels
        self.blocks = nn.Sequential(*[WangResidualBlock(widths[index], widths[index + 1], config.kernel_sizes) for index in range(len(config.channels))])
        self.feature_channels = config.channels[-1]

    def forward(self, signals):
        return self.blocks(signals)

class XResNetBottleneck(nn.Module):
    def __init__(self, in_channels, width, stride, kernel_size, expansion):
        super().__init__()
        out_channels = width * expansion
        self.convolutions = nn.Sequential(
            ConvBnRelu(in_channels, width, 1),
            ConvBnRelu(width, width, kernel_size, stride=stride),
            ConvBnRelu(width, out_channels, 1, activation=False)
        )
        nn.init.zeros_(self.convolutions[-1][1].weight)
        pool = [nn.AvgPool1d(2, ceil_mode=True)] if stride > 1 else []
        projection = [ConvBnRelu(in_channels, out_channels, 1, activation=False)] if in_channels != out_channels else []
        self.shortcut = nn.Sequential(*pool, *projection)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activation(self.convolutions(features) + self.shortcut(features))

@dataclass(frozen=True)
class XResNet1dConfig:
    stem_channels: tuple = (32, 32, 64)
    stage_depths: tuple = (3, 4, 6, 3)
    stage_widths: tuple = (32, 64, 128, 256)
    kernel_size: int = 5
    expansion: int = 4

class XResNet1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        stem_widths = (in_channels,) + config.stem_channels
        stem_layers = [
            ConvBnRelu(stem_widths[index], stem_widths[index + 1], config.kernel_size, stride=2 if index == 0 else 1)
            for index in range(len(config.stem_channels))
        ]
        self.stem = nn.Sequential(*stem_layers, nn.MaxPool1d(3, stride=2, padding=1))
        stages, channels = [], config.stem_channels[-1]
        for stage, (depth, width) in enumerate(zip(config.stage_depths, config.stage_widths)):
            for block in range(depth):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(XResNetBottleneck(channels, width, stride, config.kernel_size, config.expansion))
                channels = width * config.expansion
        self.stages = nn.Sequential(*stages)
        self.feature_channels = channels

    def forward(self, signals):
        return self.stages(self.stem(signals))

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, features):
        return features * self.gate(features.mean(dim=-1)).unsqueeze(-1)

class SeBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, reduction):
        super().__init__()
        self.convolutions = nn.Sequential(
            ConvBnRelu(in_channels, out_channels, kernel_size, stride=stride),
            ConvBnRelu(out_channels, out_channels, kernel_size, activation=False),
            SqueezeExcitation(out_channels, reduction)
        )
        needs_projection = stride > 1 or in_channels != out_channels
        self.shortcut = ConvBnRelu(in_channels, out_channels, 1, stride=stride, activation=False) if needs_projection else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.activation(self.convolutions(features) + self.shortcut(features))

@dataclass(frozen=True)
class SeResNet1dConfig:
    stem_channels: int = 64
    stem_kernel_size: int = 15
    stage_depths: tuple = (2, 2, 2, 2)
    stage_widths: tuple = (32, 64, 128, 256)
    kernel_size: int = 7
    reduction: int = 16

class SeResNet1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.stem = nn.Sequential(ConvBnRelu(in_channels, config.stem_channels, config.stem_kernel_size, stride=2), nn.MaxPool1d(3, stride=2, padding=1))
        stages, channels = [], config.stem_channels
        for stage, (depth, width) in enumerate(zip(config.stage_depths, config.stage_widths)):
            for block in range(depth):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(SeBasicBlock(channels, width, stride, config.kernel_size, config.reduction))
                channels = width
        self.stages = nn.Sequential(*stages)
        self.feature_channels = channels

    def forward(self, signals):
        return self.stages(self.stem(signals))

class InceptionModule(nn.Module):
    def __init__(self, in_channels, filters, kernel_sizes, bottleneck):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False) if in_channels > 1 else nn.Identity()
        branch_input = bottleneck if in_channels > 1 else in_channels
        self.branches = nn.ModuleList([nn.Conv1d(branch_input, filters, kernel_size, padding="same", bias=False) for kernel_size in kernel_sizes])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, filters, 1, bias=False))
        self.output = nn.Sequential(nn.BatchNorm1d(filters * (len(kernel_sizes) + 1)), nn.ReLU(inplace=True))

    def forward(self, features):
        bottlenecked = self.bottleneck(features)
        return self.output(torch.cat([branch(bottlenecked) for branch in self.branches] + [self.pool_branch(features)], dim=1))

@dataclass(frozen=True)
class Inception1dConfig:
    depth: int = 6
    filters: int = 32
    kernel_sizes: tuple = (39, 19, 9)
    bottleneck: int = 32
    residual_interval: int = 3

class Inception1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        out_channels = config.filters * (len(config.kernel_sizes) + 1)
        self.inception_modules = nn.ModuleList([
            InceptionModule(in_channels if index == 0 else out_channels, config.filters, config.kernel_sizes, config.bottleneck)
            for index in range(config.depth)
        ])
        self.shortcuts = nn.ModuleList([
            ConvBnRelu(in_channels if index == 0 else out_channels, out_channels, 1, activation=False)
            for index in range(config.depth // config.residual_interval)
        ])
        self.residual_interval = config.residual_interval
        self.activation = nn.ReLU(inplace=True)
        self.feature_channels = out_channels

    def forward(self, signals):
        features, residual = signals, signals
        for index, module in enumerate(self.inception_modules):
            features = module(features)
            if (index + 1) % self.residual_interval == 0:
                features = self.activation(features + self.shortcuts[index // self.residual_interval](residual))
                residual = features
        return features