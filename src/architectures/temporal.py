from dataclasses import dataclass
from torch import nn
from torch.nn import functional
from torch.nn.utils.parametrizations import weight_norm

INIT_STD = 0.01

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        convolution = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.normal_(convolution.weight, 0.0, INIT_STD)
        self.convolution = weight_norm(convolution)

    def forward(self, features):
        batch, channels, length = features.shape
        padded = functional.pad(features, (self.padding, -(length + self.padding) % self.dilation))
        phases = padded.view(batch, channels, -1, self.dilation).permute(0, 3, 1, 2).reshape(batch * self.dilation, channels, -1)
        outputs = self.convolution(phases)
        outputs = outputs.view(batch, self.dilation, outputs.size(1), -1).permute(0, 2, 3, 1)
        return outputs.reshape(batch, outputs.size(1), -1)[..., :length]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if isinstance(self.downsample, nn.Conv1d):
            nn.init.normal_(self.downsample.weight, 0.0, INIT_STD)
        self.activation = nn.ReLU()

    def forward(self, features):
        return self.activation(self.net(features) + self.downsample(features))

@dataclass(frozen=True)
class TcnConfig:
    channels: tuple = (48,) * 7
    kernel_size: int = 7
    dropout: float = 0.05

class Tcn(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        widths = (in_channels,) + config.channels
        self.blocks = nn.Sequential(*[
            TemporalBlock(widths[level], widths[level + 1], config.kernel_size, 2 ** level, config.dropout)
            for level in range(len(config.channels))
        ])
        self.feature_channels = config.channels[-1]

    def forward(self, signals):
        return self.blocks(signals)
