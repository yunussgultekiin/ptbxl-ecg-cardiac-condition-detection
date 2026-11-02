from dataclasses import dataclass
from torch import nn
from src.architectures.base import ConvBnRelu

@dataclass(frozen=True)
class CnnBiLstmConfig:
    channels: tuple = (64, 128, 128)
    kernel_sizes: tuple = (7, 5, 3)
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.2

class CnnBiLstm(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        widths = (in_channels,) + config.channels
        self.encoder = nn.Sequential(*[
            nn.Sequential(ConvBnRelu(widths[index], widths[index + 1], kernel_size), nn.MaxPool1d(2))
            for index, kernel_size in enumerate(config.kernel_sizes)
        ])
        self.lstm = nn.LSTM(config.channels[-1], config.hidden_size, num_layers=config.layers, dropout=config.dropout, batch_first=True, bidirectional=True)
        self.feature_channels = 2 * config.hidden_size

    def forward(self, signals):
        return self.lstm(self.encoder(signals).transpose(1, 2))[0].transpose(1, 2)