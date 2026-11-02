import math
from dataclasses import dataclass
import torch
from torch import nn

class S4DKernel(nn.Module):
    def __init__(self, model_dim, state_dim, dt_min, dt_max):
        super().__init__()
        log_dt = torch.rand(model_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.c = nn.Parameter(torch.randn(model_dim, state_dim // 2, 2) * 0.5 ** 0.5)
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(model_dim, state_dim // 2)))
        self.a_imaginary = nn.Parameter(math.pi * torch.arange(state_dim // 2, dtype=torch.float32).repeat(model_dim, 1))

    def forward(self, length):
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        a = torch.complex(-torch.exp(self.log_a_real), self.a_imaginary)
        dt_a = a * dt
        c = torch.view_as_complex(self.c) * (torch.exp(dt_a) - 1.0) / a
        steps = torch.arange(length, device=dt.device, dtype=torch.float32)
        return 2 * torch.einsum("hn,hnl->hl", c, torch.exp(dt_a.unsqueeze(-1) * steps)).real

class S4DLayer(nn.Module):
    def __init__(self, model_dim, state_dim, dropout, dt_min, dt_max):
        super().__init__()
        self.kernel = S4DKernel(model_dim, state_dim, dt_min, dt_max)
        self.skip = nn.Parameter(torch.randn(model_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Conv1d(model_dim, 2 * model_dim, 1), nn.GLU(dim=1))

    def forward(self, features):
        length = features.size(-1)
        kernel_frequency = torch.fft.rfft(self.kernel(length), n=2 * length)
        signal_frequency = torch.fft.rfft(features, n=2 * length)
        convolved = torch.fft.irfft(signal_frequency * kernel_frequency, n=2 * length)[..., :length]
        return self.output(self.dropout(self.activation(convolved + features * self.skip.unsqueeze(-1))))

class S4DBlock(nn.Module):
    def __init__(self, model_dim, state_dim, dropout, dt_min, dt_max):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.layer = S4DLayer(model_dim, state_dim, dropout, dt_min, dt_max)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        normalized = self.norm(features.transpose(1, 2)).transpose(1, 2)
        return features + self.dropout(self.layer(normalized))

@dataclass(frozen=True)
class S4DConfig:
    model_dim: int = 256
    state_dim: int = 8
    layers: int = 4
    dropout: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1

class S4D(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.encoder = nn.Conv1d(in_channels, config.model_dim, 1)
        self.blocks = nn.Sequential(*[S4DBlock(config.model_dim, config.state_dim, config.dropout, config.dt_min, config.dt_max) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.model_dim)
        self.feature_channels = config.model_dim

    def forward(self, signals):
        features = self.blocks(self.encoder(signals))
        return self.norm(features.transpose(1, 2)).transpose(1, 2)