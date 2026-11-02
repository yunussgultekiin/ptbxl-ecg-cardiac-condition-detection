from dataclasses import dataclass
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, model_dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = model_dim // heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        batch, length, model_dim = tokens.shape
        query, key, value = self.qkv(tokens).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        weights = self.dropout(torch.softmax(query @ key.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1))
        return self.output((weights @ value).transpose(1, 2).reshape(batch, length, model_dim))

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, heads, feedforward_dim, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = SelfAttention(model_dim, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        tokens = tokens + self.dropout(self.attention(self.attention_norm(tokens)))
        return tokens + self.dropout(self.feedforward(self.feedforward_norm(tokens)))

@dataclass(frozen=True)
class Transformer1dConfig:
    patch_size: int = 10
    model_dim: int = 128
    heads: int = 4
    layers: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    signal_length: int = 1000

class Transformer1d(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.patch_embedding = nn.Conv1d(in_channels, config.model_dim, config.patch_size, stride=config.patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.signal_length // config.patch_size, config.model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(config.model_dim, config.heads, config.feedforward_dim, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.model_dim)
        self.feature_channels = config.model_dim

    def forward(self, signals):
        tokens = self.patch_embedding(signals).transpose(1, 2) + self.position_embedding
        return self.norm(self.blocks(tokens)).transpose(1, 2)