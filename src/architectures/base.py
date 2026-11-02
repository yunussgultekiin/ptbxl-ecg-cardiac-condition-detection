import torch
from torch import nn

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=True):
        padding = "same" if stride == 1 else kernel_size // 2
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm1d(out_channels)]
        super().__init__(*layers, *([nn.ReLU(inplace=True)] if activation else []))

class ConcatPool(nn.Module):
    def forward(self, features):
        return torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)

class EcgClassifier(nn.Module):
    def __init__(self, backbone, class_count, metadata_size, dropout):
        super().__init__()
        self.backbone = backbone
        self.pool = ConcatPool()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * backbone.feature_channels + metadata_size, class_count))

    def forward(self, signals, metadata):
        return self.head(torch.cat([self.pool(self.backbone(signals)), metadata], dim=1))