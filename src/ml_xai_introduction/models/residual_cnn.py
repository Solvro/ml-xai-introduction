"""Residual CNN model from Damian's branch."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ml_xai_introduction.models.model_base import Model
from ml_xai_introduction.models.model_factory import model_registry


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, channel_expansion: int = 1) -> None:
        super().__init__()
        self.expanded_channels = out_channels * channel_expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.expanded_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.downsample: nn.Sequential | None = None
        if in_channels != self.expanded_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expanded_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.leaky_relu(x + residual)


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_block1 = ResidualBlock(64, 64, channel_expansion=4)
        self.residual_block2 = ResidualBlock(256, 64, channel_expansion=4)
        self.residual_block3 = ResidualBlock(256, 64, channel_expansion=4)
        self.residual_block4 = ResidualBlock(256, 128, channel_expansion=4, stride=2)
        self.residual_block5 = ResidualBlock(512, 128, channel_expansion=4)
        self.residual_block6 = ResidualBlock(512, 128, channel_expansion=4)
        self.residual_block7 = ResidualBlock(512, 128, channel_expansion=4)
        self.residual_block8 = ResidualBlock(512, 256, channel_expansion=2)
        self.residual_block9 = ResidualBlock(512, 256, channel_expansion=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.residual_block7(x)
        x = self.residual_block8(x)
        x = self.residual_block9(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.leaky_relu(self.fc1(x)))
        x = self.dropout2(self.leaky_relu(self.fc2(x)))
        return self.fc3(x)


@model_registry.register("residual_cnn")
def build_residual_cnn(cfg: DictConfig) -> Model:
    return ResidualCNN(int(cfg.data.num_classes))
