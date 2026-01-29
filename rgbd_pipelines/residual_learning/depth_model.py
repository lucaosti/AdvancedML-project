"""CNN for depth map processing."""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """Convolutional encoder for depth maps."""
    
    def __init__(self, depth_channels=64):
        super(DepthEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, depth_channels, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(depth_channels)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """Encode depth map into feature representation."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x
