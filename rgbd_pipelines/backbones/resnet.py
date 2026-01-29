"""ResNet-based feature extractors for RGB images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class ResNetBackbone(nn.Module):
    """ResNet backbone with spatial and global feature output."""
    
    def __init__(self, arch='resnet18', pretrained=True, feature_dim=256):
        super().__init__()
        
        if arch == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            in_channels = 512
        elif arch == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            in_channels = 512
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            in_channels = 2048
        else:
            raise ValueError(f"Unknown arch: {arch}")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.proj = nn.Conv2d(in_channels, feature_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (spatial [B,C,H',W'], global [B,C])."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        spatial = self.proj(x)
        global_feat = self.global_pool(spatial).view(x.size(0), -1)
        
        return spatial, global_feat
    
    def get_intermediate_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns features at multiple scales (c1, c2, c3, c4)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return c1, c2, c3, c4


class PSPNet(nn.Module):
    """Pyramid Spatial Pooling Network for dense features."""
    
    def __init__(self, arch='resnet18', pretrained=True, feature_dim=256):
        super().__init__()
        
        self.backbone = ResNetBackbone(arch, pretrained, feature_dim)
        self.pool_sizes = [1, 2, 3, 6]
        
        self.psp_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(feature_dim, feature_dim // 4, 1),
                nn.BatchNorm2d(feature_dim // 4),
                nn.ReLU(inplace=True)
            ) for size in self.pool_sizes
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns dense features [B, C, H', W']."""
        spatial, _ = self.backbone(x)
        h, w = spatial.shape[2:]
        
        psp_out = [spatial]
        for module in self.psp_modules:
            pooled = module(spatial)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            psp_out.append(upsampled)
        
        concat = torch.cat(psp_out, dim=1)
        return self.final_conv(concat)
