"""RGB-D Fusion Network for 6D Pose Estimation (DenseFusion-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class PointNetEncoder(nn.Module):
    """PointNet encoder for 3D point cloud features."""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 256):
        super(PointNetEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns per-point features [B, N, C] and global features [B, C]."""
        x = points.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        point_features = x.transpose(1, 2)
        global_features = torch.max(x, dim=2)[0]
        
        return point_features, global_features


class RGBEncoder(nn.Module):
    """CNN encoder using ResNet18 backbone."""
    
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super(RGBEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        if pretrained:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except (TypeError, AttributeError):
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(pretrained=False)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(512, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns spatial features [B, C, H', W'] and global features [B, C]."""
        x = self.backbone(rgb)
        spatial = self.feature_conv(x)
        
        global_feat = self.global_pool(spatial)
        global_feat = global_feat.view(global_feat.size(0), -1)
        
        return spatial, global_feat


class DenseFusionModule(nn.Module):
    """Feature fusion module combining RGB and point features."""
    
    def __init__(self, rgb_dim: int = 256, point_dim: int = 256, fused_dim: int = 512):
        super(DenseFusionModule, self).__init__()
        
        self.fused_dim = fused_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(rgb_dim + point_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self, 
        rgb_global: torch.Tensor, 
        point_global: torch.Tensor,
        point_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse RGB and point cloud global features."""
        concat = torch.cat([rgb_global, point_global], dim=1)
        fused = self.fusion_mlp(concat)
        return fused


class RotationHead(nn.Module):
    """Rotation prediction head outputting normalized quaternions."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super(RotationHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict normalized quaternion [B, 4]."""
        raw_quat = self.mlp(features)
        quaternion = F.normalize(raw_quat, p=2, dim=1)
        return quaternion


class TranslationResidualHead(nn.Module):
    """Translation residual prediction head (T_final = T_geo + delta_T)."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super(TranslationResidualHead, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        self._init_small_residual()
        
    def _init_small_residual(self):
        """Initialize final layer for small initial residuals."""
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear) and module.out_features == 3:
                nn.init.zeros_(module.bias)
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                break
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict translation residual [B, 3] in meters."""
        delta_t = self.mlp(features)
        return delta_t


class RGBDFusionNetwork(nn.Module):
    """DenseFusion-style RGB-D Network for 6D Pose Estimation."""
    
    def __init__(
        self,
        num_points: int = 500,
        rgb_feature_dim: int = 256,
        point_feature_dim: int = 256,
        fused_dim: int = 512,
        pretrained_rgb: bool = True
    ):
        super(RGBDFusionNetwork, self).__init__()
        
        self.num_points = num_points
        self.rgb_feature_dim = rgb_feature_dim
        self.point_feature_dim = point_feature_dim
        self.fused_dim = fused_dim
        
        # Feature Extractors
        self.rgb_encoder = RGBEncoder(
            pretrained=pretrained_rgb,
            feature_dim=rgb_feature_dim
        )
        
        self.point_encoder = PointNetEncoder(
            input_dim=3,
            feature_dim=point_feature_dim
        )
        
        # Fusion Module
        self.fusion = DenseFusionModule(
            rgb_dim=rgb_feature_dim,
            point_dim=point_feature_dim,
            fused_dim=fused_dim
        )
        
        # Prediction Heads
        self.rotation_head = RotationHead(
            input_dim=fused_dim,
            hidden_dim=256
        )
        
        self.translation_head = TranslationResidualHead(
            input_dim=fused_dim,
            hidden_dim=256
        )
        
    def forward(
        self,
        rgb: torch.Tensor,
        points: torch.Tensor,
        T_geo: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning quaternion, delta_t, and T_final."""
        batch_size = rgb.size(0)
        
        rgb_spatial, rgb_global = self.rgb_encoder(rgb)
        point_features, point_global = self.point_encoder(points)
        fused_features = self.fusion(rgb_global, point_global)
        
        quaternion = self.rotation_head(fused_features)
        delta_t = self.translation_head(fused_features)
        
        if T_geo is not None:
            T_final = T_geo + delta_t
        else:
            T_final = delta_t
        
        return quaternion, delta_t, T_final
    
    def predict_pose(
        self,
        rgb: torch.Tensor,
        points: torch.Tensor,
        T_geo: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict full 6D pose given inputs and geometric anchor."""
        quaternion, delta_t, T_final = self.forward(rgb, points, T_geo)
        return quaternion, T_final
    
    def get_num_parameters(self) -> dict:
        """Get parameter counts for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'rgb_encoder': count_params(self.rgb_encoder),
            'point_encoder': count_params(self.point_encoder),
            'fusion': count_params(self.fusion),
            'rotation_head': count_params(self.rotation_head),
            'translation_head': count_params(self.translation_head),
            'total': count_params(self)
        }


class RGBDFusionNetworkLite(nn.Module):
    """Lightweight version of RGBDFusionNetwork for faster training."""
    
    def __init__(
        self,
        num_points: int = 500,
        rgb_feature_dim: int = 128,
        point_feature_dim: int = 128,
        fused_dim: int = 256,
        pretrained_rgb: bool = True
    ):
        super(RGBDFusionNetworkLite, self).__init__()
        
        self.num_points = num_points
        
        # Simplified RGB encoder (just a few conv layers)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, rgb_feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(rgb_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # PointNet encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, point_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Fusion and heads
        self.fusion = nn.Sequential(
            nn.Linear(rgb_feature_dim + point_feature_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.rotation_head = nn.Linear(fused_dim, 4)
        self.translation_head = nn.Linear(fused_dim, 3)
        
        # Small initialization for translation residual
        nn.init.zeros_(self.translation_head.bias)
        nn.init.normal_(self.translation_head.weight, std=0.001)
        
    def forward(
        self,
        rgb: torch.Tensor,
        points: torch.Tensor,
        T_geo: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size = rgb.size(0)
        
        # RGB features
        rgb_feat = self.rgb_encoder(rgb).view(batch_size, -1)
        
        # Point features (with max pooling)
        point_feat = self.point_encoder(points)  # [B, N, feat]
        point_feat = torch.max(point_feat, dim=1)[0]  # [B, feat]
        
        # Fusion
        fused = self.fusion(torch.cat([rgb_feat, point_feat], dim=1))
        
        # Predictions
        quaternion = F.normalize(self.rotation_head(fused), p=2, dim=1)
        delta_t = self.translation_head(fused)
        
        T_final = T_geo + delta_t if T_geo is not None else delta_t
        
        return quaternion, delta_t, T_final


def create_rgbd_fusion_model(
    variant: str = 'standard',
    num_points: int = 500,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """Factory function to create RGB-D Fusion models."""
    if variant == 'standard':
        return RGBDFusionNetwork(
            num_points=num_points,
            pretrained_rgb=pretrained,
            **kwargs
        )
    elif variant == 'lite':
        return RGBDFusionNetworkLite(
            num_points=num_points,
            pretrained_rgb=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'standard' or 'lite'")
