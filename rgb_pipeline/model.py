"""ResNet-based 6D pose estimator for RGB images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class RGBPoseEstimator(nn.Module):
    """ResNet-based network for 6D pose estimation from RGB crops."""
    
    def __init__(self, pretrained: bool = True, backbone: str = 'resnet50'):
        super().__init__()
        self.backbone_name = backbone.lower()
        
        # Load backbone
        backbone_cls = {
            'resnet18': (models.resnet18, models.ResNet18_Weights),
            'resnet34': (models.resnet34, models.ResNet34_Weights),
            'resnet50': (models.resnet50, models.ResNet50_Weights),
            'resnet101': (models.resnet101, models.ResNet101_Weights),
            'resnet152': (models.resnet152, models.ResNet152_Weights),
        }
        
        if self.backbone_name not in backbone_cls:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        model_fn, weights_cls = backbone_cls[self.backbone_name]
        try:
            weights = weights_cls.DEFAULT if pretrained else None
            self.backbone = model_fn(weights=weights)
        except AttributeError:
            self.backbone = model_fn(pretrained=pretrained)
        
        # Get backbone output features
        backbone_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Pinhole translation embedding (3D -> 64D)
        self.pinhole_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Fusion layer: backbone features + pinhole embedding
        fusion_input_dim = backbone_out_features + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Rotation head: outputs unit quaternion [w, x, y, z]
        self.rotation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
        # Translation head: outputs 3D translation [x, y, z]
        self.translation_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        pinhole_translation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict rotation quaternion and translation from RGB and pinhole estimate."""
        visual_features = self.backbone(rgb)
        pinhole_embed = self.pinhole_embed(pinhole_translation)
        
        fused = torch.cat([visual_features, pinhole_embed], dim=1)
        fused = self.fusion(fused)
        
        raw_quat = self.rotation_head(fused)
        rotation = F.normalize(raw_quat, p=2, dim=1)
        translation = self.translation_head(fused)
        
        return rotation, translation
    
    def get_num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze backbone weights for fine-tuning heads only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for full training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
