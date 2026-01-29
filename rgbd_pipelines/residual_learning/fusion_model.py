"""Feature fusion and pose regression for RGB-D."""

import torch
import torch.nn as nn
import torchvision.models as models
from rgbd_pipelines.residual_learning.depth_model import DepthEncoder


class RGBDPoseEstimator(nn.Module):
    """RGB-D pose estimator with feature fusion."""
    
    def __init__(self, depth_channels=64, pretrained_rgb=True, pretrained_path=None):
        super(RGBDPoseEstimator, self).__init__()
        
        self.rgb_backbone = models.resnet50(pretrained=pretrained_rgb)

        if pretrained_path:
            print("Loading pretrained weights from '{}'".format(pretrained_path))
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    clean_state_dict[k.replace('backbone.', '')] = v
                elif k.startswith('rgb_backbone.'):
                    clean_state_dict[k.replace('rgb_backbone.', '')] = v
                else:
                    clean_state_dict[k] = v
            self.rgb_backbone.load_state_dict(clean_state_dict, strict=False)

        self.rgb_backbone.fc = nn.Identity()
        rgb_features = 2048

        self.depth_encoder = DepthEncoder(depth_channels=depth_channels)

        fused_features = rgb_features + depth_channels
        self.fusion = nn.Sequential(
            nn.Linear(fused_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 4)
    def forward(self, rgb, depth):
        """Extract features from RGB and depth, return translation and quaternion."""
        rgb_feat = self.rgb_backbone(rgb)
        depth_feat = self.depth_encoder(depth)
        depth_feat = depth_feat.view(depth.size(0), -1)

        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        fused_feat = self.fusion(fused)

        translation = self.translation_head(fused_feat)
        quaternion = self.rotation_head(fused_feat)
        quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=1)
        
        return translation, quaternion
