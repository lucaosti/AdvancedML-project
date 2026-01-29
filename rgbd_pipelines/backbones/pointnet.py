"""PointNet-based feature extractors for point clouds."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PointNetEncoder(nn.Module):
    """Basic PointNet encoder with per-point and global features."""
    
    def __init__(self, input_dim=3, feature_dim=256):
        super().__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True)
        )
        self.feature_dim = feature_dim
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return point features [B, N, C] and global features [B, C]."""
        x = points.transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        point_features = x.transpose(1, 2)
        global_features = torch.max(x, dim=2)[0]
        
        return point_features, global_features


class PointNetPPEncoder(nn.Module):
    """PointNet++ with hierarchical set abstraction layers."""
    
    def __init__(self, input_dim=3, feature_dim=256):
        super().__init__()
        
        self.sa1 = SetAbstraction(npoint=512, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, mlp=[128, 128, 256], in_channels=128)
        self.sa3 = SetAbstraction(npoint=32, mlp=[256, 256, 512], in_channels=256)
        
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True)
        )
        self.feature_dim = feature_dim
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hierarchical features [B, 32, 512] and global features [B, C]."""
        xyz = points[..., :3]
        
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        
        global_feat = torch.max(feat3, dim=1)[0]
        global_feat = self.fc(global_feat)
        
        return feat3, global_feat


class SetAbstraction(nn.Module):
    """Set Abstraction layer for PointNet++."""
    
    def __init__(self, npoint: int, mlp: list, in_channels: int = 0):
        super().__init__()
        
        self.npoint = npoint
        
        layers = []
        prev_ch = 3 + in_channels
        for out_ch in mlp:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ])
            prev_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = mlp[-1]
    
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        
        if self.npoint < N:
            idx = torch.randperm(N, device=xyz.device)[:self.npoint]
            new_xyz = xyz[:, idx, :]
        else:
            new_xyz = xyz
            idx = torch.arange(N, device=xyz.device)
        
        if features is not None:
            x = torch.cat([new_xyz, features[:, idx, :]], dim=-1)
        else:
            x = new_xyz
        
        x = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        return new_xyz, x
