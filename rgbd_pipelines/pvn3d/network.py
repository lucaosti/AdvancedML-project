"""PVN3D - Point-wise 3D Keypoint Voting Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from rgbd_pipelines.base import BasePoseModel
from rgbd_pipelines.backbones.resnet import ResNetBackbone
from rgbd_pipelines.backbones.pointnet import PointNetPPEncoder


def svd_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute rotation from source to target keypoints via SVD (Procrustes)."""
    B = source.shape[0]
    
    source_centered = source - source.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    
    H = torch.bmm(source_centered.transpose(1, 2), target_centered)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    
    det = torch.det(R)
    sign_matrix = torch.ones(B, 3, device=source.device)
    sign_matrix[:, 2] = det.sign()
    R = torch.bmm(Vt.transpose(1, 2) * sign_matrix.unsqueeze(1), U.transpose(1, 2))
    
    return R


class PVN3D(BasePoseModel):
    """Keypoint voting network for 6D pose estimation."""
    
    def __init__(
        self,
        num_points: int = 1024,
        num_obj: int = 13,
        num_kp: int = 8,
        rgb_feature_dim: int = 256,
        geo_feature_dim: int = 256,
        fused_dim: int = 256,
        pretrained_rgb: bool = True
    ):
        super().__init__(num_points, num_obj)
        self.model_name = "pvn3d"
        self.num_kp = num_kp
        
        self.rgb_backbone = ResNetBackbone(
            arch='resnet34', pretrained=pretrained_rgb, feature_dim=rgb_feature_dim
        )
        self.geo_backbone = PointNetPPEncoder(input_dim=3, feature_dim=geo_feature_dim)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(rgb_feature_dim + geo_feature_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, fused_dim, 1),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )
        
        self.kp_head = nn.Sequential(
            nn.Conv1d(fused_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, num_obj * num_kp * 3, 1)
        )
        
        self.ctr_head = nn.Sequential(
            nn.Conv1d(fused_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, num_obj * 3, 1)
        )
        
        self.sem_head = nn.Sequential(
            nn.Conv1d(fused_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, num_obj + 1, 1)
        )
    
    def forward(self, rgb, points, choose, obj_idx, **kwargs) -> Dict[str, torch.Tensor]:
        B, N, _ = points.shape
        
        rgb_spatial, _ = self.rgb_backbone(rgb)
        _, C, H, W = rgb_spatial.shape
        
        rgb_flat = rgb_spatial.view(B, C, -1)
        choose_exp = choose.unsqueeze(1).expand(-1, C, -1).clamp(0, H*W-1)
        rgb_feat = torch.gather(rgb_flat, 2, choose_exp)
        
        _, geo_global = self.geo_backbone(points)
        geo_feat = geo_global.unsqueeze(2).expand(-1, -1, N)
        
        fused = self.fusion(torch.cat([rgb_feat, geo_feat], dim=1))
        
        kp_raw = self.kp_head(fused)
        ctr_raw = self.ctr_head(fused)
        sem_logits = self.sem_head(fused).transpose(1, 2)
        
        kp_raw = kp_raw.view(B, self.num_obj, self.num_kp, 3, N)
        idx = obj_idx.view(B, 1, 1, 1, 1).expand(-1, -1, self.num_kp, 3, N)
        kp_offsets = torch.gather(kp_raw, 1, idx).squeeze(1).permute(0, 3, 1, 2)
        
        ctr_raw = ctr_raw.view(B, self.num_obj, 3, N)
        idx_c = obj_idx.view(B, 1, 1, 1).expand(-1, -1, 3, N)
        ctr_offsets = torch.gather(ctr_raw, 1, idx_c).squeeze(1).transpose(1, 2)
        
        points_exp = points.unsqueeze(2).expand(-1, -1, self.num_kp, -1)
        voted_kp_all = points_exp + kp_offsets
        voted_keypoints = voted_kp_all.mean(dim=1)  # (B, K, 3)
        voted_center = (points + ctr_offsets).mean(dim=1)  # (B, 3)
        
        # Compute rotation using SVD from canonical keypoints to voted keypoints
        # Use voted_keypoints centered at voted_center as target
        # Use identity-centered canonical keypoints as source (learn them implicitly)
        # For simplicity, we estimate rotation from the keypoint configuration
        target_kp_centered = voted_keypoints - voted_center.unsqueeze(1)  # (B, K, 3)
        
        # Create canonical keypoints (unit cube corners for K=8)
        canonical_kp = self._get_canonical_keypoints(B, self.num_kp, points.device)  # (B, K, 3)
        
        # Compute rotation via Procrustes (SVD)
        rotation = svd_rotation(canonical_kp, target_kp_centered)  # (B, 3, 3)
        
        return {
            'rotation': rotation,
            'translation': voted_center,
            'kp_offsets': kp_offsets,
            'ctr_offsets': ctr_offsets,
            'sem_logits': sem_logits,
            'voted_keypoints': voted_keypoints,
            'voted_center': voted_center,
            'voted_kp_all': voted_kp_all
        }
    
    def _get_canonical_keypoints(self, batch_size: int, num_kp: int, device: torch.device) -> torch.Tensor:
        """Get canonical keypoints (corners of unit cube centered at origin)."""
        if num_kp == 8:
            # 8 corners of a unit cube centered at origin
            kp = torch.tensor([
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5,  0.5],
                [-0.5,  0.5, -0.5],
                [-0.5,  0.5,  0.5],
                [ 0.5, -0.5, -0.5],
                [ 0.5, -0.5,  0.5],
                [ 0.5,  0.5, -0.5],
                [ 0.5,  0.5,  0.5],
            ], dtype=torch.float32, device=device)
        else:
            # For other num_kp, use random but fixed keypoints
            torch.manual_seed(42)
            kp = torch.randn(num_kp, 3, device=device) * 0.5
        
        return kp.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_loss(self, pred_dict, gt_rotation, gt_translation, model_points, **kwargs):
        from rgbd_pipelines.pvn3d.loss import PVN3DLoss
        
        # Compute target keypoints from GT pose
        # canonical_kp: [B, K, 3] - keypoints in model frame
        # target_kp = R @ canonical_kp + t
        B = gt_rotation.shape[0]
        device = gt_rotation.device
        
        # Get canonical keypoints (same as used in forward pass)
        canonical_kp = self._get_canonical_keypoints(B, self.num_kp, device)  # [B, K, 3]
        
        # Transform to scene frame: kp_target = R @ kp_canonical + t
        kp_targ = torch.bmm(canonical_kp, gt_rotation.transpose(1, 2)) + gt_translation.unsqueeze(1)  # [B, K, 3]
        
        # Center target is simply the GT translation
        ctr_targ = gt_translation  # [B, 3]
        
        # Extract points from kwargs to avoid duplication
        points = kwargs.pop('points', None)
        
        return PVN3DLoss(num_kp=self.num_kp)(
            pred_dict,
            kp_targ=kp_targ,
            ctr_targ=ctr_targ,
            points=points,
            **kwargs
        )
