"""FFB6D - Full Flow Bidirectional Fusion Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from rgbd_pipelines.base import BasePoseModel
from rgbd_pipelines.backbones.resnet import ResNetBackbone


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


class FFB6D(BasePoseModel):
    """Bidirectional fusion at every encoder level."""
    
    def __init__(
        self,
        num_points: int = 1024,
        num_obj: int = 13,
        num_kp: int = 8,
        pretrained_rgb: bool = True
    ):
        super().__init__(num_points, num_obj)
        self.model_name = "ffb6d"
        self.num_kp = num_kp
        
        self.rgb_backbone = ResNetBackbone(arch='resnet34', pretrained=pretrained_rgb, feature_dim=256)
        
        self.point_layer1 = PointLayer(in_dim=0, out_dim=64, npoint=512)
        self.point_layer2 = PointLayer(in_dim=64, out_dim=128, npoint=256)
        self.point_layer3 = PointLayer(in_dim=128, out_dim=256, npoint=128)
        self.point_layer4 = PointLayer(in_dim=256, out_dim=512, npoint=64)
        
        self.fusion1 = BidirectionalFusion(rgb_dim=64, point_dim=64)
        self.fusion2 = BidirectionalFusion(rgb_dim=128, point_dim=128)
        self.fusion3 = BidirectionalFusion(rgb_dim=256, point_dim=256)
        self.fusion4 = BidirectionalFusion(rgb_dim=512, point_dim=512)
        
        self.rgb_adapt1 = nn.Conv2d(64, 64, 1)
        self.rgb_adapt2 = nn.Conv2d(128, 128, 1)
        self.rgb_adapt3 = nn.Conv2d(256, 256, 1)
        self.rgb_adapt4 = nn.Conv2d(512, 512, 1)
        
        self.fp4 = FeatureProp(512, 256, 256)
        self.fp3 = FeatureProp(256, 128, 128)
        self.fp2 = FeatureProp(128, 64, 64)
        self.fp1 = FeatureProp(64, 3, 64)
        
        self.final_fusion = nn.Sequential(
            nn.Conv1d(64 + 256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True)
        )
        
        self.kp_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_obj * num_kp * 3, 1)
        )
        self.ctr_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_obj * 3, 1)
        )
        self.sem_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_obj + 1, 1)
        )
    
    def forward(self, rgb, points, choose, obj_idx, **kwargs) -> Dict[str, torch.Tensor]:
        B, N, _ = points.shape
        
        c1, c2, c3, c4 = self.rgb_backbone.get_intermediate_features(rgb)
        c1 = self.rgb_adapt1(c1)
        c2 = self.rgb_adapt2(c2)
        c3 = self.rgb_adapt3(c3)
        c4 = self.rgb_adapt4(c4)
        
        xyz0 = points
        xyz1, p1 = self.point_layer1(xyz0, None)
        xyz2, p2 = self.point_layer2(xyz1, p1)
        xyz3, p3 = self.point_layer3(xyz2, p2)
        xyz4, p4 = self.point_layer4(xyz3, p3)
        
        rgb_p1 = self._sample_rgb(c1, choose, p1.shape[1])
        _, p1_fused = self.fusion1(rgb_p1, p1)
        
        rgb_p2 = self._sample_rgb(c2, choose, p2.shape[1])
        _, p2_fused = self.fusion2(rgb_p2, p2)
        
        rgb_p3 = self._sample_rgb(c3, choose, p3.shape[1])
        _, p3_fused = self.fusion3(rgb_p3, p3)
        
        rgb_p4 = self._sample_rgb(c4, choose, p4.shape[1])
        _, p4_fused = self.fusion4(rgb_p4, p4)
        
        up4 = self.fp4(xyz3, xyz4, p3_fused, p4_fused)
        up3 = self.fp3(xyz2, xyz3, p2_fused, up4)
        up2 = self.fp2(xyz1, xyz2, p1_fused, up3)
        up1 = self.fp1(xyz0, xyz1, xyz0, up2)
        
        rgb_final = self._sample_rgb(c3, choose, N)
        combined = torch.cat([up1, rgb_final], dim=-1).transpose(1, 2)
        fused = self.final_fusion(combined)
        
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
        target_kp_centered = voted_keypoints - voted_center.unsqueeze(1)  # (B, K, 3)
        
        # Create canonical keypoints
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
            'voted_center': voted_center
        }
    
    def _sample_rgb(self, feat_map, choose, target_n):
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.view(B, C, -1)
        N = choose.shape[1]
        
        if N >= target_n:
            indices = choose[:, :target_n]
        else:
            indices = choose.repeat(1, (target_n // N) + 1)[:, :target_n]
        
        indices = indices.clamp(0, H*W-1).unsqueeze(1).expand(-1, C, -1)
        return torch.gather(feat_flat, 2, indices).transpose(1, 2)
    
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
        from rgbd_pipelines.ffb6d.loss import FFB6DLoss
        
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
        
        return FFB6DLoss(num_kp=self.num_kp)(
            pred_dict,
            kp_targ=kp_targ,
            ctr_targ=ctr_targ,
            points=points,
            **kwargs
        )


class PointLayer(nn.Module):
    """Single point cloud processing layer with subsampling."""
    
    def __init__(self, in_dim, out_dim, npoint):
        super().__init__()
        self.npoint = npoint
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim + 3, out_dim // 2, 1), nn.BatchNorm1d(out_dim // 2), nn.ReLU(inplace=True),
            nn.Conv1d(out_dim // 2, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)
        )
    
    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        
        if self.npoint < N:
            idx = torch.randperm(N, device=xyz.device)[:self.npoint]
            new_xyz = xyz[:, idx, :]
        else:
            new_xyz = xyz
            idx = torch.arange(N, device=xyz.device)
        
        if features is not None:
            sel_feat = features[:, idx, :]
            x = torch.cat([new_xyz, sel_feat], dim=-1)
        else:
            x = new_xyz
        
        x = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        return new_xyz, x


class BidirectionalFusion(nn.Module):
    """Bidirectional RGB-Point fusion block."""
    
    def __init__(self, rgb_dim, point_dim):
        super().__init__()
        self.rgb2point = nn.Linear(rgb_dim, point_dim)
        self.point2rgb = nn.Linear(point_dim, rgb_dim)
        self.rgb_out = nn.Sequential(nn.Linear(rgb_dim*2, rgb_dim), nn.ReLU())
        self.point_out = nn.Sequential(nn.Linear(point_dim*2, point_dim), nn.ReLU())
    
    def forward(self, rgb_feat, point_feat):
        point_from_rgb = self.rgb2point(rgb_feat)
        rgb_from_point = self.point2rgb(point_feat)
        
        rgb_out = self.rgb_out(torch.cat([rgb_feat, rgb_from_point], dim=-1))
        point_out = self.point_out(torch.cat([point_feat, point_from_rgb], dim=-1))
        
        return rgb_out, point_out


class FeatureProp(nn.Module):
    """Feature propagation for upsampling point features."""
    
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim + skip_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)
        )
    
    def forward(self, xyz1, xyz2, feat1, feat2):
        B, N1, _ = xyz1.shape
        _, N2, C2 = feat2.shape
        
        dist = torch.cdist(xyz1, xyz2)
        _, idx = dist.topk(3, dim=-1, largest=False)
        
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C2)
        feat2_exp = feat2.unsqueeze(1).expand(-1, N1, -1, -1)
        interpolated = torch.gather(feat2_exp, 2, idx_exp).mean(dim=2)
        
        concat = torch.cat([feat1, interpolated], dim=-1).transpose(1, 2)
        return self.mlp(concat).transpose(1, 2)
