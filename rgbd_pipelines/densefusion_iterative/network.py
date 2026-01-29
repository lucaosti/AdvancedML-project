"""
DenseFusion Iterative - Full per-pixel fusion with refinement.

Features per-point dense fusion, confidence estimation, and iterative
pose refinement. Based on Wang et al., CVPR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from rgbd_pipelines.base import BasePoseModel
from rgbd_pipelines.backbones.resnet import PSPNet


class DenseFusionIterative(BasePoseModel):
    """Per-pixel dense fusion with iterative refinement."""
    
    def __init__(
        self,
        num_points: int = 1024,
        num_obj: int = 13,
        num_iter: int = 2,
        rgb_feature_dim: int = 256,
        point_feature_dim: int = 128,
        global_feature_dim: int = 1024,
        fused_dim: int = 256,
        pretrained_rgb: bool = True
    ):
        super().__init__(num_points, num_obj)
        self.model_name = "densefusion_iterative"
        self.num_iter = num_iter
        
        self.rgb_encoder = PSPNet(
            arch='resnet18', pretrained=pretrained_rgb, feature_dim=rgb_feature_dim
        )
        self.point_encoder = PointNetFeat(
            input_dim=3, point_feat_dim=point_feature_dim, global_feat_dim=global_feature_dim
        )
        self.fusion = DenseFusionModule(
            rgb_dim=rgb_feature_dim, point_dim=point_feature_dim,
            global_dim=global_feature_dim, fused_dim=fused_dim
        )
        self.estimator = PoseEstimator(fused_dim=fused_dim, num_obj=num_obj)
        self.refiner = PoseRefiner(fused_dim=fused_dim, num_obj=num_obj) if num_iter > 0 else None
    
    def forward(self, rgb, points, choose, obj_idx, **kwargs) -> Dict[str, torch.Tensor]:
        B, N, _ = points.shape
        
        rgb_feat_map = self.rgb_encoder(rgb)
        _, C, H, W = rgb_feat_map.shape
        
        rgb_feat_flat = rgb_feat_map.view(B, C, -1)
        choose_exp = choose.unsqueeze(1).expand(-1, C, -1).clamp(0, H*W-1)
        rgb_feat = torch.gather(rgb_feat_flat, 2, choose_exp).transpose(1, 2)
        
        point_feat, global_feat = self.point_encoder(points)
        fused_feat = self.fusion(rgb_feat, point_feat, global_feat)
        
        per_point_quat, per_point_trans, confidence = self.estimator(fused_feat, obj_idx)
        
        conf_flat = confidence.squeeze(-1)
        max_idx = torch.argmax(conf_flat, dim=1)
        
        idx_q = max_idx.view(B, 1, 1).expand(-1, -1, 4)
        best_quat = torch.gather(per_point_quat, 1, idx_q).squeeze(1)
        
        idx_t = max_idx.view(B, 1, 1).expand(-1, -1, 3)
        best_trans = torch.gather(per_point_trans, 1, idx_t).squeeze(1)
        best_point = torch.gather(points, 1, idx_t).squeeze(1)
        
        initial_trans = best_trans + best_point
        initial_rot = self._quaternion_to_matrix(best_quat)
        
        refined_rotations = [initial_rot]
        refined_translations = [initial_trans]
        current_rot, current_trans = initial_rot, initial_trans
        
        if self.refiner is not None:
            for _ in range(self.num_iter):
                current_rot, current_trans = self.refiner(
                    fused_feat, points, current_rot, current_trans, obj_idx
                )
                refined_rotations.append(current_rot)
                refined_translations.append(current_trans)
        
        return {
            'rotation': current_rot,
            'translation': current_trans,
            'confidence': confidence,
            'per_point_rotation': per_point_quat,
            'per_point_translation': per_point_trans,
            'refined_rotations': refined_rotations,
            'refined_translations': refined_translations,
            'fused_features': fused_feat
        }
    
    def get_loss(self, pred_dict, gt_rotation, gt_translation, model_points, **kwargs):
        from rgbd_pipelines.densefusion_iterative.loss import DenseFusionIterativeLoss
        return DenseFusionIterativeLoss()(pred_dict, gt_rotation, gt_translation, model_points, **kwargs)
    
    @staticmethod
    def _quaternion_to_matrix(q):
        q = F.normalize(q, p=2, dim=1)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        B = q.shape[0]
        R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2*y*y - 2*z*z
        R[:, 0, 1] = 2*x*y - 2*w*z
        R[:, 0, 2] = 2*x*z + 2*w*y
        R[:, 1, 0] = 2*x*y + 2*w*z
        R[:, 1, 1] = 1 - 2*x*x - 2*z*z
        R[:, 1, 2] = 2*y*z - 2*w*x
        R[:, 2, 0] = 2*x*z - 2*w*y
        R[:, 2, 1] = 2*y*z + 2*w*x
        R[:, 2, 2] = 1 - 2*x*x - 2*y*y
        return R


class PointNetFeat(nn.Module):
    """PointNet with per-point and global features."""
    
    def __init__(self, input_dim=3, point_feat_dim=128, global_feat_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, point_feat_dim, 1)
        self.conv4 = nn.Conv1d(point_feat_dim, 256, 1)
        self.conv5 = nn.Conv1d(256, global_feat_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(point_feat_dim)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(global_feat_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_feat = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(point_feat)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feat = torch.max(x, dim=2)[0]
        return point_feat.transpose(1, 2), global_feat


class DenseFusionModule(nn.Module):
    """Per-point dense fusion of RGB, point, and global features."""
    
    def __init__(self, rgb_dim, point_dim, global_dim, fused_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(rgb_dim + point_dim + global_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, fused_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(fused_dim)
    
    def forward(self, rgb_feat, point_feat, global_feat):
        B, N, _ = rgb_feat.shape
        global_tiled = global_feat.unsqueeze(1).expand(-1, N, -1)
        concat = torch.cat([rgb_feat, point_feat, global_tiled], dim=2).transpose(1, 2)
        fused = F.relu(self.bn1(self.conv1(concat)))
        fused = F.relu(self.bn2(self.conv2(fused)))
        return fused.transpose(1, 2)


class PoseEstimator(nn.Module):
    """Per-point pose estimation with confidence scores."""
    
    def __init__(self, fused_dim, num_obj):
        super().__init__()
        self.num_obj = num_obj
        self.conv1 = nn.Conv1d(fused_dim, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv_r = nn.Conv1d(256, num_obj * 4, 1)
        self.conv_t = nn.Conv1d(256, num_obj * 3, 1)
        self.conv_c = nn.Conv1d(256, num_obj * 1, 1)
    
    def forward(self, fused_feat, obj_idx):
        B, N, _ = fused_feat.shape
        x = fused_feat.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        r = self.conv_r(x).view(B, self.num_obj, 4, N)
        t = self.conv_t(x).view(B, self.num_obj, 3, N)
        c = self.conv_c(x).view(B, self.num_obj, 1, N)
        
        idx = obj_idx.view(B, 1, 1, 1).expand(-1, -1, 4, N)
        rotation = torch.gather(r, 1, idx).squeeze(1).permute(0, 2, 1)
        
        idx_t = obj_idx.view(B, 1, 1, 1).expand(-1, -1, 3, N)
        translation = torch.gather(t, 1, idx_t).squeeze(1).permute(0, 2, 1)
        
        idx_c = obj_idx.view(B, 1, 1, 1).expand(-1, -1, 1, N)
        confidence = torch.gather(c, 1, idx_c).squeeze(1).permute(0, 2, 1)
        
        return F.normalize(rotation, p=2, dim=2), translation, confidence


class PoseRefiner(nn.Module):
    """Iterative pose refinement module."""
    
    def __init__(self, fused_dim, num_obj):
        super().__init__()
        self.num_obj = num_obj
        self.conv1 = nn.Conv1d(fused_dim + 3, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc_r = nn.Linear(256, num_obj * 4)
        self.fc_t = nn.Linear(256, num_obj * 3)
    
    def forward(self, fused_feat, points, current_rot, current_trans, obj_idx):
        B, N, _ = fused_feat.shape
        
        centered = points - current_trans.unsqueeze(1)
        transformed = torch.bmm(centered, current_rot.transpose(1, 2))
        
        x = torch.cat([fused_feat, transformed], dim=2).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        delta_r = self.fc_r(x).view(B, self.num_obj, 4)
        delta_t = self.fc_t(x).view(B, self.num_obj, 3)
        
        idx_r = obj_idx.view(B, 1, 1).expand(-1, -1, 4)
        delta_quat = F.normalize(torch.gather(delta_r, 1, idx_r).squeeze(1), p=2, dim=1)
        
        idx_t = obj_idx.view(B, 1, 1).expand(-1, -1, 3)
        delta_trans = torch.gather(delta_t, 1, idx_t).squeeze(1)
        
        delta_R = DenseFusionIterative._quaternion_to_matrix(delta_quat)
        
        return torch.bmm(delta_R, current_rot), current_trans + delta_trans
