"""
Common loss functions for RGB-D pose estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicLoss(nn.Module):
    """Geodesic loss for rotation matrices on SO(3) manifold."""
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance between predicted and ground truth rotations."""
        R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        trace = torch.clamp(trace, -1 + self.eps, 3 - self.eps)
        angle = torch.acos((trace - 1) / 2)
        return angle.mean()


class ADDLoss(nn.Module):
    """Average Distance loss for 6D pose (for non-symmetric objects)."""
    
    def forward(
        self,
        pred_R: torch.Tensor,
        pred_t: torch.Tensor,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor,
        model_points: torch.Tensor
    ) -> torch.Tensor:
        """Compute ADD loss between predicted and ground truth poses."""
        B = pred_R.shape[0]
        points = model_points.unsqueeze(0).expand(B, -1, -1)
        
        pred_transformed = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_transformed = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        distances = torch.norm(pred_transformed - gt_transformed, dim=2)
        return distances.mean()


class ADDSLoss(nn.Module):
    """Average Distance of Symmetric objects (ADD-S) loss."""
    
    def forward(
        self,
        pred_R: torch.Tensor,
        pred_t: torch.Tensor,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor,
        model_points: torch.Tensor
    ) -> torch.Tensor:
        """
        ADD-S = mean min_j ||pred_i - gt_j||
        """
        B = pred_R.shape[0]
        points = model_points.unsqueeze(0).expand(B, -1, -1)
        
        pred_transformed = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_transformed = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        distances = torch.cdist(pred_transformed, gt_transformed)
        min_distances = distances.min(dim=2)[0]
        
        return min_distances.mean()
