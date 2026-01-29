"""Loss functions for 6D Pose Estimation with RGB-D Fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GeodesicRotationLoss(nn.Module):
    """Geodesic distance loss on SO(3) manifold for rotation matrices."""
    
    def __init__(self, eps: float = 1e-7):
        super(GeodesicRotationLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
        """Compute geodesic rotation loss (mean angle in radians)."""
        R_rel = torch.bmm(pred_R, gt_R.transpose(1, 2))
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) / 2.0, min=-1.0 + self.eps, max=1.0 - self.eps)
        angle = torch.acos(cos_angle)
        return angle.mean()


class QuaternionDistanceLoss(nn.Module):
    """Quaternion distance loss handling double cover."""
    
    def __init__(self, eps: float = 1e-7):
        super(QuaternionDistanceLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion angular distance loss (mean radians)."""
        pred_q = F.normalize(pred_q, p=2, dim=1)
        gt_q = F.normalize(gt_q, p=2, dim=1)
        
        dot = torch.abs(torch.sum(pred_q * gt_q, dim=1))
        dot = torch.clamp(dot, min=-1.0 + self.eps, max=1.0 - self.eps)
        angle = 2.0 * torch.acos(dot)
        
        return angle.mean()


class LogQuaternionLoss(nn.Module):
    """Logarithmic quaternion loss in axis-angle space."""
    
    def __init__(self, eps: float = 1e-7):
        super(LogQuaternionLoss, self).__init__()
        self.eps = eps
    
    def _quat_to_log(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to log representation (rotation vector)."""
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        v_norm = torch.sqrt(x**2 + y**2 + z**2 + self.eps)
        half_angle = torch.atan2(v_norm, w)
        
        scale = torch.where(
            v_norm > self.eps,
            2.0 * half_angle / v_norm,
            torch.ones_like(v_norm) * 2.0
        )
        
        log_q = torch.stack([x * scale, y * scale, z * scale], dim=1)
        return log_q
    
    def forward(self, pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
        """Compute log quaternion loss (mean L2 distance in log space)."""
        pred_q = F.normalize(pred_q, p=2, dim=1)
        gt_q = F.normalize(gt_q, p=2, dim=1)
        
        dot = torch.sum(pred_q * gt_q, dim=1, keepdim=True)
        gt_q = torch.where(dot < 0, -gt_q, gt_q)
        
        gt_q_inv = torch.cat([-gt_q[:, :3], gt_q[:, 3:]], dim=1)
        q_rel = self._quat_multiply(pred_q, gt_q_inv)
        
        log_rel = self._quat_to_log(q_rel)
        loss = torch.norm(log_rel, p=2, dim=1).mean()
        
        return loss
    
    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions: q1 * q2."""
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w], dim=1)


class TranslationLoss(nn.Module):
    """SmoothL1 loss for translation vectors."""
    
    def __init__(self, beta: float = 0.01):
        super(TranslationLoss, self).__init__()
        self.beta = beta  # ~1cm for meter-scale translations
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(
        self, 
        pred_t: torch.Tensor, 
        gt_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute translation loss."""
        return self.smooth_l1(pred_t, gt_t)


class TranslationResidualLoss(nn.Module):
    """Loss for translation residual with optional regularization."""
    
    def __init__(self, beta: float = 0.01, residual_weight: float = 0.0):
        super(TranslationResidualLoss, self).__init__()
        self.beta = beta
        self.residual_weight = residual_weight
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(
        self,
        T_geo: torch.Tensor,
        delta_t: torch.Tensor,
        gt_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute translation residual loss, returning loss and final translation."""
        T_final = T_geo + delta_t
        
        main_loss = self.smooth_l1(T_final, gt_t)
        
        if self.residual_weight > 0:
            residual_reg = torch.norm(delta_t, p=2, dim=1).mean()
            loss = main_loss + self.residual_weight * residual_reg
        else:
            loss = main_loss
        
        return loss, T_final


class ADDLoss(nn.Module):
    """Average Distance (ADD) loss for pose estimation."""
    
    def __init__(self, model_points: torch.Tensor):
        super(ADDLoss, self).__init__()
        self.register_buffer('model_points', model_points)
    
    def forward(
        self,
        pred_R: torch.Tensor,
        pred_t: torch.Tensor,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute ADD loss between predicted and ground truth poses."""
        B = pred_R.shape[0]
        M = self.model_points.shape[0]
        
        points = self.model_points.unsqueeze(0).expand(B, -1, -1)
        pred_transformed = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_transformed = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
        distances = torch.norm(pred_transformed - gt_transformed, p=2, dim=2)
        add = distances.mean()
        
        return add


class ADDSLoss(nn.Module):
    """ADD-S loss for symmetric objects using closest point distance."""
    
    def __init__(self, model_points: torch.Tensor):
        super(ADDSLoss, self).__init__()
        self.register_buffer('model_points', model_points)
    
    def forward(
        self,
        pred_R: torch.Tensor,
        pred_t: torch.Tensor,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute ADD-S loss using minimum point-to-point distance."""
        B = pred_R.shape[0]
        M = self.model_points.shape[0]
        
        points = self.model_points.unsqueeze(0).expand(B, -1, -1)
        
        pred_transformed = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_transformed = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        diff = pred_transformed.unsqueeze(2) - gt_transformed.unsqueeze(1)
        pairwise_dist = torch.norm(diff, p=2, dim=3)
        
        min_dist = pairwise_dist.min(dim=2)[0]
        adds = min_dist.mean()
        
        return adds


class PoseLoss(nn.Module):
    """Combined pose loss with configurable rotation, translation, and ADD components."""
    
    def __init__(
        self,
        rotation_loss_type: str = 'geodesic',
        translation_weight: float = 1.0,
        use_add: bool = False,
        symmetric: bool = False,
        model_points: Optional[torch.Tensor] = None,
        add_weight: float = 0.1
    ):
        super(PoseLoss, self).__init__()
        
        self.translation_weight = translation_weight
        self.use_add = use_add
        self.add_weight = add_weight
        
        # Rotation loss
        if rotation_loss_type == 'geodesic':
            self.rotation_loss = GeodesicRotationLoss()
        elif rotation_loss_type == 'quaternion':
            self.rotation_loss = QuaternionDistanceLoss()
        elif rotation_loss_type == 'log':
            self.rotation_loss = LogQuaternionLoss()
        else:
            raise ValueError(f"Unknown rotation loss: {rotation_loss_type}")
        
        self.rotation_loss_type = rotation_loss_type
        
        self.translation_loss = TranslationResidualLoss(beta=0.01)
        
        if use_add and model_points is not None:
            if symmetric:
                self.add_loss = ADDSLoss(model_points)
            else:
                self.add_loss = ADDLoss(model_points)
        else:
            self.add_loss = None
    
    def forward(
        self,
        pred_q: torch.Tensor,
        pred_R: torch.Tensor,
        T_geo: torch.Tensor,
        delta_t: torch.Tensor,
        gt_R: torch.Tensor,
        gt_t: torch.Tensor
    ) -> dict:
        """Compute combined pose loss with rotation and translation components."""
        if self.rotation_loss_type == 'geodesic':
            loss_rot = self.rotation_loss(pred_R, gt_R)
        else:
            from utils.geometry import rotation_matrix_to_quaternion_torch
            gt_q = rotation_matrix_to_quaternion_torch(gt_R)
            loss_rot = self.rotation_loss(pred_q, gt_q)
        
        loss_trans, T_final = self.translation_loss(T_geo, delta_t, gt_t)
        
        total_loss = loss_rot + self.translation_weight * loss_trans
        
        if self.add_loss is not None:
            loss_add = self.add_loss(pred_R, T_final, gt_R, gt_t)
            total_loss = total_loss + self.add_weight * loss_add
        else:
            loss_add = torch.tensor(0.0, device=pred_q.device)
        
        return {
            'total': total_loss,
            'rotation': loss_rot,
            'translation': loss_trans,
            'add': loss_add,
            'T_final': T_final,
            'delta_t_norm': torch.norm(delta_t, p=2, dim=1).mean()
        }
