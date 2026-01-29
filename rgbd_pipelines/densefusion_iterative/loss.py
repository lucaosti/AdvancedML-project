"""
Loss functions for DenseFusion Iterative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DenseFusionIterativeLoss(nn.Module):
    """Combined loss for DenseFusion Iterative with ADD metric."""
    
    def __init__(self, refine_weight=1.0, confidence_margin=0.01):
        super().__init__()
        self.refine_weight = refine_weight
        self.confidence_margin = confidence_margin
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_rotation: torch.Tensor,
        gt_translation: torch.Tensor,
        model_points: torch.Tensor,
        points: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # Initial loss from final refined pose
        pred_R = pred_dict['rotation']
        pred_t = pred_dict['translation']
        
        initial_loss = self._add_loss(pred_R, pred_t, gt_rotation, gt_translation, model_points)
        
        # Refinement losses
        refinement_loss = torch.tensor(0.0, device=pred_R.device)
        refined_Rs = pred_dict.get('refined_rotations', [])
        refined_ts = pred_dict.get('refined_translations', [])
        
        if len(refined_Rs) > 1:
            for i, (R, t) in enumerate(zip(refined_Rs[1:], refined_ts[1:])):
                ref_loss = self._add_loss(R, t, gt_rotation, gt_translation, model_points)
                weight = (i + 1) / len(refined_Rs[1:])
                refinement_loss = refinement_loss + weight * ref_loss
        
        total = initial_loss + self.refine_weight * refinement_loss
        
        return {
            'total': total,
            'initial': initial_loss,
            'refinement': refinement_loss
        }
    
    @staticmethod
    def _add_loss(pred_R, pred_t, gt_R, gt_t, model_points):
        """ADD (Average Distance) loss."""
        B = pred_R.shape[0]
        M = model_points.shape[0]
        
        points = model_points.unsqueeze(0).expand(B, -1, -1)
        
        pred_transformed = torch.bmm(points, pred_R.transpose(1, 2)) + pred_t.unsqueeze(1)
        gt_transformed = torch.bmm(points, gt_R.transpose(1, 2)) + gt_t.unsqueeze(1)
        
        distances = torch.norm(pred_transformed - gt_transformed, dim=2)
        return distances.mean()
