"""
Loss functions for FFB6D (same as PVN3D - keypoint voting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FFB6DLoss(nn.Module):
    """Keypoint voting loss for FFB6D."""
    
    def __init__(self, num_kp=8, w_kp=1.0, w_ctr=1.0, w_sem=0.1):
        super().__init__()
        self.num_kp = num_kp
        self.w_kp = w_kp
        self.w_ctr = w_ctr
        self.w_sem = w_sem
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        kp_targ: torch.Tensor = None,
        ctr_targ: torch.Tensor = None,
        sem_targ: torch.Tensor = None,
        points: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        kp_offsets = pred_dict['kp_offsets']
        ctr_offsets = pred_dict['ctr_offsets']
        sem_logits = pred_dict.get('sem_logits')
        
        B, N, K, _ = kp_offsets.shape
        
        if kp_targ is not None and points is not None:
            kp_targ_exp = kp_targ.unsqueeze(1).expand(-1, N, -1, -1)
            points_exp = points.unsqueeze(2).expand(-1, -1, K, -1)
            gt_kp_offsets = kp_targ_exp - points_exp
        else:
            gt_kp_offsets = torch.zeros_like(kp_offsets)
        
        if ctr_targ is not None and points is not None:
            ctr_targ_exp = ctr_targ.unsqueeze(1).expand(-1, N, -1)
            gt_ctr_offsets = ctr_targ_exp - points
        else:
            gt_ctr_offsets = torch.zeros_like(ctr_offsets)
        
        kp_loss = F.smooth_l1_loss(kp_offsets, gt_kp_offsets)
        ctr_loss = F.smooth_l1_loss(ctr_offsets, gt_ctr_offsets)
        
        sem_loss = torch.tensor(0.0, device=kp_offsets.device)
        if sem_logits is not None and sem_targ is not None:
            sem_logits_flat = sem_logits.view(-1, sem_logits.shape[-1])
            sem_targ_flat = sem_targ.view(-1)
            sem_loss = F.cross_entropy(sem_logits_flat, sem_targ_flat)
        
        total = self.w_kp * kp_loss + self.w_ctr * ctr_loss + self.w_sem * sem_loss
        
        return {
            'total': total,
            'kp': kp_loss,
            'ctr': ctr_loss,
            'sem': sem_loss
        }
