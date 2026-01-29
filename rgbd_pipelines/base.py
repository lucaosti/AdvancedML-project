"""
Abstract base class for RGB-D pose estimation models.

Defines the common interface that all pose models must implement:
forward() for inference and get_loss() for training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BasePoseModel(nn.Module, ABC):
    """
    Base class for pose estimation models.
    Subclasses must implement forward() and get_loss() methods.
    """
    
    def __init__(self, num_points: int = 1024, num_obj: int = 13):
        super(BasePoseModel, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.model_name = "base"
    
    @abstractmethod
    def forward(
        self,
        rgb: torch.Tensor,
        points: torch.Tensor,
        choose: torch.Tensor,
        obj_idx: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. Must return dict with 'rotation' [B,3,3] and 'translation' [B,3].
        """
        pass
    
    @abstractmethod
    def get_loss(
        self,
        pred_dict: Dict[str, torch.Tensor],
        gt_rotation: torch.Tensor,
        gt_translation: torch.Tensor,
        model_points: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss. Must return dict with 'total' key.
        """
        pass
    
    @torch.no_grad()
    def predict_pose(
        self,
        rgb: torch.Tensor,
        points: torch.Tensor,
        choose: torch.Tensor,
        obj_idx: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference wrapper returning rotation and translation."""
        self.eval()
        output = self.forward(rgb, points, choose, obj_idx, **kwargs)
        return output['rotation'], output['translation']
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Return trainable parameter count."""
        return {
            'total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters. Override in subclass."""
        pass
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters. Override in subclass."""
        pass
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"num_points={self.num_points}, "
            f"num_obj={self.num_obj}, "
            f"params={self.get_num_parameters()['total']:,})"
        )
