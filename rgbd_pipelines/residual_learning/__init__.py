"""RGB-D pose estimator with DenseFusion-style architecture and geometric anchor residual learning."""

from rgbd_pipelines.residual_learning.fusion_network import (
    RGBDFusionNetwork,
    RGBDFusionNetworkLite,
    PointNetEncoder,
    RGBEncoder,
    create_rgbd_fusion_model
)

from utils.dataset import (
    LineMODDataset,
    rgbd_collate_fn,
)

from rgbd_pipelines.residual_learning.geometric_anchor import (
    get_geometric_anchor,
    get_geometric_anchor_batch,
    get_geometric_anchor_torch,
    shrink_bbox,
    compute_robust_depth,
    filter_depth_outliers
)

from rgbd_pipelines.residual_learning.fusion_loss import (
    GeodesicRotationLoss,
    QuaternionDistanceLoss,
    LogQuaternionLoss,
    TranslationLoss,
    TranslationResidualLoss,
    ADDLoss,
    ADDSLoss,
    PoseLoss
)

from rgbd_pipelines.residual_learning.depth_model import DepthEncoder
from rgbd_pipelines.residual_learning.fusion_model import RGBDPoseEstimator

from rgbd_pipelines.base import BasePoseModel


class ResidualLearningModel(BasePoseModel):
    """Residual Learning model wrapper implementing BasePoseModel interface."""
    
    def __init__(self, num_points: int = 500, num_obj: int = 13, pretrained: bool = True):
        super().__init__(num_points, num_obj)
        self.model_name = "residual_learning"
        
        self.network = create_rgbd_fusion_model(
            variant='standard',
            num_points=num_points,
            pretrained=pretrained
        )
        
        self.rot_criterion = GeodesicRotationLoss()
        self.trans_criterion = TranslationLoss()
    
    def forward(self, rgb, points, choose, obj_idx, T_geo=None, R_geo=None, **kwargs):
        """Forward pass returning rotation matrix and translation vector."""
        from utils.geometry import quaternion_to_rotation_matrix_torch
        
        if T_geo is None:
            T_geo = points.mean(dim=1)
        
        quaternion, delta_t, T_final = self.network(rgb, points, T_geo)
        R = quaternion_to_rotation_matrix_torch(quaternion)
        
        return {
            'rotation': R,
            'translation': T_final,
            'quaternion': quaternion,
            'delta_translation': delta_t,
            'T_geo': T_geo
        }
    
    def get_loss(self, pred_dict, gt_rotation, gt_translation, model_points=None, **kwargs):
        """Compute training loss combining rotation and translation components."""
        pred_R = pred_dict['rotation']
        pred_t = pred_dict['translation']
        
        loss_rot = self.rot_criterion(pred_R, gt_rotation)
        loss_trans = self.trans_criterion(pred_t, gt_translation)
        
        total_loss = loss_rot + loss_trans
        
        return {
            'total': total_loss,
            'rotation': loss_rot,
            'translation': loss_trans
        }

__all__ = [
    'RGBDFusionNetwork',
    'RGBDFusionNetworkLite',
    'PointNetEncoder',
    'RGBEncoder',
    'create_rgbd_fusion_model',
    'LineMODDataset',
    'rgbd_collate_fn',
    'get_geometric_anchor',
    'get_geometric_anchor_batch',
    'get_geometric_anchor_torch',
    'shrink_bbox',
    'compute_robust_depth',
    'filter_depth_outliers',
    'GeodesicRotationLoss',
    'QuaternionDistanceLoss',
    'LogQuaternionLoss',
    'TranslationLoss',
    'TranslationResidualLoss',
    'ADDLoss',
    'ADDSLoss',
    'PoseLoss',
    'DepthEncoder',
    'RGBDPoseEstimator',
    'ResidualLearningModel',
]
