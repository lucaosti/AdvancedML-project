"""Shared utilities for RGB-D pose estimation pipelines."""

from rgbd_pipelines.common.geometric_anchor import (
    get_geometric_anchor,
    get_geometric_anchor_batch,
    get_geometric_anchor_torch,
    filter_depth_outliers,
    compute_robust_depth
)
from rgbd_pipelines.common.losses import (
    GeodesicLoss,
    ADDLoss,
    ADDSLoss
)

__all__ = [
    'get_geometric_anchor',
    'get_geometric_anchor_batch',
    'get_geometric_anchor_torch',
    'filter_depth_outliers',
    'compute_robust_depth',
    'GeodesicLoss',
    'ADDLoss',
    'ADDSLoss',
]
