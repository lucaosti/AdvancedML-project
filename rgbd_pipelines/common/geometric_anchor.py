"""Geometric anchor estimation for translation initialization via Pinhole Camera Model."""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def filter_depth_outliers(
    depths: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    iqr_factor: float = 1.5
) -> np.ndarray:
    """Filter invalid and outlier depth values using range and IQR filtering."""
    if len(depths) == 0:
        return depths
    
    valid_mask = (depths > min_depth) & (depths < max_depth)
    filtered = depths[valid_mask]
    
    if len(filtered) < 3:
        return filtered
    
    q1 = np.percentile(filtered, 25)
    q3 = np.percentile(filtered, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    
    iqr_mask = (filtered >= lower_bound) & (filtered <= upper_bound)
    
    return filtered[iqr_mask]


def compute_robust_depth(
    depth_region: np.ndarray,
    percentile: float = 20.0,
    min_depth: float = 0.1,
    max_depth: float = 3.0
) -> float:
    """Compute robust depth estimate using percentile-based filtering."""
    depths = depth_region.flatten()
    filtered_depths = filter_depth_outliers(depths, min_depth, max_depth)
    
    if len(filtered_depths) == 0:
        non_zero = depths[depths > 0]
        if len(non_zero) > 0:
            return float(np.median(non_zero))
        return 1.0
    
    return float(np.percentile(filtered_depths, percentile))


def get_geometric_anchor(
    bbox: Union[np.ndarray, Tuple[float, float, float, float]],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    shrink_ratio: float = 0.85,
    depth_percentile: float = 20.0
) -> np.ndarray:
    """Estimate translation from bounding box and depth using Pinhole Camera Model."""
    x, y, w, h = bbox
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    cx_bbox = x + w / 2
    cy_bbox = y + h / 2
    
    w_shrink = w * shrink_ratio
    h_shrink = h * shrink_ratio
    
    H, W = depth.shape
    x1 = max(0, int(cx_bbox - w_shrink / 2))
    y1 = max(0, int(cy_bbox - h_shrink / 2))
    x2 = min(W, int(cx_bbox + w_shrink / 2))
    y2 = min(H, int(cy_bbox + h_shrink / 2))
    
    depth_roi = depth[y1:y2, x1:x2]
    Z = compute_robust_depth(depth_roi, percentile=depth_percentile)
    
    X = (cx_bbox - cx) * Z / fx
    Y = (cy_bbox - cy) * Z / fy
    
    return np.array([X, Y, Z], dtype=np.float32)


def get_geometric_anchor_batch(
    bboxes: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    shrink_ratio: float = 0.85
) -> torch.Tensor:
    """Batch version of geometric anchor estimation (PyTorch)."""
    B = bboxes.shape[0]
    device = bboxes.device
    dtype = bboxes.dtype
    
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
    
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    
    cx_bbox = bboxes[:, 0] + bboxes[:, 2] / 2
    cy_bbox = bboxes[:, 1] + bboxes[:, 3] / 2
    
    H, W = depths.shape[1], depths.shape[2]
    
    iu = cx_bbox.long().clamp(0, W - 1)
    iv = cy_bbox.long().clamp(0, H - 1)
    
    Z = depths[torch.arange(B, device=device), iv, iu]
    Z = torch.where(Z > 0, Z, torch.ones_like(Z))
    
    X = (cx_bbox - cx) * Z / fx
    Y = (cy_bbox - cy) * Z / fy
    
    return torch.stack([X, Y, Z], dim=1)


def get_geometric_anchor_torch(
    bbox: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """PyTorch-native geometric anchor computation (differentiable)."""
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)
        depth = depth.unsqueeze(0) if depth.dim() == 0 else depth
        intrinsics = intrinsics.unsqueeze(0) if intrinsics.dim() == 2 else intrinsics
        squeeze_output = True
    else:
        squeeze_output = False
    
    cx_bbox = bbox[:, 0] + bbox[:, 2] / 2.0
    cy_bbox = bbox[:, 1] + bbox[:, 3] / 2.0
    
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    
    Z = depth.view(-1) if depth.dim() > 1 else depth
    
    X = (cx_bbox - cx) * Z / fx
    Y = (cy_bbox - cy) * Z / fy
    
    T_geo = torch.stack([X, Y, Z], dim=1)
    
    if squeeze_output:
        T_geo = T_geo.squeeze(0)
    
    return T_geo
