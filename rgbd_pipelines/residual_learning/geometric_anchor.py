"""Geometric anchor strategy for robust translation estimation via Pinhole Camera Model."""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def shrink_bbox(
    bbox: Union[np.ndarray, torch.Tensor],
    shrink_factor: float = 0.15
) -> Union[np.ndarray, torch.Tensor]:
    """Shrink bounding box symmetrically to avoid background pixels."""
    is_torch = isinstance(bbox, torch.Tensor)
    
    if is_torch:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        margin_x = w * shrink_factor / 2
        margin_y = h * shrink_factor / 2
        new_x = x + margin_x
        new_y = y + margin_y
        new_w = w * (1 - shrink_factor)
        new_h = h * (1 - shrink_factor)
        return torch.stack([new_x, new_y, new_w, new_h])
    else:
        x, y, w, h = bbox
        margin_x = w * shrink_factor / 2
        margin_y = h * shrink_factor / 2
        new_x = x + margin_x
        new_y = y + margin_y
        new_w = w * (1 - shrink_factor)
        new_h = h * (1 - shrink_factor)
        return np.array([new_x, new_y, new_w, new_h], dtype=bbox.dtype)


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
    filtered = filtered[iqr_mask]
    
    return filtered


def compute_robust_depth(
    depth_region: np.ndarray,
    percentile: float = 20.0,
    min_depth: float = 0.1,
    max_depth: float = 3.0
) -> float:
    """Compute robust depth using percentile-based filtering for front surface estimation."""
    depths = depth_region.flatten()
    filtered_depths = filter_depth_outliers(depths, min_depth=min_depth, max_depth=max_depth)
    
    if len(filtered_depths) == 0:
        non_zero = depths[depths > 0]
        if len(non_zero) > 0:
            return float(np.median(non_zero))
        else:
            return 1.0
    
    robust_depth = np.percentile(filtered_depths, percentile)
    return float(robust_depth)


def get_geometric_anchor(
    bbox: Union[np.ndarray, torch.Tensor],
    depth_map: Union[np.ndarray, torch.Tensor],
    intrinsics: Union[np.ndarray, torch.Tensor],
    shrink_factor: float = 0.15,
    depth_percentile: float = 20.0,
    return_debug: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Compute geometric translation anchor T_geo using Pinhole Camera Model."""
    is_torch = isinstance(bbox, torch.Tensor)
    
    if is_torch:
        bbox_np = bbox.cpu().numpy()
        depth_np = depth_map.cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
        K_np = intrinsics.cpu().numpy() if isinstance(intrinsics, torch.Tensor) else intrinsics
    else:
        bbox_np = np.array(bbox)
        depth_np = np.array(depth_map)
        K_np = np.array(intrinsics)
    
    shrunk_bbox = shrink_bbox(bbox_np, shrink_factor)
    x, y, w, h = shrunk_bbox.astype(int)
    
    H, W = depth_np.shape[:2]
    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    
    depth_region = depth_np[y:y2, x:x2]
    
    Z = compute_robust_depth(
        depth_region,
        percentile=depth_percentile
    )
    
    original_x, original_y, original_w, original_h = bbox_np
    center_u = original_x + original_w / 2.0
    center_v = original_y + original_h / 2.0
    
    fx = K_np[0, 0]
    fy = K_np[1, 1]
    cx = K_np[0, 2]
    cy = K_np[1, 2]
    
    X = (center_u - cx) * Z / fx
    Y = (center_v - cy) * Z / fy
    
    T_geo = np.array([X, Y, Z], dtype=np.float32)
    
    if return_debug:
        debug_info = {
            'shrunk_bbox': shrunk_bbox,
            'depth_region_shape': depth_region.shape,
            'robust_depth': Z,
            'bbox_center': (center_u, center_v),
            'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        }
        return T_geo, debug_info
    
    return T_geo


def get_geometric_anchor_batch(
    bboxes: torch.Tensor,
    depth_crops: list,
    intrinsics: torch.Tensor,
    crop_offsets: torch.Tensor,
    shrink_factor: float = 0.15,
    depth_percentile: float = 20.0
) -> torch.Tensor:
    """Compute geometric anchors for a batch of samples."""
    batch_size = bboxes.shape[0]
    device = bboxes.device
    
    T_geo_list = []
    
    for i in range(batch_size):
        bbox = bboxes[i].cpu().numpy()
        depth_crop = depth_crops[i].cpu().numpy()
        K = intrinsics[i].cpu().numpy()
        
        shrunk_crop = _shrink_depth_crop(depth_crop, shrink_factor)
        Z = compute_robust_depth(shrunk_crop, percentile=depth_percentile)
        
        x, y, w, h = bbox
        center_u = x + w / 2.0
        center_v = y + h / 2.0
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        X = (center_u - cx) * Z / fx
        Y = (center_v - cy) * Z / fy
        
        T_geo_list.append([X, Y, Z])
    
    T_geo = torch.tensor(T_geo_list, dtype=torch.float32, device=device)
    
    return T_geo


def _shrink_depth_crop(depth_crop: np.ndarray, shrink_factor: float) -> np.ndarray:
    """Apply shrink to a depth crop by extracting the central region."""
    H, W = depth_crop.shape
    
    margin_y = int(H * shrink_factor / 2)
    margin_x = int(W * shrink_factor / 2)
    
    y1 = margin_y
    y2 = H - margin_y
    x1 = margin_x
    x2 = W - margin_x
    
    if y2 <= y1 or x2 <= x1:
        return depth_crop
    
    return depth_crop[y1:y2, x1:x2]


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
    
    B = bbox.shape[0]
    
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    center_u = x + w / 2.0
    center_v = y + h / 2.0
    
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    
    Z = depth.view(-1) if depth.dim() > 1 else depth
    
    X = (center_u - cx) * Z / fx
    Y = (center_v - cy) * Z / fy
    
    T_geo = torch.stack([X, Y, Z], dim=1)
    
    if squeeze_output:
        T_geo = T_geo.squeeze(0)
    
    return T_geo
