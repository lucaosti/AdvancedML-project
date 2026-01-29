"""Pinhole Camera Model for 3D translation estimation from 2D bounding boxes."""

import numpy as np
import torch
from typing import Union, Tuple
from utils.bbox import get_bbox_center


def estimate_translation_from_bbox(
    bbox: Union[np.ndarray, Tuple[float, float, float, float]],
    depth: Union[float, np.ndarray],
    intrinsics: Union[np.ndarray, torch.Tensor],
    object_size: Union[np.ndarray, Tuple[float, float, float]] = None,
    format: str = 'xywh'
) -> np.ndarray:
    """Estimate 3D translation from 2D bbox using Pinhole Camera Model."""
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox, dtype=np.float32)
    
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    
    if isinstance(depth, (list, np.ndarray, torch.Tensor)):
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()
        depth = float(np.mean(depth))
    
    f_x = intrinsics[0, 0]
    f_y = intrinsics[1, 1]
    c_x = intrinsics[0, 2]
    c_y = intrinsics[1, 2]
    
    center = get_bbox_center(bbox, format=format)
    center_x = float(center[0])
    center_y = float(center[1])
    
    Z = depth
    X = (center_x - c_x) * Z / f_x
    Y = (center_y - c_y) * Z / f_y
    
    translation = np.array([X, Y, Z], dtype=np.float32)
    return translation


def estimate_translation_from_bbox_torch(
    bbox: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    object_size: torch.Tensor = None
) -> torch.Tensor:
    """Batched Pinhole Camera Model translation estimation (PyTorch)."""
    device = bbox.device
    batch_size = bbox.shape[0]
    
    if depth.dim() == 1:
        depth = depth.unsqueeze(1)
    elif depth.dim() > 2:
        depth = depth.mean(dim=tuple(range(1, depth.dim()))).unsqueeze(1)
    
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    
    f_x = intrinsics[:, 0, 0]  
    f_y = intrinsics[:, 1, 1]  
    c_x = intrinsics[:, 0, 2]  
    c_y = intrinsics[:, 1, 2]  
    
    x, y, param3, param4 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    
    is_xyxy = (param3 > x).float() * (param4 > y).float()
    
    center_x_xyxy = (x + param3) / 2.0
    center_y_xyxy = (y + param4) / 2.0
    center_x_xywh = x + param3 / 2.0
    center_y_xywh = y + param4 / 2.0
    
    center_x = is_xyxy * center_x_xyxy + (1 - is_xyxy) * center_x_xywh
    center_y = is_xyxy * center_y_xyxy + (1 - is_xyxy) * center_y_xywh
    
    Z = depth.squeeze(-1) if depth.dim() > 1 else depth
    if Z.dim() == 0:
        Z = Z.unsqueeze(0)
    
    X = (center_x - c_x) * Z / f_x
    Y = (center_y - c_y) * Z / f_y
    
    translation = torch.stack([X, Y, Z], dim=1)
    return translation


def project_3d_to_2d(
    points_3d: Union[np.ndarray, torch.Tensor],
    intrinsics: Union[np.ndarray, torch.Tensor],
    translation: Union[np.ndarray, torch.Tensor] = None,
    rotation: Union[np.ndarray, torch.Tensor] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Project 3D points to 2D image coordinates using Pinhole Camera Model."""
    is_torch = isinstance(points_3d, torch.Tensor)
    
    if is_torch:
        return _project_3d_to_2d_torch(points_3d, intrinsics, translation, rotation)
    else:
        return _project_3d_to_2d_numpy(points_3d, intrinsics, translation, rotation)


def _project_3d_to_2d_numpy(points_3d, intrinsics, translation, rotation):
    """NumPy implementation of 3D to 2D projection."""
    if rotation is not None or translation is not None:
        if rotation is None:
            rotation = np.eye(3)
        if translation is None:
            translation = np.zeros(3)
        points_3d = (rotation @ points_3d.T).T + translation
    
    points_2d_homogeneous = (intrinsics @ points_3d.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:3]
    return points_2d


def _project_3d_to_2d_torch(points_3d, intrinsics, translation, rotation):
    """PyTorch implementation of 3D to 2D projection."""
    if rotation is not None or translation is not None:
        if rotation is None:
            rotation = torch.eye(3, device=points_3d.device)
        if translation is None:
            translation = torch.zeros(3, device=points_3d.device)
        
        if points_3d.dim() == 3:
            if rotation.dim() == 2:
                rotation = rotation.unsqueeze(0)
            if translation.dim() == 1:
                translation = translation.unsqueeze(0).unsqueeze(1)
            elif translation.dim() == 2:
                translation = translation.unsqueeze(1)
            points_3d = torch.bmm(points_3d, rotation.transpose(1, 2)) + translation
        else:
            points_3d = (rotation @ points_3d.T).T + translation
    
    if points_3d.dim() == 3:
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
        points_2d_homogeneous = torch.bmm(points_3d, intrinsics.transpose(1, 2))
    else:
        points_2d_homogeneous = (intrinsics @ points_3d.T).T
    
    points_2d = points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3]
    return points_2d


def compute_depth_from_bbox(
    depth_map: Union[np.ndarray, torch.Tensor],
    bbox: Union[np.ndarray, Tuple[float, float, float, float]],
    method: str = 'median'
) -> float:
    """Extract depth value from depth map within bounding box region."""
    is_torch = isinstance(depth_map, torch.Tensor)
    
    # Handle shape
    if is_torch:
        if depth_map.dim() == 3:
            depth_map = depth_map.squeeze(0)
    else:
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(0)
    
    # Parse bbox
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox) if not is_torch else torch.tensor(bbox)
    
    x, y, param3, param4 = bbox if not is_torch else bbox.tolist()
    
    # Detect format
    if param3 > x and param4 > y:
        x1, y1, x2, y2 = int(x), int(y), int(param3), int(param4)
    else:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + param3), int(y + param4)
    
    # Crop depth region
    depth_crop = depth_map[y1:y2, x1:x2]
    
    # Aggregate
    if method == 'median':
        if is_torch:
            depth = torch.median(depth_crop[depth_crop > 0])
        else:
            valid_depths = depth_crop[depth_crop > 0]
            depth = np.median(valid_depths) if len(valid_depths) > 0 else 0.0
    elif method == 'mean':
        if is_torch:
            depth = torch.mean(depth_crop[depth_crop > 0])
        else:
            valid_depths = depth_crop[depth_crop > 0]
            depth = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0
    elif method == 'center':
        center_y = (y1 + y2) // 2
        center_x = (x1 + x2) // 2
        depth = depth_map[center_y, center_x]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median', 'mean', or 'center'")
    
    return float(depth) if is_torch else depth


def estimate_depth_with_rotation(bbox_h, f_y, model_points_3d, predicted_rotation_matrix):
    """Estimate depth using bbox height and object rotation."""
    rotated_points = model_points_3d @ predicted_rotation_matrix.T
    max_y = np.max(rotated_points[:, 1])
    min_y = np.min(rotated_points[:, 1])
    current_physical_height = max_y - min_y
    depth = (f_y * current_physical_height) / bbox_h
    return depth