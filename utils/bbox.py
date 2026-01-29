"""Bounding box format conversion utilities."""

import numpy as np
import torch
from typing import Union, Tuple


def xywh_to_xyxy(bbox: Union[np.ndarray, torch.Tensor, Tuple, list]) -> Union[np.ndarray, torch.Tensor]:
    """Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format."""
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox, dtype=np.float32)
        was_list = True
    else:
        was_list = False
    
    is_torch = isinstance(bbox, torch.Tensor)
    
    if bbox.ndim == 1:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2 = x + w
        y2 = y + h
        
        if is_torch:
            bbox_xyxy = torch.stack([x, y, x2, y2])
        else:
            bbox_xyxy = np.array([x, y, x2, y2], dtype=bbox.dtype)
    else:
        x = bbox[:, 0]
        y = bbox[:, 1]
        w = bbox[:, 2]
        h = bbox[:, 3]
        x2 = x + w
        y2 = y + h
        
        if is_torch:
            bbox_xyxy = torch.stack([x, y, x2, y2], dim=1)
        else:
            bbox_xyxy = np.stack([x, y, x2, y2], axis=1).astype(bbox.dtype)
    
    return bbox_xyxy


def xyxy_to_xywh(bbox: Union[np.ndarray, torch.Tensor, Tuple, list]) -> Union[np.ndarray, torch.Tensor]:
    """Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h) format."""
    if isinstance(bbox, (list, tuple)):
        bbox = np.array(bbox, dtype=np.float32)
        was_list = True
    else:
        was_list = False
    
    is_torch = isinstance(bbox, torch.Tensor)
    
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = x2 - x1
        h = y2 - y1
        
        if is_torch:
            bbox_xywh = torch.stack([x1, y1, w, h])
        else:
            bbox_xywh = np.array([x1, y1, w, h], dtype=bbox.dtype)
    else:
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        w = x2 - x1
        h = y2 - y1
        
        if is_torch:
            bbox_xywh = torch.stack([x1, y1, w, h], dim=1)
        else:
            bbox_xywh = np.stack([x1, y1, w, h], axis=1).astype(bbox.dtype)
    
    return bbox_xywh


def normalize_bbox(
    bbox: Union[np.ndarray, torch.Tensor], 
    image_width: float, 
    image_height: float,
    format: str = 'xywh'
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize bounding box coordinates to [0, 1] range."""
    is_torch = isinstance(bbox, torch.Tensor)
    
    if bbox.ndim == 1:
        bbox_norm = bbox.clone().float() if is_torch else bbox.astype(np.float32)
        if format == 'xywh':
            bbox_norm[0] = bbox[0] / image_width
            bbox_norm[1] = bbox[1] / image_height
            bbox_norm[2] = bbox[2] / image_width
            bbox_norm[3] = bbox[3] / image_height
        elif format == 'xyxy':
            bbox_norm[0] = bbox[0] / image_width
            bbox_norm[1] = bbox[1] / image_height
            bbox_norm[2] = bbox[2] / image_width
            bbox_norm[3] = bbox[3] / image_height
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    else:
        bbox_norm = bbox.clone().float() if is_torch else bbox.astype(np.float32)
        if format == 'xywh':
            bbox_norm[:, 0] = bbox[:, 0] / image_width
            bbox_norm[:, 1] = bbox[:, 1] / image_height
            bbox_norm[:, 2] = bbox[:, 2] / image_width
            bbox_norm[:, 3] = bbox[:, 3] / image_height
        elif format == 'xyxy':
            bbox_norm[:, 0] = bbox[:, 0] / image_width
            bbox_norm[:, 1] = bbox[:, 1] / image_height
            bbox_norm[:, 2] = bbox[:, 2] / image_width
            bbox_norm[:, 3] = bbox[:, 3] / image_height
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    
    return bbox_norm


def denormalize_bbox(
    bbox: Union[np.ndarray, torch.Tensor], 
    image_width: float, 
    image_height: float,
    format: str = 'xywh'
) -> Union[np.ndarray, torch.Tensor]:
    """Denormalize bounding box coordinates from [0, 1] to pixel values."""
    is_torch = isinstance(bbox, torch.Tensor)
    
    if bbox.ndim == 1:
        bbox_pixels = bbox.clone() if is_torch else bbox.copy()
        if format == 'xywh':
            bbox_pixels[0] = bbox[0] * image_width   # x
            bbox_pixels[1] = bbox[1] * image_height  # y
            bbox_pixels[2] = bbox[2] * image_width   # w
            bbox_pixels[3] = bbox[3] * image_height  # h
        elif format == 'xyxy':
            bbox_pixels[0] = bbox[0] * image_width   # x1
            bbox_pixels[1] = bbox[1] * image_height  # y1
            bbox_pixels[2] = bbox[2] * image_width   # x2
            bbox_pixels[3] = bbox[3] * image_height  # y2
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    else:
        bbox_pixels = bbox.clone() if is_torch else bbox.copy()
        if format == 'xywh':
            bbox_pixels[:, 0] = bbox[:, 0] * image_width
            bbox_pixels[:, 1] = bbox[:, 1] * image_height
            bbox_pixels[:, 2] = bbox[:, 2] * image_width
            bbox_pixels[:, 3] = bbox[:, 3] * image_height
        elif format == 'xyxy':
            bbox_pixels[:, 0] = bbox[:, 0] * image_width
            bbox_pixels[:, 1] = bbox[:, 1] * image_height
            bbox_pixels[:, 2] = bbox[:, 2] * image_width
            bbox_pixels[:, 3] = bbox[:, 3] * image_height
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    
    return bbox_pixels


def get_bbox_center(bbox: Union[np.ndarray, torch.Tensor], format: str = 'xywh') -> Union[np.ndarray, torch.Tensor]:
    """Calculate the center point of a bounding box."""
    is_torch = isinstance(bbox, torch.Tensor)
    
    if bbox.ndim == 1:
        if format == 'xywh':
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
        elif format == 'xyxy':
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
        
        if is_torch:
            center = torch.stack([cx, cy])
        else:
            center = np.array([cx, cy], dtype=bbox.dtype)
    else:
        if format == 'xywh':
            cx = bbox[:, 0] + bbox[:, 2] / 2.0
            cy = bbox[:, 1] + bbox[:, 3] / 2.0
        elif format == 'xyxy':
            cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
            cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
        
        if is_torch:
            center = torch.stack([cx, cy], dim=1)
        else:
            center = np.stack([cx, cy], axis=1).astype(bbox.dtype)
    
    return center


def get_bbox_area(bbox: Union[np.ndarray, torch.Tensor], format: str = 'xywh') -> Union[float, np.ndarray, torch.Tensor]:
    """Calculate the area of a bounding box."""
    if bbox.ndim == 1:
        if format == 'xywh':
            area = bbox[2] * bbox[3]
        elif format == 'xyxy':
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
        
        if isinstance(bbox, torch.Tensor):
            return area.item()
        else:
            return float(area)
    else:
        if format == 'xywh':
            area = bbox[:, 2] * bbox[:, 3]
        elif format == 'xyxy':
            area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        else:
            raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
        
        return area


def clip_bbox(
    bbox: Union[np.ndarray, torch.Tensor],
    image_width: float,
    image_height: float,
    format: str = 'xywh'
) -> Union[np.ndarray, torch.Tensor]:
    """Clip bounding box coordinates to image boundaries."""
    is_torch = isinstance(bbox, torch.Tensor)
    bbox_clipped = bbox.clone() if is_torch else bbox.copy()
    
    if format == 'xywh':
        bbox_xyxy = xywh_to_xyxy(bbox)
        
        if bbox_xyxy.ndim == 1:
            if is_torch:
                bbox_xyxy[0] = torch.clamp(bbox_xyxy[0], 0, image_width)
                bbox_xyxy[1] = torch.clamp(bbox_xyxy[1], 0, image_height)
                bbox_xyxy[2] = torch.clamp(bbox_xyxy[2], 0, image_width)
                bbox_xyxy[3] = torch.clamp(bbox_xyxy[3], 0, image_height)
            else:
                bbox_xyxy[0] = np.clip(bbox_xyxy[0], 0, image_width)
                bbox_xyxy[1] = np.clip(bbox_xyxy[1], 0, image_height)
                bbox_xyxy[2] = np.clip(bbox_xyxy[2], 0, image_width)
                bbox_xyxy[3] = np.clip(bbox_xyxy[3], 0, image_height)
        else:
            if is_torch:
                bbox_xyxy[:, 0] = torch.clamp(bbox_xyxy[:, 0], 0, image_width)
                bbox_xyxy[:, 1] = torch.clamp(bbox_xyxy[:, 1], 0, image_height)
                bbox_xyxy[:, 2] = torch.clamp(bbox_xyxy[:, 2], 0, image_width)
                bbox_xyxy[:, 3] = torch.clamp(bbox_xyxy[:, 3], 0, image_height)
            else:
                bbox_xyxy[:, 0] = np.clip(bbox_xyxy[:, 0], 0, image_width)
                bbox_xyxy[:, 1] = np.clip(bbox_xyxy[:, 1], 0, image_height)
                bbox_xyxy[:, 2] = np.clip(bbox_xyxy[:, 2], 0, image_width)
                bbox_xyxy[:, 3] = np.clip(bbox_xyxy[:, 3], 0, image_height)
        
        bbox_clipped = xyxy_to_xywh(bbox_xyxy)
        
    elif format == 'xyxy':
        if bbox.ndim == 1:
            if is_torch:
                bbox_clipped[0] = torch.clamp(bbox[0], 0, image_width)
                bbox_clipped[1] = torch.clamp(bbox[1], 0, image_height)
                bbox_clipped[2] = torch.clamp(bbox[2], 0, image_width)
                bbox_clipped[3] = torch.clamp(bbox[3], 0, image_height)
            else:
                bbox_clipped[0] = np.clip(bbox[0], 0, image_width)
                bbox_clipped[1] = np.clip(bbox[1], 0, image_height)
                bbox_clipped[2] = np.clip(bbox[2], 0, image_width)
                bbox_clipped[3] = np.clip(bbox[3], 0, image_height)
        else:
            if is_torch:
                bbox_clipped[:, 0] = torch.clamp(bbox[:, 0], 0, image_width)
                bbox_clipped[:, 1] = torch.clamp(bbox[:, 1], 0, image_height)
                bbox_clipped[:, 2] = torch.clamp(bbox[:, 2], 0, image_width)
                bbox_clipped[:, 3] = torch.clamp(bbox[:, 3], 0, image_height)
            else:
                bbox_clipped[:, 0] = np.clip(bbox[:, 0], 0, image_width)
                bbox_clipped[:, 1] = np.clip(bbox[:, 1], 0, image_height)
                bbox_clipped[:, 2] = np.clip(bbox[:, 2], 0, image_width)
                bbox_clipped[:, 3] = np.clip(bbox[:, 3], 0, image_height)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    
    return bbox_clipped
