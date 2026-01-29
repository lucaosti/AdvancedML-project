"""Shared utilities for 6D pose estimation."""

from .pinhole_camera import (
    estimate_translation_from_bbox,
    estimate_translation_from_bbox_torch,
    project_3d_to_2d,
    compute_depth_from_bbox
)

from .bbox import (
    xywh_to_xyxy,
    xyxy_to_xywh,
    normalize_bbox,
    denormalize_bbox,
    get_bbox_center,
    get_bbox_area,
    clip_bbox
)

from .geometry import load_ply_vertices

from .dataset import (
    LineMODDataset,
    UnifiedRGBDDataset,
    YOLODetector,
    linemod_collate_fn,
    rgbd_collate_fn,
    unified_collate_fn,
    load_gt_for_object,
    load_camera_intrinsics,
    load_depth_scale,
    depth_to_pointcloud,
    LINEMOD_OBJECTS,
    LINEMOD_ID_TO_NAME,
    OBJECT_NAME_TO_OBJ_ID,
)

__all__ = [
    'estimate_translation_from_bbox',
    'estimate_translation_from_bbox_torch',
    'project_3d_to_2d',
    'compute_depth_from_bbox',
    'xywh_to_xyxy',
    'xyxy_to_xywh',
    'normalize_bbox',
    'denormalize_bbox',
    'get_bbox_center',
    'get_bbox_area',
    'clip_bbox',
    'load_ply_vertices',
    'LineMODDataset',
    'UnifiedRGBDDataset',
    'YOLODetector',
    'linemod_collate_fn',
    'rgbd_collate_fn',
    'unified_collate_fn',
    'load_gt_for_object',
    'load_camera_intrinsics',
    'load_depth_scale',
    'depth_to_pointcloud',
    'LINEMOD_OBJECTS',
    'LINEMOD_ID_TO_NAME',
    'OBJECT_NAME_TO_OBJ_ID',
]
