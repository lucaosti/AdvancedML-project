"""RGB-only 6D pose estimation pipeline using ResNet backbone with pinhole camera guidance."""

from rgb_pipeline.model import RGBPoseEstimator
from rgb_pipeline.config import (
    ConfigRGB,
    ConfigInference,
    CAMERA_INTRINSICS,
    FX, FY, CX, CY,
    load_camera_intrinsics,
    OBJECT_DIMENSIONS,
    LINEMOD_ID_MAP,
    OBJECT_ID_TO_NAME,
    YOLO_NAME_TO_LINEMOD_ID,
    get_linemod_id_from_name,
)

__all__ = [
    'RGBPoseEstimator',
    'ConfigRGB',
    'ConfigInference',
    'CAMERA_INTRINSICS',
    'FX', 'FY', 'CX', 'CY',
    'load_camera_intrinsics',
    'OBJECT_DIMENSIONS',
    'LINEMOD_ID_MAP',
    'OBJECT_ID_TO_NAME',
    'YOLO_NAME_TO_LINEMOD_ID',
    'get_linemod_id_from_name',
]

