"""Configuration for RGB pose estimation pipeline."""

import numpy as np
import os
import yaml
from typing import Dict, Optional


def load_camera_intrinsics(data_root: Optional[str] = None, object_id: str = '01') -> np.ndarray:
    """Load 3x3 camera intrinsic matrix from LineMOD info.yml file."""
    if data_root is None:
        # Default path relative to this file's location
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.join(project_root, 'data/Linemod_preprocessed/data')
    
    info_path = os.path.join(data_root, object_id, 'info.yml')
    
    if not os.path.exists(info_path):
        # Fallback to hardcoded LineMOD camera intrinsics
        # These are the standard values for the LineMOD dataset camera
        print(f"WARNING: info.yml not found at {info_path}, using default intrinsics")
        return np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.5704, 242.0489],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    # Load from YAML file
    with open(info_path, 'r') as f:
        info_data = yaml.safe_load(f)
    
    # Get camera matrix from first entry (all frames have same intrinsics)
    first_key = list(info_data.keys())[0]
    cam_K = info_data[first_key]['cam_K']
    
    # cam_K is stored as flat list: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    K = np.array(cam_K, dtype=np.float32).reshape(3, 3)
    
    return K


# Load camera intrinsics at module import time
CAMERA_INTRINSICS = load_camera_intrinsics()

# Convenience accessors for individual parameters
FX = float(CAMERA_INTRINSICS[0, 0])  # Focal length X (pixels)
FY = float(CAMERA_INTRINSICS[1, 1])  # Focal length Y (pixels)
CX = float(CAMERA_INTRINSICS[0, 2])  # Principal point X (pixels)
CY = float(CAMERA_INTRINSICS[1, 2])


LINEMOD_ID_MAP: Dict[int, str] = {
    1: 'ape',
    2: 'benchvise',
    4: 'camera',
    5: 'can',
    6: 'cat',
    8: 'driller',
    9: 'duck',
    10: 'eggbox',
    11: 'glue',
    12: 'holepuncher',
    13: 'iron',
    14: 'lamp',
    15: 'phone'
}

YOLO_NAME_TO_LINEMOD_ID: Dict[str, str] = {
    'ape': '01',
    'benchvise': '02',
    'camera': '04',
    'can': '05',
    'cat': '06',
    'driller': '08',
    'duck': '09',
    'eggbox': '10',
    'glue': '11',
    'holepuncher': '12',
    'iron': '13',
    'lamp': '14',
    'phone': '15'
}

OBJECT_ID_TO_NAME: Dict[str, str] = {f"{k:02d}": v for k, v in LINEMOD_ID_MAP.items()}

YOLO_FINETUNED_ID_MAP: Dict[int, str] = {
    0: 'ape',
    1: 'benchvise',
    2: 'camera',
    3: 'can',
    4: 'cat',
    5: 'driller',
    6: 'duck',
    7: 'eggbox',
    8: 'glue',
    9: 'holepuncher',
    10: 'iron',
    11: 'lamp',
    12: 'phone'
}


def get_linemod_id_from_name(object_name: str) -> Optional[str]:
    """Convert object name to LineMOD ID string."""
    return YOLO_NAME_TO_LINEMOD_ID.get(object_name.lower(), None)


def load_object_dimensions(yaml_path: str) -> Dict[str, Dict[str, float]]:
    """Load 3D object dimensions from LineMOD models_info.yml in meters."""
    if not os.path.exists(yaml_path):
        print(f"WARNING: models_info.yml not found at {yaml_path}")
        return {}

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    dimensions = {}
    for obj_id, info in data.items():
        obj_id = int(obj_id)
        if obj_id not in LINEMOD_ID_MAP:
            continue

        obj_name = LINEMOD_ID_MAP[obj_id]
        scale = 1000.0

        dimensions[obj_name] = {
            'diameter': info.get('diameter', 0.0) / scale,
            'width': info.get('size_x', 0.0) / scale,
            'height': info.get('size_y', 0.0) / scale,
            'depth': info.get('size_z', 0.0) / scale
        }

    return dimensions


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(PROJECT_ROOT, "data/Linemod_preprocessed/models/models_info.yml")

OBJECT_DIMENSIONS = load_object_dimensions(YAML_PATH)


class ConfigRGB:
    """Training configuration for RGB pose estimation."""
    
    DATA_ROOT = "data/Linemod_preprocessed"
    IMG_SIZE = (640, 480)
    INPUT_SIZE = (224, 224)
    OBJECT_NAME = 'ape'
    SPLIT_RATIO = 0.8
    SEED = 42
    NEED_PREPARATION = True
    YOLO_FALLBACK_ENABLED = True
    
    BATCH_SIZE = 32
    NUM_WORKERS = 20
    NUM_EPOCHS = 200
    LEARNING_RATE_START = 1e-4
    LEARNING_RATE_END = 1e-6
    WEIGHT_DECAY = 1e-3
    
    USE_EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 20
    
    BACKBONE = "resnet50"
    PRETRAINED = True
    
    ADD_THRESHOLD = 0.1
    SAVE_DIR = "trained_checkpoints/"


class ConfigInference:
    """Inference configuration for the pose estimation pipeline."""
    
    def __init__(self, object_id: str = "01"):
        """Initialize with LineMOD object ID."""
        self.OBJECT_ID = object_id
        
        self.YOLO_PATH = os.path.join(PROJECT_ROOT, "yolo/weights/fine-tuned-yolo-weights.pt")
        self.RESNET_PATH = os.path.join(PROJECT_ROOT, "trained_checkpoints/baseline/baseline_best.pth")
        self.YOLO_TYPE = "finetuned"
        self.IMAGE_PATH = os.path.join(PROJECT_ROOT, f"data/Linemod_preprocessed/data/{self.OBJECT_ID}/rgb/0000.png")
        self.DEPTH_PATH = os.path.join(PROJECT_ROOT, f"data/Linemod_preprocessed/data/{self.OBJECT_ID}/depth/0000.png")
