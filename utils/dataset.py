"""Unified Dataset Loader for 6D Pose Estimation."""

import os
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, Optional, Tuple, List, Union, Callable
from pathlib import Path


LINEMOD_OBJECTS = {
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

LINEMOD_ID_TO_NAME = {v: k for k, v in LINEMOD_OBJECTS.items()}

OBJECT_NAME_TO_OBJ_ID = {
    'ape': 1,
    'benchvise': 2,
    'camera': 4,
    'can': 5,
    'cat': 6,
    'driller': 8,
    'duck': 9,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15
}

DEPTH_SCALE_MM_TO_M = 1.0 / 1000.0

IMAGE_SIZE = (640, 480)


class YOLODetector:
    """YOLO detector wrapper with caching and class filtering."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        padding_pct: float = 0.1,
        conf_threshold: float = 0.01,
        target_class: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize YOLO detector."""
        self.model = None
        self.model_path = model_path
        self.padding_pct = padding_pct
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        self.verbose = verbose
        self._loaded = False
        self._class_id = None  # Will be set when model loads
    
    def _ensure_loaded(self):
        """Lazy load YOLO model."""
        if not self._loaded:
            try:
                from ultralytics import YOLO
                path = self.model_path or 'yolov8n.pt'
                if self.verbose:
                    print(f"Loading YOLO model from {path}...")
                self.model = YOLO(path)
                self._loaded = True
                
                # Find target class ID if specified
                if self.target_class and self.model is not None:
                    names = self.model.names  # dict: {0: 'ape', 1: 'benchvise', ...}
                    for cls_id, cls_name in names.items():
                        if cls_name.lower() == self.target_class.lower():
                            self._class_id = cls_id
                            if self.verbose:
                                print(f"Target class '{self.target_class}' has ID {cls_id}")
                            break
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
                self.model = None
                self._loaded = True
    
    def detect(
        self,
        image: np.ndarray,
        return_normalized: bool = False
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float]:
        """Detect object and return bounding box with padding."""
        self._ensure_loaded()
        
        if self.model is None:
            return None, 0.0
        
        H, W = image.shape[:2]
        image_bgr = image[:, :, ::-1].copy()
        results = self.model(image_bgr, verbose=False, conf=self.conf_threshold)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0
        
        boxes = results[0].boxes
        
        if self._class_id is not None:
            class_ids = boxes.cls.cpu().numpy()
            mask = class_ids == self._class_id
            
            if not mask.any():
                return None, 0.0
            
            matching_indices = np.where(mask)[0]
            confs = boxes.conf.cpu().numpy()[matching_indices]
            best_local_idx = confs.argmax()
            best_idx = matching_indices[best_local_idx]
        else:
            best_idx = boxes.conf.argmax().item()
        
        best_box = boxes[best_idx]
        conf = float(best_box.conf[0])
        
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * self.padding_pct
        pad_y = bh * self.padding_pct
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(W, x2 + pad_x)
        y2_pad = min(H, y2 + pad_y)
        
        if return_normalized:
            bbox = (x1_pad / W, y1_pad / H, x2_pad / W, y2_pad / H)
        else:
            bbox = (x1_pad, y1_pad, x2_pad, y2_pad)
        
        return bbox, conf
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[Optional[Tuple[float, float, float, float]], float]]:
        """Detect objects in batch of images."""
        return [self.detect(img) for img in images]


def load_gt_for_object(
    gt_path: str,
    target_obj_id: int
) -> Dict[int, Dict]:
    """Load ground truth from gt.yml filtered by object ID."""
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    
    filtered_gt = {}
    
    for frame_idx, entries in gt_data.items():
        frame_idx = int(frame_idx)
        
        for entry in entries:
            if entry.get('obj_id') == target_obj_id:
                R_flat = entry['cam_R_m2c']
                R = np.array(R_flat, dtype=np.float32).reshape(3, 3)
                t = np.array(entry['cam_t_m2c'], dtype=np.float32) * DEPTH_SCALE_MM_TO_M
                bbox_gt = np.array(entry['obj_bb'], dtype=np.float32)
                
                filtered_gt[frame_idx] = {
                    'rotation': R,
                    'translation': t,
                    'bbox_gt': bbox_gt
                }
                break
    
    return filtered_gt


def load_camera_intrinsics(info_path: str) -> np.ndarray:
    """Load camera intrinsic matrix from info.yml."""
    with open(info_path, 'r') as f:
        info_data = yaml.safe_load(f)
    
    first_key = list(info_data.keys())[0]
    cam_K = info_data[first_key]['cam_K']
    K = np.array(cam_K, dtype=np.float32).reshape(3, 3)
    return K


def load_depth_scale(info_path: str) -> float:
    """Load depth scale from info.yml."""
    with open(info_path, 'r') as f:
        info_data = yaml.safe_load(f)
    
    first_key = list(info_data.keys())[0]
    depth_scale = info_data[first_key].get('depth_scale', 1.0)
    return depth_scale * DEPTH_SCALE_MM_TO_M


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    num_points: int = 500,
    depth_scale: float = DEPTH_SCALE_MM_TO_M,
    min_depth: float = 0.1,
    max_depth: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D point cloud using pinhole camera model."""
    H, W = depth.shape
    depth_m = depth.astype(np.float32) * depth_scale
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(W, x2)), int(min(H, y2))
        depth_roi = depth_m[y1:y2, x1:x2]
        u_roi = u[y1:y2, x1:x2]
        v_roi = v[y1:y2, x1:x2]
    else:
        depth_roi = depth_m
        u_roi = u
        v_roi = v
    
    depth_flat = depth_roi.flatten()
    u_flat = u_roi.flatten().astype(np.float32)
    v_flat = v_roi.flatten().astype(np.float32)
    
    valid_mask = (depth_flat > min_depth) & (depth_flat < max_depth)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return np.zeros((num_points, 3), dtype=np.float32), np.zeros(num_points, dtype=np.int64)
    
    if len(valid_indices) >= num_points:
        chosen_indices = np.random.choice(valid_indices, num_points, replace=False)
    else:
        chosen_indices = np.random.choice(valid_indices, num_points, replace=True)
    
    Z = depth_flat[chosen_indices]
    u_sel = u_flat[chosen_indices]
    v_sel = v_flat[chosen_indices]
    
    X = (u_sel - cx) * Z / fx
    Y = (v_sel - cy) * Z / fy
    
    points = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return points, chosen_indices


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to zero mean and unit variance."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
    
    return normalized


class LineMODDataset(Dataset):
    """Unified LineMOD dataset for RGB and RGB-D pipelines."""
    
    def __init__(
        self,
        root_dir: str,
        object_name: str,
        split: str = 'train',
        mode: str = 'rgb',
        transform: Optional[Callable] = None,
        yolo_model_path: Optional[str] = None,
        yolo_bbox_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        yolo_padding_pct: float = 0.1,
        num_points: int = 500,
        normalize_points: bool = False,
        rgb_size: int = 128
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.object_name = object_name.lower()
        self.split = split
        self.mode = mode.lower()
        self.transform = transform
        self.num_points = num_points
        self.normalize_points = normalize_points
        self.rgb_size = rgb_size
        self.yolo_padding_pct = yolo_padding_pct
        
        # Validate mode
        assert self.mode in ['rgb', 'rgbd'], f"Mode must be 'rgb' or 'rgbd', got {mode}"
        
        # Get object folder ID and obj_id
        self.obj_folder_id = LINEMOD_OBJECTS.get(self.object_name)
        if self.obj_folder_id is None:
            raise ValueError(f"Unknown object: {object_name}. Valid: {list(LINEMOD_OBJECTS.keys())}")
        
        self.obj_id = OBJECT_NAME_TO_OBJ_ID[self.object_name]
        
        self.data_dir = self.root_dir / 'data' / self.obj_folder_id
        self.rgb_dir = self.data_dir / 'rgb'
        self.depth_dir = self.data_dir / 'depth'
        self.gt_path = self.data_dir / 'gt.yml'
        self.info_path = self.data_dir / 'info.yml'
        self.split_file = self.data_dir / f'{split}.txt'
        
        # Validate paths
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            self.frame_indices = [int(line.strip()) for line in f if line.strip()]
        
        self.gt_data = load_gt_for_object(str(self.gt_path), self.obj_id)
        self.frame_indices = [idx for idx in self.frame_indices if idx in self.gt_data]
        
        if len(self.frame_indices) == 0:
            raise ValueError(f"No valid frames found for {object_name} in {split} split")
        
        self.intrinsics = load_camera_intrinsics(str(self.info_path))
        self.depth_scale = load_depth_scale(str(self.info_path))
        
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            padding_pct=yolo_padding_pct,
            conf_threshold=0.01,
            target_class=self.object_name,
            verbose=False
        )
        
        self.yolo_bbox_cache = yolo_bbox_cache or {}
        
        print(f"LineMODDataset ({mode}): {object_name}/{split} - {len(self.frame_indices)} frames")
    
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def _get_yolo_bbox(
        self,
        frame_idx: int,
        rgb_image: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float, float, float]], bool]:
        """Get YOLO bounding box, using cache if available. Returns None if detection fails."""
        cache_key = str(frame_idx)
        
        if cache_key in self.yolo_bbox_cache:
            cached = self.yolo_bbox_cache[cache_key]
            # Cache stores (bbox, is_detection) tuple
            if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], bool):
                return cached
            else:
                # Old cache format - assume it was a detection
                return cached, True
        
        # Run YOLO detection
        bbox, conf = self.yolo_detector.detect(rgb_image, return_normalized=False)
        
        is_yolo_detection = bbox is not None
        
        # NO GT FALLBACK - if YOLO fails, bbox stays None
        # Cache the result (bbox, is_detection)
        self.yolo_bbox_cache[cache_key] = (bbox, is_yolo_detection)
        
        return bbox, is_yolo_detection
    
    def _load_rgb(self, frame_idx: int) -> np.ndarray:
        """Load RGB image as numpy array."""
        rgb_path = self.rgb_dir / f'{frame_idx:04d}.png'
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        return rgb
    
    def _load_depth(self, frame_idx: int) -> np.ndarray:
        """Load depth map as numpy array."""
        depth_path = self.depth_dir / f'{frame_idx:04d}.png'
        # Load as 16-bit (LineMOD depth is typically 16-bit PNG)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Depth not found: {depth_path}")
        return depth
    
    def _crop_rgb(
        self,
        rgb: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Crop and resize RGB image from bbox."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = rgb[y1:y2, x1:x2]
        
        if crop.size == 0:
            crop = rgb
        
        if self.mode == 'rgbd' or self.transform is None:
            crop = cv2.resize(crop, (self.rgb_size, self.rgb_size), interpolation=cv2.INTER_LINEAR)
        
        return crop
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a sample from the dataset. Returns None if YOLO fails to detect."""
        frame_idx = self.frame_indices[idx]
        gt = self.gt_data[frame_idx]
        
        rgb = self._load_rgb(frame_idx)
        H, W = rgb.shape[:2]
        
        bbox, yolo_detected = self._get_yolo_bbox(frame_idx, rgb)
        
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        
        bbox_normalized = (
            x1 / W,
            y1 / H,
            (x2 - x1) / W,
            (y2 - y1) / H
        )
        
        rgb_crop = self._crop_rgb(rgb, bbox)
        
        if self.transform is not None:
            rgb_pil = Image.fromarray(rgb_crop)
            rgb_tensor = self.transform(rgb_pil)
        else:
            rgb_tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_tensor = (rgb_tensor - mean) / std
        
        sample = {
            'rgb': rgb_tensor,
            'rotation': torch.from_numpy(gt['rotation']),
            'translation': torch.from_numpy(gt['translation']),
            'bbox': torch.tensor(bbox_normalized, dtype=torch.float32),
            'intrinsics': torch.from_numpy(self.intrinsics),
            'frame_idx': frame_idx
        }
        
        if self.mode == 'rgbd':
            depth = self._load_depth(frame_idx)
            
            points, choose = depth_to_pointcloud(
                depth=depth,
                intrinsics=self.intrinsics,
                bbox=bbox,
                num_points=self.num_points,
                depth_scale=self.depth_scale
            )
            
            if self.normalize_points:
                points = normalize_pointcloud(points)
            
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            depth_crop = depth[y1i:y2i, x1i:x2i].astype(np.float32) * self.depth_scale
            
            sample['points'] = torch.from_numpy(points)
            sample['choose'] = torch.from_numpy(choose.astype(np.int64))
            sample['depth_crop'] = torch.from_numpy(depth_crop)
            sample['crop_offset'] = torch.tensor([x1, y1], dtype=torch.float32)
            sample['bbox_abs'] = torch.tensor([x1, y1, x2 - x1, y2 - y1], dtype=torch.float32)
            
            gt_bbox = gt.get('bbox_gt', np.array([0, 0, 0, 0], dtype=np.float32))
            gt_x, gt_y, gt_w, gt_h = gt_bbox
            sample['bbox_gt'] = torch.tensor([gt_x, gt_y, gt_x + gt_w, gt_y + gt_h], dtype=torch.float32)
            sample['yolo_detected'] = torch.tensor(yolo_detected, dtype=torch.bool)
        
        return sample
    
    @staticmethod
    def get_available_objects(root_dir: str) -> List[str]:
        """Get list of available objects in the dataset."""
        data_path = Path(root_dir) / 'data'
        available = []
        for obj_name, folder_id in LINEMOD_OBJECTS.items():
            if (data_path / folder_id).exists():
                available.append(obj_name)
        return available


def linemod_collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """Collate function for LineMODDataset. Filters out None samples."""
    batch = [sample for sample in batch if sample is not None]
    
    if len(batch) == 0:
        return None
    
    tensor_keys = ['rgb', 'rotation', 'translation', 'bbox', 'intrinsics', 'points', 'choose', 'crop_offset', 'bbox_abs', 'bbox_gt', 'yolo_detected']
    
    result = {}
    
    for key in tensor_keys:
        if key in batch[0]:
            result[key] = torch.stack([sample[key] for sample in batch])
    
    if 'frame_idx' in batch[0]:
        result['frame_idx'] = [sample['frame_idx'] for sample in batch]
    
    if 'depth_crop' in batch[0]:
        result['depth_crop'] = [sample['depth_crop'] for sample in batch]
    
    return result


def rgbd_collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, torch.Tensor]]:
    """Collate function for RGBD mode (alias for backward compatibility)."""
    return linemod_collate_fn(batch)


UnifiedRGBDDataset = LineMODDataset
unified_collate_fn = linemod_collate_fn


__all__ = [
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
    'normalize_pointcloud',
    'LINEMOD_OBJECTS',
    'LINEMOD_ID_TO_NAME',
    'OBJECT_NAME_TO_OBJ_ID',
]
