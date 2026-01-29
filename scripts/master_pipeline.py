#!/usr/bin/env python3
"""Master Pipeline for training all 6D Pose Estimation models."""

import os
import sys
import platform
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm


ALL_OBJECTS = [
    'ape', 'benchvise', 'camera', 'can', 'cat', 'driller',
    'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
]

ALL_PIPELINES = ['rgb', 'residual_learning', 'densefusion_iterative', 'pvn3d', 'ffb6d']

OBJECT_ID_MAP = {
    'ape': '01', 'benchvise': '02', 'camera': '04', 'can': '05', 'cat': '06',
    'driller': '08', 'duck': '09', 'eggbox': '10', 'glue': '11',
    'holepuncher': '12', 'iron': '13', 'lamp': '14', 'phone': '15'
}

OBJECT_INDEX_MAP = {
    'ape': 0, 'benchvise': 1, 'camera': 2, 'can': 3, 'cat': 4,
    'driller': 5, 'duck': 6, 'eggbox': 7, 'glue': 8,
    'holepuncher': 9, 'iron': 10, 'lamp': 11, 'phone': 12
}

NUM_EPOCHS_RGB = 500
BATCH_SIZE_RGB = 32
LEARNING_RATE_RGB = 1e-4
WEIGHT_DECAY_RGB = 1e-3
INPUT_SIZE_RGB = (224, 224)
BACKBONE_RGB = 'resnet50'

NUM_EPOCHS_RGBD = 500
BATCH_SIZE_RGBD = 16
LEARNING_RATE_RGBD = 1e-4
WEIGHT_DECAY_RGBD = 1e-3
NUM_POINTS_RGBD = 500
RGB_SIZE_RGBD = 128

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 20

NUM_WORKERS = 0 if platform.system() == 'Windows' else 10

USE_AMP = torch.cuda.is_available()

RANDOM_SEED = 42

_SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = _SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / 'data' / 'Linemod_preprocessed'

SAVE_DIR_RGB = PROJECT_ROOT / 'trained_checkpoints'
SAVE_DIR_RGBD = PROJECT_ROOT / 'trained_checkpoints_rgbd'

YOLO_WEIGHTS_PATH = PROJECT_ROOT / 'yolo' / 'weights' / 'fine-tuned-yolo-weights.pt'


def setup_environment():
    """Setup Python path and random seeds."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    return get_device()


def get_device() -> torch.device:
    """Get optimal training device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        print(f"CUDA: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print("Device: CPU (GPU not available)")
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    return device


def adjust_batch_size_for_gpu(base_batch_size: int, model_type: str) -> int:
    """Adjust batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return base_batch_size
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if model_type == 'rgb':
        if gpu_memory_gb < 8:
            return min(base_batch_size, 16)
        elif gpu_memory_gb >= 16:
            return min(base_batch_size * 2, 64)
    else:  # rgbd
        if gpu_memory_gb < 8:
            return min(base_batch_size, 4)
        elif gpu_memory_gb < 12:
            return min(base_batch_size, 8)
        elif gpu_memory_gb >= 16:
            return min(base_batch_size * 2, 32)
    
    return base_batch_size


def train_rgb_pipeline(
    object_name: str,
    device: torch.device,
    epochs: int = NUM_EPOCHS_RGB,
    batch_size: int = BATCH_SIZE_RGB,
    num_workers: int = NUM_WORKERS,
    yolo_bbox_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None
) -> Dict[str, Any]:
    """Train RGB pose estimation model for a single object."""
    from rgb_pipeline.config import ConfigRGB, ConfigInference, OBJECT_DIMENSIONS
    from rgb_pipeline.model import RGBPoseEstimator
    from rgb_pipeline.train import train_rgb_model
    from utils.dataset import LineMODDataset, linemod_collate_fn
    from scripts.prepare_data import prepare_linemod_data
    
    try:
        print(f"\n[RGB] Training on {object_name}...")
        start_time = time.time()
        
        # Configuration
        cfg = ConfigRGB()
        cfg_yolo = ConfigInference()
        
        # Prepare data splits if needed
        obj_data_path = DATA_ROOT / 'data' / OBJECT_ID_MAP.get(object_name, object_name)
        if not (obj_data_path / 'train.txt').exists():
            prepare_linemod_data(str(obj_data_path), str(obj_data_path), 
                               train_ratio=cfg.SPLIT_RATIO, seed=RANDOM_SEED)
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg.INPUT_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(cfg.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Datasets using unified loader in rgb mode (no depth loading)
        train_dataset = LineMODDataset(
            root_dir=str(DATA_ROOT), object_name=object_name, split='train',
            mode='rgb', transform=train_transform, yolo_model_path=cfg_yolo.YOLO_PATH,
            yolo_bbox_cache=yolo_bbox_cache, yolo_padding_pct=0.1
        )
        
        val_dataset = LineMODDataset(
            root_dir=str(DATA_ROOT), object_name=object_name, split='test',
            mode='rgb', transform=val_transform, yolo_model_path=cfg_yolo.YOLO_PATH,
            yolo_bbox_cache=yolo_bbox_cache, yolo_padding_pct=0.1
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=linemod_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=linemod_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        # Model
        model = RGBPoseEstimator(pretrained=cfg.PRETRAINED, backbone=cfg.BACKBONE)
        
        # Save directory
        save_dir = SAVE_DIR_RGB / object_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        history = train_rgb_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            num_epochs=epochs, learning_rate=cfg.LEARNING_RATE_START,
            weight_decay=cfg.WEIGHT_DECAY, device=device, save_dir=str(save_dir),
            object_name=object_name,
            early_stopping_patience=EARLY_STOPPING_PATIENCE if EARLY_STOPPING_ENABLED else 0
        )
        
        # Save history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        elapsed = time.time() - start_time
        best_loss = min(history['val_losses'])
        best_angle = history['val_angle_errors'][history['val_losses'].index(best_loss)]
        
        print(f"[RGB] {object_name}: completed in {elapsed/60:.1f} min "
              f"(loss={best_loss:.4f}, angle={best_angle:.2f} deg)")
        
        return {
            'status': 'success',
            'best_loss': best_loss,
            'best_angle': best_angle,
            'time_seconds': elapsed,
            'history': history
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def train_rgbd_pipeline(
    model_name: str,
    object_name: str,
    device: torch.device,
    epochs: int = NUM_EPOCHS_RGBD,
    batch_size: int = BATCH_SIZE_RGBD,
    num_points: int = NUM_POINTS_RGBD,
    num_workers: int = NUM_WORKERS,
    yolo_bbox_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None
) -> Dict[str, Any]:
    """Train RGB-D pose estimation model for a single object."""
    try:
        print(f"\n[{model_name}] Training on {object_name}...")
        start_time = time.time()
        
        if model_name == 'residual_learning':
            result = _train_residual_learning(
                object_name, device, epochs, batch_size, num_points, num_workers, yolo_bbox_cache
            )
        else:
            result = _train_generic_rgbd(
                model_name, object_name, device, epochs, batch_size, num_points, num_workers, yolo_bbox_cache
            )
        
        elapsed = time.time() - start_time
        result['time_seconds'] = elapsed
        
        if result['status'] == 'success':
            print(f"[{model_name}] {object_name}: completed in {elapsed/60:.1f} min "
                  f"(rot={result.get('best_rot', 0):.2f} deg, trans={result.get('best_trans', 0):.2f} cm)")
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def _train_residual_learning(
    object_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_points: int,
    num_workers: int,
    yolo_bbox_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None
) -> Dict[str, Any]:
    """Train residual learning model using the unified dataset."""
    from rgbd_pipelines.residual_learning.fusion_network import create_rgbd_fusion_model
    from rgbd_pipelines.residual_learning.fusion_loss import GeodesicRotationLoss
    from rgbd_pipelines.residual_learning.geometric_anchor import get_geometric_anchor_batch
    from utils.dataset import LineMODDataset, rgbd_collate_fn
    from utils.geometry import quaternion_to_rotation_matrix_torch
    
    train_dataset = LineMODDataset(
        root_dir=str(DATA_ROOT),
        object_name=object_name,
        split='train',
        mode='rgbd',
        num_points=num_points,
        yolo_model_path=str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else None,
        yolo_bbox_cache=yolo_bbox_cache,
        yolo_padding_pct=0.1
    )
    
    val_dataset = LineMODDataset(
        root_dir=str(DATA_ROOT),
        object_name=object_name,
        split='test',
        mode='rgbd',
        num_points=num_points,
        yolo_model_path=str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else None,
        yolo_bbox_cache=yolo_bbox_cache,
        yolo_padding_pct=0.1
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=rgbd_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=rgbd_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    model = create_rgbd_fusion_model(
        variant='standard',
        num_points=num_points,
        pretrained=True
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_RGBD, weight_decay=WEIGHT_DECAY_RGBD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    rot_criterion = GeodesicRotationLoss()
    trans_criterion = nn.SmoothL1Loss(beta=0.01)
    trans_weight = 1.0
    
    history = {
        'train_losses': [], 'val_losses': [],
        'val_rot_error': [], 'val_trans_error': [], 'lr': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    save_dir = SAVE_DIR_RGBD / 'residual_learning' / object_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            points = batch['points'].to(device)
            gt_R = batch['rotation'].to(device)
            gt_t = batch['translation'].to(device)
            
            T_geo = get_geometric_anchor_batch(
                bboxes=batch['bbox_abs'],
                depth_crops=batch['depth_crop'],
                intrinsics=batch['intrinsics'],
                crop_offsets=batch['crop_offset'],
                shrink_factor=0.15,
                depth_percentile=20.0
            ).to(device)
            
            optimizer.zero_grad()
            quaternion, delta_t, T_final = model(rgb, points, T_geo)
            pred_R = quaternion_to_rotation_matrix_torch(quaternion)
            
            loss_rot = rot_criterion(pred_R, gt_R)
            loss_trans = trans_criterion(T_final, gt_t)
            loss = loss_rot + trans_weight * loss_trans
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        history['train_losses'].append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        rot_errors = []
        trans_errors = []
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                points = batch['points'].to(device)
                gt_R = batch['rotation'].to(device)
                gt_t = batch['translation'].to(device)
                
                T_geo = get_geometric_anchor_batch(
                    bboxes=batch['bbox_abs'],
                    depth_crops=batch['depth_crop'],
                    intrinsics=batch['intrinsics'],
                    crop_offsets=batch['crop_offset'],
                    shrink_factor=0.15,
                    depth_percentile=20.0
                ).to(device)
                
                quaternion, delta_t, T_final = model(rgb, points, T_geo)
                pred_R = quaternion_to_rotation_matrix_torch(quaternion)
                
                loss_rot = rot_criterion(pred_R, gt_R)
                loss_trans = trans_criterion(T_final, gt_t)
                loss = loss_rot + trans_weight * loss_trans
                
                if torch.isfinite(loss):
                    val_loss += loss.item()
                    num_val += 1
                    
                    R_diff = torch.bmm(pred_R, gt_R.transpose(1, 2))
                    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                    cos_angle = torch.clamp((trace - 1) / 2, -1, 1)
                    rot_err = torch.acos(cos_angle).mean() * 180 / np.pi
                    rot_errors.append(rot_err.item())
                    
                    trans_err = torch.norm(T_final - gt_t, dim=1).mean() * 100
                    trans_errors.append(trans_err.item())
        
        avg_val_loss = val_loss / max(num_val, 1)
        avg_rot_err = np.mean(rot_errors) if rot_errors else float('nan')
        avg_trans_err = np.mean(trans_errors) if trans_errors else float('nan')
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['val_losses'].append(avg_val_loss)
        history['val_rot_error'].append(avg_rot_err)
        history['val_trans_error'].append(avg_trans_err)
        history['lr'].append(current_lr)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }, save_dir / 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_val_loss:.4f}, rot={avg_rot_err:.2f} deg, "
              f"trans={avg_trans_err:.2f} cm, lr={current_lr:.2e}")
        
        if EARLY_STOPPING_ENABLED and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    best_idx = history['val_losses'].index(min(history['val_losses']))
    return {
        'status': 'success',
        'best_loss': min(history['val_losses']),
        'best_rot': history['val_rot_error'][best_idx],
        'best_trans': history['val_trans_error'][best_idx],
        'history': history
    }


def _load_model_points(object_name: str, num_sample: int = 500) -> Optional[torch.Tensor]:
    """Load 3D model points from PLY file."""
    from utils.geometry import load_ply_vertices
    
    obj_id = OBJECT_ID_MAP.get(object_name.lower(), '01')
    ply_path = DATA_ROOT / 'models' / f'obj_{obj_id}.ply'
    
    points = load_ply_vertices(str(ply_path))
    if points is None:
        print(f"Warning: Could not load model points from {ply_path}")
        return None
    
    if np.abs(points).max() > 1.0:
        points = points / 1000.0
    
    if len(points) > num_sample:
        indices = np.random.choice(len(points), num_sample, replace=False)
        points = points[indices]
    
    return torch.tensor(points, dtype=torch.float32)


def _train_generic_rgbd(
    model_name: str,
    object_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_points: int,
    num_workers: int,
    yolo_bbox_cache: Optional[Dict[str, Tuple[float, float, float, float]]] = None
) -> Dict[str, Any]:
    """Train RGB-D models (densefusion_iterative, pvn3d, ffb6d)."""
    from rgbd_pipelines import create_model
    from utils.dataset import LineMODDataset, rgbd_collate_fn
    
    obj_idx = OBJECT_INDEX_MAP.get(object_name.lower(), 0)
    
    model_points = _load_model_points(object_name)
    if model_points is not None:
        model_points = model_points.to(device)
        print(f"  Loaded {model_points.shape[0]} model points for ADD loss")
    
    model_kwargs = {'num_points': num_points, 'num_obj': 13}
    if 'iterative' in model_name:
        model_kwargs['num_iter'] = 2
    if model_name in ['pvn3d', 'ffb6d']:
        model_kwargs['num_kp'] = 8
    
    model = create_model(model_name, **model_kwargs).to(device)
    
    train_dataset = LineMODDataset(
        root_dir=str(DATA_ROOT), object_name=object_name, split='train',
        mode='rgbd', num_points=num_points,
        yolo_model_path=str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else None,
        yolo_bbox_cache=yolo_bbox_cache,
        yolo_padding_pct=0.1
    )
    
    val_dataset = LineMODDataset(
        root_dir=str(DATA_ROOT), object_name=object_name, split='test',
        mode='rgbd', num_points=num_points,
        yolo_model_path=str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else None,
        yolo_bbox_cache=yolo_bbox_cache,
        yolo_padding_pct=0.1
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=rgbd_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=rgbd_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_RGBD, weight_decay=WEIGHT_DECAY_RGBD)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = {'train_loss': [], 'val_loss': [], 'rot_err': [], 'trans_err': []}
    best_loss = float('inf')
    epochs_without_improvement = 0
    save_dir = SAVE_DIR_RGBD / model_name / object_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            rgb = batch['rgb'].to(device)
            points = batch['points'].to(device)
            choose = batch['choose'].to(device)
            gt_R = batch['rotation'].to(device)
            gt_t = batch['translation'].to(device)
            
            points = torch.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
            B = rgb.size(0)
            obj_idx_tensor = torch.full((B,), obj_idx, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            pred_dict = model(rgb, points, choose, obj_idx_tensor)
            loss_dict = model.get_loss(pred_dict, gt_R, gt_t, model_points=model_points, points=points)
            loss = loss_dict['total']
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        history['train_loss'].append(total_loss / max(num_batches, 1))
        
        model.eval()
        val_loss = 0.0
        rot_errors = []
        trans_errors = []
        num_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch['rgb'].to(device)
                points = batch['points'].to(device)
                choose = batch['choose'].to(device)
                gt_R = batch['rotation'].to(device)
                gt_t = batch['translation'].to(device)
                
                points = torch.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
                B = rgb.size(0)
                obj_idx_tensor = torch.full((B,), obj_idx, dtype=torch.long, device=device)
                
                pred_dict = model(rgb, points, choose, obj_idx_tensor)
                loss_dict = model.get_loss(pred_dict, gt_R, gt_t, model_points=model_points, points=points)
                
                if torch.isfinite(loss_dict['total']):
                    val_loss += loss_dict['total'].item()
                    num_val += 1
                    
                    pred_R = pred_dict['rotation']
                    R_diff = torch.bmm(pred_R, gt_R.transpose(1, 2))
                    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
                    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
                    theta = torch.acos(cos_theta)
                    rot_errors.extend(torch.rad2deg(theta).cpu().numpy().tolist())
                    
                    pred_t = pred_dict['translation']
                    t_err = torch.norm(pred_t - gt_t, dim=1)
                    trans_errors.extend((t_err * 100).cpu().numpy().tolist())
        
        avg_val_loss = val_loss / max(num_val, 1)
        avg_rot_err = np.mean(rot_errors) if rot_errors else float('nan')
        avg_trans_err = np.mean(trans_errors) if trans_errors else float('nan')
        
        if not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)
        
        history['val_loss'].append(avg_val_loss)
        history['rot_err'].append(avg_rot_err)
        history['trans_err'].append(avg_trans_err)
        
        if avg_val_loss < best_loss and not np.isnan(avg_val_loss):
            best_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        else:
            epochs_without_improvement += 1
        
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_val_loss:.4f}, rot={avg_rot_err:.2f} deg, trans={avg_trans_err:.2f} cm")
        
        if EARLY_STOPPING_ENABLED and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break
    
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    best_idx = history['val_loss'].index(min(history['val_loss']))
    return {
        'status': 'success',
        'best_loss': best_loss,
        'best_rot': history['rot_err'][best_idx],
        'best_trans': history['trans_err'][best_idx],
        'history': history
    }


def _precompute_yolo_cache(object_name: str, yolo_model_path: Optional[str] = None) -> Dict[str, Tuple[float, float, float, float]]:
    """Pre-compute YOLO bounding boxes for all frames of an object."""
    from ultralytics import YOLO
    from PIL import Image
    import yaml
    
    obj_id = OBJECT_ID_MAP.get(object_name.lower(), '01')
    data_dir = DATA_ROOT / 'data' / obj_id
    rgb_dir = data_dir / 'rgb'
    gt_path = data_dir / 'gt.yml'
    
    if yolo_model_path is None or not Path(yolo_model_path).exists():
        yolo_model_path = str(YOLO_WEIGHTS_PATH) if YOLO_WEIGHTS_PATH.exists() else 'yolov8n.pt'
    
    yolo_model = YOLO(yolo_model_path)
    
    target_class_id = None
    for cls_id, cls_name in yolo_model.names.items():
        if cls_name.lower() == object_name.lower():
            target_class_id = cls_id
            break
    
    all_frames = sorted([int(f.stem) for f in rgb_dir.glob('*.png')])
    
    print(f"Pre-computing YOLO bboxes for {len(all_frames)} frames of {object_name}...")
    
    cache = {}
    detection_count = 0
    skipped_count = 0
    padding_pct = 0.1
    
    for frame_idx in all_frames:
        rgb_path = rgb_dir / f'{frame_idx:04d}.png'
        if not rgb_path.exists():
            continue
        
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_array = np.array(rgb_image)
        H, W = rgb_array.shape[:2]
        
        bgr_array = rgb_array[:, :, ::-1].copy()
        results = yolo_model(bgr_array, verbose=False, conf=0.01)
        
        bbox = None
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            if target_class_id is not None:
                class_ids = boxes.cls.cpu().numpy()
                mask = class_ids == target_class_id
                
                if mask.any():
                    matching_indices = np.where(mask)[0]
                    confs = boxes.conf.cpu().numpy()[matching_indices]
                    best_local_idx = confs.argmax()
                    best_idx = matching_indices[best_local_idx]
                    
                    x1, y1, x2, y2 = boxes[best_idx].xyxy[0].cpu().numpy()
                    
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = bw * padding_pct, bh * padding_pct
                    
                    bbox = (
                        max(0, float(x1 - pad_x)),
                        max(0, float(y1 - pad_y)),
                        min(W, float(x2 + pad_x)),
                        min(H, float(y2 + pad_y))
                    )
                    detection_count += 1
        
        if bbox is None:
            skipped_count += 1
        
        cache[str(frame_idx)] = bbox
    
    print(f"YOLO cache ready: {len(cache)} frames for {object_name} "
          f"(detected: {detection_count}, skipped: {skipped_count})")
    
    del yolo_model
    
    return cache


def run_training_sequential(
    pipelines: List[str],
    objects: List[str],
    device: torch.device,
    epochs_rgb: int = NUM_EPOCHS_RGB,
    epochs_rgbd: int = NUM_EPOCHS_RGBD,
    batch_size_rgb: int = BATCH_SIZE_RGB,
    batch_size_rgbd: int = BATCH_SIZE_RGBD,
    num_points: int = NUM_POINTS_RGBD,
    num_workers: int = NUM_WORKERS
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run training for all pipeline and object pairs."""
    results = {p: {} for p in pipelines}
    total_tasks = len(pipelines) * len(objects)
    task_count = 0
    
    # Iterate over OBJECTS first, then PIPELINES
    for obj in objects:
        print(f"\n{'#'*60}")
        print(f"# OBJECT: {obj.upper()}")
        print(f"{'#'*60}")
        
        # Pre-compute YOLO cache ONCE for this object
        yolo_cache = _precompute_yolo_cache(obj)
        
        for pipeline in pipelines:
            task_count += 1
            print(f"\n{'='*60}")
            print(f"[{task_count}/{total_tasks}] Training {pipeline} on {obj}")
            print(f"{'='*60}")
            
            if pipeline == 'rgb':
                batch_size = adjust_batch_size_for_gpu(batch_size_rgb, 'rgb')
                result = train_rgb_pipeline(obj, device, epochs_rgb, batch_size, num_workers, yolo_cache)
            else:
                batch_size = adjust_batch_size_for_gpu(batch_size_rgbd, 'rgbd')
                result = train_rgbd_pipeline(pipeline, obj, device, epochs_rgbd, batch_size, num_points, num_workers, yolo_cache)
            
            results[pipeline][obj] = result
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del yolo_cache
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Master Pipeline - Train all models on all objects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/master_pipeline.py
    python scripts/master_pipeline.py --objects ape cat duck
    python scripts/master_pipeline.py --pipelines rgb residual_learning
    python scripts/master_pipeline.py --epochs_rgb 100 --epochs_rgbd 150
        """
    )
    
    # Objects and pipelines
    parser.add_argument('--objects', nargs='+', default=ALL_OBJECTS,
                        help='Objects to train (default: all)')
    parser.add_argument('--pipelines', nargs='+', default=ALL_PIPELINES,
                        help='Pipelines to train (default: all)')
    
    # Training parameters
    parser.add_argument('--epochs_rgb', type=int, default=NUM_EPOCHS_RGB,
                        help=f'Epochs for RGB pipeline (default: {NUM_EPOCHS_RGB})')
    parser.add_argument('--epochs_rgbd', type=int, default=NUM_EPOCHS_RGBD,
                        help=f'Epochs for RGB-D pipelines (default: {NUM_EPOCHS_RGBD})')
    parser.add_argument('--batch_size_rgb', type=int, default=BATCH_SIZE_RGB,
                        help=f'Batch size for RGB (default: {BATCH_SIZE_RGB})')
    parser.add_argument('--batch_size_rgbd', type=int, default=BATCH_SIZE_RGBD,
                        help=f'Batch size for RGB-D (default: {BATCH_SIZE_RGBD})')
    parser.add_argument('--num_points', type=int, default=NUM_POINTS_RGBD,
                        help=f'Points in point cloud (default: {NUM_POINTS_RGBD})')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                        help=f'DataLoader workers (default: {NUM_WORKERS})')
    
    args = parser.parse_args()
    
    epochs_rgb = args.epochs_rgb
    epochs_rgbd = args.epochs_rgbd
    batch_size_rgb = args.batch_size_rgb
    batch_size_rgbd = args.batch_size_rgbd
    num_points = args.num_points
    num_workers = args.num_workers
    
    for obj in args.objects:
        if obj not in ALL_OBJECTS:
            print(f"Warning: Unknown object '{obj}'. Available: {ALL_OBJECTS}")
    for pipe in args.pipelines:
        if pipe not in ALL_PIPELINES:
            print(f"Warning: Unknown pipeline '{pipe}'. Available: {ALL_PIPELINES}")
    
    objects = [o for o in args.objects if o in ALL_OBJECTS]
    pipelines = [p for p in args.pipelines if p in ALL_PIPELINES]
    
    if not objects or not pipelines:
        print("Error: No valid objects or pipelines specified.")
        return 1
    
    print("="*60)
    print("MASTER PIPELINE - 6D Pose Estimation Training")
    print("="*60)
    
    device = setup_environment()
    
    print(f"\nConfiguration:")
    print(f"  Objects: {len(objects)} ({', '.join(objects)})")
    print(f"  Pipelines: {len(pipelines)} ({', '.join(pipelines)})")
    print(f"  RGB epochs: {epochs_rgb}, batch: {batch_size_rgb}")
    print(f"  RGB-D epochs: {epochs_rgbd}, batch: {batch_size_rgbd}, points: {num_points}")
    print(f"  Data root: {DATA_ROOT}")
    
    if not DATA_ROOT.exists():
        print(f"\nError: Data directory not found: {DATA_ROOT}")
        return 1
    
    start_time = time.time()
    
    results = run_training_sequential(
        pipelines, objects, device,
        epochs_rgb=epochs_rgb, epochs_rgbd=epochs_rgbd,
        batch_size_rgb=batch_size_rgb, batch_size_rgbd=batch_size_rgbd,
        num_points=num_points, num_workers=num_workers
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/3600:.2f} hours")
    
    for pipeline in pipelines:
        success = sum(1 for r in results[pipeline].values() if r.get('status') == 'success')
        total = len(results[pipeline])
        print(f"  {pipeline}: {success}/{total} successful")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
