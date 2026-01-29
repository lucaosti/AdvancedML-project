"""Training pipeline for RGB-D fusion network with residual learning."""

import os
import sys
import time
import argparse
from typing import Dict, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Local imports
from rgbd_pipelines.residual_learning.fusion_network import RGBDFusionNetwork, create_rgbd_fusion_model
from utils.dataset import LineMODDataset, rgbd_collate_fn
from rgbd_pipelines.residual_learning.fusion_loss import PoseLoss, GeodesicRotationLoss, TranslationLoss
from rgbd_pipelines.residual_learning.geometric_anchor import get_geometric_anchor_batch
from utils.geometry import quaternion_to_rotation_matrix_torch


def compute_batch_geometric_anchors(batch: Dict, device: torch.device) -> torch.Tensor:
    """Compute geometric anchors for a batch using Pinhole estimation."""
    # Use absolute bbox coordinates for geometric anchor computation
    T_geo = get_geometric_anchor_batch(
        bboxes=batch['bbox_abs'],
        depth_crops=batch['depth_crop'],
        intrinsics=batch['intrinsics'],
        crop_offsets=batch['crop_offset'],
        shrink_factor=0.15,
        depth_percentile=20.0
    )
    
    return T_geo.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    rot_criterion: nn.Module,
    trans_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    trans_weight: float = 1.0,
    quiet: bool = False
) -> Dict[str, float]:
    """Execute one training epoch and return metrics."""
    model.train()
    
    running_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0
    running_delta_norm = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False, disable=quiet)
    
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        points = batch['points'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)
        
        T_geo = compute_batch_geometric_anchors(batch, device)
        
        optimizer.zero_grad()
        
        quaternion, delta_t, T_final = model(rgb, points, T_geo)
        
        pred_R = quaternion_to_rotation_matrix_torch(quaternion)
        
        loss_rot = rot_criterion(pred_R, gt_R)
        loss_trans = trans_criterion(T_final, gt_t)
        
        total_loss = loss_rot + trans_weight * loss_trans
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += total_loss.item()
        running_rot_loss += loss_rot.item()
        running_trans_loss += loss_trans.item()
        running_delta_norm += torch.norm(delta_t, p=2, dim=1).mean().item()
        num_batches += 1
        
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Rot': f'{loss_rot.item():.4f}',
            'Trans': f'{loss_trans.item():.4f}',
            '|ΔT|': f'{torch.norm(delta_t, p=2, dim=1).mean().item():.4f}'
        })
    
    return {
        'loss': running_loss / num_batches,
        'rot_loss': running_rot_loss / num_batches,
        'trans_loss': running_trans_loss / num_batches,
        'delta_norm': running_delta_norm / num_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    rot_criterion: nn.Module,
    trans_criterion: nn.Module,
    device: torch.device,
    trans_weight: float = 1.0,
    quiet: bool = False
) -> Dict[str, float]:
    """Validate on validation set and return metrics."""
    model.eval()
    
    running_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0
    running_delta_norm = 0.0
    running_trans_error = 0.0
    running_rot_error = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating", leave=False, disable=quiet):
        rgb = batch['rgb'].to(device)
        points = batch['points'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)
        
        T_geo = compute_batch_geometric_anchors(batch, device)
        
        quaternion, delta_t, T_final = model(rgb, points, T_geo)
        pred_R = quaternion_to_rotation_matrix_torch(quaternion)
        
        loss_rot = rot_criterion(pred_R, gt_R)
        loss_trans = trans_criterion(T_final, gt_t)
        total_loss = loss_rot + trans_weight * loss_trans
        
        trans_error = torch.norm(T_final - gt_t, p=2, dim=1).mean()
        R_diff = torch.bmm(pred_R, gt_R.transpose(1, 2))
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = torch.clamp((trace - 1) / 2, -1, 1)
        rot_error = torch.acos(cos_angle).mean() * 180 / np.pi
        
        running_loss += total_loss.item()
        running_rot_loss += loss_rot.item()
        running_trans_loss += loss_trans.item()
        running_delta_norm += torch.norm(delta_t, p=2, dim=1).mean().item()
        running_trans_error += trans_error.item()
        running_rot_error += rot_error.item()
        num_batches += 1
    
    return {
        'loss': running_loss / num_batches,
        'rot_loss': running_rot_loss / num_batches,
        'trans_loss': running_trans_loss / num_batches,
        'delta_norm': running_delta_norm / num_batches,
        'trans_error_m': running_trans_error / num_batches,
        'rot_error_deg': running_rot_error / num_batches
    }


def train_fusion_model(
    object_name: str,
    data_root: str = "data/Linemod_preprocessed",
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_points: int = 500,
    rgb_size: int = 128,
    model_variant: str = 'standard',
    save_dir: str = "trained_checkpoints/fusion",
    device: Optional[str] = None,
    num_workers: int = 4,
    trans_weight: float = 1.0,
    bbox_noise: bool = True,
    quiet: bool = False
) -> Dict:
    """Full training pipeline for RGB-D fusion network."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    save_path = Path(save_dir) / object_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Loading {object_name} dataset ===")
    
    train_dataset = LineMODDataset(
        root_dir=data_root,
        object_name=object_name,
        split='train',
        mode='rgbd',
        num_points=num_points,
        rgb_size=rgb_size,
        yolo_padding_pct=0.1
    )
    
    test_dataset = LineMODDataset(
        root_dir=data_root,
        object_name=object_name,
        split='test',
        mode='rgbd',
        num_points=num_points,
        rgb_size=rgb_size,
        yolo_padding_pct=0.1
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rgbd_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rgbd_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\n=== Creating {model_variant} RGB-D Fusion Network ===")
    
    model = create_rgbd_fusion_model(
        variant=model_variant,
        num_points=num_points,
        pretrained=True
    )
    model = model.to(device)
    
    param_counts = model.get_num_parameters()
    print(f"Model parameters: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if name != 'total':
            print(f"  - {name}: {count:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )
    
    rot_criterion = GeodesicRotationLoss()
    trans_criterion = nn.SmoothL1Loss(beta=0.01)
    
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"Translation Weight: {trans_weight}, BBox Noise: {bbox_noise}")
    
    history = {
        'train_losses': [], 'val_losses': [],
        'train_rot_losses': [], 'val_rot_losses': [],
        'train_trans_losses': [], 'val_trans_losses': [],
        'train_delta_norm': [], 'val_delta_norm': [],
        'val_trans_error': [], 'val_rot_error': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            rot_criterion, trans_criterion,
            device, epoch, trans_weight, quiet=quiet
        )
        
        val_metrics = validate(
            model, test_loader,
            rot_criterion, trans_criterion,
            device, trans_weight, quiet=quiet
        )
        
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_losses'].append(train_metrics['loss'])
        history['val_losses'].append(val_metrics['loss'])
        history['train_rot_losses'].append(train_metrics['rot_loss'])
        history['val_rot_losses'].append(val_metrics['rot_loss'])
        history['train_trans_losses'].append(train_metrics['trans_loss'])
        history['val_trans_losses'].append(val_metrics['trans_loss'])
        history['train_delta_norm'].append(train_metrics['delta_norm'])
        history['val_delta_norm'].append(val_metrics['delta_norm'])
        history['val_trans_error'].append(val_metrics['trans_error_m'])
        history['val_rot_error'].append(val_metrics['rot_error_deg'])
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - start_time
        print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Rot: {train_metrics['rot_loss']:.4f} | "
              f"Trans: {train_metrics['trans_loss']:.4f} | "
              f"|ΔT|: {train_metrics['delta_norm']*100:.2f}cm")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Rot: {val_metrics['rot_loss']:.4f} ({val_metrics['rot_error_deg']:.1f}°) | "
              f"Trans: {val_metrics['trans_loss']:.4f} ({val_metrics['trans_error_m']*100:.2f}cm)")
        print(f"  LR: {current_lr:.2e}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'config': {
                    'object_name': object_name,
                    'num_points': num_points,
                    'rgb_size': rgb_size,
                    'model_variant': model_variant,
                    'trans_weight': trans_weight
                }
            }, save_path / 'best_model.pth')
            print(f"  New best model saved!")
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    return history


def example_forward_pass():
    """Example showing RGB-D fusion network usage with residual learning."""
    model = RGBDFusionNetwork(
        num_points=500,
        rgb_feature_dim=256,
        point_feature_dim=256,
        fused_dim=512,
        pretrained_rgb=True
    )
    model.eval()
    
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 128, 128)
    points = torch.randn(batch_size, 500, 3)
    
    T_geo = torch.tensor([
        [0.05, 0.02, 0.80],
        [0.03, -0.01, 0.75],
        [-0.02, 0.04, 0.90],
        [0.01, 0.00, 0.85],
    ])
    
    with torch.no_grad():
        quaternion, delta_t, T_final = model(rgb, points, T_geo)
        
    print("=== RGB-D Fusion Network Output ===")
    print(f"Quaternion shape: {quaternion.shape}")
    print(f"Delta T shape: {delta_t.shape}")
    print(f"T_final shape: {T_final.shape}")
    
    print("\n=== Sample Output (first batch element) ===")
    print(f"Quaternion (x,y,z,w): {quaternion[0].numpy()}")
    print(f"T_geo (geometric anchor): {T_geo[0].numpy()} meters")
    print(f"Delta T (learned residual): {delta_t[0].numpy()} meters")
    print(f"T_final = T_geo + Delta T: {T_final[0].numpy()} meters")
    
    assert torch.allclose(T_final, T_geo + delta_t), "Residual learning check failed!"
    print("\nResidual learning verified: T_final = T_geo + Delta T")
    
    return quaternion, delta_t, T_final
