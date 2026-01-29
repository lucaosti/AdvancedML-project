"""
Training script for the RGB 6D pose estimator.
Uses pinhole camera model for initial translation estimate as network input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
from typing import Dict, Tuple

from utils.geometry import rotation_matrix_to_quaternion_torch
from utils.pinhole_camera import estimate_translation_from_bbox
from rgb_pipeline import config as cfg


LAMBDA_ROT = 10.0
LAMBDA_TRANS = 1.0


class GeodesicLoss(nn.Module):
    """Geodesic distance loss for quaternions on SO(3) manifold."""
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        dot = torch.abs(torch.sum(pred * target, dim=1))
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(dot).mean()


def compute_angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute angular error between quaternions in degrees."""
    pred = F.normalize(pred, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    dot = torch.clamp(torch.sum(pred * target, dim=1), -1.0, 1.0)
    theta_rad = 2 * torch.acos(torch.abs(dot))
    return torch.rad2deg(theta_rad).mean()


def estimate_depth_from_bbox(
    bbox_hw: Tuple[float, float],
    object_name: str,
    shrink_factor: float = 0.15
) -> float:
    """Estimate object depth from bounding box size using similar triangles."""
    h_px, w_px = bbox_hw
    
    # Apply shrink factor to compensate for YOLO padding
    effective_h = h_px * (1.0 - shrink_factor)
    effective_w = w_px * (1.0 - shrink_factor)
    
    # Get object dimensions
    obj_dims = cfg.OBJECT_DIMENSIONS.get(object_name, {})
    real_h = obj_dims.get('height', 0.1)  # meters
    real_w = obj_dims.get('width', 0.1)   # meters
    diameter = obj_dims.get('diameter', 0.1)  # meters
    
    depth_estimates = []
    
    # From height
    if effective_h > 10:
        z_from_h = (cfg.FY * real_h) / effective_h
        depth_estimates.append(z_from_h)
    
    # From width
    if effective_w > 10:
        z_from_w = (cfg.FX * real_w) / effective_w
        depth_estimates.append(z_from_w)
    
    # From diagonal (using diameter)
    diag_px = (effective_h**2 + effective_w**2)**0.5
    if diag_px > 10:
        f_avg = (cfg.FX + cfg.FY) / 2
        z_from_diag = (f_avg * diameter) / diag_px
        depth_estimates.append(z_from_diag)
    
    if not depth_estimates:
        return 1.0  # Fallback default depth
    
    return sum(depth_estimates) / len(depth_estimates)


def compute_pinhole_translations(
    bboxes: torch.Tensor,
    object_name: str,
    im_w: int = 640,
    im_h: int = 480,
    shrink_factor: float = 0.15
) -> torch.Tensor:
    """Compute pinhole translation estimates for a batch of bounding boxes."""
    B = bboxes.shape[0]
    translations = []
    
    for i in range(B):
        x_norm, y_norm, w_norm, h_norm = bboxes[i]
        
        x_px = x_norm.item() * im_w
        y_px = y_norm.item() * im_h
        w_px = w_norm.item() * im_w
        h_px = h_norm.item() * im_h
        
        if w_px <= 0 or h_px <= 0:
            translations.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            continue
        
        estimated_depth = estimate_depth_from_bbox(
            bbox_hw=(h_px, w_px),
            object_name=object_name,
            shrink_factor=shrink_factor
        )
        
        bbox_xywh = (x_px, y_px, w_px, h_px)
        t = estimate_translation_from_bbox(bbox_xywh, estimated_depth, cfg.CAMERA_INTRINSICS)
        translations.append(t)
    
    return torch.tensor(np.stack(translations), dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, device, object_name):
    """Execute a single training epoch."""
    model.train()
    running_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0
    
    rot_criterion = GeodesicLoss()
    trans_criterion = nn.SmoothL1Loss()

    for batch in train_loader:
        if batch is None:
            continue
        
        images = batch['rgb'].to(device)
        gt_rot_matrix = batch['rotation'].to(device)
        gt_trans = batch['translation'].to(device)
        bboxes = batch['bbox']
        
        pinhole_trans = compute_pinhole_translations(bboxes, object_name).to(device)
        gt_rot = F.normalize(rotation_matrix_to_quaternion_torch(gt_rot_matrix), p=2, dim=1)
        
        pred_rot, pred_trans = model(images, pinhole_trans)
        loss_rot = rot_criterion(pred_rot, gt_rot) * LAMBDA_ROT
        loss_trans = trans_criterion(pred_trans, gt_trans) * LAMBDA_TRANS
        loss = loss_rot + loss_trans

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        running_rot_loss += loss_rot.item()
        running_trans_loss += loss_trans.item()

    n = len(train_loader)
    return running_loss / n, running_rot_loss / n, running_trans_loss / n


def validate(model, val_loader, device, object_name):
    """Compute validation loss and metrics."""
    model.eval()
    val_loss = 0.0
    angle_errors = []
    trans_errors = []

    rot_criterion = GeodesicLoss()
    trans_criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            
            images = batch['rgb'].to(device)
            gt_rot_matrix = batch['rotation'].to(device)
            gt_trans = batch['translation'].to(device)
            bboxes = batch['bbox']
            
            pinhole_trans = compute_pinhole_translations(bboxes, object_name).to(device)
            gt_rot = F.normalize(rotation_matrix_to_quaternion_torch(gt_rot_matrix), p=2, dim=1)
            
            pred_rot, pred_trans = model(images, pinhole_trans)
            
            loss_rot = rot_criterion(pred_rot, gt_rot) * LAMBDA_ROT
            loss_trans = trans_criterion(pred_trans, gt_trans) * LAMBDA_TRANS
            val_loss += (loss_rot + loss_trans).item()
            
            angle_err = compute_angular_error(pred_rot, gt_rot)
            angle_errors.append(angle_err.item())
            
            trans_err = torch.norm(pred_trans - gt_trans, dim=1).mean()
            trans_errors.append(trans_err.item())

    n_batches = len(val_loader)
    avg_angle_err = np.mean(angle_errors)
    avg_trans_err = np.mean(trans_errors)
    
    return val_loss / n_batches, avg_angle_err, avg_trans_err


def train_rgb_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    save_dir: str,
    object_name: str,
    early_stopping_patience: int = 20
) -> Dict[str, list]:
    """Main training loop for RGB pose estimator. Returns training history."""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=cfg.ConfigRGB.LEARNING_RATE_END
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_angle_errors': [],
        'val_trans_errors': []
    }

    print(f"Training RGB model on {object_name} [{device}]")
    print(f"(6D pose: rotation + translation learned, pinhole as input)")

    for epoch in range(num_epochs):
        train_loss, train_rot, train_trans = train_epoch(
            model, train_loader, optimizer, device, object_name
        )
        val_loss, val_angle_err, val_trans_err = validate(
            model, val_loader, device, object_name
        )
        scheduler.step()

        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['val_angle_errors'].append(val_angle_err)
        history['val_trans_errors'].append(val_trans_err)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            epochs_without_improvement += 1

        print(f"[{epoch+1:3d}/{num_epochs}] "
              f"loss={val_loss:.4f} "
              f"angle={val_angle_err:.1f}deg "
              f"trans={val_trans_err*100:.2f}cm "
              f"{'*' if improved else ''}")
        
        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
            break
    
    print(f"Done. Best loss: {best_val_loss:.4f}")
    return history
