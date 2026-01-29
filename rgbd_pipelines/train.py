"""
Training script for RGB-D pose estimation models.
"""

import os
import argparse
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from rgbd_pipelines import create_model, list_models
from utils.dataset import LineMODDataset, rgbd_collate_fn
from rgbd_pipelines.common.geometric_anchor import get_geometric_anchor_batch
from rgb_pipeline.config import ConfigInference


class EarlyStopping:
    """Stops training when validation loss stops improving."""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(model, dataloader, optimizer, device, obj_idx):
    """Execute a single training epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        n_batches += 1
        rgb = batch['rgb'].to(device)
        points = batch['points'].to(device)
        choose = batch['choose'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)
        
        B = rgb.size(0)
        obj_idx_tensor = torch.full((B,), obj_idx, dtype=torch.long, device=device)
        
        T_geo = None
        R_geo = None
        if hasattr(model, 'model_name') and 'residual' in model.model_name:
            if 'bbox' in batch and 'intrinsics' in batch:
                bboxes = batch['bbox'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                depth_avg = points[:, :, 2].mean(dim=1, keepdim=True)
                fx = intrinsics[:, 0, 0] if intrinsics.dim() == 3 else intrinsics[0, 0].unsqueeze(0).expand(B)
                fy = intrinsics[:, 1, 1] if intrinsics.dim() == 3 else intrinsics[1, 1].unsqueeze(0).expand(B)
                cx = intrinsics[:, 0, 2] if intrinsics.dim() == 3 else intrinsics[0, 2].unsqueeze(0).expand(B)
                cy = intrinsics[:, 1, 2] if intrinsics.dim() == 3 else intrinsics[1, 2].unsqueeze(0).expand(B)
                
                cx_bbox = bboxes[:, 0] + bboxes[:, 2] / 2
                cy_bbox = bboxes[:, 1] + bboxes[:, 3] / 2
                
                Z = depth_avg.squeeze(1)
                X = (cx_bbox - cx) * Z / fx
                Y = (cy_bbox - cy) * Z / fy
                
                T_geo = torch.stack([X, Y, Z], dim=1)
                R_geo = torch.eye(3, device=device, dtype=rgb.dtype).unsqueeze(0).expand(B, -1, -1)
        
        pred_dict = model(rgb, points, choose, obj_idx_tensor, T_geo=T_geo, R_geo=R_geo)
        loss_dict = model.get_loss(pred_dict, gt_R, gt_t, model_points=None, points=points)
        loss = loss_dict['total']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, obj_idx):
    """Compute validation metrics."""
    model.eval()
    total_loss = 0.0
    total_rot_err = 0.0
    total_trans_err = 0.0
    n_samples = 0
    
    for batch in dataloader:
        # Skip batch if all samples had YOLO detection failures
        if batch is None:
            continue
        
        rgb = batch['rgb'].to(device)
        points = batch['points'].to(device)
        choose = batch['choose'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)
        
        B = rgb.size(0)
        obj_idx_tensor = torch.full((B,), obj_idx, dtype=torch.long, device=device)
        
        # Compute geometric anchors for residual learning models
        T_geo = None
        R_geo = None
        if hasattr(model, 'model_name') and 'residual' in model.model_name:
            if 'bbox' in batch and 'intrinsics' in batch:
                bboxes = batch['bbox'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                # Compute translation anchor from bbox and average depth of points
                depth_avg = points[:, :, 2].mean(dim=1, keepdim=True)  # [B, 1]
                fx = intrinsics[:, 0, 0] if intrinsics.dim() == 3 else intrinsics[0, 0].unsqueeze(0).expand(B)
                fy = intrinsics[:, 1, 1] if intrinsics.dim() == 3 else intrinsics[1, 1].unsqueeze(0).expand(B)
                cx = intrinsics[:, 0, 2] if intrinsics.dim() == 3 else intrinsics[0, 2].unsqueeze(0).expand(B)
                cy = intrinsics[:, 1, 2] if intrinsics.dim() == 3 else intrinsics[1, 2].unsqueeze(0).expand(B)
                cx_bbox = bboxes[:, 0] + bboxes[:, 2] / 2
                cy_bbox = bboxes[:, 1] + bboxes[:, 3] / 2
                Z = depth_avg.squeeze(1)
                X = (cx_bbox - cx) * Z / fx
                Y = (cy_bbox - cy) * Z / fy
                T_geo = torch.stack([X, Y, Z], dim=1)
                R_geo = torch.eye(3, device=device, dtype=rgb.dtype).unsqueeze(0).expand(B, -1, -1)
        
        pred_dict = model(rgb, points, choose, obj_idx_tensor, T_geo=T_geo, R_geo=R_geo)
        loss_dict = model.get_loss(pred_dict, gt_R, gt_t, model_points=None, points=points)
        
        total_loss += loss_dict['total'].item()
        
        pred_R = pred_dict['rotation']
        R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        total_rot_err += angle.sum().item() * 180 / 3.14159
        
        pred_t = pred_dict['translation']
        trans_err = torch.norm(pred_t - gt_t, dim=1)
        total_trans_err += trans_err.sum().item() * 100  # to cm
        
        n_samples += B
    
    return {
        'loss': total_loss / len(dataloader),
        'rot_err': total_rot_err / n_samples,
        'trans_err': total_trans_err / n_samples
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list_models())
    parser.add_argument('--object', type=str, default='ape')
    parser.add_argument('--data_root', type=str, default='data/Linemod_preprocessed/data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_iter', type=int, default=2)
    parser.add_argument('--num_kp', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='trained_checkpoints_rgbd')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model} on {args.object} [{device}]")
    
    obj_map = {'ape': 0, 'benchvise': 1, 'camera': 2, 'can': 3, 'cat': 4,
               'driller': 5, 'duck': 6, 'eggbox': 7, 'glue': 8,
               'holepuncher': 9, 'iron': 10, 'lamp': 11, 'phone': 12}
    obj_idx = obj_map.get(args.object.lower(), 0)
    
    model_kwargs = {'num_points': args.num_points, 'num_obj': 13}
    if 'iterative' in args.model:
        model_kwargs['num_iter'] = args.num_iter
    if args.model in ['pvn3d', 'ffb6d']:
        model_kwargs['num_kp'] = args.num_kp
    
    model = create_model(args.model, **model_kwargs).to(device)
    
    # Use consistent YOLO path from config (same as RGB pipeline)
    yolo_cfg = ConfigInference()
    
    # Use unified dataset in rgbd mode
    train_dataset = LineMODDataset(
        root_dir=args.data_root, object_name=args.object, split='train',
        mode='rgbd', num_points=args.num_points,
        yolo_model_path=yolo_cfg.YOLO_PATH, yolo_padding_pct=0.1
    )
    val_dataset = LineMODDataset(
        root_dir=args.data_root, object_name=args.object, split='test',
        mode='rgbd', num_points=args.num_points,
        yolo_model_path=yolo_cfg.YOLO_PATH, yolo_padding_pct=0.1
    )
    
    # Windows compatibility: use 0 workers to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=num_workers,
                              collate_fn=rgbd_collate_fn,
                              pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=rgbd_collate_fn,
                            pin_memory=True if torch.cuda.is_available() else False)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    early_stopper = EarlyStopping(patience=args.patience) if args.early_stopping else None
    
    save_dir = os.path.join(args.save_dir, args.model, args.object)
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'rot_err': [], 'trans_err': []}
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, obj_idx)
        val_metrics = validate(model, val_loader, device, obj_idx)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['rot_err'].append(val_metrics['rot_err'])
        history['trans_err'].append(val_metrics['trans_err'])
        
        improved = val_metrics['loss'] < best_loss
        if improved:
            best_loss = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
        
        print(f"[{epoch+1:3d}/{args.epochs}] "
              f"loss={val_metrics['loss']:.4f} "
              f"rot={val_metrics['rot_err']:.1f}deg "
              f"trans={val_metrics['trans_err']:.1f}cm "
              f"{'*' if improved else ''}")
        
        if early_stopper and early_stopper(val_metrics['loss']):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    print(f"Done. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
