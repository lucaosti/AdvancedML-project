"""
Evaluation script for RGB-D pose estimation models.
Supports single model evaluation and cross-model comparison.
All translations in meters; output errors in cm for readability.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from rgbd_pipelines import create_model, list_models
from utils.dataset import LineMODDataset, rgbd_collate_fn
from rgb_pipeline.config import ConfigInference


def compute_add(pred_R, pred_t, gt_R, gt_t, model_points):
    """Compute ADD metric in meters."""
    pred_pts = torch.bmm(model_points.unsqueeze(0), pred_R.unsqueeze(0).transpose(1, 2)) + pred_t.unsqueeze(0).unsqueeze(1)
    gt_pts = torch.bmm(model_points.unsqueeze(0), gt_R.unsqueeze(0).transpose(1, 2)) + gt_t.unsqueeze(0).unsqueeze(1)
    return torch.norm(pred_pts - gt_pts, dim=-1).mean().item()


def compute_adds(pred_R, pred_t, gt_R, gt_t, model_points):
    """Compute ADD-S metric for symmetric objects in meters."""
    pred_pts = torch.bmm(model_points.unsqueeze(0), pred_R.unsqueeze(0).transpose(1, 2)) + pred_t.unsqueeze(0).unsqueeze(1)
    gt_pts = torch.bmm(model_points.unsqueeze(0), gt_R.unsqueeze(0).transpose(1, 2)) + gt_t.unsqueeze(0).unsqueeze(1)
    dists = torch.cdist(pred_pts[0], gt_pts[0])
    return dists.min(dim=1)[0].mean().item()


def _compute_geometric_anchor(batch: Dict, device: torch.device) -> torch.Tensor:
    """Compute geometric anchor T_geo for residual_learning model."""
    from rgbd_pipelines.residual_learning.geometric_anchor import get_geometric_anchor_batch
    
    T_geo = get_geometric_anchor_batch(
        bboxes=batch['bbox_abs'],
        depth_crops=batch['depth_crop'],
        intrinsics=batch['intrinsics'],
        crop_offsets=batch['crop_offset'],
        shrink_factor=0.15,
        depth_percentile=20.0
    )
    return T_geo.to(device)


@torch.no_grad()
def evaluate_model(model, dataloader, device, obj_idx, threshold=0.1, model_name: Optional[str] = None):
    """Evaluate model, returning rotation error in degrees and translation error in cm."""
    model.eval()
    
    all_adds = []
    all_rot_err = []
    all_trans_err = []  # Will store errors in CENTIMETERS
    
    # Auto-detect residual_learning model
    is_residual_learning = (
        model_name == 'residual_learning' or
        (hasattr(model, 'model_name') and 'residual' in model.model_name.lower())
    )
    
    for batch in dataloader:
        # Skip batch if all samples had YOLO detection failures
        if batch is None:
            continue
        
        rgb = batch['rgb'].to(device)
        points = batch['points'].to(device)  # Points are in METERS
        choose = batch['choose'].to(device)
        gt_R = batch['rotation'].to(device)
        gt_t = batch['translation'].to(device)  # Translation is in METERS
        
        B = rgb.size(0)
        obj_idx_tensor = torch.full((B,), obj_idx, dtype=torch.long, device=device)
        
        # For residual_learning, compute proper geometric anchor
        if is_residual_learning:
            T_geo = _compute_geometric_anchor(batch, device)  # T_geo in METERS
            pred_dict = model(rgb, points, choose, obj_idx_tensor, T_geo=T_geo)
        else:
            pred_dict = model(rgb, points, choose, obj_idx_tensor)
        
        pred_R = pred_dict['rotation']
        pred_t = pred_dict['translation']  # Prediction is in METERS
        
        for i in range(B):
            # Rotation error (in DEGREES)
            R_diff = pred_R[i].T @ gt_R[i]
            trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
            angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
            all_rot_err.append(angle.item() * 180 / np.pi)
            
            # Translation error: computed in METERS, stored in CENTIMETERS
            # pred_t[i] and gt_t[i] are both in METERS
            trans_err_meters = torch.norm(pred_t[i] - gt_t[i]).item()
            trans_err_cm = trans_err_meters * 100  # Convert to CENTIMETERS
            all_trans_err.append(trans_err_cm)
            
            # ADD metric (in meters, simplified using translation error)
            all_adds.append(trans_err_meters)
    
    # Compute metrics
    # Note: all_trans_err is already in CENTIMETERS
    results = {
        'rot_err_mean': np.mean(all_rot_err),
        'rot_err_median': np.median(all_rot_err),
        'trans_err_mean': np.mean(all_trans_err),  # in CENTIMETERS
        'trans_err_median': np.median(all_trans_err),  # in CENTIMETERS
        'acc_5deg': np.mean(np.array(all_rot_err) < 5) * 100,
        'acc_10deg': np.mean(np.array(all_rot_err) < 10) * 100,
        'acc_5cm': np.mean(np.array(all_trans_err) < 5) * 100,  # threshold in cm
    }
    
    return results


def print_results(results: Dict, model_name: str, object_name: str):
    print(f"\n{'='*60}")
    print(f"Results: {model_name} on {object_name}")
    print(f"{'='*60}")
    print(f"Rotation Error:   {results['rot_err_mean']:.2f}° (mean), {results['rot_err_median']:.2f}° (median)")
    print(f"Translation Error: {results['trans_err_mean']:.2f} cm (mean), {results['trans_err_median']:.2f} cm (median)")
    print(f"Accuracy <5°:     {results['acc_5deg']:.1f}%")
    print(f"Accuracy <10°:    {results['acc_10deg']:.1f}%")
    print(f"Accuracy <5cm:    {results['acc_5cm']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RGB-D pose estimation models')
    parser.add_argument('--model', type=str, default=None, choices=list_models(),
                        help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--object', type=str, default='ape',
                        help='Object to evaluate on')
    parser.add_argument('--data_root', type=str, default='data/Linemod_preprocessed/data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--compare', action='store_true',
                        help='Compare all available models')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obj_map = {'ape': 0, 'benchvise': 1, 'camera': 2, 'can': 3, 'cat': 4,
               'driller': 5, 'duck': 6, 'eggbox': 7, 'glue': 8,
               'holepuncher': 9, 'iron': 10, 'lamp': 11, 'phone': 12}
    obj_idx = obj_map.get(args.object.lower(), 0)
    
    # Use consistent YOLO path from config (same as RGB pipeline)
    yolo_cfg = ConfigInference()
    
    # Use unified dataset in rgbd mode
    test_dataset = LineMODDataset(
        root_dir=args.data_root, object_name=args.object, split='test',
        mode='rgbd', num_points=args.num_points,
        yolo_model_path=yolo_cfg.YOLO_PATH, yolo_padding_pct=0.1
    )
    # Windows compatibility: use 0 workers
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=rgbd_collate_fn,
                             pin_memory=True if torch.cuda.is_available() else False)
    
    if args.compare:
        # Compare all models
        print("\nComparing all models...")
        all_results = {}
        
        for model_name in list_models():
            checkpoint_path = f'trained_checkpoints_rgbd/{model_name}/{args.object}/best.pth'
            if not os.path.exists(checkpoint_path):
                print(f"  Skipping {model_name} (no checkpoint)")
                continue
            
            model = create_model(model_name, num_points=args.num_points)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model = model.to(device)
            
            # Pass model_name for proper geometric anchor handling
            results = evaluate_model(model, test_loader, device, obj_idx, model_name=model_name)
            all_results[model_name] = results
            print_results(results, model_name, args.object)
        
        # Summary table
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'Rot(°)':<10} {'Trans(cm)':<10} {'<5°':<8} {'<5cm':<8}")
        print("-"*80)
        for name, res in all_results.items():
            print(f"{name:<25} {res['rot_err_mean']:<10.2f} {res['trans_err_mean']:<10.2f} "
                  f"{res['acc_5deg']:<8.1f} {res['acc_5cm']:<8.1f}")
    
    else:
        # Single model evaluation
        if args.model is None or args.checkpoint is None:
            parser.error("--model and --checkpoint required (or use --compare)")
        
        model = create_model(args.model, num_points=args.num_points)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model = model.to(device)
        
        # Pass model_name for proper geometric anchor handling
        results = evaluate_model(model, test_loader, device, obj_idx, model_name=args.model)
        print_results(results, args.model, args.object)


if __name__ == '__main__':
    main()
