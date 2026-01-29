"""ADD metric computation for RGB-D variant."""

import numpy as np
import torch
from utils.metrics import add_metric_batch
from utils.geometry import quaternion_to_rotation_matrix


def evaluate_add_rgbd(model, dataloader, model_points, device='cuda', threshold=0.1):
    """Evaluate ADD metric on test set for RGB-D model."""
    model.eval()
    all_add_scores = []

    print("Evaluating ADD Metric...")

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)

            gt_pose = batch['pose'].numpy()

            pred_trans, pred_quat = model(rgb, depth)

            pred_rot_mat = quaternion_to_rotation_matrix(pred_quat)

            B = pred_trans.shape[0]
            pred_pose = np.eye(4)[None, :, :].repeat(B, axis=0)
            pred_pose[:, :3, :3] = pred_rot_mat.cpu().numpy()
            pred_pose[:, :3, 3] = pred_trans.cpu().numpy()

            scores = add_metric_batch(pred_pose, gt_pose, model_points)
            all_add_scores.extend(scores)

    all_add_scores = np.array(all_add_scores)
    success_count = np.sum(all_add_scores < threshold)
    success_rate = success_count / len(all_add_scores)

    return all_add_scores, success_rate
