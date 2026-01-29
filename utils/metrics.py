"""Evaluation metrics: ADD, IoU, and helper functions."""

import numpy as np


def transform_points(pose, points):
    """Apply 4x4 pose transformation to 3D points."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    return np.dot(points, R.T) + t


def average_distance_metric(pred_pose, gt_pose, model_points, threshold=0.1):
    """Compute Average Distance Metric (ADD)."""
    pred_points_transformed = transform_points(pred_pose, model_points)
    gt_points_transformed = transform_points(gt_pose, model_points)
    distances = np.linalg.norm(pred_points_transformed - gt_points_transformed, axis=1)
    add_distance = np.mean(distances)
    is_correct = bool(add_distance < threshold)
    return add_distance, is_correct


def intersection_over_union(pred_mask, gt_mask):
    """Compute IoU between predicted and ground truth masks."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


def add_metric_batch(pred_poses, gt_poses, model_points, threshold=0.1):
    """Compute ADD metric for a batch of poses."""
    total_add = 0.0
    correct_count = 0
    batch_size = len(pred_poses)

    for i in range(batch_size):
        dist, is_correct = average_distance_metric(
            pred_poses[i],
            gt_poses[i],
            model_points,
            threshold
        )
        total_add += dist
        if is_correct:
            correct_count += 1

    return total_add / batch_size, correct_count / batch_size