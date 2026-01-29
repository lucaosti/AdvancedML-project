"""Geometric utilities: quaternion operations, transformations, PLY loading."""

import numpy as np
import torch
import os


def rotation_matrix_to_quaternion_torch(R):
    """Convert batched rotation matrices [B, 3, 3] to quaternions [B, 4] (x, y, z, w)."""
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, dtype=R.dtype, device=R.device)
    
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    mask1 = tr > 0
    if mask1.any():
        S = torch.sqrt(tr[mask1] + 1.0) * 2
        q[mask1, 3] = 0.25 * S
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
        q[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S
        q[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S
    
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        S = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S
        q[mask2, 0] = 0.25 * S
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
        q[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S
    
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        S = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S
        q[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S
        q[mask3, 1] = 0.25 * S
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S
    
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        S = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S
        q[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S
        q[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S
        q[mask4, 2] = 0.25 * S
    
    if squeeze_output:
        q = q.squeeze(0)
    
    return q


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix (numpy)."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return R


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w) (numpy)."""
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([x, y, z, w])


def pose_to_matrix(rotation, translation):
    """Convert rotation (3x3 or quaternion) + translation to 4x4 transformation matrix."""
    rotation = np.array(rotation)
    if rotation.shape == (4,):
        R = quaternion_to_rotation_matrix(rotation)
    else:
        R = rotation

    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = translation
    return matrix


def matrix_to_pose(T):
    """Extract rotation (3x3) and translation (3,) from 4x4 transformation matrix."""
    rotation = T[:3, :3]
    translation = T[:3, 3]
    return rotation, translation


def project_3d_to_2d(points_3d, intrinsics):
    """Project 3D points to 2D pixel coordinates using camera intrinsics."""
    projected = np.dot(intrinsics, points_3d.T).T
    z = projected[:, 2:3]
    z[z == 0] = 1e-5
    points_2d = projected[:, :2] / z
    return points_2d


def quaternion_to_rotation_matrix_torch(q):
    """Convert batched quaternions [B, 4] (x, y, z, w) to rotation matrices [B, 3, 3]."""
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q)

    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    batch_size = q.shape[0]
    R = torch.zeros(batch_size, 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    if squeeze_output:
        R = R.squeeze(0)

    return R


def load_ply_vertices(ply_path):
    """Load PLY vertices for ADD metric computation. Returns Nx3 array in meters."""
    if not os.path.exists(ply_path):
        return None
    
    vertices = []
    header_ended = False
    num_verts = 0
    
    with open(ply_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith("element vertex"):
            num_verts = int(line.split()[-1])
        elif line == "end_header":
            header_ended = True
            continue
        
        if header_ended:
            if len(vertices) >= num_verts:
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    
    if len(vertices) == 0:
        return None
    
    pts = np.array(vertices, dtype=np.float32)
    
    if np.max(np.abs(pts)) > 10.0:
        pts /= 1000.0
    
    return pts