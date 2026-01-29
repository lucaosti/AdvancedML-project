# Technical Documentation

## Overview

6D pose estimation using hybrid learning-geometry approaches:
- **Rotation**: Learned via neural networks
- **Translation**: Learned with Pinhole Camera Model as input guidance

## Pipelines

| Pipeline | Input | Method |
|----------|-------|--------|
| RGB | RGB + Pinhole | ResNet learns 6D pose with pinhole translation as input |
| Residual Learning | RGB + Depth | MLP-based residual learning |
| DenseFusion Iterative | RGB + Depth | Per-pixel fusion + refinement |
| PVN3D | RGB + Depth | Keypoint voting |
| FFB6D | RGB + Depth | Bidirectional multi-scale fusion |

## RGB Pipeline (Baseline)

The RGB baseline follows this architecture:
```
YOLO -> Bounding Boxes -> Translation (Pinhole Camera Model) -> ResNet -> 6D Pose
```

The Pinhole Camera Model provides an initial translation estimate which is fed
as INPUT to the ResNet. The network then learns to predict the full 6D pose
(both rotation AND translation).

Components:
- **Detection**: YOLO fine-tuned on LineMOD, filtered by target class
- **Pinhole Translation (Input)**: Initial estimate computed geometrically
  - X, Y: From bbox center using camera intrinsics
  - Z (depth): Estimated from bbox size and known object dimensions
- **ResNet**: Takes RGB crop + pinhole translation, outputs 6D pose
- **Rotation**: Learned as unit quaternion [w, x, y, z]
- **Translation**: Learned, with pinhole estimate guiding the network

**Model**: `rgb_pipeline/model.py`

```python
from rgb_pipeline import RGBPoseEstimator

model = RGBPoseEstimator(backbone='resnet50', pretrained=True)
# Model takes RGB image + pinhole translation estimate as input
rotation, translation = model(rgb_image, pinhole_translation)
# rotation: [B, 4] unit quaternion
# translation: [B, 3] in meters
```

**Loss**: Combined rotation + translation loss
- Rotation: GeodesicLoss - angular distance on SO(3): `arccos(|<q_pred, q_gt>|)`
- Translation: SmoothL1Loss on 3D coordinates

**Pinhole Translation Estimation** (input to network):
The depth is estimated from bounding box size using similar triangles:
```
Z = (focal_length * real_object_size) / projected_size
```
Then X, Y are computed from the bbox center using the pinhole model:
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

## RGB-D Pipelines

**Dataset**: All pipelines (RGB and RGB-D) use `LineMODDataset` from `utils/dataset.py`:
- Mode `'rgb'`: Loads only RGB images (no depth) for the RGB baseline
- Mode `'rgbd'`: Loads RGB, depth, and generates point clouds for RGB-D pipelines
- Filters gt.yml to only include data for the current object
- Uses YOLO for object detection with class-specific filtering
- Pre-computes YOLO bounding boxes once per object for efficiency

### Residual Learning

MLP-based residual learning with geometric anchors:
- RGB Crop -> ResNet18 Encoder
- Point Cloud -> PointNet Encoder
- Fusion Module
- Rotation Residual Head -> delta_R (quaternion)
- Translation Residual Head -> delta_T
- Final: R_final = R_geo * delta_R, T_final = T_geo + delta_T

Geometric anchors (R_geo, T_geo) are computed from YOLO detections and depth information.

### DenseFusion Iterative

Per-pixel fusion with refinement:
- RGB: PSPNet -> dense features
- Points: PointNet -> per-point features
- Per-point pose estimation with confidence
- Iterative refinement

### PVN3D

Keypoint voting:
- Each point votes for K keypoints
- Votes aggregated via mean

### FFB6D

Bidirectional fusion at every encoder level:
- RGB <-> Point feature exchange at multiple scales
- Same keypoint voting as PVN3D

## Mathematical Background

### Rotation

**Quaternion**: q = (w, x, y, z), ||q|| = 1
**Double cover**: q and -q represent same rotation

**Quaternion to Matrix**:
```
R[0,0] = 1 - 2(y^2 + z^2)
R[0,1] = 2(xy - wz)
R[0,2] = 2(xz + wy)
...
```

### Pinhole Camera

**Back-projection** (2D + depth -> 3D):
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

**Depth from bbox** (similar triangles):
```
Z = (f * real_size) / projected_size
```

### ADD Metric

Average distance between transformed model points:
```
ADD = (1/M) * sum(||R_pred * p + t_pred - R_gt * p - t_gt||)
```

Success if ADD < 0.1 * diameter

## Dataset

**LineMOD**: 13 objects (ape, benchvise, camera, can, cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)

## Unit Conventions

**All spatial values are in METERS.**

| Data | Unit | Source | Notes |
|------|------|--------|-------|
| Point clouds | meters | Depth back-projection | Depth mm → meters (×0.001) |
| Ground truth translation | meters | gt.yml | Converted from mm |
| Predicted translation | meters | Model output | Direct output |
| Geometric anchors (T_geo) | meters | Pinhole + depth | Computed from bbox + depth |
| Object models | meters | PLY files | Converted from mm if needed |
| Depth maps | meters | Dataset | Scaled from raw mm values |

**Evaluation Output (for readability)**:
- **Rotation errors**: DEGREES
- **Translation errors**: CENTIMETERS (computed from meters, ×100)

**residual_learning model evaluation**

The residual_learning model uses geometric anchors (T_geo) computed from:
1. YOLO bounding box (absolute pixel coordinates)
2. Depth crop
3. Camera intrinsics
4. Crop offsets

Using point cloud centroid instead of proper T_geo causes ~4cm error!
Always use `get_geometric_anchor_batch()` for correct evaluation.

**Bounding Box Formats**:
- `bbox_abs`: (x, y, w, h) - YOLO prediction, pixels, top-left + dimensions
- `bbox_gt`: (x1, y1, x2, y2) - Ground truth, pixels, corner coordinates

**Dataset Loader** (`utils/dataset.py`):
- **LineMODDataset**: Unified dataloader supporting both RGB and RGBD modes
  - `mode='rgb'`: Loads RGB images only (no depth data)
  - `mode='rgbd'`: Loads RGB, depth, and generates point clouds
  - Filters gt.yml by target object ID
  - Uses YOLO for object detection with class-specific filtering
  - Pre-computes YOLO bounding boxes once per object for efficiency
- **Bounding Box Format**: All bboxes are in xywh format (x, y, width, height), normalized to [0, 1]
- **YOLO Configuration**: 
  - Low confidence threshold (0.01) for fine-tuned models
  - Class filtering to detect only the target object
  - 10% padding by default (configurable)
