# AML - 6D Pose Estimation

6D object pose estimation using hybrid geometric-learning approaches.

## Authors

This project was developed as part of the **Advanced Machine Learning** course examination at [Politecnico di Torino](https://www.polito.it/), under the supervision of Professor [Tatiana Tommasi](https://www.polito.it/personale?p=051287) and Assistant Professor [Stephany Ortuno Chanelo](https://www.polito.it/personale?p=stephany.ortuno).

| Name | Email |
|------|-------|
| [Matin Bayramli](https://github.com/matinbyrml) | s330991@studenti.polito.it |
| [Alessio Meini](https://github.com/Tartayoshi) | alessio.meini@studenti.polito.it |
| [Luca Ostinelli](https://github.com/lucaosti) | luca.ostinelli@studenti.polito.it |
| [Luca Visconti](https://github.com/longobucco) | s348650@studenti.polito.it |

## Overview

| Approach | Input | Method |
|----------|-------|--------|
| RGB Pipeline | RGB only | ResNet + Pinhole Camera Model |
| Residual Learning | RGB + Depth | MLP-based residual learning |
| DenseFusion Iterative | RGB + Depth | Per-pixel fusion + refinement |
| PVN3D | RGB + Depth | Keypoint voting |
| FFB6D | RGB + Depth | Bidirectional fusion |

## Project Structure

```
AML/
├── rgb_pipeline/           # RGB-only baseline
│   ├── model.py            # RGBPoseEstimator
│   ├── train.py            # Training script
│   └── config.py
│
├── rgbd_pipelines/         # RGB-D models
│   ├── __init__.py         # Factory: create_model(...)
│   ├── base.py             # BasePoseModel interface
│   ├── dataset.py          # UnifiedRGBDDataset
│   ├── train.py
│   ├── evaluate.py
│   ├── backbones/          # ResNet, PointNet
│   ├── residual_learning/
│   ├── densefusion_iterative/
│   ├── pvn3d/
│   └── ffb6d/
│
├── utils/                  # Shared utilities
│   ├── dataset.py          # Main dataset loader (LineMODDataset)
│   ├── dataset_unified.py  # Re-exports for convenience
│   ├── geometry.py         # Geometry and PLY loading
│   ├── pinhole_camera.py   # Pinhole camera model
│   ├── bbox.py             # Bounding box utilities
│   ├── metrics.py          # ADD/ADD-S metrics
│   └── visualization.py    # Visualization helpers
├── scripts/                # Run scripts
└── data/                   # LineMOD dataset
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

#### Option 1: Command-Line Script

```bash
# Train all models on all objects
python scripts/master_pipeline.py

# Train specific objects
python scripts/master_pipeline.py --objects ape cat duck

# Train specific pipelines
python scripts/master_pipeline.py --pipelines rgb residual_learning

# Customize training parameters
python scripts/master_pipeline.py --epochs_rgb 100 --epochs_rgbd 150
```

#### Option 2: Jupyter Notebooks

```bash
cd notebooks
jupyter notebook
```

**Available Notebooks:**
- `evaluation.ipynb` - Model evaluation and comparison
- `visualization.ipynb` - Results visualization with tables and graphs

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete details.

### RGB Pipeline

```python
from rgb_pipeline import RGBPoseEstimator

model = RGBPoseEstimator(backbone='resnet50')
rotation, translation = model(rgb_image, pinhole_translation)
# rotation: [B, 4] quaternion (w, x, y, z)
# translation: [B, 3] in meters
```

## Input/Output

### RGB Pipeline

- Input: RGB `[B, 3, H, W]` (ImageNet normalized) + Pinhole translation `[B, 3]`
- Output: Quaternion `[B, 4]` (w, x, y, z) + Translation `[B, 3]` in meters
- Pinhole translation computed from bbox and used as input to the network

### RGB-D Pipelines

- `rgb`: RGB image `[B, 3, H, W]`
- `points`: Point cloud `[B, N, 3]` in meters
- `choose`: Pixel indices `[B, N]`
- `obj_idx`: Object class `[B]`
- Output: `rotation [B, 3, 3]`, `translation [B, 3]`

## Dataset

LineMOD dataset with 13 objects. Place at `data/Linemod_preprocessed/`

Objects: ape, benchvise, camera, can, cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone

**Important**: 
- All pipelines use the unified dataset loader (`UnifiedLineMODDataset`) which automatically filters gt.yml to only include data for the current object (handles contaminated files)
- RGB-D pipelines use YOLO for object detection (always)
- RGB pipelines use YOLO for object detection (always)

**Training Outputs**: All checkpoints and history are saved automatically:
- RGB: `trained_checkpoints/{object}/`
- RGB-D: `trained_checkpoints_rgbd/{model}/{object}/`

## Requirements

- Python 3.8+
- PyTorch >= 1.9 (with CUDA for GPU)
- torchvision
- numpy, opencv-python, PyYAML
- jupyter, ipython (for notebooks)

## Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Detailed architecture, training workflow, data organization and project structure
