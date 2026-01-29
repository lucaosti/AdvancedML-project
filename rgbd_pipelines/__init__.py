"""RGB-D pose estimation pipelines with depth-based point cloud processing."""

from rgbd_pipelines.base import BasePoseModel

from rgbd_pipelines.residual_learning import ResidualLearningModel
from rgbd_pipelines.densefusion_iterative import DenseFusionIterative
from rgbd_pipelines.pvn3d import PVN3D
from rgbd_pipelines.ffb6d import FFB6D

MODEL_REGISTRY = {
    'residual_learning': ResidualLearningModel,
    'densefusion_iterative': DenseFusionIterative,
    'pvn3d': PVN3D,
    'ffb6d': FFB6D,
}

ALIASES = {
    'densefusion': 'densefusion_iterative',
    'df_iter': 'densefusion_iterative',
    'df': 'densefusion_iterative',
    'residual': 'residual_learning',
    'mlp_residual': 'residual_learning',
}


def create_model(model_name: str, **kwargs):
    """Create pose estimation model by name."""
    name = model_name.lower()
    if name in ALIASES:
        name = ALIASES[name]
    
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(ALIASES.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Available: {sorted(set(available))}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)


def list_models():
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    'BasePoseModel',
    'ResidualLearningModel',
    'DenseFusionIterative',
    'PVN3D',
    'FFB6D',
    'create_model',
    'list_models',
    'MODEL_REGISTRY',
]

