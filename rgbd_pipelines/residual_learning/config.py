"""Configuration for RGB-D model training."""

# Note: Camera intrinsics and object dimensions are imported from pose_rgb.config
# to maintain single source of truth across RGB and RGB-D models

class ConfigRGBD:
    """RGB-D model configuration."""
    
    # Data
    DATA_ROOT = "data/Linemod_preprocessed"  # Consistent with ConfigRGB
    IMG_SIZE = (640, 480)
    NUM_WORKERS = 10  
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    # WEIGHT_DECAY = 1e-5


    
    # Model
    RGB_BACKBONE = "resnet50"
    PRETRAINED = True
    DEPTH_CHANNELS = 64

    # Evaluation
    ADD_THRESHOLD = 0.1
    
    # Output
    SAVE_DIR = "trained_checkpoints_rgbd/"  # Save it separately from RGB


class ConfigFusion:
    """Fusion model specific configuration."""
    
   # Data
    DATA_ROOT = "data/Linemod_preprocessed"  # Consistent with ConfigRGB
    IMG_SIZE = 128
    NUM_WORKERS = 10 
    BBOX_NOISE = True 
    OBJECT_NAME = "ape"
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    # WEIGHT_DECAY = 1e-5


    
    # Model
    NUM_POINTS = 500

    # Evaluation
    ADD_THRESHOLD = 0.1
    
    # Output
    SAVE_DIR = "trained_checkpoints_fusion/"  # Save it separately from RGB



