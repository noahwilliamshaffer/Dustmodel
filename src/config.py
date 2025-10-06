"""
Configuration settings for the Elemental Distribution Neural Network.

This module contains all configurable parameters for the model architecture,
training process, and data handling. Modify these values to adapt the system
to different datasets or requirements.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "MLtrainFiles"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Model Architecture Parameters
NUM_ELEMENTS = 33  # Number of chemical elements (H to Au)
NUM_RADIAL = 100   # Number of radial distance bins (r0 to r99)
CONDITION_SETS = 2 # Number of condition states (initial and final)

# Element names (H to Au in order)
ELEMENT_NAMES = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As'
]

# Neural Network Architecture
HIDDEN_LAYER_1_SIZE = 512
HIDDEN_LAYER_2_SIZE = 256
DROPOUT_RATE = 0.2
USE_BATCH_NORM = True

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience
MIN_DELTA = 1e-6  # Minimum change for early stopping

# Continual Learning Settings
INCREMENTAL_MODE = False  # True for incremental learning, False for full retraining
RETRAIN_ON_NEW_DATA = True
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N epochs

# File Monitoring
WATCH_RECURSIVELY = True
FILE_PATTERNS = ['*.npy', '*.csv', '*.txt']
IGNORE_PATTERNS = ['*.tmp', '*.bak', '.*']

# Visualization Settings
FIGURE_SIZE = (12, 8)
DPI = 300
COLORMAP = 'viridis'
SAVE_PLOTS = True
INTERACTIVE_PLOTS = False

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = PROJECT_ROOT / 'training.log'
VERBOSE = True

# Data Processing
NORMALIZE_INPUTS = True
NORMALIZE_OUTPUTS = True
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# File Format Settings
DEFAULT_DELIMITER = None  # Auto-detect
ELEMENT_HEADER_ROWS = 0  # Number of header rows to skip in element files
CONDITION_HEADER_ROWS = 0  # Number of header rows to skip in condition files

# Performance Settings
NUM_WORKERS = 0  # Number of workers for data loading (0 for main thread)
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# Model Saving
MODEL_SAVE_FORMAT = 'pytorch'  # 'pytorch' or 'onnx'
SAVE_OPTIMIZER_STATE = True
SAVE_TRAINING_HISTORY = True

# Advanced Features
USE_GPU = True
MIXED_PRECISION = False  # Use automatic mixed precision training
GRADIENT_CLIPPING = 1.0  # Clip gradients to prevent exploding gradients

# Model Performance Monitoring
ENABLE_PROFILING = False  # Enable PyTorch profiler for performance analysis
LOG_MEMORY_USAGE = True   # Log GPU/CPU memory usage during training
SAVE_TRAINING_METRICS = True  # Save detailed training metrics to JSON

# Experimental Features
USE_ATTENTION = False  # Experimental: Use attention mechanisms
USE_CONVOLUTION = False  # Experimental: Use 1D convolutions along radial dimension
USE_RESIDUAL_CONNECTIONS = False  # Experimental: Add residual connections

def get_input_features():
    """Calculate total input features based on current configuration."""
    element_features = NUM_ELEMENTS * NUM_RADIAL
    condition_features = 3 * CONDITION_SETS * NUM_RADIAL  # P, T, σ for each state
    return element_features + condition_features

def get_output_features():
    """Calculate total output features."""
    return NUM_ELEMENTS * NUM_RADIAL

def validate_config():
    """Validate configuration parameters."""
    assert NUM_ELEMENTS > 0, "NUM_ELEMENTS must be positive"
    assert NUM_RADIAL > 0, "NUM_RADIAL must be positive"
    assert CONDITION_SETS >= 2, "CONDITION_SETS must be at least 2"
    assert len(ELEMENT_NAMES) == NUM_ELEMENTS, "ELEMENT_NAMES length must match NUM_ELEMENTS"
    assert 0 <= DROPOUT_RATE <= 1, "DROPOUT_RATE must be between 0 and 1"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert EPOCHS > 0, "EPOCHS must be positive"
    assert 0 <= VALIDATION_SPLIT <= 1, "VALIDATION_SPLIT must be between 0 and 1"
    assert 0 <= TEST_SPLIT <= 1, "TEST_SPLIT must be between 0 and 1"
    assert VALIDATION_SPLIT + TEST_SPLIT < 1, "VALIDATION_SPLIT + TEST_SPLIT must be less than 1"
    
    print("✅ Configuration validation passed!")
    return True

if __name__ == "__main__":
    validate_config()
    print(f"Input features: {get_input_features()}")
    print(f"Output features: {get_output_features()}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
