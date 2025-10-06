"""
Neural network model for elemental distribution prediction.

This module implements a flexible neural network architecture designed to predict
elemental distributions across radial distances. The model supports various
architectural configurations and continual learning capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .config import (
    NUM_ELEMENTS, NUM_RADIAL, CONDITION_SETS,
    HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE,
    DROPOUT_RATE, USE_BATCH_NORM, USE_GPU,
    MODELS_DIR, get_input_features, get_output_features
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElementDistributionModel(nn.Module):
    """
    Neural network model for predicting elemental distributions.
    
    The model takes as input:
    - Initial elemental distribution (NUM_ELEMENTS × NUM_RADIAL)
    - Condition arrays (pressure, temperature, sigma for initial and final states)
    
    And outputs:
    - Predicted elemental distribution (NUM_ELEMENTS × NUM_RADIAL)
    """
    
    def __init__(
        self,
        input_features: Optional[int] = None,
        output_features: Optional[int] = None,
        hidden1_size: int = HIDDEN_LAYER_1_SIZE,
        hidden2_size: int = HIDDEN_LAYER_2_SIZE,
        dropout_rate: float = DROPOUT_RATE,
        use_batch_norm: bool = USE_BATCH_NORM,
        activation: str = 'relu'
    ):
        """
        Initialize the ElementDistributionModel.
        
        Args:
            input_features: Number of input features (auto-calculated if None)
            output_features: Number of output features (auto-calculated if None)
            hidden1_size: Size of first hidden layer
            hidden2_size: Size of second hidden layer
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super(ElementDistributionModel, self).__init__()
        
        self.input_features = input_features or get_input_features()
        self.output_features = output_features or get_output_features()
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build the network
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ElementDistributionModel:")
        logger.info(f"  Input features: {self.input_features}")
        logger.info(f"  Output features: {self.output_features}")
        logger.info(f"  Hidden layers: {hidden1_size} -> {hidden2_size}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Batch norm: {use_batch_norm}")
    
    def _build_network(self):
        """Build the neural network architecture."""
        # Input layer
        self.fc1 = nn.Linear(self.input_features, self.hidden1_size)
        self.bn1 = nn.BatchNorm1d(self.hidden1_size) if self.use_batch_norm else None
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Hidden layer 1
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.bn2 = nn.BatchNorm1d(self.hidden2_size) if self.use_batch_norm else None
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        # Hidden layer 2 (optional)
        self.fc3 = nn.Linear(self.hidden2_size, self.hidden2_size // 2)
        self.bn3 = nn.BatchNorm1d(self.hidden2_size // 2) if self.use_batch_norm else None
        self.dropout3 = nn.Dropout(self.dropout_rate)
        
        # Output layer
        self.fc_out = nn.Linear(self.hidden2_size // 2, self.output_features)
        
        # Optional: Add residual connections
        if hasattr(self, 'use_residual') and self.use_residual:
            self.residual_fc = nn.Linear(self.input_features, self.hidden2_size // 2)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, NUM_ELEMENTS, NUM_RADIAL)
        """
        # Store input for residual connection if needed
        input_x = x
        
        # First layer
        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # Third layer
        x = self.fc3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc_out(x)
        
        # Reshape to (batch_size, NUM_ELEMENTS, NUM_RADIAL)
        x = x.view(-1, NUM_ELEMENTS, NUM_RADIAL)
        
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on numpy array input.
        
        Args:
            x: Input array of shape (n_samples, input_features) or (input_features,)
            
        Returns:
            Prediction array of shape (n_samples, NUM_ELEMENTS, NUM_RADIAL) or (NUM_ELEMENTS, NUM_RADIAL)
        """
        self.eval()
        
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x
        
        # Add batch dimension if needed
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Move to device
        if USE_GPU and torch.cuda.is_available():
            x_tensor = x_tensor.cuda()
            self.to('cuda')
        
        # Make prediction
        with torch.no_grad():
            output = self.forward(x_tensor)
            
        # Convert back to numpy
        if USE_GPU and torch.cuda.is_available():
            output = output.cpu()
        
        output_np = output.numpy()
        
        # Remove batch dimension if input was 1D
        if squeeze_output:
            output_np = output_np.squeeze(0)
        
        return output_np
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_features': self.input_features,
            'output_features': self.output_features,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class ConvolutionalElementModel(nn.Module):
    """
    Alternative convolutional model for elemental distribution prediction.
    
    This model treats the elemental distribution as a 2D image and uses
    convolutional layers to capture spatial patterns.
    """
    
    def __init__(
        self,
        input_features: Optional[int] = None,
        dropout_rate: float = DROPOUT_RATE,
        use_batch_norm: bool = USE_BATCH_NORM
    ):
        """
        Initialize the ConvolutionalElementModel.
        
        Args:
            input_features: Number of input features
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(ConvolutionalElementModel, self).__init__()
        
        self.input_features = input_features or get_input_features()
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Calculate how to reshape input for convolutions
        # Assume we can reshape to (channels, height, width)
        self.input_channels = NUM_ELEMENTS + (3 * CONDITION_SETS)
        self.input_height = NUM_RADIAL
        
        # Build convolutional layers
        self.conv1 = nn.Conv1d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64) if use_batch_norm else None
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128) if use_batch_norm else None
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256) if use_batch_norm else None
        
        # Calculate size after convolutions and pooling
        conv_output_size = 256 * (NUM_RADIAL // 4)  # After two pooling operations
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, NUM_ELEMENTS * NUM_RADIAL)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        logger.info(f"Initialized ConvolutionalElementModel with {conv_output_size} conv features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional network.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, NUM_ELEMENTS, NUM_RADIAL)
        """
        batch_size = x.size(0)
        
        # Reshape input for convolutions
        x = x.view(batch_size, self.input_channels, self.input_height)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        if self.bn3 is not None:
            x = self.bn3(x)
        
        # Flatten for fully connected layers
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Reshape to (batch_size, NUM_ELEMENTS, NUM_RADIAL)
        x = x.view(batch_size, NUM_ELEMENTS, NUM_RADIAL)
        
        return x

class ModelManager:
    """Manager class for handling model operations like saving, loading, and checkpointing."""
    
    def __init__(self, model: nn.Module, models_dir: Path = MODELS_DIR):
        """
        Initialize the ModelManager.
        
        Args:
            model: The neural network model
            models_dir: Directory to save/load models
        """
        self.model = model
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        # Move model to GPU if available
        if USE_GPU and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model moved to GPU")
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss: float,
        filename: Optional[str] = None,
        save_optimizer: bool = True
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            optimizer: Optimizer state
            loss: Current loss value
            filename: Custom filename (optional)
            save_optimizer: Whether to save optimizer state
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.models_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }
        
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def save_model(self, filename: str = "model.pt") -> Path:
        """
        Save the complete model.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to saved model
        """
        model_path = self.models_dir / filename
        torch.save(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        return model_path
    
    def load_model(self, model_path: Path) -> nn.Module:
        """
        Load a complete model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        self.model = torch.load(model_path, map_location='cpu')
        if USE_GPU and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        logger.info(f"Loaded model from {model_path}")
        return self.model

def create_model(model_type: str = "feedforward", **kwargs) -> nn.Module:
    """
    Factory function to create different types of models.
    
    Args:
        model_type: Type of model to create ('feedforward' or 'convolutional')
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Initialized model
    """
    if model_type == "feedforward":
        return ElementDistributionModel(**kwargs)
    elif model_type == "convolutional":
        return ConvolutionalElementModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    print("Testing ElementDistributionModel...")
    
    model = ElementDistributionModel()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 4
    input_features = get_input_features()
    x = torch.randn(batch_size, input_features)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    x_np = np.random.randn(2, input_features)
    pred = model.predict(x_np)
    print(f"Prediction shape: {pred.shape}")
    
    # Test model info
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    print("Model tests completed successfully!")
