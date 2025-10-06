"""
Training pipeline for the elemental distribution neural network.

This module implements the training loop, continual learning capabilities,
and model evaluation. It supports both full retraining and incremental
learning modes.
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm

from .model import ElementDistributionModel, ModelManager, count_parameters
from .data_loader import ElementalDataLoader, create_sample_data
from .config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    CHECKPOINT_FREQUENCY, USE_GPU, MIXED_PRECISION, GRADIENT_CLIPPING,
    MODELS_DIR, OUTPUTS_DIR, NUM_ELEMENTS, NUM_RADIAL
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingHistory:
    """Class to track training history and metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_mae = []
        self.val_mae = []
        self.learning_rates = []
        self.epochs = []
        
    def add_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  train_mae: float, val_mae: float, lr: float):
        """Add metrics for a training epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_mae.append(train_mae)
        self.val_mae.append(val_mae)
        self.learning_rates.append(lr)
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best validation loss."""
        return self.epochs[np.argmin(self.val_losses)]
    
    def plot_history(self, save_path: Optional[Path] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.epochs, self.val_losses, label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(self.epochs, self.train_mae, label='Train MAE')
        axes[0, 1].plot(self.epochs, self.val_mae, label='Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.epochs, self.learning_rates)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)
        
        # Combined plot
        axes[1, 1].plot(self.epochs, self.train_losses, label='Train Loss', alpha=0.7)
        axes[1, 1].plot(self.epochs, self.val_losses, label='Val Loss', alpha=0.7)
        axes[1, 1].plot(self.epochs, self.train_mae, label='Train MAE', alpha=0.7)
        axes[1, 1].plot(self.epochs, self.val_mae, label='Val MAE', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('All Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

class ContinualLearningTrainer:
    """
    Trainer class that supports continual learning and automatic retraining.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        data_loader: Optional[ElementalDataLoader] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model (created if None)
            data_loader: Data loader instance (created if None)
            device: Device to use for training ('cpu', 'cuda', or None for auto)
        """
        self.model = model or ElementDistributionModel()
        self.data_loader = data_loader or ElementalDataLoader()
        self.device = device or ('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.history = TrainingHistory()
        self.early_stopping = EarlyStopping()
        
        # Model manager for saving/loading
        self.model_manager = ModelManager(self.model, MODELS_DIR)
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
    
    def setup_optimizer(self, learning_rate: float = LEARNING_RATE, weight_decay: float = WEIGHT_DECAY):
        """Set up optimizer and learning rate scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        logger.info(f"Setup optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = BATCH_SIZE
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders for training and validation."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Calculate loss
            loss = self.criterion(outputs.view(batch_x.size(0), -1), batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if GRADIENT_CLIPPING > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIPPING)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs.view(batch_x.size(0), -1) - batch_y))
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validation", leave=False):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.view(batch_x.size(0), -1), batch_y)
                mae = torch.mean(torch.abs(outputs.view(batch_x.size(0), -1) - batch_y))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        resume_from_checkpoint: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training input features
            y_train: Training targets
            X_val: Validation input features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Setup optimizer
        self.setup_optimizer(learning_rate)
        
        # Load checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint and resume_from_checkpoint.exists():
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = self.model_manager.load_checkpoint(resume_from_checkpoint, self.optimizer)
            start_epoch = checkpoint['epoch'] + 1
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Train and validate
            train_loss, train_mae = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            # Record history
            self.history.add_epoch(epoch, train_loss, val_loss, train_mae, val_mae, current_lr)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch:3d}/{epochs}: "
                f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
                f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, "
                f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s"
            )
            
            # Save checkpoint
            if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
                self.model_manager.save_checkpoint(
                    epoch, self.optimizer, val_loss,
                    f"checkpoint_epoch_{epoch:04d}.pt"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model_manager.save_checkpoint(
                    epoch, self.optimizer, val_loss,
                    "best_model.pt"
                )
                logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        total_time = time.time() - training_start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Plot training history
        history_plot_path = OUTPUTS_DIR / "training_history.png"
        self.history.plot_history(history_plot_path)
        
        # Return training results
        results = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_train_mae': train_mae,
            'final_val_mae': val_mae,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'history': self.history
        }
        
        return results
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = BATCH_SIZE
    ) -> Dict[str, float]:
        """Evaluate the model on test data."""
        logger.info("Evaluating model on test data")
        
        # Create test data loader
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.view(batch_x.size(0), -1), batch_y)
                mae = torch.mean(torch.abs(outputs.view(batch_x.size(0), -1) - batch_y))
                
                total_loss += loss.item()
                total_mae += mae.item()
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # Calculate overall metrics
        num_batches = len(test_loader)
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        # Calculate additional metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mse = mean_squared_error(all_targets.flatten(), all_predictions.flatten())
        mae_sklearn = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
        
        results = {
            'test_loss': avg_loss,
            'test_mae': avg_mae,
            'test_mse': mse,
            'test_mae_sklearn': mae_sklearn,
            'rmse': np.sqrt(mse)
        }
        
        logger.info(f"Test Results: Loss={avg_loss:.6f}, MAE={avg_mae:.6f}, MSE={mse:.6f}")
        
        return results
    
    def continual_learn(
        self,
        new_data_path: Path,
        epochs: int = 10,
        learning_rate: Optional[float] = None,
        mode: str = "incremental"
    ) -> Dict[str, Any]:
        """
        Perform continual learning with new data.
        
        Args:
            new_data_path: Path to new training data
            epochs: Number of epochs for retraining
            learning_rate: Learning rate for retraining (uses current if None)
            mode: Learning mode ('incremental' or 'full')
            
        Returns:
            Dictionary containing retraining results
        """
        logger.info(f"Starting continual learning in {mode} mode")
        
        # Load new data
        new_data = self.data_loader.load_dataset(str(new_data_path.name))
        new_pairs = self.data_loader.combine_input_output_pairs(new_data)
        
        if not new_pairs:
            logger.warning("No valid input-output pairs found in new data")
            return {}
        
        # Prepare new training data
        X_new, y_new = self.data_loader.prepare_training_data(new_pairs)
        
        if mode == "full":
            # Load all existing data and combine with new data
            logger.info("Full retraining mode: loading all existing data")
            all_data = self.data_loader.load_dataset("*")
            all_pairs = self.data_loader.combine_input_output_pairs(all_data)
            
            if all_pairs:
                X_all, y_all = self.data_loader.prepare_training_data(all_pairs)
                X_combined = np.vstack([X_all, X_new])
                y_combined = np.vstack([y_all, y_new])
            else:
                X_combined, y_combined = X_new, y_new
            
            # Split combined data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_combined, y_combined, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
            
        else:  # incremental mode
            # Use only new data for fine-tuning
            logger.info("Incremental learning mode: using only new data")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_new, y_new, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        
        # Adjust learning rate for incremental learning
        if mode == "incremental" and learning_rate is None:
            learning_rate = LEARNING_RATE * 0.1  # Use smaller learning rate
        
        # Train with new data
        results = self.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            learning_rate=learning_rate or LEARNING_RATE
        )
        
        # Evaluate on test set
        test_results = self.evaluate(X_test, y_test)
        results.update(test_results)
        
        logger.info("Continual learning completed successfully")
        return results

def main():
    """Main training function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train elemental distribution model")
    parser.add_argument("--data_dir", type=str, default="MLtrainFiles",
                       help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--model_type", type=str, default="feedforward",
                       choices=["feedforward", "convolutional"],
                       help="Type of model to train")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample data for testing")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        logger.info("Creating sample data for testing")
        X, y = create_sample_data(200)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        # Load real data
        data_loader = ElementalDataLoader(Path(args.data_dir))
        dataset = data_loader.load_dataset("*")
        
        if not dataset:
            logger.error("No data found. Use --create_sample to generate test data.")
            return
        
        pairs = data_loader.combine_input_output_pairs(dataset)
        if not pairs:
            logger.error("No valid input-output pairs found.")
            return
        
        X, y = data_loader.prepare_training_data(pairs)
        X_train, X_temp, y_train, y_temp = data_loader.split_data(X, y)[:4]
        X_val, X_test, y_val, y_test = data_loader.split_data(X, y)[4:]
    
    # Create and train model
    model = ElementDistributionModel()
    trainer = ContinualLearningTrainer(model)
    
    # Resume from checkpoint if specified
    resume_path = Path(args.resume) if args.resume else None
    
    # Train the model
    results = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=resume_path
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate(X_test, y_test)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final validation loss: {results['final_val_loss']:.6f}")
    logger.info(f"Test loss: {test_results['test_loss']:.6f}")
    logger.info(f"Test MAE: {test_results['test_mae']:.6f}")

if __name__ == "__main__":
    main()
