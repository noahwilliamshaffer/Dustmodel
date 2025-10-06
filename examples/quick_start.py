#!/usr/bin/env python3
"""
Quick Start Example for DustModel

This script demonstrates how to quickly get started with the elemental
distribution neural network using sample data.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import *
from data_loader import create_sample_data
from model import ElementDistributionModel
from train import ContinualLearningTrainer
from visualize import ElementalVisualizer

def main():
    """Run a quick demonstration of the system."""
    print("🚀 DustModel Quick Start Example")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    X, y = create_sample_data(n_samples=50)
    print(f"   Created {len(X)} samples")
    
    # Create model
    print("🧠 Creating neural network model...")
    model = ElementDistributionModel()
    print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = ContinualLearningTrainer(model)
    
    # Split data
    from data_loader import ElementalDataLoader
    loader = ElementalDataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Train model (quick training for demo)
    print("🏋️ Training model (5 epochs for demo)...")
    results = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=5,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"   Final validation loss: {results['final_val_loss']:.6f}")
    
    # Test prediction
    print("🔮 Making predictions...")
    predictions = model.predict(X_test[:3])
    print(f"   Predictions shape: {predictions.shape}")
    
    # Create visualization
    print("📈 Creating visualization...")
    visualizer = ElementalVisualizer()
    
    # Use first prediction as example
    sample_data = predictions[0]
    fig = visualizer.plot_elemental_heatmap(
        sample_data,
        title="Sample Prediction - Elemental Distribution",
        save_path=visualizer.output_dir / "quick_start_example.png"
    )
    
    print(f"   Visualization saved to: {visualizer.output_dir / 'quick_start_example.png'}")
    
    print("\n✅ Quick start example completed successfully!")
    print("\nNext steps:")
    print("1. Explore the Jupyter notebook: notebooks/ElementalDistributionModel.ipynb")
    print("2. Try training with your own data in MLtrainFiles/")
    print("3. Set up automation: python src/automation.py --data_dir MLtrainFiles")

if __name__ == "__main__":
    main()
