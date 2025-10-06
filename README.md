# Continual Learning Neural Network for Elemental Radial Distributions

A standalone neural network pipeline designed for modeling and predicting the distribution of chemical elements across radial distances in scientific simulations.

## Overview

This project implements a continual learning neural network that processes elemental abundance profiles across 100 radial distance bins. The system is designed for scientific data (e.g., astrophysical simulations or materials experiments) that produce:

- **33 elemental abundance profiles** (from Hydrogen to Gold)
- **100 radial distance bins** (r0, r1, ..., r99)
- **Environmental conditions** (pressure, temperature, sigma values) as 100×N arrays

The neural network outputs a 33×100 grid of elemental distribution values, visualized as a heatmap of element vs radius.

## Key Features

- 🔄 **Continual Learning**: Automatically detects new training files and retrains the model
- 🧠 **Flexible Architecture**: Modular neural network design supporting various input formats
- 📊 **Rich Visualization**: Interactive heatmaps and analysis plots
- 🔧 **Industry Standards**: Built with PyTorch, NumPy, Pandas, and Matplotlib
- 📁 **Multiple Input Formats**: Supports tab/space-separated files, CSVs, and NumPy arrays
- ⚡ **Real-time Monitoring**: File system monitoring for automatic retraining

## Project Structure

```
DustModel/
├── src/
│   ├── data_loader.py      # Flexible data parsing and loading
│   ├── model.py           # Neural network architecture
│   ├── train.py           # Training pipeline and continual learning
│   ├── visualize.py       # Visualization and analysis tools
│   └── automation.py      # File monitoring and auto-retraining
├── notebooks/
│   └── ElementalDistributionModel.ipynb  # Interactive demonstration
├── MLtrainFiles/          # Training data directory
├── models/               # Saved model checkpoints
├── outputs/              # Generated plots and results
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noahwilliamshaffer/Dustmodel.git
cd Dustmodel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Exploration (Recommended)
```bash
jupyter notebook notebooks/ElementalDistributionModel.ipynb
```

### Command Line Training
```bash
python src/train.py --data_dir "MLtrainFiles" --epochs 50
```

### Enable Automatic Retraining
```bash
python src/automation.py --watch_dir "MLtrainFiles"
```

## Data Format

The system supports multiple input formats:

### Elemental Distribution Files
- **Format**: Tab/space-separated text files
- **Structure**: 33 rows (elements) × 100 columns (radial bins)
- **Elements**: H, He, C, N, O, ..., Au

### Condition Arrays
- **Pressure, Temperature, Sigma**: 100×N arrays (N=2 for initial/final states)
- **Format**: CSV or NumPy binary files

### File Naming Convention
- Input files: `MLIN*.npy`, `MLINCO*.npy`
- Output files: `MLOUT*.npy`, `MLOUTCO*.npy`

## Neural Network Architecture

The model uses a flexible feedforward architecture:

```
Input Layer: (33×100 + 3×N×100) features
    ↓
Hidden Layer 1: 512 neurons (ReLU)
    ↓
Hidden Layer 2: 256 neurons (ReLU)
    ↓
Output Layer: 33×100 elemental distribution
```

**Key Features:**
- Fully-connected layers for maximum flexibility
- Configurable input/output dimensions
- Support for multiple condition states (N≥2)
- Regularization with dropout and L2 weight decay

## Continual Learning

The system implements two retraining strategies:

### Full Retraining
- Accumulates all available data
- Trains model from scratch or last checkpoint
- Ensures comprehensive integration of new data

### Incremental Learning
- Fine-tunes on new data only
- Faster updates with minimal computation
- Suitable for frequent, small data additions

## Visualization

### Primary Output: 33×100 Heatmap
- **Y-axis**: 33 elements (H to Au)
- **X-axis**: 100 radial positions (r0 to r99)
- **Color**: Element abundance/density values

### Additional Plots
- Individual element radial profiles
- Initial vs. final distribution comparisons
- Training loss curves and metrics
- Interactive exploration with Plotly

## Configuration

Key parameters can be adjusted in `src/config.py`:

```python
# Model Architecture
NUM_ELEMENTS = 33
NUM_RADIAL = 100
CONDITION_SETS = 2  # initial and final states

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Continual Learning
INCREMENTAL_MODE = False  # True for incremental, False for full retraining
```

## Automation

### File Monitoring
The system uses the `watchdog` library to monitor the training data directory:

```python
# Automatic retraining on new files
from watchdog.observers import Observer
from src.automation import NewFileHandler

observer = Observer()
handler = NewFileHandler()
observer.schedule(handler, path="MLtrainFiles", recursive=True)
observer.start()
```

### Scheduled Retraining
Alternative cron-based approach for periodic updates:

```bash
# Run every hour
0 * * * * cd /path/to/DustModel && python src/train.py --refresh
```

## API Usage

### Basic Prediction
```python
from src.model import ElementDistributionModel
from src.data_loader import load_sample_data

# Load model
model = ElementDistributionModel()
model.load_checkpoint('models/latest_model.pt')

# Make prediction
input_data = load_sample_data('MLtrainFiles/sample.npy')
prediction = model.predict(input_data)  # Returns 33×100 array
```

### Visualization
```python
from src.visualize import plot_elemental_heatmap, plot_radial_profile

# Create heatmap
plot_elemental_heatmap(prediction, save_path='outputs/heatmap.png')

# Plot specific element
oxygen_profile = prediction[4, :]  # Oxygen is index 4
plot_radial_profile(oxygen_profile, element='O', save_path='outputs/oxygen.png')
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dustmodel2024,
  title={Continual Learning Neural Network for Elemental Radial Distributions},
  author={Williams, Noah},
  year={2024},
  url={https://github.com/noahwilliamshaffer/Dustmodel}
}
```

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

---

**Built for scientific research in astrophysics and materials science**
