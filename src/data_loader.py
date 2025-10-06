"""
Data loading and parsing module for elemental distribution neural network.

This module provides flexible parsing of multiple input formats including:
- Elemental distribution files (tab/space-separated)
- Condition arrays (pressure, temperature, sigma)
- CSV files with aggregated data
- NumPy binary files

The module is designed to handle various file formats and automatically
detect the structure of input data.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR, NUM_ELEMENTS, NUM_RADIAL, CONDITION_SETS,
    ELEMENT_NAMES, NORMALIZE_INPUTS, NORMALIZE_OUTPUTS,
    VALIDATION_SPLIT, TEST_SPLIT, ELEMENT_HEADER_ROWS,
    CONDITION_HEADER_ROWS, DEFAULT_DELIMITER
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElementalDataLoader:
    """
    Flexible data loader for elemental distribution datasets.
    
    Handles multiple file formats and automatically detects data structure.
    Supports both individual file loading and batch processing of datasets.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory. Defaults to config.DATA_DIR.
        """
        self.data_dir = data_dir or DATA_DIR
        self.input_scaler = StandardScaler() if NORMALIZE_INPUTS else None
        self.output_scaler = StandardScaler() if NORMALIZE_OUTPUTS else None
        self.element_names = ELEMENT_NAMES
        
        logger.info(f"Initialized ElementalDataLoader with data directory: {self.data_dir}")
    
    def detect_file_format(self, filepath: Path) -> str:
        """
        Detect the format of a data file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            String indicating file format: 'numpy', 'csv', 'text', 'unknown'
        """
        extension = filepath.suffix.lower()
        
        if extension == '.npy':
            return 'numpy'
        elif extension == '.csv':
            return 'csv'
        elif extension in ['.txt', '.dat']:
            return 'text'
        else:
            return 'unknown'
    
    def load_numpy_file(self, filepath: Path) -> np.ndarray:
        """
        Load a NumPy binary file.
        
        Args:
            filepath: Path to the .npy file
            
        Returns:
            NumPy array containing the data
        """
        try:
            data = np.load(filepath)
            logger.info(f"Loaded NumPy file {filepath.name}: shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading NumPy file {filepath}: {e}")
            raise
    
    def load_text_file(self, filepath: Path, delimiter: Optional[str] = None) -> np.ndarray:
        """
        Load a text file (tab/space-separated).
        
        Args:
            filepath: Path to the text file
            delimiter: Delimiter to use (None for auto-detection)
            
        Returns:
            NumPy array containing the data
        """
        try:
            if delimiter is None:
                # Try to auto-detect delimiter
                with open(filepath, 'r') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif ',' in first_line:
                        delimiter = ','
                    else:
                        delimiter = None  # Use whitespace
            
            data = np.loadtxt(filepath, delimiter=delimiter, skiprows=ELEMENT_HEADER_ROWS)
            logger.info(f"Loaded text file {filepath.name}: shape {data.shape}, delimiter='{delimiter}'")
            return data
        except Exception as e:
            logger.error(f"Error loading text file {filepath}: {e}")
            raise
    
    def load_csv_file(self, filepath: Path) -> pd.DataFrame:
        """
        Load a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Pandas DataFrame containing the data
        """
        try:
            df = pd.read_csv(filepath, skiprows=ELEMENT_HEADER_ROWS)
            logger.info(f"Loaded CSV file {filepath.name}: shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {e}")
            raise
    
    def parse_elemental_distribution(self, data: np.ndarray, filepath: Path) -> np.ndarray:
        """
        Parse elemental distribution data from array.
        
        Args:
            data: Raw data array
            filepath: Original file path for logging
            
        Returns:
            Array of shape (NUM_ELEMENTS, NUM_RADIAL)
        """
        # Handle different input shapes
        if data.ndim == 1:
            # Flattened data, reshape to (NUM_ELEMENTS, NUM_RADIAL)
            if len(data) == NUM_ELEMENTS * NUM_RADIAL:
                return data.reshape(NUM_ELEMENTS, NUM_RADIAL)
            else:
                raise ValueError(f"1D data length {len(data)} doesn't match expected {NUM_ELEMENTS * NUM_RADIAL}")
        
        elif data.ndim == 2:
            # 2D data, check dimensions
            if data.shape == (NUM_ELEMENTS, NUM_RADIAL):
                return data
            elif data.shape == (NUM_RADIAL, NUM_ELEMENTS):
                return data.T
            elif data.shape[0] == NUM_ELEMENTS or data.shape[1] == NUM_RADIAL:
                # Assume elements are rows, radial bins are columns
                if data.shape[0] == NUM_ELEMENTS:
                    return data[:, :NUM_RADIAL]
                else:
                    return data[:NUM_ELEMENTS, :]
            else:
                logger.warning(f"Unexpected 2D shape {data.shape} for {filepath.name}")
                return data[:NUM_ELEMENTS, :NUM_RADIAL]
        
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}D")
    
    def parse_condition_data(self, data: np.ndarray, filepath: Path) -> np.ndarray:
        """
        Parse condition data (pressure, temperature, sigma).
        
        Args:
            data: Raw data array
            filepath: Original file path for logging
            
        Returns:
            Array of shape (3 * CONDITION_SETS, NUM_RADIAL)
        """
        expected_features = 3 * CONDITION_SETS
        
        if data.ndim == 1:
            if len(data) == expected_features * NUM_RADIAL:
                return data.reshape(expected_features, NUM_RADIAL)
            else:
                raise ValueError(f"1D condition data length {len(data)} doesn't match expected {expected_features * NUM_RADIAL}")
        
        elif data.ndim == 2:
            if data.shape[0] == expected_features:
                return data[:, :NUM_RADIAL]
            elif data.shape[1] == expected_features:
                return data[:NUM_RADIAL, :].T
            else:
                # Try to infer structure
                if data.shape[1] == NUM_RADIAL:
                    return data[:expected_features, :]
                else:
                    logger.warning(f"Unexpected condition data shape {data.shape} for {filepath.name}")
                    return data[:expected_features, :NUM_RADIAL]
        
        else:
            raise ValueError(f"Unsupported condition data dimensions: {data.ndim}D")
    
    def load_single_file(self, filepath: Path) -> Dict[str, np.ndarray]:
        """
        Load and parse a single data file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Dictionary containing parsed data with keys:
            - 'elemental_distribution': (NUM_ELEMENTS, NUM_RADIAL)
            - 'conditions': (3 * CONDITION_SETS, NUM_RADIAL)
            - 'metadata': Dictionary with file information
        """
        file_format = self.detect_file_format(filepath)
        
        if file_format == 'numpy':
            raw_data = self.load_numpy_file(filepath)
        elif file_format in ['csv', 'text']:
            if file_format == 'csv':
                df = self.load_csv_file(filepath)
                raw_data = df.values
            else:
                raw_data = self.load_text_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Determine data type based on filename patterns
        filename = filepath.name.lower()
        
        if 'mlin' in filename and 'mlout' not in filename:
            # Input elemental distribution
            elemental_dist = self.parse_elemental_distribution(raw_data, filepath)
            conditions = np.zeros((3 * CONDITION_SETS, NUM_RADIAL))  # Placeholder
        elif 'mlout' in filename:
            # Output elemental distribution
            elemental_dist = self.parse_elemental_distribution(raw_data, filepath)
            conditions = np.zeros((3 * CONDITION_SETS, NUM_RADIAL))  # Placeholder
        elif any(cond in filename for cond in ['pressure', 'temp', 'sigma']):
            # Condition data
            conditions = self.parse_condition_data(raw_data, filepath)
            elemental_dist = np.zeros((NUM_ELEMENTS, NUM_RADIAL))  # Placeholder
        else:
            # Try to infer from shape
            if raw_data.shape == (NUM_ELEMENTS, NUM_RADIAL) or len(raw_data.flatten()) == NUM_ELEMENTS * NUM_RADIAL:
                elemental_dist = self.parse_elemental_distribution(raw_data, filepath)
                conditions = np.zeros((3 * CONDITION_SETS, NUM_RADIAL))
            elif len(raw_data.flatten()) == 3 * CONDITION_SETS * NUM_RADIAL:
                conditions = self.parse_condition_data(raw_data, filepath)
                elemental_dist = np.zeros((NUM_ELEMENTS, NUM_RADIAL))
            else:
                logger.warning(f"Could not determine data type for {filepath.name}")
                elemental_dist = np.zeros((NUM_ELEMENTS, NUM_RADIAL))
                conditions = np.zeros((3 * CONDITION_SETS, NUM_RADIAL))
        
        return {
            'elemental_distribution': elemental_dist,
            'conditions': conditions,
            'metadata': {
                'filepath': str(filepath),
                'filename': filepath.name,
                'format': file_format,
                'original_shape': raw_data.shape
            }
        }
    
    def load_dataset(self, pattern: str = "*") -> List[Dict[str, np.ndarray]]:
        """
        Load all files matching a pattern from the data directory.
        
        Args:
            pattern: Glob pattern for file matching
            
        Returns:
            List of dictionaries containing parsed data
        """
        data_files = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(data_files)} files matching pattern '{pattern}'")
        
        dataset = []
        for filepath in data_files:
            if filepath.is_file():
                try:
                    data = self.load_single_file(filepath)
                    dataset.append(data)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    continue
        
        logger.info(f"Successfully loaded {len(dataset)} files")
        return dataset
    
    def combine_input_output_pairs(self, dataset: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """
        Combine input and output pairs for supervised learning.
        
        Args:
            dataset: List of data dictionaries
            
        Returns:
            List of combined input-output pairs
        """
        # Group files by experiment/simulation ID
        experiments = {}
        
        for data in dataset:
            filename = data['metadata']['filename']
            
            # Extract experiment ID (remove MLIN/MLOUT prefixes and suffixes)
            if 'mlin' in filename.lower():
                exp_id = filename.lower().replace('mlin', '').split('_')[0].split('.')[0]
                exp_type = 'input'
            elif 'mlout' in filename.lower():
                exp_id = filename.lower().replace('mlout', '').split('_')[0].split('.')[0]
                exp_type = 'output'
            else:
                continue
            
            if exp_id not in experiments:
                experiments[exp_id] = {}
            
            experiments[exp_id][exp_type] = data
        
        # Create input-output pairs
        pairs = []
        for exp_id, exp_data in experiments.items():
            if 'input' in exp_data and 'output' in exp_data:
                pair = {
                    'input_elemental': exp_data['input']['elemental_distribution'],
                    'input_conditions': exp_data['input']['conditions'],
                    'output_elemental': exp_data['output']['elemental_distribution'],
                    'output_conditions': exp_data['output']['conditions'],
                    'metadata': {
                        'experiment_id': exp_id,
                        'input_file': exp_data['input']['metadata']['filename'],
                        'output_file': exp_data['output']['metadata']['filename']
                    }
                }
                pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} input-output pairs from {len(dataset)} files")
        return pairs
    
    def prepare_training_data(self, pairs: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data in the format expected by the neural network.
        
        Args:
            pairs: List of input-output pairs
            
        Returns:
            Tuple of (X, y) where:
            - X: Input features of shape (n_samples, input_features)
            - y: Output targets of shape (n_samples, output_features)
        """
        X_list = []
        y_list = []
        
        for pair in pairs:
            # Flatten input data
            input_elemental = pair['input_elemental'].flatten()
            input_conditions = pair['input_conditions'].flatten()
            input_features = np.concatenate([input_elemental, input_conditions])
            
            # Flatten output data
            output_elemental = pair['output_elemental'].flatten()
            
            X_list.append(input_features)
            y_list.append(output_elemental)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        
        # Normalize data if requested
        if self.input_scaler is not None:
            X = self.input_scaler.fit_transform(X)
            logger.info("Normalized input features")
        
        if self.output_scaler is not None:
            y = self.output_scaler.fit_transform(y)
            logger.info("Normalized output targets")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Input features
            y: Output targets
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=42, shuffle=True
        )
        
        # Second split: separate training and validation sets
        val_size = VALIDATION_SPLIT / (1 - TEST_SPLIT)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, shuffle=True
        )
        
        logger.info(f"Data split: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessed_data(self, X: np.ndarray, y: np.ndarray, filename: str = "preprocessed_data.npz"):
        """
        Save preprocessed data for later use.
        
        Args:
            X: Input features
            y: Output targets
            filename: Name of the file to save
        """
        save_path = self.data_dir / filename
        np.savez_compressed(save_path, X=X, y=y)
        logger.info(f"Saved preprocessed data to {save_path}")
    
    def load_preprocessed_data(self, filename: str = "preprocessed_data.npz") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load previously preprocessed data.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Tuple of (X, y)
        """
        load_path = self.data_dir / filename
        data = np.load(load_path)
        X, y = data['X'], data['y']
        logger.info(f"Loaded preprocessed data from {load_path}: X shape {X.shape}, y shape {y.shape}")
        return X, y

def create_sample_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample data for testing and demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X, y) with synthetic data
    """
    logger.info(f"Creating {n_samples} synthetic samples")
    
    # Generate random elemental distributions
    input_elemental = np.random.rand(n_samples, NUM_ELEMENTS, NUM_RADIAL)
    output_elemental = input_elemental + np.random.normal(0, 0.1, (n_samples, NUM_ELEMENTS, NUM_RADIAL))
    
    # Generate random condition data
    input_conditions = np.random.rand(n_samples, 3 * CONDITION_SETS, NUM_RADIAL)
    output_conditions = input_conditions + np.random.normal(0, 0.05, (n_samples, 3 * CONDITION_SETS, NUM_RADIAL))
    
    # Flatten for neural network input
    X = np.concatenate([
        input_elemental.reshape(n_samples, -1),
        input_conditions.reshape(n_samples, -1)
    ], axis=1)
    
    y = output_elemental.reshape(n_samples, -1)
    
    logger.info(f"Generated synthetic data: X shape {X.shape}, y shape {y.shape}")
    return X, y

if __name__ == "__main__":
    # Test the data loader
    loader = ElementalDataLoader()
    
    # Create sample data for testing
    X, y = create_sample_data(50)
    
    # Test data splitting
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    print("Data loader test completed successfully!")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
