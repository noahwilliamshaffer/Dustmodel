"""
Visualization module for elemental distribution analysis.

This module provides comprehensive visualization tools for:
- 33x100 elemental distribution heatmaps
- Individual element radial profiles
- Training progress and model evaluation
- Interactive plots for data exploration
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .config import (
    NUM_ELEMENTS, NUM_RADIAL, ELEMENT_NAMES, OUTPUTS_DIR,
    FIGURE_SIZE, DPI, COLORMAP, SAVE_PLOTS, INTERACTIVE_PLOTS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up matplotlib style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class ElementalVisualizer:
    """
    Main visualization class for elemental distribution data.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots (defaults to config.OUTPUTS_DIR)
        """
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # Element colors for consistent plotting
        self.element_colors = self._create_element_colors()
        
        logger.info(f"Initialized ElementalVisualizer with output directory: {self.output_dir}")
    
    def _create_element_colors(self) -> Dict[str, str]:
        """Create a consistent color mapping for elements."""
        # Use a colormap to generate distinct colors
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in np.linspace(0, 1, NUM_ELEMENTS)]
        
        return {element: colors[i] for i, element in enumerate(ELEMENT_NAMES)}
    
    def plot_elemental_heatmap(
        self,
        data: np.ndarray,
        title: str = "Elemental Distribution",
        element_labels: Optional[List[str]] = None,
        radial_labels: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = FIGURE_SIZE,
        cmap: str = COLORMAP,
        show_values: bool = False,
        log_scale: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Create a heatmap of elemental distribution.
        
        Args:
            data: Array of shape (NUM_ELEMENTS, NUM_RADIAL) or (NUM_RADIAL, NUM_ELEMENTS)
            title: Plot title
            element_labels: Labels for elements (defaults to ELEMENT_NAMES)
            radial_labels: Labels for radial positions
            save_path: Path to save the plot
            figsize: Figure size
            cmap: Colormap name
            show_values: Whether to show values in cells
            log_scale: Whether to use log scale for colors
            **kwargs: Additional arguments for imshow
            
        Returns:
            Matplotlib figure object
        """
        # Ensure data is in correct format (elements as rows, radial as columns)
        if data.shape == (NUM_RADIAL, NUM_ELEMENTS):
            data = data.T
        
        # Set default labels
        if element_labels is None:
            element_labels = ELEMENT_NAMES
        if radial_labels is None:
            radial_labels = [f"r{i}" for i in range(NUM_RADIAL)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Apply log scale if requested
        if log_scale:
            # Add small value to avoid log(0)
            data_plot = np.log10(data + 1e-10)
            title += " (Log Scale)"
        else:
            data_plot = data
        
        # Create heatmap
        im = ax.imshow(
            data_plot,
            cmap=cmap,
            aspect='auto',
            interpolation='nearest',
            **kwargs
        )
        
        # Set labels and title
        ax.set_xlabel('Radial Distance (bins)', fontsize=12)
        ax.set_ylabel('Chemical Elements', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set ticks
        ax.set_xticks(range(0, NUM_RADIAL, max(1, NUM_RADIAL // 10)))
        ax.set_xticklabels([radial_labels[i] for i in range(0, NUM_RADIAL, max(1, NUM_RADIAL // 10))])
        ax.set_yticks(range(NUM_ELEMENTS))
        ax.set_yticklabels(element_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if log_scale:
            cbar.set_label('Log₁₀(Abundance)', fontsize=12)
        else:
            cbar.set_label('Abundance', fontsize=12)
        
        # Show values if requested (only for small arrays)
        if show_values and NUM_ELEMENTS <= 20 and NUM_RADIAL <= 50:
            for i in range(NUM_ELEMENTS):
                for j in range(NUM_RADIAL):
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_radial_profile(
        self,
        data: np.ndarray,
        elements: Union[str, List[str]] = None,
        title: str = "Element Radial Profiles",
        radial_labels: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = FIGURE_SIZE,
        log_scale: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot radial profiles for specific elements.
        
        Args:
            data: Array of shape (NUM_ELEMENTS, NUM_RADIAL)
            elements: Element name(s) to plot (defaults to all elements)
            title: Plot title
            radial_labels: Labels for radial positions
            save_path: Path to save the plot
            figsize: Figure size
            log_scale: Whether to use log scale for y-axis
            **kwargs: Additional arguments for plot
            
        Returns:
            Matplotlib figure object
        """
        if radial_labels is None:
            radial_labels = [f"r{i}" for i in range(NUM_RADIAL)]
        
        # Determine which elements to plot
        if elements is None:
            element_indices = range(NUM_ELEMENTS)
            element_names = ELEMENT_NAMES
        elif isinstance(elements, str):
            element_indices = [ELEMENT_NAMES.index(elements)]
            element_names = [elements]
        else:
            element_indices = [ELEMENT_NAMES.index(elem) for elem in elements]
            element_names = elements
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each element
        for i, (idx, name) in enumerate(zip(element_indices, element_names)):
            profile = data[idx, :]
            
            if log_scale:
                profile = np.log10(profile + 1e-10)
            
            ax.plot(
                range(NUM_RADIAL),
                profile,
                label=name,
                color=self.element_colors[name],
                linewidth=2,
                **kwargs
            )
        
        # Customize plot
        ax.set_xlabel('Radial Distance (bins)', fontsize=12)
        if log_scale:
            ax.set_ylabel('Log₁₀(Abundance)', fontsize=12)
            title += " (Log Scale)"
        else:
            ax.set_ylabel('Abundance', fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(0, NUM_RADIAL, max(1, NUM_RADIAL // 10)))
        ax.set_xticklabels([radial_labels[i] for i in range(0, NUM_RADIAL, max(1, NUM_RADIAL // 10))])
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Radial profile saved to {save_path}")
        
        return fig
    
    def plot_element_comparison(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        title: str = "Element Comparison",
        labels: Tuple[str, str] = ("Initial", "Final"),
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = FIGURE_SIZE,
        **kwargs
    ) -> plt.Figure:
        """
        Compare two elemental distributions.
        
        Args:
            data1: First distribution array (NUM_ELEMENTS, NUM_RADIAL)
            data2: Second distribution array (NUM_ELEMENTS, NUM_RADIAL)
            title: Plot title
            labels: Labels for the two distributions
            save_path: Path to save the plot
            figsize: Figure size
            **kwargs: Additional arguments for subplots
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot first distribution
        self._plot_single_heatmap(data1, axes[0, 0], f"{labels[0]} Distribution")
        
        # Plot second distribution
        self._plot_single_heatmap(data2, axes[0, 1], f"{labels[1]} Distribution")
        
        # Plot difference
        diff = data2 - data1
        self._plot_single_heatmap(diff, axes[1, 0], "Difference (Final - Initial)", 
                                 cmap='RdBu_r', center=0)
        
        # Plot relative difference
        rel_diff = np.where(data1 != 0, (data2 - data1) / data1 * 100, 0)
        self._plot_single_heatmap(rel_diff, axes[1, 1], "Relative Difference (%)", 
                                 cmap='RdBu_r', center=0)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Element comparison saved to {save_path}")
        
        return fig
    
    def _plot_single_heatmap(
        self,
        data: np.ndarray,
        ax: plt.Axes,
        title: str,
        cmap: str = COLORMAP,
        center: Optional[float] = None,
        **kwargs
    ):
        """Helper method to plot a single heatmap."""
        im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', **kwargs)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Radial Distance (bins)', fontsize=10)
        ax.set_ylabel('Elements', fontsize=10)
        
        # Set ticks
        ax.set_xticks(range(0, NUM_RADIAL, max(1, NUM_RADIAL // 10)))
        ax.set_yticks(range(0, NUM_ELEMENTS, max(1, NUM_ELEMENTS // 10)))
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = FIGURE_SIZE
    ) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training metrics
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[0, 1].plot(epochs, history['train_mae'], label='Train MAE', color='blue')
        axes[0, 1].plot(epochs, history['val_mae'], label='Validation MAE', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].set_title('Training and Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[1, 0].plot(epochs, history['learning_rate'], color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined metrics plot
        axes[1, 1].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
        axes[1, 1].plot(epochs, history['val_loss'], label='Val Loss', alpha=0.7)
        axes[1, 1].plot(epochs, history['train_mae'], label='Train MAE', alpha=0.7)
        axes[1, 1].plot(epochs, history['val_mae'], label='Val MAE', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('All Training Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        return fig
    
    def create_interactive_heatmap(
        self,
        data: np.ndarray,
        title: str = "Interactive Elemental Distribution",
        element_labels: Optional[List[str]] = None,
        radial_labels: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> go.Figure:
        """
        Create an interactive heatmap using Plotly.
        
        Args:
            data: Array of shape (NUM_ELEMENTS, NUM_RADIAL)
            title: Plot title
            element_labels: Labels for elements
            radial_labels: Labels for radial positions
            save_path: Path to save the interactive plot
            
        Returns:
            Plotly figure object
        """
        if element_labels is None:
            element_labels = ELEMENT_NAMES
        if radial_labels is None:
            radial_labels = [f"r{i}" for i in range(NUM_RADIAL)]
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=radial_labels,
            y=element_labels,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Element: %{y}<br>Radial: %{x}<br>Abundance: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Radial Distance (bins)',
            yaxis_title='Chemical Elements',
            width=800,
            height=600
        )
        
        # Save interactive plot
        if save_path:
            save_path = Path(save_path)
            fig.write_html(str(save_path))
            logger.info(f"Interactive heatmap saved to {save_path}")
        
        return fig
    
    def create_element_dashboard(
        self,
        data: np.ndarray,
        title: str = "Elemental Distribution Dashboard",
        save_path: Optional[Union[str, Path]] = None
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            data: Array of shape (NUM_ELEMENTS, NUM_RADIAL)
            title: Dashboard title
            save_path: Path to save the dashboard
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Elemental Heatmap', 'Top 5 Elements by Total Abundance',
                          'Radial Distribution', 'Element Abundance Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Heatmap
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=[f"r{i}" for i in range(NUM_RADIAL)],
                y=ELEMENT_NAMES,
                colorscale='Viridis',
                name='Distribution'
            ),
            row=1, col=1
        )
        
        # Top 5 elements by total abundance
        total_abundance = np.sum(data, axis=1)
        top_indices = np.argsort(total_abundance)[-5:]
        
        fig.add_trace(
            go.Bar(
                x=[ELEMENT_NAMES[i] for i in top_indices],
                y=[total_abundance[i] for i in top_indices],
                name='Total Abundance'
            ),
            row=1, col=2
        )
        
        # Radial distribution (average across all elements)
        radial_avg = np.mean(data, axis=0)
        
        fig.add_trace(
            go.Scatter(
                x=[f"r{i}" for i in range(NUM_RADIAL)],
                y=radial_avg,
                mode='lines+markers',
                name='Radial Average'
            ),
            row=2, col=1
        )
        
        # Element abundance distribution
        fig.add_trace(
            go.Histogram(
                x=total_abundance,
                nbinsx=20,
                name='Abundance Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        # Save dashboard
        if save_path:
            save_path = Path(save_path)
            fig.write_html(str(save_path))
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig

def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    logger.info("Creating sample visualizations")
    
    visualizer = ElementalVisualizer()
    
    # Generate sample data
    np.random.seed(42)
    sample_data = np.random.rand(NUM_ELEMENTS, NUM_RADIAL)
    
    # Add some structure to make it more realistic
    for i in range(NUM_ELEMENTS):
        # Create radial profiles that decrease with distance
        profile = np.exp(-np.arange(NUM_RADIAL) / 20) * (1 + 0.5 * np.random.randn(NUM_RADIAL))
        sample_data[i, :] = profile * (1 + i * 0.1)  # Different scales for different elements
    
    # Create various visualizations
    visualizer.plot_elemental_heatmap(
        sample_data,
        title="Sample Elemental Distribution",
        save_path=visualizer.output_dir / "sample_heatmap.png"
    )
    
    # Plot radial profiles for a few elements
    visualizer.plot_radial_profile(
        sample_data,
        elements=['H', 'He', 'C', 'N', 'O'],
        title="Sample Radial Profiles",
        save_path=visualizer.output_dir / "sample_radial_profiles.png"
    )
    
    # Create comparison plot
    sample_data2 = sample_data * 1.5 + 0.1 * np.random.randn(NUM_ELEMENTS, NUM_RADIAL)
    visualizer.plot_element_comparison(
        sample_data, sample_data2,
        title="Sample Element Comparison",
        save_path=visualizer.output_dir / "sample_comparison.png"
    )
    
    # Create interactive heatmap
    visualizer.create_interactive_heatmap(
        sample_data,
        title="Sample Interactive Heatmap",
        save_path=visualizer.output_dir / "sample_interactive.html"
    )
    
    logger.info("Sample visualizations created successfully")

if __name__ == "__main__":
    create_sample_visualizations()
    print("Visualization module test completed!")
