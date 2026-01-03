"""
Common utilities and configuration for report figure generation.

Shared by all figure modules - single source of truth for styling.
No Streamlit dependency - pure matplotlib.
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
from io import BytesIO


# Force non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output quality
DPI = 300  # Print-ready resolution

# Default figure size (inches)
DEFAULT_FIGSIZE: Tuple[float, float] = (10, 6)

# Font configuration
FONT_FAMILY = "Helvetica"
FONT_SIZE = 10
TITLE_SIZE = 14

# Color palette (from PDF spec)
COLORS = {
    "primary": "#1F77B4",      # Blue - power, CP
    "secondary": "#7F8C8D",    # Gray - text, grids
    "power": "#1F77B4",        # Blue
    "hr": "#D62728",           # Red
    "ve": "#FF7F0E",           # Orange (VE)
    "vt1": "#FFA15A",          # Orange
    "vt2": "#EF553B",          # Red-orange
    "smo2": "#2CA02C",         # Green
    "lt1": "#2CA02C",          # Green
    "lt2": "#D62728",          # Red
    "cp": "#1F77B4",           # Blue
    "mmp": "#00CC96",          # Teal
    "confidence_ok": "#2ECC71",
    "confidence_warn": "#F1C40F",
}


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    dpi: int = DPI
    format: str = "png"  # png or svg
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE
    
    # Font settings
    font_family: str = FONT_FAMILY
    font_size: int = FONT_SIZE
    title_size: int = TITLE_SIZE
    
    # Method version (for footer)
    method_version: str = "1.0.0"
    
    def get_color(self, name: str) -> str:
        """Get color by name from palette."""
        return COLORS.get(name, COLORS["primary"])


def apply_common_style(fig: Figure, ax, config: FigureConfig) -> None:
    """Apply common styling to figure and axes.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        config: Figure configuration
    """
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Tick params
    ax.tick_params(axis='both', labelsize=config.font_size)


def add_version_footer(fig: Figure, config: FigureConfig) -> None:
    """Add method version footer to figure.
    
    Args:
        fig: Matplotlib figure
        config: Figure configuration
    """
    fig.text(
        0.99, 0.01, 
        f"v{config.method_version}", 
        ha='right', va='bottom',
        fontsize=8, color=COLORS["secondary"], style='italic'
    )


def save_figure(
    fig: Figure, 
    config: FigureConfig, 
    output_path: Optional[str] = None
) -> bytes:
    """Save figure to file or return as bytes.
    
    Args:
        fig: Matplotlib figure to save
        config: Figure configuration
        output_path: Optional file path to save (returns bytes if None)
        
    Returns:
        Figure bytes (PNG or SVG)
    """
    save_kwargs = {
        "format": config.format,
        "dpi": config.dpi,
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "none",
    }
    
    if output_path:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(output_path, **save_kwargs)
        plt.close(fig)
        
        with open(output_path, 'rb') as f:
            return f.read()
    else:
        buf = BytesIO()
        fig.savefig(buf, **save_kwargs)
        plt.close(fig)
        buf.seek(0)
        return buf.read()


def create_empty_figure(
    message: str, 
    title: str, 
    config: FigureConfig
) -> Figure:
    """Create an empty figure with a message (for missing data cases).
    
    Args:
        message: Message to display
        title: Figure title
        config: Figure configuration
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.text(0.5, 0.5, message, ha='center', va='center', 
            fontsize=14, transform=ax.transAxes, color=COLORS["secondary"])
    ax.set_title(title, fontsize=config.title_size)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig
