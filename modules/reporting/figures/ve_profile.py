"""
VE (Ventilation) Profile Chart.

Generates a chart showing VE over time with VT1/VT2 markers.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, Optional
import pandas as pd

from .common import FigureConfig, save_figure, COLORS


def generate_ve_profile_chart(
    report_data: Dict[str, Any],
    config: FigureConfig,
    output_path: str,
    source_df: Optional[pd.DataFrame] = None
) -> str:
    """Generate VE profile chart with VT1/VT2 markers.
    
    Args:
        report_data: Canonical JSON report
        config: Figure configuration
        output_path: Output file path
        source_df: Optional DataFrame with raw VE data
        
    Returns:
        Path to saved figure
    """
    thresholds = report_data.get("thresholds", {})
    test_date = report_data.get("metadata", {}).get("test_date", "")[:10]
    
    # Get VT thresholds
    vt1 = thresholds.get("vt1", {})
    vt2 = thresholds.get("vt2", {})
    vt1_ve = vt1.get("midpoint_ve")
    vt2_ve = vt2.get("midpoint_ve")
    
    # Try to get data from source_df
    time_data = []
    ve_data = []
    power_data = []
    
    if source_df is not None and not source_df.empty:
        source_df.columns = source_df.columns.str.lower().str.strip()
        
        if 'time' in source_df.columns:
            time_data = source_df['time'].values / 60  # Convert to minutes
        
        if 'tymeventilation' in source_df.columns:
            ve_data = source_df['tymeventilation'].values
        
        if 'watts' in source_df.columns:
            power_data = source_df['watts'].rolling(5, center=True).mean().values
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    if len(time_data) > 0 and len(ve_data) > 0:
        # Smooth VE data
        ve_smooth = pd.Series(ve_data).rolling(10, center=True).mean()
        
        # Plot VE
        ax.plot(time_data, ve_smooth, 
                color=config.get_color("ve"), linewidth=2, 
                label="VE (L/min)")
        
        # Add power as light secondary
        if len(power_data) > 0:
            ax2 = ax.twinx()
            ax2.plot(time_data, power_data, 
                    color=config.get_color("power"), linewidth=1, alpha=0.4,
                    label="Moc (W)")
            ax2.set_ylabel("Moc [W]", fontsize=config.font_size, alpha=0.6)
            ax2.tick_params(axis='y', labelcolor=config.get_color("power"), alpha=0.6)
        
        # VT1 marker
        if vt1_ve:
            ax.axhline(y=vt1_ve, color=config.get_color("vt1"), 
                      linestyle='--', linewidth=2, alpha=0.8)
            ax.text(time_data[-1] * 0.02, vt1_ve + 2, 
                   f"VT1: {vt1_ve:.0f} L/min", 
                   color=config.get_color("vt1"), fontsize=9, fontweight='bold')
        
        # VT2 marker
        if vt2_ve:
            ax.axhline(y=vt2_ve, color=config.get_color("vt2"), 
                      linestyle='--', linewidth=2, alpha=0.8)
            ax.text(time_data[-1] * 0.02, vt2_ve + 2, 
                   f"VT2: {vt2_ve:.0f} L/min", 
                   color=config.get_color("vt2"), fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "Brak danych VE", 
               ha='center', va='center', fontsize=12, color='gray',
               transform=ax.transAxes)
    
    # Axis labels
    ax.set_xlabel("Czas [min]", fontsize=config.font_size, fontweight='medium')
    ax.set_ylabel("VE [L/min]", fontsize=config.font_size, fontweight='medium')
    
    # Title
    ax.set_title(f"Profil Wentylacji - {test_date}", 
                fontsize=config.title_size, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=config.font_size - 2, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    save_figure(fig, output_path, config)
    plt.close(fig)
    
    return output_path
