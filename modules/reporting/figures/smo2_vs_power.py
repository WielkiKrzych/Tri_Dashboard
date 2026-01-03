"""
SmO₂ vs Power Chart Generator.

Generates SmO₂ saturation vs power scatter with LT1/LT2 markers.
Input: Canonical JSON report
Output: PNG file
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

from .common import (
    FigureConfig, 
    apply_common_style, 
    add_version_footer, 
    save_figure,
    create_empty_figure,
)


def generate_smo2_power_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate SmO₂ vs Power chart with LT1/LT2 markers.
    
    Shows muscle oxygen saturation against power output,
    with polynomial trend line and lactate threshold markers.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save
        
    Returns:
        PNG/SVG bytes
    """
    config = config or FigureConfig()
    
    # Extract data
    time_series = report_data.get("time_series", {})
    thresholds = report_data.get("thresholds", {})
    
    power_data = time_series.get("power_watts", [])
    smo2_data = time_series.get("smo2_pct", [])
    
    # Handle missing data
    if not power_data or not smo2_data:
        fig = create_empty_figure("Brak danych SmO₂", "SmO₂ vs Moc", config)
        return save_figure(fig, config, output_path)
    
    lt1_watts = thresholds.get("smo2_lt1_watts", 0)
    lt2_watts = thresholds.get("smo2_lt2_watts", 0)
    lt1_smo2 = thresholds.get("smo2_lt1_value", 0)
    lt2_smo2 = thresholds.get("smo2_lt2_value", 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Scatter plot SmO2 vs Power
    ax.scatter(power_data, smo2_data, c=config.get_color("smo2"), 
               alpha=0.3, s=10, label="SmO₂")
    
    # Trend line (polynomial)
    if len(power_data) > 10:
        try:
            z = np.polyfit(power_data, smo2_data, 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(power_data), max(power_data), 100)
            ax.plot(x_smooth, p(x_smooth), color=config.get_color("smo2"), 
                    linewidth=2, label="Trend")
        except Exception:
            pass  # Skip trend if polyfit fails
    
    # LT1/LT2 markers
    if lt1_watts > 0:
        ax.axvline(x=lt1_watts, color=config.get_color("lt1"), 
                   linewidth=2, linestyle='--',
                   label=f"LT1: {lt1_watts}W ({lt1_smo2:.1f}%)")
    if lt2_watts > 0:
        ax.axvline(x=lt2_watts, color=config.get_color("lt2"), 
                   linewidth=2, linestyle='--',
                   label=f"LT2: {lt2_watts}W ({lt2_smo2:.1f}%)")
    
    # Marker points at thresholds
    if lt1_watts > 0 and lt1_smo2 > 0:
        ax.scatter([lt1_watts], [lt1_smo2], c=config.get_color("lt1"), 
                   s=100, zorder=5, marker='o', edgecolors='white')
    if lt2_watts > 0 and lt2_smo2 > 0:
        ax.scatter([lt2_watts], [lt2_smo2], c=config.get_color("lt2"), 
                   s=100, zorder=5, marker='o', edgecolors='white')
    
    ax.set_xlabel("Moc [W]", fontsize=config.font_size)
    ax.set_ylabel("SmO₂ [%]", fontsize=config.font_size)
    ax.set_title("SmO₂ vs Moc – Progi LT", fontsize=config.title_size, fontweight='bold')
    ax.legend(loc='upper right', fontsize=config.font_size - 1)
    
    apply_common_style(fig, ax, config)
    add_version_footer(fig, config)
    
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
