"""
SmO₂ vs Power Chart Generator.

Generates SmO₂ saturation vs power scatter with LT1/LT2 range bands.
Input: Canonical JSON report
Output: PNG file

Chart shows:
- Raw scatter of SmO₂ vs Power (NO smoothing/interpolation)
- LT1/LT2 as HORIZONTAL RANGE BANDS if available
- Annotation about SmO₂ being a local signal
- Footer with test_id and method version
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from .common import (
    FigureConfig, 
    apply_common_style, 
    save_figure,
    create_empty_figure,
    COLORS,
)


def generate_smo2_power_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate SmO₂ vs Power chart with LT1/LT2 range bands.
    
    Shows raw muscle oxygen saturation against power output,
    WITHOUT any smoothing or interpolation.
    
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
    metadata = report_data.get("metadata", {})
    
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
    
    # Define LT ranges (±5% for visual band width)
    lt1_range = (lt1_watts * 0.95, lt1_watts * 1.05) if lt1_watts else None
    lt2_range = (lt2_watts * 0.95, lt2_watts * 1.05) if lt2_watts else None
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Raw scatter plot SmO2 vs Power (NO SMOOTHING)
    ax.scatter(power_data, smo2_data, c=config.get_color("smo2"), 
               alpha=0.4, s=12, label="SmO₂", zorder=3, edgecolors='none')
    
    # LT1 vertical range band (semi-transparent)
    if lt1_range:
        ax.axvspan(lt1_range[0], lt1_range[1], 
                   alpha=0.2, color=config.get_color("lt1"), 
                   zorder=1, label=f"LT1: {lt1_watts} W")
        # Marker point at LT1
        if lt1_smo2 > 0:
            ax.scatter([lt1_watts], [lt1_smo2], c=config.get_color("lt1"), 
                       s=120, zorder=5, marker='o', edgecolors='white', linewidths=2)
    
    # LT2 vertical range band (semi-transparent)
    if lt2_range:
        ax.axvspan(lt2_range[0], lt2_range[1], 
                   alpha=0.2, color=config.get_color("lt2"), 
                   zorder=1, label=f"LT2: {lt2_watts} W")
        # Marker point at LT2
        if lt2_smo2 > 0:
            ax.scatter([lt2_watts], [lt2_smo2], c=config.get_color("lt2"), 
                       s=120, zorder=5, marker='o', edgecolors='white', linewidths=2)
    
    # Axis labels
    ax.set_xlabel("Moc [W]", fontsize=config.font_size, fontweight='medium')
    ax.set_ylabel("SmO₂ [%]", fontsize=config.font_size, fontweight='medium')
    
    # Title
    ax.set_title("SmO₂ vs Moc", fontsize=config.title_size, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=config.font_size - 1, 
              framealpha=0.9, edgecolor='none')
    
    # Apply common styling
    apply_common_style(fig, ax, config)
    
    # Important annotation about SmO₂ interpretation
    ax.text(0.5, -0.12, 
            "ℹ️ SmO₂ jest sygnałem lokalnym – interpretować kontekstowo",
            ha='center', va='top', fontsize=9, style='italic',
            color=COLORS["secondary"], transform=ax.transAxes)
    
    # Footer with test_id and method version
    session_id = metadata.get("session_id", "unknown")[:8]
    fig.text(0.01, 0.01, f"ID: {session_id}", 
             ha='left', va='bottom', fontsize=8, 
             color=COLORS["secondary"], style='italic')
    fig.text(0.99, 0.01, f"v{config.method_version}", 
             ha='right', va='bottom', fontsize=8, 
             color=COLORS["secondary"], style='italic')
    
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
