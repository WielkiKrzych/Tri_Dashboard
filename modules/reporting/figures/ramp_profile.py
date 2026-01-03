"""
Ramp Profile Chart Generator.

Generates power + HR profile over time with VT1/VT2 markers.
Input: Canonical JSON report
Output: PNG file
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from .common import (
    FigureConfig, 
    apply_common_style, 
    add_version_footer, 
    save_figure,
    create_empty_figure,
)


def generate_ramp_profile_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate ramp profile chart with VT1/VT2 markers.
    
    Shows power and HR traces over time, with vertical markers
    at detected threshold points.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save (None = return bytes)
        
    Returns:
        PNG/SVG bytes
    """
    config = config or FigureConfig()
    
    # Extract data from report
    time_series = report_data.get("time_series", {})
    thresholds = report_data.get("thresholds", {})
    metadata = report_data.get("metadata", {})
    
    time_data = time_series.get("time_sec", [])
    power_data = time_series.get("power_watts", [])
    hr_data = time_series.get("hr_bpm", [])
    
    # Handle missing data
    if not power_data or not time_data:
        fig = create_empty_figure("Brak danych mocy", "Profil Ramp Test", config)
        return save_figure(fig, config, output_path)
    
    vt1_watts = thresholds.get("vt1_watts", 0)
    vt2_watts = thresholds.get("vt2_watts", 0)
    vt1_time = thresholds.get("vt1_time_sec", 0)
    vt2_time = thresholds.get("vt2_time_sec", 0)
    
    # Convert time to minutes
    time_min = [t / 60 for t in time_data]
    vt1_time_min = vt1_time / 60 if vt1_time else 0
    vt2_time_min = vt2_time / 60 if vt2_time else 0
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Power trace
    ax1.plot(time_min, power_data, color=config.get_color("power"), 
             linewidth=1.5, label="Moc [W]")
    ax1.fill_between(time_min, power_data, alpha=0.2, 
                     color=config.get_color("power"))
    
    ax1.set_xlabel("Czas [min]", fontsize=config.font_size)
    ax1.set_ylabel("Moc [W]", fontsize=config.font_size, 
                   color=config.get_color("power"))
    ax1.tick_params(axis='y', labelcolor=config.get_color("power"))
    
    # HR trace (secondary axis)
    ax2 = ax1.twinx()
    if hr_data:
        ax2.plot(time_min, hr_data, color=config.get_color("hr"), 
                 linewidth=1.5, linestyle='--', label="HR [bpm]")
    ax2.set_ylabel("Tętno [bpm]", fontsize=config.font_size, 
                   color=config.get_color("hr"))
    ax2.tick_params(axis='y', labelcolor=config.get_color("hr"))
    
    # VT1/VT2 vertical markers
    if vt1_time_min > 0:
        ax1.axvline(x=vt1_time_min, color=config.get_color("vt1"), 
                    linewidth=2, linestyle='--', 
                    label=f"VT1: {vt1_watts}W")
    if vt2_time_min > 0:
        ax1.axvline(x=vt2_time_min, color=config.get_color("vt2"), 
                    linewidth=2, linestyle='--',
                    label=f"VT2: {vt2_watts}W")
    
    # Shade VT1-VT2 zone
    if vt1_time_min > 0 and vt2_time_min > 0:
        ax1.axvspan(vt1_time_min, vt2_time_min, alpha=0.1, color='orange', 
                    label=f"Strefa {vt1_watts}–{vt2_watts}W")
    
    # Title and legend
    test_date = metadata.get("test_date", "")
    ax1.set_title(f"Profil Ramp Test – {test_date}", 
                  fontsize=config.title_size, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               loc='upper left', fontsize=config.font_size - 1)
    
    apply_common_style(fig, ax1, config)
    add_version_footer(fig, config)
    
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
