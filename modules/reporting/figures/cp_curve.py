"""
Power-Duration Curve (CP Curve) Chart Generator.

Generates MMP scatter with CP model overlay.
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


def generate_cp_curve_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate Power-Duration Curve chart with CP model.
    
    Shows Mean Maximal Power points and fitted CP/W' model curve.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save
        
    Returns:
        PNG/SVG bytes
    """
    config = config or FigureConfig()
    
    # Extract data
    pdc_data = report_data.get("power_duration_curve", {})
    cp_model = report_data.get("cp_model", {})
    
    durations = pdc_data.get("durations_sec", [])
    powers = pdc_data.get("powers_watts", [])
    
    cp_watts = cp_model.get("cp_watts", 0)
    w_prime_j = cp_model.get("w_prime_joules", 0)
    
    # Handle missing data
    if not durations or not powers:
        fig = create_empty_figure("Brak danych PDC", "Power-Duration Curve", config)
        return save_figure(fig, config, output_path)
    
    # Convert to minutes for display
    durations_min = [d / 60 for d in durations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # MMP points
    ax.scatter(durations_min, powers, c=config.get_color("mmp"), 
               s=60, zorder=5, label="Twoje MMP", 
               edgecolors='white', linewidths=1)
    
    # CP model curve
    if cp_watts > 0 and w_prime_j > 0:
        t_model = np.linspace(30, max(durations) if durations else 1800, 100)
        p_model = [cp_watts + (w_prime_j / t) for t in t_model]
        t_model_min = [t / 60 for t in t_model]
        ax.plot(t_model_min, p_model, color=config.get_color("cp"), 
                linewidth=2, linestyle='--',
                label=f"Model CP ({cp_watts}W)")
        
        # CP horizontal line
        ax.axhline(y=cp_watts, color=config.get_color("cp"), 
                   linewidth=1.5, linestyle=':', alpha=0.7)
        ax.annotate(f"CP = {cp_watts}W", 
                    xy=(max(durations_min) * 0.8, cp_watts + 5),
                    fontsize=config.font_size, color=config.get_color("cp"))
    
    ax.set_xlabel("Czas [min]", fontsize=config.font_size)
    ax.set_ylabel("Moc [W]", fontsize=config.font_size)
    ax.set_title("Power-Duration Curve (PDC)", 
                 fontsize=config.title_size, fontweight='bold')
    ax.legend(loc='upper right', fontsize=config.font_size - 1)
    
    # W' annotation box
    if w_prime_j > 0:
        ax.text(0.15, 0.85, f"W' = {w_prime_j/1000:.1f} kJ", 
                fontsize=config.font_size,
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    apply_common_style(fig, ax, config)
    add_version_footer(fig, config)
    
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)


# Alias for backward compatibility
generate_pdc_chart = generate_cp_curve_chart
