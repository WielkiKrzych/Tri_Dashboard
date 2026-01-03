"""
Ramp Profile Chart Generator.

Generates power profile over time with VT1/VT2 range bands.
Input: Canonical JSON report
Output: PNG file

Chart shows:
- Power trace over time
- VT1 and VT2 as HORIZONTAL RANGE BANDS (semi-transparent)
- No vertical "magic lines"
- Footer with test_id and method version
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, Optional

from .common import (
    FigureConfig, 
    apply_common_style, 
    save_figure,
    create_empty_figure,
    COLORS,
)


def generate_ramp_profile_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional["pd.DataFrame"] = None
) -> bytes:
    """Generate ramp profile chart with VT1/VT2 as horizontal range bands.
    
    Shows power trace over time with semi-transparent horizontal bands
    indicating VT1 and VT2 power zones.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save (None = return bytes)
        source_df: Optional source DataFrame with raw time/power/hr data
        
    Returns:
        PNG/SVG bytes
    """
    config = config or FigureConfig()
    
    # Extract data from source_df if available, otherwise from report
    time_series = report_data.get("time_series", {})
    thresholds = report_data.get("thresholds", {})
    metadata = report_data.get("metadata", {})
    
    # Try to get data from source_df first
    if source_df is not None and len(source_df) > 0:
        # Normalize column names
        df = source_df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        # Get time data
        if 'time' in df.columns:
            time_data = df['time'].tolist()
        else:
            time_data = list(range(len(df)))
        
        # Get power data
        power_col = None
        for col in ['watts', 'power', 'watts_smooth_5s']:
            if col in df.columns:
                power_col = col
                break
        
        if power_col:
            power_data = df[power_col].fillna(0).tolist()
        else:
            power_data = []
    else:
        # Fallback to time_series from JSON
        time_data = time_series.get("time_sec", [])
        power_data = time_series.get("power_watts", [])
    
    # Handle missing data
    if not power_data or not time_data:
        fig = create_empty_figure("Brak danych mocy", "Profil Ramp Test", config)
        return save_figure(fig, config, output_path)
    
    # Get threshold values from thresholds dict (nested structure)
    vt1_data = thresholds.get("vt1", {})
    vt2_data = thresholds.get("vt2", {})
    vt1_watts = vt1_data.get("midpoint_watts", 0) if isinstance(vt1_data, dict) else 0
    vt2_watts = vt2_data.get("midpoint_watts", 0) if isinstance(vt2_data, dict) else 0
    
    # Define VT ranges (±5% for visual band width)
    vt1_range = (vt1_watts * 0.95, vt1_watts * 1.05) if vt1_watts else None
    vt2_range = (vt2_watts * 0.95, vt2_watts * 1.05) if vt2_watts else None
    
    # Convert time to minutes
    time_min = [t / 60 for t in time_data]
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Power trace
    ax.plot(time_min, power_data, color=config.get_color("power"), 
            linewidth=1.5, label="Moc", zorder=3)
    ax.fill_between(time_min, power_data, alpha=0.15, 
                    color=config.get_color("power"), zorder=2)
    
    # VT1 horizontal band (semi-transparent)
    if vt1_range:
        ax.axhspan(vt1_range[0], vt1_range[1], 
                   alpha=0.25, color=config.get_color("vt1"), 
                   zorder=1, label=f"VT1: {vt1_watts} W")
        # Add center line for clarity
        ax.axhline(y=vt1_watts, color=config.get_color("vt1"), 
                   linewidth=1, linestyle=':', alpha=0.7, zorder=2)
    
    # VT2 horizontal band (semi-transparent)
    if vt2_range:
        ax.axhspan(vt2_range[0], vt2_range[1], 
                   alpha=0.25, color=config.get_color("vt2"), 
                   zorder=1, label=f"VT2: {vt2_watts} W")
        # Add center line for clarity
        ax.axhline(y=vt2_watts, color=config.get_color("vt2"), 
                   linewidth=1, linestyle=':', alpha=0.7, zorder=2)
    
    # Axis labels
    ax.set_xlabel("Czas [min]", fontsize=config.font_size, fontweight='medium')
    ax.set_ylabel("Moc [W]", fontsize=config.font_size, fontweight='medium')
    
    # Title
    test_date = metadata.get("test_date", "")
    ax.set_title(f"Profil Ramp Test – {test_date}", 
                 fontsize=config.title_size, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper left', fontsize=config.font_size - 1, 
              framealpha=0.9, edgecolor='none')
    
    # Apply common styling
    apply_common_style(fig, ax, config)
    
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
