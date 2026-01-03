"""
VE Profile Chart Generator.

Generates ventilation (VE) profile over time with VT1/VT2 thresholds.
Input: Canonical JSON report + Source DataFrame
Output: PNG file

Chart shows:
- Ventilation (VE) on Left Y-Axis
- Power (Watts) on Right Y-Axis (background)
- VT1 and VT2 vertical lines
- Footer with test_id and method version
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import pandas as pd

from .common import (
    FigureConfig, 
    apply_common_style, 
    save_figure,
    create_empty_figure,
    COLORS,
)

def generate_ve_profile_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate VE profile chart with Power overlay.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save (None = return bytes)
        source_df: Optional source DataFrame with raw time/ve/power/hr data
        
    Returns:
        PNG bytes
    """
    config = config or FigureConfig()
    
    # Needs source_df for VE data
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych (VE, Power)", "Dynamika Wentylacji", config)
        return save_figure(fig, config, output_path)
        
    df = source_df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    ve_col = next((c for c in ['tymeventilation', 've', 'ventilation'] if c in df.columns), None)
    power_col = next((c for c in ['watts', 'power', 'watts_smooth_5s'] if c in df.columns), None)
    hr_col = next((c for c in ['hr', 'heart_rate', 'bpm'] if c in df.columns), None) # Optional, strictly requested VE profile but Power helps context
    time_col = next((c for c in ['time', 'seconds'] if c in df.columns), None)
    
    if not ve_col or not time_col:
        fig = create_empty_figure("Brak kolumn VE lub czasu", "Dynamika Wentylacji", config)
        return save_figure(fig, config, output_path)
    
    time_data = df[time_col].tolist()
    ve_data = df[ve_col].fillna(0).tolist()
    power_data = df[power_col].fillna(0).tolist() if power_col else []
    
    # Get threshold values
    thresholds = report_data.get("thresholds", {})
    vt1_data = thresholds.get("vt1", {})
    vt2_data = thresholds.get("vt2", {})
    
    # Use step midpoints or timestamps if available? 
    # Actually, vertical lines at specific WATTS/TIME are tricky if using watts as x-axis?
    # No, request says "Dynamika Wentylacji" (Ventilation Dynamics) -> usually Time X-Axis.
    # UI screenshot shows Time on X-Axis.
    
    # Need to find TIME of VT1/VT2.
    # Thresholds only store WATTS.
    # We can approximate time by finding first time point where power >= vt_watts.
    
    vt1_watts = vt1_data.get("midpoint_watts", 0)
    vt2_watts = vt2_data.get("midpoint_watts", 0)
    
    vt1_time = None
    vt2_time = None
    
    if power_data and vt1_watts:
        # Find time of VT1 (simple search)
        for t, p in zip(time_data, power_data):
            if p >= vt1_watts:
                vt1_time = t
                break
                
    if power_data and vt2_watts:
        for t, p in zip(time_data, power_data):
            if p >= vt2_watts:
                vt2_time = t
                break

    # Create figure
    fig, ax1 = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot Power on Right Axis (Background)
    ax2 = ax1.twinx()
    if power_data:
        ax2.plot(time_data, power_data, color=COLORS["power"], alpha=0.3, linewidth=1, label="Moc")
        ax2.fill_between(time_data, power_data, color=COLORS["power"], alpha=0.05)
        ax2.set_ylabel("Moc [W]", color=COLORS["power"], fontsize=config.font_size)
        ax2.tick_params(axis='y', labelcolor=COLORS["power"])
    
    # Plot VE on Left Axis (Foreground)
    ax1.plot(time_data, ve_data, color=COLORS["ve"], linewidth=2, label="VE (Wentylacja)")
    ax1.set_xlabel("Czas [s]", fontsize=config.font_size)
    ax1.set_ylabel("VE [L/min]", color=COLORS["ve"], fontsize=config.font_size, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=COLORS["ve"])
    
    # Vertical Lines for VT1/VT2
    if vt1_time:
        ax1.axvline(x=vt1_time, color=COLORS["vt1"], linestyle='--', alpha=0.9, linewidth=1.5,
                   label=f"VT1: {int(vt1_watts)} W")
        # Add label at top
        ax1.text(vt1_time, max(ve_data)*0.95, f"VT1\n{int(vt1_watts)} W", 
                 color=COLORS["vt1"], ha='center', va='top', fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                 
    if vt2_time:
        ax1.axvline(x=vt2_time, color=COLORS["vt2"], linestyle='--', alpha=0.9, linewidth=1.5,
                   label=f"VT2: {int(vt2_watts)} W")
        ax1.text(vt2_time, max(ve_data)*0.95, f"VT2\n{int(vt2_watts)} W", 
                 color=COLORS["vt2"], ha='center', va='top', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Title
    metadata = report_data.get("metadata", {})
    test_date = metadata.get("test_date", "")
    ax1.set_title(f"Dynamika Wentylacji (VE) – {test_date}", 
                 fontsize=config.title_size, fontweight='bold', pad=15)
                 
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=config.font_size - 1)
    
    # Common style bits (grid, spines)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Footer
    session_id = metadata.get("session_id", "unknown")[:8]
    fig.text(0.01, 0.01, f"ID: {session_id}", 
             ha='left', va='bottom', fontsize=8, 
             color=COLORS["secondary"], style='italic')

    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
