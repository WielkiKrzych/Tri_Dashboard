"""
Ramp Test Static Figure Generator.

Generates publication-ready charts for PDF reports.
Based solely on JSON report data - no Streamlit dependency.

Charts:
1. Ramp Profile (Power + HR + VT1/VT2 markers)
2. SmO₂ vs Power (with LT1/LT2 markers)
3. Power-Duration Curve (MMP + CP model)

Per specification: methodology/ramp_test/10_pdf_report_spec.md
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from io import BytesIO

# Avoid GUI backend issues
plt.switch_backend('Agg')


@dataclass
class FigureConfig:
    """Configuration for figure generation."""
    dpi: int = 150
    format: str = "png"  # png or svg
    figsize: Tuple[float, float] = (10, 6)
    
    # Colors per spec
    color_power: str = "#1F77B4"
    color_hr: str = "#D62728"
    color_vt1: str = "#FFA15A"
    color_vt2: str = "#EF553B"
    color_smo2: str = "#2CA02C"
    color_lt1: str = "#2CA02C"
    color_lt2: str = "#D62728"
    color_cp: str = "#1F77B4"
    color_mmp: str = "#00CC96"
    
    # Font
    font_family: str = "sans-serif"
    font_size: int = 10
    title_size: int = 14
    
    # Method version (for footer)
    method_version: str = "1.0.0"


def generate_ramp_profile_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate ramp profile chart with VT1/VT2 markers.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save (None = return bytes)
        
    Returns:
        PNG/SVG bytes if output_path is None
    """
    config = config or FigureConfig()
    
    # Extract data from report
    time_series = report_data.get("time_series", {})
    thresholds = report_data.get("thresholds", {})
    metadata = report_data.get("metadata", {})
    
    time_data = time_series.get("time_sec", [])
    power_data = time_series.get("power_watts", [])
    hr_data = time_series.get("hr_bpm", [])
    
    vt1_watts = thresholds.get("vt1_watts", 0)
    vt2_watts = thresholds.get("vt2_watts", 0)
    vt1_time = thresholds.get("vt1_time_sec", 0)
    vt2_time = thresholds.get("vt2_time_sec", 0)
    
    # Convert time to minutes
    time_min = [t / 60 for t in time_data] if time_data else []
    vt1_time_min = vt1_time / 60 if vt1_time else 0
    vt2_time_min = vt2_time / 60 if vt2_time else 0
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Power trace
    if power_data and time_min:
        ax1.plot(time_min, power_data, color=config.color_power, linewidth=1.5, label="Moc [W]")
        ax1.fill_between(time_min, power_data, alpha=0.2, color=config.color_power)
    
    ax1.set_xlabel("Czas [min]", fontsize=config.font_size)
    ax1.set_ylabel("Moc [W]", fontsize=config.font_size, color=config.color_power)
    ax1.tick_params(axis='y', labelcolor=config.color_power)
    
    # HR trace (secondary axis)
    ax2 = ax1.twinx()
    if hr_data and time_min:
        ax2.plot(time_min, hr_data, color=config.color_hr, linewidth=1.5, linestyle='--', label="HR [bpm]")
    ax2.set_ylabel("Tętno [bpm]", fontsize=config.font_size, color=config.color_hr)
    ax2.tick_params(axis='y', labelcolor=config.color_hr)
    
    # VT1/VT2 vertical markers with range shading
    if vt1_time_min > 0:
        ax1.axvline(x=vt1_time_min, color=config.color_vt1, linewidth=2, linestyle='--', 
                   label=f"VT1: {vt1_watts}W")
    if vt2_time_min > 0:
        ax1.axvline(x=vt2_time_min, color=config.color_vt2, linewidth=2, linestyle='--',
                   label=f"VT2: {vt2_watts}W")
    
    # Shade VT1-VT2 zone
    if vt1_time_min > 0 and vt2_time_min > 0:
        ax1.axvspan(vt1_time_min, vt2_time_min, alpha=0.1, color='orange', 
                   label=f"Strefa {vt1_watts}–{vt2_watts}W")
    
    # Title and legend
    test_date = metadata.get("test_date", "")
    ax1.set_title(f"Profil Ramp Test – {test_date}", fontsize=config.title_size, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=config.font_size - 1)
    
    # Footer with method version
    fig.text(0.99, 0.01, f"v{config.method_version}", ha='right', va='bottom', 
             fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    
    return _save_or_return(fig, config, output_path)


def generate_smo2_power_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate SmO₂ vs Power chart with LT1/LT2 markers.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save
        
    Returns:
        PNG/SVG bytes if output_path is None
    """
    config = config or FigureConfig()
    
    # Extract data
    time_series = report_data.get("time_series", {})
    thresholds = report_data.get("thresholds", {})
    metadata = report_data.get("metadata", {})
    
    power_data = time_series.get("power_watts", [])
    smo2_data = time_series.get("smo2_pct", [])
    
    lt1_watts = thresholds.get("smo2_lt1_watts", 0)
    lt2_watts = thresholds.get("smo2_lt2_watts", 0)
    lt1_smo2 = thresholds.get("smo2_lt1_value", 0)
    lt2_smo2 = thresholds.get("smo2_lt2_value", 0)
    
    if not power_data or not smo2_data:
        # Return empty chart with message
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "Brak danych SmO₂", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title("SmO₂ vs Moc", fontsize=config.title_size)
        return _save_or_return(fig, config, output_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Scatter plot SmO2 vs Power
    ax.scatter(power_data, smo2_data, c=config.color_smo2, alpha=0.3, s=10, label="SmO₂")
    
    # Trend line
    if len(power_data) > 10:
        z = np.polyfit(power_data, smo2_data, 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(power_data), max(power_data), 100)
        ax.plot(x_smooth, p(x_smooth), color=config.color_smo2, linewidth=2, label="Trend")
    
    # LT1/LT2 markers
    if lt1_watts > 0:
        ax.axvline(x=lt1_watts, color=config.color_lt1, linewidth=2, linestyle='--',
                  label=f"LT1: {lt1_watts}W ({lt1_smo2:.1f}%)")
    if lt2_watts > 0:
        ax.axvline(x=lt2_watts, color=config.color_lt2, linewidth=2, linestyle='--',
                  label=f"LT2: {lt2_watts}W ({lt2_smo2:.1f}%)")
    
    # Add marker points
    if lt1_watts > 0 and lt1_smo2 > 0:
        ax.scatter([lt1_watts], [lt1_smo2], c=config.color_lt1, s=100, zorder=5, marker='o', edgecolors='white')
    if lt2_watts > 0 and lt2_smo2 > 0:
        ax.scatter([lt2_watts], [lt2_smo2], c=config.color_lt2, s=100, zorder=5, marker='o', edgecolors='white')
    
    ax.set_xlabel("Moc [W]", fontsize=config.font_size)
    ax.set_ylabel("SmO₂ [%]", fontsize=config.font_size)
    ax.set_title("SmO₂ vs Moc – Progi LT", fontsize=config.title_size, fontweight='bold')
    ax.legend(loc='upper right', fontsize=config.font_size - 1)
    ax.grid(True, alpha=0.3)
    
    # Footer
    fig.text(0.99, 0.01, f"v{config.method_version}", ha='right', va='bottom', 
             fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    
    return _save_or_return(fig, config, output_path)


def generate_pdc_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None
) -> bytes:
    """Generate Power-Duration Curve chart with CP model.
    
    Args:
        report_data: Canonical JSON report dictionary
        config: Figure configuration
        output_path: Optional file path to save
        
    Returns:
        PNG/SVG bytes if output_path is None
    """
    config = config or FigureConfig()
    
    # Extract data
    pdc_data = report_data.get("power_duration_curve", {})
    cp_model = report_data.get("cp_model", {})
    metadata = report_data.get("metadata", {})
    
    durations = pdc_data.get("durations_sec", [])
    powers = pdc_data.get("powers_watts", [])
    
    cp_watts = cp_model.get("cp_watts", 0)
    w_prime_j = cp_model.get("w_prime_joules", 0)
    
    if not durations or not powers:
        # Return empty chart with message
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        ax.text(0.5, 0.5, "Brak danych PDC", ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Power-Duration Curve", fontsize=config.title_size)
        return _save_or_return(fig, config, output_path)
    
    # Convert to minutes for display
    durations_min = [d / 60 for d in durations]
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # MMP points
    ax.scatter(durations_min, powers, c=config.color_mmp, s=60, zorder=5, 
              label="Twoje MMP", edgecolors='white', linewidths=1)
    
    # CP model curve
    if cp_watts > 0 and w_prime_j > 0:
        t_model = np.linspace(30, max(durations) if durations else 1800, 100)
        p_model = [cp_watts + (w_prime_j / t) for t in t_model]
        t_model_min = [t / 60 for t in t_model]
        ax.plot(t_model_min, p_model, color=config.color_cp, linewidth=2, linestyle='--',
               label=f"Model CP ({cp_watts}W)")
        
        # CP horizontal line
        ax.axhline(y=cp_watts, color=config.color_cp, linewidth=1.5, linestyle=':', alpha=0.7)
        ax.annotate(f"CP = {cp_watts}W", xy=(max(durations_min) * 0.8, cp_watts + 5),
                   fontsize=config.font_size, color=config.color_cp)
    
    ax.set_xlabel("Czas [min]", fontsize=config.font_size)
    ax.set_ylabel("Moc [W]", fontsize=config.font_size)
    ax.set_title("Power-Duration Curve (PDC)", fontsize=config.title_size, fontweight='bold')
    ax.legend(loc='upper right', fontsize=config.font_size - 1)
    ax.grid(True, alpha=0.3)
    
    # Add W' annotation
    if w_prime_j > 0:
        fig.text(0.15, 0.85, f"W' = {w_prime_j/1000:.1f} kJ", fontsize=config.font_size,
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Footer
    fig.text(0.99, 0.01, f"v{config.method_version}", ha='right', va='bottom', 
             fontsize=8, color='gray', style='italic')
    
    plt.tight_layout()
    
    return _save_or_return(fig, config, output_path)


def generate_all_ramp_figures(
    report_data: Dict[str, Any],
    output_dir: str,
    config: Optional[FigureConfig] = None
) -> Dict[str, str]:
    """Generate all ramp test figures and save to directory.
    
    Args:
        report_data: Canonical JSON report dictionary
        output_dir: Directory to save figures
        config: Figure configuration
        
    Returns:
        Dict mapping figure name to file path
    """
    config = config or FigureConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_id = report_data.get("metadata", {}).get("session_id", "unknown")[:8]
    ext = config.format
    
    paths = {}
    
    # 1. Ramp profile
    ramp_path = output_path / f"ramp_profile_{session_id}.{ext}"
    generate_ramp_profile_chart(report_data, config, str(ramp_path))
    paths["ramp_profile"] = str(ramp_path)
    
    # 2. SmO2 vs Power
    smo2_path = output_path / f"smo2_power_{session_id}.{ext}"
    generate_smo2_power_chart(report_data, config, str(smo2_path))
    paths["smo2_power"] = str(smo2_path)
    
    # 3. PDC
    pdc_path = output_path / f"pdc_{session_id}.{ext}"
    generate_pdc_chart(report_data, config, str(pdc_path))
    paths["pdc"] = str(pdc_path)
    
    return paths


def _save_or_return(fig: plt.Figure, config: FigureConfig, output_path: Optional[str]) -> bytes:
    """Save figure to file or return as bytes."""
    if output_path:
        fig.savefig(output_path, format=config.format, dpi=config.dpi, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        with open(output_path, 'rb') as f:
            return f.read()
    else:
        buf = BytesIO()
        fig.savefig(buf, format=config.format, dpi=config.dpi, 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
