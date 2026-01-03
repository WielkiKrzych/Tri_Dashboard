"""
Thermal Analysis Charts Generator.

Generates:
1. Core Temperature vs Heat Strain Index (Time Series)
2. Efficiency Factor (W/HR) vs Core Temperature (Scatter with Trend)
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .common import (
    FigureConfig, 
    apply_common_style, 
    save_figure,
    create_empty_figure,
    COLORS,
)

def _find_column(df: pd.DataFrame, aliases: list) -> Optional[str]:
    """Find first existing column from aliases."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def generate_thermal_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Core Temperature vs HSI chart (Time Series)."""
    config = config or FigureConfig()
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Termoregulacja", config)
        return save_figure(fig, config, output_path)

    # Resolve columns
    df = source_df.copy()
    temp_col = _find_column(df, ['core_temperature_smooth', 'core_temperature', 'core_temp', 'temp', 'temperature'])
    hsi_col = _find_column(df, ['hsi', 'heat_strain_index'])
    time_col = _find_column(df, ['time_min', 'time'])
    
    if not temp_col or not hsi_col:
        fig = create_empty_figure("Brak danych Temp/HSI", "Termoregulacja", config)
        return save_figure(fig, config, output_path)

    # Normalize time to minutes if needed
    if time_col == 'time':
        df['time_min'] = df['time'] / 60.0
        time_vals = df['time_min']
    else:
        time_vals = df[time_col]

    fig, ax1 = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Core Temp (Left Axis)
    l1, = ax1.plot(time_vals, df[temp_col], color='#ff7f0e', label="Core Temp", linewidth=2)
    ax1.set_xlabel("Czas [min]", fontsize=config.font_size)
    ax1.set_ylabel("Temperatura [°C]", fontsize=config.font_size, color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Threshold lines for Temp
    ax1.axhline(y=38.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label="Krytyczna (38.5°C)")
    ax1.axhline(y=37.5, color='green', linestyle=':', alpha=0.5, linewidth=1, label="Optymalna (37.5°C)")

    # HSI (Right Axis)
    ax2 = ax1.twinx()
    l2, = ax2.plot(time_vals, df[hsi_col], color='#d62728', linestyle=':', label="HSI", linewidth=2)
    ax2.set_ylabel("Heat Strain Index (HSI)", fontsize=config.font_size, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, 12)  # HSI scale usually 0-10+

    # Title & Legend
    ax1.set_title("Termoregulacja: Core Temp vs HSI", fontsize=config.title_size, fontweight='bold')
    
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
    
    apply_common_style(fig, ax1, config)
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)


def generate_efficiency_chart(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Efficiency Factor vs Temperature scatter plot."""
    config = config or FigureConfig()
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Spadek Efektywności", config)
        return save_figure(fig, config, output_path)
        
    df = source_df.copy()
    temp_col = _find_column(df, ['core_temperature_smooth', 'core_temperature', 'core_temp'])
    hr_col = _find_column(df, ['heartrate', 'heartrate_smooth', 'hr'])
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power'])
    
    if not (temp_col and hr_col and pwr_col):
        fig = create_empty_figure("Brak danych Power/HR/Temp", "Spadek Efektywności", config)
        return save_figure(fig, config, output_path)
        
    # Filter valid data
    mask = (df[pwr_col] > 10) & (df[hr_col] > 60)
    df_clean = df[mask].copy()
    
    if df_clean.empty:
        fig = create_empty_figure("Zbyt mało danych", "Spadek Efektywności", config)
        return save_figure(fig, config, output_path)

    # Calculate Efficiency (W/HR)
    df_clean['eff_raw'] = df_clean[pwr_col] / df_clean[hr_col]
    # Filter outliers
    df_clean = df_clean[df_clean['eff_raw'] < 6.0]
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Scatter points
    sc = ax.scatter(df_clean[temp_col], df_clean['eff_raw'], 
                    c='#1f77b4', alpha=0.3, s=20, label="Punkty pomiarowe")
                    
    # Trend line (Linear fit for simplicity in PDF, instead of LOWESS which needs statsmodels)
    try:
        z = np.polyfit(df_clean[temp_col], df_clean['eff_raw'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_clean[temp_col].min(), df_clean[temp_col].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f"Trend (Slope: {z[0]:.3f})")
    except:
        pass # Ignore trend error if singular matrix

    ax.set_xlabel("Core Temperature [°C]", fontsize=config.font_size)
    ax.set_ylabel("Efficiency Factor [W/bpm]", fontsize=config.font_size)
    ax.set_title("Spadek Efektywności (Cardiac Drift) vs Temperatura", fontsize=config.title_size, fontweight='bold')
    ax.legend(loc='upper right')
    
    apply_common_style(fig, ax, config)
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
