"""
Drift Analysis Scatter Charts Generator.

Generates:
1. Power vs HR Scatter (Decoupling Map)
2. Power vs SmO2 Scatter (Muscle Oxygen Map)
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

def generate_power_hr_scatter(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Power vs Heart Rate scatter plot with time coloring."""
    config = config or FigureConfig()
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Power vs HR", config)
        return save_figure(fig, config, output_path)

    df = source_df.copy()
    hr_col = _find_column(df, ['heartrate', 'heartrate_smooth', 'hr'])
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power', 'Power'])
    time_col = _find_column(df, ['time_min', 'time'])
    
    if not hr_col or not pwr_col:
        fig = create_empty_figure("Brak danych Power/HR", "Power vs HR", config)
        return save_figure(fig, config, output_path)

    # Filter invalid
    mask = (df[pwr_col] > 10) & (df[hr_col] > 30)
    df_clean = df[mask].copy()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Scatter with time coloring
    if time_col:
        c_vals = df_clean[time_col]
        sc = ax.scatter(df_clean[pwr_col], df_clean[hr_col], 
                       c=c_vals, cmap='viridis', alpha=0.5, s=20)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Czas' + (' [min]' if time_col == 'time_min' else ' [s]'))
    else:
        ax.scatter(df_clean[pwr_col], df_clean[hr_col], 
                   c='#1f77b4', alpha=0.5, s=20)
        
    ax.set_xlabel("Moc [W]", fontsize=config.font_size)
    ax.set_ylabel("HR [bpm]", fontsize=config.font_size)
    ax.set_title("Relacja: Moc vs Tętno (Decoupling)", fontsize=config.title_size, fontweight='bold')
    
    apply_common_style(fig, ax, config)
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)


def generate_power_smo2_scatter(
    report_data: Dict[str, Any],
    config: Optional[FigureConfig] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Power vs SmO2 scatter plot with time coloring."""
    config = config or FigureConfig()
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Power vs SmO2", config)
        return save_figure(fig, config, output_path)

    df = source_df.copy()
    smo2_col = _find_column(df, ['smo2', 'SmO2', 'muscle_oxygen'])
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power', 'Power'])
    time_col = _find_column(df, ['time_min', 'time'])
    
    if not smo2_col or not pwr_col:
        fig = create_empty_figure("Brak danych Power/SmO2", "Power vs SmO2", config)
        return save_figure(fig, config, output_path)

    # Filter invalid
    mask = (df[pwr_col] > 10) & (df[smo2_col] > 0)
    df_clean = df[mask].copy()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Scatter with time coloring
    if time_col:
        c_vals = df_clean[time_col]
        sc = ax.scatter(df_clean[pwr_col], df_clean[smo2_col], 
                       c=c_vals, cmap='inferno', alpha=0.5, s=20) # inferno looks good for muscles/heat
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Czas' + (' [min]' if time_col == 'time_min' else ' [s]'))
    else:
        ax.scatter(df_clean[pwr_col], df_clean[smo2_col], 
                   c='#d62728', alpha=0.5, s=20)
        
    ax.set_xlabel("Moc [W]", fontsize=config.font_size)
    ax.set_ylabel("SmO₂ [%]", fontsize=config.font_size)
    ax.set_title("Relacja: Moc vs Saturacja Mięśniowa", fontsize=config.title_size, fontweight='bold')
    
    apply_common_style(fig, ax, config)
    plt.tight_layout()
    
    return save_figure(fig, config, output_path)
