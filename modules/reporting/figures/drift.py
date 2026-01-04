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
    apply_common_style, 
    save_figure,
    create_empty_figure,
    COLORS,
    get_color
)

def _find_column(df: pd.DataFrame, aliases: list) -> Optional[str]:
    """Find first existing column from aliases."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def generate_power_hr_scatter(
    report_data: Dict[str, Any],
    config: Optional[Any] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Power vs Heart Rate scatter plot with time coloring."""
    # Handle config as dict if passed, or use defaults
    if hasattr(config, '__dict__'):
        cfg = config.__dict__
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = {}

    figsize = cfg.get('figsize', (10, 6))
    dpi = cfg.get('dpi', 150)
    font_size = cfg.get('font_size', 10)
    title_size = cfg.get('title_size', 14)
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Power vs HR", **cfg)
        return save_figure(fig, output_path, **cfg)

    df = source_df.copy()
    hr_col = _find_column(df, ['heartrate', 'heartrate_smooth', 'hr'])
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power', 'Power'])
    time_col = _find_column(df, ['time_min', 'time'])
    
    if not hr_col or not pwr_col:
        fig = create_empty_figure("Brak danych Power/HR", "Power vs HR", **cfg)
        return save_figure(fig, output_path, **cfg)

    # Filter invalid
    mask = (df[pwr_col] > 10) & (df[hr_col] > 30)
    df_clean = df[mask].copy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Scatter with time coloring
    if time_col:
        c_vals = df_clean[time_col]
        sc = ax.scatter(df_clean[pwr_col], df_clean[hr_col], 
                       c=c_vals, cmap='viridis', alpha=0.5, s=20)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Czas' + (' [min]' if time_col == 'time_min' else ' [s]'))
    else:
        ax.scatter(df_clean[pwr_col], df_clean[hr_col], 
                   c=get_color("primary"), alpha=0.5, s=20)
        
    ax.set_xlabel("Moc [W]", fontsize=font_size)
    ax.set_ylabel("HR [bpm]", fontsize=font_size)
    ax.set_title("Relacja: Moc vs Tętno (Decoupling)", fontsize=title_size, fontweight='bold')
    
    apply_common_style(fig, ax, **cfg)
    plt.tight_layout()
    
    return save_figure(fig, output_path, **cfg)


def generate_power_smo2_scatter(
    report_data: Dict[str, Any],
    config: Optional[Any] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Power vs SmO2 scatter plot with time coloring."""
    # Handle config as dict if passed, or use defaults
    if hasattr(config, '__dict__'):
        cfg = config.__dict__
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = {}

    figsize = cfg.get('figsize', (10, 6))
    dpi = cfg.get('dpi', 150)
    font_size = cfg.get('font_size', 10)
    title_size = cfg.get('title_size', 14)
    
    if source_df is None or source_df.empty:
        fig = create_empty_figure("Brak danych źródłowych", "Power vs SmO2", **cfg)
        return save_figure(fig, output_path, **cfg)

    df = source_df.copy()
    smo2_col = _find_column(df, ['smo2', 'SmO2', 'muscle_oxygen'])
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power', 'Power'])
    time_col = _find_column(df, ['time_min', 'time'])
    
    if not smo2_col or not pwr_col:
        fig = create_empty_figure("Brak danych Power/SmO2", "Power vs SmO2", **cfg)
        return save_figure(fig, output_path, **cfg)

    # Filter invalid
    mask = (df[pwr_col] > 10) & (df[smo2_col] > 0)
    df_clean = df[mask].copy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Scatter with time coloring
    if time_col:
        c_vals = df_clean[time_col]
        sc = ax.scatter(df_clean[pwr_col], df_clean[smo2_col], 
                       c=c_vals, cmap='inferno', alpha=0.5, s=20) 
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Czas' + (' [min]' if time_col == 'time_min' else ' [s]'))
    else:
        ax.scatter(df_clean[pwr_col], df_clean[hr_col], 
                   c=get_color("smo2"), alpha=0.5, s=20)
        
    ax.set_xlabel("Moc [W]", fontsize=font_size)
    ax.set_ylabel("SmO₂ [%]", fontsize=font_size)
    ax.set_title("Relacja: Moc vs Saturacja Mięśniowa", fontsize=title_size, fontweight='bold')
    
    apply_common_style(fig, ax, **cfg)
    plt.tight_layout()
    
    return save_figure(fig, output_path, **cfg)
