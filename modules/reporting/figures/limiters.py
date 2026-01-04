"""
Limiters Analysis Chart Generator.

Generates:
1. Metabolic Profile Radar Chart (5min Peak Window)
   - Dimensions: Heart, Lungs, Muscles, Power
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

def generate_radar_chart(
    report_data: Dict[str, Any],
    config: Optional[Any] = None,
    output_path: Optional[str] = None,
    source_df: Optional[pd.DataFrame] = None
) -> bytes:
    """Generate Metabolic Profile Radar Chart (5min Peak Window)."""
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
        fig = create_empty_figure("Brak danych źródłowych", "Profil Metaboliczny", **cfg)
        return save_figure(fig, output_path, **cfg)

    # Resolve columns
    df = source_df.copy()
    pwr_col = _find_column(df, ['watts', 'watts_smooth', 'power', 'Power'])
    
    if not pwr_col:
        fig = create_empty_figure("Brak danych Mocy", "Profil Metaboliczny", **cfg)
        return save_figure(fig, output_path, **cfg)

    # Rolling 5min (300s)
    window_sec = 300
    df['rolling_watts'] = df[pwr_col].rolling(window=window_sec, min_periods=window_sec).mean()
    
    if df['rolling_watts'].isna().all():
        fig = create_empty_figure("Za krótki trening (<5min)", "Profil Metaboliczny", **cfg)
        return save_figure(fig, output_path, **cfg)
    
    peak_idx = df['rolling_watts'].idxmax()
    start_idx = max(0, peak_idx - window_sec + 1)
    df_peak = df.iloc[start_idx:peak_idx+1]
    
    # --- Metrics Calculation ---
    # 1. Heart (% HRmax)
    hr_col = _find_column(df, ['heartrate', 'hr'])
    max_hr = report_data.get("metadata", {}).get("max_hr")
    
    if not max_hr:
         max_hr = 190.0
         
    if hr_col and max_hr:
        peak_hr_avg = df_peak[hr_col].mean()
        pct_hr = min(100, (peak_hr_avg / max_hr * 100))
    else:
        pct_hr = 0
        
    # 2. Lungs (% VEmax)
    ve_col = _find_column(df, ['tymeventilation', 've', 'ventilation'])
    vt2_ve = 0
    try:
        thresholds = report_data.get("thresholds", {})
        vt2 = thresholds.get("vt2_result", thresholds.get("vt2", {}))
        vt2_ve = vt2.get("ve", 0)
    except:
        pass
    
    if ve_col:
        ve_max_user = vt2_ve * 1.1 if vt2_ve > 0 else df[ve_col].max()
        peak_ve_avg = df_peak[ve_col].mean()
        pct_ve = min(100, (peak_ve_avg / ve_max_user * 100) if ve_max_user > 0 else 0)
    else:
        pct_ve = 0
        
    # 3. Muscles (% Desaturation)
    smo2_col = _find_column(df, ['smo2', 'SmO2'])
    if smo2_col:
        peak_smo2_avg = df_peak[smo2_col].mean()
        pct_smo2 = min(100, 100 - peak_smo2_avg) # Utilization
    else:
        pct_smo2 = 0
        
    # 4. Power (% CP)
    cp_watts = report_data.get("cp_model", {}).get("cp_watts", 0)
    peak_w_avg = df_peak[pwr_col].mean()
    pct_power = min(120, (peak_w_avg / cp_watts * 100) if cp_watts and cp_watts > 0 else 0)
    
    # --- Plotting ---
    categories = ['Serce\n(% HRmax)', 'Płuca\n(% VEmax)', 'Mięśnie\n(% Desat)', 'Moc\n(% CP)']
    values = [pct_hr, pct_ve, pct_smo2, pct_power]
    
    # Close the loop
    values_closed = values + [values[0]]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=dict(polar=True))
    
    # Draw polygon
    ax.plot(angles, values_closed, color='#00cc96', linewidth=2)
    ax.fill(angles, values_closed, color='#00cc96', alpha=0.25)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=font_size)
    
    # Y-grids
    ax.set_ylim(0, 100)
    plt.yticks([20, 40, 60, 80, 100], color="grey", size=8)
    
    # Title
    ax.set_title(f"Profil Obciążenia (5 min @ {peak_w_avg:.0f} W)", 
                 fontsize=title_size, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return save_figure(fig, output_path, **cfg)
