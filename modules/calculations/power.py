"""
SRP: Moduł odpowiedzialny za obliczenia związane z mocą.
"""
from typing import Union, Any, Tuple
import numpy as np
import pandas as pd

from .common import ensure_pandas


def calculate_normalized_power(df_pl: Union[pd.DataFrame, Any]) -> float:
    """
    Calculate Normalized Power (NP) using Coggan's formula.
    
    NP = 4th root of (mean of 4th power of 30s rolling average power)
    
    Args:
        df_pl: DataFrame with 'watts' or 'watts_smooth' column
    
    Returns:
        Normalized Power value
    """
    df = ensure_pandas(df_pl)
    col = 'watts' if 'watts' in df.columns else ('watts_smooth' if 'watts_smooth' in df.columns else None)
    
    if col is None:
        return 0.0
        
    # Rolling 30s avg
    rolling_30s = df[col].rolling(window=30, min_periods=1).mean()
    # 4th power
    rolling_pow4 = np.power(rolling_30s, 4)
    # Mean
    avg_pow4 = np.mean(rolling_pow4)
    # 4th root
    np_val = np.power(avg_pow4, 0.25)
    
    if pd.isna(np_val):
        return df[col].mean()
        
    return np_val


def calculate_pulse_power_stats(df_pl: Union[pd.DataFrame, Any]) -> Tuple[float, float, pd.DataFrame]:
    """Calculate Pulse Power (Efficiency) statistics: Avg PP, Trend Drop %.
    
    Pulse Power = Watts / Heart Rate - indicates cardiac efficiency.
    
    Args:
        df_pl: DataFrame with power and HR data
    
    Returns:
        Tuple of (average PP, drop percentage, filtered DataFrame)
    """
    df = ensure_pandas(df_pl)
    
    col_w = 'watts_smooth' if 'watts_smooth' in df.columns else 'watts'
    col_hr = 'heartrate_smooth' if 'heartrate_smooth' in df.columns else 'heartrate'
    
    if col_w not in df.columns or col_hr not in df.columns:
        return 0.0, 0.0, pd.DataFrame()
        
    # Filter sensible values
    mask = (df[col_w] > 50) & (df[col_hr] > 90)
    df_pp = df[mask].copy()
    
    if df_pp.empty:
        return 0.0, 0.0, pd.DataFrame()
        
    df_pp['pulse_power'] = df_pp[col_w] / df_pp[col_hr]
    avg_pp = df_pp['pulse_power'].mean()
    
    # Trend
    if len(df_pp) > 100:
        x = df_pp['time'] if 'time' in df_pp.columns else np.arange(len(df_pp))
        y = df_pp['pulse_power'].values
        idx = np.isfinite(x) & np.isfinite(y)
        if np.sum(idx) > 10:
            z = np.polyfit(x[idx], y[idx], 1)
            slope = z[0]
            intercept = z[1]
            # Total drop in % over the session
            start_val = intercept + slope * x.iloc[0]
            end_val = intercept + slope * x.iloc[-1]
             
            if start_val != 0:
                drop_pct = (end_val - start_val) / start_val * 100
            else:
                drop_pct = 0.0
        else:
            drop_pct = 0.0
    else:
        drop_pct = 0.0
        
    return avg_pp, drop_pct, df_pp
