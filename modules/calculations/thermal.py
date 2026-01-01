"""
SRP: Moduł odpowiedzialny za obliczenia termiczne (Heat Strain Index).
"""
from typing import Union, Any
import pandas as pd
import numpy as np

from .common import ensure_pandas


def calculate_heat_strain_index(df_pl: Union[pd.DataFrame, Any]) -> pd.DataFrame:
    """Calculate Heat Strain Index (HSI) based on core temp and HR.
    
    HSI is a composite index (0-10) indicating heat stress level:
    - 0-3: Low strain
    - 4-6: Moderate strain  
    - 7-10: High strain (risk of heat illness)
    
    Args:
        df_pl: DataFrame with 'core_temperature_smooth' and 'heartrate_smooth'
    
    Returns:
        DataFrame with added 'hsi' column
    """
    df = ensure_pandas(df_pl)
    core_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df.columns else None
    
    if not core_col or 'heartrate_smooth' not in df.columns:
        df['hsi'] = None
        return df
    
    # HSI formula: weighted combination of temperature and HR deviation from baseline
    # Temperature contribution: (CoreTemp - 37.0) / 2.5 * 5 (max 5 points)
    # HR contribution: (HR - 60) / 120 * 5 (max 5 points)
    df['hsi'] = (
        (5 * (df[core_col] - 37.0) / 2.5) + 
        (5 * (df['heartrate_smooth'] - 60.0) / 120.0)
    ).clip(0.0, 10.0)
    
    return df
def calculate_thermal_decay(df_pl: Union[pd.DataFrame, Any]) -> dict:
    """Calculate the thermal cost of performance as % efficiency loss per 1°C.
    
    Standard WKO5/INSCYD approach:
    - Efficiency Factor (EF) = Power / Heart Rate
    - Decay = Percentage change in EF for every +1°C of Core Temperature.
    
    Returns:
        dict: {
            'decay_pct_per_c': float, # e.g. -5.2 means 5.2% drop per 1°C
            'r_squared': float,       # statistical confidence
            'is_significant': bool,
            'message': str
        }
    """
    df = ensure_pandas(df_pl)
    
    # Column mapping (finding best candidates)
    pwr_col = next((c for c in ['watts_smooth', 'watts', 'power'] if c in df.columns), None)
    hr_col = next((c for c in ['heartrate_smooth', 'heartrate', 'hr'] if c in df.columns), None)
    temp_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df.columns else None
    
    if not all([pwr_col, hr_col, temp_col]):
        return {'decay_pct_per_c': 0, 'r_squared': 0, 'is_significant': False, 'message': "Brak wymaganych kolumn (Moc, HR, Temp)"}

    # Filter for active state: Power > 100W (or 50% FTP), HR > 100, Temp > 37.2
    # We focus on the "Heat Load" phase where temperature is significantly above baseline
    mask = (df[pwr_col] > 50) & (df[hr_col] > 80) & (df[temp_col] > 37.0)
    df_act = df[mask].copy()
    
    if len(df_act) < 300: # Min 5 minutes of data
        return {'decay_pct_per_c': 0, 'r_squared': 0, 'is_significant': False, 'message': "Zbyt mało danych aktywnych (>37°C)"}
        
    # Efficiency Factor
    df_act['ef'] = df_act[pwr_col] / df_act[hr_col]
    
    # Linear Regression (EF ~ Temp)
    # y = m*x + b
    x = df_act[temp_col].values
    y = df_act['ef'].values
    
    # Basic OLS
    n = len(x)
    m = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    b = (np.mean(y)) - m * np.mean(x)
    
    # R-squared
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate % decay per 1 degree
    # Decay = (Slope / Mean EF at 37.5°C) * 100
    ref_ef = m * 37.5 + b
    if ref_ef <= 0: ref_ef = np.mean(y)
    
    decay_pct = (m / ref_ef) * 100
    
    return {
        'decay_pct_per_c': round(decay_pct, 2),
        'r_squared': round(r2, 3),
        'is_significant': r2 > 0.3 and decay_pct < 0,
        'message': f"Spadek o {abs(decay_pct):.1f}% na każdy +1°C" if decay_pct < 0 else "Stabilna termoregulacja"
    }
