"""
SRP: Moduł odpowiedzialny za obliczenia termiczne (Heat Strain Index).
"""
from typing import Union, Any
import pandas as pd

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
