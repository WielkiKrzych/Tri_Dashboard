"""
Moduł pomocniczy - wspólne funkcje i stałe dla pakietu calculations.
"""
from typing import Union, Any
import pandas as pd

# Stałe używane w wielu modułach (przeniesione z constants.py dla izolacji)
MIN_SAMPLES_HRV = 100
MIN_SAMPLES_DFA_WINDOW = 30
MIN_SAMPLES_ACTIVE = 60
MIN_SAMPLES_Z2_DRIFT = 300
EFFICIENCY_FACTOR = 0.22
KCAL_PER_JOULE = 0.000239
KCAL_PER_GRAM_CARB = 4.0
CARB_FRACTION_BELOW_VT1 = 0.4
CARB_FRACTION_VT1_VT2 = 0.7
CARB_FRACTION_ABOVE_VT2 = 1.0
DFA_ALPHA_MIN = 0.3
DFA_ALPHA_MAX = 1.5
MIN_WATTS_ACTIVE = 50
MIN_HR_ACTIVE = 90
MIN_WATTS_DECOUPLING = 50
MIN_HR_DECOUPLING = 60
WINDOW_LONG = 30
WINDOW_SHORT = 5


def ensure_pandas(df: Union[pd.DataFrame, Any]) -> pd.DataFrame:
    """
    Convert any DataFrame-like object to pandas DataFrame.
    Minimizes unnecessary copying when already a pandas DataFrame.
    
    Args:
        df: Input data (pandas DataFrame, Polars DataFrame, or dict)
        
    Returns:
        pandas DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, 'to_pandas'):
        return df.to_pandas()
    if isinstance(df, dict):
        return pd.DataFrame(df)
    return pd.DataFrame(df)
