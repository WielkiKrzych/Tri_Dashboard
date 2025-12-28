"""
SRP: Moduł odpowiedzialny za obliczenia związane z żywieniem.
"""
from typing import Union, Any
import numpy as np
import pandas as pd

from .common import (
    ensure_pandas, 
    EFFICIENCY_FACTOR, 
    KCAL_PER_JOULE, 
    KCAL_PER_GRAM_CARB,
    CARB_FRACTION_BELOW_VT1,
    CARB_FRACTION_VT1_VT2,
    CARB_FRACTION_ABOVE_VT2
)


def estimate_carbs_burned(df_pl: Union[pd.DataFrame, Any], vt1_watts: float, vt2_watts: float) -> float:
    """
    Estimate carbohydrate consumption based on power zones.
    
    Assumption: 22% mechanical efficiency.
    Carb utilization varies by zone:
    - Below VT1: ~40% carbs
    - VT1 to VT2: ~70% carbs
    - Above VT2: ~100% carbs
    
    Args:
        df_pl: DataFrame with 'watts' column
        vt1_watts: Ventilatory Threshold 1 power [W]
        vt2_watts: Ventilatory Threshold 2 power [W]
    
    Returns:
        Total carbohydrates burned [g]
    """
    df = ensure_pandas(df_pl)
    if 'watts' not in df.columns:
        return 0.0
        
    # Energy per second (kcal/s)
    # Power (W = J/s). Efficiency ~22% -> Total Energy = Power / efficiency
    energy_kcal_sec = (df['watts'] / EFFICIENCY_FACTOR) * KCAL_PER_JOULE
    
    # Carb fraction by zone
    conditions = [
        (df['watts'] < vt1_watts),
        (df['watts'] >= vt1_watts) & (df['watts'] < vt2_watts),
        (df['watts'] >= vt2_watts)
    ]
    choices = [CARB_FRACTION_BELOW_VT1, CARB_FRACTION_VT1_VT2, CARB_FRACTION_ABOVE_VT2]
    carb_fraction = np.select(conditions, choices, default=1.0)
    
    # 1 g carbs = 4 kcal
    carbs_burned_sec = (energy_kcal_sec * carb_fraction) / KCAL_PER_GRAM_CARB
    
    return carbs_burned_sec.sum()
