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


# ============================================================
# NEW FEATURES: PDC, FRI, Match Burns
# Based on WKO5/INSCYD/TrainerRoad methodologies
# ============================================================

# Default durations for PDC (seconds)
DEFAULT_PDC_DURATIONS = [1, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 2400, 3600]


def calculate_power_duration_curve(
    df_pl: Union[pd.DataFrame, Any], 
    durations: list = None
) -> dict:
    """Calculate Power Duration Curve (Mean Maximal Power for each duration).
    
    The PDC shows the maximum average power that can be sustained for
    different durations - fundamental metric for power profiling.
    
    Args:
        df_pl: DataFrame with 'watts' column
        durations: List of durations in seconds (default: 1s to 1h)
        
    Returns:
        Dict mapping duration (seconds) to MMP (watts)
        
    Example:
        >>> pdc = calculate_power_duration_curve(df)
        >>> pdc[300]  # 5-minute max power
        320.5
    """
    df = ensure_pandas(df_pl)
    
    if 'watts' not in df.columns:
        return {}
    
    if durations is None:
        durations = DEFAULT_PDC_DURATIONS
    
    watts = df['watts'].fillna(0).values
    n = len(watts)
    
    results = {}
    for duration in durations:
        if n < duration:
            results[duration] = None
            continue
            
        # Rolling mean for this duration
        rolling = pd.Series(watts).rolling(window=duration, min_periods=duration).mean()
        mmp = rolling.max()
        
        if pd.notna(mmp):
            results[duration] = float(mmp)
        else:
            results[duration] = None
    
    return results


def calculate_fatigue_resistance_index(
    mmp_5min: float, 
    mmp_20min: float
) -> float:
    """Calculate Fatigue Resistance Index (FRI).
    
    FRI = MMP20 / MMP5
    
    Shows how well an athlete can sustain power over longer durations.
    Key indicator of endurance capability vs anaerobic capacity.
    
    Interpretation:
    - 0.95+: Exceptional endurance (diesel engine)
    - 0.90-0.95: Pro-level stamina
    - 0.85-0.90: Well-trained amateur
    - 0.80-0.85: Average trained cyclist
    - <0.80: Anaerobic-dominant (sprinter profile)
    
    Args:
        mmp_5min: 5-minute max power in watts
        mmp_20min: 20-minute max power in watts
        
    Returns:
        FRI ratio (0.0-1.0+)
    """
    if mmp_5min is None or mmp_20min is None:
        return 0.0
    if mmp_5min <= 0:
        return 0.0
    
    return mmp_20min / mmp_5min


def count_match_burns(
    w_bal: np.ndarray, 
    w_prime_capacity: float,
    threshold_pct: float = 0.3
) -> int:
    """Count how many 'matches' were burned during the workout.
    
    A 'match' is burned when W' balance drops below threshold.
    Each significant W' depletion event counts as burning a match.
    
    Concept from TrainerRoad/WKO5 - helps athletes understand
    how many high-intensity efforts they can make.
    
    Args:
        w_bal: W' balance array (J remaining)
        w_prime_capacity: Full W' capacity (J)
        threshold_pct: Threshold as fraction of W' (default 30%)
        
    Returns:
        Number of matches burned
    """
    if w_bal is None or len(w_bal) == 0 or w_prime_capacity <= 0:
        return 0
    
    threshold = w_prime_capacity * threshold_pct
    burns = 0
    below_threshold = False
    
    for val in w_bal:
        if val < threshold and not below_threshold:
            # Just dropped below threshold
            burns += 1
            below_threshold = True
        elif val >= threshold * 1.5:  # Recovery buffer
            # Recovered enough to count next drop as new match
            below_threshold = False
    
    return burns


def calculate_power_zones_time(
    df_pl: Union[pd.DataFrame, Any],
    cp: float,
    zones: dict = None
) -> dict:
    """Calculate time spent in each power zone.
    
    Default zones based on Coggan's model:
    - Z1 Recovery: <55% CP
    - Z2 Endurance: 55-75% CP
    - Z3 Tempo: 75-90% CP
    - Z4 Threshold: 90-105% CP
    - Z5 VO2max: 105-120% CP
    - Z6 Anaerobic: >120% CP
    
    Args:
        df_pl: DataFrame with 'watts' column
        cp: Critical Power in watts
        zones: Optional custom zone definitions
        
    Returns:
        Dict mapping zone name to seconds spent
    """
    df = ensure_pandas(df_pl)
    
    if 'watts' not in df.columns or cp <= 0:
        return {}
    
    if zones is None:
        zones = {
            'Z1 Recovery': (0, 0.55),
            'Z2 Endurance': (0.55, 0.75),
            'Z3 Tempo': (0.75, 0.90),
            'Z4 Threshold': (0.90, 1.05),
            'Z5 VO2max': (1.05, 1.20),
            'Z6 Anaerobic': (1.20, 10.0)  # 10x as upper bound
        }
    
    watts = df['watts'].fillna(0)
    results = {}
    
    for zone_name, (low_pct, high_pct) in zones.items():
        low_watts = cp * low_pct
        high_watts = cp * high_pct
        
        mask = (watts >= low_watts) & (watts < high_watts)
        seconds_in_zone = mask.sum()
        results[zone_name] = int(seconds_in_zone)
    
    return results


def get_fri_interpretation(fri: float) -> str:
    """Get human-readable interpretation of FRI value.
    
    Args:
        fri: Fatigue Resistance Index value
        
    Returns:
        Polish interpretation string
    """
    if fri >= 0.95:
        return "🟢 Wyjątkowa wytrzymałość (diesel)"
    elif fri >= 0.90:
        return "🟢 Poziom Pro - świetna wytrzymałość"
    elif fri >= 0.85:
        return "🟡 Dobrze wytrenowany amator"
    elif fri >= 0.80:
        return "🟠 Przeciętny poziom"
    else:
        return "🔴 Profil sprinterski (niska wytrzymałość)"

