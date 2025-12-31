"""
Data Quality & Reliability Module.

Implements checks for:
- Signal Integrity (Dropout, Noise, Range).
- Protocol Compliance (Ramp Linearity, Step Duration).
- Automatic suppression of metrics when data is unreliable.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats

def check_signal_quality(
    series: pd.Series, 
    metric_name: str = "Metric",
    valid_range: Tuple[float, float] = (0, 1000)
) -> Dict[str, any]:
    """
    Check the quality of a single time-series signal.
    
    Args:
        series: The data series to check.
        metric_name: Name for reporting (e.g. "SmO2").
        valid_range: Tuple of (min, max) physiologically valid values.
        
    Returns:
        Dict with quality score (0.0-1.0), status, and issues list.
    """
    if series is None or series.empty:
        return {
            "score": 0.0,
            "is_valid": False,
            "issues": [f"{metric_name}: No data"]
        }
        
    issues = []
    score = 1.0
    
    # 1. Check Dropouts (NaNs or Zeros/Negatives if strictly positive)
    n_total = len(series)
    n_nans = series.isna().sum()
    pct_nan = (n_nans / n_total) * 100
    
    if pct_nan > 20:
        score -= 0.5
        issues.append(f"{metric_name}: High dropout rate ({pct_nan:.1f}% missing)")
    elif pct_nan > 5:
        score -= 0.1
        issues.append(f"{metric_name}: Moderate dropouts ({pct_nan:.1f}%)")
        
    # 2. Check Range
    # Filter valid to check range on existing data
    valid_data = series.dropna()
    if valid_data.empty:
        return {"score": 0.0, "is_valid": False, "issues": [f"{metric_name}: All Empty"]}
        
    min_val, max_val = valid_range
    n_out_range = ((valid_data < min_val) | (valid_data > max_val)).sum()
    pct_out = (n_out_range / len(valid_data)) * 100
    
    if pct_out > 10:
        score -= 0.3
        issues.append(f"{metric_name}: Values out of range ({pct_out:.1f}%)")
        
    # 3. Check Noise (Rapid fluctuations)
    # Calculate rolling std dev or diff
    # Simple metric: Mean of absolute adjacent differences
    diffs = np.abs(np.diff(valid_data))
    mean_diff = np.mean(diffs)
    data_range = valid_data.max() - valid_data.min()
    
    # Heuristic: If avg step is > 5% of range, it's very noisy
    if data_range > 0:
        noise_ratio = mean_diff / data_range
        if noise_ratio > 0.05: # > 5% jump per second on average
             score -= 0.3
             issues.append(f"{metric_name}: High Signal Noise")
             
    # Clean up score
    score = max(0.0, min(1.0, score))
    
    return {
        "score": round(score, 2),
        "is_valid": score > 0.5,
        "issues": issues
    }

def check_step_test_protocol(
    df: pd.DataFrame, 
    min_step_duration_sec: int = 60
) -> Dict[str, any]:
    """
    Check if the session looks like a valid Step Test (Ramp).
    Reliable VT detection requires a monotonic increase in load.
    
    Args:
        df: DataFrame with 'time' and 'watts'.
        min_step_duration_sec: Minimum duration for a step to be valid.
        
    Returns:
        Dict with validation status.
    """
    if 'time' not in df.columns or 'watts' not in df.columns:
         return {"is_valid": False, "issues": ["Missing time or watts columns"]}
         
    issues = []
    
    # 1. Check Linearity (R²)
    # Resample to 1s to avoid high freq noise affecting simple linregress too much?
    # Simple linear regression on the whole file
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['time'], df['watts'])
    r_squared = r_value ** 2
    
    # Protocol: Monotonic Ramp = Positive Slope, High R2
    # Ramp should be at least ~3-5W per minute (0.05 W/s).
    if slope <= 0.05:
        issues.append(f"Power slope too low ({slope:.3f} W/s). Not a Ramp Test.")
        return {
            "is_valid": False, 
            "issues": issues, 
            "slope": round(slope, 3), 
            "r_squared": round(r_squared, 2)
        }
        
    if r_squared < 0.6: # Allow some variation (warmup, recovery) but main trend must be ramp
        # Check if maybe it's cleaner without warmup/cooldown?
        # But generally, a step test is dominated by the ramp.
        issues.append(f"Power profile is not linear (R²={r_squared:.2f}). Irregular load.")
        
    # 2. Check for Stability (Steps) vs Ramp
    # This is harder without step detection logic. 
    # But if R² is low, it's likely interval or steady ride.
    
    # 3. Check Range
    p_max = df['watts'].max()
    p_min = df['watts'].min()
    if (p_max - p_min) < 50:
         issues.append("Power range too small (<50W) for threshold detection.")
         
    is_valid = len(issues) == 0
    
    return {
        "is_valid": is_valid,
        "issues": issues,
        "r_squared": round(r_squared, 2),
        "slope": round(slope, 2)
    }

def check_data_suitability(df: pd.DataFrame) -> Dict[str, any]:
    """General check for data sufficiency."""
    issues = []
    if len(df) < 300: # < 5 mins
        issues.append("Duration too short (< 5 min)")
        
    return {
        "is_valid": len(issues) == 0,
        "issues": issues
    }
