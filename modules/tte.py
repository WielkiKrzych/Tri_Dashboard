"""
Time-to-Exhaustion (TTE) Detection Module.

Computes the maximum continuous duration an athlete can sustain
a target power percentage (e.g., 100% FTP ±5%).
"""
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TTEResult:
    """Result of TTE computation for a single session."""
    session_id: str
    tte_seconds: int
    target_pct: float
    ftp: float
    tolerance_pct: float
    target_power_min: float
    target_power_max: float
    timestamp: Optional[str] = None


def compute_tte(
    power_series: pd.Series,
    target_pct: float,
    ftp: float,
    tol_pct: float = 5.0
) -> int:
    """Compute Time-to-Exhaustion at a given FTP percentage.
    
    Finds the longest continuous segment where power stays within
    target_pct ± tol_pct of FTP.
    
    Args:
        power_series: Power values at 1Hz (or will be resampled)
        target_pct: Target percentage of FTP (e.g., 100 for 100% FTP)
        ftp: Functional Threshold Power in watts
        tol_pct: Tolerance percentage (default 5%)
        
    Returns:
        Maximum continuous duration in seconds
    """
    if power_series is None or len(power_series) == 0:
        return 0
    
    # Calculate target power range
    target_power = ftp * (target_pct / 100.0)
    power_min = target_power * (1 - tol_pct / 100.0)
    power_max = target_power * (1 + tol_pct / 100.0)
    
    # Fill NaN with 0 (treated as below threshold)
    power = power_series.fillna(0).values
    
    # Create boolean mask for valid power range
    in_range = (power >= power_min) & (power <= power_max)
    
    # Find longest continuous run of True values
    max_duration = 0
    current_duration = 0
    
    for val in in_range:
        if val:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def compute_tte_result(
    power_series: pd.Series,
    target_pct: float,
    ftp: float,
    tol_pct: float = 5.0,
    session_id: Optional[str] = None
) -> TTEResult:
    """Compute TTE and return a structured result.
    
    Args:
        power_series: Power values
        target_pct: Target percentage of FTP
        ftp: Functional Threshold Power
        tol_pct: Tolerance percentage
        session_id: Optional session identifier
        
    Returns:
        TTEResult with all computed values
    """
    tte_seconds = compute_tte(power_series, target_pct, ftp, tol_pct)
    
    target_power = ftp * (target_pct / 100.0)
    power_min = target_power * (1 - tol_pct / 100.0)
    power_max = target_power * (1 + tol_pct / 100.0)
    
    return TTEResult(
        session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        tte_seconds=tte_seconds,
        target_pct=target_pct,
        ftp=ftp,
        tolerance_pct=tol_pct,
        target_power_min=power_min,
        target_power_max=power_max,
        timestamp=datetime.now().isoformat()
    )


def rolling_tte(
    history: List[Dict],
    window_days: int = 30
) -> Dict[str, float]:
    """Compute rolling TTE statistics from historical data.
    
    Args:
        history: List of dicts with 'date' and 'tte_seconds' keys
        window_days: Rolling window size in days
        
    Returns:
        Dict with 'median', 'mean', 'max', 'min', 'count'
    """
    if not history:
        return {"median": 0, "mean": 0, "max": 0, "min": 0, "count": 0}
    
    # Filter to window
    cutoff = datetime.now() - timedelta(days=window_days)
    
    filtered = []
    for entry in history:
        entry_date = entry.get("date")
        if entry_date:
            if isinstance(entry_date, str):
                try:
                    entry_date = datetime.fromisoformat(entry_date)
                except ValueError:
                    continue
            if entry_date >= cutoff:
                tte = entry.get("tte_seconds", 0)
                if tte > 0:
                    filtered.append(tte)
    
    if not filtered:
        return {"median": 0, "mean": 0, "max": 0, "min": 0, "count": 0}
    
    return {
        "median": float(np.median(filtered)),
        "mean": float(np.mean(filtered)),
        "max": float(max(filtered)),
        "min": float(min(filtered)),
        "count": len(filtered)
    }


def compute_trend_data(
    history: List[Dict],
    windows: List[int] = [30, 90]
) -> Dict[int, Dict]:
    """Compute trend data for multiple rolling windows.
    
    Args:
        history: List of session dicts with 'date' and 'tte_seconds'
        windows: List of window sizes in days
        
    Returns:
        Dict mapping window_days to rolling stats
    """
    return {w: rolling_tte(history, w) for w in windows}


def export_tte_json(result: TTEResult) -> str:
    """Export TTE result to JSON string.
    
    Args:
        result: TTEResult object
        
    Returns:
        JSON string representation
    """
    data = {
        "session_id": result.session_id,
        "tte_seconds": result.tte_seconds,
        "tte_formatted": format_tte(result.tte_seconds),
        "target_pct": result.target_pct,
        "ftp": result.ftp,
        "tolerance_pct": result.tolerance_pct,
        "target_power_range": {
            "min": round(result.target_power_min, 0),
            "max": round(result.target_power_max, 0)
        },
        "timestamp": result.timestamp
    }
    return json.dumps(data, indent=2)


def format_tte(seconds: int) -> str:
    """Format TTE duration to mm:ss string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "05:30")
    """
    if seconds <= 0:
        return "00:00"
    
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"
