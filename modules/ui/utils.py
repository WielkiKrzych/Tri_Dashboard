"""
Shared UI utility functions — deduplicated from vent.py, smo2.py, heart_rate.py, etc.
"""

import hashlib
from typing import Optional

import pandas as pd


def parse_time_to_seconds(t_str: str) -> Optional[int]:
    """Parse a time string (hh:mm:ss, mm:ss, or ss) to total seconds."""
    try:
        parts = list(map(int, t_str.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 1:
            return parts[0]
    except (ValueError, AttributeError):
        return None
    return None


def format_time(s: float) -> str:
    """Format seconds to hh:mm:ss or mm:ss string."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def hash_dataframe(df: pd.DataFrame) -> str:
    """Create a hash of DataFrame for cache key generation."""
    if df is None or df.empty:
        return "empty"
    sample = df.head(100).to_json() if hasattr(df, 'to_json') else str(df)
    shape_str = f"{df.shape}_{list(df.columns)}" if hasattr(df, 'shape') else str(df)
    return hashlib.md5(f"{shape_str}_{sample}".encode()).hexdigest()[:16]


def hash_params(**kwargs) -> str:
    """Create a hash of parameters for cache key."""
    param_str = str(sorted(kwargs.items()))
    return hashlib.md5(param_str.encode()).hexdigest()[:16]
