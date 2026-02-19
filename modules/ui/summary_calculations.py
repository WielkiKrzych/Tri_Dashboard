"""
Summary Calculations — pure math helpers, no Streamlit dependency.
"""

import hashlib
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Create a hash of DataFrame for cache key generation."""
    if df is None or df.empty:
        return "empty"
    sample = df.head(100).to_json()
    shape_str = f"{df.shape}_{list(df.columns)}"
    return hashlib.md5(f"{shape_str}_{sample}".encode()).hexdigest()[:16]


def _calculate_np(watts_series: pd.Series) -> float:
    """Obliczenie Normalized Power."""
    if len(watts_series) < 30:
        return watts_series.mean()
    rolling_avg = watts_series.rolling(30, min_periods=1).mean()
    fourth_power = rolling_avg**4
    return fourth_power.mean() ** 0.25


def _estimate_cp_wprime(df_plot: pd.DataFrame) -> Tuple[float, float]:
    """Estymacja CP i W' z danych MMP."""
    if "watts" not in df_plot.columns or len(df_plot) < 1200:
        return 0, 0

    durations = [180, 300, 600, 900, 1200]
    valid_durations = [d for d in durations if d < len(df_plot)]

    if len(valid_durations) < 3:
        return 0, 0

    work_values = []
    for d in valid_durations:
        p = df_plot["watts"].rolling(window=d).mean().max()
        if not pd.isna(p):
            work_values.append(p * d)
        else:
            return 0, 0

    try:
        slope, intercept, _, _, _ = stats.linregress(valid_durations, work_values)
        return slope, intercept
    except Exception:
        return 0, 0


def _get_vent_metrics_for_power(
    df_plot: pd.DataFrame, power_watts: float
) -> Tuple[float, float, float]:
    """
    Pobiera metryki wentylacyjne (HR, VE, BR) dla zadanej mocy z df_plot.
    Znajduje najbliższy punkt w danych do zadanej mocy i zwraca wartości.
    """
    if not power_watts or power_watts <= 0:
        return 0, 0, 0

    if "watts" not in df_plot.columns:
        return 0, 0, 0

    power_col = "watts_smooth_5s" if "watts_smooth_5s" in df_plot.columns else "watts"

    idx = (df_plot[power_col] - power_watts).abs().idxmin()

    start_idx = max(0, idx - 5)
    end_idx = min(len(df_plot), idx + 5)
    window_data = df_plot.iloc[start_idx:end_idx]

    hr_col = None
    for alias in ["hr", "heartrate", "heart_rate", "bpm"]:
        if alias in df_plot.columns:
            hr_col = alias
            break
    hr_val = window_data[hr_col].mean() if hr_col else 0

    ve_val = window_data["tymeventilation"].mean() if "tymeventilation" in df_plot.columns else 0

    br_col = None
    for alias in ["tymebreathrate", "br", "rr", "breath_rate"]:
        if alias in df_plot.columns:
            br_col = alias
            break
    br_val = window_data[br_col].mean() if br_col else 0

    return hr_val, ve_val, br_val
