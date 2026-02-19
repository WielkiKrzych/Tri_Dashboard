"""
Ventilatory Threshold — sliding-window detection.

Detects VT1/VT2 transition zones by scanning through the dataset with a
sliding window and checking where the VE slope confidence interval overlaps
physiological thresholds. Includes sensitivity analysis across 4 window sizes.

Results are cached per (DataFrame, parameters) to avoid redundant computation.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .threshold_types import TransitionZone, SensitivityResult
from .vt_utils import calculate_slope

# Module-level result cache (same pattern as hrv.py dfa_cache)
_vt_cache: dict = {}


def _vt_cache_key(*args) -> str:
    """Generate a cache key from function arguments. DataFrames are hashed by content."""
    parts = []
    for a in args:
        if isinstance(a, pd.DataFrame):
            parts.append(str(pd.util.hash_pandas_object(a).sum()))
        else:
            parts.append(repr(a))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def detect_vt_transition_zone(
    df: pd.DataFrame,
    window_duration: int = 60,
    step_size: int = 10,
    ve_column: str = "tymeventilation",
    power_column: str = "watts",
    hr_column: str = "hr",
    time_column: str = "time",
) -> Tuple[Optional[TransitionZone], Optional[TransitionZone]]:
    """Detect VT transition zones using sliding window."""
    cache_key = _vt_cache_key(df, window_duration, step_size, ve_column, power_column, hr_column, time_column)
    if cache_key in _vt_cache:
        return _vt_cache[cache_key]

    if len(df) < window_duration:
        return None, None
    min_time, max_time = df[time_column].min(), df[time_column].max()
    vt1_c, vt2_c = [], []
    Z = 1.96

    for t in range(int(min_time), int(max_time) - window_duration, step_size):
        mask = (df[time_column] >= t) & (df[time_column] < t + window_duration)
        w = df[mask]
        if len(w) < 10:
            continue
        slope, _, err = calculate_slope(w[time_column], w[ve_column])
        l, u = slope - Z * err, slope + Z * err
        if l <= 0.05 <= u and 0.02 <= slope <= 0.08:
            vt1_c.append(
                {
                    "avg_watts": w[power_column].mean(),
                    "avg_hr": w[hr_column].mean() if hr_column in w else None,
                    "std_err": err,
                }
            )
        if l <= 0.15 <= u and 0.10 <= slope <= 0.20:
            vt2_c.append(
                {
                    "avg_watts": w[power_column].mean(),
                    "avg_hr": w[hr_column].mean() if hr_column in w else None,
                    "std_err": err,
                }
            )

    def process_c(c, threshold, err_scale):
        if not c:
            return None
        df_c = pd.DataFrame(c)
        avg_err = df_c["std_err"].mean()
        return TransitionZone(
            range_watts=(df_c["avg_watts"].min(), df_c["avg_watts"].max()),
            range_hr=(df_c["avg_hr"].min(), df_c["avg_hr"].max())
            if "avg_hr" in df_c and df_c["avg_hr"].min()
            else None,
            confidence=max(0.1, min(1.0, 1.0 - (avg_err * err_scale))),
            method=f"Sliding Window (threshold {threshold})",
            description=f"Region where slope CI overlaps {threshold}.",
        )

    result = process_c(vt1_c, 0.05, 100), process_c(vt2_c, 0.15, 50)
    _vt_cache[cache_key] = result
    return result


def run_sensitivity_analysis(
    df: pd.DataFrame, ve_column: str, power_column: str, hr_column: str, time_column: str
) -> SensitivityResult:
    """Check stability by varying window size."""
    cache_key = _vt_cache_key(df, ve_column, power_column, hr_column, time_column)
    if cache_key in _vt_cache:
        return _vt_cache[cache_key]

    wins = [30, 45, 60, 90]
    r1, r2 = [], []
    for w in wins:
        v1, v2 = detect_vt_transition_zone(
            df, w, 5, ve_column, power_column, hr_column, time_column
        )
        if v1:
            r1.append(sum(v1.range_watts) / 2)
        if v2:
            r2.append(sum(v2.range_watts) / 2)

    res = SensitivityResult()

    ve_noise = 0.0
    if ve_column in df.columns:
        ve_data = df[ve_column].dropna()
        if len(ve_data) > 10:
            rolling_std = ve_data.rolling(window=30, min_periods=5).std().mean()
            ve_noise = rolling_std if not pd.isna(rolling_std) else 0.0

    def analyze(r, name):
        if len(r) >= 2:
            std = np.std(r)
            base_score = max(0.0, min(1.0, 1.0 - (std / 50.0)))
            noise_penalty = min(0.3, max(0.0, (ve_noise - 1.0) / 10))
            score = max(0.0, base_score - noise_penalty)
            res.details.append(f"{name} variability: {std:.1f}W")
            return std, score, std > 30.0
        res.details.append(f"{name} not enough data points.")
        return 0.0, 0.5, False

    res.vt1_variability_watts, res.vt1_stability_score, res.is_vt1_unreliable = analyze(r1, "VT1")
    res.vt2_variability_watts, res.vt2_stability_score, res.is_vt2_unreliable = analyze(r2, "VT2")
    _vt_cache[cache_key] = res
    return res
