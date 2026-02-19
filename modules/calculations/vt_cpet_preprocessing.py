"""
CPET Preprocessing — data copy, unit normalization, and signal smoothing.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def preprocess_cpet_data(
    df: pd.DataFrame,
    cols: dict,
    smoothing_window_sec: int,
    result: dict,
) -> Tuple[pd.DataFrame, bool, bool, bool]:
    """
    Preprocess CPET data: copy, validate, normalize units, smooth signals.

    Args:
        df: Input DataFrame
        cols: Column mapping dict with keys: power, ve, time, vo2, vco2, hr
        smoothing_window_sec: Smoothing window size in seconds
        result: Result dict — modified in place for has_gas_exchange and analysis_notes

    Returns:
        (data, has_vo2, has_vco2, has_hr)
        On validation failure, sets result["error"] and returns early.
    """
    data = df.copy()
    data.columns = data.columns.str.lower().str.strip()

    if cols["power"] not in data.columns:
        result["error"] = f"Missing {cols['power']}"
        return data, False, False, False

    if cols["ve"] not in data.columns:
        result["error"] = f"Missing {cols['ve']}"
        return data, False, False, False

    has_vo2 = cols["vo2"] in data.columns and data[cols["vo2"]].notna().sum() > 10
    has_vco2 = cols["vco2"] in data.columns and data[cols["vco2"]].notna().sum() > 10
    has_hr = cols["hr"] in data.columns and data[cols["hr"]].notna().sum() > 10
    result["has_gas_exchange"] = has_vo2 and has_vco2

    # Unit normalization
    if data[cols["ve"]].mean() < 10:
        data["ve_lmin"] = data[cols["ve"]] * 60
    else:
        data["ve_lmin"] = data[cols["ve"]]

    if has_vo2:
        if data[cols["vo2"]].mean() > 100:
            data["vo2_lmin"] = data[cols["vo2"]] / 1000
        else:
            data["vo2_lmin"] = data[cols["vo2"]]

    if has_vco2:
        if data[cols["vco2"]].mean() > 100:
            data["vco2_lmin"] = data[cols["vco2"]] / 1000
        else:
            data["vco2_lmin"] = data[cols["vco2"]]

    # Smoothing
    window = min(smoothing_window_sec, len(data) // 4)
    if window < 3:
        window = 3

    data["ve_smooth"] = data["ve_lmin"].rolling(window, center=True, min_periods=1).mean()

    if has_vo2:
        data["vo2_smooth"] = data["vo2_lmin"].rolling(window, center=True, min_periods=1).mean()
    if has_vco2:
        data["vco2_smooth"] = data["vco2_lmin"].rolling(window, center=True, min_periods=1).mean()

    # Artifact removal (only when gas exchange available)
    if has_vo2 and has_vco2:
        ve_diff = data["ve_smooth"].diff().abs()
        vo2_diff = data["vo2_smooth"].diff().abs()
        vco2_diff = data["vco2_smooth"].diff().abs()

        ve_threshold = ve_diff.std() * 3
        gas_threshold = max(vo2_diff.std(), vco2_diff.std())

        artifact_mask = (ve_diff > ve_threshold) & (
            (vo2_diff < gas_threshold) & (vco2_diff < gas_threshold)
        )
        artifact_count = artifact_mask.sum()
        if artifact_count > 0:
            result["analysis_notes"].append(f"Removed {artifact_count} respiratory artifacts")
            data.loc[artifact_mask, "ve_smooth"] = np.nan
            data["ve_smooth"] = data["ve_smooth"].interpolate(method="linear")

    return data, has_vo2, has_vco2, has_hr
