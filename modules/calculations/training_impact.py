"""Aerobic vs Anaerobic training impact decomposition.

Decomposes session TSS into aerobic (below CP) and anaerobic (above CP)
contributions, classifies session intensity, and tracks rolling balance.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrainingImpact:
    """Decomposed training impact for a session."""

    aerobic_tss: float
    anaerobic_tss: float
    total_tss: float
    aerobic_fraction: float
    intensity_type: str


_INTENSITY_MAP = [
    (0.90, "recovery"),
    (0.75, "endurance"),
    (0.60, "tempo"),
    (0.45, "threshold"),
    (0.30, "vo2max"),
    (0.0, "anaerobic"),
]


def classify_session_intensity(aerobic_fraction: float) -> str:
    """Classify session from aerobic fraction."""
    for threshold, label in _INTENSITY_MAP:
        if aerobic_fraction >= threshold:
            return label
    return "anaerobic"


def calculate_session_impact(
    df: pd.DataFrame,
    cp: float,
    w_prime: float,
    power_col: str = "watts",
    duration_col: str = "time",
) -> Optional[TrainingImpact]:
    """Decompose session TSS into aerobic and anaerobic components.

    TSS ≈ (duration_hours × NP² / FTP²) × 100. We approximate
    by splitting seconds-above-CP (anaerobic) vs below (aerobic),
    weighted by relative power.

    Args:
        df: DataFrame with power data.
        cp: Critical Power in watts.
        w_prime: W' in joules.
        power_col: Column name for power.
        duration_col: Column name for time.

    Returns:
        TrainingImpact or None if insufficient data.
    """
    if df is None or df.empty or cp <= 0:
        return None
    if power_col not in df.columns:
        return None

    power = df[power_col].dropna()
    if power.empty or len(power) < 10:
        return None

    ftp = cp * 0.95

    seconds_total = len(power)
    above_cp = (power > cp).sum()
    below_cp = seconds_total - above_cp

    mean_below = float(power[power <= cp].mean()) if below_cp > 0 else 0.0
    mean_above = float(power[power > cp].mean()) if above_cp > 0 else 0.0

    hours = seconds_total / 3600.0

    np_below_sq = (mean_below / ftp) ** 2 if ftp > 0 else 0
    np_above_sq = (mean_above / ftp) ** 2 if ftp > 0 else 0

    below_frac = below_cp / seconds_total if seconds_total > 0 else 1.0
    above_frac = above_cp / seconds_total if seconds_total > 0 else 0.0

    raw_total = hours * (np_below_sq * below_frac + np_above_sq * above_frac) * 100
    raw_aerobic = hours * np_below_sq * below_frac * 100
    raw_anaerobic = hours * np_above_sq * above_frac * 100

    total_tss = max(raw_total, 0.1)
    aerobic_tss = max(raw_aerobic, 0.0)
    anaerobic_tss = max(raw_anaerobic, 0.0)

    aerobic_fraction = aerobic_tss / total_tss if total_tss > 0 else 1.0

    return TrainingImpact(
        aerobic_tss=round(aerobic_tss, 1),
        anaerobic_tss=round(anaerobic_tss, 1),
        total_tss=round(total_tss, 1),
        aerobic_fraction=round(aerobic_fraction, 3),
        intensity_type=classify_session_intensity(aerobic_fraction),
    )


def calculate_rolling_impact(
    impacts: List[TrainingImpact],
    window: int = 7,
) -> dict:
    """Compute rolling aerobic/anaerobic balance.

    Args:
        impacts: List of TrainingImpact from recent sessions.
        window: Rolling window in days.

    Returns:
        Dict with 'aerobic_tss', 'anaerobic_tss', 'total_tss',
        'aerobic_fraction', 'balance_status'.
    """
    if not impacts:
        return {
            "aerobic_tss": 0.0,
            "anaerobic_tss": 0.0,
            "total_tss": 0.0,
            "aerobic_fraction": 0.5,
            "balance_status": "Brak danych",
        }

    recent = impacts[-window:]
    aero = sum(i.aerobic_tss for i in recent)
    anaero = sum(i.anaerobic_tss for i in recent)
    total = aero + anaero

    frac = aero / total if total > 0 else 0.5

    if frac >= 0.80:
        status = "🟢 Zrównoważony (dominacja tlenowa)"
    elif frac >= 0.60:
        status = "🟡 Umiarkowany"
    elif frac >= 0.40:
        status = "🟠 Duża intensywność"
    else:
        status = "🔴 Dominacja beztlenowa"

    return {
        "aerobic_tss": round(aero, 1),
        "anaerobic_tss": round(anaero, 1),
        "total_tss": round(total, 1),
        "aerobic_fraction": round(frac, 3),
        "balance_status": status,
    }
