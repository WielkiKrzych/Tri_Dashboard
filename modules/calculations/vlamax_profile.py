"""VLamax metabolic profile estimation.

Builds a complete metabolic profile from power-duration curve data,
including rider type classification and aerobic/anaerobic decomposition.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .stamina import (
    estimate_vlamax_from_pdc,
    get_vlamax_interpretation,
    calculate_aerobic_contribution,
)


@dataclass
class VLamaxProfile:
    """Complete metabolic profile for a rider."""

    vlamax: float
    confidence: float
    rider_type: str
    aerobic_pct_dict: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None


_DURATIONS_SEC = [5, 15, 30, 60, 120, 240, 480, 960, 1920]
_DURATION_LABELS = {
    5: "5s",
    15: "15s",
    30: "30s",
    60: "1min",
    120: "2min",
    240: "4min",
    480: "8min",
    960: "16min",
    1920: "32min",
}


def _classify_rider(vlamax: float) -> str:
    if vlamax >= 1.0:
        return "Sprinter"
    elif vlamax >= 0.7:
        return "Puncheur"
    elif vlamax >= 0.45:
        return "All-rounder"
    else:
        return "Climber"


def build_vlamax_profile(
    pdc: Dict[float, float],
    weight: float,
    vo2max: Optional[float] = None,
) -> Optional[VLamaxProfile]:
    """Build a VLamax profile from power-duration curve data.

    Args:
        pdc: Dict mapping duration (seconds) → mean max power (watts).
        weight: Rider weight in kg.
        vo2max: Optional VO2max in ml/kg/min for aerobic decomposition.

    Returns:
        VLamaxProfile or None if estimation fails.
    """
    if not pdc or weight <= 0:
        return None

    vlamax = estimate_vlamax_from_pdc(pdc, weight)
    if vlamax is None or np.isnan(vlamax) or vlamax <= 0:
        return None

    interp = get_vlamax_interpretation(vlamax)
    confidence = interp.get("confidence", 0.5) if isinstance(interp, dict) else 0.5
    rider_type = _classify_rider(vlamax)

    aerobic_pct_dict: Dict[str, float] = {}
    if vo2max and vo2max > 0:
        for dur in _DURATIONS_SEC:
            if dur in pdc:
                contrib = calculate_aerobic_contribution(pdc, vo2max, weight)
                if contrib is not None:
                    aerobic_pct_dict[_DURATION_LABELS[dur]] = round(contrib * 100, 1)

    return VLamaxProfile(
        vlamax=round(vlamax, 3),
        confidence=round(confidence, 2),
        rider_type=rider_type,
        aerobic_pct_dict=aerobic_pct_dict,
    )


def get_metabolic_profile_chart_data(profile: VLamaxProfile) -> dict:
    """Return data dicts suitable for charting.

    Returns:
        Dict with 'durations', 'aerobic_pct', 'anaerobic_pct' lists.
    """
    if not profile.aerobic_pct_dict:
        return {"durations": [], "aerobic_pct": [], "anaerobic_pct": []}

    durations = list(profile.aerobic_pct_dict.keys())
    aerobic = list(profile.aerobic_pct_dict.values())
    anaerobic = [round(100 - a, 1) for a in aerobic]

    return {
        "durations": durations,
        "aerobic_pct": aerobic,
        "anaerobic_pct": anaerobic,
    }


def compare_vlamax_longitudinal(profiles: List[VLamaxProfile]) -> dict:
    """Compare VLamax values across multiple sessions.

    Args:
        profiles: List of VLamaxProfile from different sessions.

    Returns:
        Dict with 'timestamps', 'vlamax_values', 'rider_types', 'trend_slope'.
    """
    if not profiles:
        return {
            "timestamps": [],
            "vlamax_values": [],
            "rider_types": [],
            "trend_slope": 0.0,
        }

    timestamps = [p.timestamp or "unknown" for p in profiles]
    values = [p.vlamax for p in profiles]
    types = [p.rider_type for p in profiles]

    slope = 0.0
    if len(values) >= 2:
        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        if np.std(y) > 1e-9:
            coeffs = np.polyfit(x, y, 1)
            slope = round(float(coeffs[0]), 4)

    return {
        "timestamps": timestamps,
        "vlamax_values": values,
        "rider_types": types,
        "trend_slope": slope,
    }
