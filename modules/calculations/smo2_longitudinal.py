"""SmO2 NIRS Longitudinal Threshold Tracker.

Extracts threshold data from individual sessions and tracks
threshold power trends over time using linear regression.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .smo2_thresholds import detect_smo2_thresholds_moxy, SmO2ThresholdResult


@dataclass
class SmO2SessionThreshold:
    """Threshold data extracted from a single session."""

    date: str
    t1_power: Optional[float] = None
    t1_smo2: Optional[float] = None
    t2_power: Optional[float] = None
    t2_smo2: Optional[float] = None
    quality_grade: str = "C"


@dataclass
class SmO2LongitudinalTrend:
    """Longitudinal trend across multiple sessions."""

    sessions: List[SmO2SessionThreshold] = field(default_factory=list)
    power_trend_slope: float = 0.0
    power_trend_r: float = 0.0
    direction: str = "stable"
    confidence: float = 0.0


_QUALITY_GRADES = {"A", "B", "C", "D"}


def _grade_quality(result: SmO2ThresholdResult) -> str:
    """Assign quality grade based on threshold completeness."""
    has_t1 = result.t1_watts is not None and result.t1_smo2 is not None
    has_t2 = result.t2_onset_watts is not None or result.t2_watts is not None

    if has_t1 and has_t2:
        return "A"
    elif has_t1:
        return "B"
    elif has_t2:
        return "C"
    return "D"


def extract_session_thresholds(
    session_data: dict,
) -> Optional[SmO2SessionThreshold]:
    """Extract thresholds from a single session's data.

    Args:
        session_data: Dict with 'date', 'df' (DataFrame), optional 'cp'.

    Returns:
        SmO2SessionThreshold or None if detection fails.
    """
    date = session_data.get("date", "")
    df = session_data.get("df")

    if df is None or (hasattr(df, "empty") and df.empty):
        return None

    required_cols = {"smo2", "watts"}
    if hasattr(df, "columns"):
        available = {c.lower().strip() for c in df.columns}
        if not required_cols.issubset(available):
            return None

    cp = session_data.get("cp")

    try:
        result = detect_smo2_thresholds_moxy(df, cp_watts=cp)
    except Exception:
        return None

    if result is None:
        return None

    t1_power = float(result.t1_watts) if result.t1_watts is not None else None
    t1_smo2 = float(result.t1_smo2) if result.t1_smo2 is not None else None

    t2_w = result.t2_onset_watts if result.t2_onset_watts is not None else result.t2_watts
    t2_s = result.t2_onset_smo2 if result.t2_onset_smo2 is not None else result.t2_smo2

    t2_power = float(t2_w) if t2_w is not None else None
    t2_smo2 = float(t2_s) if t2_s is not None else None

    grade = _grade_quality(result)

    return SmO2SessionThreshold(
        date=date,
        t1_power=t1_power,
        t1_smo2=t1_smo2,
        t2_power=t2_power,
        t2_smo2=t2_smo2,
        quality_grade=grade,
    )


def calculate_longitudinal_trend(
    thresholds: List[SmO2SessionThreshold],
    min_grade: str = "B",
) -> SmO2LongitudinalTrend:
    """Calculate longitudinal trend from threshold data.

    Args:
        thresholds: List of session thresholds.
        min_grade: Minimum quality grade to include (A > B > C > D).

    Returns:
        SmO2LongitudinalTrend with regression results.
    """
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    min_idx = grade_order.get(min_grade, 1)

    filtered = [
        t
        for t in thresholds
        if t.t1_power is not None and grade_order.get(t.quality_grade, 3) <= min_idx
    ]

    if not filtered:
        return SmO2LongitudinalTrend(sessions=thresholds)

    powers = np.array([t.t1_power for t in filtered], dtype=float)
    x = np.arange(len(powers), dtype=float)

    slope, r = 0.0, 0.0
    if len(powers) >= 2 and np.std(powers) > 1e-6:
        coeffs = np.polyfit(x, powers, 1)
        slope = float(coeffs[0])
        correlation_matrix = np.corrcoef(x, powers)
        r = float(abs(correlation_matrix[0, 1])) if not np.isnan(correlation_matrix[0, 1]) else 0.0

    if slope > 1.0:
        direction = "improving"
    elif slope < -1.0:
        direction = "declining"
    else:
        direction = "stable"

    confidence = min(r, 1.0)

    return SmO2LongitudinalTrend(
        sessions=thresholds,
        power_trend_slope=round(slope, 2),
        power_trend_r=round(r, 3),
        direction=direction,
        confidence=round(confidence, 2),
    )


def interpret_trend(trend: SmO2LongitudinalTrend) -> str:
    """Polish interpretation of the longitudinal trend."""
    if not trend.sessions:
        return "Brak danych do analizy trendu."

    arrows = {"improving": "⬆️ Poprawa", "stable": "➡️ Stabilny", "declining": "⬇️ Spadek"}

    direction_text = arrows.get(trend.direction, "➡️ Stabilny")
    n = len(trend.sessions)

    text = f"{direction_text} — {n} sesji analizowanych"

    if trend.power_trend_r > 0.7:
        text += " (trend wyraźny)"
    elif trend.power_trend_r > 0.4:
        text += " (trend umiarkowany)"
    else:
        text += " (trend słaby/brak)"

    if trend.direction == "improving":
        text += ". Progi tlenowe przesuwają się w górę — organizm adaptuje się do treningu."
    elif trend.direction == "declining":
        text += ". Progi tlenowe spadają — możliwe przepracowanie lub detrenning."

    return text
