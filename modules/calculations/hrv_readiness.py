"""HRV-based Readiness Score calculation.

Computes a 0-100 readiness score from RMSSD history using
7-day coefficient of variation, with Polish recommendations.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ReadinessScore:
    """Training readiness assessment from HRV data."""

    date: str
    score: float
    level: str
    color: str
    rmssd_7day_avg: float
    rmssd_cv: float
    recommendation: str


_LEVEL_THRESHOLDS = [
    (80, ("Wysoka gotowość", "#27ae60")),
    (60, ("Gotowość umiarkowana", "#f39c12")),
    (40, ("Obniżona gotowość", "#e67e22")),
    (20, ("Niska gotowość", "#e74c3c")),
    (0, ("Krytycznie niska", "#8e44ad")),
]


def get_readiness_level(score: float) -> Tuple[str, str]:
    """Map score 0-100 to (level_name, color_hex)."""
    for threshold, (name, color) in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return name, color
    return "Krytycznie niska", "#8e44ad"


def get_readiness_recommendation(level: str, score: float) -> str:
    """Polish recommendation based on readiness level."""
    recommendations = {
        "Wysoka gotowość": (
            "Organizm gotowy na intensywny trening. "
            "Możesz wykonać trening interwałowy lub test wydolnościowy."
        ),
        "Gotowość umiarkowana": (
            "Stan dobry, ale nie na peak performance. Wykonaj trening w strefie tempa/threshold."
        ),
        "Obniżona gotowość": (
            "Wskazówki na zmęczenie. Zalecany trening tlenowy Z1-Z2 lub dzień regeneracji aktywnej."
        ),
        "Niska gotowość": (
            "Znaczne zmęczenie organizmu. Zalecany dzień regeneracji lub bardzo lekki trening Z1."
        ),
        "Krytycznie niska": (
            "Poważne zmęczenie lub stres. "
            "Całkowity odpoczynek, sen, nawodnienie. Skonsultuj się z trenerem."
        ),
    }
    return recommendations.get(level, "Brak rekomendacji.")


def calculate_readiness_score(
    rmssd_history: List[dict],
) -> Optional[ReadinessScore]:
    """Calculate readiness from RMSSD history.

    Args:
        rmssd_history: List of dicts with 'date' and 'rmssd' keys.

    Returns:
        ReadinessScore or None if insufficient data.
    """
    if not rmssd_history or len(rmssd_history) < 3:
        return None

    values = []
    latest_date = ""
    for entry in rmssd_history[-7:]:
        rmssd = entry.get("rmssd")
        if rmssd is not None and not np.isnan(rmssd) and rmssd > 0:
            values.append(float(rmssd))
            latest_date = entry.get("date", "")

    if len(values) < 3:
        return None

    arr = np.array(values)
    mean_rmssd = float(np.mean(arr))
    std_rmssd = float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0
    cv = (std_rmssd / mean_rmssd * 100) if mean_rmssd > 0 else 100.0

    # Score: low CV = high readiness
    # CV < 10% → 100, CV > 60% → 0
    cv_clamped = np.clip(cv, 0, 60)
    score = float(100 * (1 - cv_clamped / 60))
    score = round(np.clip(score, 0, 100), 1)

    level, color = get_readiness_level(score)
    recommendation = get_readiness_recommendation(level, score)

    return ReadinessScore(
        date=latest_date,
        score=score,
        level=level,
        color=color,
        rmssd_7day_avg=round(mean_rmssd, 1),
        rmssd_cv=round(cv, 1),
        recommendation=recommendation,
    )
