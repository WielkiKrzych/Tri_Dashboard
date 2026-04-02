"""
Periodization templates and TSS baseline suggestions.

Provides preset phase distributions, baseline TSS recommendations,
and plan validation helpers.
"""

from __future__ import annotations

from typing import List, Tuple

from .models import PeriodizationPhase


# ---------------------------------------------------------------------------
# Phase distribution presets
# ---------------------------------------------------------------------------
# Each entry: list of (phase, number_of_weeks) tuples.

PERIODIZATION_TEMPLATES: dict[str, List[Tuple[PeriodizationPhase, int]]] = {
    "13_week": [
        (PeriodizationPhase.BASE, 4),
        (PeriodizationPhase.BUILD, 5),
        (PeriodizationPhase.PEAK, 2),
        (PeriodizationPhase.TAPER, 2),
    ],
    "standard": [
        (PeriodizationPhase.BASE, 4),
        (PeriodizationPhase.BUILD, 4),
        (PeriodizationPhase.PEAK, 2),
        (PeriodizationPhase.TAPER, 2),
        (PeriodizationPhase.RECOVERY, 1),
    ],
    "base_build_only": [
        (PeriodizationPhase.BASE, 4),
        (PeriodizationPhase.BUILD, 4),
    ],
    "peak_race": [
        (PeriodizationPhase.BASE, 2),
        (PeriodizationPhase.BUILD, 4),
        (PeriodizationPhase.PEAK, 2),
        (PeriodizationPhase.TAPER, 2),
    ],
}

# Display names for templates (Polish)
_TEMPLATE_DISPLAY: dict[str, str] = {
    "13_week": "13 tyg (4+5+2+2)",
    "standard": "Standard (4+4+2+2+1)",
    "base_build_only": "Baza + Budowa (4+4)",
    "peak_race": "Start (2+4+2+2)",
}


def template_display_name(key: str) -> str:
    """Return human-readable name for a periodization template key."""
    return _TEMPLATE_DISPLAY.get(key, key)


def suggest_baseline_tss(ctl: float | None = None, experience_years: float = 0.0) -> float:
    """Suggest a weekly TSS baseline.

    Args:
        ctl: Current Chronic Training Load (from PMC). Takes precedence if > 0.
        experience_years: Athlete experience in years (used when CTL is unavailable).

    Returns:
        Recommended weekly TSS baseline.
    """
    if ctl is not None and ctl > 0:
        # Use CTL as baseline — athletes typically train at ~100% of CTL
        return round(ctl * 5, 0)  # CTL is daily avg → weekly = CTL * 5 training days

    # Fallback: experience-based estimation
    if experience_years >= 5:
        return 400.0
    if experience_years >= 3:
        return 300.0
    if experience_years >= 1:
        return 200.0
    return 150.0


def validate_plan_tss_ramp(weekly_tss_values: List[float], max_ramp_pct: float = 7.0) -> List[str]:
    """Check that weekly TSS doesn't ramp too aggressively.

    Args:
        weekly_tss_values: List of weekly TSS totals.
        max_ramp_pct: Maximum allowed week-to-week increase as a percentage.

    Returns:
        List of warning strings (empty if no warnings).
    """
    warnings: list[str] = []
    for i in range(1, len(weekly_tss_values)):
        prev = weekly_tss_values[i - 1]
        curr = weekly_tss_values[i]
        if prev > 0:
            ramp_pct = ((curr - prev) / prev) * 100
            if ramp_pct > max_ramp_pct:
                warnings.append(
                    f"Tydzień {i + 1}: skok TSS {ramp_pct:.1f}% "
                    f"(limit: {max_ramp_pct:.0f}%) — ryzyko przetrenowania"
                )
    return warnings
