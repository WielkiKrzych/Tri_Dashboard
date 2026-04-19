"""
Plateau Detection — identifies when physiological metrics have stopped improving.

Uses rolling regression and confidence intervals to determine if a metric
has plateaued, with configurable sensitivity parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class PlateauResult:
    """Result of plateau detection for a single metric."""

    metric_name: str
    is_plateaued: bool
    plateau_start_date: Optional[str]
    confidence: float  # 0.0-1.0
    weeks_since_change: float
    rate_ci_lower: float  # %/week
    rate_ci_upper: float  # %/week
    description: str  # Polish summary


def detect_plateau(
    values: list[float],
    dates: list[str],
    metric_name: str = "",
    min_data_points: int = 4,
    significance_level: float = 0.05,
    min_plateau_weeks: float = 4.0,
) -> PlateauResult:
    """Detect if a metric has plateaued using rolling regression.

    Algorithm:
    1. Convert dates to numeric (days since first date)
    2. Fit linear regression on full dataset
    3. If slope CI includes zero AND p-value > significance_level -> plateau candidate
    4. Find earliest date where rolling 3-point slope is not significantly positive
    5. Require at least min_plateau_weeks since that date
    """
    n = len(values)
    from scipy import stats as scipy_stats

    if n < min_data_points:
        return PlateauResult(
            metric_name=metric_name,
            is_plateaued=False,
            plateau_start_date=None,
            confidence=0.0,
            weeks_since_change=0.0,
            rate_ci_lower=0.0,
            rate_ci_upper=0.0,
            description="Za mało danych do analizy",
        )

    # Convert dates to days since first date
    date_objs: list[datetime] = []
    for d in dates:
        try:
            date_objs.append(datetime.strptime(str(d)[:10], "%Y-%m-%d"))
        except (ValueError, TypeError):
            date_objs.append(datetime.now())

    days = [(d - date_objs[0]).days for d in date_objs]
    arr_values = np.array(values, dtype=float)
    arr_days = np.array(days, dtype=float)

    # Full regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(arr_days, arr_values)

    # Rate as %/week
    mean_val = np.mean(arr_values)
    rate_pct_week = (slope * 7 / mean_val * 100) if mean_val != 0 else 0.0

    # 95% CI for slope
    n_pts = len(arr_days)
    t_crit = scipy_stats.t.ppf(0.975, df=n_pts - 2) if n_pts > 2 else 1.96
    ci_lower = ((slope - t_crit * std_err) * 7 / mean_val * 100) if mean_val != 0 else 0.0
    ci_upper = ((slope + t_crit * std_err) * 7 / mean_val * 100) if mean_val != 0 else 0.0

    # Check plateau: CI includes zero and not significantly positive
    is_plateau = ci_lower <= 0 <= ci_upper and p_value > significance_level

    # Find plateau start date (earliest point where rolling slope not positive)
    plateau_start: Optional[str] = None
    weeks_since = 0.0
    if is_plateau and n >= 3:
        for i in range(n - 2):
            window_slope, _, _, _, _ = scipy_stats.linregress(
                arr_days[i : i + 3], arr_values[i : i + 3]
            )
            if window_slope <= 0:
                plateau_start = str(date_objs[i].date())
                total_days = (date_objs[-1] - date_objs[i]).days
                weeks_since = total_days / 7.0
                break

    # Require minimum plateau duration
    if is_plateau and weeks_since < min_plateau_weeks:
        is_plateau = False

    confidence = 1.0 - p_value if is_plateau else max(0.0, 1.0 - p_value)

    if is_plateau:
        desc = (
            f"Plateau od {weeks_since:.0f} tygodni "
            f"(p={p_value:.2f}, CI: {ci_lower:.1f}..{ci_upper:.1f}%/tydz)"
        )
    elif ci_lower > 0:
        desc = f"Trend wzrostowy (+{rate_pct_week:.1f}%/tydz, p={p_value:.3f})"
    else:
        desc = f"Trend stabilny ({rate_pct_week:+.1f}%/tydz, p={p_value:.3f})"

    return PlateauResult(
        metric_name=metric_name,
        is_plateaued=is_plateau,
        plateau_start_date=plateau_start,
        confidence=confidence,
        weeks_since_change=weeks_since,
        rate_ci_lower=ci_lower,
        rate_ci_upper=ci_upper,
        description=desc,
    )
