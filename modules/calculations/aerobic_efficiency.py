"""
Aerobic efficiency trends — Power/HR ratio analysis.

Computes Efficiency Factor (EF = Power/HR) overall and per-zone,
tracks trends over time, and detects cardiac drift via EF delta.

References:
    Coggan AR (2003). Efficiency Factor as training metric.
    Sanders D et al. (2022). Cardiac drift and EF decline. IJSPP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np


@dataclass
class SessionEfficiency:
    date: str
    overall_ef: float
    zone1_ef: float
    zone2_ef: float
    zone3_ef: float
    zone4_ef: float
    zone5_ef: float
    ef_start: float
    ef_end: float
    ef_delta_pct: float


@dataclass
class EfficiencyTrend:
    sessions: List[SessionEfficiency]
    overall_trend_slope: float
    overall_trend_r: float
    direction: str
    zone_trends: Dict[str, float]


def calculate_session_efficiency(
    power: np.ndarray,
    hr: np.ndarray,
    time: np.ndarray,
    cp: float,
) -> SessionEfficiency:
    """Calculate EF (Power/HR) metrics for a session."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ef = np.where(hr > 50, power / hr, np.nan)

    valid = np.isfinite(ef) & np.isfinite(power) & (power > 0) & (hr > 50)
    if np.sum(valid) < 10:
        return SessionEfficiency(
            date="",
            overall_ef=0.0,
            zone1_ef=0.0,
            zone2_ef=0.0,
            zone3_ef=0.0,
            zone4_ef=0.0,
            zone5_ef=0.0,
            ef_start=0.0,
            ef_end=0.0,
            ef_delta_pct=0.0,
        )

    overall_ef = float(np.nanmean(ef[valid]))

    zone_bounds = [
        ("zone1", 0.0, 0.55),
        ("zone2", 0.55, 0.75),
        ("zone3", 0.75, 0.90),
        ("zone4", 0.90, 1.05),
        ("zone5", 1.05, 999.0),
    ]
    zone_pcts = [cp * lo for _, lo, _ in zone_bounds], [cp * hi for _, _, hi in zone_bounds]
    zone_ef_values = {}
    for (name, lo, hi), lo_val, hi_val in zip(zone_bounds, zone_pcts[0], zone_pcts[1]):
        mask = valid & (power >= lo_val) & (power < hi_val)
        zone_ef_values[name] = float(np.nanmean(ef[mask])) if np.sum(mask) > 5 else 0.0

    n_valid = np.sum(valid)
    quarter = max(n_valid // 4, 1)
    ef_start = float(np.nanmean(ef[valid][:quarter]))
    ef_end = float(np.nanmean(ef[valid][-quarter:]))

    ef_delta_pct = ((ef_end - ef_start) / max(ef_start, 0.01)) * 100 if ef_start > 0 else 0.0

    return SessionEfficiency(
        date="",
        overall_ef=overall_ef,
        zone1_ef=zone_ef_values.get("zone1", 0.0),
        zone2_ef=zone_ef_values.get("zone2", 0.0),
        zone3_ef=zone_ef_values.get("zone3", 0.0),
        zone4_ef=zone_ef_values.get("zone4", 0.0),
        zone5_ef=zone_ef_values.get("zone5", 0.0),
        ef_start=ef_start,
        ef_end=ef_end,
        ef_delta_pct=ef_delta_pct,
    )


def calculate_efficiency_trend(sessions: List[SessionEfficiency]) -> EfficiencyTrend:
    """Linear regression on overall EF over time."""
    if len(sessions) < 2:
        return EfficiencyTrend(
            sessions=sessions,
            overall_trend_slope=0.0,
            overall_trend_r=0.0,
            direction="insufficient_data",
            zone_trends={},
        )

    ef_vals = np.array([s.overall_ef for s in sessions], dtype=float)
    x = np.arange(len(ef_vals), dtype=float)
    n = len(x)
    sx, sy = np.sum(x), np.sum(ef_vals)
    sxx, sxy = np.sum(x * x), np.sum(x * ef_vals)
    denom = n * sxx - sx * sx

    if abs(denom) < 1e-12:
        slope, r = 0.0, 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
        mean_y = sy / n
        ss_res = np.sum((ef_vals - (slope * x + (sy - slope * sx) / n)) ** 2)
        ss_tot = np.sum((ef_vals - mean_y) ** 2)
        r = np.sqrt(max(0, 1 - ss_res / max(ss_tot, 1e-12)))

    direction = "improving" if slope > 0.001 else ("declining" if slope < -0.001 else "stable")

    zone_trends = {}
    for zone_name in ["zone1", "zone2", "zone3", "zone4", "zone5"]:
        vals = np.array([getattr(s, f"{zone_name}_ef", 0.0) for s in sessions], dtype=float)
        if np.any(vals > 0):
            zone_trends[zone_name] = float(vals[-1] - vals[0])

    return EfficiencyTrend(
        sessions=sessions,
        overall_trend_slope=float(slope),
        overall_trend_r=float(r),
        direction=direction,
        zone_trends=zone_trends,
    )


def interpret_trend(direction: str) -> str:
    """Polish interpretation of efficiency trend direction."""
    return {
        "improving": "📈 Poprawa — EF rośnie, organizm staje się bardziej wydajny",
        "declining": "📉 Spadek — EF maleje, możliwe zmęczenie lub przeładowanie",
        "stable": "➡️ Stabilny — EF utrzymuje się na stałym poziomie",
        "insufficient_data": "❓ Za mało danych do oceny trendu",
    }.get(direction, "❓ Nieznany")
