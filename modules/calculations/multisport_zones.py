"""
Multi-sport training zone calculations — Cycling, Running, Swimming.

Coggan 7-zone model for cycling, pace-based zones for running (from threshold pace),
and CSS-based zones for swimming.

References:
    Coggan AR (2003). Training zones for cycling.
    Skiba PF (2015). Running critical pace model.
    Wakayoshi et al. (1992). CSS (Critical Swim Speed).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class SportZones:
    sport: str
    threshold_value: float
    zones: List[Tuple[str, float, float]]


def calculate_cycling_zones(ftp: float) -> SportZones:
    """Coggan Classic 7-zone model from FTP."""
    zones = [
        ("Z1 - Aktywna Regeneracja", 0.0, ftp * 0.55),
        ("Z2 - Wytrzymałość", ftp * 0.55, ftp * 0.75),
        ("Z3 - Tempo", ftp * 0.75, ftp * 0.90),
        ("Z4 - Próg Mleczanowy", ftp * 0.90, ftp * 1.05),
        ("Z5 - VO2max", ftp * 1.05, ftp * 1.20),
        ("Z6 - Anaerobowa", ftp * 1.20, ftp * 1.50),
        ("Z7 - Neuromięśniowa", ftp * 1.50, 9999.0),
    ]
    return SportZones(sport="Kolarstwo", threshold_value=ftp, zones=zones)


def calculate_running_zones(threshold_pace_sec_km: float) -> SportZones:
    """Running pace zones based on threshold pace (sec/km)."""
    zones = [
        ("Z1 - Łatwy", threshold_pace_sec_km * 1.40, threshold_pace_sec_km * 1.20),
        ("Z2 - Aerobowy", threshold_pace_sec_km * 1.20, threshold_pace_sec_km * 1.05),
        ("Z3 - Tempo", threshold_pace_sec_km * 1.05, threshold_pace_sec_km * 0.95),
        ("Z4 - Próg", threshold_pace_sec_km * 0.95, threshold_pace_sec_km * 0.87),
        ("Z5 - VO2max", threshold_pace_sec_km * 0.87, threshold_pace_sec_km * 0.78),
        ("Z6 - Anaerobowa", threshold_pace_sec_km * 0.78, threshold_pace_sec_km * 0.65),
    ]
    return SportZones(sport="Bieg", threshold_value=threshold_pace_sec_km, zones=zones)


def calculate_swim_zones(css_sec_100m: float) -> SportZones:
    """Swim pace zones based on Critical Swim Speed (sec/100m)."""
    zones = [
        ("Z1 - Łatwy", css_sec_100m * 1.30, css_sec_100m * 1.15),
        ("Z2 - Aerobowy", css_sec_100m * 1.15, css_sec_100m * 1.05),
        ("Z3 - Tempo", css_sec_100m * 1.05, css_sec_100m * 0.97),
        ("Z4 - Próg", css_sec_100m * 0.97, css_sec_100m * 0.90),
        ("Z5 - VO2max", css_sec_100m * 0.90, css_sec_100m * 0.80),
        ("Z6 - Sprint", css_sec_100m * 0.80, 0.0),
    ]
    return SportZones(sport="Pływanie", threshold_value=css_sec_100m, zones=zones)


def estimate_critical_pace(
    distances: List[float],
    times: List[float],
) -> Tuple[float, float]:
    """Estimate CP (critical pace, sec/km) and D' (m) from distance-time pairs.

    Uses linear regression: t = D'/v + CP_distance * distance.
    Returns (critical_pace_sec_km, d_prime_meters).
    """
    if len(distances) < 2 or len(times) < 2:
        return 0.0, 0.0
    d_arr = np.array(distances, dtype=float)
    t_arr = np.array(times, dtype=float)
    if d_arr.min() <= 0 or t_arr.min() <= 0:
        return 0.0, 0.0
    speeds = d_arr / t_arr
    A = np.column_stack([np.ones_like(speeds), 1.0 / speeds])
    result = np.linalg.lstsq(A, t_arr / d_arr, rcond=None)
    coeffs = result[0]
    inv_cp = coeffs[0]
    d_prime = coeffs[1]
    if inv_cp <= 0:
        return 0.0, 0.0
    cp_sec_per_km = (1.0 / inv_cp) * 1000.0
    return float(cp_sec_per_km), float(d_prime)


def pace_to_str(seconds_per_km: float) -> str:
    """Format pace as 'M:SS/km'."""
    if seconds_per_km <= 0 or not np.isfinite(seconds_per_km):
        return "—"
    mins = int(seconds_per_km // 60)
    secs = int(seconds_per_km % 60)
    return f"{mins}:{secs:02d}/km"


def swim_pace_to_str(seconds_per_100m: float) -> str:
    """Format swim pace as 'M:SS/100m'."""
    if seconds_per_100m <= 0 or not np.isfinite(seconds_per_100m):
        return "—"
    mins = int(seconds_per_100m // 60)
    secs = int(seconds_per_100m % 60)
    return f"{mins}:{secs:02d}/100m"
