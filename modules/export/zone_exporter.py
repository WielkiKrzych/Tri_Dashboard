"""
Power and HR zone calculations + CSV export.

Implements 7-zone Coggan power zones and 5-zone HR zones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PowerZone:
    """Single training zone with percentage and absolute bounds."""

    zone: int
    name: str
    low_pct: float
    high_pct: float
    low_watts: float
    high_watts: float


# Coggan 7-zone power model (fractions of CP)
POWER_ZONE_DEFS: list[tuple[int, str, float, float]] = [
    (1, "Active Recovery", 0.0, 0.55),
    (2, "Endurance", 0.55, 0.75),
    (3, "Tempo", 0.75, 0.90),
    (4, "Threshold", 0.90, 1.05),
    (5, "VO2 Max", 1.05, 1.20),
    (6, "Anaerobic Capacity", 1.20, 1.50),
    (7, "Neuromuscular", 1.50, 999.0),
]

# 5-zone HR model (fractions of max HR)
HR_ZONE_DEFS: list[tuple[int, str, float, float]] = [
    (1, "Recovery", 0.0, 0.60),
    (2, "Endurance", 0.60, 0.70),
    (3, "Tempo", 0.70, 0.80),
    (4, "Threshold", 0.80, 0.90),
    (5, "VO2 Max", 0.90, 1.05),
]


def calculate_power_zones(cp: float) -> list[PowerZone]:
    """Calculate 7-zone Coggan power zones from Critical Power.

    Args:
        cp: Critical Power in watts.

    Returns:
        List of PowerZone objects. Empty if cp <= 0.
    """
    if cp <= 0:
        return []
    return [
        PowerZone(
            zone=z,
            name=n,
            low_pct=lp,
            high_pct=hp,
            low_watts=cp * lp,
            high_watts=min(cp * hp, 9999),
        )
        for z, n, lp, hp in POWER_ZONE_DEFS
    ]


def calculate_hr_zones(max_hr: float) -> list[PowerZone]:
    """Calculate 5-zone HR zones from maximum heart rate.

    Reuses PowerZone dataclass with bpm values in low_watts/high_watts.

    Args:
        max_hr: Maximum heart rate in bpm.

    Returns:
        List of PowerZone objects. Empty if max_hr <= 0.
    """
    if max_hr <= 0:
        return []
    return [
        PowerZone(
            zone=z,
            name=n,
            low_pct=lp,
            high_pct=hp,
            low_watts=max_hr * lp,
            high_watts=min(max_hr * hp, 250),
        )
        for z, n, lp, hp in HR_ZONE_DEFS
    ]


def export_power_zones_csv(cp: float, rider_weight: float | None = None) -> bytes:
    """Export power zones as CSV bytes.

    Args:
        cp: Critical Power in watts.
        rider_weight: Optional rider weight in kg for w/kg columns.

    Returns:
        UTF-8 encoded CSV content.
    """
    zones = calculate_power_zones(cp)
    lines = ["zone,name,low_pct,high_pct,low_watts,high_watts"]
    if rider_weight and rider_weight > 0:
        lines[0] += ",low_wkg,high_wkg"
    for z in zones:
        line = (
            f"{z.zone},{z.name},{z.low_pct:.2f},{z.high_pct:.2f},"
            f"{z.low_watts:.0f},{z.high_watts:.0f}"
        )
        if rider_weight and rider_weight > 0:
            line += f",{z.low_watts / rider_weight:.2f},{z.high_watts / rider_weight:.2f}"
        lines.append(line)
    return "\n".join(lines).encode("utf-8")


def export_hr_zones_csv(max_hr: float) -> bytes:
    """Export HR zones as CSV bytes.

    Args:
        max_hr: Maximum heart rate in bpm.

    Returns:
        UTF-8 encoded CSV content.
    """
    zones = calculate_hr_zones(max_hr)
    lines = ["zone,name,low_bpm,high_bpm"]
    for z in zones:
        lines.append(f"{z.zone},{z.name},{z.low_watts:.0f},{z.high_watts:.0f}")
    return "\n".join(lines).encode("utf-8")
