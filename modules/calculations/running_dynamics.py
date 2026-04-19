"""Running Dynamics calculations.

Estimates ground contact time, vertical oscillation, leg spring stiffness,
and stride metrics from accelerometer data.

Pure functions — no Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class RunningDynamicsMetrics:
    """Running dynamics metrics from accelerometer data."""

    ground_contact_time_ms: float
    vertical_oscillation_mm: float
    leg_spring_stiffness_kn_m: float
    vertical_ratio_pct: float
    cadence_spm: float
    stride_length_m: float


def calculate_running_dynamics(
    accel_x: list | np.ndarray,
    accel_y: list | np.ndarray,
    accel_z: list | np.ndarray,
    time_arr: list | np.ndarray,
    cadence: Optional[float] = None,
    body_mass_kg: float = 75.0,
    pace_min_km: Optional[float] = None,
) -> Optional[RunningDynamicsMetrics]:
    """Calculate running dynamics from accelerometer data.

    Args:
        accel_x: Lateral acceleration (g or m/s²).
        accel_y: Vertical acceleration (g or m/s²).
        accel_z: Forward acceleration (g or m/s²).
        time_arr: Time array in seconds.
        cadence: Optional cadence in spm (if not provided, estimated).
        body_mass_kg: Body mass for spring stiffness calculation.
        pace_min_km: Pace in min/km for stride length estimation.

    Returns:
        RunningDynamicsMetrics or None if insufficient data.
    """
    ax = np.asarray(accel_x, dtype=float)
    ay = np.asarray(accel_y, dtype=float)
    az = np.asarray(accel_z, dtype=float)
    t = np.asarray(time_arr, dtype=float)

    min_samples = 100
    if len(t) < min_samples or len(ax) < min_samples:
        return None

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.01
    if dt <= 0:
        return None

    # Ground contact time from vertical acceleration zero-crossings
    gct_ms = _estimate_ground_contact_time(ay, dt)

    # Vertical oscillation from RMS of vertical displacement
    vert_osc_mm = _estimate_vertical_oscillation(ay, dt)

    # Leg spring stiffness
    if vert_osc_mm > 0 and body_mass_kg > 0:
        displacement_m = vert_osc_mm / 1000.0
        peak_force_n = body_mass_kg * 9.81 * 2.5  # ~2.5x BW at midstance
        stiffness = peak_force_n / displacement_m / 1000.0  # kN/m
    else:
        stiffness = 0.0

    # Cadence estimation
    if cadence is None or cadence <= 0:
        cadence = _estimate_cadence(ay, dt)

    # Stride length
    stride_length = 0.0
    if pace_min_km is not None and pace_min_km > 0 and cadence > 0:
        speed_m_s = 1000.0 / (pace_min_km * 60.0)
        stride_length = speed_m_s / (cadence / 60.0)

    # Vertical ratio: oscillation / stride length
    vertical_ratio = (vert_osc_mm / 1000.0 / stride_length * 100) if stride_length > 0 else 0.0

    return RunningDynamicsMetrics(
        ground_contact_time_ms=round(gct_ms, 1),
        vertical_oscillation_mm=round(vert_osc_mm, 1),
        leg_spring_stiffness_kn_m=round(stiffness, 2),
        vertical_ratio_pct=round(vertical_ratio, 1),
        cadence_spm=round(cadence, 0),
        stride_length_m=round(stride_length, 3),
    )


def _estimate_ground_contact_time(vert_accel: np.ndarray, dt: float) -> float:
    """Estimate ground contact time from vertical acceleration peaks."""
    abs_ay = np.abs(vert_accel)
    if len(abs_ay) < 10:
        return 0.0

    threshold = np.percentile(abs_ay, 75)
    above = abs_ay > threshold

    contact_times = []
    in_contact = False
    start = 0

    for i in range(len(above)):
        if above[i] and not in_contact:
            in_contact = True
            start = i
        elif not above[i] and in_contact:
            contact_times.append((i - start) * dt * 1000)
            in_contact = False

    if not contact_times:
        return 250.0  # typical default

    return float(np.median(contact_times))


def _estimate_vertical_oscillation(vert_accel: np.ndarray, dt: float) -> float:
    """Estimate vertical oscillation from vertical acceleration RMS."""
    if len(vert_accel) < 10:
        return 0.0

    # Double integration of acceleration → displacement
    vert_vel = np.cumsum(vert_accel) * dt
    vert_vel -= np.mean(vert_vel)  # remove drift
    vert_disp = np.cumsum(vert_vel) * dt
    vert_disp -= np.mean(vert_disp)  # remove drift

    # RMS displacement in meters → mm
    rms_m = np.sqrt(np.mean(vert_disp**2))
    return float(rms_m * 1000 * 2)  # peak-to-peak ≈ 2× RMS


def _estimate_cadence(vert_accel: np.ndarray, dt: float) -> float:
    """Estimate cadence from vertical acceleration frequency."""
    if len(vert_accel) < 50:
        return 170.0  # typical default

    # FFT to find dominant frequency
    n = len(vert_accel)
    fft_vals = np.abs(np.fft.rfft(vert_accel - np.mean(vert_accel)))
    freqs = np.fft.rfftfreq(n, dt)

    # Cadence typically 140-200 spm → 2.3-3.3 Hz (steps, not strides)
    # Strides = half steps, so 70-100 spm stride freq → 1.2-1.7 Hz
    mask = (freqs > 1.5) & (freqs < 4.5)
    if not np.any(mask):
        return 170.0

    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return float(peak_freq * 60)  # Hz → steps/min


def classify_running_economics(metrics: RunningDynamicsMetrics) -> str:
    """Classify running economy based on metrics.

    Returns:
        Polish interpretation: good/average/poor.
    """
    score = 0

    if 200 < metrics.ground_contact_time_ms < 260:
        score += 1
    elif metrics.ground_contact_time_ms >= 300:
        score -= 1

    if 5 < metrics.vertical_oscillation_mm < 10:
        score += 1
    elif metrics.vertical_oscillation_mm >= 12:
        score -= 1

    if metrics.leg_spring_stiffness_kn_m > 10:
        score += 1
    elif metrics.leg_spring_stiffness_kn_m < 5:
        score -= 1

    if score >= 2:
        return "🟢 Dobra ekonomia biegu"
    elif score >= 0:
        return "🟡 Przeciętna ekonomia biegu"
    return "🔴 Słaba ekonomia biegu — rozważ poprawę techniki"


def get_ideal_ranges(pace_min_km: float) -> Dict[str, tuple]:
    """Get ideal running dynamics ranges for a given pace.

    Args:
        pace_min_km: Pace in min/km.

    Returns:
        Dict with ideal ranges for each metric.
    """
    if pace_min_km < 4.0:
        return {
            "ground_contact_time_ms": (190, 230),
            "vertical_oscillation_mm": (5, 8),
            "leg_spring_stiffness_kn_m": (12, 20),
            "vertical_ratio_pct": (5, 8),
            "cadence_spm": (178, 195),
        }
    elif pace_min_km < 5.0:
        return {
            "ground_contact_time_ms": (220, 270),
            "vertical_oscillation_mm": (6, 10),
            "leg_spring_stiffness_kn_m": (8, 14),
            "vertical_ratio_pct": (6, 9),
            "cadence_spm": (172, 188),
        }
    elif pace_min_km < 6.0:
        return {
            "ground_contact_time_ms": (250, 300),
            "vertical_oscillation_mm": (7, 12),
            "leg_spring_stiffness_kn_m": (6, 11),
            "vertical_ratio_pct": (7, 10),
            "cadence_spm": (165, 182),
        }
    return {
        "ground_contact_time_ms": (270, 340),
        "vertical_oscillation_mm": (8, 14),
        "leg_spring_stiffness_kn_m": (5, 9),
        "vertical_ratio_pct": (8, 12),
        "cadence_spm": (158, 178),
    }
