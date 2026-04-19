"""
MPA (Maximum Power Available) / W'bal Enhancement.

Computes the instantaneous MPA envelope based on the Sufferfest/Wahoo model:
    MPA = W'_bal_remaining / tau_recovery + CP

When power > CP, W' depletes and MPA drops.
When power < CP, W' recovers and MPA rises toward Pmax.

References:
    Skiba PF et al. (2012, 2015). W' balance integral model.
    Caen K et al. (2021). Bi-exponential W' reconstitution. EJAP.
    Clarke DC, Skiba PF (2013). Rationale and resources for W'.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MPAProfile:
    """Result of an MPA analysis for a single session."""

    mpa_array: np.ndarray  # MPA at each second [W]
    time_array: np.ndarray  # Time axis [s]
    wbal_array: np.ndarray  # W' balance at each second [J]
    cp: float  # Critical Power [W]
    w_prime: float  # W' capacity [J]
    time_to_exhaustion_at_peak: Optional[float] = None  # TTE at peak power [s]


# ── Sport-specific time constants (same as w_prime.py bi-exponential) ──────
_SPORT_TAU: dict[int, tuple[float, float, float]] = {
    0: (120.0, 600.0, 0.50),  # Cycling
    1: (150.0, 750.0, 0.45),  # Running
    2: (90.0, 500.0, 0.55),  # Swimming
}


def calculate_time_to_exhaustion(
    wbal_remaining: float,
    current_power: float,
    cp: float,
) -> float:
    """Estimate seconds until exhaustion at *current_power*.

    TTE = W'_bal / (P - CP)  when P > CP, else +inf (not depleting).

    Args:
        wbal_remaining: Current W' balance [J].
        current_power: Current power output [W].
        cp: Critical Power [W].

    Returns:
        Seconds until W' is fully depleted, or float('inf') if below CP.
    """
    surplus = current_power - cp
    if surplus <= 0:
        return float("inf")
    return max(0.0, wbal_remaining / surplus)


def calculate_mpa(
    watts: np.ndarray,
    time: np.ndarray,
    cp: float,
    w_prime_cap: float,
    model: str = "biexp",
    sport: int = 0,
) -> MPAProfile:
    """Calculate MPA (Maximum Power Available) throughout a session.

    MPA = W'_bal_remaining / tau_recovery + CP

    Uses the same bi-exponential reconstitution model as w_prime.py to
    determine the effective recovery time constant.

    Args:
        watts: Power array [W].
        time: Time array [s].
        cp: Critical Power [W].
        w_prime_cap: W' capacity [J].
        model: "biexp" (Caen 2021) or "skiba" (Skiba 2015).
        sport: 0=cycling, 1=running, 2=swimming.

    Returns:
        MPAProfile with MPA envelope, W' balance, and key metrics.
    """
    n = len(watts)
    wbal = np.empty(n, dtype=np.float64)
    mpa = np.empty(n, dtype=np.float64)

    curr_w = w_prime_cap
    prev_time = time[0]

    tau_fast, tau_slow, fast_frac = _SPORT_TAU.get(sport, _SPORT_TAU[0])
    slow_frac = 1.0 - fast_frac

    for i in range(n):
        dt = 1.0 if i == 0 else max(time[i] - prev_time, 1.0)
        if i > 0:
            prev_time = time[i]

        p = watts[i]

        if p > cp:
            # Depletion
            curr_w += (cp - p) * dt
        elif p < cp:
            dcp = cp - p
            if model == "biexp":
                tau_f = tau_fast * np.exp(-0.008 * dcp) + tau_fast * 0.3
                tau_s = tau_slow * np.exp(-0.005 * dcp) + tau_slow * 0.3
                rec = fast_frac * (1.0 - np.exp(-dt / tau_f)) + slow_frac * (
                    1.0 - np.exp(-dt / tau_s)
                )
            else:
                # Skiba mono-exponential
                tau = 546.0 * np.exp(-0.01 * dcp) + 316.0
                rec = 1.0 - np.exp(-dt / tau)
            depleted = w_prime_cap - curr_w
            curr_w += depleted * rec

        curr_w = np.clip(curr_w, 0.0, w_prime_cap)
        wbal[i] = curr_w

        # MPA = W'bal / tau_eff + CP
        # Effective tau: how fast W' would recover at current power.
        # When above CP, tau_eff is negative → MPA drops.
        if p <= cp:
            dcp = cp - p
            if model == "biexp":
                tau_f = tau_fast * np.exp(-0.008 * dcp) + tau_fast * 0.3
                tau_s = tau_slow * np.exp(-0.005 * dcp) + tau_slow * 0.3
                tau_eff = 1.0 / (fast_frac / tau_f + slow_frac / tau_s)
            else:
                tau_eff = 546.0 * np.exp(-0.01 * dcp) + 316.0
            mpa[i] = curr_w / max(tau_eff, 1.0) + cp
        else:
            # Above CP → MPA is what you could sustain if W' were full:
            # We show actual capacity remaining + CP
            mpa[i] = curr_w / max(1.0, dt) + cp

    # Time to exhaustion at peak power
    peak_power = float(np.max(watts))
    tte_peak = calculate_time_to_exhaustion(w_prime_cap, peak_power, cp)

    return MPAProfile(
        mpa_array=mpa,
        time_array=time,
        wbal_array=wbal,
        cp=float(cp),
        w_prime=float(w_prime_cap),
        time_to_exhaustion_at_peak=tte_peak,
    )
