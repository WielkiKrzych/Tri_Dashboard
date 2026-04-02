"""
VO2/VCO2 Estimation Module.

Provides estimation of oxygen consumption (VO2) and carbon dioxide production (VCO2)
from power and heart rate data when direct gas exchange measurements are unavailable.

Uses ACSM and Wasserman equations for estimation.
"""

from typing import Optional

import numpy as np


def estimate_vo2_from_power(
    power_watts: np.ndarray,
    body_weight_kg: float = 70.0,
    efficiency: float = 0.23,
    resting_vo2_ml_min: float = 250.0,
) -> np.ndarray:
    """
    Estimate VO2 from power using ACSM cycling equation.

    VO2 (mL/min) = (Power (W) × 60) / (efficiency × 20.9 J/mL O2) + Resting VO2

    Args:
        power_watts: Array of power values in watts
        body_weight_kg: Rider body weight in kg (default 70kg)
        efficiency: Cycling efficiency (default 0.23 or 23%)
        resting_vo2_ml_min: Resting VO2 in mL/min (default 250)

    Returns:
        Array of estimated VO2 values in mL/min
    """
    # Convert power to oxygen cost
    # 1 Watt = 1 Joule/second
    # Energy cost = Power / efficiency
    # VO2 (mL/min) = (Power × 60) / (efficiency × 20.9 J/mL O2)
    # Simplified: VO2 = Power × 2.87 + resting

    power_array = np.asarray(power_watts)
    vo2_ml_min = (power_array * 60) / (efficiency * 20.9) + resting_vo2_ml_min

    return vo2_ml_min


def estimate_vo2_from_hr(
    hr_bpm: np.ndarray,
    hr_max: float = 185.0,
    hr_rest: float = 60.0,
    vo2max_ml_kg_min: float = 55.0,
    body_weight_kg: float = 70.0,
) -> np.ndarray:
    """
    Estimate VO2 from heart rate using %HRmax-%VO2max relationship.

    Args:
        hr_bpm: Array of heart rate values in bpm
        hr_max: Maximum heart rate in bpm
        hr_rest: Resting heart rate in bpm
        vo2max_ml_kg_min: VO2max in mL/kg/min
        body_weight_kg: Body weight in kg

    Returns:
        Array of estimated VO2 values in mL/min
    """
    hr_array = np.asarray(hr_bpm)

    # %HRR = (HR - HRrest) / (HRmax - HRrest)
    hrr_fraction = (hr_array - hr_rest) / (hr_max - hr_rest)
    hrr_fraction = np.clip(hrr_fraction, 0.0, 1.0)

    # Swain et al. (1998): %VO2R ≈ %HRR (linear relationship)
    # VO2 = %HRR × (VO2max - VO2rest) + VO2rest
    resting_vo2_ml_min = 3.5 * body_weight_kg  # 1 MET
    vo2max_ml_min = vo2max_ml_kg_min * body_weight_kg
    vo2_reserve = vo2max_ml_min - resting_vo2_ml_min
    vo2_ml_min = hrr_fraction * vo2_reserve + resting_vo2_ml_min

    return vo2_ml_min


def estimate_vco2_from_vo2(
    vo2_ml_min: np.ndarray,
    rer: Optional[np.ndarray] = None,
    power_watts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate VCO2 from VO2 using RER (Respiratory Exchange Ratio).

    If RER not provided, estimate from power intensity:
    - Low intensity (<VT1): RER ≈ 0.85
    - Moderate (VT1-VT2): RER ≈ 0.90
    - High (>VT2): RER ≈ 1.00+

    Args:
        vo2_ml_min: Array of VO2 values in mL/min
        rer: Optional array of RER values
        power_watts: Optional array of power for RER estimation

    Returns:
        Array of estimated VCO2 values in mL/min
    """
    vo2_array = np.asarray(vo2_ml_min)

    if rer is not None:
        rer_array = np.asarray(rer)
    elif power_watts is not None:
        # Estimate RER from power percentiles
        power_array = np.asarray(power_watts)
        p25, p75 = np.percentile(power_array, [25, 75])

        # RER increases with intensity
        rer_array = np.where(power_array < p25, 0.85, np.where(power_array < p75, 0.90, 1.00))
    else:
        # Default moderate intensity
        rer_array = np.full_like(vo2_array, 0.90)

    vco2_ml_min = vo2_array * rer_array
    return vco2_ml_min
