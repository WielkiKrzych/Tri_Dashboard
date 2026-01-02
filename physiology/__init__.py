"""
Physiology Module - Physiological Calculations

This module contains all physiological calculation functions.
NO STREAMLIT OR UI DEPENDENCIES ALLOWED.

Sub-modules:
- hrv: HRV and DFA Alpha-1 analysis
- power: Power-based calculations (NP, TSS, W'bal)
- thresholds: VT1/VT2, LT detection
- thermal: Heat strain calculations
"""

# Re-export from modules.calculations for now (gradual migration)
from modules.calculations.hrv import (
    calculate_dynamic_dfa,
)

from modules.calculations.power import (
    calculate_normalized_power,
    calculate_pulse_power_stats,
    calculate_power_duration_curve,
    calculate_fatigue_resistance_index,
    count_match_burns,
    calculate_power_zones_time,
    estimate_tte,
    classify_phenotype,
)

from modules.calculations.w_prime import (
    calculate_w_prime_balance,
    calculate_w_prime_fast,
    calculate_recovery_score,
    get_recovery_recommendation,
)

from modules.calculations.thermal import (
    calculate_heat_strain_index,
    calculate_thermal_decay,
)

from modules.calculations.thresholds import (
    detect_vt_transition_zone,
    analyze_step_test,
    calculate_training_zones_from_thresholds,
)

from modules.calculations.stamina import (
    calculate_stamina_score,
    estimate_vlamax_from_pdc,
    calculate_durability_index,
)

__all__ = [
    # HRV
    'calculate_dynamic_dfa',
    # Power
    'calculate_normalized_power',
    'calculate_pulse_power_stats',
    'calculate_power_duration_curve',
    'calculate_fatigue_resistance_index',
    'count_match_burns',
    'calculate_power_zones_time',
    'estimate_tte',
    'classify_phenotype',
    # W' Balance
    'calculate_w_prime_balance',
    'calculate_w_prime_fast',
    'calculate_recovery_score',
    'get_recovery_recommendation',
    # Thermal
    'calculate_heat_strain_index',
    'calculate_thermal_decay',
    # Thresholds
    'detect_vt_transition_zone',
    'analyze_step_test',
    'calculate_training_zones_from_thresholds',
    # Stamina
    'calculate_stamina_score',
    'estimate_vlamax_from_pdc',
    'calculate_durability_index',
]
