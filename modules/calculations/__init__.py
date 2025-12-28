"""
SOLID: Single Responsibility Principle - Reorganizacja obliczeń.

Ten pakiet grupuje funkcje obliczeniowe według odpowiedzialności:
- w_prime.py: Obliczenia W' Balance
- hrv.py: Analiza HRV / DFA
- thermal.py: Indeks ciepła HSI
- power.py: NP, strefy mocy, PDC, FRI, Match Burns
- nutrition.py: Spalanie węglowodanów
- metrics.py: Podstawowe metryki treningowe
- stamina.py: Stamina Score, VLamax estimation
- data_processing.py: Przetwarzanie danych

Dla wstecznej kompatybilności, wszystkie funkcje są re-eksportowane z tego modułu.
"""

# ============================================================
# Re-eksport dla wstecznej kompatybilności z istniejącym kodem
# Import: from modules.calculations import calculate_metrics
# nadal działa jak wcześniej
# ============================================================

from .w_prime import (
    calculate_w_prime_balance,
    calculate_w_prime_fast,
    # Recovery Score (NEW)
    calculate_recovery_score,
    get_recovery_recommendation,
    estimate_w_prime_reconstitution,
)

from .hrv import (
    calculate_dynamic_dfa,
)

from .thermal import (
    calculate_heat_strain_index,
)

from .power import (
    calculate_normalized_power,
    calculate_pulse_power_stats,
    # Advanced power analytics
    calculate_power_duration_curve,
    calculate_fatigue_resistance_index,
    count_match_burns,
    calculate_power_zones_time,
    get_fri_interpretation,
    DEFAULT_PDC_DURATIONS,
    # TTE & Phenotype (NEW)
    estimate_tte,
    estimate_tte_range,
    classify_phenotype,
    get_phenotype_description,
)

from .nutrition import (
    estimate_carbs_burned,
)

from .metrics import (
    calculate_metrics,
    calculate_advanced_kpi,
    calculate_z2_drift,
    calculate_vo2max,
    calculate_trend,
)

from .data_processing import (
    process_data,
    ensure_pandas,
)

from .stamina import (
    calculate_stamina_score,
    estimate_vlamax_from_pdc,
    get_stamina_interpretation,
    get_vlamax_interpretation,
    calculate_aerobic_contribution,
    # NEW
    calculate_durability_index,
    get_durability_interpretation,
)

from .kinetics import (
    fit_smo2_kinetics,
    get_tau_interpretation,
    calculate_o2_deficit,
    detect_smo2_breakpoints,
)

# Eksport wszystkich symboli dla import *
__all__ = [
    # W' Balance
    'calculate_w_prime_balance',
    'calculate_w_prime_fast',
    # W' Recovery (NEW)
    'calculate_recovery_score',
    'get_recovery_recommendation',
    'estimate_w_prime_reconstitution',
    # HRV
    'calculate_dynamic_dfa',
    # Thermal
    'calculate_heat_strain_index',
    # Power - Basic
    'calculate_normalized_power',
    'calculate_pulse_power_stats',
    # Power - Advanced
    'calculate_power_duration_curve',
    'calculate_fatigue_resistance_index',
    'count_match_burns',
    'calculate_power_zones_time',
    'get_fri_interpretation',
    'DEFAULT_PDC_DURATIONS',
    # Power - TTE & Phenotype (NEW)
    'estimate_tte',
    'estimate_tte_range',
    'classify_phenotype',
    'get_phenotype_description',
    # Nutrition
    'estimate_carbs_burned',
    # Metrics
    'calculate_metrics',
    'calculate_advanced_kpi',
    'calculate_z2_drift',
    'calculate_vo2max',
    'calculate_trend',
    # Stamina
    'calculate_stamina_score',
    'estimate_vlamax_from_pdc',
    'get_stamina_interpretation',
    'get_vlamax_interpretation',
    'calculate_aerobic_contribution',
    # Durability (NEW)
    'calculate_durability_index',
    'get_durability_interpretation',
    # Kinetics (NEW)
    'fit_smo2_kinetics',
    'get_tau_interpretation',
    'calculate_o2_deficit',
    'detect_smo2_breakpoints',
    # Data Processing
    'process_data',
    'ensure_pandas',
]
