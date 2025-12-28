"""
SOLID: Single Responsibility Principle - Reorganizacja obliczeń.

Ten pakiet grupuje funkcje obliczeniowe według odpowiedzialności:
- w_prime.py: Obliczenia W' Balance
- hrv.py: Analiza HRV / DFA
- thermal.py: Indeks ciepła HSI
- power.py: NP, strefy mocy
- nutrition.py: Spalanie węglowodanów
- metrics.py: Podstawowe metryki treningowe
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

# Eksport wszystkich symboli dla import *
__all__ = [
    # W' Balance
    'calculate_w_prime_balance',
    'calculate_w_prime_fast',
    # HRV
    'calculate_dynamic_dfa',
    # Thermal
    'calculate_heat_strain_index',
    # Power
    'calculate_normalized_power',
    'calculate_pulse_power_stats',
    # Nutrition
    'estimate_carbs_burned',
    # Metrics
    'calculate_metrics',
    'calculate_advanced_kpi',
    'calculate_z2_drift',
    'calculate_vo2max',
    'calculate_trend',
    # Data Processing
    'process_data',
    'ensure_pandas',
]
