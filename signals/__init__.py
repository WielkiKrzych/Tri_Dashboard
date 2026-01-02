"""
Signals Module - Signal Processing

This module contains signal processing and analysis functions.
NO STREAMLIT OR UI DEPENDENCIES ALLOWED.

Sub-modules:
- processing: Signal filtering, smoothing
- metrics: Signal-level metrics (SmO2, etc.)
- kinetics: O2/SmO2 kinetics analysis
"""

# Re-export from modules.calculations for now (gradual migration)
from modules.calculations.kinetics import (
    fit_smo2_kinetics,
    get_tau_interpretation,
    calculate_o2_deficit,
    detect_smo2_breakpoints,
    normalize_smo2_series,
    detect_smo2_trend,
    classify_smo2_context,
    calculate_resaturation_metrics,
    calculate_signal_lag,
    analyze_temporal_sequence,
    detect_physiological_state,
    generate_state_timeline,
)

from modules.calculations.data_processing import (
    process_data,
    ensure_pandas,
)

from modules.calculations.quality import (
    check_signal_quality,
    check_step_test_protocol,
    check_data_suitability,
)

__all__ = [
    # Kinetics
    'fit_smo2_kinetics',
    'get_tau_interpretation',
    'calculate_o2_deficit',
    'detect_smo2_breakpoints',
    'normalize_smo2_series',
    'detect_smo2_trend',
    'classify_smo2_context',
    'calculate_resaturation_metrics',
    'calculate_signal_lag',
    'analyze_temporal_sequence',
    'detect_physiological_state',
    'generate_state_timeline',
    # Processing
    'process_data',
    'ensure_pandas',
    # Quality
    'check_signal_quality',
    'check_step_test_protocol',
    'check_data_suitability',
]
