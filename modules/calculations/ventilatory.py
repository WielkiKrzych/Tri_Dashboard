"""
Ventilatory Threshold Detection (VT1/VT2) — facade module.

Re-exports all public API from focused sub-modules so that existing
callers require no import changes:

- vt_utils   : calculate_slope, detect_vt1_peaks_heuristic
- vt_step    : detect_vt_from_steps
- vt_sliding : detect_vt_transition_zone, run_sensitivity_analysis
- vt_cpet    : detect_vt_cpet, detect_vt_vslope_savgol
"""

from .vt_utils import calculate_slope, detect_vt1_peaks_heuristic
from .vt_step import detect_vt_from_steps
from .vt_sliding import detect_vt_transition_zone, run_sensitivity_analysis
from .vt_cpet import detect_vt_cpet, detect_vt_vslope_savgol

__all__ = [
    "calculate_slope",
    "detect_vt1_peaks_heuristic",
    "detect_vt_from_steps",
    "detect_vt_transition_zone",
    "run_sensitivity_analysis",
    "detect_vt_cpet",
    "detect_vt_vslope_savgol",
]
