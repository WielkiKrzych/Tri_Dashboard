"""
Models Module - Data Models and Complex Computations

This module contains data models, dataclasses, and complex ML models.
NO STREAMLIT OR UI DEPENDENCIES ALLOWED.

Sub-modules:
- session: Session data model
- pdc: Power Duration Curve model
- training: Training load models
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Re-export data classes from modules.calculations
from modules.calculations.thresholds import (
    TransitionZone,
    ThresholdResult,
    StepTestResult,
    HysteresisResult,
    SensitivityResult,
)


@dataclass
class SessionRecord:
    """Session data record for storage."""
    filename: str
    timestamp: str
    duration_sec: int
    avg_power: float
    np_power: float
    tss: float
    intensity_factor: float
    avg_hr: Optional[float] = None
    avg_smo2: Optional[float] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass  
class AthleteProfile:
    """Athlete profile with physiological parameters."""
    weight_kg: float
    ftp: float
    max_hr: int
    lthr: Optional[int] = None
    vo2max: Optional[float] = None
    w_prime: Optional[float] = None


__all__ = [
    # Threshold models
    'TransitionZone',
    'ThresholdResult',
    'StepTestResult',
    'HysteresisResult',
    'SensitivityResult',
    # Session models
    'SessionRecord',
    'AthleteProfile',
]
