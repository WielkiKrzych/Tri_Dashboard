"""
Models Module - Data Models and Complex Computations

This module contains data models, dataclasses, and complex ML models.
NO STREAMLIT OR UI DEPENDENCIES ALLOWED.

Sub-modules:
- session: Session data model
- pdc: Power Duration Curve model
- training: Training load models
- results: Calculation result objects
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

# Re-export data classes from modules.calculations
from modules.calculations.thresholds import (
    TransitionZone,
    ThresholdResult,
    StepTestResult,
    HysteresisResult,
    SensitivityResult,
)


# ============================================================
# Calculation Result Dataclasses
# ============================================================

@dataclass
class NormalizedPowerResult:
    """Result of Normalized Power calculation."""
    value: float
    rolling_window_sec: int
    samples_count: int
    data_quality: float  # 0.0-1.0 based on missing/outlier data
    
    def __float__(self) -> float:
        """Allow using result as float for backward compatibility."""
        return self.value


@dataclass
class PulsePowerResult:
    """Result of Pulse Power (cardiac efficiency) calculation."""
    avg_pulse_power: float
    trend_drop_pct: float
    df_filtered: pd.DataFrame
    # Parameters used
    min_watts: float
    min_hr: float
    min_samples_trend: int
    # Quality metrics
    samples_count: int
    data_quality: float


@dataclass
class DFAResult:
    """Result of DFA Alpha-1 (HRV) calculation."""
    df_results: Optional[pd.DataFrame]
    mean_alpha1: Optional[float]
    alpha1_range: Tuple[float, float]  # (min, max)
    # Parameters used
    window_sec: int
    step_sec: int
    min_samples_hrv: int
    # Quality metrics
    samples_count: int
    windows_analyzed: int
    data_quality: float
    error: Optional[str] = None


@dataclass
class RecoveryScoreResult:
    """Result of W' recovery score calculation."""
    score: float
    w_pct: float  # W' percentage
    time_bonus: float
    # Parameters used
    tau_seconds: float
    time_bonus_max: float
    # Interpretation
    recommendation: Tuple[str, str]  # (emoji+label, description)


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
    # Result models
    'NormalizedPowerResult',
    'PulsePowerResult',
    'DFAResult',
    'RecoveryScoreResult',
    # Session models
    'SessionRecord',
    'AthleteProfile',
]

