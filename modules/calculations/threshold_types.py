"""
Common types and dataclasses for threshold detection.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class TransitionZone:
    """Represents a transition zone instead of a single point."""
    range_watts: Tuple[float, float]
    range_hr: Optional[Tuple[float, float]]
    confidence: float  # 0.0 to 1.0
    method: str
    description: str = ""

@dataclass
class ThresholdResult:
    """Result of threshold detection (Legacy/Simple)."""
    zone_name: str
    zone_type: str  # info, success, warning, error
    description: str
    slope_value: float
    power_at_threshold: Optional[float] = None
    hr_at_threshold: Optional[float] = None

@dataclass
class HysteresisResult:
    """Result of directional analysis (hysteresis)."""
    vt1_inc_zone: Optional[TransitionZone] = None
    vt1_dec_zone: Optional[TransitionZone] = None
    vt2_inc_zone: Optional[TransitionZone] = None
    vt2_dec_zone: Optional[TransitionZone] = None
    
    vt1_shift_watts: Optional[float] = None
    vt2_shift_watts: Optional[float] = None
    
    warnings: List[str] = field(default_factory=list)

@dataclass
class SensitivityResult:
    """Result of sensitivity/stability analysis."""
    vt1_stability_score: float = 0.0
    vt2_stability_score: float = 0.0
    vt1_variability_watts: float = 0.0
    vt2_variability_watts: float = 0.0
    is_vt1_unreliable: bool = False
    is_vt2_unreliable: bool = False
    details: List[str] = field(default_factory=list)

@dataclass
class StepTestResult:
    """Complete result of step test analysis."""
    vt1_watts: Optional[float] = None
    vt2_watts: Optional[float] = None
    lt1_watts: Optional[float] = None
    lt2_watts: Optional[float] = None
    vt1_zone: Optional[TransitionZone] = None
    vt2_zone: Optional[TransitionZone] = None
    hysteresis: Optional[HysteresisResult] = None
    sensitivity: Optional[SensitivityResult] = None
    vt1_hr: Optional[float] = None
    vt2_hr: Optional[float] = None
    steps_analyzed: int = 0
    analysis_notes: List[str] = field(default_factory=list)
    step_ve_analysis: List[dict] = field(default_factory=list)
    vt1_ve: Optional[float] = None
    vt1_br: Optional[float] = None
    vt2_ve: Optional[float] = None
    vt2_br: Optional[float] = None
    smo2_1_watts: Optional[float] = None
    smo2_2_watts: Optional[float] = None
    smo2_1_hr: Optional[float] = None
    smo2_2_hr: Optional[float] = None
    step_smo2_analysis: List[dict] = field(default_factory=list)

@dataclass
class DetectedStep:
    """Represents a detected step in the step test."""
    step_number: int
    start_time: float
    end_time: float
    duration_sec: float
    avg_power: float
    power_diff_from_prev: float = 0.0

@dataclass
class StepTestRange:
    """Detected range of a valid step test."""
    start_time: float
    end_time: float
    steps: List[DetectedStep]
    min_power: float
    max_power: float
    is_valid: bool = True
    notes: List[str] = field(default_factory=list)

@dataclass
class StepVTResult:
    """Result of step-by-step VT detection."""
    vt1_watts: Optional[float] = None
    vt1_hr: Optional[float] = None
    vt1_ve: Optional[float] = None
    vt1_br: Optional[float] = None
    vt1_step_number: Optional[int] = None
    vt1_ve_slope: Optional[float] = None
    vt2_watts: Optional[float] = None
    vt2_hr: Optional[float] = None
    vt2_ve: Optional[float] = None
    vt2_br: Optional[float] = None
    vt2_step_number: Optional[int] = None
    vt2_ve_slope: Optional[float] = None
    step_analysis: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class StepSmO2Result:
    """Result of step-by-step SmO2 detection."""
    smo2_1_watts: Optional[float] = None
    smo2_1_hr: Optional[float] = None
    smo2_1_step_number: Optional[int] = None
    smo2_1_slope: Optional[float] = None
    smo2_2_watts: Optional[float] = None
    smo2_2_hr: Optional[float] = None
    smo2_2_step_number: Optional[int] = None
    smo2_2_slope: Optional[float] = None
    step_analysis: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
