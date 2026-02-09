"""
SmO2 Types Module

Data classes for SmO2 analysis.
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SmO2AdvancedMetrics:
    """
    Immutable container for advanced SmO2 metrics.
    
    Attributes:
        slope_per_100w: SmO2 drop per 100W [%/100W]
        halftime_reoxy_sec: Half-time to reoxygenation [s]
        hr_coupling_r: Correlation SmO2 vs HR changes
        drift_pct: SmO2 drift first half vs second half [%]
        limiter_type: local, central, metabolic, balanced, unknown
        limiter_confidence: 0-1 confidence score
        interpretation: Coach-oriented interpretation text
        recommendations: List of training recommendations
        slope_r2: R-squared of slope calculation
        data_quality: good, low, no_smo2, no_power
    """
    slope_per_100w: float = 0.0
    halftime_reoxy_sec: Optional[float] = None
    hr_coupling_r: float = 0.0
    drift_pct: float = 0.0
    limiter_type: str = "unknown"
    limiter_confidence: float = 0.0
    interpretation: str = ""
    recommendations: Tuple[str, ...] = field(default_factory=tuple)
    slope_r2: float = 0.0
    data_quality: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slope_per_100w": round(self.slope_per_100w, 2),
            "halftime_reoxy_sec": round(self.halftime_reoxy_sec, 1) if self.halftime_reoxy_sec else None,
            "hr_coupling_r": round(self.hr_coupling_r, 3),
            "drift_pct": round(self.drift_pct, 2),
            "limiter_type": self.limiter_type,
            "limiter_confidence": round(self.limiter_confidence, 2),
            "interpretation": self.interpretation,
            "recommendations": list(self.recommendations),
            "data_quality": self.data_quality
        }


@dataclass
class SmO2ThresholdResult:
    """
    Result of 3-point SmO2 threshold detection.
    
    Detects T1 (LT1 analog), T2_onset (RCP analog), and optionally T2_steady.
    """
    # T1 (LT1 analog - onset of desaturation)
    t1_watts: Optional[int] = None
    t1_hr: Optional[int] = None
    t1_smo2: Optional[float] = None
    t1_gradient: Optional[float] = None
    t1_trend: Optional[float] = None
    t1_sd: Optional[float] = None
    t1_step: Optional[int] = None
    
    # T2_onset (Heavy→Severe transition)
    t2_onset_watts: Optional[int] = None
    t2_onset_hr: Optional[int] = None
    t2_onset_smo2: Optional[float] = None
    t2_onset_gradient: Optional[float] = None
    t2_onset_curvature: Optional[float] = None
    t2_onset_sd: Optional[float] = None
    t2_onset_step: Optional[int] = None
    
    # T2_steady (MLSS_local / RCP_steady analog)
    t2_steady_watts: Optional[int] = None
    t2_steady_hr: Optional[int] = None
    t2_steady_smo2: Optional[float] = None
    t2_steady_gradient: Optional[float] = None
    t2_steady_trend: Optional[float] = None
    t2_steady_sd: Optional[float] = None
    t2_steady_step: Optional[int] = None
    
    # Legacy compatibility
    t2_watts: Optional[int] = None
    t2_hr: Optional[int] = None
    t2_smo2: Optional[float] = None
    t2_gradient: Optional[float] = None
    t2_step: Optional[int] = None
    
    # Zones and validation
    zones: List[Dict] = field(default_factory=list)
    vt1_correlation_watts: Optional[int] = None
    rcp_onset_correlation_watts: Optional[int] = None
    rcp_steady_correlation_watts: Optional[int] = None
    physiological_agreement: str = "not_checked"
    analysis_notes: List[str] = field(default_factory=list)
    method: str = "moxy_3point"
    step_data: List[Dict] = field(default_factory=list)
