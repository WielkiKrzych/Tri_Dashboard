"""
Domain Module.

Contains domain-level concepts and value objects.
"""
from .session_type import (
    SessionType,
    classify_session_type,
    RampClassificationResult,
    classify_ramp_test,
    RAMP_CONFIDENCE_THRESHOLD,
)
from .data_quality import (
    DataQuality,
    quality_band,
    format_band_badge_html,
)
from .threshold_crosscheck import (
    AgreementLevel,
    CrosscheckVerdict,
    crosscheck_threshold,
)

__all__ = [
    "SessionType",
    "classify_session_type",
    "RampClassificationResult",
    "classify_ramp_test",
    "RAMP_CONFIDENCE_THRESHOLD",
    "DataQuality",
    "quality_band",
    "format_band_badge_html",
    "AgreementLevel",
    "CrosscheckVerdict",
    "crosscheck_threshold",
]
