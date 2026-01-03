"""
Domain Module.

Contains domain-level concepts and value objects.
"""
from .session_type import (
    SessionType, 
    classify_session_type,
    RampClassificationResult,
    classify_ramp_test,
)

__all__ = [
    "SessionType",
    "classify_session_type",
    "RampClassificationResult",
    "classify_ramp_test",
]
