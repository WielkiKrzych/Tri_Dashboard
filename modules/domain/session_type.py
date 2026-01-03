"""
Session Type Domain Module.

Defines the SessionType enum and classification logic.
Every CSV analysis MUST go through session type classification
before any physiological pipeline runs.
"""
from enum import Enum, auto
from typing import Optional
import pandas as pd
import numpy as np


class SessionType(Enum):
    """Domain enum for classifying training session types."""
    RAMP_TEST = auto()
    TRAINING = auto()
    UNKNOWN = auto()
    
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
    
    @property
    def emoji(self) -> str:
        """Return an emoji representation of the session type."""
        return {
            SessionType.RAMP_TEST: "📈",
            SessionType.TRAINING: "🚴",
            SessionType.UNKNOWN: "❓",
        }.get(self, "❓")


def classify_session_type(
    df: pd.DataFrame,
    filename: str = ""
) -> SessionType:
    """Classify the session type based on filename and data patterns.
    
    This function MUST be called before any physiological pipeline runs.
    
    Args:
        df: Raw DataFrame loaded from CSV
        filename: Original filename (used for heuristic matching)
        
    Returns:
        SessionType enum value
        
    Classification Rules:
        1. RAMP_TEST: Filename contains 'ramp' OR power shows step pattern
        2. TRAINING: Default for typical ride files
        3. UNKNOWN: Fallback when classification is uncertain
    """
    if df is None or df.empty:
        return SessionType.UNKNOWN
    
    filename_lower = filename.lower() if filename else ""
    
    # Rule 1: Filename-based classification
    if "ramp" in filename_lower:
        return SessionType.RAMP_TEST
    
    # Rule 2: Power pattern analysis for ramp test detection
    if "watts" in df.columns or "power" in df.columns:
        power_col = "watts" if "watts" in df.columns else "power"
        power = df[power_col].dropna()
        
        if len(power) >= 60:  # At least 1 minute of data
            if _detect_ramp_pattern(power):
                return SessionType.RAMP_TEST
    
    # Rule 3: If we have valid power data, it's likely a training session
    if "watts" in df.columns or "power" in df.columns:
        power_col = "watts" if "watts" in df.columns else "power"
        valid_power = df[power_col].dropna()
        if len(valid_power) > 0 and valid_power.mean() > 0:
            return SessionType.TRAINING
    
    # Fallback
    return SessionType.UNKNOWN


def _detect_ramp_pattern(power: pd.Series, min_steps: int = 3) -> bool:
    """Detect if power data shows a ramp/step test pattern.
    
    A ramp test typically shows:
    - Multiple distinct power plateaus
    - Consistently increasing power steps
    - Step duration of ~1-5 minutes
    
    Args:
        power: Power series (1Hz assumed)
        min_steps: Minimum number of steps to qualify as ramp
        
    Returns:
        True if ramp pattern detected
    """
    if len(power) < 180:  # Need at least 3 minutes
        return False
    
    # Smooth power to reduce noise
    window = min(30, len(power) // 10)
    if window < 5:
        return False
        
    smoothed = power.rolling(window=window, center=True).mean().dropna()
    
    if len(smoothed) < 60:
        return False
    
    # Divide into chunks and check for step pattern
    chunk_size = 60  # 1-minute chunks
    chunks = [smoothed.iloc[i:i+chunk_size].mean() 
              for i in range(0, len(smoothed) - chunk_size, chunk_size)]
    
    if len(chunks) < min_steps:
        return False
    
    # Check for consistent increases
    increases = 0
    for i in range(1, len(chunks)):
        if chunks[i] > chunks[i-1] + 5:  # At least 5W increase
            increases += 1
    
    # If most chunks show increases, it's likely a ramp
    increase_ratio = increases / (len(chunks) - 1) if len(chunks) > 1 else 0
    
    return increase_ratio >= 0.6  # 60% of steps should be increasing
