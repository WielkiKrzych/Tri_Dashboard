"""
SmO2 Metrics Calculator Module

Pure functions for SmO2 metric calculations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


class SmO2MetricsCalculator:
    """Calculator for SmO2 metrics."""
    
    @staticmethod
    def calculate_smo2_slope(
        df: pd.DataFrame,
        smo2_col: str = "SmO2",
        power_col: str = "watts",
        min_power: float = 100.0
    ) -> Tuple[float, float]:
        """
        Calculate SmO2 slope per 100W of power increase.
        
        Returns:
            Tuple of (slope_per_100w, r_squared)
        """
        if smo2_col not in df.columns or power_col not in df.columns:
            return 0.0, 0.0
            
        mask = df[power_col] >= min_power
        if mask.sum() < 10:
            return 0.0, 0.0
        
        filtered = df.loc[mask, [power_col, smo2_col]].dropna()
        if len(filtered) < 10:
            return 0.0, 0.0
        
        slope, intercept, r, p, se = stats.linregress(
            filtered[power_col], 
            filtered[smo2_col]
        )
        
        return slope * 100, r ** 2
    
    @staticmethod
    def calculate_halftime_reoxygenation(
        df: pd.DataFrame,
        smo2_col: str = "SmO2",
        time_col: str = "seconds",
        power_col: str = "watts"
    ) -> Optional[float]:
        """
        Calculate half-time for SmO2 reoxygenation after ramp ends.
        
        Returns:
            Half-time in seconds, or None if not detectable.
        """
        if power_col not in df.columns or smo2_col not in df.columns:
            return None
        
        power = df[power_col].values
        smo2 = df[smo2_col].values
        
        if time_col in df.columns:
            time = df[time_col].values
        elif "time" in df.columns:
            try:
                time = pd.to_timedelta(df["time"]).dt.total_seconds().values
            except:
                time = np.arange(len(df))
        else:
            time = np.arange(len(df))
        
        peak_idx = np.argmax(power)
        if peak_idx >= len(power) - 30 or peak_idx == 0:
            return None
        
        smo2_at_peak = smo2[peak_idx]
        smo2_min_during_ramp = np.min(smo2[:peak_idx])
        
        recovery_smo2 = smo2[peak_idx:]
        recovery_time = time[peak_idx:]
        
        if len(recovery_smo2) < 30:
            return None
        
        drop = smo2[0] - smo2_min_during_ramp
        if drop <= 0:
            return None
        
        half_recovery_target = smo2_min_during_ramp + (drop * 0.5)
        
        # Vectorized search
        above_half = recovery_smo2 >= half_recovery_target
        if not np.any(above_half):
            return None
        
        first_idx = np.argmax(above_half)
        return float(recovery_time[first_idx] - recovery_time[0])
    
    @staticmethod
    def calculate_hr_coupling_index(
        df: pd.DataFrame,
        smo2_col: str = "SmO2",
        hr_col: str = "hr",
        window: int = 30
    ) -> float:
        """
        Calculate coupling index between SmO2 and HR.
        
        Returns:
            Pearson correlation coefficient (-1 to 1)
        """
        if hr_col not in df.columns or smo2_col not in df.columns:
            return 0.0
        
        smo2_smooth = df[smo2_col].rolling(window=window, min_periods=1).mean()
        hr_smooth = df[hr_col].rolling(window=window, min_periods=1).mean()
        
        valid = pd.DataFrame({"smo2": smo2_smooth, "hr": hr_smooth}).dropna()
        if len(valid) < 30:
            return 0.0
        
        r, p = stats.pearsonr(valid["smo2"], valid["hr"])
        return float(r)
    
    @staticmethod
    def calculate_smo2_drift(
        df: pd.DataFrame,
        smo2_col: str = "SmO2",
        power_col: str = "watts",
        min_power: float = 100.0
    ) -> float:
        """
        Calculate SmO2 drift as percentage change.
        
        Returns:
            Drift percentage (e.g., -7.5 means 7.5% drop)
        """
        if smo2_col not in df.columns or power_col not in df.columns:
            return 0.0
        
        mask = df[power_col] > min_power
        if mask.sum() < 20:
            return 0.0
        
        filtered = df.loc[mask, smo2_col].dropna()
        if len(filtered) < 20:
            return 0.0
        
        n = len(filtered)
        mid = n // 2
        
        first_half_avg = filtered.iloc[:mid].mean()
        second_half_avg = filtered.iloc[mid:].mean()
        
        if first_half_avg == 0:
            return 0.0
        
        return float((second_half_avg - first_half_avg) / first_half_avg * 100)
