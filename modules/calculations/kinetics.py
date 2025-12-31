"""
VO2/SmO2 Kinetics Module.

Implements kinetics analysis for oxygen consumption and muscle oxygenation.
Inspired by INSCYD O2 Deficit and academic VO2 kinetics research.
Refactored to support Relative SmO2 analysis (normalization and trend classification).
Includes Context-Aware Analysis (Power/HR/Cadence fusion).
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy.optimize import curve_fit
from scipy import stats


def normalize_smo2_series(series: pd.Series) -> pd.Series:
    """
    Normalize SmO2 series to 0.0 - 1.0 (0-100%) range based on session min/max.
    Uses 5th and 95th percentiles to be robust against artifacts/outliers.
    
    Args:
        series: Pandas Series with raw SmO2 values
        
    Returns:
        Normalized Pandas Series (0.0 to 1.0)
    """
    if series.empty:
        return series
        
    # Robust min/max
    val_min = series.quantile(0.02) # 2nd percentile as physiological min
    val_max = series.quantile(0.98) # 98th percentile as physiological max
    
    if val_max <= val_min:
        return series.apply(lambda x: 0.5) # Flat line if no range
        
    normalized = (series - val_min) / (val_max - val_min)
    return normalized.clip(0.0, 1.0)


def detect_smo2_trend(
    time_series: pd.Series, 
    smo2_series: pd.Series
) -> Dict[str, any]:
    """
    Detect SmO2 trend (Slope) and classify kinetics state.
    
    Args:
        time_series: Time in seconds
        smo2_series: SmO2 values (can be raw or normalized)
        
    Returns:
        Dictionary with slope, interpretation, and category.
    """
    if len(time_series) < 10:
        return {
            "slope": 0.0,
            "category": "Insufficient Data",
            "description": "Too few data points"
        }
        
    slope, _, _, _, std_err = stats.linregress(time_series, smo2_series)
    
    # Classification thresholds (assuming %/s for raw, or unit/s for norm)
    # If raw SmO2 (0-100), slope is %/s.
    
    category = "Stable"
    description = "Equilibrium (Steady State)"
    
    if slope < -0.05:
        category = "Rapid Deoxygenation"
        description = "High Intensity (>> Critical Power). Very fast O2 extraction."
    elif slope < -0.01:
        category = "Deoxygenation"
        description = "Non-Steady State (> Critical Power). O2 debt accumulating."
    elif slope <= 0.01:
        category = "Equilibrium"
        description = "Steady State. Supply matches Demand."
    elif slope > 0.05:
        category = "Rapid Reoxygenation"
        description = "Recovery / Reperfusion."
    else: # > 0.01
        category = "Reoxygenation"
        description = "Supply exceeds Demand (Recovery)."
        
    return {
        "slope": slope,
        "std_err": std_err,
        "category": category,
        "description": description
    }


def classify_smo2_context(
    df_window: pd.DataFrame,
    smo2_trend_result: Dict[str, any]
) -> Dict[str, str]:
    """
    Infer the physiological cause of the SmO2 trend using concurrent signals.
    
    Args:
        df_window: DataFrame with 'time', 'watts', 'hr', 'cadence' (optional)
        smo2_trend_result: Result from detect_smo2_trend
        
    Returns:
        Dict with 'cause', 'explanation', 'confidence'
    """
    category = smo2_trend_result.get('category', '')
    slope_smo2 = smo2_trend_result.get('slope', 0)
    
    if "Insufficient" in category:
        return {"cause": "Unknown", "explanation": "Insufficient data"}

    # Calculate trends for other metrics
    time = df_window['time']
    
    # Power Trend
    if 'watts' in df_window.columns:
        slope_watts, _, _, _, _ = stats.linregress(time, df_window['watts'])
    else:
        slope_watts = 0
        
    # HR Trend
    if 'hr' in df_window.columns:
        slope_hr, _, _, _, _ = stats.linregress(time, df_window['hr'])
    else:
        slope_hr = 0
        
    # Cadence Stats
    avg_cadence = df_window['cadence'].mean() if 'cadence' in df_window.columns else 90
    
    # === HEURISTICS ===
    
    # 1. DEOXYGENATION CASES (Slope < -0.01)
    if slope_smo2 < -0.01:
        
        # A. Mechanical Occlusion (Grinding)
        # Low cadence + Stable/Rising Torque implies high muscle tension restricting flow
        if avg_cadence < 65 and avg_cadence > 10: # >10 to ignore coasting
            return {
                "cause": "Mechanical Occlusion",
                "explanation": f"Low cadence ({avg_cadence:.0f} rpm) creates high muscle tension, physically restricting blood flow.",
                "type": "mechanical"
            }
            
        # B. Demand Driven
        # Power is rising significantly
        if slope_watts > 0.5: # Rising by >0.5 W/s
            return {
                "cause": "Demand Driven",
                "explanation": "Normal response to increasing power output.",
                "type": "normal"
            }
            
        # C. Efficiency Loss (Fading)
        # Power is dropping but we are STILL desaturating? Very bad sign.
        if slope_watts < -0.5:
             return {
                "cause": "Efficiency Loss",
                "explanation": "Desaturation continues despite decreasing power. Indicates severe metabolic fatigue or decoupling.",
                "type": "warning"
            }
            
        # D. Delivery Limitation (Supply constrained)
        # Power is steady, but we are desaturating.
        # Check HR. If HR is flat/maxed, it might be cardiac limit.
        if abs(slope_watts) <= 0.5:
             return {
                "cause": "Delivery Limitation",
                "explanation": "Desaturation at constant power. Oxygen supply cannot meet steady demand.",
                "type": "limit"
            }
            
        return {"cause": "Metabolic Stress", "explanation": "General utilization exceeds supply.", "type": "normal"}

    # 2. EQUILIBRIUM / STABLE
    elif abs(slope_smo2) <= 0.01:
        return {
            "cause": "Steady State",
            "explanation": "Oxygen supply matches demand.",
            "type": "success"
        }

    # 3. REOXYGENATION
    else: # > 0.01
        if slope_watts < -0.5:
            return {
                "cause": "Recovery",
                "explanation": "Normal recovery due to reduced load.",
                "type": "success"
            }
        else:
             return {
                "cause": "Overshoot / Priming",
                "explanation": "Supply exceeds demand despite steady/rising load (Warm-up effect).",
                "type": "success"
            }


def _mono_exponential(t: np.ndarray, a: float, tau: float, td: float) -> np.ndarray:
    """Mono-exponential model: y = a * (1 - exp(-(t-td)/tau))"""
    result = np.zeros_like(t, dtype=float)
    mask = t > td
    result[mask] = a * (1 - np.exp(-(t[mask] - td) / tau))
    return result


def fit_smo2_kinetics(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    column: str = 'smo2'
) -> Optional[dict]:
    """Fit mono-exponential model to SmO2 response."""
    if column not in df.columns or 'time' not in df.columns:
        return None
    
    if end_idx <= start_idx or end_idx > len(df):
        return None
    
    # Extract window
    window = df.iloc[start_idx:end_idx].copy()
    
    if len(window) < 30:  # Need minimum data points
        return None
    
    # Normalize time to start from 0
    t = window['time'].values - window['time'].values[0]
    y = window[column].values
    
    # Remove NaNs
    mask = ~np.isnan(y) & ~np.isnan(t)
    t = t[mask]
    y = y[mask]
    
    if len(t) < 30:
        return None
    
    try:
        # Initial guesses
        amplitude_guess = y[-1] - y[0]
        tau_guess = 30.0
        td_guess = 5.0
        
        # Fit the model
        popt, pcov = curve_fit(
            _mono_exponential,
            t, y - y[0],  # Fit delta from baseline
            p0=[amplitude_guess, tau_guess, td_guess],
            bounds=(
                [-100, 1, 0],      # Lower bounds
                [100, 120, 30]     # Upper bounds
            ),
            maxfev=5000
        )
        
        amplitude, tau, td = popt
        
        # Calculate R² for goodness of fit
        y_pred = _mono_exponential(t, amplitude, tau, td) + y[0]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'amplitude': round(amplitude, 2),
            'tau': round(tau, 1),
            'time_delay': round(td, 1),
            'r_squared': round(r_squared, 3),
            'baseline': round(y[0], 1),
            'steady_state': round(y[0] + amplitude, 1)
        }
        
    except (RuntimeError, ValueError):
        return None


def get_tau_interpretation(tau: float) -> str:
    """Get interpretation of SmO2 time constant (tau)."""
    if tau < 15:
        return "⚡ Bardzo szybka kinetyka - doskonały system tlenowy"
    elif tau < 25:
        return "🟢 Szybka kinetyka - dobra wydolność tlenowa"
    elif tau < 40:
        return "🟡 Umiarkowana kinetyka - przeciętna adaptacja"
    elif tau < 60:
        return "🟠 Wolna kinetyka - ograniczona dostawa tlenu"
    else:
        return "🔴 Bardzo wolna kinetyka - wymagana praca nad bazą tlenową"


def calculate_o2_deficit(
    df: pd.DataFrame,
    interval_start: int,
    interval_end: int,
    steady_state_smo2: float
) -> Optional[float]:
    """Calculate oxygen deficit during interval onset."""
    if 'smo2' not in df.columns:
        return None
    
    window = df.iloc[interval_start:interval_end]
    
    if window.empty:
        return None
    
    # O2 deficit = integral of (steady_state - actual) over time
    smo2_values = window['smo2'].values
    
    # Calculate deficit (area where actual < steady state)
    deficit = np.sum(np.maximum(0, steady_state_smo2 - smo2_values))
    
    return round(deficit, 1)


def detect_smo2_breakpoints(
    df: pd.DataFrame,
    window_size: int = 60,
    threshold: float = 0.5
) -> list:
    """Detect breakpoints in SmO2 response."""
    if 'smo2' not in df.columns:
        return []
    
    smo2 = df['smo2'].values
    
    # Calculate smoothed derivative
    smo2_smooth = pd.Series(smo2).rolling(window=10).mean().values
    derivative = np.gradient(smo2_smooth)
    
    # Find points where derivative changes sign significantly
    breakpoints = []
    
    for i in range(window_size, len(derivative) - window_size):
        # Check for significant change in derivative
        before = np.mean(derivative[i-window_size:i])
        after = np.mean(derivative[i:i+window_size])
        
        if abs(after - before) > threshold:
            breakpoints.append(i)
    
    # Merge nearby breakpoints (within 30 seconds)
    if breakpoints:
        merged = [breakpoints[0]]
        for bp in breakpoints[1:]:
            if bp - merged[-1] > 30:
                merged.append(bp)
        return merged
    
    return []
