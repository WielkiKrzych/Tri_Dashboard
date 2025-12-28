"""
VO2/SmO2 Kinetics Module.

Implements kinetics analysis for oxygen consumption and muscle oxygenation.
Inspired by INSCYD O2 Deficit and academic VO2 kinetics research.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.optimize import curve_fit


def _mono_exponential(t: np.ndarray, a: float, tau: float, td: float) -> np.ndarray:
    """Mono-exponential model: y = a * (1 - exp(-(t-td)/tau))
    
    Args:
        t: Time array
        a: Amplitude (steady-state value)
        tau: Time constant
        td: Time delay
    """
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
    """Fit mono-exponential model to SmO2 response.
    
    Analyzes how quickly SmO2 responds to a change in power.
    The time constant (tau) indicates metabolic responsiveness:
    - Fast tau (<20s): Quick oxygen kinetics, good aerobic system
    - Slow tau (>40s): Sluggish response, limited oxygen delivery
    
    Args:
        df: DataFrame with SmO2 and time data
        start_idx: Start index of the analysis window
        end_idx: End index of the analysis window
        column: Column name for SmO2 data
        
    Returns:
        Dict with fitted parameters or None if fitting fails
    """
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
    """Get interpretation of SmO2 time constant (tau).
    
    Args:
        tau: Time constant in seconds
        
    Returns:
        Polish interpretation string
    """
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
    """Calculate oxygen deficit during interval onset.
    
    O2 deficit is the difference between oxygen demand and supply
    during the initial phase of high-intensity exercise.
    
    Args:
        df: DataFrame with SmO2 data
        interval_start: Start index of interval
        interval_end: End index (or point where steady state reached)
        steady_state_smo2: Steady-state SmO2 value
        
    Returns:
        O2 deficit as area under curve (arbitrary units)
    """
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
    """Detect breakpoints in SmO2 response.
    
    Identifies sudden changes in SmO2 that may indicate
    metabolic threshold crossings.
    
    Args:
        df: DataFrame with SmO2 data
        window_size: Rolling window size for derivative
        threshold: Threshold for breakpoint detection
        
    Returns:
        List of indices where breakpoints occur
    """
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
