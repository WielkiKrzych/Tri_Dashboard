"""
SRP: Moduł odpowiedzialny za analizę HRV i DFA Alpha-1.
"""
from typing import Union, Any, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from numba import jit

from .common import ensure_pandas, MIN_SAMPLES_HRV


@jit(nopython=True)
def _calc_alpha1_numba(rr_values: np.ndarray) -> float:
    """True DFA Alpha-1 calculation for short-term fractal correlation (Rogers et al. methodology)."""
    if len(rr_values) < 20: 
        return np.nan
        
    # Scale range for Alpha-1 (short-range correlations)
    scales = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.float64)
    
    # Standardize and Integrate signal
    y = np.cumsum(rr_values - np.mean(rr_values))
    
    fluctuations = np.zeros(len(scales))
    
    for idx_n, n in enumerate(scales):
        n_int = int(n)
        num_windows = len(y) // n_int
        if num_windows == 0: continue
        
        rms_sum = 0.0
        # Indices for linear fit optimization: 0...n-1
        x_indices = np.arange(n_int).astype(np.float64)
        x_mean = (n_int - 1.0) / 2.0
        x_var = np.sum((x_indices - x_mean)**2)
        
        for i in range(num_windows):
            start = i * n_int
            seg = y[start : start + n_int]
            
            # Fast Linear Regression in Numba
            seg_mean = np.mean(seg)
            slope = np.sum((x_indices - x_mean) * (seg - seg_mean)) / x_var
            intercept = seg_mean - slope * x_mean
            
            # Residual Sum of Squares
            fit = slope * x_indices + intercept
            rms_sum += np.sum((seg - fit)**2)
            
        fluctuations[idx_n] = np.sqrt(rms_sum / (num_windows * n_int))
    
    # Linear regression in log-log space (Alpha = slope)
    log_scales = np.log(scales)
    log_flucts = np.log(fluctuations)
    
    # Filter out invalid logs
    mask = ~(np.isnan(log_flucts) | np.isinf(log_flucts))
    if np.sum(mask) < 4: return np.nan
    
    ls = log_scales[mask]
    lf = log_flucts[mask]
    
    ls_mean = np.mean(ls)
    lf_mean = np.mean(lf)
    alpha = np.sum((ls - ls_mean) * (lf - lf_mean)) / np.sum((ls - ls_mean)**2)
    
    return alpha

@jit(nopython=True)
def _fast_dfa_loop(time_values, rr_values, window_sec, step_sec):
    """Szybka pętla DFA z użyciem Numba - sliding window analysis."""
    n = len(time_values)
    results_time = []
    results_alpha = []
    results_rmssd = []
    results_sdnn = []
    results_mean_rr = []
    
    start_t = time_values[0]
    end_t = time_values[-1]
    
    curr_t = start_t + window_sec
    
    left_idx = 0
    right_idx = 0
    
    while curr_t < end_t:
        while right_idx < n and time_values[right_idx] <= curr_t:
            right_idx += 1
            
        win_start = curr_t - window_sec
        while left_idx < right_idx and time_values[left_idx] < win_start:
            left_idx += 1
            
        window_len = right_idx - left_idx
        
        if window_len >= 30:
            window_rr = rr_values[left_idx:right_idx]
            
            # IQR Outlier Removal
            q25 = np.nanpercentile(window_rr, 25)
            q75 = np.nanpercentile(window_rr, 75)
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            
            clean_rr = window_rr[(window_rr > lower) & (window_rr < upper)]
            
            if len(clean_rr) >= 20:
                # RMSSD
                diffs_sq_sum = 0.0
                for k in range(len(clean_rr) - 1):
                    d = clean_rr[k+1] - clean_rr[k]
                    diffs_sq_sum += d*d
                rmssd = np.sqrt(diffs_sq_sum / (len(clean_rr) - 1))
                
                # SDNN
                sdnn = np.std(clean_rr)
                mean_rr = np.mean(clean_rr)
                
                # Real True Alpha1
                alpha1 = _calc_alpha1_numba(clean_rr)
                
                if not np.isnan(alpha1):
                    # Clip to sensible physiology
                    alpha1 = max(0.2, min(1.8, alpha1))
                    
                    results_time.append(curr_t)
                    results_alpha.append(alpha1)
                    results_rmssd.append(rmssd)
                    results_sdnn.append(sdnn)
                    results_mean_rr.append(mean_rr)
        
        curr_t += step_sec
    return results_time, results_alpha, results_rmssd, results_sdnn, results_mean_rr


@st.cache_data
def calculate_dynamic_dfa(df_pl, window_sec: int = 300, step_sec: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Calculate HRV metrics (RMSSD, SDNN, Alpha-1) in a sliding window.
    Optimized version with Numba.
    
    Args:
        df_pl: DataFrame with RR data
        window_sec: Window size in seconds
        step_sec: Step size in seconds
    
    Returns:
        Tuple of (results DataFrame, error message or None)
    """
    df = ensure_pandas(df_pl)
    
    rr_col = next((c for c in ['rr', 'rr_interval', 'hrv', 'ibi', 'r-r', 'rr_ms'] if c in df.columns), None)
    
    if rr_col is None:
        return None, "Missing R-R/HRV data column"

    rr_data = df[['time', rr_col]].dropna()
    rr_data = rr_data[rr_data[rr_col] > 0]
    
    if len(rr_data) < MIN_SAMPLES_HRV:
        return None, f"Za mało danych R-R ({len(rr_data)} < 100)"

    # Automatic unit detection
    mean_val = rr_data[rr_col].mean()
    if mean_val < 2.0:  # Seconds -> ms
        rr_data[rr_col] = rr_data[rr_col] * 1000
    elif mean_val > 2000:  # Microseconds -> ms
        rr_data[rr_col] = rr_data[rr_col] / 1000

    rr_values = rr_data[rr_col].values.astype(np.float64)
    time_values = rr_data['time'].values.astype(np.float64)

    try:
        r_time, r_alpha, r_rmssd, r_sdnn, r_mean_rr = _fast_dfa_loop(
            time_values, rr_values, float(window_sec), float(step_sec)
        )
        
        if not r_time:
            return None, "Brak wyników (zbyt mało danych w oknach?)"

        results = pd.DataFrame({
            'time': r_time,
            'alpha1': r_alpha,
            'rmssd': r_rmssd,
            'sdnn': r_sdnn,
            'mean_rr': r_mean_rr
        })
        return results, None
        
    except Exception as e:
        return None, f"Błąd obliczeń Numba: {e}"
