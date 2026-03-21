"""
SRP: Moduł odpowiedzialny za obliczenia termiczne (Heat Strain Index).
"""
from typing import Union, Any
import pandas as pd
import numpy as np

from .common import ensure_pandas


def calculate_heat_strain_index(
    df_pl: Union[pd.DataFrame, Any],
    resting_hr: float = 0.0,
    hr_max: float = 0.0,
    baseline_core_temp: float = 0.0,
    acclimatization_days: int = 0,
) -> pd.DataFrame:
    """Calculate Physiological Strain Index (PSI) based on Moran et al. 1998.

    PSI is a validated composite index (0-10) indicating heat stress:
    - 0-3: Low strain (no concern)
    - 4-6: Moderate strain (monitor)
    - 7-8: High strain (consider stopping)
    - 9-10: Very high / dangerous

    Reference: Moran DS, Shitzer A, Pandolf KB (1998).
    "A physiological strain index to evaluate heat stress."
    Am J Physiol 275: R129-R134.

    Extended with adaptive PSI (aPSI) per Buller et al. (2023).
    "Individualized monitoring of heat illness risk." Physiol Measurement 44(10).
    Acclimatized athletes (10+ days) show reduced effective strain.

    Formula:
        PSI = 5 × (Tcore_t - Tcore_0) / (39.5 - Tcore_0)
            + 5 × (HR_t - HR_0) / (HRmax - HR_0)

    Args:
        df_pl: DataFrame with 'core_temperature_smooth' and 'heartrate_smooth'
        resting_hr: Resting heart rate [bpm]. If 0, estimated from first 60s.
        hr_max: Maximum heart rate [bpm]. If 0, estimated from data max.
        baseline_core_temp: Baseline core temperature [°C]. If 0, estimated from first 60s.
        acclimatization_days: Days of heat acclimatization. 10+ days enables aPSI correction.

    Returns:
        DataFrame with added 'hsi' column (PSI values 0-10)
    """
    df = ensure_pandas(df_pl)
    core_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df.columns else None

    if not core_col or 'heartrate_smooth' not in df.columns:
        df['hsi'] = None
        return df

    # Estimate baselines from initial data if not provided
    warmup_samples = min(60, len(df) // 4)
    if warmup_samples < 5:
        warmup_samples = 5

    tcore_0 = baseline_core_temp if baseline_core_temp > 35.0 else float(
        df[core_col].iloc[:warmup_samples].median()
    )
    hr_0 = resting_hr if resting_hr > 30 else float(
        df['heartrate_smooth'].iloc[:warmup_samples].quantile(0.10)
    )
    hr_max_val = hr_max if hr_max > 100 else float(df['heartrate_smooth'].max())

    # Guard against division by zero
    temp_denom = max(0.5, 39.5 - tcore_0)
    hr_denom = max(10.0, hr_max_val - hr_0)

    # PSI = 5 × ΔTcore / (39.5 - Tcore_0) + 5 × ΔHR / (HRmax - HR_0)
    temp_component = 5.0 * (df[core_col] - tcore_0) / temp_denom
    hr_component = 5.0 * (df['heartrate_smooth'] - hr_0) / hr_denom

    df['hsi'] = (temp_component + hr_component).clip(0.0, 10.0)

    # Adaptive PSI correction (Buller et al. 2023)
    # Acclimatized athletes tolerate higher strain before performance decrement
    # aPSI adjusts thresholds upward after 10+ days of heat acclimatization
    if acclimatization_days >= 10:
        # Acclimatized: effective PSI is reduced (better tolerance)
        # Buller 2023: dynamically adjusts for Tcore-to-Tsk gradient
        acclim_factor = min(0.85, 1.0 - (acclimatization_days - 10) * 0.01)
        df['hsi'] = (df['hsi'] * acclim_factor).clip(0.0, 10.0)
        df['hsi_acclimated'] = True
    else:
        df['hsi_acclimated'] = False

    return df


def calculate_thermal_decay(df_pl: Union[pd.DataFrame, Any]) -> dict:
    """Calculate the thermal cost of performance as % efficiency loss per 1°C.
    
    Standard WKO5/INSCYD approach:
    - Efficiency Factor (EF) = Power / Heart Rate
    - Decay = Percentage change in EF for every +1°C of Core Temperature.
    
    Returns:
        dict: {
            'decay_pct_per_c': float, # e.g. -5.2 means 5.2% drop per 1°C
            'r_squared': float,       # statistical confidence
            'is_significant': bool,
            'message': str
        }
    """
    df = ensure_pandas(df_pl)
    
    # Column mapping (finding best candidates)
    pwr_col = next((c for c in ['watts_smooth', 'watts', 'power'] if c in df.columns), None)
    hr_col = next((c for c in ['heartrate_smooth', 'heartrate', 'hr'] if c in df.columns), None)
    temp_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df.columns else None
    
    if not all([pwr_col, hr_col, temp_col]):
        return {'decay_pct_per_c': 0, 'r_squared': 0, 'is_significant': False, 'message': "Brak wymaganych kolumn (Moc, HR, Temp)"}

    # Filter for active state: Power > 100W (or 50% FTP), HR > 100, Temp > 37.2
    # We focus on the "Heat Load" phase where temperature is significantly above baseline
    mask = (df[pwr_col] > 50) & (df[hr_col] > 80) & (df[temp_col] > 37.0)
    df_act = df[mask].copy()
    
    if len(df_act) < 300: # Min 5 minutes of data
        return {'decay_pct_per_c': 0, 'r_squared': 0, 'is_significant': False, 'message': "Zbyt mało danych aktywnych (>37°C)"}
        
    # Efficiency Factor
    df_act['ef'] = df_act[pwr_col] / df_act[hr_col]
    
    # Linear Regression (EF ~ Temp)
    # y = m*x + b
    x = df_act[temp_col].values
    y = df_act['ef'].values
    
    # Basic OLS
    n = len(x)
    m = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    b = (np.mean(y)) - m * np.mean(x)
    
    # R-squared
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate % decay per 1 degree
    # Decay = (Slope / Mean EF at 37.5°C) * 100
    ref_ef = m * 37.5 + b
    if ref_ef <= 0: ref_ef = np.mean(y)
    
    decay_pct = (m / ref_ef) * 100
    
    return {
        'decay_pct_per_c': round(decay_pct, 2),
        'r_squared': round(r2, 3),
        'is_significant': r2 > 0.3 and decay_pct < 0,
        'message': f"Spadek o {abs(decay_pct):.1f}% na każdy +1°C" if decay_pct < 0 else "Stabilna termoregulacja"
    }


def predict_thermal_performance(
    cp: float,
    ftp: float,
    w_prime: float,
    baseline_hr: float,
    target_temp: float,
    decay_pct_per_c: float = -3.0,
    hr_increase_per_c: float = 3.0,
    baseline_temp: float = 37.5,
    resting_hr: float = 0.0,
    max_hr: float = 0.0,
) -> dict:
    """
    Predict performance degradation at a given core temperature.
    
    Based on:
    - dEF/dT: Efficiency Factor decay per degree Celsius
    - dHR/dT: HR increase per degree Celsius
    - HSI: Heat Strain Index (derived from temp and HR)
    
    Args:
        cp: Critical Power [W]
        ftp: Functional Threshold Power [W]
        w_prime: W' anaerobic capacity [kJ]
        baseline_hr: Baseline heart rate at threshold [bpm]
        target_temp: Target core temperature [°C]
        decay_pct_per_c: Efficiency decay % per °C (default -3.0%)
        hr_increase_per_c: HR increase per °C [bpm] (default 3.0)
        baseline_temp: Baseline core temperature [°C] (default 37.5)
        
    Returns:
        dict with predictions: CP, FTP, W' at target temp, HR cost, TTE reduction
    """
    
    # Calculate temperature delta
    temp_delta = max(0, target_temp - baseline_temp)
    
    # === CP/FTP DEGRADATION ===
    # Power drops by decay_pct_per_c for each degree above baseline
    decay_factor = 1 + (decay_pct_per_c / 100) * temp_delta
    decay_factor = max(0.5, min(1.0, decay_factor))  # Clamp to 50-100%
    
    cp_degraded = cp * decay_factor
    ftp_degraded = ftp * decay_factor
    
    # === W' DEGRADATION ===
    # W' degrades slightly faster than CP in heat due to increased glycolytic cost
    # and reduced muscle efficiency. Périard et al. (2011) report 0.8-1.2x multiplier
    # depending on humidity and acclimatization. Using 1.2x as conservative estimate
    # for non-acclimatized athletes.
    w_prime_decay_factor = 1 + (decay_pct_per_c * 1.2 / 100) * temp_delta
    w_prime_decay_factor = max(0.3, min(1.0, w_prime_decay_factor))
    w_prime_degraded = w_prime * w_prime_decay_factor
    
    # === HR COST INCREASE ===
    # HR increases by hr_increase_per_c per degree
    hr_cost_increase = hr_increase_per_c * temp_delta
    hr_at_threshold = baseline_hr + hr_cost_increase
    
    # === TIME TO EXHAUSTION (TTE) REDUCTION ===
    # At threshold power, TTE is theoretically infinite (at CP)
    # But in practice, heat reduces sustainable time
    # Model: TTE_reduction = temp_delta * 5% per degree
    tte_reduction_pct = temp_delta * 5.0  # 5% per degree
    
    # For a 60-minute effort at FTP, calculate reduced time
    base_tte_min = 60.0
    tte_degraded_min = base_tte_min * (1 - tte_reduction_pct / 100)
    tte_degraded_min = max(20.0, tte_degraded_min)  # Min 20 min
    
    # === PSI ESTIMATE (Moran et al. 1998) ===
    # PSI = 5 × ΔTcore / (39.5 - Tcore_0) + 5 × ΔHR / (HRmax - HR_0)
    tcore_0 = baseline_temp
    # Resting HR for trained cyclists: typically 45-65 bpm
    # baseline_hr is HR at threshold (~150-170 bpm), so we use a population-based
    # estimate rather than subtracting from threshold HR
    hr_0 = resting_hr if resting_hr > 30 else 55.0
    hr_max_est = max_hr if max_hr > 100 else max(baseline_hr + 20, 190)
    temp_denom = max(0.5, 39.5 - tcore_0)
    hr_denom = max(10.0, hr_max_est - hr_0)
    hsi_estimated = min(10.0, max(0.0,
        5.0 * (target_temp - tcore_0) / temp_denom +
        5.0 * (hr_at_threshold - hr_0) / hr_denom
    ))
    
    # === CLASSIFICATION ===
    if temp_delta < 1.0:
        risk_level = "low"
        risk_color = "#27AE60"
        risk_label = "Niskie"
    elif temp_delta < 2.0:
        risk_level = "moderate"
        risk_color = "#F39C12"
        risk_label = "Umiarkowane"
    else:
        risk_level = "high"
        risk_color = "#E74C3C"
        risk_label = "Wysokie"
    
    return {
        "target_temp": target_temp,
        "temp_delta": round(temp_delta, 1),
        "baseline_temp": baseline_temp,
        
        # Power degradation
        "cp_baseline": cp,
        "cp_degraded": round(cp_degraded, 0),
        "cp_loss_pct": round((1 - decay_factor) * 100, 1),
        
        "ftp_baseline": ftp,
        "ftp_degraded": round(ftp_degraded, 0),
        "ftp_loss_pct": round((1 - decay_factor) * 100, 1),
        
        "w_prime_baseline": w_prime,
        "w_prime_degraded": round(w_prime_degraded, 1),
        "w_prime_loss_pct": round((1 - w_prime_decay_factor) * 100, 1),
        
        # HR cost
        "hr_baseline": baseline_hr,
        "hr_at_threshold": round(hr_at_threshold, 0),
        "hr_cost_increase": round(hr_cost_increase, 1),
        
        # TTE
        "tte_baseline_min": base_tte_min,
        "tte_degraded_min": round(tte_degraded_min, 0),
        "tte_reduction_pct": round(tte_reduction_pct, 1),
        
        # HSI
        "hsi_estimated": round(hsi_estimated, 1),
        
        # Risk
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_label": risk_label,
        
        # Decay parameters used
        "decay_pct_per_c": decay_pct_per_c,
        "hr_increase_per_c": hr_increase_per_c
    }
