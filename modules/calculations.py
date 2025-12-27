from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import io
import streamlit as st
from numba import jit
from .utils import _serialize_df_to_parquet_bytes
from .constants import (
    MIN_SAMPLES_HRV, MIN_SAMPLES_DFA_WINDOW, MIN_SAMPLES_ACTIVE,
    MIN_SAMPLES_Z2_DRIFT, EFFICIENCY_FACTOR, KCAL_PER_JOULE,
    KCAL_PER_GRAM_CARB, CARB_FRACTION_BELOW_VT1, CARB_FRACTION_VT1_VT2,
    CARB_FRACTION_ABOVE_VT2, DFA_ALPHA_MIN, DFA_ALPHA_MAX,
    MIN_WATTS_ACTIVE, MIN_HR_ACTIVE, MIN_WATTS_DECOUPLING, MIN_HR_DECOUPLING,
    WINDOW_LONG, WINDOW_SHORT
)


def ensure_pandas(df: Union[pd.DataFrame, Any]) -> pd.DataFrame:
    """
    Convert any DataFrame-like object to pandas DataFrame.
    Minimizes unnecessary copying when already a pandas DataFrame.
    
    Args:
        df: Input data (pandas DataFrame, Polars DataFrame, or dict)
        
    Returns:
        pandas DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, 'to_pandas'):
        return df.to_pandas()
    if isinstance(df, dict):
        return pd.DataFrame(df)
    return pd.DataFrame(df)


@jit(nopython=True, fastmath=True)
def calculate_w_prime_fast(watts, time, cp, w_prime_cap):
    n = len(watts)
    w_bal = np.empty(n, dtype=np.float64)
    curr_w = w_prime_cap
    
    # Pre-calculate dt locally to keep it in cache
    prev_time = time[0]
    
    for i in range(n):
        # Calculate dt on the fly to save memory/pass
        if i == 0:
            dt = 1.0
        else:
            dt = time[i] - prev_time
            if dt <= 0: dt = 1.0 # fix artifacts
            prev_time = time[i]
            
        # Differential W' Model
        # dW/dt = CP - P
        delta = (cp - watts[i]) * dt
        
        # Integral
        curr_w += delta
        
        # Boundary conditions
        if curr_w > w_prime_cap:
            curr_w = w_prime_cap
        elif curr_w < 0:
            curr_w = 0.0
            
        w_bal[i] = curr_w
        
    return w_bal

@st.cache_data
def _calculate_w_prime_balance_cached(df_bytes: bytes, cp: float, w_prime: float):
    try:
        # 1. Wczytanie danych z bajtów (tak jak było)
        bio = io.BytesIO(df_bytes)
        try:
            df_pd = pd.read_parquet(bio)
        except Exception:
            bio.seek(0)
            df_pd = pd.read_csv(bio)

        if 'watts' not in df_pd.columns:
            df_pd['w_prime_balance'] = np.nan
            return df_pd

        # 2. Przygotowanie tablic dla Numby (musi dostać czyste tablice numpy)
        watts_arr = df_pd['watts'].to_numpy(dtype=np.float64)
        
        if 'time' in df_pd.columns:
            time_arr = df_pd['time'].to_numpy(dtype=np.float64)
        else:
            # Jak nie ma czasu, zakładamy co 1 sekundę
            time_arr = np.arange(len(watts_arr), dtype=np.float64)

        # 3. Uruchomienie TURBO SILNIKA!
        # Tu dzieje się magia - to trwa milisekundy zamiast sekund
        w_bal = calculate_w_prime_fast(watts_arr, time_arr, float(cp), float(w_prime))

        # 4. Zapisanie wyniku
        df_pd['w_prime_balance'] = w_bal
        return df_pd

    except Exception as e:
        # Awaryjnie zwróć pusty wynik, żeby apka się nie wywaliła
        print(f"Błąd obliczeń W': {e}")
        try:
            bio = io.BytesIO(df_bytes)
            try:
                df_pd = pd.read_parquet(bio)
            except:
                bio.seek(0)
                df_pd = pd.read_csv(bio)
            df_pd['w_prime_balance'] = 0.0
            return df_pd
        except:
            return pd.DataFrame({'w_prime_balance': []})

def calculate_w_prime_balance(_df_pl_active, cp: float, w_prime: float):
    if isinstance(_df_pl_active, dict):
        df_pd = pd.DataFrame(_df_pl_active)
    elif hasattr(_df_pl_active, 'to_pandas'):
        df_pd = _df_pl_active.to_pandas()
    else:
        df_pd = _df_pl_active.copy()
    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd), dtype=float)
    df_bytes = _serialize_df_to_parquet_bytes(df_pd)
    result_df = _calculate_w_prime_balance_cached(df_bytes, float(cp), float(w_prime))
    return result_df

@jit(nopython=True)
def _fast_dfa_loop(time_values, rr_values, window_sec, step_sec):
    n = len(time_values)
    results_time = []
    results_alpha = []
    results_rmssd = []
    results_sdnn = []
    results_mean_rr = []
    
    # Znajdź początek i koniec
    start_t = time_values[0]
    end_t = time_values[-1]
    
    curr_t = start_t + window_sec
    
    # Optymalizacja: Indeksy "przesuwnego okna"
    # Zamiast szukać maską (O(N)), trzymamy wskaźniki left_idx i right_idx (O(N) łącznie)
    left_idx = 0
    right_idx = 0
    
    while curr_t < end_t:
        # Przesuń prawy wskaźnik
        while right_idx < n and time_values[right_idx] <= curr_t:
            right_idx += 1
            
        # Przesuń lewy wskaźnik (okno ma długość window_sec)
        win_start = curr_t - window_sec
        while left_idx < right_idx and time_values[left_idx] < win_start:
            left_idx += 1
            
        # Mamy zakres [left_idx, right_idx)
        window_len = right_idx - left_idx
        
        if window_len >= 30:
            # Kopiujemy wycinek do tablicy (wymagane dla obliczeń w Numba)
            window_rr = rr_values[left_idx:right_idx]
            
            # --- OUTLIER REMOVAL (IQR) ---
            q25 = np.nanpercentile(window_rr, 25)
            q75 = np.nanpercentile(window_rr, 75)
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            
            # Ręczne filtrowanie w pętli (szybsze w Numba niż boolean indexing z nową alokacją)
            # Ale boolean indexing w Numba też jest OK. Zróbmy prosto:
            clean_rr = window_rr[(window_rr > lower) & (window_rr < upper)]
            
            if len(clean_rr) >= 20:
                # --- METRYKI ---
                # RMSSD
                diffs_sq_sum = 0.0
                for k in range(len(clean_rr) - 1):
                    d = clean_rr[k+1] - clean_rr[k]
                    diffs_sq_sum += d*d
                rmssd = np.sqrt(diffs_sq_sum / (len(clean_rr) - 1))
                
                # SDNN
                sdnn = np.std(clean_rr)
                mean_rr = np.mean(clean_rr)
                
                # Pseudo-Alpha1
                if mean_rr > 0:
                    cv = (rmssd / mean_rr) * 100
                    alpha1 = 0.4 + (cv / 15.0)
                    if alpha1 < 0.3: alpha1 = 0.3
                    if alpha1 > 1.5: alpha1 = 1.5
                else:
                    alpha1 = 0.5
                
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
    Calculate HRV metrics (RMSSD, SDNN) in a sliding window.
    Optimized version with Numba.
    """
    df = ensure_pandas(df_pl)
    
    rr_col = next((c for c in ['rr', 'rr_interval', 'hrv', 'ibi', 'r-r', 'rr_ms'] if c in df.columns), None)
    
    if rr_col is None:
        return None, "Missing R-R/HRV data column"

    rr_data = df[['time', rr_col]].dropna()
    rr_data = rr_data[rr_data[rr_col] > 0]
    
    if len(rr_data) < MIN_SAMPLES_HRV:
        return None, f"Za mało danych R-R ({len(rr_data)} < 100)"

    # Automatyczna detekcja jednostek
    mean_val = rr_data[rr_col].mean()
    if mean_val < 2.0:  # Sekundy -> ms
        rr_data[rr_col] = rr_data[rr_col] * 1000
    elif mean_val > 2000:  # Mikrosekundy -> ms
        rr_data[rr_col] = rr_data[rr_col] / 1000

    # Konwersja do float64 dla Numby
    rr_values = rr_data[rr_col].values.astype(np.float64)
    time_values = rr_data['time'].values.astype(np.float64)

    # --- URUCHOMIENIE NUMBA ENGINE ---
    try:
        r_time, r_alpha, r_rmssd, r_sdnn, r_mean_rr = _fast_dfa_loop(time_values, rr_values, float(window_sec), float(step_sec))
        
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

def calculate_advanced_kpi(df_pl: Union[pd.DataFrame, Any]) -> Tuple[float, float]:
    """Calculate decoupling percentage and efficiency factor."""
    df = ensure_pandas(df_pl)
    if 'watts_smooth' not in df.columns or 'heartrate_smooth' not in df.columns:
        return 0.0, 0.0
    df_active = df[(df['watts_smooth'] > MIN_WATTS_DECOUPLING) & (df['heartrate_smooth'] > MIN_HR_DECOUPLING)]
    if len(df_active) < MIN_SAMPLES_ACTIVE: 
        return 0.0, 0.0
    mid = len(df_active) // 2
    p1, p2 = df_active.iloc[:mid], df_active.iloc[mid:]
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: 
        return 0.0, 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    if ef1 == 0: 
        return 0.0, 0.0
    return ((ef1 - ef2) / ef1) * 100, (df_active['watts_smooth'] / df_active['heartrate_smooth']).mean()

def calculate_z2_drift(df_pl: Union[pd.DataFrame, Any], cp: float) -> float:
    """Calculate cardiac drift in Zone 2."""
    df = ensure_pandas(df_pl)
    if 'watts_smooth' not in df.columns or 'heartrate_smooth' not in df.columns:
        return 0.0
    # Z2 is 55-75% of CP
    df_z2 = df[(df['watts_smooth'] >= 0.55*cp) & (df['watts_smooth'] <= 0.75*cp) & (df['heartrate_smooth'] > 60)]
    if len(df_z2) < MIN_SAMPLES_Z2_DRIFT: 
        return 0.0
    mid = len(df_z2) // 2
    p1, p2 = df_z2.iloc[:mid], df_z2.iloc[mid:]
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: 
        return 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    return ((ef1 - ef2) / ef1) * 100 if ef1 != 0 else 0.0

def calculate_heat_strain_index(df_pl: Union[pd.DataFrame, Any]) -> pd.DataFrame:
    """Calculate Heat Strain Index (HSI) based on core temp and HR."""
    df = ensure_pandas(df_pl)
    core_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df.columns else None
    if not core_col or 'heartrate_smooth' not in df.columns:
        df['hsi'] = None
        return df
    df['hsi'] = ((5 * (df[core_col] - 37.0) / 2.5) + (5 * (df['heartrate_smooth'] - 60.0) / 120.0)).clip(0.0, 10.0)
    return df

def calculate_vo2max(mmp_5m, rider_weight):
    if mmp_5m is None or pd.isna(mmp_5m) or rider_weight <= 0: return 0.0
    return (10.8 * mmp_5m / rider_weight) + 7

def calculate_trend(x, y):
    try:
        idx = np.isfinite(x) & np.isfinite(y)
        if np.sum(idx) < 2: return None
        z = np.polyfit(x[idx], y[idx], 1)
        p = np.poly1d(z)
        return p(x)
    except: return None

def process_data(df: Union[pd.DataFrame, Any]) -> pd.DataFrame:
    """Process raw data: resample, smooth, and add time columns."""
    df_pd = ensure_pandas(df)

    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)
    df_pd['time'] = pd.to_numeric(df_pd['time'], errors='coerce')
    
    # Usuń wiersze z NaN w kolumnie time przed utworzeniem indeksu
    df_pd = df_pd.dropna(subset=['time'])
    
    # Wypełnij brakujące wartości time sekwencyjnie jeśli są duplikaty lub luki
    if df_pd['time'].isna().any() or len(df_pd) == 0:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)

    df_pd = df_pd.sort_values('time').reset_index(drop=True)
    df_pd['time_dt'] = pd.to_timedelta(df_pd['time'], unit='s')
    
    # Upewnij się, że indeks nie ma NaN
    df_pd = df_pd[df_pd['time_dt'].notna()]
    df_pd = df_pd.set_index('time_dt')

    num_cols = df_pd.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        # Użyj metody 'linear' zamiast 'time' dla większej niezawodności
        df_pd[num_cols] = df_pd[num_cols].interpolate(method='linear').ffill().bfill()

    try:
        df_numeric = df_pd.select_dtypes(include=[np.number])
        df_resampled = df_numeric.resample('1S').mean()
        df_resampled = df_resampled.interpolate(method='linear').ffill().bfill()
    except Exception:
        df_resampled = df_pd
    df_resampled['time'] = df_resampled.index.total_seconds()
    df_resampled['time_min'] = df_resampled['time'] / 60.0

    window_long = WINDOW_LONG
    window_short = WINDOW_SHORT
    smooth_cols = ['watts', 'heartrate', 'cadence', 'smo2', 'torque', 'core_temperature',
                   'skin_temperature', 'velocity_smooth', 'tymebreathrate', 'tymeventilation', 'thb']
    
    for col in smooth_cols:
        if col in df_resampled.columns:
            df_resampled[f'{col}_smooth'] = df_resampled[col].rolling(window=window_long, min_periods=1).mean()
            df_resampled[f'{col}_smooth_5s'] = df_resampled[col].rolling(window=window_short, min_periods=1).mean()

    df_resampled = df_resampled.reset_index(drop=True)

    return df_resampled

def calculate_metrics(df_pl, cp_val):
    cols = df_pl.columns
    avg_watts = df_pl['watts'].mean() if 'watts' in cols else 0
    avg_hr = df_pl['heartrate'].mean() if 'heartrate' in cols else 0
    avg_cadence = df_pl['cadence'].mean() if 'cadence' in cols else 0
    avg_vent = df_pl['tymeventilation'].mean() if 'tymeventilation' in cols else 0
    avg_rr = df_pl['tymebreathrate'].mean() if 'tymebreathrate' in cols else 0
    power_hr = (avg_watts / avg_hr) if avg_hr > 0 else 0
    np_est = avg_watts * 1.05
    ef_factor = (np_est / avg_hr) if avg_hr > 0 else 0
    work_above_cp_kj = 0.0
    if 'watts' in cols:
        try:
            if hasattr(df_pl, "select"):
                t = df_pl['time'].to_numpy().astype(float)
                w = df_pl['watts'].to_numpy().astype(float)
            else:
                t = df_pl['time'].values.astype(float)
                w = df_pl['watts'].values.astype(float)
            dt = np.diff(t, prepend=t[0])
            if len(dt) > 1:
                dt[0] = dt[1] if dt[1] > 0 else np.median(dt[1:]) if len(dt)>2 else 1.0
            else:
                dt = np.ones_like(w)
            excess = np.maximum(w - cp_val, 0.0)
            energy_j = np.sum(excess * dt)  # w·s = J
            work_above_cp_kj = energy_j / 1000.0
        except Exception:
            df_above_cp = df_pl[df_pl['watts'] > cp_val] if 'watts' in df_pl.columns else pd.DataFrame()
            work_above_cp_kj = (df_above_cp['watts'].sum() / 1000) if len(df_above_cp)>0 else 0.0

    return {
        "avg_watts": avg_watts,
        "avg_hr": avg_hr,
        "avg_cadence": avg_cadence,
        "avg_vent": avg_vent,
        "avg_rr": avg_rr,
        "power_hr": power_hr,
        "ef_factor": ef_factor,
        "work_above_cp_kj": work_above_cp_kj
    }

def calculate_normalized_power(df_pl: Union[pd.DataFrame, Any]) -> float:
    """
    Calculate Normalized Power (NP) using Coggan's formula.
    Requires 'watts' or 'watts_smooth' column.
    """
    df = ensure_pandas(df_pl)
    col = 'watts' if 'watts' in df.columns else ('watts_smooth' if 'watts_smooth' in df.columns else None)
    
    if col is None:
        return 0.0
        
    # Rolling 30s avg
    rolling_30s = df[col].rolling(window=30, min_periods=1).mean()
    # 4th power
    rolling_pow4 = np.power(rolling_30s, 4)
    # Mean
    avg_pow4 = np.mean(rolling_pow4)
    # 4th root
    np_val = np.power(avg_pow4, 0.25)
    
    if pd.isna(np_val):
        return df[col].mean()
        
    return np_val

def estimate_carbs_burned(df_pl: Union[pd.DataFrame, Any], vt1_watts: float, vt2_watts: float) -> float:
    """
    Estimate carbohydrate consumption based on power zones.
    Assumption: 22% mechanical efficiency.
    """
    df = ensure_pandas(df_pl)
    if 'watts' not in df.columns:
        return 0.0
        
    # Energy per second (kcal/s)
    # Power (W = J/s). Efficiency ~22% -> Total Energy = Power / efficiency
    energy_kcal_sec = (df['watts'] / EFFICIENCY_FACTOR) * KCAL_PER_JOULE
    
    # Carb fraction by zone
    conditions = [
        (df['watts'] < vt1_watts),
        (df['watts'] >= vt1_watts) & (df['watts'] < vt2_watts),
        (df['watts'] >= vt2_watts)
    ]
    choices = [CARB_FRACTION_BELOW_VT1, CARB_FRACTION_VT1_VT2, CARB_FRACTION_ABOVE_VT2]
    carb_fraction = np.select(conditions, choices, default=1.0)
    
    # 1 g carbs = 4 kcal
    carbs_burned_sec = (energy_kcal_sec * carb_fraction) / KCAL_PER_GRAM_CARB
    
    return carbs_burned_sec.sum()

def calculate_pulse_power_stats(df_pl: Union[pd.DataFrame, Any]) -> Tuple[float, float, pd.DataFrame]:
    """Calculate Pulse Power (Efficiency) statistics: Avg PP, Trend Drop %."""
    df = ensure_pandas(df_pl)
    
    col_w = 'watts_smooth' if 'watts_smooth' in df.columns else 'watts'
    col_hr = 'heartrate_smooth' if 'heartrate_smooth' in df.columns else 'heartrate'
    
    if col_w not in df.columns or col_hr not in df.columns:
        return 0.0, 0.0, pd.DataFrame() # Avg, Drop, DF_PP
        
    # Filtrujemy sensowne wartości
    mask = (df[col_w] > 50) & (df[col_hr] > 90)
    df_pp = df[mask].copy()
    
    if df_pp.empty:
        return 0.0, 0.0, pd.DataFrame()
        
    df_pp['pulse_power'] = df_pp[col_w] / df_pp[col_hr]
    avg_pp = df_pp['pulse_power'].mean()
    
    # Trend
    if len(df_pp) > 100:
        x = df_pp['time'] if 'time' in df_pp.columns else np.arange(len(df_pp))
        y = df_pp['pulse_power'].values
        idx = np.isfinite(x) & np.isfinite(y)
        if np.sum(idx) > 10:
             z = np.polyfit(x[idx], y[idx], 1)
             slope = z[0]
             intercept = z[1]
             # Total drop in % over the session
             start_val = intercept + slope * x.iloc[0]
             end_val = intercept + slope * x.iloc[-1]
             
             if start_val != 0:
                 drop_pct = (end_val - start_val) / start_val * 100
             else:
                 drop_pct = 0.0
        else:
            drop_pct = 0.0
    else:
        drop_pct = 0.0
        
    return avg_pp, drop_pct, df_pp
