import numpy as np
import pandas as pd
import io
import streamlit as st
from numba import jit
from .utils import _serialize_df_to_parquet_bytes

@jit(nopython=True)
def calculate_w_prime_fast(watts, time, cp, w_prime_cap):
    n = len(watts)
    w_bal = np.empty(n, dtype=np.float64)
    curr_w = w_prime_cap
    
    # Obliczamy różnice czasu (dt) wewnątrz Numby dla szybkości
    # Dla pierwszego punktu zakładamy 1 sekundę, dla reszty różnicę
    dt = np.empty(n, dtype=np.float64)
    dt[0] = 1.0 
    for i in range(1, n):
        val = time[i] - time[i-1]
        # Zabezpieczenie przed zerowym czasem lub ujemnym (błędy w pliku)
        if val <= 0:
            dt[i] = 1.0
        else:
            dt[i] = val

    for i in range(n):
        # Logika: CP - Moc = delta. 
        # Jeśli delta > 0 (jedziesz lekko) -> regeneracja.
        # Jeśli delta < 0 (jedziesz mocno) -> spalanie.
        delta = (cp - watts[i]) * dt[i]
        curr_w += delta
        
        # Nie możemy mieć więcej niż 100% baterii
        if curr_w > w_prime_cap:
            curr_w = w_prime_cap
        # Nie możemy mieć mniej niż 0% baterii
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

@st.cache_data
def calculate_dynamic_dfa(df_pl, window_sec=300, step_sec=30):
    """
    Oblicza metryki HRV (RMSSD, SDNN) w oknie przesuwnym.
    Działa z danymi resamplowanymi (1 Hz) i surowymi R-R.
    Zwraca pseudo-DFA bazujący na zmienności HRV.
    """

    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl.copy()
    
    rr_col = next((c for c in ['rr', 'rr_interval', 'hrv', 'ibi', 'r-r', 'rr_ms'] if c in df.columns), None)
    
    if rr_col is None:
        return None, "Brak kolumny z danymi R-R/HRV"

    rr_data = df[['time', rr_col]].dropna()
    rr_data = rr_data[rr_data[rr_col] > 0]
    
    if len(rr_data) < 100:
        return None, f"Za mało danych R-R ({len(rr_data)} < 100)"

    # Automatyczna detekcja jednostek
    mean_val = rr_data[rr_col].mean()
    if mean_val < 2.0:  # Prawdopodobnie sekundy
        rr_data = rr_data.copy()
        rr_data[rr_col] = rr_data[rr_col] * 1000
    elif mean_val > 2000:  # Prawdopodobnie mikrosekundy
        rr_data = rr_data.copy()
        rr_data[rr_col] = rr_data[rr_col] / 1000

    rr_values = rr_data[rr_col].values
    time_values = rr_data['time'].values

    results = []
    
    max_time = time_values[-1]
    curr_time = time_values[0] + window_sec

    while curr_time < max_time:
        mask = (time_values >= (curr_time - window_sec)) & (time_values <= curr_time)
        window_rr = rr_values[mask]
        
        if len(window_rr) >= 30:
            try:
                # Usuwamy outliers
                q1, q3 = np.percentile(window_rr, [25, 75])
                iqr = q3 - q1
                mask_valid = (window_rr > q1 - 1.5*iqr) & (window_rr < q3 + 1.5*iqr)
                clean_rr = window_rr[mask_valid]
                
                if len(clean_rr) >= 20:
                    # Oblicz RMSSD (różnice kolejnych interwałów)
                    diffs = np.diff(clean_rr)
                    rmssd = np.sqrt(np.mean(diffs**2))
                    sdnn = np.std(clean_rr)
                    mean_rr = np.mean(clean_rr)
                    
                    # Pseudo-Alpha1: normalizacja RMSSD/SDNN do skali 0.5-1.5
                    # Wysoki RMSSD/SDNN = wysoka zmienność = wysoki alpha (stan zrelaksowany)
                    # Niski RMSSD/SDNN = niska zmienność = niski alpha (stres)
                    cv = (rmssd / mean_rr) * 100  # Coefficient of variation
                    
                    # Mapowanie CV do alpha1 (empiryczne)
                    # CV ~1-2% = niska zmienność = alpha ~0.5 (stres)
                    # CV ~5-10% = wysoka zmienność = alpha ~1.0 (relaks)
                    alpha1 = 0.4 + (cv / 15.0)  # Skalowanie
                    alpha1 = np.clip(alpha1, 0.3, 1.5)
                    
                    results.append({
                        'time': curr_time, 
                        'alpha1': alpha1,
                        'rmssd': rmssd,
                        'sdnn': sdnn,
                        'mean_rr': mean_rr
                    })
            except Exception:
                pass 
        
        curr_time += step_sec

    if not results:
        return None, f"Nie udało się obliczyć HRV. Dane: {len(rr_data)} próbek"

    return pd.DataFrame(results), None

def calculate_advanced_kpi(df_pl):
    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl.copy()
    if 'watts_smooth' not in df.columns or 'heartrate_smooth' not in df.columns:
        return 0.0, 0.0
    df_active = df[(df['watts_smooth'] > 100) & (df['heartrate_smooth'] > 80)]
    if len(df_active) < 600: return 0.0, 0.0
    mid = len(df_active) // 2
    p1, p2 = df_active.iloc[:mid], df_active.iloc[mid:]
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: return 0.0, 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    if ef1 == 0: return 0.0, 0.0
    return ((ef1 - ef2) / ef1) * 100, (df_active['watts_smooth'] / df_active['heartrate_smooth']).mean()

def calculate_z2_drift(df_pl, cp):
    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl.copy()
    if 'watts_smooth' not in df.columns or 'heartrate_smooth' not in df.columns:
        return 0.0
    df_z2 = df[(df['watts_smooth'] >= 0.55*cp) & (df['watts_smooth'] <= 0.75*cp) & (df['heartrate_smooth'] > 60)]
    if len(df_z2) < 300: return 0.0
    mid = len(df_z2) // 2
    p1, p2 = df_z2.iloc[:mid], df_z2.iloc[mid:]
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: return 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    return ((ef1 - ef2) / ef1) * 100 if ef1 != 0 else 0.0

def calculate_heat_strain_index(df_pl):
    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl.copy()
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

def process_data(df):
    df_pd = df.to_pandas() if hasattr(df, "to_pandas") else df.copy()

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

    window_long = '30s'
    window_short = '5s'
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
        'avg_watts': avg_watts, 'avg_hr': avg_hr, 'avg_cadence': avg_cadence,
        'avg_vent': avg_vent, 'avg_rr': avg_rr, 'power_hr': power_hr,
        'np_est': np_est, 'ef_factor': ef_factor, 'work_above_cp_kj': work_above_cp_kj
    }
