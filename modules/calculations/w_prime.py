"""
SRP: Moduł odpowiedzialny za obliczenia W' Balance (Skarbiec Beztlenowy).
"""
from typing import Union, Any
import numpy as np
import pandas as pd
import io
import streamlit as st
from numba import jit

from .common import ensure_pandas
from ..utils import _serialize_df_to_parquet_bytes


@jit(nopython=True, fastmath=True)
def calculate_w_prime_fast(watts, time, cp, w_prime_cap):
    """Szybkie obliczenie W' Balance przy użyciu Numba JIT.
    
    Implementacja modelu różnicowego W' Skiba/Morton.
    
    Args:
        watts: Tablica mocy [W]
        time: Tablica czasów [s]
        cp: Critical Power [W]
        w_prime_cap: Pojemność W' [J]
    
    Returns:
        Tablica wartości W' Balance w czasie
    """
    n = len(watts)
    w_bal = np.empty(n, dtype=np.float64)
    curr_w = w_prime_cap
    
    prev_time = time[0]
    
    for i in range(n):
        if i == 0:
            dt = 1.0
        else:
            dt = time[i] - prev_time
            if dt <= 0: dt = 1.0
            prev_time = time[i]
            
        # Differential W' Model: dW/dt = CP - P
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
    """Cached version of W' Balance calculation."""
    try:
        bio = io.BytesIO(df_bytes)
        try:
            df_pd = pd.read_parquet(bio)
        except Exception:
            bio.seek(0)
            df_pd = pd.read_csv(bio)

        if 'watts' not in df_pd.columns:
            df_pd['w_prime_balance'] = np.nan
            return df_pd

        watts_arr = df_pd['watts'].to_numpy(dtype=np.float64)
        
        if 'time' in df_pd.columns:
            time_arr = df_pd['time'].to_numpy(dtype=np.float64)
        else:
            time_arr = np.arange(len(watts_arr), dtype=np.float64)

        w_bal = calculate_w_prime_fast(watts_arr, time_arr, float(cp), float(w_prime))

        df_pd['w_prime_balance'] = w_bal
        return df_pd

    except Exception as e:
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


def calculate_w_prime_balance(_df_pl_active, cp: float, w_prime: float) -> pd.DataFrame:
    """Calculate W' Balance for the entire workout.
    
    Args:
        _df_pl_active: DataFrame with workout data
        cp: Critical Power [W]
        w_prime: W' capacity [J]
    
    Returns:
        DataFrame with added 'w_prime_balance' column
    """
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
