"""Utility functions for data loading and parsing."""
import streamlit as st
import pandas as pd
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)


def parse_time_input(t_str: str) -> int | None:
    """Parse time string (HH:MM:SS, MM:SS, or SS) to seconds.
    
    Args:
        t_str: Time string in format HH:MM:SS, MM:SS, or SS
        
    Returns:
        Total seconds or None if parsing fails
    """
    try:
        parts = list(map(int, t_str.split(':')))
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        if len(parts) == 1: return parts[0]
    except (ValueError, AttributeError) as e:
        logger.debug(f"Failed to parse time '{t_str}': {e}")
        return None
    return None


def _serialize_df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to bytes for caching.
    
    Tries parquet first (faster), falls back to CSV.
    """
    bio = io.BytesIO()
    try:
        df.to_parquet(bio, index=False)
        return bio.getvalue()
    except (ImportError, ValueError) as e:
        logger.debug(f"Parquet serialization failed, using CSV: {e}")
        bio = io.BytesIO()
        df.to_csv(bio, index=False)
        return bio.getvalue()


def normalize_columns_pandas(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase and apply standard mappings.
    
    Mappings:
    - 've' / 'ventilation' -> 'tymeventilation'
    - 'total_hemoglobin' -> 'thb'
    """
    # Lowercase all columns first
    df_pd.columns = [str(c).lower().strip() for c in df_pd.columns]
    
    # Apply standard mappings
    mapping = {}
    cols = list(df_pd.columns)
    
    if 've' in cols and 'tymeventilation' not in cols:
        mapping['ve'] = 'tymeventilation'
    if 'ventilation' in cols and 'tymeventilation' not in cols:
        mapping['ventilation'] = 'tymeventilation'
    if 'total_hemoglobin' in cols and 'thb' not in cols:
        mapping['total_hemoglobin'] = 'thb'
    
    if mapping:
        df_pd = df_pd.rename(columns=mapping)
    
    return df_pd


def _clean_hrv_value(val: str) -> float:
    """Clean a single HRV value string.
    
    Handles formats: plain numbers, colon-separated values (e.g., "50:60:55")
    """
    val = val.strip().lower()
    if val == 'nan' or val == '': 
        return np.nan
    
    if ':' in val:
        try:
            parts = [float(x) for x in val.split(':') if x]
            return np.mean(parts) if parts else np.nan
        except ValueError:
            return np.nan
    
    try:
        return float(val)
    except ValueError:
        return np.nan


@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load CSV/TXT file into DataFrame with column normalization.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Processed DataFrame with normalized columns
    """
    # Try comma separator first, then semicolon
    try:
        file.seek(0)
        df_pd = pd.read_csv(file, low_memory=False) 
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        logger.info(f"Standard CSV parse failed, trying semicolon separator: {e}")
        file.seek(0)
        df_pd = pd.read_csv(file, sep=';', low_memory=False)

    # Normalize columns (DRY - use shared function)
    df_pd = normalize_columns_pandas(df_pd)

    # Process HRV column if present
    if 'hrv' in df_pd.columns:
        df_pd['hrv'] = df_pd['hrv'].astype(str).apply(_clean_hrv_value)
        df_pd['hrv'] = pd.to_numeric(df_pd['hrv'], errors='coerce')
        df_pd['hrv'] = df_pd['hrv'].interpolate(method='linear').ffill().bfill()

    # Ensure time column exists
    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)

    # Convert numeric columns
    numeric_cols = [
        'watts', 'heartrate', 'cadence', 'smo2', 'thb', 'temp', 'torque', 
        'core_temperature', 'skin_temperature', 'velocity_smooth', 
        'tymebreathrate', 'tymeventilation', 'rr', 'rr_interval', 'hrv', 
        'ibi', 'time', 'skin_temp', 'core_temp', 'power'
    ]
    
    for col in numeric_cols:
        if col in df_pd.columns:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

    return df_pd
