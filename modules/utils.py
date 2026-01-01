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
    
    # Mapping definitions: {alias: canonical}
    aliases = {
        # Heart Rate
        'hr': 'heartrate', 'heart rate': 'heartrate', 'bpm': 'heartrate', 
        'tętno': 'heartrate', 'heartrate': 'heartrate', 'heart_rate': 'heartrate',
        'heart-rate': 'heartrate', 'pulse': 'heartrate', 'heart_rate_bpm': 'heartrate',
        'heartrate_bpm': 'heartrate', 'hr_bpm': 'heartrate',
        # Power
        'power': 'watts', 'pwr': 'watts', 'moc': 'watts', 'w': 'watts', 'watts': 'watts',
        # Core Temp
        'core temp': 'core_temperature', 'core_temp': 'core_temperature', 
        'temp_central': 'core_temperature', 'temp': 'core_temperature',
        'core temperature': 'core_temperature',
        # Skin Temp
        'skin temp': 'skin_temperature', 'skin_temp': 'skin_temperature',
        'skin temperature': 'skin_temperature',
        # Ventilation
        've': 'tymeventilation', 'ventilation': 'tymeventilation', 
        'vent': 'tymeventilation',
        # Breath Rate
        'br': 'tymebreathrate', 'rr': 'tymebreathrate', 'breath rate': 'tymebreathrate',
        'breathing rate': 'tymebreathrate', 'respiration rate': 'tymebreathrate',
        # Cadence
        'cad': 'cadence', 'rpm': 'cadence',
        # Hematology
        'total_hemoglobin': 'thb', 'total hemoglobin': 'thb'
    }

    for alias, canonical in aliases.items():
        if alias in cols and canonical not in cols:
            mapping[alias] = canonical
    
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


def _read_raw_file(file) -> pd.DataFrame:
    """Read file content into raw DataFrame using Polars/Pandas."""
    # Try Polars first for speed
    try:
        import polars as pl
        file.seek(0)
        content = file.read()
        file.seek(0)
        
        # Try comma separator
        try:
            pl_df = pl.read_csv(io.BytesIO(content))
        except Exception:
            # Try semicolon
            pl_df = pl.read_csv(io.BytesIO(content), separator=';')
        
        df_pd = pl_df.to_pandas()
        logger.debug("Loaded data with Polars (fast mode)")
        return df_pd
    except Exception as e:
        logger.debug(f"Polars load failed, using Pandas: {e}")
        # Pandas fallback
        try:
            file.seek(0)
            return pd.read_csv(file, low_memory=False) 
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.info(f"Standard CSV parse failed, trying semicolon separator: {e}")
            file.seek(0)
            return pd.read_csv(file, sep=';', low_memory=False)


def _process_hrv_column(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean HRV column if present."""
    if 'hrv' in df.columns:
        df['hrv'] = df['hrv'].astype(str).apply(_clean_hrv_value)
        df['hrv'] = pd.to_numeric(df['hrv'], errors='coerce')
        df['hrv'] = df['hrv'].interpolate(method='linear').ffill().bfill()
    return df


def _convert_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known columns to numeric types."""
    numeric_cols = [
        'watts', 'heartrate', 'cadence', 'smo2', 'thb', 'temp', 'torque', 
        'core_temperature', 'skin_temperature', 'velocity_smooth', 
        'tymebreathrate', 'tymeventilation', 'rr', 'rr_interval', 'hrv', 
        'ibi', 'time', 'skin_temp', 'core_temp', 'power'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load CSV/TXT file into DataFrame with column normalization.
    
    Uses Polars for faster reading if available, falls back to Pandas.
    Refactored to single responsibility steps.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Processed DataFrame with normalized columns
    """
    # 1. IO -> Raw DataFrame
    df_pd = _read_raw_file(file)

    # 2. Normalization -> Standard Column Names
    df_pd = normalize_columns_pandas(df_pd)
    
    # 3. Data Cleaning (HRV)
    df_pd = _process_hrv_column(df_pd)

    # 4. Structure Enforcement (Time)
    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)
        
    # 5. Type Conversion
    df_pd = _convert_numeric_types(df_pd)

    return df_pd

