import streamlit as st
import pandas as pd
import numpy as np
import io

def parse_time_input(t_str):
    try:
        parts = list(map(int, t_str.split(':')))
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        if len(parts) == 2: return parts[0]*60 + parts[1]
        if len(parts) == 1: return parts[0]
    except: return None
    return None

def _serialize_df_to_parquet_bytes(df):
    bio = io.BytesIO()
    try:
        df.to_parquet(bio, index=False)
        return bio.getvalue()
    except Exception:
        bio = io.BytesIO()
        df.to_csv(bio, index=False)
        return bio.getvalue()

@st.cache_data
def load_data(file):
    try:
        file.seek(0)
        df_pd = pd.read_csv(file, low_memory=False) 
    except:
        file.seek(0)
        df_pd = pd.read_csv(file, sep=';', low_memory=False)

    df_pd.columns = [str(c).lower().strip() for c in df_pd.columns]
    rename_map = {}
    if 've' in df_pd.columns and 'tymeventilation' not in df_pd.columns: rename_map['ve'] = 'tymeventilation'
    if 'ventilation' in df_pd.columns and 'tymeventilation' not in df_pd.columns: rename_map['ventilation'] = 'tymeventilation'
    if 'total_hemoglobin' in df_pd.columns and 'thb' not in df_pd.columns: rename_map['total_hemoglobin'] = 'thb'
    if rename_map: 
        df_pd = df_pd.rename(columns=rename_map)

    if 'hrv' in df_pd.columns:
        df_pd['hrv'] = df_pd['hrv'].astype(str)
        def clean_hrv_hardcore(val):
            val = val.strip().lower()
            if val == 'nan' or val == '': 
                return np.nan
            if ':' in val:
                try:
                    parts = [float(x) for x in val.split(':') if x]
                    return np.mean(parts) if parts else np.nan
                except:
                    return np.nan
            try:
                return float(val)
            except:
                return np.nan

        df_pd['hrv'] = df_pd['hrv'].apply(clean_hrv_hardcore)
        df_pd['hrv'] = pd.to_numeric(df_pd['hrv'], errors='coerce')
        df_pd['hrv'] = df_pd['hrv'].interpolate(method='linear').ffill().bfill()

    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)

    numeric_cols = ['watts', 'heartrate', 'cadence', 'smo2', 'thb', 'temp', 'torque', 'core_temperature', 
                    'skin_temperature', 'velocity_smooth', 'tymebreathrate', 'tymeventilation', 'rr', 'rr_interval', 'hrv', 'ibi', 'time', 'skin_temp', 'core_temp', 'power']
    
    for col in numeric_cols:
        if col in df_pd.columns:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

    return df_pd

def normalize_columns_pandas(df_pd):
    mapping = {}
    cols = [c.lower() for c in df_pd.columns]
    if 've' in cols and 'tymeventilation' not in cols:
        mapping[[c for c in df_pd.columns if c.lower() == 've'][0]] = 'tymeventilation'
    if 'ventilation' in cols and 'tymeventilation' not in cols:
        mapping[[c for c in df_pd.columns if c.lower() == 'ventilation'][0]] = 'tymeventilation'
    if 'total_hemoglobin' in cols and 'thb' not in cols:
        mapping[[c for c in df_pd.columns if c.lower() == 'total_hemoglobin'][0]] = 'thb'
    df_pd = df_pd.rename(columns=mapping)
    df_pd.columns = [c.lower() for c in df_pd.columns]
    return df_pd
