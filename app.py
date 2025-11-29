import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from io import BytesIO
from scipy import stats
import neurokit2 as nk

try:
    import sweatpy as sw
    SWEATP_AVAILABLE = True
except Exception:
    SWEATP_AVAILABLE = False

from fpdf import FPDF
import base64

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Pro Athlete Dashboard - Raport Treningowy', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Strona {self.page_no()}', 0, 0, 'C')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">📥 Pobierz Raport PDF</a>'

# --- KONFIGURACJA STRONY I DESIGN SYSTEM (ORYGINALNY Z APP_V1) ---
st.set_page_config(page_title="Pro Athlete Dashboard", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Rajdhani:wght@500;600;700&display=swap');

    /* GLOBALNE TŁO */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1a1f25 0%, #0e1117 90%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* TYPOGRAFIA */
    h1, h2, h3, h4, h5 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    h1 { font-weight: 700; color: #00cc96 !important; }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #30363d;
    }

    /* METRYKI */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #00cc96;
        box-shadow: 0 8px 15px rgba(0, 204, 150, 0.2);
    }
    
    /* Kolor etykiety metryki */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #8b949e !important;
        font-family: 'Rajdhani', sans-serif;
    }
    /* Kolor wartości metryki */
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #f0f6fc !important;
    }

    /* ZAKŁADKI */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.05);
        color: #c9d1d9;
        border: none;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00cc96 !important;
        color: #000000 !important;
    }

    /* INFO BOXY */
    .stAlert {
        background-color: rgba(22, 27, 34, 0.9);
        border: 1px solid #30363d;
        border-left: 5px solid #58a6ff;
        border-radius: 8px;
        color: #c9d1d9;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. KONFIGURACJA STAŁYCH ---
class Config:
    SMOOTH_WINDOW = 30
    SMOOTH_WINDOW_SHORT = 5
    COLOR_POWER = '#00cc96'
    COLOR_HR = '#ef553b'
    COLOR_SMO2 = '#ab63fa'
    COLOR_VE = '#ffa15a'
    COLOR_RR = '#19d3f3'
    COLOR_THB = '#e377c2'
    COLOR_TORQUE = '#e377c2'

# --- 3. FUNKCJE POMOCNICZE ---

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
def _calculate_w_prime_balance_cached(df_bytes: bytes, cp: float, w_prime: float):
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

        watts = df_pd['watts'].to_numpy(dtype=float)

        if 'time' in df_pd.columns:
            time_arr = df_pd['time'].to_numpy(dtype=float)
            if not np.all(np.diff(time_arr) >= 0):
                order = np.argsort(time_arr)
                time_arr = time_arr[order]
                watts = watts[order]
                inv_order = np.argsort(order)
            else:
                inv_order = None
        else:
            time_arr = np.arange(len(watts), dtype=float)
            inv_order = None

        dt = np.diff(time_arr, prepend=time_arr[0])
        if len(dt) > 1:
            dt[0] = dt[1] if dt[1] > 0 else 1.0
        else:
            dt[0] = 1.0

        w_bal = np.empty_like(watts, dtype=float)
        curr_w = float(w_prime)

        for i in range(len(watts)):
            p = watts[i]
            delta = (cp - p) * dt[i]   # >0 -> ładowanie, <0 -> zużycie
            curr_w += delta
            if curr_w > w_prime:
                curr_w = float(w_prime)
            if curr_w < 0:
                curr_w = 0.0
            w_bal[i] = curr_w

        if inv_order is not None:
            w_bal = w_bal[inv_order]

        df_pd['w_prime_balance'] = w_bal
        return df_pd

    except Exception as e:
        try:
            bio = io.BytesIO(df_bytes)
            try:
                df_pd = pd.read_parquet(bio)
            except Exception:
                bio.seek(0)
                df_pd = pd.read_csv(bio)
            df_pd['w_prime_balance'] = np.zeros(len(df_pd))
            return df_pd
        except Exception:
            return pd.DataFrame({'w_prime_balance': []})

def calculate_w_prime_balance(_df_pl_active, cp: float, w_prime: float):
    try:
        is_polars = isinstance(_df_pl_active, pl.DataFrame)
    except Exception:
        is_polars = False
    if is_polars:
        df_pd = _df_pl_active.to_pandas()
    elif isinstance(_df_pl_active, dict):
        df_pd = pd.DataFrame(_df_pl_active)
    else:
        df_pd = _df_pl_active.copy()
    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd), dtype=float)
    df_bytes = _serialize_df_to_parquet_bytes(df_pd)
    result_df = _calculate_w_prime_balance_cached(df_bytes, float(cp), float(w_prime))
    # Konwersja z powrotem do Polars DataFrame
    return pl.from_pandas(result_df)

def load_data(file):
    if file.name.endswith('.csv'):
        try:
            df = pl.read_csv(file)
            df = df.to_pandas()
            df = normalize_columns_pandas(df)
            df = pl.from_pandas(df)
        except:
            df = pl.read_csv(file, separator=';')
    else:
        df = pl.read_csv(file)

    df = df.select([pl.col(c).alias(c.lower()) for c in df.columns])
    
    rename_map = {}
    if 've' in df.columns and 'tymeventilation' not in df.columns: rename_map['ve'] = 'tymeventilation'
    if 'ventilation' in df.columns and 'tymeventilation' not in df.columns: rename_map['ventilation'] = 'tymeventilation'
    if 'total_hemoglobin' in df.columns and 'thb' not in df.columns: rename_map['total_hemoglobin'] = 'thb'
    if rename_map: df = df.rename(rename_map)

    numeric_cols = ['watts', 'heartrate', 'cadence', 'smo2', 'thb', 'temp', 'torque', 'core_temperature', 
                    'skin_temperature', 'velocity_smooth', 'tymebreathrate', 'tymeventilation', 'rr', 'rr_interval', 'hrv', 'ibi', 'time', 'skin_temp', 'core_temp', 'power']
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    if 'time' not in df.columns:
        df = df.with_columns(pl.Series("time", range(len(df))))
    return df

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


def process_data(df):
    df_pd = df.to_pandas() if hasattr(df, "to_pandas") else df.copy()

    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd)).astype(float)
    df_pd['time'] = df_pd['time'].astype(float)

    df_pd = df_pd.sort_values('time').reset_index(drop=True)
    df_pd['time_dt'] = pd.to_timedelta(df_pd['time'], unit='s')
    df_pd = df_pd.set_index('time_dt')

    num_cols = df_pd.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        df_pd[num_cols] = df_pd[num_cols].interpolate(method='time').ffill().bfill()

    try:
        df_resampled = df_pd.resample('1S').mean()
        df_resampled = df_resampled.interpolate(method='time').ffill().bfill()
    except Exception:
        df_resampled = df_pd  # fallback

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
    import polars as pl
    df_clean = pl.from_pandas(df_resampled)
    return df_clean

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
            df_above_cp = df_pl.filter(pl.col('watts') > cp_val) if hasattr(df_pl, "filter") else df_pl[df_pl['watts'] > cp_val]
            work_above_cp_kj = (df_above_cp['watts'].sum() / 1000) if len(df_above_cp)>0 else 0.0
    return {
        'avg_watts': avg_watts, 'avg_hr': avg_hr, 'avg_cadence': avg_cadence,
        'avg_vent': avg_vent, 'avg_rr': avg_rr, 'power_hr': power_hr,
        'np_est': np_est, 'ef_factor': ef_factor, 'work_above_cp_kj': work_above_cp_kj
    }

def calculate_dynamic_dfa(df_pl, window_sec=120, step_sec=10):
    """
    Oblicza DFA Alpha-1 w oknie przesuwnym.
    Wymaga kolumny z interwałami R-R (w milisekundach lub sekundach).
    """

    df = df_pl.to_pandas() if hasattr(df_pl, "to_pandas") else df_pl.copy()
    
    rr_col = next((c for c in ['rr', 'rr_interval', 'hrv', 'ibi', 'r-r', 'rr_ms'] if c in df.columns), None)
    
    if rr_col is None:
        return None # Brak danych RR

    rr_data = df[['time', rr_col]].dropna()
    rr_data = rr_data[rr_data[rr_col] > 0]
    
    if len(rr_data) < 300: # Za mało danych
        return None

    if rr_data[rr_col].mean() < 2.0: 
        rr_data[rr_col] = rr_data[rr_col] * 1000

    rr_values = rr_data[rr_col].values
    time_values = rr_data['time'].values

    results = []
    
    max_time = time_values[-1]
    curr_time = time_values[0] + window_sec

    while curr_time < max_time:
        mask = (time_values >= (curr_time - window_sec)) & (time_values <= curr_time)
        window_rr = rr_values[mask]
        
        if len(window_rr) > 100:
            try:
                clean_rr = nk.signal_sanitize(window_rr)
                dfa_metrics = nk.complexity_dfa(clean_rr, scale='default', show=False)
                alpha1 = dfa_metrics['DFA_Alpha1']
                
                if not np.isnan(alpha1):
                    results.append({'time': curr_time, 'alpha1': alpha1})
            except Exception:
                pass 
        
        curr_time += step_sec

    if not results:
        return None

    return pd.DataFrame(results)

def calculate_advanced_kpi(df_pl):
    if 'watts_smooth' not in df_pl.columns or 'heartrate_smooth' not in df_pl.columns:
        return 0.0, 0.0
    df_active = df_pl.filter((pl.col('watts_smooth') > 100) & (pl.col('heartrate_smooth') > 80))
    if df_active.height < 600: return 0.0, 0.0
    mid = df_active.height // 2
    p1, p2 = df_active.slice(0, mid), df_active.slice(mid, mid)
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: return 0.0, 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    if ef1 == 0: return 0.0, 0.0
    return ((ef1 - ef2) / ef1) * 100, (df_active['watts_smooth'] / df_active['heartrate_smooth']).mean()

def calculate_z2_drift(df_pl, cp):
    if 'watts_smooth' not in df_pl.columns or 'heartrate_smooth' not in df_pl.columns:
        return 0.0
    df_z2 = df_pl.filter((pl.col('watts_smooth') >= 0.55*cp) & (pl.col('watts_smooth') <= 0.75*cp) & (pl.col('heartrate_smooth') > 60))
    if df_z2.height < 300: return 0.0
    mid = df_z2.height // 2
    p1, p2 = df_z2.slice(0, mid), df_z2.slice(mid, mid)
    hr1 = p1['heartrate_smooth'].mean()
    hr2 = p2['heartrate_smooth'].mean()
    if hr1 == 0 or hr2 == 0: return 0.0
    ef1 = p1['watts_smooth'].mean() / hr1
    ef2 = p2['watts_smooth'].mean() / hr2
    return ((ef1 - ef2) / ef1) * 100 if ef1 != 0 else 0.0

def calculate_heat_strain_index(df_pl):
    cols = df_pl.columns
    core_col = 'core_temperature_smooth' if 'core_temperature_smooth' in cols else None
    if not core_col: return df_pl.with_columns(pl.lit(None).alias('hsi'))
    return df_pl.with_columns(
        ((5 * (pl.col(core_col) - 37.0) / 2.5) + (5 * (pl.col('heartrate_smooth') - 60.0) / 120.0))
        .clip(0.0, 10.0).alias('hsi')
    )

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

def apply_chart_style(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=title, 
            font=dict(family="Rajdhani", size=24, color="#f0f6fc")
        ) if title else None,
        font=dict(family="Inter", size=12, color="#c9d1d9"),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    return fig

# --- APP START ---

st.title("⚡ Pro Athlete Dashboard")

st.sidebar.header("Ustawienia Zawodnika")
with st.sidebar.expander("⚙️ Parametry Fizyczne", expanded=True):
    rider_weight = st.number_input("Waga Zawodnika [kg]", value=95.0, step=0.5, min_value=30.0, max_value=200.0, key="weight")
    rider_height = st.number_input("Wzrost [cm]", value=180, step=1, min_value=100, max_value=250, key="height")
    
    st.markdown("---")
    vt1_watts = st.number_input("VT1 (Próg Tlenowy) [W]", value=280, min_value=0, key="vt1_w")
    vt2_watts = st.number_input("VT2 (Próg Beztlenowy/FTP) [W]", value=400, min_value=0, key="vt2_w")
    
    st.divider()
    st.markdown("### 🫁 Wentylacja [L/min]")
    vt1_vent = st.number_input("VT1 (Próg Tlenowy) [L/min]", value=79.0, min_value=0.0, key="vt1_v")
    vt2_vent = st.number_input("VT2 (Próg Beztlenowy) [L/min]", value=136.0, min_value=0.0, key="vt2_v")

st.sidebar.divider()
cp_input = st.sidebar.number_input("Moc Krytyczna (CP/FTP) [W]", value=410, min_value=1, key="cp_in")
w_prime_input = st.sidebar.number_input("W' (W Prime) [J]", value=31000, min_value=0, key="wp_in")
st.sidebar.divider()
crank_length = st.sidebar.number_input("Długość korby [mm]", value=160.0, key="crank")
uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV / TXT)", type=['csv', 'txt'])

if rider_weight <= 0 or cp_input <= 0:
    st.error("Błąd: Waga i CP muszą być większe od zera.")
    st.stop()

if uploaded_file is not None:
    with st.spinner('Przetwarzanie danych...'):
        try:
            df_raw = load_data(uploaded_file)
            df_clean_pl = process_data(df_raw)
            metrics = calculate_metrics(df_clean_pl, cp_input)
            df_w_prime = calculate_w_prime_balance(df_clean_pl, cp_input, w_prime_input)
            decoupling_percent, ef_factor = calculate_advanced_kpi(df_clean_pl)
            drift_z2 = calculate_z2_drift(df_clean_pl, cp_input)
            df_with_hsi = calculate_heat_strain_index(df_w_prime)
            df_plot = df_with_hsi.to_pandas()
            
            if 'smo2' in df_plot.columns:
                 df_plot['smo2_smooth_ultra'] = df_plot['smo2'].rolling(window=60, center=True, min_periods=1).mean()
            df_plot_resampled = df_plot.iloc[::5, :] if len(df_plot) > 10000 else df_plot
        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {e}")
            st.stop()

       # --- HEADER METRICS ---
        if 'watts' in df_plot.columns:
            rolling_30s_header = df_plot['watts'].rolling(window=30, min_periods=1).mean()
            np_header = np.power(np.mean(np.power(rolling_30s_header, 4)), 0.25)
            if pd.isna(np_header): np_header = metrics['avg_watts']
        else:
            np_header = 0

        if cp_input > 0:
            if_header = np_header / cp_input
            duration_sec = len(df_plot)
            tss_header = (duration_sec * np_header * if_header) / (cp_input * 3600) * 100
        else:
            tss_header = 0; if_header = 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NP (Norm. Power)", f"{np_header:.0f} W", help="Normalized Power (Coggan Formula)")
        m2.metric("TSS", f"{tss_header:.0f}", help=f"IF: {if_header:.2f}")
        m3.metric("Praca [kJ]", f"{df_plot['watts'].sum()/1000:.0f}")
        m4.metric("W' Min [J]", f"{df_plot['w_prime_balance'].min():.0f}", delta_color="inverse")

        # --- ZAKŁADKI ---
        tab_raport, tab_kpi, tab_power, tab_hrv, tab_biomech, tab_thermal, tab_trends, tab_nutrition, tab_smo2, tab_hemo, tab_vent, tab_limiters, tab_model = st.tabs(
            ["Raport", "KPI", "Power", "HRV", "Biomech", "Thermal", "Trends", "Nutrition", "SmO2 Analysis", "Hematology Analysis", "Ventilation Analysis", "Limiters Analysis", "Model Analysis"]
        )

       # --- TAB RAPORT ---
        with tab_raport:
            st.header("Executive Summary")
            
            st.subheader("Przebieg Treningu")
            fig_exec = go.Figure()
            
            if 'watts_smooth' in df_plot:
                fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['watts_smooth'], name='Moc', fill='tozeroy', line=dict(color=Config.COLOR_POWER, width=1), hovertemplate="Moc: %{y:.0f} W<extra></extra>"))
            if 'heartrate_smooth' in df_plot:
                fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['heartrate_smooth'], name='HR', line=dict(color=Config.COLOR_HR, width=2), yaxis='y2', hovertemplate="HR: %{y:.0f} bpm<extra></extra>"))
            if 'smo2_smooth' in df_plot:
                fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['smo2_smooth'], name='SmO2', line=dict(color=Config.COLOR_SMO2, width=2, dash='dot'), yaxis='y3', hovertemplate="SmO2: %{y:.1f}%<extra></extra>"))
            if 'tymeventilation_smooth' in df_plot:
                fig_exec.add_trace(go.Scatter(x=df_plot['time_min'], y=df_plot['tymeventilation_smooth'], name='VE', line=dict(color=Config.COLOR_VE, width=2, dash='dash'), yaxis='y4', hovertemplate="VE: %{y:.1f} L/min<extra></extra>"))

            fig_exec.update_layout(
                template="plotly_dark", height=500,
                yaxis=dict(title="Moc [W]"),
                yaxis2=dict(title="HR", overlaying='y', side='right', showgrid=False),
                yaxis3=dict(title="SmO2", overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, 100]),
                yaxis4=dict(title="VE", overlaying='y', side='right', showgrid=False, showticklabels=False),
                legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified"
            )
            st.plotly_chart(fig_exec, use_container_width=True)

            st.markdown("---")
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.subheader("Czas w Strefach (Moc)")
                if 'watts' in df_plot:
                    bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
                    labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6']
                    colors = ['#808080', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
                    df_z = df_plot.copy()
                    df_z['Zone'] = pd.cut(df_z['watts'], bins=bins, labels=labels, right=False)
                    pcts = (df_z['Zone'].value_counts().sort_index() / len(df_z) * 100).round(1)
                    fig_hist = go.Figure(go.Bar(x=pcts.values, y=labels, orientation='h', marker_color=colors, text=pcts.apply(lambda x: f"{x}%"), textposition='auto'))
                    fig_hist.update_layout(template="plotly_dark", height=250, xaxis=dict(visible=False), yaxis=dict(showgrid=False), margin=dict(t=20, b=20))
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_dist2:
                st.subheader("Rozkład Tętna")
                if 'heartrate' in df_plot:
                    hr_counts = df_plot['heartrate'].dropna().round().astype(int).value_counts().sort_index()
                    fig_hr = go.Figure(go.Bar(x=hr_counts.index, y=hr_counts.values, marker_color=Config.COLOR_HR, hovertemplate="<b>%{x} BPM</b><br>Czas: %{y} s<extra></extra>"))
                    fig_hr.update_layout(template="plotly_dark", height=250, xaxis_title="BPM", yaxis=dict(visible=False), bargap=0.1, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_hr, use_container_width=True)

            st.markdown("---")
            c_bot1, c_bot2 = st.columns(2)
            with c_bot1:
                st.subheader("🏆 Peak Power")
                mmp_windows = {'5s': 5, '1m': 60, '5m': 300, '20m': 1200, '60m': 3600}
                cols = st.columns(5)
                if 'watts' in df_plot:
                    for c, (l, s) in zip(cols, mmp_windows.items()):
                        val = df_plot['watts'].rolling(s).mean().max()
                        with c:
                            if not pd.isna(val): st.metric(l, f"{val:.0f} W", f"{val/rider_weight:.1f} W/kg")
                            else: st.metric(l, "--")
            
            with c_bot2:
                st.subheader("🎯 Strefy (wg CP)")
                z2_l, z2_h = int(0.56*cp_input), int(0.75*cp_input)
                z3_l, z3_h = int(0.76*cp_input), int(0.90*cp_input)
                z4_l, z4_h = int(0.91*cp_input), int(1.05*cp_input)
                z5_l, z5_h = int(1.06*cp_input), int(1.20*cp_input)
                st.info(f"**Z2 (Baza):** {z2_l}-{z2_h} W | **Z3 (Tempo):** {z3_l}-{z3_h} W | **Z4 (Próg):** {z4_l}-{z4_h} W")

        # --- TAB KPI ---
        with tab_kpi:
            st.header("Kluczowe Wskaźniki Wydajności (KPI)")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Średnia Moc", f"{metrics['avg_watts']:.0f} W")
            c2.metric("Średnie Tętno", f"{metrics['avg_hr']:.0f} BPM")
            c3.metric("Średnie SmO2", f"{df_plot['smo2'].mean() if 'smo2' in df_plot else 0:.1f} %")
            c4.metric("Kadencja", f"{metrics['avg_cadence']:.0f} RPM")
            vo2max_est = calculate_vo2max(df_plot['watts'].rolling(window=300).mean().max(), rider_weight)
            c5.metric("Szac. VO2max", f"{vo2max_est:.1f}", help="Estymowane na podstawie mocy 5-minutowej (ACSM).")

            st.markdown("---")
            st.subheader("🧠 Kompendium Analityczne (Średnie)")
            with st.expander("⚡️ Szacowane VO2max (Wydolność Tlenowa)", expanded=False):
                st.markdown("""
                * **Teoria:** VO2max to maksymalna ilość tlenu, jaką Twój organizm potrafi pochłonąć, przetransportować i zużyć w ciągu minuty. Jest to kluczowy wskaźnik wydolności tlenowej.
                * **Estymacja:** Obliczone na podstawie Twojej maksymalnej mocy utrzymanej przez 5 minut (MMP 5') oraz wagi, z użyciem formuły ACSM. To jest *szacunek*, a nie pomiar laboratoryjny.
                * **Praktyka:** Wyższy VO2max generalnie koreluje z lepszymi wynikami w sportach wytrzymałościowych. Trening interwałowy o wysokiej intensywności (HIIT) jest skuteczną metodą podnoszenia tego wskaźnika.
                """)
            with st.expander("🚴 Średnia Moc (Mechanika)", expanded=True):
                st.markdown("""
                * **Teoria:** Mierzy zewnętrzne obciążenie (External Load). Mówi "ile pracy wykonałeś", ale nie "jak bardzo Cię to kosztowało".
                * **Praktyka (Triathlon):** * Ważniejsza jest **Moc Znormalizowana (NP)**. Jeśli NP jest dużo wyższe od średniej (>10%), jazda była szarpana (nieekonomiczna).
                    * W "czasówkach" dążymy do **Variability Index (VI) = 1.00-1.05**.
                """)
            with st.expander("❤️ Średnie Tętno (Fizjologia)", expanded=True):
                st.markdown("""
                * **Teoria:** Mierzy wewnętrzny koszt (Internal Load). Zależy od rzutu serca (Stroke Volume x HR).
                * **Praktyka:**
                    * Jest opóźnione względem mocy.
                    * Podatne na temperaturę, kofeinę i stres (tzw. cardiac drift).
                    * Niskie tętno przy wysokiej mocy = **Wysoka wydajność**.
                """)
            with st.expander("🩸 Średnie SmO2 (Metabolizm Lokalny)", expanded=True):
                st.markdown("""
                * **Teoria:** Bilans dostaw i zużycia tlenu bezpośrednio w mięśniu.
                * **Praktyka:**
                    * **> 60-70%:** Praca tlenowa, stabilna. Pełna resynteza ATP.
                    * **< 40-30%:** Praca beztlenowa. Mięsień korzysta z rezerw mioglobiny i glikolizy beztlenowej.
                    * Stabilne SmO2 przy rosnącym tętnie to często oznaka przegrzania, a nie braku paliwa.
                """)
            with st.expander("🦶 Kadencja (Układ Nerwowy)", expanded=True):
                st.markdown("""
                * **Teoria:** Relacja między siłą (mięśnie) a szybkością skurczu (układ nerwowy).
                * **Praktyka (Triathlon):**
                    * Zalecana wyższa kadencja (**85-95 RPM**).
                    * Oszczędza glikogen (mniej siły na obrót) i "rozluźnia" nogi przed biegiem.
                    * Zbyt niska kadencja ("ubijanie kapusty") niszczy włókna mięśniowe (mikrourazy).
                """)

            st.divider()
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Power/HR", f"{metrics['power_hr']:.2f}")
            c6.metric("Efficiency (EF)", f"{metrics['ef_factor']:.2f}")
            c7.metric("Praca > CP", f"{metrics['work_above_cp_kj']:.0f} kJ")
            c8.metric("Wentylacja (VE)", f"{metrics['avg_vent']:.1f} L/min")
            st.divider()
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Dryf (Pa:Hr)", f"{decoupling_percent:.1f} %", delta_color="inverse" if decoupling_percent<5 else "normal")
            c10.metric("Dryf Z2", f"{drift_z2:.1f} %", delta_color="inverse" if drift_z2<5 else "normal")
            max_hsi = df_plot['hsi'].max() if 'hsi' in df_plot else 0
            c11.metric("Max HSI", f"{max_hsi:.1f}", delta_color="normal" if max_hsi>5 else "inverse")
            c12.metric("Oddechy (RR)", f"{metrics['avg_rr']:.1f} /min")

            st.subheader("Wizualizacja Dryfu i Zmienności")
            if 'watts_smooth' in df_plot.columns:
                fig_dec = go.Figure()
                fig_dec.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'], name='Moc', line=dict(color=Config.COLOR_POWER, width=1.5), hovertemplate="Moc: %{y:.0f} W<extra></extra>"))
                if 'heartrate_smooth' in df_plot: fig_dec.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['heartrate_smooth'], name='HR', yaxis='y2', line=dict(color=Config.COLOR_HR, width=1.5), hovertemplate="HR: %{y:.0f} BPM<extra></extra>"))
                if 'smo2_smooth' in df_plot: fig_dec.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['smo2_smooth'], name='SmO2', yaxis='y3', line=dict(color=Config.COLOR_SMO2, dash='dot', width=1.5), hovertemplate="SmO2: %{y:.1f}%<extra></extra>"))
                
                fig_dec.update_layout(template="plotly_dark", title="Dryf Mocy, Tętna i SmO2 w Czasie", hovermode="x unified",
                    yaxis=dict(title="Moc [W]"),
                    yaxis2=dict(title="HR [bpm]", overlaying='y', side='right', showgrid=False),
                    yaxis3=dict(title="SmO2 [%]", overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, 100]),
                    legend=dict(orientation="h", y=1.1, x=0))
                st.plotly_chart(fig_dec, use_container_width=True)
                
                st.info("""
                **💡 Interpretacja: Fizjologia Zmęczenia (Triada: Moc - HR - SmO2)**

                Ten wykres pokazuje "koszt fizjologiczny" utrzymania zadanej mocy w czasie.

                **1. Stan Idealny (Brak Dryfu):**
                * **Moc (Zielony):** Linia płaska (stałe obciążenie).
                * **Tętno (Czerwony):** Linia płaska (równoległa do mocy).
                * **SmO2 (Fiolet):** Stabilne.
                * **Wniosek:** Jesteś w pełnej równowadze tlenowej. Możesz tak jechać godzinami.

                **2. Dryf Sercowo-Naczyniowy (Cardiac Drift):**
                * **Moc:** Stała.
                * **Tętno:** Powoli rośnie (rozjeżdża się z linią mocy).
                * **SmO2:** Stabilne.
                * **Przyczyna:** Odwodnienie (spadek objętości osocza) lub przegrzanie (krew ucieka do skóry). Serce musi bić szybciej, by pompować tę samą ilość tlenu.

                **3. Zmęczenie Metaboliczne (Metabolic Fatigue):**
                * **Moc:** Stała.
                * **Tętno:** Stabilne lub lekko rośnie.
                * **SmO2:** **Zaczyna spadać.**
                * **Przyczyna:** Mięśnie tracą wydajność (rekrutacja włókien szybkokurczliwych II typu, które zużywają więcej tlenu). To pierwszy sygnał nadchodzącego "odcięcia".

                **4. "Zgon" (Bonking/Failure):**
                * **Moc:** Zaczyna spadać (nie jesteś w stanie jej utrzymać).
                * **Tętno:** Może paradoksalnie spadać (zmęczenie układu nerwowego) lub rosnąć (panika organizmu).
                * **SmO2:** Gwałtowny spadek lub chaotyczne skoki.
                """)

        # --- TAB POWER ---
        with tab_power:
            st.subheader("Wykres Mocy i W'")
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['watts_smooth'], name="Moc", fill='tozeroy', line=dict(color=Config.COLOR_POWER, width=1), hovertemplate="Moc: %{y:.0f} W<extra></extra>"))
            fig_pw.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['w_prime_balance'], name="W' Bal", yaxis="y2", line=dict(color=Config.COLOR_HR, width=2), hovertemplate="W' Bal: %{y:.0f} J<extra></extra>"))
            fig_pw.update_layout(template="plotly_dark", title="Zarządzanie Energią (Moc vs W')", hovermode="x unified", yaxis=dict(title="Moc [W]"), yaxis2=dict(title="W' Balance [J]", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig_pw, use_container_width=True)
            
            st.info("""
            **💡 Interpretacja: Energia Beztlenowa (W' Balance)**

            Ten wykres pokazuje, ile "zapałek" masz jeszcze w pudełku.

            * **Czerwona Linia (W' Bal):** Poziom energii beztlenowej w Dżulach [J].
            * **Moc Krytyczna (CP):** To Twoja granica tlenowa (jak FTP, ale fizjologicznie precyzyjniejsza).

            **Jak to działa?**
            * **Moc < CP (Strefa Tlenowa):** Nie spalasz W'. Jeśli jechałeś mocno wcześniej, bateria się ładuje (czerwona linia rośnie).
            * **Moc > CP (Strefa Beztlenowa):** Zaczynasz "palić zapałki". Czerwona linia spada. Im mocniej depczesz, tym szybciej spada.
            * **W' = 0 J (Wyczerpanie):** "Odcina prąd". Nie jesteś w stanie utrzymać mocy powyżej CP ani sekundy dłużej. Musisz zwolnić, żeby zregenerować.

            **Scenariusze:**
            1.  **Interwały:** W' powinno spadać w trakcie powtórzenia (wysiłek) i rosnąć w przerwie (regeneracja). Jeśli nie wraca do 100% przed kolejnym startem, kumulujesz zmęczenie.
            2.  **Finisz:** Idealnie rozegrany wyścig to taki, gdzie W' spada do zera dokładnie na linii mety. Jeśli zostało Ci 10kJ, mogłeś finiszować mocniej. Jeśli spadło do zera 500m przed metą - przeszarżowałeś.
            3.  **Błędne CP:** Jeśli podczas spokojnej jazdy W' ciągle spada, Twoje CP jest ustawione za wysoko. Jeśli finiszujesz "w trupa", a W' pokazuje wciąż 50% - Twoje CP lub W' są niedoszacowane.
            """)

            st.subheader("Czas w Strefach Mocy (Time in Zones)")
            if 'watts' in df_plot.columns:
                bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
                labels = ['Z1: Regeneracja', 'Z2: Wytrzymałość', 'Z3: Tempo', 'Z4: Próg', 'Z5: VO2Max', 'Z6: Beztlenowa']
                colors = ['#A0A0A0', '#32CD32', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']
                df_z = df_plot.copy()
                df_z['Zone'] = pd.cut(df_z['watts'], bins=bins, labels=labels, right=False)
                pcts = (df_z['Zone'].value_counts().sort_index() / len(df_z) * 100).round(1)
                fig_z = px.bar(x=pcts.values, y=labels, orientation='h', text=pcts.apply(lambda x: f"{x}%"), color=labels, color_discrete_sequence=colors)
                fig_z.update_layout(template="plotly_dark", showlegend=False)
                st.plotly_chart(apply_chart_style(fig_z), use_container_width=True)

                st.info("""
                **💡 Interpretacja Treningowa:**
                * **Polaryzacja:** Dobry plan często ma dużo Z1/Z2 (baza) i trochę Z5/Z6 (bodziec), a mało "śmieciowych kilometrów" w Z3. Strefa Z3 to "szara strefa", która męczy, ale nie daje dużych korzyści adaptacyjnych, jednakże zużywa dużo glikogenu. Mimo tego, w triathlonie Z3 ma swoje miejsce (jazda na czas) i warto ją stosować taktycznie.
                * **Długie Wyścigi (Triathlon):** Większość czasu powinna być w Z2, z akcentami w Z4 (próg mleczanowy) i Z5 (VO2Max) dla poprawy wydolności. Spędzanie czasu w Z3 powinno być ograniczone ale taktyczne (np. jazda na czas).
                * **Sprinty i Criterium:** Więcej czasu w Z4/Z5/Z6, ale z odpowiednią regeneracją w Z1. Dużo interwałów wysokiej intensywności. Ważne jest, aby nie zaniedbywać Z2 dla budowy bazy tlenowej.
                * **Regeneracja:** Z1 to strefa regeneracyjna, idealna na dni odpoczynku lub bardzo lekkie sesje. Może pomóc w usuwaniu metabolitów i poprawie krążenia bez dodatkowego stresu. "Nie trenować" to też trening.
                * **Adaptacje Fizjologiczne:**
                * **Z1 (Szary):** Regeneracja i krążenie.
                * **Z2 (Zielony):** Kluczowe dla budowania mitochondriów i spalania tłuszczu. Podstawa wytrzymałości.
                * **Z3 (Żółty):** Mieszana strefa, poprawia ekonomię jazdy i tolerancję na wysiłek, ale może prowadzić do zmęczenia bez odpowiedniej regeneracji.
                * **Z4/Z5 (Pomarańczowy/Czerwony):** Budują tolerancję na mleczan i VO2Max, ale wymagają długiej regeneracji. Nie powinny dominować w planie treningowym.
                """)

                st.markdown("### 📚 Kompendium Fizjologii Stref (Deep Dive)")
                with st.expander("🟩 Z1/Z2: Fundament Tlenowy (< 75% CP)", expanded=True):
                    st.markdown("""
                    * **Metabolizm:** Dominacja Wolnych Kwasów Tłuszczowych (WKT). RER ~0.7-0.85. Oszczędność glikogenu.
                    * **Fizjologia:**
                        * Biogeneza mitochondriów (więcej "pieców" energetycznych).
                        * Angiogeneza (tworzenie nowych naczyń włosowatych).
                        * Wzrost aktywności enzymów oksydacyjnych.
                    * **Biomechanika:** Rekrutacja głównie włókien wolnokurczliwych (Typ I).
                    * **SmO2:** Stabilne, wysokie wartości (Równowaga Podaż=Popyt).
                    * **Oddech (VT):** Poniżej VT1. Pełna konwersacja.
                    * **Typowy Czas:** 1.5h - 6h+.
                    """)

                with st.expander("🟨 Z3: Tempo / Sweet Spot (76-90% CP)"):
                    st.markdown("""
                    * **Metabolizm:** Miks węglowodanów i tłuszczów (RER ~0.85-0.95). Zaczyna się znaczne zużycie glikogenu.
                    * **Fizjologia:** "Strefa Szara". Bodziec tlenowy, ale już z narastającym zmęczeniem.
                    * **Zastosowanie:** Trening specyficzny pod 70.3 / Ironman (długie utrzymanie mocy).
                    * **SmO2:** Stabilne, ale niższe niż w Z2. Możliwy powolny trend spadkowy.
                    * **Oddech (VT):** Okolice VT1. Głęboki, rytmiczny oddech.
                    * **Typowy Czas:** 45 min - 2.5h.
                    """)

                with st.expander("🟧 Z4: Próg Mleczanowy (91-105% CP)"):
                    st.markdown("""
                    * **Metabolizm:** Dominacja glikogenu (RER ~1.0). Produkcja mleczanu równa się jego utylizacji (MLSS).
                    * **Fizjologia:** Poprawa tolerancji na kwasicę. Zwiększenie magazynów glikogenu.
                    * **Biomechanika:** Rekrutacja włókien pośrednich (Typ IIa).
                    * **SmO2:** Granica równowagi. Utrzymuje się na stałym, niskim poziomie.
                    * **Oddech (VT):** Pomiędzy VT1 a VT2. Oddech mocny, utrudniona mowa.
                    * **Typowy Czas:** Interwały 8-30 min (łącznie do 60-90 min w sesji).
                    """)

                with st.expander("🟥 Z5/Z6: VO2Max i Beztlenowa (> 106% CP)"):
                    st.markdown("""
                    * **Metabolizm:** Wyłącznie glikogen + Fosfokreatyna (PCr). RER > 1.1.
                    * **Fizjologia:** Maksymalny pobór tlenu (pułap tlenowy). Szybkie narastanie długu tlenowego.
                    * **Biomechanika:** Pełna rekrutacja wszystkich włókien (Typ IIx). Duży moment siły.
                    * **SmO2:** Gwałtowny spadek (Desaturacja).
                    * **Oddech (VT):** Powyżej VT2 (RCP). Hiperwentylacja.
                    * **Typowy Czas:** Z5: 3-8 min. Z6: < 2 min.
                    """)
            
            st.divider()
            st.subheader("🔥 Symulator 'Spalania Zapałek' (W' Attack)")
            st.markdown("Sprawdź, jak konkretny atak wpłynie na Twoje rezerwy energii.")

            c_sim1, c_sim2 = st.columns(2)
            with c_sim1:
                sim_watts = st.slider("Moc Ataku [W]", min_value=int(cp_input), max_value=int(cp_input*2.5), value=int(cp_input*1.2), step=10)
                sim_dur = st.slider("Czas Trwania [sek]", min_value=10, max_value=300, value=60, step=10)

                if sim_watts > cp_input:
                    w_burned = (sim_watts - cp_input) * sim_dur
                    w_rem = w_prime_input - w_burned
                    w_rem_pct = (w_rem / w_prime_input) * 100
                else:
                    w_burned = 0; w_rem = w_prime_input; w_rem_pct = 100
                if w_rem < 0: w_rem = 0; w_rem_pct = 0
                st.markdown(f"**Spalone:** {w_burned:.0f} J\n**Pozostało:** {w_rem:.0f} J ({w_rem_pct:.1f}%)")
            with c_sim2:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = w_rem,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Pozostałe W'"},
                    gauge = {
                        'axis': {'range': [0, w_prime_input], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, w_prime_input*0.25], 'color': "red"},
                            {'range': [w_prime_input*0.25, w_prime_input*0.5], 'color': "orange"},
                            {'range': [w_prime_input*0.5, w_prime_input], 'color': "green"}],
                    }
                ))
                # fig_g.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), template="plotly_dark")
                st.plotly_chart(apply_chart_style(fig_g), use_container_width=True)
            
            if w_rem_pct == 0:
                st.error("💀 **TOTAL FAILURE!** Ten atak wyczerpie Cię całkowicie. Nie dojedziesz.")
            elif w_rem_pct < 25:
                st.warning("⚠️ **KRYTYCZNIE:** Bardzo ryzykowny atak. Zostaniesz na oparach.")
            else:
                st.success("✅ **BEZPIECZNIE:** Masz zapas na taki ruch.")

        # --- TAB HRV ---
        with tab_hrv:
            st.header("Analiza Zmienności Rytmu Serca (HRV & DFA)")
            
            with st.spinner("Obliczanie fraktalnej złożoności serca (DFA Alpha-1)..."):
                df_dfa = calculate_dynamic_dfa(df_clean_pl)

            if df_dfa is not None and not df_dfa.empty:
                
                df_dfa = df_dfa.sort_values('time')
                orig_times = df_clean_pl['time'].to_numpy()
                orig_watts = df_clean_pl['watts_smooth'].to_numpy() if 'watts_smooth' in df_clean_pl else np.zeros(len(orig_times))
                orig_hr = df_clean_pl['heartrate_smooth'].to_numpy() if 'heartrate_smooth' in df_clean_pl else np.zeros(len(orig_times))
                df_dfa['watts'] = np.interp(df_dfa['time'], orig_times, orig_watts)
                df_dfa['hr'] = np.interp(df_dfa['time'], orig_times, orig_hr)
                df_dfa['time_min'] = df_dfa['time'] / 60.0

                st.subheader("Detekcja Progu Aerobowego (VT1)")
                
                fig_dfa = go.Figure()
                fig_dfa.add_trace(go.Scatter(
                    x=df_dfa['time_min'], 
                    y=df_dfa['alpha1'],
                    name='DFA Alpha-1',
                    mode='lines',
                    line=dict(color='#00cc96', width=2),
                    hovertemplate="Alpha-1: %{y:.2f}<extra></extra>"
                ))

                fig_dfa.add_trace(go.Scatter(
                    x=df_dfa['time_min'], 
                    y=df_dfa['watts'],
                    name='Moc',
                    yaxis='y2',
                    fill='tozeroy',
                    line=dict(width=0.5, color='rgba(255,255,255,0.1)'),
                    hovertemplate="Moc: %{y:.0f} W<extra></extra>"
                ))

                fig_dfa.add_hline(y=0.75, line_dash="solid", line_color="#ef553b", line_width=2, 
                                annotation_text="VT1 (0.75)", annotation_position="top left")
                
                fig_dfa.add_hline(y=0.50, line_dash="dot", line_color="#ab63fa", line_width=1, 
                                annotation_text="~VT2 (0.50)", annotation_position="bottom left")

                fig_dfa.update_layout(
                    template="plotly_dark",
                    title="DFA Alpha-1 vs Czas",
                    hovermode="x unified",
                    xaxis=dict(title="Czas [min]"),
                    yaxis=dict(title="DFA Alpha-1", range=[0.2, 1.6]), # Typowy zakres
                    yaxis2=dict(title="Moc [W]", overlaying='y', side='right', showgrid=False),
                    height=500,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", y=1.05, x=0)
                )

                st.plotly_chart(fig_dfa, use_container_width=True)

                mask_threshold = (df_dfa['time_min'] > 5) & (df_dfa['alpha1'] < 0.75)
                
                if mask_threshold.any():
                    # Bierzemy pierwszy moment
                    row = df_dfa[mask_threshold].iloc[0]
                    vt1_est_power = row['watts']
                    vt1_est_hr = row['hr']
                    vt1_time = row['time_min']
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Estymowane VT1 (Moc)", f"{vt1_est_power:.0f} W", help="Moc w momencie przecięcia linii 0.75")
                    c2.metric("Estymowane VT1 (HR)", f"{vt1_est_hr:.0f} bpm", help="Tętno w momencie przecięcia linii 0.75")
                    c3.metric("Czas przecięcia", f"{vt1_time:.0f} min")
                    
                    if vt1_est_power < 100:
                        st.warning("⚠️ Wykryto bardzo niskie VT1. Sprawdź jakość danych HRV (artefakty mogą zaniżać wynik).")
                else:
                    st.info("Nie przekroczono progu 0.75 w trakcie tego treningu (cały czas praca tlenowa lub krótkie dane).")

                # --- TEORIA ---
                with st.expander("🧠 O co chodzi z DFA Alpha-1?", expanded=True):
                    st.markdown(r"""
                    **Detrended Fluctuation Analysis ($\alpha_1$)** mierzy tzw. korelacje fraktalne w odstępach między uderzeniami serca.
                    
                    * **$\alpha_1 \approx 1.0$ (Szum Różowy):** Stan zdrowy, wypoczęty. Serce bije w sposób złożony, elastyczny. Organizuje się samo.
                    * **$\alpha_1 \approx 0.5$ (Szum Biały/Losowy):** Silny stres metaboliczny. Układ nerwowy "bombarduje" węzeł zatokowy, rytm staje się nieskorelowany.
                    
                    **Dlaczego 0.75?**
                    Badania (m.in. Rogers et al.) wykazały, że przejście przez wartość **0.75** idealnie pokrywa się z **Pierwszym Progiem Wentylacyjnym (VT1)**. Jest to punkt, w którym zaczynasz tracić "luz tlenowy", a organizm zaczyna rekrutować więcej włókien szybkokurczliwych.
                    """)

            else:
                st.warning("⚠️ **Brak danych R-R (Inter-Beat Intervals).**")
                st.markdown("""
                Aby analiza DFA zadziałała, plik musi zawierać surowe dane o każdym uderzeniu serca, a nie tylko uśrednione tętno.
                * Sprawdź, czy Twój pas HR obsługuje HRV (np. Polar H10, Garmin HRM-Pro).
                * Upewnij się, że włączyłeś zapis zmienności tętna w zegarku/komputerze (często opcja "Log HRV").
                """)
            
            st.divider()
            
            c1, c2 = st.columns(2)
            
            # LEWA KOLUMNA: SmO2 + TREND
            with c1:
                st.subheader("SmO2")
                # Szukamy odpowiedniej kolumny
                col_smo2 = 'smo2_smooth_ultra' if 'smo2_smooth_ultra' in df_plot else ('smo2_smooth' if 'smo2_smooth' in df_plot else None)
                
                if col_smo2:
                    fig_s = go.Figure()
                    
                    # 1. SmO2 (Linia)
                    fig_s.add_trace(go.Scatter(
                        x=df_plot_resampled['time_min'], 
                        y=df_plot_resampled[col_smo2], 
                        name='SmO2', 
                        line=dict(color='#ab63fa', width=2), 
                        hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
                    ))
                    
                    # 2. Trend (Linia przerywana)
                    trend_y = calculate_trend(df_plot_resampled['time_min'].values, df_plot_resampled[col_smo2].values)
                    if trend_y is not None:
                        fig_s.add_trace(go.Scatter(
                            x=df_plot_resampled['time_min'], 
                            y=trend_y, 
                            name='Trend', 
                            line=dict(color='white', dash='dash', width=1.5), 
                            hovertemplate="Trend: %{y:.1f}%<extra></extra>"
                        ))
                    
                    # Layout "Pro"
                    fig_s.update_layout(
                        template="plotly_dark",
                        title="Lokalna Oksydacja (SmO2)",
                        hovermode="x unified", # <--- To robi robotę
                        yaxis=dict(title="SmO2 [%]", range=[0, 100]), # Sztywna skala dla czytelności
                        legend=dict(orientation="h", y=1.1, x=0),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=400
                    )
                    
                    st.plotly_chart(fig_s, use_container_width=True)
                    
                    st.info("""
                    **💡 Hemodynamika Mięśniowa (SmO2) - Lokalny Monitoring:**
                    
                    SmO2 to "wskaźnik paliwa" bezpośrednio w pracującym mięśniu (zazwyczaj czworogłowym uda).
                    * **Równowaga (Linia Płaska):** Podaż tlenu = Zapotrzebowanie. To stan zrównoważony (Steady State).
                    * **Desaturacja (Spadek):** Popyt > Podaż. Wchodzisz w dług tlenowy. Jeśli dzieje się to przy stałej mocy -> zmęczenie metaboliczne.
                    * **Reoksygenacja (Wzrost):** Odpoczynek. Szybkość powrotu do normy to doskonały wskaźnik wytrenowania (regeneracji).
                    """)
                else:
                     st.info("Brak danych SmO2")

            # PRAWA KOLUMNA: TĘTNO (HR)
            with c2:
                st.subheader("Tętno")
                
                # Przepisane na go.Figure dla spójności stylu z resztą aplikacji
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=df_plot_resampled['time_min'], 
                    y=df_plot_resampled['heartrate_smooth'], 
                    name='HR', 
                    fill='tozeroy', # Ładne wypełnienie pod wykresem
                    line=dict(color='#ef553b', width=2), 
                    hovertemplate="HR: %{y:.0f} BPM<extra></extra>"
                ))
                
                fig_h.update_layout(
                    template="plotly_dark",
                    title="Odpowiedź Sercowa (HR)",
                    hovermode="x unified", # <--- To robi robotę
                    yaxis=dict(title="HR [bpm]"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=400
                )
                
                st.plotly_chart(fig_h, use_container_width=True)
                
                st.info("""
                **💡 Reakcja Sercowo-Naczyniowa (HR) - Globalny System:**
                
                Serce to pompa centralna. Jego reakcja jest **opóźniona** względem wysiłku.
                * **Lag (Opóźnienie):** W krótkich interwałach (np. 30s) tętno nie zdąży wzrosnąć, mimo że moc jest max. Nie steruj sprintami na tętno!
                * **Decoupling (Rozjazd):** Jeśli moc jest stała, a tętno rośnie (dryfuje) -> organizm walczy z przegrzaniem lub odwodnieniem.
                * **Recovery HR:** Jak szybko tętno spada po wysiłku? Szybki spadek = sprawne przywspółczulne układu nerwowego (dobra forma).
                """)

            st.divider()

            st.subheader("Wentylacja (VE) i Oddechy (RR)")
            
            fig_v = go.Figure()
            
            # 1. WENTYLACJA (Oś Lewa)
            if 'tymeventilation_smooth' in df_plot_resampled:
                fig_v.add_trace(go.Scatter(
                    x=df_plot_resampled['time_min'], 
                    y=df_plot_resampled['tymeventilation_smooth'], 
                    name="VE", 
                    line=dict(color='#ffa15a', width=2), 
                    hovertemplate="VE: %{y:.1f} L/min<extra></extra>"
                ))
                
                # Trend VE
                trend_ve = calculate_trend(df_plot_resampled['time_min'].values, df_plot_resampled['tymeventilation_smooth'].values)
                if trend_ve is not None:
                     fig_v.add_trace(go.Scatter(
                         x=df_plot_resampled['time_min'], 
                         y=trend_ve, 
                         name="Trend VE", 
                         line=dict(color='#ffa15a', dash='dash', width=1.5), 
                         hovertemplate="Trend: %{y:.1f} L/min<extra></extra>"
                     ))
            
            # 2. ODDECHY / RR (Oś Prawa)
            if 'tymebreathrate_smooth' in df_plot_resampled:
                fig_v.add_trace(go.Scatter(
                    x=df_plot_resampled['time_min'], 
                    y=df_plot_resampled['tymebreathrate_smooth'], 
                    name="RR", 
                    yaxis="y2", # Druga oś
                    line=dict(color='#19d3f3', dash='dot', width=2), 
                    hovertemplate="RR: %{y:.1f} /min<extra></extra>"
                ))
            
            # Linie Progi Wentylacyjne (Zostawiamy jako stałe linie odniesienia)
            fig_v.add_hline(y=vt1_vent, line_dash="dot", line_color="green", annotation_text="VT1", annotation_position="bottom right")
            fig_v.add_hline(y=vt2_vent, line_dash="dot", line_color="red", annotation_text="VT2", annotation_position="bottom right")

            # LAYOUT (Unified Hover)
            fig_v.update_layout(
                template="plotly_dark",
                title="Mechanika Oddechu (Wydajność vs Częstość)",
                hovermode="x unified", # <--- To łączy dane w jeden dymek
                
                # Oś Lewa
                yaxis=dict(title="Wentylacja [L/min]"),
                
                # Oś Prawa
                yaxis2=dict(
                    title="Kadencja Oddechu [RR]", 
                    overlaying="y", 
                    side="right", 
                    showgrid=False
                ),
                
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )
            
            st.plotly_chart(fig_v, use_container_width=True)
            
            st.info("""
            **💡 Interpretacja: Mechanika Oddychania**

            * **Wzorzec Prawidłowy (Efektywność):** Wentylacja (VE) rośnie liniowo wraz z mocą, a częstość (RR) jest stabilna. Oznacza to głęboki, spokojny oddech.
            * **Wzorzec Niekorzystny (Płytki Oddech):** Bardzo wysokie RR (>40-50) przy stosunkowo niskim VE. Oznacza to "dyszenie" - powietrze wchodzi tylko do "martwej strefy" płuc, nie biorąc udziału w wymianie gazowej.
            * **Dryf Wentylacyjny:** Jeśli przy stałej mocy VE ciągle rośnie (rosnący trend pomarańczowej linii), oznacza to narastającą kwasicę (organizm próbuje wydmuchać CO2) lub zmęczenie mięśni oddechowych.
            * **Próg VT2 (RCP):** Punkt załamania, gdzie VE wystrzeliwuje pionowo w górę. To Twoja "czerwona linia" metaboliczna.
            """)
            
            col_vent_full = 'tymeventilation_smooth' if 'tymeventilation_smooth' in df_plot.columns else ('tymeventilation' if 'tymeventilation' in df_plot.columns else None)
            
            if col_vent_full:
                st.markdown("#### Czas w Strefach Wentylacyjnych")
                total_samples = len(df_plot)
                z1_count = len(df_plot[df_plot[col_vent_full] < vt1_vent])
                z2_count = len(df_plot[(df_plot[col_vent_full] >= vt1_vent) & (df_plot[col_vent_full] < vt2_vent)])
                z3_count = len(df_plot[df_plot[col_vent_full] >= vt2_vent])
                
                def format_time(seconds):
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    if h > 0: return f"{int(h)}h {int(m)}m {int(s)}s"
                    return f"{int(m)}m {int(s)}s"

                z1_time = format_time(z1_count)
                z2_time = format_time(z2_count)
                z3_time = format_time(z3_count)
                
                z1_pct = z1_count / total_samples * 100 if total_samples > 0 else 0
                z2_pct = z2_count / total_samples * 100 if total_samples > 0 else 0
                z3_pct = z3_count / total_samples * 100 if total_samples > 0 else 0
                
                c_z1, c_z2, c_z3 = st.columns(3)
                c_z1.metric(f"Tlenowa (< {vt1_vent} L)", z1_time, f"{z1_pct:.1f}%")
                c_z2.metric(f"Mieszana ({vt1_vent}-{vt2_vent} L)", z2_time, f"{z2_pct:.1f}%")
                c_z3.metric(f"Beztlenowa (> {vt2_vent} L)", z3_time, f"{z3_pct:.1f}%")

            if 'tymeventilation' in df_plot:
                st.markdown("#### Średnie Wartości (10 min)")
                df_s = df_plot.copy()
                df_s['Int'] = (df_s['time_min'] // 10).astype(int)
                grp = df_s.groupby('Int')[['tymeventilation', 'tymebreathrate']].mean().reset_index()
                grp['Czas'] = grp['Int'].apply(lambda x: f"{x*10}-{(x+1)*10} min")
                st.dataframe(grp[['Czas', 'tymeventilation', 'tymebreathrate']].style.format("{:.1f}", subset=['tymeventilation', 'tymebreathrate']), use_container_width=True, hide_index=True)

        # --- TAB BIOMECH ---
        with tab_biomech:
            st.header("Biomechaniczny Stres")
            
            if 'torque_smooth' in df_plot_resampled:
                fig_b = go.Figure()
                
                # 1. MOMENT OBROTOWY (Oś Lewa)
                # Kolor różowy/magenta - symbolizuje napięcie/siłę
                fig_b.add_trace(go.Scatter(
                    x=df_plot_resampled['time_min'], 
                    y=df_plot_resampled['torque_smooth'], 
                    name='Moment (Torque)', 
                    line=dict(color='#e377c2', width=1.5), 
                    hovertemplate="Moment: %{y:.1f} Nm<extra></extra>"
                ))
                
                # 2. KADENCJA (Oś Prawa)
                # Kolor cyan/turkus - symbolizuje szybkość/obroty
                if 'cadence_smooth' in df_plot_resampled:
                    fig_b.add_trace(go.Scatter(
                        x=df_plot_resampled['time_min'], 
                        y=df_plot_resampled['cadence_smooth'], 
                        name='Kadencja', 
                        yaxis="y2", # Druga oś
                        line=dict(color='#19d3f3', width=1.5), 
                        hovertemplate="Kadencja: %{y:.0f} RPM<extra></extra>"
                    ))
                
                # LAYOUT (Unified Hover)
                fig_b.update_layout(
                    template="plotly_dark",
                    title="Analiza Generowania Mocy (Siła vs Szybkość)",
                    hovermode="x unified", # <--- Klucz do sukcesu
                    
                    # Oś Lewa
                    yaxis=dict(title="Moment [Nm]"),
                    
                    # Oś Prawa
                    yaxis2=dict(
                        title="Kadencja [RPM]", 
                        overlaying="y", 
                        side="right", 
                        showgrid=False
                    ),
                    
                    legend=dict(orientation="h", y=1.1, x=0),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=450
                )
                
                st.plotly_chart(fig_b, use_container_width=True)
                
                # --- ZMIANA: ROZBUDOWANE KOMPENDIUM BIOMECH ---
                st.info("""
                **💡 Kompendium: Moment Obrotowy (Siła) vs Kadencja (Szybkość)**

                Wykres pokazuje, w jaki sposób generujesz moc.
                Pamiętaj: `Moc = Moment x Kadencja`. Tę samą moc (np. 200W) możesz uzyskać "siłowo" (50 RPM) lub "szybkościowo" (100 RPM).

                **1. Interpretacja Stylu Jazdy:**
                * **Grinding (Niska Kadencja < 70, Wysoki Moment):**
                    * **Fizjologia:** Dominacja włókien szybkokurczliwych (beztlenowych). Szybkie zużycie glikogenu.
                    * **Skutek:** "Betonowe nogi" na biegu.
                    * **Ryzyko:** Przeciążenie stawu rzepkowo-udowego (ból kolan) i odcinka lędźwiowego.
                * **Spinning (Wysoka Kadencja > 90, Niski Moment):**
                    * **Fizjologia:** Przeniesienie obciążenia na układ krążenia (serce i płuca). Lepsze ukrwienie mięśni (pompa mięśniowa).
                    * **Skutek:** Świeższe nogi do biegu (T2).
                    * **Wyzwanie:** Wymaga dobrej koordynacji nerwowo-mięśniowej (żeby nie podskakiwać na siodełku).

                **2. Praktyczne Przykłady (Kiedy co stosować?):**
                * **Podjazd:** Naturalna tendencja do spadku kadencji. **Błąd:** "Przepychanie" na twardym biegu. **Korekta:** Zredukuj bieg, utrzymaj 80+ RPM, nawet jeśli prędkość spadnie. Oszczędzisz mięśnie.
                * **Płaski odcinek (TT):** Utrzymuj "Sweet Spot" kadencji (zazwyczaj 85-95 RPM). To balans między zmęczeniem mięśniowym a sercowym.
                * **Finisz / Atak:** Chwilowe wejście w wysoki moment I wysoką kadencję. Kosztowne energetycznie, ale daje max prędkość.

                **3. Możliwe Komplikacje i Sygnały Ostrzegawcze:**
                * **Ból przodu kolana:** Zbyt duży moment obrotowy (za twarde przełożenia). -> Zwiększ kadencję.
                * **Ból bioder / "skakanie":** Zbyt wysoka kadencja przy słabej stabilizacji (core). -> Wzmocnij brzuch lub nieco zwolnij obroty.
                * **Drętwienie stóp:** Często wynik ciągłego nacisku przy niskiej kadencji. Wyższa kadencja poprawia krążenie (faza luzu w obrocie).
                """)
            
            st.divider()
            st.subheader("Wpływ Momentu na Oksydację (Torque vs SmO2)")
            
            if 'torque' in df_plot.columns and 'smo2' in df_plot.columns:
                # Przygotowanie danych (Binning)
                df_bins = df_plot.copy()
                # Grupujemy moment co 2 Nm
                df_bins['Torque_Bin'] = (df_bins['torque'] // 2 * 2).astype(int)
                
                # Liczymy statystyki dla każdego koszyka
                bin_stats = df_bins.groupby('Torque_Bin')['smo2'].agg(['mean', 'std', 'count']).reset_index()
                # Filtrujemy szum (musi być min. 10 próbek dla danej siły)
                bin_stats = bin_stats[bin_stats['count'] > 10]
                
                fig_ts = go.Figure()
                
                # 1. GÓRNA GRANICA (Mean + STD) - Niewidoczna linia, potrzebna do cieniowania
                fig_ts.add_trace(go.Scatter(
                    x=bin_stats['Torque_Bin'], 
                    y=bin_stats['mean'] + bin_stats['std'], 
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False, 
                    name='Górny zakres (+1SD)',
                    hovertemplate="Max (zakres): %{y:.1f}%<extra></extra>"
                ))
                
                # 2. DOLNA GRANICA (Mean - STD) - Wypełnienie
                fig_ts.add_trace(go.Scatter(
                    x=bin_stats['Torque_Bin'], 
                    y=bin_stats['mean'] - bin_stats['std'], 
                    mode='lines', 
                    line=dict(width=0), 
                    fill='tonexty', # Wypełnia do poprzedniej ścieżki (Górnej granicy)
                    fillcolor='rgba(255, 75, 75, 0.15)', # Lekka czerwień
                    showlegend=False, 
                    name='Dolny zakres (-1SD)',
                    hovertemplate="Min (zakres): %{y:.1f}%<extra></extra>"
                ))
                
                # 3. ŚREDNIA (Główna Linia)
                fig_ts.add_trace(go.Scatter(
                    x=bin_stats['Torque_Bin'], 
                    y=bin_stats['mean'], 
                    mode='lines+markers', 
                    name='Średnie SmO2', 
                    line=dict(color='#FF4B4B', width=3), 
                    marker=dict(size=6, color='#FF4B4B', line=dict(width=1, color='white')),
                    hovertemplate="<b>Śr. SmO2:</b> %{y:.1f}%<extra></extra>"
                ))
                
                # LAYOUT (Unified Hover)
                fig_ts.update_layout(
                    template="plotly_dark",
                    title="Agregacja: Jak Siła (Moment) wpływa na Tlen (SmO2)?",
                    hovermode="x unified", # <--- Skanujemy w pionie dla konkretnej wartości Nm
                    xaxis=dict(title="Moment Obrotowy [Nm]"),
                    yaxis=dict(title="SmO2 [%]"),
                    legend=dict(orientation="h", y=1.1, x=0),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=450
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
                
                st.info("""
                **💡 Fizjologia Okluzji (Analiza Koszykowa):**
                
                **Mechanizm Okluzji:** Kiedy mocno napinasz mięsień (wysoki moment), ciśnienie wewnątrzmięśniowe przewyższa ciśnienie w naczyniach włosowatych. Krew przestaje płynąć, tlen nie dociera, a metabolity (kwas mlekowy) nie są usuwane. To "duszenie" mięśnia od środka.
                
                **Punkt Krytyczny:** Szukaj momentu (na osi X), gdzie czerwona linia gwałtownie opada w dół. To Twój limit siłowy. Powyżej tej wartości generujesz waty 'na kredyt' beztlenowy.
                
                **Praktyczny Wniosek (Scenario):** * Masz do wygenerowania 300W. Możesz to zrobić siłowo (70 RPM, wysoki moment) lub kadencyjnie (90 RPM, niższy moment).
                * Spójrz na wykres: Jeśli przy momencie odpowiadającym 70 RPM Twoje SmO2 spada do 30%, a przy momencie dla 90 RPM wynosi 50% -> **Wybierz wyższą kadencję!** Oszczędzasz nogi (glikogen) kosztem nieco wyższego tętna.
                """)

        # --- TAB THERMAL ---
        with tab_thermal:
            st.header("Wydajność Chłodzenia")
            
            fig_t = go.Figure()
            
            # 1. CORE TEMP (Oś Lewa)
            # Kolor pomarańczowy - symbolizuje ciepło
            if 'core_temperature_smooth' in df_plot:
                fig_t.add_trace(go.Scatter(
                    x=df_plot['time_min'], 
                    y=df_plot['core_temperature_smooth'], 
                    name='Core Temp', 
                    line=dict(color='#ff7f0e', width=2), 
                    hovertemplate="Temp: %{y:.2f}°C<extra></extra>"
                ))
            
            # 2. HSI - HEAT STRAIN INDEX (Oś Prawa)
            # Kolor czerwony przerywany - symbolizuje ryzyko/alarm
            if 'hsi' in df_plot:
                fig_t.add_trace(go.Scatter(
                    x=df_plot['time_min'], 
                    y=df_plot['hsi'], 
                    name='HSI', 
                    yaxis="y2", # Druga oś
                    line=dict(color='#d62728', width=2, dash='dot'), 
                    hovertemplate="HSI: %{y:.1f}<extra></extra>"
                ))
            
            # Linie referencyjne dla temperatury (Strefy)
            fig_t.add_hline(y=38.5, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Krytyczna (38.5°C)", annotation_position="top left")
            fig_t.add_hline(y=37.5, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Optymalna (37.5°C)", annotation_position="bottom left")

            # LAYOUT (Unified Hover)
            fig_t.update_layout(
                template="plotly_dark",
                title="Termoregulacja: Temperatura Głęboka vs Indeks Zmęczenia (HSI)",
                hovermode="x unified", # <--- Skanujemy obie wartości na raz
                
                # Oś Lewa
                yaxis=dict(title="Core Temp [°C]"),
                
                # Oś Prawa
                yaxis2=dict(
                    title="HSI [0-10]", 
                    overlaying="y", 
                    side="right", 
                    showgrid=False,
                    range=[0, 12] # Lekki zapas na skali, żeby wykres nie dotykał sufitu
                ),
                
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )
            
            st.plotly_chart(fig_t, use_container_width=True)
            
            st.info("""
            **🌡️ Kompendium Termoregulacji: Fizjologia i Strategia**

            **1. Fizjologiczny Koszt Ciepła (Konkurencja o Krew)**
            Twój układ krążenia to system zamknięty o ograniczonej pojemności (ok. 5L krwi). Podczas wysiłku w upale serce musi obsłużyć dwa konkurencyjne cele:
            * **Mięśnie:** Dostarczenie tlenu i paliwa (priorytet wysiłkowy).
            * **Skóra:** Oddanie ciepła przez pot i konwekcję (priorytet przeżycia).
            * **Efekt:** Mniej krwi trafia do mięśni -> Spadek VO2max -> Wzrost tętna przy tej samej mocy (Cardiac Drift). Dodatkowo, utrata osocza (pot) zagęszcza krew, zmuszając serce do cięższej pracy.

            **2. Strefy Temperaturowe (Core Temp):**
            * **36.5°C - 37.5°C:** Homeostaza. Strefa komfortu i rozgrzewki.
            * **37.5°C - 38.4°C:** **Strefa Wydajności.** Optymalna temperatura pracy mięśni (enzymy działają najszybciej). Tutaj chcesz być podczas wyścigu.
            * **> 38.5°C:** **Strefa Krytyczna ("The Meltdown").** Ośrodkowy Układ Nerwowy (mózg) zaczyna "zaciągać hamulec ręczny", redukując rekrutację jednostek motorycznych, by chronić organy przed ugotowaniem. Odczuwasz to jako nagły brak mocy ("odcięcie").

            **3. HSI (Heat Strain Index 0-10):**
            * **0-3 (Niski):** Pełen komfort. Możesz cisnąć maxa.
            * **4-6 (Umiarkowany):** Fizjologiczny koszt rośnie. Wymagane nawadnianie.
            * **7-9 (Wysoki):** Znaczący spadek wydajności. Skup się na chłodzeniu, nie na watach.
            * **10 (Ekstremalny):** Ryzyko udaru. Zwolnij natychmiast.

            **4. Protokół Chłodzenia (Strategia):**
            * **Internal (Wewnętrzne):** Pij zimne napoje (tzw. ice slurry). Obniża to temp. żołądka i core temp.
            * **External (Zewnętrzne):** Polewaj wodą głowę, kark i **nadgarstki** (duże naczynia krwionośne blisko skóry). Lód w stroju startowym (na karku/klatce) to game-changer.

            **5. Czerwone Flagi (Kiedy przerwać):**
            * Gęsia skórka lub dreszcze w upale (paradoksalna reakcja - mózg "wariuje").
            * Nagły spadek tętna przy utrzymaniu wysiłku.
            * Zaburzenia widzenia lub koordynacji.
            """)

            st.header("Koszt Termiczny Wydajności (Cardiac Drift)")
            
            # Sprawdzamy czy mamy potrzebne kolumny
            temp_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df_plot else 'core_temperature'
            
            if 'watts' in df_plot and temp_col in df_plot and 'heartrate' in df_plot:
                
                # 1. FILTROWANIE DANYCH
                # Wywalamy zera i postoje
                mask = (df_plot['watts'] > 10) & (df_plot['heartrate'] > 60)
                df_clean = df_plot[mask].copy()
                
                # 2. OBLICZENIE EFEKTYWNOŚCI (EF)
                df_clean['eff_raw'] = df_clean['watts'] / df_clean['heartrate']
                
                # 3. USUWANIE OUTLIERÓW
                df_clean = df_clean[df_clean['eff_raw'] < 6.0]

                if not df_clean.empty:
                    # Tworzymy wykres z linią trendu (Lowess - lokalna regresja)
                    fig_te = px.scatter(
                        df_clean, 
                        x=temp_col, 
                        y='eff_raw', 
                        trendline="lowess", 
                        trendline_options=dict(frac=0.3), 
                        trendline_color_override="#FF4B4B", 
                        template="plotly_dark",
                        opacity=0.3 # Przezroczyste punkty, żeby widzieć gęstość
                    )
                    
                    # Formatowanie punktów (Scatter)
                    fig_te.update_traces(
                        selector=dict(mode='markers'),
                        marker=dict(size=5, color='#1f77b4'),
                        hovertemplate="<b>Temp:</b> %{x:.2f}°C<br><b>EF:</b> %{y:.2f} W/bpm<extra></extra>"
                    )
                    
                    # Formatowanie linii trendu
                    fig_te.update_traces(
                        selector=dict(mode='lines'),
                        line=dict(width=4),
                        hovertemplate="<b>Trend:</b> %{y:.2f} W/bpm<extra></extra>"
                    )
                    
                    # LAYOUT (Unified Hover)
                    fig_te.update_layout(
                        title="Spadek Efektywności (W/HR) vs Temperatura",
                        hovermode="x unified", # <--- To robi robotę
                        
                        xaxis=dict(title="Temperatura Głęboka [°C]"),
                        yaxis=dict(title="Efficiency Factor [W/bpm]"),
                        
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=450
                    )

                    st.plotly_chart(fig_te, use_container_width=True, config={'scrollZoom': False}, key="thermal_eff")
                    
                    st.info("""
                    ℹ️ **Jak to czytać?**
                    Ten wykres pokazuje **Cardiac Drift** w funkcji temperatury.
                    * **Oś Y (W/HR):** Ile watów generujesz z jednego uderzenia serca. Wyższa wartość = lepsza efektywność.
                    * **Oś X (Core Temp):** Twoja temperatura wewnętrzna. Wyższa wartość = większy stres cieplny.
                    * **Trend spadkowy:** Oznacza, że wraz ze wzrostem temperatury Twoje serce musi bić szybciej dla tej samej mocy (krew idzie do skóry na chłodzenie = mniejszy rzut serca dla mięśni).
                    * **Filtracja:** Usunąłem momenty, gdy nie pedałujesz (Moc < 10W), żeby nie zaburzać wyniku.
                    """)
                else:
                    st.warning("Zbyt mało danych po przefiltrowaniu (sprawdź czy masz odczyty mocy i tętna).")
            else:
                st.error("Brak wymaganych kolumn (watts, heartrate, core_temperature).")
                
                st.info("""
                **💡 Interpretacja: Koszt Fizjologiczny Ciepła (Decoupling Termiczny)**

                Ten wykres pokazuje, jak Twoje "serce płaci" za każdy wat mocy w miarę wzrostu temperatury ciała.
                * **Oś X:** Temperatura Centralna (Core Temp).
                * **Oś Y:** Efektywność (Waty na 1 uderzenie serca).
                * **Czerwona Linia:** Trend zmian.

                **🔍 Scenariusze:**
                1.  **Linia Płaska (Idealnie):** Twoja termoregulacja działa świetnie. Mimo wzrostu temperatury, serce pracuje tak samo wydajnie. Jesteś dobrze nawodniony i zaadaptowany do ciepła.
                2.  **Linia Opadająca (Typowe):** Wraz ze wzrostem temp. serce musi bić szybciej, by utrzymać tę samą moc (Dryf). Krew ucieka do skóry, by Cię chłodzić, zamiast napędzać mięśnie.
                3.  **Gwałtowny Spadek:** "Zawał termiczny" wydajności. Zazwyczaj powyżej 38.5°C. W tym momencie walczysz o przetrwanie, a nie o wynik.

                **Wniosek:** Jeśli linia leci mocno w dół, musisz poprawić chłodzenie (polewanie wodą, lód) lub strategię nawadniania przed startem.
                """)

        # --- TAB TRENDS ---
        with tab_trends:
            st.header("Trendy")
            
            if 'watts_smooth' in df_plot and 'heartrate_smooth' in df_plot:
                # Przygotowanie danych do ścieżki (Rolling Average 5 min)
                df_trend = df_plot.copy()
                df_trend['w_trend'] = df_trend['watts'].rolling(window=300, min_periods=60).mean()
                df_trend['hr_trend'] = df_trend['heartrate'].rolling(window=300, min_periods=60).mean()
                
                # Próbkowanie co 60 wierszy (co minutę), żeby nie zamulić wykresu tysiącami kropek
                df_path = df_trend.iloc[::60, :]
                
                fig_d = go.Figure()
                
                fig_d.add_trace(go.Scatter(
                    x=df_path['w_trend'], 
                    y=df_path['hr_trend'], 
                    mode='markers+lines', 
                    name='Ścieżka',
                    # Kolorowanie wg czasu (Gradient)
                    marker=dict(
                        size=8, 
                        color=df_path['time_min'], 
                        colorscale='Viridis', 
                        showscale=True, 
                        colorbar=dict(title="Czas [min]"),
                        line=dict(width=1, color='white')
                    ),
                    line=dict(color='rgba(255,255,255,0.3)', width=1), # Cienka linia łącząca
                    
                    # Bogaty Tooltip (Stylizowany jak w innych zakładkach)
                    hovertemplate="<b>Czas: %{marker.color:.0f} min</b><br>" +
                                  "Moc (5min): %{x:.0f} W<br>" +
                                  "HR (5min): %{y:.0f} BPM<extra></extra>"
                ))
                
                fig_d.update_layout(
                    template="plotly_dark",
                    title="Ścieżka Dryfu: Relacja Moc vs Tętno w Czasie",
                    
                    # Tutaj używamy 'closest', bo oś X to Moc, a nie czas. 
                    # 'x unified' zrobiłoby bałagan pokazując wszystkie momenty z tą samą mocą na raz.
                    hovermode="closest", 
                    
                    xaxis=dict(title="Moc (Średnia 5 min) [W]"),
                    yaxis=dict(title="Tętno (Średnia 5 min) [BPM]"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=500
                )
                
                st.plotly_chart(fig_d, use_container_width=True)
                
                st.info("""
                **💡 Interpretacja Ścieżki:**
                * **Pionowo w górę:** Czysty dryf tętna (rosnące zmęczenie przy stałej mocy). Związane jest to z odwodnieniem lub nagromadzeniem ciepła. Zazwyczaj obserwowane w długotrwałych wysiłkach (>60 min) w ciepłych warunkach. Protip: nawadniaj się regularnie i stosuj chłodzenie.
                * **Poziomo w prawo:** Zwiększenie mocy bez wzrostu tętna. Oznacza poprawę efektywności (np. zjazd, lepsza aerodynamika, wiatr w plecy).
                * **Poziomo w lewo:** Spadek mocy przy stałym tętnie. Może wskazywać na zmęczenie mięśniowe lub pogorszenie warunków (podjazd pod wiatr).
                * **W lewo i w dół:** Niekorzystna reakcja organizmu (spadek mocy i tętna) - możliwe początki wyczerpania energetycznego lub przegrzania.
                * **W prawo i w górę:** Zdrowa reakcja na zwiększenie intensywności. Twoje ciało efektywnie dostosowuje się do rosnącego wysiłku. Oznaka odpowiedniego poziomu wytrenowania.
                """)

            st.divider()
            st.subheader("Analiza Kwadrantowa 3D")
            if 'torque' in df_plot and 'cadence' in df_plot and 'watts' in df_plot:
                df_q = df_plot.sample(min(len(df_plot), 5000))
                color_col = 'smo2_smooth' if 'smo2_smooth' in df_q else 'watts'
                title_col = 'SmO2' if 'smo2_smooth' in df_q else 'Moc'
                scale = 'Spectral' if 'smo2_smooth' in df_q else 'Viridis'
                
                fig_3d = px.scatter_3d(df_q, x='cadence', y='torque', z='watts', color=color_col, title=f"3D Quadrant Analysis (Kolor: {title_col})", labels={'cadence': 'Kadencja', 'torque': 'Moment', 'watts': 'Moc'}, color_continuous_scale=scale, template='plotly_dark')
                fig_3d.update_traces(marker=dict(size=3, opacity=0.6), hovertemplate="Kadencja: %{x:.0f}<br>Moment: %{y:.1f}<br>Moc: %{z:.0f}<br>Val: %{marker.color:.1f}<extra></extra>")
                # W 3D używamy wbudowanego w px template, więc tylko update layout dla wysokości
                fig_3d.update_layout(height=700) 
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.info("""
                **💡 Jak czytać ten wykres 3D? (Instrukcja i Przykłady)**

                Ten wykres to "mapa Twojego silnika". Każdy punkt to jedna sekunda jazdy.
                * **Oś X (Kadencja):** Szybkość obrotu korbą.
                * **Oś Y (Moment):** Siła nacisku na pedał.
                * **Oś Z (Wysokość - Moc):** Wynik końcowy (Siła x Szybkość).
                * **Kolor (SmO2):** Poziom tlenu w mięśniu (Czerwony = Niedotlenienie, Niebieski = Komfort).

                **🔍 Przykłady z Życia (Szukaj tych obszarów na wykresie):**
                1.  **"Młynek" (Prawa Strona, Nisko):** Wysoka kadencja, niski moment. To jazda ekonomiczna (np. na płaskim). Punkty powinny być **niebieskie/zielone** (dobre ukrwienie, "pompa mięśniowa" działa).
                2.  **"Przepychanie" (Lewa Strona, Wysoko):** Niska kadencja, duża siła (np. sztywny podjazd na twardym przełożeniu). Mięśnie są napięte, krew nie dopływa. Punkty mogą być **czerwone** (hipoksja/okluzja). To męczy mięśnie szybciej niż serce.
                3.  **Sprint (Prawy Górny Róg, Wysoko w górę):** Max kadencja i max siła. Generujesz szczytową moc (Oś Z). To stan beztlenowy, punkty szybko zmienią się na **czerwone**.
                4.  **Jazda w Grupie (Środek):** Umiarkowana kadencja i siła. To Twój "Sweet Spot" biomechaniczny.

                **Wniosek:** Jeśli widzisz dużo czerwonych punktów przy niskiej kadencji, zredukuj bieg i kręć szybciej, aby dotlenić nogi!
                """)

        # --- NEW TAB: NUTRITION ---
        with tab_nutrition:
            st.header("⚡ Kalkulator Spalania Glikogenu (The Bonk Prediction)")
            
            # Interaktywne suwaki
            c1, c2, c3 = st.columns(3)
            carb_intake = c1.number_input("Spożycie Węglowodanów [g/h]", min_value=0, max_value=120, value=60, step=10)
            initial_glycogen = c2.number_input("Początkowy Zapas Glikogenu [g]", min_value=200, max_value=800, value=450, step=50, help="Standardowo: 400-500g dla wytrenowanego sportowca.")
            efficiency_input = c3.number_input("Sprawność Mechaniczna [%]", min_value=18.0, max_value=26.0, value=22.0, step=0.5, help="Amator: 18-21%, Pro: 23%+")
            
            # --- ZMIANA: "MENU KOLARSKIE" (CHEAT SHEET) ---
            with st.expander("🍬 Menu Kolarskie (Ile to węglowodanów?)", expanded=False):
                st.markdown("""
                Aby dostarczyć 90g węgli na godzinę, potrzebujesz np.:
                * **3 x Żel Energetyczny** (standardowo ~25-30g CHO / sztukę)
                * **1.5 Bidonu Izotonika** (standardowo ~40g CHO / 500ml)
                * **3 x Banan** (~25-30g CHO / sztukę)
                * **2 x Baton Energetyczny** (~40-50g CHO / sztukę)
                * **Garść Żelków (100g)** (~75g CHO)
                
                *Pamiętaj: Trening jelita jest równie ważny jak trening nóg! Nie testuj 90g/h pierwszy raz na zawodach.*
                """)
            
            if 'watts' in df_plot.columns:
                intensity_factor = df_plot['watts'] / cp_input
                
                # Model metaboliczny (Logika bez zmian)
                conditions = [
                    (df_plot['watts'] < vt1_watts),
                    (df_plot['watts'] >= vt1_watts) & (df_plot['watts'] < vt2_watts),
                    (df_plot['watts'] >= vt2_watts)
                ]
                choices = [0.3, 0.8, 1.1] 
                carb_fraction = np.select(conditions, choices, default=1.0)
                
                # Obliczenia energii
                energy_kcal_sec = df_plot['watts'] / (efficiency_input/100.0) / 4184.0
                carbs_burned_per_sec = (energy_kcal_sec * carb_fraction) / 4.0
                cumulative_burn = carbs_burned_per_sec.cumsum()
                
                intake_per_sec = carb_intake / 3600.0
                cumulative_intake = np.cumsum(np.full(len(df_plot), intake_per_sec))
                
                glycogen_balance = initial_glycogen - cumulative_burn + cumulative_intake
                
                df_nutri = pd.DataFrame({
                    'Czas [min]': df_plot['time_min'],
                    'Bilans Glikogenu [g]': glycogen_balance,
                    'Spalone [g]': cumulative_burn,
                    'Spożyte [g]': cumulative_intake,
                    'Burn Rate [g/h]': carbs_burned_per_sec * 3600
                })
                
                # --- WYKRES 1: BILANS GLIKOGENU ---
                fig_nutri = go.Figure()
                
                # Linia Balansu
                line_color = '#00cc96' if df_nutri['Bilans Glikogenu [g]'].min() > 0 else '#ef553b'
                
                fig_nutri.add_trace(go.Scatter(
                    x=df_nutri['Czas [min]'], 
                    y=df_nutri['Bilans Glikogenu [g]'], 
                    name='Zapas Glikogenu', 
                    fill='tozeroy', 
                    line=dict(color=line_color, width=2), 
                    hovertemplate="<b>Czas: %{x:.0f} min</b><br>Zapas: %{y:.0f} g<extra></extra>"
                ))
                
                # Linia "Ściana" (Bonk)
                fig_nutri.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Ściana (Bonk)", annotation_position="bottom right")
                
                fig_nutri.update_layout(
                    template="plotly_dark",
                    title=f"Symulacja Baku Paliwa (Start: {initial_glycogen}g, Intake: {carb_intake}g/h)",
                    hovermode="x unified",
                    yaxis=dict(title="Glikogen [g]"),
                    # ZMIANA TUTAJ: tickformat=".0f" wymusza liczby całkowite
                    xaxis=dict(title="Czas [min]", tickformat=".0f"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_nutri, use_container_width=True)
                
                # --- WYKRES 2: TEMPO SPALANIA (BURN RATE) ---
                st.subheader("🔥 Tempo Spalania (Burn Rate)")
                fig_burn = go.Figure()
                
                burn_rate_smooth = df_nutri['Burn Rate [g/h]'].rolling(window=60, center=True, min_periods=1).mean()
                
                fig_burn.add_trace(go.Scatter(
                    x=df_nutri['Czas [min]'], 
                    y=burn_rate_smooth, 
                    name='Spalanie', 
                    line=dict(color='#ff7f0e', width=2), 
                    fill='tozeroy', 
                    hovertemplate="<b>Czas: %{x:.0f} min</b><br>Spalanie: %{y:.0f} g/h<extra></extra>"
                ))
                
                # Linia Spożycia (Intake)
                fig_burn.add_hline(y=carb_intake, line_dash="dot", line_color="#00cc96", annotation_text=f"Intake: {carb_intake}g/h", annotation_position="top right")
                
                fig_burn.update_layout(
                    template="plotly_dark",
                    title="Zapotrzebowanie na Węglowodany",
                    hovermode="x unified",
                    yaxis=dict(title="Burn Rate [g/h]"),
                    xaxis=dict(title="Czas [min]", tickformat=".0f"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_burn, use_container_width=True)

                # PODSUMOWANIE LICZBOWE
                total_burn = cumulative_burn.iloc[-1]
                total_intake = cumulative_intake[-1]
                final_balance = glycogen_balance.iloc[-1]
                
                n1, n2, n3 = st.columns(3)
                n1.metric("Spalone Węgle", f"{total_burn:.0f} g", help="Suma węglowodanów zużytych na wysiłek")
                n2.metric("Spożyte Węgle", f"{total_intake:.0f} g", help="Suma węglowodanów dostarczonych z jedzenia/napojów")
                n3.metric("Wynik Końcowy", f"{final_balance:.0f} g", delta=f"{final_balance - initial_glycogen:.0f} g", delta_color="inverse" if final_balance < 0 else "normal")
                
                if final_balance < 0:
                    st.error(f"⚠️ **UWAGA:** Według symulacji, Twoje zapasy glikogenu wyczerpały się w okolicach {df_nutri[df_nutri['Bilans Glikogenu [g]'] < 0]['Czas [min]'].iloc[0]:.0f} minuty! To oznacza ryzyko 'odcięcia' (bonk).")
                else:
                    st.success(f"✅ **OK:** Zakończyłeś trening z zapasem {final_balance:.0f}g glikogenu. Strategia żywieniowa wystarczająca dla tej intensywności.")
                
                st.info("""
                **💡 Fizjologia Spalania (Model VT1/VT2):**
                
                * **Strefa Tłuszczowa (< VT1):** Spalasz ok. **20-40g węgli/h**. Reszta to tłuszcz. Tutaj możesz jechać godzinami na samej wodzie.
                * **Strefa Mieszana (VT1 - VT2):** Spalanie węgli skacze do **60-90g/h**. Musisz zacząć jeść (żele/izotonik), żeby nie opróżniać baku.
                * **Strefa Cukrowa (> VT2):** "Turbo". Spalasz **120g/h i więcej**. Twoje jelita nie są w stanie tyle wchłonąć (max ~90g/h). Każda minuta tutaj to "pożyczka", której nie spłacisz w trakcie jazdy.
                
                *Model uwzględnia Twoją wagę, sprawność (Efficiency) oraz progi mocy.*
                """)
            else:
                st.warning("Brak danych mocy (Watts) do obliczenia wydatku energetycznego.")

    # --- TAB SmO2 ---
    with tab_smo2:
        st.header("Analiza Kinetyki SmO2 (LT1 / LT2 Detection)")
        st.markdown("Tutaj szukamy punktów przełamania. Wybierz stabilny odcinek (interwał), a obliczymy trend desaturacji.")

        # 1. Przygotowanie danych (Wygładzanie)
        # Wybierz pierwszą dostępną ramkę danych w porządku: df_plot, df_with_hsi, df_clean_pl, df_raw
        if 'df_plot' in locals():
            target_df = df_plot
        elif 'df_with_hsi' in locals():
            # df_with_hsi może być polars lub pandas
            target_df = df_with_hsi.to_pandas() if hasattr(df_with_hsi, "to_pandas") else df_with_hsi
        elif 'df_clean_pl' in locals():
            target_df = df_clean_pl.to_pandas() if hasattr(df_clean_pl, "to_pandas") else df_clean_pl
        elif 'df_raw' in locals():
            target_df = df_raw.to_pandas() if hasattr(df_raw, "to_pandas") else df_raw
        else:
            st.error("Brak wczytanych danych. Najpierw wgraj plik w sidebar.")
            st.stop()

        if 'time' not in target_df.columns:
            st.error("Brak kolumny 'time' w danych!")
            st.stop()

        # Wygładzanie
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
        target_df['smo2_smooth'] = target_df['smo2'].rolling(window=3, center=True).mean()
        
        # TWORZENIE FORMATU CZASU (h:mm:ss) DLA TOOLTIPÓW
        target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

        # 2. Interfejs do wprowadzania interwałów (START -> KONIEC)
        col_inp1, col_inp2 = st.columns(2)
        
        with col_inp1:
            start_time_str = st.text_input("Start Interwału (h:mm:ss)", value="0:10:00")
        
        with col_inp2:
            end_time_str = st.text_input("Koniec Interwału (h:mm:ss)", value="0:12:00")

        # Funkcja parsująca czas
        def parse_time_to_seconds(t_str):
            try:
                parts = list(map(int, t_str.split(':')))
                if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
                if len(parts) == 2: return parts[0]*60 + parts[1]
                if len(parts) == 1: return parts[0]
            except:
                return None
            return None

        start_sec = parse_time_to_seconds(start_time_str)
        end_sec = parse_time_to_seconds(end_time_str)
        
        # Logika główna
        if start_sec is not None and end_sec is not None:
            if end_sec > start_sec:
                duration_sec = end_sec - start_sec
                
                # 3. Wycinanie danych
                mask = (target_df['time'] >= start_sec) & (target_df['time'] <= end_sec)
                interval_data = target_df.loc[mask]

                if not interval_data.empty:
                    # 4. Obliczenia
                    avg_watts = interval_data['watts'].mean()
                    avg_smo2 = interval_data['smo2'].mean()
                    max_smo2 = interval_data['smo2'].max()
                    min_smo2 = interval_data['smo2'].min()
                    
                    # Trend (Slope)
                    if len(interval_data) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(interval_data['time'], interval_data['smo2'])
                        trend_desc = f"{slope:.4f} %/s"
                    else:
                        slope = 0
                        intercept = 0
                        trend_desc = "N/A"

                    # Wyświetlenie Metryk
                    st.subheader(f"Metryki dla odcinka: {start_time_str} - {end_time_str} (Czas trwania: {duration_sec}s)")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Śr. Moc", f"{avg_watts:.0f} W")
                    m2.metric("Śr. SmO2", f"{avg_smo2:.1f} %")
                    m3.metric("Min SmO2", f"{min_smo2:.1f} %", delta_color="inverse")
                    m4.metric("Max SmO2", f"{max_smo2:.1f} %")
                    
                    delta_color = "normal" if slope >= -0.01 else "inverse" 
                    m5.metric("SmO2 Trend (Slope)", trend_desc, delta=trend_desc, delta_color=delta_color)

                    # 5. Wykres Główny
                    fig_smo2 = go.Figure()

                    # Oś lewa: SmO2
                    fig_smo2.add_trace(go.Scatter(
                        x=target_df['time'], 
                        y=target_df['smo2_smooth'],
                        customdata=target_df['time_str'],
                        mode='lines', 
                        name='SmO2',
                        line=dict(color='#FF4B4B', width=2),
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>SmO2:</b> %{y:.0f}%<extra></extra>"
                    ))

                    # Oś prawa: Moc
                    fig_smo2.add_trace(go.Scatter(
                        x=target_df['time'], 
                        y=target_df['watts_smooth_5s'],
                        customdata=target_df['time_str'],
                        mode='lines', 
                        name='Power',
                        line=dict(color='#1f77b4', width=1),
                        yaxis='y2',
                        opacity=0.3,
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
                    ))

                    # Zaznaczenie interwału
                    fig_smo2.add_vrect(
                        x0=start_sec, x1=end_sec,
                        fillcolor="green", opacity=0.1,
                        layer="below", line_width=0,
                        annotation_text="ANALIZA", annotation_position="top left"
                    )
                    
                    # Linia trendu
                    if len(interval_data) > 1:
                        trend_line = intercept + slope * interval_data['time']
                        fig_smo2.add_trace(go.Scatter(
                            x=interval_data['time'], 
                            y=trend_line,
                            customdata=interval_data['time_str'],
                            mode='lines', 
                            name='Trend SmO2',
                            line=dict(color='yellow', width=3, dash='dash'),
                            hovertemplate="<b>Czas:</b> %{customdata}<br><b>Trend:</b> %{y:.1f}%<extra></extra>"
                        ))

                    # Layout
                    fig_smo2.update_layout(
                        title="Analiza Przebiegu SmO2 vs Power",
                        xaxis_title="Czas",
                        yaxis=dict(title="SmO2 (%)", range=[0, 100]),
                        yaxis2=dict(title="Power (W)", overlaying='y', side='right', showgrid=False),
                        legend=dict(x=0.01, y=0.99),
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_smo2, use_container_width=True)
                    
                    # 6. SEKCJA TEORII (Rozwijana)
                    with st.expander("📚 TEORIA: Jak wyznaczyć LT1 i LT2 z SmO2? (Kliknij, aby rozwinąć)", expanded=False):
                        st.markdown("""
                        ### 1. Interpretacja Slope (Nachylenia Trendu)
                        Slope mówi nam o równowadze między dostawą a zużyciem tlenu w mięśniu.
                        
                        * **Slope > 0 (Dodatni): "Luksus Tlenowy"**
                            * *Co się dzieje:* Dostawa tlenu przewyższa zużycie.
                            * *Kiedy:* Rozgrzewka, regeneracja, początek interwału (rzut serca rośnie szybciej niż zużycie).
                        
                        * **Slope ≈ 0 (Bliski Zera): "Steady State"**
                            * *Wartości:* Zazwyczaj od **-0.005 do +0.005 %/s**.
                            * *Co się dzieje:* Równowaga. Tyle ile mięsień potrzebuje, tyle krew dostarcza.
                            * *Kiedy:* Jazda w strefie tlenowej (Z2), Sweet Spot (jeśli wytrenowany).
                        
                        * **Slope < 0 (Ujemny): "Desaturacja / Dług Tlenowy"**
                            * *Wartości:* Poniżej **-0.01 %/s** (wyraźny spadek).
                            * *Co się dzieje:* Mitochondria zużywają więcej tlenu niż jest dostarczane. Mioglobina traci tlen.
                            * *Kiedy:* Jazda powyżej progu beztlenowego (LT2), mocne skoki mocy.

                        ---

                        ### 2. Jak znaleźć progi (Breakpoints)?
                        
                        #### 🟢 LT1 (Aerobic Threshold)
                        Szukaj mocy, przy której Slope zmienia się z **dodatniego na płaski (bliski 0)**.
                        * *Przykład:* Przy 180W SmO2 jeszcze rośnie, przy 200W staje w miejscu. **LT1 ≈ 200W**.
                        
                        #### 🔴 LT2 (Anaerobic Threshold / Critical Power)
                        Szukaj mocy, przy której **nie jesteś w stanie ustabilizować SmO2** (brak Steady State).
                        * *Scenariusz:*
                            * 280W: SmO2 spada, ale po minucie się poziomuje (Slope wraca do 0). -> **Jesteś pod progiem.**
                            * 300W: SmO2 leci w dół ciągle i nie chce się zatrzymać (Slope ciągle ujemny). -> **Jesteś nad progiem (powyżej LT2).**
                        
                        ---
                        
                        ### ⚠️ WAŻNE: Pro Tip Biomechaniczny
                        **Uważaj na niską kadencję (Grinding)!**
                        Przy tej samej mocy, niska kadencja = wyższy moment siły (Torque). To powoduje większy ucisk mechaniczny na naczynia krwionośne w mięśniu (okluzja).
                        * *Efekt:* SmO2 może spadać gwałtownie (sztuczna desaturacja) tylko przez mechanikę, mimo że metabolicznie organizm dałby radę.
                        * *Rada:* Testy progowe rób na swojej naturalnej, stałej kadencji.
                        """)

                else:
                    st.warning("Brak danych w wybranym zakresie. Sprawdź poprawność wpisanego czasu.")
            else:
                st.error("Czas zakończenia musi być późniejszy niż czas rozpoczęcia!")
        else:
            st.warning("Wprowadź poprawne czasy w formacie h:mm:ss (np. 0:10:00).")

    # --- TAB HEMODYNAMICS (THb vs SmO2) ---
    with tab_hemo:
        st.header("Profil Hemodynamiczny (Mechanika vs Metabolizm)")
        st.markdown("Analiza relacji objętości krwi (THb) do saturacji (SmO2). Pozwala wykryć okluzję (ucisk) i limitery przepływu.")

        # 1. Próba znalezienia kolumny THb
        if 'df_plot' in locals():
            target_df = df_plot
        elif 'df_with_hsi' in locals():
            target_df = df_with_hsi.to_pandas() if hasattr(df_with_hsi, "to_pandas") else df_with_hsi
        elif 'df_clean_pl' in locals():
            target_df = df_clean_pl.to_pandas() if hasattr(df_clean_pl, "to_pandas") else df_clean_pl
        elif 'df_raw' in locals():
            target_df = df_raw.to_pandas() if hasattr(df_raw, "to_pandas") else df_raw
        else:
            st.error("Brak danych. Najpierw wgraj plik.")
            st.stop()
        col_thb = next((c for c in ['thb', 'total_hemoglobin', 'total_hgb'] if c in target_df.columns), None)
        col_smo2 = 'smo2_smooth' if 'smo2_smooth' in target_df else ('smo2' if 'smo2' in target_df else None)

        if col_thb and col_smo2:
            
            # Wygładzanie THb dla czytelności (jeśli nie jest już wygładzone)
            if f"{col_thb}_smooth" not in target_df.columns:
                target_df[f'{col_thb}_smooth'] = target_df[col_thb].rolling(window=10, center=True).mean()
            
            thb_val = f'{col_thb}_smooth'

            # Uśrednianie SmO2 dla wykresu trendu (10s)
            if 'smo2' in target_df.columns:
                target_df['smo2_smooth_10s_hemo_trend'] = target_df['smo2'].rolling(window=10, center=True).mean()
                col_smo2_hemo_trend = 'smo2_smooth_10s_hemo_trend'
            else:
                col_smo2_hemo_trend = col_smo2 # Fallback to existing or no smoothing
            
            # 2. Wykres XY (Scatter) - SmO2 vs THb
            # Kolorujemy punktami Mocy, żeby widzieć co się dzieje na wysokich watach
            
            # Próbkowanie dla szybkości (oryginalne zachowanie)
            df_hemo = target_df.sample(min(len(target_df), 5000))
            
            fig_hemo = px.scatter(
                df_hemo, 
                x=col_smo2, # Revert to original col_smo2 (3s smoothed or raw)
                y=thb_val, 
                color='watts', 
                title="Hemo-Scatter: SmO2 (Oś X) vs THb (Oś Y)", # Revert title
                labels={col_smo2: "SmO2 (Saturacja) [%]", thb_val: "THb (Objętość Krwi) [a.u.]", "watts": "Moc [W]"},
                template="plotly_dark",
                color_continuous_scale='Turbo' # Turbo jest świetne do pokazywania intensywności
            )
            
            # Odwracamy oś X dla SmO2 (zwyczajowo w fizjologii wykresy czyta się od prawej do lewej dla desaturacji)
            fig_hemo.update_xaxes(autorange="reversed")
            
            fig_hemo.update_traces(marker=dict(size=5, opacity=0.6))
            fig_hemo.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            # Dodajemy adnotacje "ćwiartek" (Uproszczona interpretacja)
            # To wymagałoby znania średnich, ale damy opisy w rogach
            fig_hemo.add_annotation(xref="paper", yref="paper", x=0.05, y=0.95, text="<b>Stres Metaboliczny</b><br>(Wazodylatacja)", showarrow=False, font=dict(color="#00cc96"))
            fig_hemo.add_annotation(xref="paper", yref="paper", x=0.05, y=0.05, text="<b>OKLUZJA / UCISK</b><br>(Limit Przepływu)", showarrow=False, font=dict(color="#ef553b"))
            fig_hemo.add_annotation(xref="paper", yref="paper", x=0.95, y=0.95, text="<b>Regeneracja</b><br>(Napływ)", showarrow=False, font=dict(color="#ffa15a"))
            
            st.plotly_chart(fig_hemo, use_container_width=True)
            
            # 3. Wykres Liniowy w czasie (Dual Axis)
            st.subheader("Trendy w Czasie (Szukanie Rozjazdu)")
            fig_trend = go.Figure()
            
            # SmO2 (Oś Lewa)
            fig_trend.add_trace(go.Scatter(
                x=target_df['time_min'], y=target_df[col_smo2_hemo_trend],
                name='SmO2', line=dict(color='#ab63fa', width=2),
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
            ))

            
            # THb (Oś Prawa)
            fig_trend.add_trace(go.Scatter(
                x=target_df['time_min'], y=target_df[thb_val],
                name='THb', line=dict(color='#ffa15a', width=2), yaxis='y2',
                hovertemplate="THb: %{y:.2f}<extra></extra>"
            ))
            
            # Tło - Moc (dla kontekstu)
            if 'watts_smooth_30s' in target_df:
                 fig_trend.add_trace(go.Scatter(
                    x=target_df['time_min'], y=target_df['watts_smooth_30s'],
                    name='Moc', line=dict(color='rgba(255,255,255,0.1)', width=1),
                    fill='tozeroy', fillcolor='rgba(255,255,255,0.05)', yaxis='y3',
                    hoverinfo='skip'
                ))

            # Poprawiony Layout dla fig_trend (bez titlefont)
            fig_trend.update_layout(
                template="plotly_dark",
                title="SmO2 vs THb w Czasie",
                hovermode="x unified",
                yaxis=dict(
                    title=dict(text="SmO2 [%]", font=dict(color='#ab63fa'))
                ),
                yaxis2=dict(
                    title=dict(text="THb [a.u.]", font=dict(color='#ffa15a')),
                    overlaying='y', side='right'
                ),
                yaxis3=dict(title="Moc", overlaying='y', side='right', showgrid=False, showticklabels=False), 
                height=450
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # 4. Teoria dla Fizjologii
            st.info("""
            **💡 Interpretacja Hemodynamiczna (THb + SmO2):**
            
            THb (Total Hemoglobin) to wskaźnik objętości krwi ("tHb = pompa paliwowa"). SmO2 to wskaźnik zużycia ("SmO2 = bak").
            
            * **Scenariusz 1: Dobra praca (Wazodylatacja)**
                * **SmO2 SPADA 📉 | THb ROŚNIE 📈**
                * *Co to znaczy:* Mięsień pracuje mocno, metabolizm zużywa tlen, ale układ krążenia reaguje prawidłowo, rozszerzając naczynia i pompując więcej krwi. To zdrowy limit metaboliczny.
            
            * **Scenariusz 2: Okluzja / Limit Mechaniczny (UWAGA!)**
                * **SmO2 SPADA 📉 | THb SPADA 📉 (lub płaskie)**
                * *Co to znaczy:* "Wyżymanie gąbki". Napięcie mięśnia jest tak duże (lub kadencja za niska), że ciśnienie wewnątrzmięśniowe blokuje dopływ świeżej krwi.
                * *Działanie:* Zwiększ kadencję, sprawdź siodełko (czy nie uciska tętnic), popraw fit.
            
            * **Scenariusz 3: Venous Pooling (Zastój)**
                * **SmO2 ROŚNIE 📈 | THb ROŚNIE 📈**
                * *Kiedy:* Często podczas nagłego zatrzymania po wysiłku. Krew napływa, ale pompa mięśniowa nie odprowadza jej z powrotem.
            """)

        else:
            st.warning("⚠️ Brak danych THb (Total Hemoglobin). Sensor Moxy/Train.Red powinien dostarczać tę kolumnę (często jako 'thb' lub 'total_hemoglobin'). Bez tego analiza hemodynamiczna jest niemożliwa.")
            st.markdown("Dostępne kolumny w pliku: " + ", ".join(target_df.columns))

    # --- TAB VENT ANALYSIS (VT1 / VT2) ---
    with tab_vent:
        st.header("Analiza Progu Wentylacyjnego (VT1 / VT2 Detection)")
        st.markdown("Analiza dynamiki oddechu. Szukamy nieliniowych przyrostów wentylacji (VE) względem mocy.")

        # 1. Przygotowanie danych
        if 'df_plot' in locals():
            target_df = df_plot
        elif 'df_with_hsi' in locals():
            target_df = df_with_hsi.to_pandas() if hasattr(df_with_hsi, "to_pandas") else df_with_hsi
        elif 'df_clean_pl' in locals():
            target_df = df_clean_pl.to_pandas() if hasattr(df_clean_pl, "to_pandas") else df_clean_pl
        elif 'df_raw' in locals():
            target_df = df_raw.to_pandas() if hasattr(df_raw, "to_pandas") else df_raw
        else:
            st.error("Brak danych.")
            st.stop()

        if 'time' not in target_df.columns or 'tymeventilation' not in target_df.columns:
            st.error("Brak danych wentylacji (tymeventilation) lub czasu!")
            st.stop()

        # Wygładzanie (VE jest szumiące, dajemy 10s smooth)
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
        target_df['ve_smooth'] = target_df['tymeventilation'].rolling(window=10, center=True).mean()
        target_df['rr_smooth'] = target_df['tymebreathrate'].rolling(window=10, center=True).mean() if 'tymebreathrate' in target_df else 0
        
        # Format czasu
        target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

        # 2. Interfejs (START -> KONIEC)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            start_time_v = st.text_input("Start Analizy (h:mm:ss)", value="0:20:00", key="vent_start")
        with col_v2:
            end_time_v = st.text_input("Koniec Analizy (h:mm:ss)", value="0:25:00", key="vent_end")

        # Parser czasu (lokalny dla pewności)
        def parse_time_vent(t_str):
            try:
                parts = list(map(int, t_str.split(':')))
                if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
                if len(parts) == 2: return parts[0]*60 + parts[1]
                if len(parts) == 1: return parts[0]
            except: return None
            return None

        s_sec = parse_time_vent(start_time_v)
        e_sec = parse_time_vent(end_time_v)

        if s_sec is not None and e_sec is not None and e_sec > s_sec:
            duration_v = e_sec - s_sec
            
            # 3. Wycinanie
            mask_v = (target_df['time'] >= s_sec) & (target_df['time'] <= e_sec)
            interval_v = target_df.loc[mask_v]

            if not interval_v.empty:
                # 4. Obliczenia
                avg_w = interval_v['watts'].mean()
                avg_ve = interval_v['tymeventilation'].mean()
                avg_rr = interval_v['tymebreathrate'].mean() if 'tymebreathrate' in interval_v else 0
                max_ve = interval_v['tymeventilation'].max()
                
                # Ve/Power Ratio (Efektywność)
                ve_power_ratio = avg_ve / avg_w if avg_w > 0 else 0
                
                # Trend (Slope) dla VE
                if len(interval_v) > 1:
                    slope_ve, intercept_ve, _, _, _ = stats.linregress(interval_v['time'], interval_v['tymeventilation'])
                    trend_desc_ve = f"{slope_ve:.4f} L/s"
                else:
                    slope_ve = 0; intercept_ve = 0; trend_desc_ve = "N/A"

                # Metryki
                st.subheader(f"Metryki Oddechowe: {start_time_v} - {end_time_v} ({duration_v}s)")
                mv1, mv2, mv3, mv4, mv5 = st.columns(5)
                mv1.metric("Śr. Moc", f"{avg_w:.0f} W")
                mv2.metric("Śr. Wentylacja (VE)", f"{avg_ve:.1f} L/min")
                mv3.metric("Częstość (RR)", f"{avg_rr:.1f} /min")
                mv4.metric("Wydajność (VE/W)", f"{ve_power_ratio:.3f}", help="Ile litrów powietrza na 1 Wat mocy. Niżej = lepiej (do pewnego momentu).")
                
                # Kolorowanie trendu (Tu odwrotnie niż w SmO2: Duży wzrost = Czerwony/Ostrzegawczy)
                trend_color = "inverse" if slope_ve > 0.1 else "normal"
                mv5.metric("Trend VE (Slope)", trend_desc_ve, delta=trend_desc_ve, delta_color=trend_color)

                # 5. Wykres
                fig_vent = go.Figure()

                # Lewa Oś: Wentylacja
                fig_vent.add_trace(go.Scatter(
                    x=target_df['time'], y=target_df['ve_smooth'],
                    customdata=target_df['time_str'],
                    mode='lines', name='VE (L/min)',
                    line=dict(color='#ffa15a', width=2),
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>"
                ))

                # Prawa Oś: Moc
                fig_vent.add_trace(go.Scatter(
                    x=target_df['time'], y=target_df['watts_smooth_5s'],
                    customdata=target_df['time_str'],
                    mode='lines', name='Power',
                    line=dict(color='#1f77b4', width=1),
                    yaxis='y2', opacity=0.3,
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
                ))

                # Zaznaczenie
                fig_vent.add_vrect(x0=s_sec, x1=e_sec, fillcolor="orange", opacity=0.1, layer="below", annotation_text="ANALIZA", annotation_position="top left")

                # Linia trendu VE
                if len(interval_v) > 1:
                    trend_line_ve = intercept_ve + slope_ve * interval_v['time']
                    fig_vent.add_trace(go.Scatter(
                        x=interval_v['time'], y=trend_line_ve,
                        customdata=interval_v['time_str'],
                        mode='lines', name='Trend VE',
                        line=dict(color='white', width=2, dash='dash'),
                        hovertemplate="<b>Trend:</b> %{y:.2f} L/min<extra></extra>"
                    ))

                fig_vent.update_layout(
                    title="Dynamika Wentylacji vs Moc",
                    xaxis_title="Czas",
                    yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
                    yaxis2=dict(title=dict(text="Moc (W)", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
                    legend=dict(x=0.01, y=0.99),
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_vent, use_container_width=True)

                # 6. TEORIA ODDECHOWA
                with st.expander("🫁 TEORIA: Jak znaleźć VT1 i VT2 na podstawie Slope?", expanded=False):
                    st.markdown("""
                    ### Interpretacja Slope (Nachylenia VE)
                    Wentylacja rośnie nieliniowo. Szukamy punktów załamania krzywej ("Kinks").

                    #### 🟢 1. Strefa Tlenowa (Poniżej VT1)
                    * **Zachowanie:** VE rośnie proporcjonalnie do mocy (liniowo).
                    * **Slope:** Stabilny, umiarkowanie dodatni (np. 0.02 - 0.05 L/s).
                    * **RR (Oddechy):** Stabilne, wolne pogłębianie oddechu.

                    #### 🟡 2. Próg VT1 (Aerobic Threshold) - "Pierwsze Przełamanie"
                    * **Co szukać:** Pierwszy moment, gdzie Slope wyraźnie wzrasta, mimo że moc rośnie liniowo.
                    * **Fizjologia:** Buforowanie kwasu mlekowego wodorowęglanami -> powstaje ekstra CO2 -> musisz go wydychać.
                    * **Test mowy:** Tutaj zaczynasz urywać zdania.

                    #### 🔴 3. Próg VT2 (Respiratory Compensation Point) - "Drugie Przełamanie"
                    * **Co szukać:** Slope wystrzeliwuje w górę ("Vertical spike"). VE rośnie wykładniczo.
                    * **Wartości Slope:** Bardzo wysokie (np. > 0.15 L/s).
                    * **RR (Oddechy):** Gwałtowny wzrost częstości (tachypnoe).
                    * **Fizjologia:** Hiperwentylacja. Organizm nie nadąża z usuwaniem CO2. Koniec równowagi.
                    ---
                    **Pro Tip:** Porównaj Slope VE ze Slope Mocy. Jeśli Moc rośnie o 5%, a VE o 15% -> właśnie przekroczyłeś próg.
                    """)
            else:
                st.warning("Brak danych w tym zakresie.")
        else:
            st.warning("Wprowadź poprawny zakres czasu.")

    # --- TAB LIMITERS (RADAR CHART) ---
    with tab_limiters:
        st.header("Analiza Limiterów Fizjologicznych (Radar)")
        st.markdown("Sprawdzamy, który układ (Serce, Płuca, Mięśnie) był 'wąskim gardłem' podczas najcięższych momentów treningu.")

        # Sprawdzamy dostępność danych
        has_hr = 'heartrate' in df_plot
        has_ve = any(c in df_plot.columns for c in ['tymeventilation', 've', 'ventilation'])
        has_smo2 = 'smo2' in df_plot
        has_watts = 'watts' in df_plot

        if has_watts and (has_hr or has_ve or has_smo2):
            
            # 1. Wybór okna czasowego (Peak Power)
            window_options = {
                "1 min (Anaerobic)": 60, 
                "5 min (VO2max)": 300, 
                "20 min (FTP)": 1200,
                "60 min (Endurance)": 3600
            }
            selected_window_name = st.selectbox("Wybierz okno analizy (MMP):", list(window_options.keys()), index=1)
            window_sec = window_options[selected_window_name]

            # Znajdujemy indeks startu dla najlepszej średniej mocy w tym oknie
            # Rolling musi mieć min_periods=window_sec, żeby nie liczyć "połówek" na początku
            df_plot['rolling_watts'] = df_plot['watts'].rolling(window=window_sec, min_periods=window_sec).mean()

            if df_plot['rolling_watts'].isna().all():
                st.warning(f"Trening jest krótszy niż {window_sec/60:.0f} min. Wybierz krótsze okno.")
                st.stop()

            peak_idx = df_plot['rolling_watts'].idxmax()

            # Sprawdzamy, czy znaleziono peak (czy trening był wystarczająco długi)
            if not pd.isna(peak_idx):
                # Wycinamy ten fragment danych
                start_idx = max(0, peak_idx - window_sec + 1)
                df_peak = df_plot.iloc[start_idx:peak_idx+1]
                
                # 2. Obliczamy % wykorzystania potencjału (Estymacja Maxów)
                
                # HR (Centralny)
                peak_hr_avg = df_peak['heartrate'].mean() if has_hr else 0
                max_hr_user = df_plot['heartrate'].max() 
                pct_hr = (peak_hr_avg / max_hr_user * 100) if max_hr_user > 0 else 0
                
                # VE (Oddechowy)
                col_ve_nm = next((c for c in ['tymeventilation', 've', 'ventilation'] if c in df_plot.columns), None)
                peak_ve_avg = df_peak[col_ve_nm].mean() if col_ve_nm else 0
                # Estymujemy Max VE jako 110% VT2 (bezpieczny margines dla RCP)
                max_ve_user = vt2_vent * 1.1 
                pct_ve = (peak_ve_avg / max_ve_user * 100) if max_ve_user > 0 else 0
                
                # SmO2 (Lokalny) - Odwrócona logika (im mniej tym "więcej" pracy)
                peak_smo2_avg = df_peak['smo2'].mean() if has_smo2 else 100
                # Używamy 100 - SmO2 jako "stopnia ekstrakcji tlenu"
                pct_smo2_util = 100 - peak_smo2_avg
                
                # Power (Mechaniczny) vs CP
                peak_w_avg = df_peak['watts'].mean()
                pct_power = (peak_w_avg / cp_input * 100) if cp_input > 0 else 0

                # 3. Rysujemy Radar
                categories = ['Serce (% HRmax)', 'Płuca (% VEmax)', 'Mięśnie (% Desat)', 'Moc (% CP)']
                values = [pct_hr, pct_ve, pct_smo2_util, pct_power]
                
                # Zamykamy koło dla wykresu radarowego
                values += [values[0]]
                categories += [categories[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_window_name,
                    line=dict(color='#00cc96'),
                    fillcolor='rgba(0, 204, 150, 0.3)',
                    hovertemplate="%{theta}: <b>%{r:.1f}%</b><extra></extra>"
                ))

                # Dynamiczna skala - jeśli moc wyskoczy poza 120% (np. przy 1 min), zwiększamy zakres
                max_val = max(values)
                range_max = 100 if max_val < 100 else (max_val + 10)

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, range_max] 
                        )
                    ),
                    template="plotly_dark",
                    title=f"Profil Obciążenia: {selected_window_name} ({peak_w_avg:.0f} W)",
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # 4. Interpretacja
                st.info(f"""
                **🔍 Diagnoza dla odcinka {selected_window_name}:**
                
                * **Serce (Central):** {pct_hr:.1f}% Maxa. (Wysokie tętno = koszt transportu).
                * **Płuca (Oddech):** {pct_ve:.1f}% Szacowanego Maxa. (Wysokie VE = koszt usunięcia CO2).
                * **Mięśnie (Lokalne):** {pct_smo2_util:.1f}% Wykorzystania tlenu (Średnie SmO2: {peak_smo2_avg:.1f}%).
                * **Moc:** {pct_power:.0f}% Twojego CP/FTP.
                
                **Co Cię zatrzymało?**
                Patrz, który "wierzchołek" jest najdalej od środka.
                * Jeśli **Serce > Mięśnie**: Ograniczenie centralne (układ krążenia nie nadąża z dostawą).
                * Jeśli **Mięśnie > Serce**: Ograniczenie peryferyjne (mięśnie zużywają wszystko, co dostają, albo jest okluzja mechaniczna).
                """)
            else:
                st.warning(f"Twój trening jest krótszy niż {window_sec/60:.0f} min, więc nie możemy wyznaczyć tego okna.")
        else:
            st.error("Brakuje kluczowych danych (Watts + HR/VE/SmO2) do stworzenia radaru.")

    # --- TAB MODEL CP (PREDICTION) ---
    with tab_model:
        st.header("Matematyczny Model CP (Critical Power Estimation)")
        st.markdown("Estymacja Twojego CP i W' na podstawie krzywej mocy (MMP) z tego treningu. Używamy modelu liniowego: `Praca = CP * t + W'`.")

        if 'watts' in df_plot and len(df_plot) > 1200: # Minimum 20 minut danych
            
            # 1. Wybieramy punkty czasowe do modelu (standardowe dla modelu 2-parametrowego)
            # Unikamy bardzo krótkich czasów (< 2-3 min), bo tam dominuje Pmax/AC
            durations = [180, 300, 600, 900, 1200] # 3min, 5min, 10min, 15min, 20min
            
            # Filtrujemy czasy dłuższe niż długość treningu
            valid_durations = [d for d in durations if d < len(df_plot)]
            
            if len(valid_durations) >= 3: # Potrzebujemy min. 3 punktów do sensownej regresji
                
                mmp_values = []
                work_values = []
                
                # Liczymy MMP i Pracę dla każdego punktu
                for d in valid_durations:
                    # Rolling mean max
                    p = df_plot['watts'].rolling(window=d).mean().max()
                    if not pd.isna(p):
                        mmp_values.append(p)
                        # Praca [J] = Moc [W] * Czas [s]
                        work_values.append(p * d)
                
                # 2. Regresja Liniowa (Work vs Time)
                # Y = Work, X = Time
                # Slope = CP, Intercept = W'
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_durations, work_values)
                
                modeled_cp = slope
                modeled_w_prime = intercept
                r_squared = r_value**2

                # 3. Wyświetlenie Wyników
                c_res1, c_res2, c_res3 = st.columns(3)
                
                c_res1.metric("Estymowane CP (z pliku)", f"{modeled_cp:.0f} W", 
                              delta=f"{modeled_cp - cp_input:.0f} W vs Ustawienia",
                              help="Moc Krytyczna wyliczona z Twoich najmocniejszych odcinków w tym pliku.")
                
                c_res2.metric("Estymowane W'", f"{modeled_w_prime:.0f} J",
                              delta=f"{modeled_w_prime - w_prime_input:.0f} J vs Ustawienia",
                              help="Pojemność beztlenowa wyliczona z modelu.")
                
                c_res3.metric("Jakość Dopasowania (R²)", f"{r_squared:.4f}", 
                              delta_color="normal" if r_squared > 0.98 else "inverse",
                              help="Jak bardzo Twoje wyniki pasują do teoretycznej krzywej. >0.98 = Bardzo wiarygodne.")

                st.markdown("---")

                # 4. Wizualizacja: Krzywa MMP vs Krzywa Modelowa
                # Generujemy punkty teoretyczne dla zakresu 1 min - 30 min
                x_theory = np.arange(60, 1800, 60) # co minutę
                y_theory = [modeled_cp + (modeled_w_prime / t) for t in x_theory]
                
                # Rzeczywiste MMP z pliku dla tych samych czasów
                y_actual = []
                x_actual = []
                for t in x_theory:
                    if t < len(df_plot):
                        val = df_plot['watts'].rolling(t).mean().max()
                        y_actual.append(val)
                        x_actual.append(t)

                fig_model = go.Figure()
                
                # Rzeczywiste MMP
                fig_model.add_trace(go.Scatter(
                    x=np.array(x_actual)/60, y=y_actual,
                    mode='markers', name='Twoje MMP (Actual)',
                    marker=dict(color='#00cc96', size=8)
                ))
                
                # Model Teoretyczny
                fig_model.add_trace(go.Scatter(
                    x=x_theory/60, y=y_theory,
                    mode='lines', name=f'Model CP ({modeled_cp:.0f}W)',
                    line=dict(color='#ef553b', dash='dash')
                ))

                fig_model.update_layout(
                    template="plotly_dark",
                    title="Power Duration Curve: Rzeczywistość vs Model",
                    xaxis_title="Czas trwania [min]",
                    yaxis_title="Moc [W]",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig_model, use_container_width=True)
                
                # 5. Interpretacja
                st.info(f"""
                **📊 Interpretacja Modelu:**
                
                Ten algorytm próbuje dopasować Twoje wysiłki do fizjologicznego prawa mocy krytycznej.
                
                * **Jeśli Estymowane CP > Ustawione CP:** Brawo! W tym treningu pokazałeś, że jesteś mocniejszy niż myślisz. Rozważ aktualizację ustawień w sidebarze.
                * **Jeśli Estymowane CP < Ustawione CP:** To normalne, jeśli nie jechałeś "do odciny" (All-Out) na odcinkach 3-20 min. Model pokazuje tylko to, co *zademonstrowałeś*, a nie Twój absolutny potencjał.
                * **R² (R-kwadrat):** Jeśli jest niskie (< 0.95), oznacza to, że Twoja jazda była nieregularna i model nie może znaleźć jednej linii, która pasuje do Twoich wyników.
                """)

            else:
                st.warning("Trening jest zbyt krótki lub brakuje mocnych odcinków, by zbudować wiarygodny model CP (wymagane wysiłki > 3 min i > 10 min).")
        else:
            st.warning("Za mało danych (wymagane min. 20 minut jazdy z pomiarem mocy).")

        # --- EXPORT DO PDF (Wersja ULTIMATE v3 - Dual Zones Logic) ---
from fpdf import FPDF
import datetime

def clean_text(text):
    replacements = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z',
        '²': '2', '³': '3', '°': 'st.', '≈': '~', 'Δ': 'delta'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Pro Athlete Dashboard - Raport Fizjologiczny', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, f'Data: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.line(10, 25, 200, 25)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Strona {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, clean_text(label), 0, 1, 'L', 1)
        self.ln(2)

    def chapter_body(self, text, size=9):
        self.set_font('Arial', '', size)
        self.multi_cell(0, 5, clean_text(text))
        self.ln(3)

# UI Exportu
st.sidebar.markdown("---")
st.sidebar.header("🖨️ Export Danych")

if 'df_plot' in locals() and uploaded_file is not None:
    
    def generate_pdf():
        pdf = PDFReport()
        pdf.add_page()
        
        # --- DANE ---
        duration_sec = len(df_plot)
        avg_w = df_plot['watts'].mean() if 'watts' in df_plot else 0
        max_w = df_plot['watts'].max() if 'watts' in df_plot else 0
        
        if 'watts' in df_plot:
            rolling_30s = df_plot['watts'].rolling(window=30, min_periods=1).mean()
            calc_np = np.power(np.mean(np.power(rolling_30s, 4)), 0.25)
            if pd.isna(calc_np): calc_np = avg_w
            calc_if = calc_np / cp_input if cp_input > 0 else 0
            calc_tss = (duration_sec * calc_np * calc_if) / (cp_input * 3600) * 100
        else:
            calc_np, calc_if, calc_tss = 0, 0, 0

        # --- 1. PROFIL ---
        pdf.chapter_title("1. Parametry Profilu")
        p_txt = f"""Zawodnik: Waga {rider_weight} kg | Wzrost {rider_height} cm
Progi Fizjologiczne (Metabolizm): VT1 {vt1_watts} W | VT2 {vt2_watts} W
Ustawienia Mechaniczne (Moc): CP {cp_input} W | W' {w_prime_input} J"""
        pdf.chapter_body(p_txt)

        # --- 2. LOAD ---
        work_kj = (df_plot['watts'].sum() / 1000) if 'watts' in df_plot else 0
        work_above_cp = metrics.get('work_above_cp_kj', 0)
        
        pdf.chapter_title("2. Obciazenie (Load)")
        load_txt = f"""Czas: {fmt_time(duration_sec)}
NP: {calc_np:.0f} W | Avg: {avg_w:.0f} W | IF: {calc_if:.2f} | TSS: {calc_tss:.0f}
Praca Calkowita: {work_kj:.0f} kJ
Praca Beztlenowa: {work_above_cp:.0f} kJ"""
        pdf.chapter_body(load_txt)

        # --- 3. FIZJOLOGIA (HSI v3) ---
        pdf.chapter_title("3. Fizjologia: Hemodynamika i Termoregulacja")
        
        # Temp metric (Active 95%)
        temp_metric = 0.0
        if 'core_temperature' in df_plot:
            working_df = df_plot[df_plot['watts'] > 100]
            if working_df.empty: working_df = df_plot
            temp_metric = working_df['core_temperature'].quantile(0.95)
        
        # HSI Logic v3 (Progresywna)
        hsi_val = 0.0
        if temp_metric < 38.0: hsi_val = 0.0
        elif 38.0 <= temp_metric <= 38.5:
            ratio = (temp_metric - 38.0) / 0.5
            hsi_val = 1.0 + (ratio * 3.0) # 1-4
        elif 38.5 < temp_metric <= 39.5:
            ratio = (temp_metric - 38.5) / 1.0
            hsi_val = 5.0 + (ratio * 3.0) # 5-8
        elif 39.5 < temp_metric <= 41.0:
            ratio = (temp_metric - 39.5) / 1.5
            hsi_val = 8.0 + (ratio * 2.0) # 8-10
        else: hsi_val = 10.0
            
        if hsi_val >= 8.0: hsi_msg = "KRYTYCZNE (Ryzyko udaru)"
        elif hsi_val >= 5.0: hsi_msg = "Wysokie (Spadek wydajnosci)"
        elif hsi_val >= 1.0: hsi_msg = "Umiarkowane (Strefa Robocza)"
        else: hsi_msg = "Niskie (Komfort)"

        therm_txt = "Brak Core."
        if 'core_temperature' in df_plot:
            avg_temp = df_plot['core_temperature'].mean()
            max_temp = df_plot['core_temperature'].quantile(0.99)
            time_optimal = len(df_plot[(df_plot['core_temperature'] >= 37.5) & (df_plot['core_temperature'] <= 38.4)])
            time_hot = len(df_plot[df_plot['core_temperature'] > 38.5])
            time_crit = len(df_plot[df_plot['core_temperature'] > 39.0])
            
            therm_txt = f"""Avg Temp: {avg_temp:.2f} C | Max Temp: {max_temp:.2f} C
Czas w strefie OPTIMAL (37.5-38.4 C): {fmt_time(time_optimal)}
Czas w strefie HOT (>38.5 C): {fmt_time(time_hot)}
Czas w strefie CRITICAL (>39.0 C): {fmt_time(time_crit)}

Heat Strain Index (Custom Logic): {hsi_val:.1f} / 10
Interpretacja: {hsi_msg}"""

        smo2_txt = "Brak SmO2."
        if 'smo2' in df_plot:
            avg_smo2 = df_plot['smo2'].mean()
            min_smo2 = df_plot['smo2'].min()
            p5_smo2 = df_plot['smo2'].quantile(0.05)
            smo2_txt = f"Avg SmO2: {avg_smo2:.1f}% | Min SmO2: {min_smo2:.1f}% | 5-ty Percentyl: {p5_smo2:.1f}%"
            if 'thb' in df_plot:
                avg_thb = df_plot['thb'].mean()
                smo2_txt += f"\nTHb (Hemo): Srednie {avg_thb:.2f} a.u."

        phys_body = f"""TETNO (HR):
Avg: {metrics.get('avg_hr',0):.0f} bpm | Max: {df_plot['heartrate'].max() if 'heartrate' in df_plot else 0:.0f} bpm

OKSYDACJA MIESNIOWA (SmO2):
{smo2_txt}

TERMOREGULACJA (Core Temp):
{therm_txt}

REZERWY BEZTLENOWE (W'):
Min W' Balance: {df_plot['w_prime_balance'].min():.0f} J"""
        pdf.chapter_body(phys_body)

        # --- 4. PALIWO (Tylda Fix) ---
        pdf.chapter_title("4. Metabolizm (Paliwo)")
        if 'watts' in df_plot:
            kcals_series = (df_plot['watts'] / 0.22) / 4184 
            total_kcal = kcals_series.sum()
            cho_fraction = pd.Series(0.40, index=df_plot.index)
            cho_fraction[df_plot['watts'] >= vt1_watts] = 0.75
            cho_fraction[df_plot['watts'] >= vt2_watts] = 1.00
            cho_g = (kcals_series * cho_fraction).sum() / 4.0
            fat_g = (kcals_series * (1.0 - cho_fraction)).sum() / 9.0
            
            # Zamiana tyldy na 'ok.'
            nutri_txt = f"""Wydatek: ok. {total_kcal:.0f} kcal
Spalone Weglowodany: {cho_g:.0f} g ({cho_g/(duration_sec/3600):.0f} g/h)
Spalone Tluszcze: {fat_g:.0f} g ({fat_g/(duration_sec/3600):.0f} g/h)"""
        else:
            nutri_txt = "Brak mocy."
        pdf.chapter_body(nutri_txt)

        # --- 5. STREFY FIZJOLOGICZNE (VT) ---
        pdf.chapter_title("5. Strefy Fizjologiczne (Metaboliczne)")
        if 'watts' in df_plot:
            t_z1 = len(df_plot[df_plot['watts'] < vt1_watts])
            t_z2 = len(df_plot[(df_plot['watts'] >= vt1_watts) & (df_plot['watts'] < vt2_watts)])
            t_z3 = len(df_plot[df_plot['watts'] >= vt2_watts])
            total_t = len(df_plot)
            p1 = t_z1/total_t*100 if total_t>0 else 0
            p2 = t_z2/total_t*100 if total_t>0 else 0
            p3 = t_z3/total_t*100 if total_t>0 else 0

            zones_phys = f"""Strefa Tlenowa (<VT1 {vt1_watts}W):
Czas: {fmt_time(t_z1)} ({p1:.1f}%) -> Regeneracja i Baza

Strefa Mieszana (VT1-VT2):
Czas: {fmt_time(t_z2)} ({p2:.1f}%) -> Tempo i Sweet Spot

Strefa Beztlenowa (>VT2 {vt2_watts}W):
Czas: {fmt_time(t_z3)} ({p3:.1f}%) -> VO2Max i Anaerobic"""
            pdf.chapter_body(zones_phys)

        # --- 6. STREFY MOCY (Coggan / FTP) - PRZYWRÓCONE ---
        pdf.chapter_title("6. Strefy Mocy (Coggan / %CP)")
        if 'watts' in df_plot:
            bins = [0, 0.55*cp_input, 0.75*cp_input, 0.90*cp_input, 1.05*cp_input, 1.20*cp_input, 10000]
            labels = ['Z1 (Active Recovery)', 'Z2 (Endurance)', 'Z3 (Tempo)', 'Z4 (Threshold)', 'Z5 (VO2Max)', 'Z6 (Anaerobic)']
            
            counts = []
            for i in range(len(labels)):
                low = bins[i]
                high = bins[i+1]
                c = len(df_plot[(df_plot['watts'] >= low) & (df_plot['watts'] < high)])
                counts.append(c)
            
            total_t = sum(counts)
            z_txt = ""
            for lab, c in zip(labels, counts):
                pct = c / total_t * 100 if total_t > 0 else 0
                z_txt += f"{lab}: {fmt_time(c)} ({pct:.1f}%)\n"
            
            pdf.chapter_body(z_txt)
        
        # --- 7. MMP ---
        pdf.chapter_title("7. Profil Mocy (MMP)")
        mmp_txt = ""
        for sec, lab in [(5, '5s'), (60, '1m'), (300, '5m'), (1200, '20m'), (3600, '60m')]:
            if len(df_plot) > sec:
                val = df_plot['watts'].rolling(sec).mean().max()
                mmp_txt += f"{lab}: {val:.0f} W ({val/rider_weight:.2f} W/kg) | "
        pdf.chapter_body(mmp_txt)

        return pdf.output(dest='S').encode('latin-1', 'replace')

    pdf_bytes = generate_pdf()
    
    st.sidebar.download_button(
        label="📥 Pobierz Raport PDF (Dual Zones)",
        data=pdf_bytes,
        file_name=f"raport_dual_{uploaded_file.name.split('.')[0]}.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.info("Wgraj plik.")
