import streamlit as st
import pandas as pd
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
from io import BytesIO
from scipy import stats
import neurokit2 as nk
from pathlib import Path
import os
import zipfile
import json
from typing import Dict, Any, Optional, Tuple

# --- MODULE IMPORTS ---
from modules.config import Config
from modules.utils import parse_time_input, load_data, _serialize_df_to_parquet_bytes, normalize_columns_pandas
from modules.calculations import (
    calculate_w_prime_balance,
    calculate_metrics,
    calculate_dynamic_dfa,
    calculate_advanced_kpi,
    calculate_z2_drift,
    calculate_heat_strain_index,
    calculate_vo2max,
    calculate_normalized_power,
    estimate_carbs_burned,
    calculate_trend,
    process_data
)
from modules.plots import apply_chart_style, add_stats_to_legend
from modules.ml_logic import (
    MLX_AVAILABLE, train_cycling_brain, predict_only, MODEL_FILE, HISTORY_FILE
)
from modules.intervals import detect_intervals
from modules.notes import TrainingNotes
from modules.reports import generate_docx_report, export_all_charts_as_png

# --- SERVICES IMPORTS (Business Logic Layer) ---
from services import (
    calculate_header_metrics,
    calculate_extended_metrics,
    apply_smo2_smoothing,
    resample_dataframe,
    validate_dataframe,
    prepare_session_record,
    prepare_sticky_header_data,
    ROLLING_WINDOW_5MIN,
    ROLLING_WINDOW_30S,
    ROLLING_WINDOW_60S,
    RESAMPLE_THRESHOLD,
    RESAMPLE_STEP,
    MIN_WATTS_ACTIVE,
    MIN_HR_ACTIVE,
    MIN_RECORDS_FOR_ROLLING,
)

# UI Modules - Lazy Loading for faster startup
# Each function only imports its module when actually called
def render_report_tab(*args, **kwargs):
    from modules.ui.report import render_report_tab as _render
    return _render(*args, **kwargs)

def render_kpi_tab(*args, **kwargs):
    from modules.ui.kpi import render_kpi_tab as _render
    return _render(*args, **kwargs)

def render_power_tab(*args, **kwargs):
    from modules.ui.power import render_power_tab as _render
    return _render(*args, **kwargs)

def render_intervals_tab(*args, **kwargs):
    from modules.ui.intervals_ui import render_intervals_tab as _render
    return _render(*args, **kwargs)

def render_hrv_tab(*args, **kwargs):
    from modules.ui.hrv import render_hrv_tab as _render
    return _render(*args, **kwargs)

def render_biomech_tab(*args, **kwargs):
    from modules.ui.biomech import render_biomech_tab as _render
    return _render(*args, **kwargs)

def render_thermal_tab(*args, **kwargs):
    from modules.ui.thermal import render_thermal_tab as _render
    return _render(*args, **kwargs)

def render_trends_tab(*args, **kwargs):
    from modules.ui.trends import render_trends_tab as _render
    return _render(*args, **kwargs)

def render_nutrition_tab(*args, **kwargs):
    from modules.ui.nutrition import render_nutrition_tab as _render
    return _render(*args, **kwargs)

def render_smo2_tab(*args, **kwargs):
    from modules.ui.smo2 import render_smo2_tab as _render
    return _render(*args, **kwargs)

def render_hemo_tab(*args, **kwargs):
    from modules.ui.hemo import render_hemo_tab as _render
    return _render(*args, **kwargs)

def render_vent_tab(*args, **kwargs):
    from modules.ui.vent import render_vent_tab as _render
    return _render(*args, **kwargs)

def render_limiters_tab(*args, **kwargs):
    from modules.ui.limiters import render_limiters_tab as _render
    return _render(*args, **kwargs)

def render_ai_coach_tab(*args, **kwargs):
    from modules.ui.ai_coach import render_ai_coach_tab as _render
    return _render(*args, **kwargs)

def render_model_tab(*args, **kwargs):
    from modules.ui.model import render_model_tab as _render
    return _render(*args, **kwargs)

# ===== NEW FEATURE TABS =====
def render_trends_history_tab(*args, **kwargs):
    from modules.ui.trends_history import render_trends_history_tab as _render
    return _render(*args, **kwargs)

def render_community_tab(*args, **kwargs):
    from modules.ui.community import render_community_tab as _render
    return _render(*args, **kwargs)
def render_pdc_tab(*args, **kwargs):
    from modules.ui.pdc_ui import render_pdc_tab as _render
    return _render(*args, **kwargs)

def render_history_import_tab(*args, **kwargs):
    from modules.ui.history_import_ui import render_history_import_tab as _render
    return _render(*args, **kwargs)

def render_threshold_analysis_tab(*args, **kwargs):
    from modules.ui.threshold_analysis_ui import render_threshold_analysis_tab as _render
    return _render(*args, **kwargs)

from modules.comparison import render_compare_dashboard
from modules.settings import SettingsManager
from modules.health_alerts import HealthMonitor
from modules.db import SessionStore, SessionRecord
from modules.ai.interval_detector import IntervalDetector


def cleanup_session_state() -> None:
    """Clean up old DataFrames from session state to free memory."""
    keys_to_check = ['_prev_df_plot', '_prev_df_resampled', '_prev_file_name', 'data_loaded']
    for key in keys_to_check:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title="Pro Athlete Dashboard", layout="wide", page_icon="⚡")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

# Inicjalizacja notatek
training_notes = TrainingNotes()


# --- SETTINGS / SESSION STATE INIT ---
settings_manager = SettingsManager()
saved_settings = settings_manager.load_settings()

# Mapowanie kluczy UI na klucze w JSON
keys_map = {
    "weight": "rider_weight",
    "height": "rider_height", 
    "age": "rider_age",
    "gender_m": "is_male",
    "vt1_w": "vt1_watts",
    "vt2_w": "vt2_watts",
    "vt1_v": "vt1_vent",
    "vt2_v": "vt2_vent",
    "cp_in": "cp",
    "wp_in": "w_prime",
    "crank": "crank_length"
}

# Inicjalizacja Session State z pliku (tylko raz na sesje, lub jeśli brakuje)
for ui_key, json_key in keys_map.items():
    if ui_key not in st.session_state:
        st.session_state[ui_key] = saved_settings.get(json_key)

def save_settings_callback():
    """Zapisuje aktualne wartości z UI do pliku JSON."""
    current_values = {}
    for ui_key, json_key in keys_map.items():
        if ui_key in st.session_state:
            current_values[json_key] = st.session_state[ui_key]
    settings_manager.save_settings(current_values)
    # Opcjonalnie: st.toast("Ustawienia zapisane!", icon="💾")

# --- APP START ---

st.title("⚡ Pro Athlete Dashboard")

st.sidebar.header("Ustawienia Zawodnika")
with st.sidebar.expander("⚙️ Parametry Fizyczne", expanded=True):
    rider_weight = st.number_input("Waga Zawodnika [kg]", step=0.5, min_value=30.0, max_value=200.0, key="weight", on_change=save_settings_callback)
    rider_height = st.number_input("Wzrost [cm]", step=1, min_value=100, max_value=250, key="height", on_change=save_settings_callback)
    rider_age = st.number_input("Wiek [lata]", step=1, min_value=10, max_value=100, key="age", on_change=save_settings_callback)
    is_male = st.checkbox("Mężczyzna?", key="gender_m", on_change=save_settings_callback)
    
    st.markdown("---")
    vt1_watts = st.number_input("VT1 (Próg Tlenowy) [W]", min_value=0, value=0, key="vt1_w", on_change=save_settings_callback)
    vt2_watts = st.number_input("VT2 (Próg Beztlenowy/FTP) [W]", min_value=0, value=0, key="vt2_w", on_change=save_settings_callback)
    st.divider()

    st.markdown("### 🫁 Wentylacja [L/min]")
    vt1_vent = st.number_input("VT1 (Próg Tlenowy) [L/min]", min_value=0.0, key="vt1_v", on_change=save_settings_callback)
    vt2_vent = st.number_input("VT2 (Próg Beztlenowy) [L/min]", min_value=0.0, key="vt2_v", on_change=save_settings_callback)

st.sidebar.divider()
cp_input = st.sidebar.number_input("Moc Krytyczna (CP) [W]", min_value=1, key="cp_in", on_change=save_settings_callback)
w_prime_input = st.sidebar.number_input("W' (W Prime) [J]", min_value=0, key="wp_in", on_change=save_settings_callback)
st.sidebar.divider()
crank_length = st.sidebar.number_input("Długość korby [mm]", key="crank", on_change=save_settings_callback)
compare_mode = st.sidebar.toggle("⚔️ Tryb Porównania (Beta)", value=False)
uploaded_file = None

if compare_mode:
    file1 = st.sidebar.file_uploader("Wgraj Plik A (CSV)", type=['csv'])
    file2 = st.sidebar.file_uploader("Wgraj Plik B (CSV)", type=['csv'])

    if file1 and file2:
        from modules.comparison import render_compare_dashboard
        render_compare_dashboard(file1, file2, cp_input)
        st.stop()
    else:
        st.info("Wgraj dwa pliki, aby rozpocząć porównanie.")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV / TXT)", type=['csv', 'txt'])

if rider_weight <= 0 or cp_input <= 0:
    st.error("Błąd: Waga i CP muszą być większe od zera.")
    st.stop()

if uploaded_file is not None:
    # Clean up old data when new file is uploaded
    cleanup_session_state()
    
    with st.spinner('Przetwarzanie danych...'):
        try:
            df_raw = load_data(uploaded_file)
            
            # VALIDATION: Use services layer
            is_valid, error_msg = validate_dataframe(df_raw)
            if not is_valid:
                st.error(f"Błąd walidacji danych: {error_msg}")
                st.stop()
            
            # PROCESSING: Use modules.calculations for base processing
            df_clean_pl = process_data(df_raw)
            metrics = calculate_metrics(df_clean_pl, cp_input)
            df_w_prime = calculate_w_prime_balance(df_clean_pl, cp_input, w_prime_input)
            decoupling_percent, ef_factor = calculate_advanced_kpi(df_clean_pl)
            drift_z2 = calculate_z2_drift(df_clean_pl, cp_input)
            df_with_hsi = calculate_heat_strain_index(df_w_prime)
            df_plot = df_with_hsi

            # EXTENDED METRICS: Use services layer
            metrics = calculate_extended_metrics(
                df_plot, metrics, rider_weight, vt1_watts, vt2_watts, ef_factor
            )
            
            # SMO2 SMOOTHING: Use services layer
            df_plot = apply_smo2_smoothing(df_plot)
            
            # RESAMPLING: Use services layer
            df_plot_resampled = resample_dataframe(df_plot)
            
            st.session_state['data_loaded'] = True
            
            # --- SEKCJA AI / MLX ---
            if MLX_AVAILABLE and os.path.exists(MODEL_FILE):
                try:
                    auto_pred = predict_only(df_plot_resampled)
                    
                    if auto_pred is not None:
                        df_plot_resampled['ai_hr'] = auto_pred
                    else:
                        st.sidebar.warning("⚠️ AI zwróciło pusty wynik (None). Sprawdź load_model.")
                except Exception as e:
                    st.sidebar.error(f"💥 Krytyczny błąd w Auto-Inference: {e}")
            elif not os.path.exists(MODEL_FILE):
                pass 
            # ----------------------------------

        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {e}")
            st.stop()

        # HEADER METRICS: Use services layer
        np_header, if_header, tss_header = calculate_header_metrics(df_plot, cp_input)

        # ===== AUTO-SAVE SESSION TO DATABASE =====
        try:
            session_data = prepare_session_record(
                uploaded_file.name, df_plot, metrics, np_header, if_header, tss_header
            )
            session_record = SessionRecord(**session_data)
            SessionStore().add_session(session_record)
        except Exception as e:
            pass  # Silent fail for session save

        # ===== HEALTH ALERTS =====
        health_monitor = HealthMonitor()
        alerts = health_monitor.analyze_session(df_plot, metrics)
        
        if alerts:
            for alert in alerts[:3]:  # Show max 3 alerts
                if alert.severity == "critical":
                    st.error(f"{alert.icon} **{alert.message}**\n\n{alert.recommendation}")
                elif alert.severity == "warning":
                    st.warning(f"{alert.icon} **{alert.message}**\n\n{alert.recommendation}")
                else:
                    st.info(f"{alert.icon} {alert.message}")

        # ===== STICKY HEADER =====
        header_data = prepare_sticky_header_data(df_plot, metrics)

        st.markdown(f"""
        <div class="sticky-metrics">
            <h4>⚡ Live Training Summary</h4>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Avg Power</div>
                    <div class="value">{header_data['avg_power']:.0f} <span class="unit">W</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg HR</div>
                    <div class="value">{header_data['avg_hr']:.0f} <span class="unit">bpm</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg SmO2</div>
                    <div class="value">{header_data['avg_smo2']:.1f} <span class="unit">%</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Cadence</div>
                    <div class="value">{header_data['avg_cadence']:.0f} <span class="unit">rpm</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Avg VE</div>
                    <div class="value">{header_data['avg_ve']:.0f} <span class="unit">L/min</span></div>
                </div>
                <div class="metric-box">
                    <div class="label">Duration</div>
                    <div class="value">{header_data['duration_min']:.0f} <span class="unit">min</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # ===== KONIEC STICKY HEADER =====

        m1, m2, m3 = st.columns(3)
        m1.metric("NP (Norm. Power)", f"{np_header:.0f} W", help="Normalized Power (Coggan Formula)")
        m2.metric("TSS", f"{tss_header:.0f}", help=f"IF: {if_header:.2f}")
        m3.metric("Praca [kJ]", f"{df_plot['watts'].sum()/1000:.0f}")
        
        # --- ZAKŁADKI (Pogrupowane dla lepszej przejrzystości) ---
        
        # Helper dla breadcrumbs
        def show_breadcrumb(group: str, section: str = None):
            """Wyświetla breadcrumb nawigację."""
            if section:
                st.markdown(f'''
                <div class="breadcrumb-nav">
                    🏠 Dashboard <span class="separator">›</span> 
                    {group} <span class="separator">›</span> 
                    <span class="current">{section}</span>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="breadcrumb-nav">
                    🏠 Dashboard <span class="separator">›</span> 
                    <span class="current">{group}</span>
                </div>
                ''', unsafe_allow_html=True)
        
        tab_overview, tab_performance, tab_physiology, tab_intelligence, tab_analytics = st.tabs([
            "📊 Overview", "⚡ Performance", "🫀 Physiology", "🧠 Intelligence", "📈 Analytics"
        ])

        # ===== 📊 OVERVIEW =====
        with tab_overview:
            show_breadcrumb("📊 Overview")
            sub_raport, sub_kpi, sub_trends = st.tabs([
                "📋 Raport", "📈 KPI", "📉 Trends"
            ])
            with sub_raport:
                render_report_tab(df_plot, rider_weight, cp_input)
            with sub_kpi:
                render_kpi_tab(df_plot, df_plot_resampled, metrics, rider_weight, decoupling_percent, drift_z2, vt1_vent, vt2_vent)
            with sub_trends:
                render_trends_tab(df_plot)

        # ===== ⚡ PERFORMANCE =====
        with tab_performance:
            show_breadcrumb("⚡ Performance")
            sub_power, sub_pdc, sub_intervals, sub_biomech, sub_model = st.tabs([
                "🔋 Power", "📊 PDC", "⏱️ Intervals", "🦵 Biomech", "📐 Model"
            ])
            with sub_power:
                render_power_tab(df_plot, df_plot_resampled, cp_input, w_prime_input)
            with sub_pdc:
                vo2max_val = metrics.get('vo2_max_est', 0)
                render_pdc_tab(df_plot, cp_input, w_prime_input, rider_weight, vo2max_val)
            with sub_intervals:
                render_intervals_tab(df_plot, df_plot_resampled, cp_input, rider_weight, rider_age, is_male)
            with sub_biomech:
                render_biomech_tab(df_plot, df_plot_resampled)
            with sub_model:
                render_model_tab(df_plot, cp_input, w_prime_input)

        # ===== 🫀 PHYSIOLOGY =====
        with tab_physiology:
            show_breadcrumb("🫀 Physiology")
            sub_hrv, sub_smo2, sub_hemo, sub_vent, sub_thermal = st.tabs([
                "💓 HRV", "🩸 SmO2", "🧬 Hematology", "🫁 Ventilation", "🌡️ Thermal"
            ])
            with sub_hrv:
                render_hrv_tab(df_clean_pl)
            with sub_smo2:
                render_smo2_tab(df_plot, training_notes, uploaded_file.name)
            with sub_hemo:
                render_hemo_tab(df_plot)
            with sub_vent:
                render_vent_tab(df_plot, training_notes, uploaded_file.name)
            with sub_thermal:
                render_thermal_tab(df_plot)

        # ===== 🧠 INTELLIGENCE =====
        with tab_intelligence:
            show_breadcrumb("🧠 Intelligence")
            sub_nutrition, sub_limiters, sub_ai, sub_thresholds = st.tabs([
                "🍎 Nutrition", "🚧 Limiters", "🤖 AI Coach", "🎯 Progi & Plan"
            ])
            with sub_nutrition:
                render_nutrition_tab(df_plot, cp_input, vt1_watts, vt2_watts)
            with sub_limiters:
                render_limiters_tab(df_plot, cp_input, vt2_vent)
            with sub_ai:
                render_ai_coach_tab(df_plot_resampled)
            with sub_thresholds:
                max_hr = int(208 - 0.7 * rider_age) if rider_age else 185
                render_threshold_analysis_tab(df_plot, training_notes, uploaded_file.name, cp_input, cp_input, max_hr)

        # ===== 📈 ANALYTICS (NOWA GRUPA) =====
        with tab_analytics:
            show_breadcrumb("📈 Analytics")
            sub_history, sub_community, sub_import = st.tabs([
                "📊 History", "👥 Community", "📂 Import"
            ])
            with sub_history:
                render_trends_history_tab()
            with sub_community:
                vo2max_val = metrics.get('vo2_max_est', 0) if 'metrics' in dir() else 0
                render_community_tab(cp_input, rider_weight, vo2max_val, rider_age, 'M' if is_male else 'F')
            with sub_import:
                render_history_import_tab(cp_input)

# ===== DOCX EXPORT BUTTON =====
st.sidebar.markdown("---")
st.sidebar.header("📄 Export Raportu")

if st.session_state.get('data_loaded', False) and uploaded_file is not None:
    # Kolumny dla przycisków
    col_docx, col_png = st.sidebar.columns(2)
    
    with col_docx:
        # Generuj DOCX
        try:
            docx_doc = generate_docx_report(
                metrics, df_plot, df_plot_resampled, uploaded_file, cp_input,
                vt1_watts, vt2_watts, rider_weight, vt1_vent, vt2_vent, w_prime_input
            )
            
            # Zapisz do BytesIO
            docx_buffer = BytesIO()
            docx_doc.save(docx_buffer)
            docx_buffer.seek(0)
            
            st.sidebar.download_button(
                label="📥 Pobierz Raport DOCX",
                data=docx_buffer.getvalue(),
                file_name=f"Raport_{uploaded_file.name.split('.')[0]}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        except Exception as e:
            st.sidebar.error(f"Błąd DOCX: {e}")
    

    with col_png:
    # Generuj PNG ZIP
        try:
            png_zip = export_all_charts_as_png(
                df_plot, df_plot_resampled, cp_input, vt1_watts, vt2_watts,
                metrics, rider_weight, uploaded_file,
                st.session_state.get('vent_start_sec'), st.session_state.get('vent_end_sec'),
                st.session_state.get('smo2_start_sec'), st.session_state.get('smo2_end_sec')
            )
            
            st.sidebar.download_button(
                label="📸 Pobierz Wykresy PNG (ZIP)",
                data=png_zip,
                file_name=f"Wykresy_{uploaded_file.name.split('.')[0]}.zip",
                mime="application/zip",
                use_container_width=True
            )
        except Exception as e:
            st.sidebar.error(f"Błąd PNG: {e}")

        
else:
    st.sidebar.info("Wgraj plik aby pobrać raport.")
# ===== KONIEC DOCX =====
