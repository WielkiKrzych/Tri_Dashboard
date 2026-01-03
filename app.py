import streamlit as st
import io
from io import BytesIO
import os

# --- FRONTEND IMPORTS ---
from modules.frontend.theme import ThemeManager
from modules.frontend.state import StateManager
from modules.frontend.layout import AppLayout
from modules.frontend.components import UIComponents

# --- MODULE IMPORTS ---
from modules.config import Config
from modules.utils import load_data
from modules.calculations import (
    calculate_w_prime_balance,
    calculate_metrics,
    calculate_advanced_kpi,
    calculate_z2_drift,
    calculate_heat_strain_index,
)
from modules.ml_logic import (
    MLX_AVAILABLE, predict_only, MODEL_FILE
)
from modules.notes import TrainingNotes
from modules.reports import generate_docx_report, export_all_charts_as_png
from modules.db import SessionStore, SessionRecord
from modules.health_alerts import HealthMonitor
from modules.reporting.persistence import check_git_tracking

# --- SERVICES IMPORTS ---
from services import (
    calculate_header_metrics,
    calculate_extended_metrics,
    apply_smo2_smoothing,
    resample_dataframe,
    validate_dataframe,
    prepare_session_record,
    prepare_sticky_header_data
)

# --- TAB REGISTRY (OCP) ---
class TabRegistry:
    """Registry for UI tabs to support Open/Closed Principle."""
    _tabs = {
        "report": ("modules.ui.report", "render_report_tab"),
        "kpi": ("modules.ui.kpi", "render_kpi_tab"),
        "power": ("modules.ui.power", "render_power_tab"),
        "intervals": ("modules.ui.intervals_ui", "render_intervals_tab"),
        "biomech": ("modules.ui.biomech", "render_biomech_tab"),
        "model": ("modules.ui.model", "render_model_tab"),
        "hrv": ("modules.ui.hrv", "render_hrv_tab"),
        "smo2": ("modules.ui.smo2", "render_smo2_tab"),
        "hemo": ("modules.ui.hemo", "render_hemo_tab"),
        "vent": ("modules.ui.vent", "render_vent_tab"),
        "vent_thresholds": ("modules.ui.vent_thresholds", "render_vent_thresholds_tab"),
        "smo2_thresholds": ("modules.ui.smo2_thresholds", "render_smo2_thresholds_tab"),
        "thermal": ("modules.ui.thermal", "render_thermal_tab"),
        "nutrition": ("modules.ui.nutrition", "render_nutrition_tab"),
        "limiters": ("modules.ui.limiters", "render_limiters_tab"),
        "ai_coach": ("modules.ui.ai_coach", "render_ai_coach_tab"),
        "thresholds": ("modules.ui.threshold_analysis_ui", "render_threshold_analysis_tab"),
        "history": ("modules.ui.trends_history", "render_trends_history_tab"),
        "community": ("modules.ui.community", "render_community_tab"),
        "import": ("modules.ui.history_import_ui", "render_history_import_tab"),
        "heart_rate": ("modules.ui.heart_rate", "render_hr_tab"),
        "manual_thresholds": ("modules.ui.manual_thresholds", "render_manual_thresholds_tab"),
        "smo2_manual_thresholds": ("modules.ui.smo2_manual_thresholds", "render_smo2_manual_thresholds_tab"),
        "summary": ("modules.ui.summary", "render_summary_tab"),
        "drift_maps": ("modules.ui.drift_maps_ui", "render_drift_maps_tab"),
        "tte": ("modules.ui.tte_ui", "render_tte_tab"),
        "ramp_archive": ("modules.ui.ramp_archive", "render_ramp_archive"),
    }

    @classmethod
    def render(cls, tab_name, *args, **kwargs):
        """Dynamic dispatcher for tab rendering (Lazy loading)."""
        if tab_name not in cls._tabs:
            st.error(f"Unknown tab: {tab_name}")
            return
        
        module_path, func_name = cls._tabs[tab_name]
        try:
            import importlib
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error loading tab {tab_name}: {e}")

def render_tab_content(tab_name, *args, **kwargs):
    """Facade for TabRegistry."""
    return TabRegistry.render(tab_name, *args, **kwargs)

# --- INIT ---
ThemeManager.set_page_config()
ThemeManager.load_css()

state = StateManager()
state.init_session_state()

# Safety Check: Git Tracking of sensitive data (reports & raw CSVs)
check_git_tracking("reports/ramp_tests")
check_git_tracking("treningi_csv")

layout = AppLayout(state)
uploaded_file, params = layout.render_sidebar()

# Parameters shorthand
rider_weight = params.get('rider_weight', 75.0)
cp_input = params.get('cp', 280)
vt1_watts = params.get('vt1_watts', 0)
vt2_watts = params.get('vt2_watts', 0)
vt1_vent = params.get('vt1_vent', 0)
vt2_vent = params.get('vt2_vent', 0)
w_prime_input = params.get('w_prime', 20000)
rider_age = params.get('rider_age', 30)
is_male = params.get('is_male', True)

layout.render_header()

if params.get('compare_mode'):
    if isinstance(uploaded_file, tuple):
        from modules.comparison import render_compare_dashboard
        render_compare_dashboard(uploaded_file[0], uploaded_file[1], cp_input)
        st.stop()
    else:
        st.stop()

if rider_weight <= 0 or cp_input <= 0:
    st.error("Błąd: Waga i CP muszą być większe od zera.")
    st.stop()

if uploaded_file is not None:
    state.cleanup_old_data()
    training_notes = TrainingNotes()

    with st.spinner('Przetwarzanie danych...'):
        try:
            df_raw = load_data(uploaded_file)
            
            # --- SESSION TYPE CLASSIFICATION (MUST run first) ---
            from modules.domain import (
                SessionType, classify_session_type, 
                classify_ramp_test, RAMP_CONFIDENCE_THRESHOLD
            )
            session_type = classify_session_type(df_raw, uploaded_file.name)
            st.session_state['session_type'] = session_type
            
            # Store detailed ramp classification for gating decisions
            ramp_classification = None
            if 'watts' in df_raw.columns or 'power' in df_raw.columns:
                power_col = 'watts' if 'watts' in df_raw.columns else 'power'
                power = df_raw[power_col].dropna()
                if len(power) >= 300:
                    ramp_classification = classify_ramp_test(power)
                    st.session_state['ramp_classification'] = ramp_classification
            
            # --- PROCESSING PIPELINE (SRP/DIP) ---
            from services.session_orchestrator import process_uploaded_session
            
            df_plot, df_plot_resampled, metrics, error_msg = process_uploaded_session(
                df_raw, cp_input, w_prime_input, rider_weight, vt1_watts, vt2_watts
            )
            
            if error_msg:
                st.error(f"Błąd analizy: {error_msg}")
                st.stop()

            # Extract intermediate results from metrics (DIP: metrics acts as a container here)
            decoupling_percent = metrics.pop('_decoupling_percent', 0.0)
            drift_z2 = metrics.pop('_drift_z2', 0.0)
            df_clean_pl = metrics.pop('_df_clean_pl', df_raw)

            state.set_data_loaded()

            # AI Section (Optional/Non-critical)
            if MLX_AVAILABLE and os.path.exists(MODEL_FILE):
                try:
                    auto_pred = predict_only(df_plot_resampled)
                    if auto_pred is not None:
                        df_plot_resampled['ai_hr'] = auto_pred
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {e}")
            st.stop()

    # --- RENDER DASHBOARD ---
    
    # 1. Header Metrics
    np_header, if_header, tss_header = calculate_header_metrics(df_plot, cp_input)
    
    # Auto-save
    try:
        session_data = prepare_session_record(
            uploaded_file.name, df_plot, metrics, np_header, if_header, tss_header
        )
        SessionStore().add_session(SessionRecord(**session_data))
    except Exception:
        pass  # Auto-save is non-critical

    # Sticky Header
    header_data = prepare_sticky_header_data(df_plot, metrics)
    UIComponents.render_sticky_header(header_data)

    m1, m2, m3 = st.columns(3)
    m1.metric("NP (Norm. Power)", f"{np_header:.0f} W")
    m2.metric("TSS", f"{tss_header:.0f}", help=f"IF: {if_header:.2f}")
    m3.metric("Praca [kJ]", f"{df_plot['watts'].sum()/1000:.0f}")

    # Session Type Badge
    session_type = st.session_state.get('session_type')
    if session_type:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(100,100,255,0.2), transparent); 
                    padding: 8px 15px; border-radius: 8px; margin-bottom: 10px; display: inline-block;">
            <span style="font-size: 1.1em;">{session_type.emoji} <b>Typ sesji:</b> {session_type}</span>
        </div>
        """, unsafe_allow_html=True)

    # Layout Tabs
    tab_overview, tab_performance, tab_intelligence, tab_physiology = st.tabs([
        "📊 Overview", "⚡ Performance", "🧠 Intelligence", "🫀 Physiology"
    ])

    with tab_overview:
        UIComponents.show_breadcrumb("📊 Overview")
        t1, t2, t3 = st.tabs(["📋 Raport", "📈 KPI", "📊 Podsumowanie"])
        with t1: render_tab_content("report", df_plot, rider_weight, cp_input)
        with t2: render_tab_content("kpi", df_plot, df_plot_resampled, metrics, rider_weight, decoupling_percent, drift_z2, vt1_vent, vt2_vent)
        with t3: render_tab_content("summary", df_plot, df_plot_resampled, metrics, training_notes, uploaded_file.name, cp_input, w_prime_input, rider_weight, vt1_watts, vt2_watts, 0, 0)

    with tab_performance:
        UIComponents.show_breadcrumb("⚡ Performance")
        t1, t3, t4, t5, t6, t7, t8, t9 = st.tabs(["🔋 Power", "⏱️ Intervals", "🦵 Biomech", "📐 Model", "❤️ HR", "🧬 Hematology", "📈 Drift Maps", "⏳ TTE"])
        with t1: render_tab_content("power", df_plot, df_plot_resampled, cp_input, w_prime_input, rider_weight, metrics.get('vo2_max_est', 0))
        with t3: render_tab_content("intervals", df_plot, df_plot_resampled, cp_input, rider_weight, rider_age, is_male)
        with t4: render_tab_content("biomech", df_plot, df_plot_resampled)
        with t5: render_tab_content("model", df_plot, cp_input, w_prime_input)
        with t6: render_tab_content("heart_rate", df_plot)
        with t7: render_tab_content("hemo", df_plot)
        with t8: render_tab_content("drift_maps", df_plot)
        filename = uploaded_file.name if uploaded_file else "manual_upload"
        with t9: render_tab_content("tte", df_plot, cp_input, filename)

    with tab_intelligence:
        UIComponents.show_breadcrumb("🧠 Intelligence")
        t1, t2, t3 = st.tabs(["🍎 Nutrition", "🚧 Limiters", "🤖 AI Coach"])
        with t1: render_tab_content("nutrition", df_plot, cp_input, vt1_watts, vt2_watts)
        with t2: render_tab_content("limiters", df_plot, cp_input, vt2_vent)
        with t3: render_tab_content("ai_coach", df_plot_resampled)

    with tab_physiology:
        UIComponents.show_breadcrumb("🫀 Physiology")
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(["💓 HRV", "🩸 SmO2", "🫁 Ventilation", "🌡️ Thermal", "🎯 Vent - Progi", "🛠️ Vent - Progi Manuals", "🎯 SmO2 - Progi", "🛠️ SmO2 - Progi Manuals", "🗄️ Ramp Archive"])
        with t1: render_tab_content("hrv", df_clean_pl)
        with t2: render_tab_content("smo2", df_plot, training_notes, uploaded_file.name)
        max_hr = int(208 - 0.7 * rider_age) if rider_age else 185
        with t3: render_tab_content("vent", df_plot, training_notes, uploaded_file.name)
        with t4: render_tab_content("thermal", df_plot)
        with t5: render_tab_content("vent_thresholds", df_plot, training_notes, uploaded_file.name, cp_input)
        with t6: render_tab_content("manual_thresholds", df_plot, training_notes, uploaded_file.name, cp_input, max_hr)
        with t7: render_tab_content("smo2_thresholds", df_plot, training_notes, uploaded_file.name, cp_input)
        with t8: render_tab_content("smo2_manual_thresholds", df_plot, training_notes, uploaded_file.name, cp_input)
        with t9: render_tab_content("ramp_archive")



    # DOCX / PNG Export
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Export Raportu")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        try:
            docx = generate_docx_report(
                metrics, df_plot, df_plot_resampled, uploaded_file, cp_input, 
                vt1_watts, vt2_watts, rider_weight, vt1_vent, vt2_vent, w_prime_input
            )
            buf = BytesIO()
            docx.save(buf)
            st.sidebar.download_button("📥 DOCX", buf.getvalue(), f"{uploaded_file.name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception:
            pass  # DOCX export failure is non-critical
    with c2:
        try:
            zip_data = export_all_charts_as_png(
                df_plot, df_plot_resampled, cp_input, vt1_watts, vt2_watts, metrics, rider_weight,
                uploaded_file, None, None, None, None
            )
            st.sidebar.download_button("📸 PNG", zip_data, f"{uploaded_file.name}.zip", mime="application/zip")
        except Exception:
            pass  # PNG export failure is non-critical

else:
    st.sidebar.info("Wgraj plik.")
