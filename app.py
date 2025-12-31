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

# UI Modules - Lazy Loading Wrappers
def render_tab_content(tab_name, *args, **kwargs):
    """Dynamic dispatcher for tab rendering."""
    # Mapping tab names to module imports and render functions
    # Using lazy import inside logic
    if tab_name == "report":
        from modules.ui.report import render_report_tab
        return render_report_tab(*args, **kwargs)
    elif tab_name == "kpi":
        from modules.ui.kpi import render_kpi_tab
        return render_kpi_tab(*args, **kwargs)
    elif tab_name == "power":
        from modules.ui.power import render_power_tab
        return render_power_tab(*args, **kwargs)
    elif tab_name == "pdc":
        from modules.ui.pdc_ui import render_pdc_tab
        return render_pdc_tab(*args, **kwargs)
    elif tab_name == "intervals":
        from modules.ui.intervals_ui import render_intervals_tab
        return render_intervals_tab(*args, **kwargs)
    elif tab_name == "biomech":
        from modules.ui.biomech import render_biomech_tab
        return render_biomech_tab(*args, **kwargs)
    elif tab_name == "model":
        from modules.ui.model import render_model_tab
        return render_model_tab(*args, **kwargs)
    elif tab_name == "hrv":
        from modules.ui.hrv import render_hrv_tab
        return render_hrv_tab(*args, **kwargs)
    elif tab_name == "smo2":
        from modules.ui.smo2 import render_smo2_tab
        return render_smo2_tab(*args, **kwargs)
    elif tab_name == "hemo":
        from modules.ui.hemo import render_hemo_tab
        return render_hemo_tab(*args, **kwargs)
    elif tab_name == "vent":
        from modules.ui.vent import render_vent_tab
        return render_vent_tab(*args, **kwargs)
    elif tab_name == "thermal":
        from modules.ui.thermal import render_thermal_tab
        return render_thermal_tab(*args, **kwargs)
    elif tab_name == "nutrition":
        from modules.ui.nutrition import render_nutrition_tab
        return render_nutrition_tab(*args, **kwargs)
    elif tab_name == "limiters":
        from modules.ui.limiters import render_limiters_tab
        return render_limiters_tab(*args, **kwargs)
    elif tab_name == "ai_coach":
        from modules.ui.ai_coach import render_ai_coach_tab
        return render_ai_coach_tab(*args, **kwargs)
    elif tab_name == "thresholds":
        from modules.ui.threshold_analysis_ui import render_threshold_analysis_tab
        return render_threshold_analysis_tab(*args, **kwargs)
    elif tab_name == "trends":
        from modules.ui.trends import render_trends_tab
        return render_trends_tab(*args, **kwargs)
    elif tab_name == "history":
        from modules.ui.trends_history import render_trends_history_tab
        return render_trends_history_tab(*args, **kwargs)
    elif tab_name == "community":
        from modules.ui.community import render_community_tab
        return render_community_tab(*args, **kwargs)
    elif tab_name == "import":
        from modules.ui.history_import_ui import render_history_import_tab
        return render_history_import_tab(*args, **kwargs)

# --- INIT ---
ThemeManager.set_page_config()
ThemeManager.load_css()

state = StateManager()
state.init_session_state()

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
            is_valid, error_msg = validate_dataframe(df_raw)
            if not is_valid:
                st.error(f"Błąd walidacji danych: {error_msg}")
                st.stop()

            # --- PROCESSING PIPELINE ---
            # TODO: Move this block to orchestrator completely in next refactor step
            from modules.calculations import process_data
            df_clean_pl = process_data(df_raw)
            metrics = calculate_metrics(df_clean_pl, cp_input)
            df_w_prime = calculate_w_prime_balance(df_clean_pl, cp_input, w_prime_input)
            decoupling_percent, ef_factor = calculate_advanced_kpi(df_clean_pl)
            drift_z2 = calculate_z2_drift(df_clean_pl, cp_input)
            df_plot = calculate_heat_strain_index(df_w_prime)

            metrics = calculate_extended_metrics(
                df_plot, metrics, rider_weight, vt1_watts, vt2_watts, ef_factor
            )
            df_plot = apply_smo2_smoothing(df_plot)
            df_plot_resampled = resample_dataframe(df_plot)

            state.set_data_loaded()

            # AI Section
            if MLX_AVAILABLE and os.path.exists(MODEL_FILE):
                try:
                    auto_pred = predict_only(df_plot_resampled)
                    if auto_pred is not None:
                        df_plot_resampled['ai_hr'] = auto_pred
                except Exception:
                    pass  # AI prediction is non-critical

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

    # Layout Tabs
    tab_overview, tab_performance, tab_physiology, tab_intelligence, tab_analytics = st.tabs([
        "📊 Overview", "⚡ Performance", "🫀 Physiology", "🧠 Intelligence", "📈 Analytics"
    ])

    with tab_overview:
        UIComponents.show_breadcrumb("📊 Overview")
        t1, t2, t3 = st.tabs(["📋 Raport", "📈 KPI", "📉 Trends"])
        with t1: render_tab_content("report", df_plot, rider_weight, cp_input)
        with t2: render_tab_content("kpi", df_plot, df_plot_resampled, metrics, rider_weight, decoupling_percent, drift_z2, vt1_vent, vt2_vent)
        with t3: render_tab_content("trends", df_plot)

    with tab_performance:
        UIComponents.show_breadcrumb("⚡ Performance")
        t1, t2, t3, t4, t5 = st.tabs(["🔋 Power", "📊 PDC", "⏱️ Intervals", "🦵 Biomech", "📐 Model"])
        with t1: render_tab_content("power", df_plot, df_plot_resampled, cp_input, w_prime_input)
        with t2: render_tab_content("pdc", df_plot, cp_input, w_prime_input, rider_weight, metrics.get('vo2_max_est', 0))
        with t3: render_tab_content("intervals", df_plot, df_plot_resampled, cp_input, rider_weight, rider_age, is_male)
        with t4: render_tab_content("biomech", df_plot, df_plot_resampled)
        with t5: render_tab_content("model", df_plot, cp_input, w_prime_input)

    with tab_physiology:
        UIComponents.show_breadcrumb("🫀 Physiology")
        t1, t2, t3, t4, t5 = st.tabs(["💓 HRV", "🩸 SmO2", "🧬 Hematology", "🫁 Ventilation", "🌡️ Thermal"])
        with t1: render_tab_content("hrv", df_clean_pl)
        with t2: render_tab_content("smo2", df_plot, training_notes, uploaded_file.name)
        with t3: render_tab_content("hemo", df_plot)
        with t4: render_tab_content("vent", df_plot, training_notes, uploaded_file.name)
        with t5: render_tab_content("thermal", df_plot)

    with tab_intelligence:
        UIComponents.show_breadcrumb("🧠 Intelligence")
        t1, t2, t3, t4 = st.tabs(["🍎 Nutrition", "🚧 Limiters", "🤖 AI Coach", "🎯 Progi & Plan"])
        with t1: render_tab_content("nutrition", df_plot, cp_input, vt1_watts, vt2_watts)
        with t2: render_tab_content("limiters", df_plot, cp_input, vt2_vent)
        with t3: render_tab_content("ai_coach", df_plot_resampled)
        with t4: 
             max_hr = int(208 - 0.7 * rider_age) if rider_age else 185
             render_tab_content("thresholds", df_plot, training_notes, uploaded_file.name, cp_input, cp_input, max_hr)

    with tab_analytics:
        UIComponents.show_breadcrumb("📈 Analytics")
        t1, t2, t3 = st.tabs(["📊 History", "👥 Community", "📂 Import"])
        with t1: render_tab_content("history")
        with t2: render_tab_content("community", cp_input, rider_weight, metrics.get('vo2_max_est', 0), rider_age, 'M' if is_male else 'F')
        with t3: render_tab_content("import", cp_input)

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
