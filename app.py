import streamlit as st
import html
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any

from io import BytesIO
import os

import pandas as pd


def sanitize_filename(name: str) -> str:
    """Strip dangerous characters from uploaded filenames to prevent path traversal."""
    # Keep only the basename (strip directories), then whitelist safe chars
    basename = os.path.basename(name)
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", basename)
    return safe[:128] if safe else "upload"


# --- FRONTEND IMPORTS ---
from modules.frontend.theme import ThemeManager
from modules.frontend.state import StateManager
from modules.frontend.layout import AppLayout
from modules.frontend.components import (
    UIComponents,
    render_header_metrics_fragment,
    render_export_buttons_fragment,
)

# --- MODULE IMPORTS ---
from modules.utils import load_data
from modules.ml_logic import MLX_AVAILABLE, predict_only, MODEL_FILE
from modules.notes import TrainingNotes
from modules.reports import export_all_charts_as_png
from modules.db import SessionStore, SessionRecord
from modules.cache_utils import get_session_store  # Use cached singleton
from modules.reporting.persistence import check_git_tracking

# --- DOMAIN IMPORTS ---
from modules.domain import SessionType, classify_session_type, classify_ramp_test

# --- SERVICES IMPORTS ---
from services import calculate_header_metrics, prepare_session_record, prepare_sticky_header_data
from services.session_orchestrator import process_uploaded_session


logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_file_hash(file) -> str:
    """
    Compute stable hash for file cache key.

    Uses MD5 for speed (not cryptographic security needed).
    Hashes actual file content for cache integrity.
    """
    file.seek(0)  # Reset position
    content_hash = hashlib.md5()
    # Hash content in chunks for memory efficiency
    for chunk in iter(lambda: file.read(8192), b""):
        content_hash.update(chunk)
    file.seek(0)  # Reset for subsequent reads
    return content_hash.hexdigest()[:16]


def classify_and_cache_session(df_raw: pd.DataFrame, file_hash: str, uploaded_file):
    """Classify session type with caching in session_state."""
    cached_hash = st.session_state.get("current_file_hash")

    if cached_hash == file_hash:
        # Cache hit - use stored values
        return (st.session_state.get("session_type"), st.session_state.get("ramp_classification"))

    # Cache miss - classify new file
    session_type = classify_session_type(df_raw, uploaded_file.name)
    st.session_state["session_type"] = session_type

    ramp_classification = None
    if "watts" in df_raw.columns or "power" in df_raw.columns:
        power_col = "watts" if "watts" in df_raw.columns else "power"
        power = df_raw[power_col].dropna()
        if len(power) >= 300:
            ramp_classification = classify_ramp_test(power)
            st.session_state["ramp_classification"] = ramp_classification

    st.session_state["current_file_hash"] = file_hash
    return session_type, ramp_classification


def render_session_badge(session_type, ramp_classification) -> None:
    """Render session type badge with confidence indicator."""
    if not session_type:
        return

    # Determine badge styling based on session type
    if session_type == SessionType.RAMP_TEST and ramp_classification:
        confidence = ramp_classification.confidence
        bg_color = "rgba(46, 204, 113, 0.2)"
        msg = f"Rozpoznano: <b>Ramp Test</b> (confidence: {confidence:.2f})"
    elif session_type == SessionType.RAMP_TEST_CONDITIONAL and ramp_classification:
        confidence = ramp_classification.confidence
        bg_color = "rgba(241, 196, 15, 0.2)"
        msg = f"Rozpoznano: <b>Ramp Test (warunkowo)</b> (confidence: {confidence:.2f})"
    elif session_type == SessionType.TRAINING:
        bg_color = "rgba(52, 152, 219, 0.2)"
        if ramp_classification and not ramp_classification.is_ramp:
            msg = f"Sesja treningowa – analiza badawcza pominięta"
        else:
            msg = f"Rozpoznano: <b>Sesja treningowa</b>"
    else:
        bg_color = "rgba(149, 165, 166, 0.2)"
        # Escape session_type to prevent XSS
        safe_type = html.escape(str(session_type))
        msg = f"Typ sesji: <b>{safe_type}</b>"

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, {bg_color}, transparent); 
                    padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; display: inline-block;">
            <span style="font-size: 1.1em;">{session_type.emoji} {msg}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB REGISTRY (OCP)
# =============================================================================
class TabRegistry:
    """Registry for UI tabs to support Open/Closed Principle."""

    _tabs = {
        "report": ("modules.ui.report", "render_report_tab"),
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
        "smo2_manual_thresholds": (
            "modules.ui.smo2_manual_thresholds",
            "render_smo2_manual_thresholds_tab",
        ),
        "summary": ("modules.ui.summary", "render_summary_tab"),
        "drift_maps": ("modules.ui.drift_maps_ui", "render_drift_maps_tab"),
        "tte": ("modules.ui.tte_ui", "render_tte_tab"),
        "ramp_archive": ("modules.ui.ramp_archive", "render_ramp_archive"),
        "durability": ("modules.ui.durability_ui", "render_durability_tab"),
        "training_dist": ("modules.ui.training_distribution_ui", "render_training_distribution_tab"),
        "heat_strain": ("modules.ui.heat_strain_ui", "render_heat_strain_tab"),
        "race_predictor": ("modules.ui.race_predictor_ui", "render_race_predictor_tab"),
        "wprime_recon": ("modules.ui.w_prime_reconstitution_ui", "render_w_prime_reconstitution_tab"),
    }

    @classmethod
    def render(cls, tab_name, *args, **kwargs):
        """Dynamic dispatcher for tab rendering (Lazy loading)."""
        if tab_name not in cls._tabs:
            logger.error("Unknown tab requested: %s", tab_name)
            st.error(f"Unknown tab: {tab_name}")
            return

        module_path, func_name = cls._tabs[tab_name]
        try:
            import importlib

            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            return func(*args, **kwargs)
        except Exception as e:
            logger.error("Tab load error [%s]: %s", tab_name, e, exc_info=True)
            st.error(f"Nie udalo sie zaladowac zakladki {tab_name}. Sprawdz logi.")


def render_tab_content(tab_name, *args, **kwargs):
    """Facade for TabRegistry."""
    return TabRegistry.render(tab_name, *args, **kwargs)


# --- INIT ---
ThemeManager.set_page_config()
# Load dark theme CSS
ThemeManager.load_css(theme="dark")

state = StateManager()
state.init_session_state()

from modules.db.athlete_profiles import AthleteProfileStore

_profile_store = AthleteProfileStore()
if st.session_state["selected_athlete_id"] == "default":
    _first_profile = _profile_store.get_or_create_default()
    st.session_state["selected_athlete_id"] = _first_profile.id
    state.load_profile_into_state(_first_profile.id)

# Safety Check: Git Tracking of sensitive data (reports & raw CSVs)
check_git_tracking("reports/ramp_tests")
check_git_tracking("treningi_csv")

layout = AppLayout(state)
uploaded_file, params = layout.render_sidebar()

# Parameters shorthand with validation
rider_weight = max(0.1, params.get("rider_weight", 75.0))
cp_input = max(1, params.get("cp", 280))
vt1_watts = max(0, params.get("vt1_watts", 0))
vt2_watts = max(0, params.get("vt2_watts", 0))
vt1_vent = max(0, params.get("vt1_vent", 0))
vt2_vent = max(0, params.get("vt2_vent", 0))
w_prime_input = max(0, params.get("w_prime", 20000))
rider_age = max(1, min(120, params.get("rider_age", 30)))
is_male = params.get("is_male", True)
max_hr = int(208 - 0.7 * rider_age) if rider_age else 185
hr_rest = max(0, params.get("hr_rest", 60))

layout.render_header()


if rider_weight <= 0 or cp_input <= 0:
    logger.warning("Invalid params: weight=%.1f, cp=%d", rider_weight, cp_input)
    st.error("Błąd: Waga i CP muszą być większe od zera.")
    st.stop()

if uploaded_file is not None:
    state.cleanup_old_data()
    safe_filename = sanitize_filename(uploaded_file.name)
    training_notes = TrainingNotes()

    with st.spinner("Przetwarzanie danych..."):
        try:
            df_raw = load_data(uploaded_file)

            # Session classification with caching
            current_file_hash = compute_file_hash(uploaded_file)
            session_type, ramp_classification = classify_and_cache_session(
                df_raw, current_file_hash, uploaded_file
            )

            # Processing pipeline

            df_plot, df_plot_resampled, metrics, error_msg = process_uploaded_session(
                df_raw, cp_input, w_prime_input, rider_weight, vt1_watts, vt2_watts
            )

            if error_msg:
                logger.error("Session processing failed: %s", error_msg)
                st.error("Blad analizy danych. Sprawdz format pliku i sprobuj ponownie.")
                st.stop()

            # Extract intermediate results from metrics (DIP: metrics acts as a container here)
            decoupling_percent = metrics.pop("_decoupling_percent", 0.0)
            drift_z2 = metrics.pop("_drift_z2", 0.0)
            df_clean_pl = metrics.pop("_df_clean_pl", df_raw)

            state.set_data_loaded()

            # Compute intelligent alerts
            try:
                from modules.calculations.alert_engine import analyze_session_alerts

                store = get_session_store()
                history_records = store.get_sessions(days=90)
                session_history_list = [
                    {
                        "date": r.date,
                        "avg_rmssd": r.avg_rmssd,
                        "session_type": r.session_type,
                        "tss": r.tss,
                    }
                    for r in history_records
                ]
                alert_report = analyze_session_alerts(
                    df_plot, metrics, session_history=session_history_list
                )
            except Exception as e:
                logger.warning("Alert engine failed: %s", e)
                from modules.calculations.alert_engine import AlertReport

                alert_report = AlertReport()

            # AI Section (Optional/Non-critical)
            if MLX_AVAILABLE and os.path.exists(MODEL_FILE):
                try:
                    auto_pred = predict_only(df_plot_resampled)
                    if auto_pred is not None:
                        df_plot_resampled["ai_hr"] = auto_pred
                except Exception as e:
                    logger.warning(f"AI prediction failed: {e}")

        except Exception as e:
            logger.error("File load error: %s", e, exc_info=True)
            st.error("Nie udalo sie wczytac pliku. Sprawdz format (CSV/TXT).")
            st.stop()

    # --- RENDER DASHBOARD ---

    # 1. Header Metrics
    np_header, if_header, tss_header = calculate_header_metrics(df_plot, cp_input)

    # Auto-save
    try:
        session_data = prepare_session_record(
            safe_filename, df_plot, metrics, np_header, if_header, tss_header
        )
        session_data["athlete_id"] = st.session_state.get("selected_athlete_id", "default")
        get_session_store().add_session(SessionRecord(**session_data))
    except Exception as e:
        logger.warning(f"Auto-save failed: {e}")

    # Sticky Header
    header_data = prepare_sticky_header_data(df_plot, metrics)
    UIComponents.render_sticky_header(header_data)

    # Header Metrics - using fragment for independent updates
    render_header_metrics_fragment(np_header, if_header, tss_header, df_plot)

    # Session Type Badge with Confidence
    session_type = st.session_state.get("session_type")
    ramp_classification = st.session_state.get("ramp_classification")

    if session_type:
        render_session_badge(session_type, ramp_classification)

    if alert_report.has_critical or alert_report.warning_count > 0:
        badge_parts = []
        if alert_report.has_critical:
            badge_parts.append(f"\U0001f6a8 {alert_report.critical_count} krytycznych")
        if alert_report.warning_count > 0:
            badge_parts.append(f"\u26a0\ufe0f {alert_report.warning_count} ostrze\u017ce\u0144")
        badge_text = " | ".join(badge_parts)
        st.markdown(
            f'<div style="background: rgba(231, 76, 60, 0.15); padding: 8px 15px; '
            f'border-radius: 8px; margin-bottom: 8px; display: inline-block;">'
            f'<span style="font-size: 1.0em;">{badge_text}</span></div>',
            unsafe_allow_html=True,
        )

    # Layout Tabs
    tab_overview, tab_performance, tab_intelligence, tab_physiology = st.tabs(
        ["📊 Overview", "⚡ Performance", "🧠 Intelligence", "🫀 Physiology"]
    )

    with tab_overview:
        UIComponents.show_breadcrumb("📊 Overview")
        t1, t2 = st.tabs(["📋 Raport z KPI", "📊 Podsumowanie"])
        with t1:
            render_tab_content(
                "report",
                df_plot,
                df_plot_resampled,
                metrics,
                rider_weight,
                cp_input,
                decoupling_percent,
                drift_z2,
                vt1_vent,
                vt2_vent,
            )
        with t2:
            render_tab_content(
                "summary",
                df_plot,
                df_plot_resampled,
                metrics,
                training_notes,
                safe_filename,
                cp_input,
                w_prime_input,
                rider_weight,
                vt1_watts,
                vt2_watts,
                0,
                0,
            )

    with tab_performance:
        UIComponents.show_breadcrumb("⚡ Performance")
        t1, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13 = st.tabs(
            [
                "🔋 Power",
                "⏱️ Intervals",
                "🦵 Biomech",
                "📐 Model",
                "❤️ HR",
                "🧬 Hematology",
                "📈 Drift Maps",
                "⏳ TTE",
                "🛡️ Wytrzymałość",
                "📊 Rozkład Treningu",
                "🌡️ Heat Strain",
                "🏁 Race Predictor",
            ]
        )
        with t1:
            render_tab_content(
                "power",
                df_plot,
                df_plot_resampled,
                cp_input,
                w_prime_input,
                rider_weight,
                metrics.get("vo2_max_est", 0),
            )
        with t3:
            render_tab_content(
                "intervals", df_plot, df_plot_resampled, cp_input, rider_weight, rider_age, is_male
            )
        with t4:
            render_tab_content("biomech", df_plot, df_plot_resampled)
        with t5:
            render_tab_content("model", df_plot, cp_input, w_prime_input)
        with t6:
            render_tab_content("heart_rate", df_plot)
        with t7:
            render_tab_content("hemo", df_plot)
        with t8:
            render_tab_content("drift_maps", df_plot)
        filename = safe_filename if uploaded_file else "manual_upload"
        with t9:
            render_tab_content("tte", df_plot, cp_input, filename)
        with t10:
            render_tab_content(
                "durability", df_plot, df_plot_resampled, metrics,
                rider_weight, cp_input, w_prime_input,
            )
        with t11:
            render_tab_content(
                "training_dist", df_plot, df_plot_resampled, metrics,
                rider_weight, cp_input, w_prime_input,
                max_hr, hr_rest,
            )
        with t12:
            render_tab_content(
                "heat_strain", df_plot, df_plot_resampled, metrics,
                rider_weight, cp_input, w_prime_input,
                max_hr, hr_rest, rider_age, is_male,
            )
        with t13:
            render_tab_content(
                "race_predictor", df_plot, df_plot_resampled, metrics,
                rider_weight, cp_input, w_prime_input,
            )

    with tab_intelligence:
        UIComponents.show_breadcrumb("\U0001f9e0 Intelligence")
        t1, t2, t3, t4 = st.tabs(
            [
                "\U0001f34e Nutrition",
                "\U0001f6a7 Limiters",
                "\U0001f916 AI Coach",
                "\U0001f514 Alerty",
            ]
        )
        with t1:
            render_tab_content("nutrition", df_plot, cp_input, vt1_watts, vt2_watts)
        with t2:
            render_tab_content("limiters", df_plot, cp_input, vt2_vent)
        with t3:
            render_tab_content("ai_coach", df_plot_resampled, cp_watts=cp_input)
        with t4:
            from modules.ui.alerts import render_alerts_tab

            render_alerts_tab(alert_report)

    with tab_physiology:
        UIComponents.show_breadcrumb("🫀 Physiology")
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs(
            [
                "💓 HRV",
                "🩸 SmO2",
                "🫁 Ventilation",
                "🌡️ Thermal",
                "🎯 Vent - Progi",
                "🛠️ Vent - Progi Manuals",
                "🎯 SmO2 - Progi",
                "🛠️ SmO2 - Progi Manuals",
                "🗄️ Ramp Archive",
                "🔋 W' Rekonstytucja",
            ]
        )
        with t1:
            render_tab_content("hrv", df_clean_pl)
        with t2:
            render_tab_content("smo2", df_plot, training_notes, safe_filename)
        with t3:
            render_tab_content("vent", df_plot, training_notes, safe_filename)
        with t4:
            render_tab_content("thermal", df_plot)
        with t5:
            render_tab_content(
                "vent_thresholds",
                df_plot,
                training_notes,
                safe_filename,
                cp_input,
                w_prime_input,
                rider_weight,
                max_hr,
            )
        with t6:
            render_tab_content(
                "manual_thresholds", df_plot, training_notes, safe_filename, cp_input, max_hr
            )
        with t7:
            render_tab_content("smo2_thresholds", df_plot, training_notes, safe_filename, cp_input)
        with t8:
            render_tab_content(
                "smo2_manual_thresholds", df_plot, training_notes, safe_filename, cp_input
            )
        with t9:
            render_tab_content("ramp_archive")
        with t10:
            render_tab_content(
                "wprime_recon", df_plot, df_plot_resampled, metrics,
                rider_weight, cp_input, w_prime_input,
            )

    # PNG Export
    st.sidebar.markdown("---")
    st.sidebar.header("📄 Export Raportu")
    try:
        zip_data = export_all_charts_as_png(
            df_plot,
            df_plot_resampled,
            cp_input,
            vt1_watts,
            vt2_watts,
            metrics,
            rider_weight,
            uploaded_file,
            None,
            None,
            None,
            None,
        )
        if zip_data:
            st.sidebar.download_button(
                "📸 PNG", zip_data, f"{safe_filename}.zip", mime="application/zip"
            )
    except Exception as e:
        st.sidebar.error(f"Błąd eksportu PNG: {e}")
        logger.warning(f"PNG export failed: {e}")

    st.sidebar.markdown("---")
    with st.sidebar.expander("📤 Export do platform", expanded=False):
        try:
            from modules.export.zone_exporter import export_hr_zones_csv, export_power_zones_csv
            from modules.export.tcx_generator import generate_tcx_bytes
            from modules.export.workout_exporter import export_trainingpeaks_csv

            export_name = safe_filename.replace(".csv", "").replace(".fit", "")[:50]

            st.download_button(
                "📁 TCX (Strava/Garmin)",
                data=generate_tcx_bytes(df_plot, metrics, cp_input),
                file_name=f"{export_name}.tcx",
                mime="application/vnd.garmin.tcx+xml",
                use_container_width=True,
            )
            st.download_button(
                "💪 Power Zones CSV",
                data=export_power_zones_csv(cp_input, rider_weight),
                file_name=f"{export_name}_power_zones.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "❤️ HR Zones CSV",
                data=export_hr_zones_csv(max_hr),
                file_name=f"{export_name}_hr_zones.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "📊 TrainingPeaks CSV",
                data=export_trainingpeaks_csv(
                    metrics, df_plot, cp_input, rider_weight, filename=export_name
                ),
                file_name=f"{export_name}_tp.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Export niedostępny: {e}")
            logger.warning(f"Platform export failed: {e}")


else:
    st.sidebar.info("Wgraj plik.")
