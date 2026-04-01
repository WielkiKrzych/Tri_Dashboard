"""
Ventilatory Thresholds tab — VT1/VT2 detection via V-slope and step-test methods.

Orchestrates:
  vent_thresholds_report   — report generation button + save validation
  vent_thresholds_display  — 4 threshold cards, zones table, analysis notes, theory
  vent_thresholds_charts   — CPET chart panels (VE/VO2, VE/VCO2, RER)
  vent_thresholds_timeline — interactive Plotly timeline with zone backgrounds
"""

import logging
import streamlit as st
import pandas as pd
from modules.calculations.vt_cpet import detect_vt_cpet
from modules.calculations.quality import check_step_test_protocol
from modules.calculations.pipeline import run_ramp_test_pipeline
from models.results import ValidityLevel

from .vent_thresholds_display import render_threshold_cards, render_theory_section
from .vent_thresholds_charts import render_cpet_charts
from .vent_thresholds_timeline import render_threshold_timeline

logger = logging.getLogger(__name__)


def render_vent_thresholds_tab(
    target_df,
    training_notes,
    uploaded_file_name,
    cp_input,
    w_prime_input=20000,
    rider_weight=75.0,
    max_hr=190.0,
):
    st.header("🎯 Detekcja Progów Wentylacyjnych (VT1 / VT2)")
    st.markdown(
        "Automatyczna detekcja progów wentylacyjnych. **Wymaga testu stopniowanego (Ramp Test).**"
    )

    # 1. Data validation
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        return

    if "time" not in target_df.columns or "tymeventilation" not in target_df.columns:
        st.info("ℹ️ Brak danych wentylacji (tymeventilation) w tym pliku.")
        return

    # Work on a copy to avoid mutating the caller's DataFrame
    target_df = target_df.copy()
    target_df.columns = target_df.columns.str.lower().str.strip()

    if "hr" not in target_df.columns:
        for alias in ["heart_rate", "heart rate", "bpm", "tętno", "heartrate", "heart_rate_bpm"]:
            if alias in target_df.columns:
                target_df = target_df.rename(columns={alias: "hr"})
                break

    if "watts_smooth_5s" not in target_df.columns and "watts" in target_df.columns:
        target_df["watts_smooth_5s"] = target_df["watts"].rolling(window=5, center=True).mean()
    target_df["ve_smooth"] = target_df["tymeventilation"].rolling(window=10, center=True).mean()
    target_df["time_str"] = pd.to_datetime(target_df["time"], unit="s").dt.strftime("%H:%M:%S")

    # 2. Protocol compliance check
    st.subheader("📋 Weryfikacja Protokołu")

    proto_check = check_step_test_protocol(target_df)

    if not proto_check["is_valid"]:
        st.error("⚠️ **Wykryto Problemy z Protokołem Testu**")
        for issue in proto_check["issues"]:
            st.warning(issue)
        st.markdown("""
        **Dlaczego to ważne?**

        Detekcja progów wentylacyjnych wymaga **testu stopniowanego (Ramp Test)** z liniowym wzrostem obciążenia.
        Dla normalnych treningów użyj zakładki **"🫁 Ventilation"** do analizy manualnej.
        """)
        if not st.checkbox("⚠️ Wymuś analizę mimo błędów protokołu (wyniki mogą być niewiarygodne)"):
            st.stop()
    else:
        st.success("✅ Protokół Testu Stopniowanego: Poprawny (Liniowy Wzrost Obciążenia)")

    st.markdown("---")

    # 3. Manual override for ramp start
    with st.expander("⚙️ Ustawienia detekcji progów", expanded=False):
        st.markdown("""
        **Manualne ustawienie początku Ramp Testu**

        Jeśli automatyczna detekcja rozgrzewki nie działa poprawnie, ustaw minimalną moc
        od której algorytm zacznie szukać progów wentylacyjnych.
        """)
        min_power_input = st.number_input(
            "Minimalna moc do analizy [W]",
            min_value=0,
            max_value=500,
            value=st.session_state.get("vt_min_power_watts", 0),
            step=10,
            help="Ustaw 0 dla automatycznej detekcji. Ustaw np. 200W aby pominąć rozgrzewkę poniżej 200W.",
        )
        st.session_state["vt_min_power_watts"] = min_power_input

        if min_power_input > 0:
            st.info(f"📌 Analiza progów rozpocznie się od mocy ≥ {min_power_input}W")
        else:
            st.caption("ℹ️ Automatyczna detekcja początku Ramp Testu")

    # 4. CPET detection + pipeline
    min_power_watts = st.session_state.get("vt_min_power_watts", 0) or None

    with st.spinner("Analizowanie progów wentylacyjnych (CPET)..."):
        cpet_result = detect_vt_cpet(
            target_df,
            step_range=None,
            power_column="watts",
            ve_column="tymeventilation",
            time_column="time",
            min_power_watts=min_power_watts,
        )
        st.session_state["cpet_vt_result"] = cpet_result

        try:
            pipeline_result = run_ramp_test_pipeline(
                target_df,
                power_column="watts",
                ve_column="tymeventilation",
                hr_column="hr" if "hr" in target_df.columns else None,
                smo2_column="smo2" if "smo2" in target_df.columns else None,
                time_column="time",
                test_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                protocol="Ramp Test",
                cp_watts=float(cp_input),
                w_prime_joules=float(w_prime_input),
                smo2_manual_lt1=st.session_state.get("smo2_lt1_m"),
                smo2_manual_lt2=st.session_state.get("smo2_lt2_m"),
                rider_weight=float(rider_weight),
                max_hr=float(max_hr),
            )

            st.session_state["pending_pipeline_result"] = pipeline_result
            st.session_state["pending_source_df"] = target_df
            st.session_state["pending_uploaded_file_name"] = uploaded_file_name
            st.session_state["pending_cp_input"] = cp_input

            if pipeline_result.validity.validity in [
                ValidityLevel.VALID,
                ValidityLevel.CONDITIONAL,
            ]:
                st.success("✅ Test poprawny - gotowy do wygenerowania raportu")
            else:
                st.warning("⚠️ Test niepoprawny - raport może być niewiarygodny")

        except Exception as e:
            st.error(f"Błąd analizy pipeline: {e}")
            logger.error("Pipeline failed: %s", e)

    # 5. Sub-sections
    st.markdown("---")
    st.subheader("🤖 Wykryte Progi Wentylacyjne (CPET)")
    cpet_result = st.session_state.get("cpet_vt_result", {})

    render_threshold_cards(cpet_result, target_df, cp_input)
    render_cpet_charts(cpet_result)
    render_threshold_timeline(cpet_result, target_df)
    render_theory_section()
