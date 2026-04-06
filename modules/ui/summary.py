"""
Moduł UI: Zakładka Podsumowanie (Summary)

Agreguje kluczowe wykresy i metryki z całego dashboardu w jednym miejscu.

Sub-modules:
  summary_calculations.py — pure math helpers (_hash_dataframe, _calculate_np, etc.)
  summary_charts.py       — Plotly chart builders (_build_training_timeline_chart, etc.)
  summary_thresholds.py   — VT/LT/TDI renderers (_render_vent_thresholds_summary, etc.)
"""

import logging

import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


def _safe_int(value, default: int = 0) -> int:
    """Convert to int, returning *default* for None / NaN / non-finite."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN
            return default
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return default

from modules.calculations.column_aliases import normalize_columns, resolve_hr_column
from modules.config import Config
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.smo2_advanced import detect_smo2_thresholds_moxy
from modules.plots import CHART_CONFIG

from .summary_calculations import _hash_dataframe, _calculate_np, _estimate_cp_wprime
from .summary_charts import (
    _build_training_timeline_chart,
    _render_cp_model_chart,
    _render_smo2_thb_chart,
)
from .summary_thresholds import (
    _render_vent_thresholds_summary,
    _render_smo2_thresholds_summary,
    _render_tdi_analysis,
)


def render_summary_tab(
    df_plot: pd.DataFrame,
    df_plot_resampled: pd.DataFrame,
    metrics: dict[str, Any] | None,
    training_notes,
    uploaded_file_name: str,
    cp_input: int,
    w_prime_input: int,
    rider_weight: float,
    vt1_watts: int = 0,
    vt2_watts: int = 0,
    lt1_watts: int = 0,
    lt2_watts: int = 0,
):
    """Renderowanie zakładki Podsumowanie z kluczowymi wykresami i metrykami."""
    st.header("📊 Podsumowanie Treningu")
    st.markdown("Wszystkie kluczowe wykresy i metryki w jednym miejscu.")

    # Work on a copy to avoid mutating the caller's DataFrame
    df_plot = df_plot.copy()
    normalize_columns(df_plot)

    # --- SHARED THRESHOLD DETECTION ---
    hr_col = resolve_hr_column(df_plot)

    analyze_kwargs: dict[str, Any] = {
        "power_column": "watts",
        "time_column": "time",
    }
    if "tymeventilation" in df_plot.columns:
        analyze_kwargs["ve_column"] = "tymeventilation"
    if "smo2" in df_plot.columns:
        analyze_kwargs["smo2_column"] = "smo2"
    if hr_col:
        analyze_kwargs["hr_column"] = hr_col

    # --- Try cached results first (from other tabs) ---
    cached_vt = st.session_state.get("cpet_vt_result") or st.session_state.get("threshold_result")
    if cached_vt and hasattr(cached_vt, "vt1_watts"):
        threshold_result = cached_vt
    else:
        try:
            threshold_result = analyze_step_test(df_plot, **analyze_kwargs)
        except Exception as e:
            logger.error("Summary: analyze_step_test failed: %s", e, exc_info=True)
            from modules.calculations.threshold_types import StepTestResult
            threshold_result = StepTestResult()

    smo2_result = None
    if "smo2" in df_plot.columns:
        cached_smo2 = st.session_state.get("smo2_threshold_result")
        if cached_smo2 and hasattr(cached_smo2, "t1_watts"):
            smo2_result = cached_smo2
        else:
            try:
                hr_max = _safe_int(df_plot[hr_col].max()) if hr_col else None
                detected_vt1 = _safe_int(threshold_result.vt1_watts) or None
                detected_vt2 = _safe_int(threshold_result.vt2_watts) or None
                smo2_kwargs: dict[str, Any] = {
                    "df": df_plot,
                    "step_duration_sec": 180,
                    "smo2_col": "smo2",
                    "power_col": "watts",
                    "time_col": "time",
                    "cp_watts": cp_input if cp_input > 0 else None,
                    "hr_max": hr_max,
                    "vt1_watts": detected_vt1,
                    "rcp_onset_watts": detected_vt2,
                }
                if hr_col:
                    smo2_kwargs["hr_col"] = hr_col

                smo2_result = detect_smo2_thresholds_moxy(
                    **smo2_kwargs,
                )
            except Exception as e:
                logger.error("Summary: detect_smo2_thresholds_moxy failed: %s", e, exc_info=True)

    try:
        eff_vt1 = _safe_int(
            vt1_watts if vt1_watts and vt1_watts > 0
            else (threshold_result.vt1_watts if threshold_result and threshold_result.vt1_watts else 0)
        )
        eff_vt2 = _safe_int(
            vt2_watts if vt2_watts and vt2_watts > 0
            else (threshold_result.vt2_watts if threshold_result and threshold_result.vt2_watts else 0)
        )
        eff_lt1 = _safe_int(
            lt1_watts if lt1_watts and lt1_watts > 0
            else (smo2_result.t1_watts if smo2_result and smo2_result.t1_watts else 0)
        )
        eff_lt2 = _safe_int(
            lt2_watts if lt2_watts and lt2_watts > 0
            else (smo2_result.t2_onset_watts if smo2_result and smo2_result.t2_onset_watts else 0)
        )
    except Exception as e:
        logger.error("Summary: eff threshold calc failed: %s", e, exc_info=True)
        eff_vt1, eff_vt2, eff_lt1, eff_lt2 = 0, 0, 0, 0

    # =========================================================================
    # 1. WYKRES PRZEBIEG TRENINGU (CACHED)
    # =========================================================================
    st.subheader("1️⃣ Przebieg Treningu")

    try:
        fig_training = _build_training_timeline_chart(
            df_plot,
            cp_input=cp_input,
            vt1_watts=_safe_int(eff_vt1),
            vt2_watts=_safe_int(eff_vt2),
        )
        if fig_training is not None:
            st.plotly_chart(fig_training, width="stretch", config=CHART_CONFIG)
    except Exception as e:
        logger.error("Summary section 1 (Training Timeline) failed: %s", e, exc_info=True)
        st.error("Blad wykresu przebiegu treningu. Sprawdz format danych.")

    # =========================================================================
    # 1a. METRYKI POD WYKRESEM
    # =========================================================================
    try:
        _render_metrics_panel(df_plot, metrics, cp_input, w_prime_input, rider_weight)
    except Exception as e:
        logger.error("Summary section 1a (Metrics Panel) failed: %s", e, exc_info=True)
        st.error("Blad panelu metryk. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 2. WYKRES WENTYLACJA (VE) I ODDECHY (BR)
    # =========================================================================
    st.subheader("2️⃣ Wentylacja (VE) i Oddechy (BR)")

    try:
        if "tymeventilation" in df_plot.columns:
            fig_ve_br = make_subplots(specs=[[{"secondary_y": True}]])

            time_x_s = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

            ve_data = (
                df_plot["tymeventilation"].rolling(10, center=True).mean()
                if "tymeventilation" in df_plot.columns
                else None
            )
            if ve_data is not None:
                fig_ve_br.add_trace(
                    go.Scatter(
                        x=time_x_s,
                        y=ve_data,
                        name="VE (L/min)",
                        line=dict(color="#ffa15a", width=2),
                        hovertemplate="VE: %{y:.1f} L/min<extra></extra>",
                    ),
                    secondary_y=False,
                )

            if "tymebreathrate" in df_plot.columns:
                br_data = df_plot["tymebreathrate"].rolling(10, center=True).mean()
                fig_ve_br.add_trace(
                    go.Scatter(
                        x=time_x_s,
                        y=br_data,
                        name="BR (oddech/min)",
                        line=dict(color="#00cc96", width=2),
                        hovertemplate="BR: %{y:.0f} /min<extra></extra>",
                    ),
                    secondary_y=True,
                )

            fig_ve_br.update_layout(
                template="plotly_dark",
                height=350,
                legend=dict(orientation="h", y=1.05, x=0),
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            fig_ve_br.update_yaxes(title_text="VE (L/min)", secondary_y=False)
            fig_ve_br.update_yaxes(title_text="BR (/min)", secondary_y=True)
            st.plotly_chart(fig_ve_br, width="stretch", config=CHART_CONFIG)

            ve_min = float(df_plot["tymeventilation"].min())
            ve_max = float(df_plot["tymeventilation"].max())
            ve_mean = float(df_plot["tymeventilation"].mean())

            br_min = float(df_plot["tymebreathrate"].min()) if "tymebreathrate" in df_plot.columns else None
            br_max = float(df_plot["tymebreathrate"].max()) if "tymebreathrate" in df_plot.columns else None
            br_mean = float(df_plot["tymebreathrate"].mean()) if "tymebreathrate" in df_plot.columns else None

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                    <h3 style="margin:0; color: #ffa15a;">🫁 VE (Wentylacja)</h3>
                    <p style="margin:5px 0; color:#aaa;"><b>Min:</b> {ve_min:.1f} L/min</p>
                    <p style="margin:5px 0; color:#aaa;"><b>Max:</b> {ve_max:.1f} L/min</p>
                    <p style="margin:5px 0; color:#aaa;"><b>Śr:</b> {ve_mean:.1f} L/min</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                if br_min is not None:
                    st.markdown(
                        f"""
                    <div style="padding:15px; border-radius:8px; border:2px solid #00cc96; background-color: #222;">
                        <h3 style="margin:0; color: #00cc96;">🌬️ BR (Oddechy)</h3>
                        <p style="margin:5px 0; color:#aaa;"><b>Min:</b> {br_min:.0f} /min</p>
                        <p style="margin:5px 0; color:#aaa;"><b>Max:</b> {br_max:.0f} /min</p>
                        <p style="margin:5px 0; color:#aaa;"><b>Śr:</b> {br_mean:.0f} /min</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Brak danych wentylacji (VE/BR) w tym pliku.")
    except Exception as e:
        logger.error("Summary section 2 (VE/BR) failed: %s", e, exc_info=True)
        st.error("Blad wykresu VE/BR. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 3. WYKRES MATEMATYCZNY MODEL CP
    # =========================================================================
    st.subheader("3️⃣ Model Matematyczny CP")
    try:
        _render_cp_model_chart(df_plot, cp_input, w_prime_input)
    except Exception as e:
        logger.error("Summary section 3 (CP Model) failed: %s", e, exc_info=True)
        st.error("Blad modelu CP. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 4. WYKRES SmO2 vs THb W CZASIE
    # =========================================================================
    st.subheader("4️⃣ SmO2 vs THb w czasie")
    try:
        _render_smo2_thb_chart(df_plot)
    except Exception as e:
        logger.error("Summary section 4 (SmO2/THb) failed: %s", e, exc_info=True)
        st.error("Blad wykresu SmO2/THb. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 5. PROGI WENTYLACYJNE VT1/VT2
    # =========================================================================
    st.subheader("5️⃣ Progi Wentylacyjne (VT1/VT2)")
    try:
        _render_vent_thresholds_summary(df_plot, cp_input, _safe_int(eff_vt1), _safe_int(eff_vt2), threshold_result)
    except Exception as e:
        logger.error("Summary section 5 (Vent Thresholds) failed: %s", e, exc_info=True)
        st.error("Blad progow wentylacyjnych. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 6. PROGI SmO2 LT1/LT2
    # =========================================================================
    st.subheader("6️⃣ Progi SmO2 (LT1/LT2)")
    try:
        _render_smo2_thresholds_summary(df_plot, cp_input, _safe_int(eff_lt1), _safe_int(eff_lt2), smo2_result)
    except Exception as e:
        logger.error("Summary section 6 (SmO2 Thresholds) failed: %s", e, exc_info=True)
        st.error("Blad progow SmO2. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 7. THRESHOLD DISCORDANCE INDEX (TDI)
    # =========================================================================
    st.subheader("7️⃣ Threshold Discordance Index (TDI)")
    try:
        _render_tdi_analysis(_safe_int(eff_vt1), _safe_int(eff_lt1))
    except Exception as e:
        logger.error("Summary section 7 (TDI) failed: %s", e, exc_info=True)
        st.error("Blad TDI. Sprawdz format danych.")

    st.markdown("---")

    # =========================================================================
    # 8. VO2max UNCERTAINTY ESTIMATION (CI95%)
    # =========================================================================
    st.subheader("8️⃣ Estymacja VO2max z Niepewnością (CI95%)")
    try:
        _render_vo2max_uncertainty(df_plot, rider_weight)
    except Exception as e:
        logger.error("Summary section 8 (VO2max) failed: %s", e, exc_info=True)
        st.error("Blad estymacji VO2max. Sprawdz format danych.")


# =============================================================================
# HELPER FUNCTIONS (stay in orchestrator — tightly coupled to render_summary_tab)
# =============================================================================


def _render_metrics_panel(df_plot, metrics, cp_input, w_prime_input, rider_weight):
    """Renderowanie panelu z metrykami pod wykresem przebiegu treningu."""

    duration_min = len(df_plot) / 60 if len(df_plot) > 0 else 0

    avg_power = df_plot["watts"].mean() if "watts" in df_plot.columns else 0
    np_power = _calculate_np(df_plot["watts"]) if "watts" in df_plot.columns else 0
    work_kj = df_plot["watts"].sum() / 1000 if "watts" in df_plot.columns else 0

    hr_col = resolve_hr_column(df_plot)
    avg_hr = df_plot[hr_col].mean() if hr_col else 0
    min_hr = df_plot[hr_col].min() if hr_col else 0
    max_hr = df_plot[hr_col].max() if hr_col else 0

    avg_smo2 = df_plot["smo2"].mean() if "smo2" in df_plot.columns else 0
    min_smo2 = df_plot["smo2"].min() if "smo2" in df_plot.columns else 0
    max_smo2 = df_plot["smo2"].max() if "smo2" in df_plot.columns else 0

    avg_ve = df_plot["tymeventilation"].mean() if "tymeventilation" in df_plot.columns else 0
    min_ve = df_plot["tymeventilation"].min() if "tymeventilation" in df_plot.columns else 0
    max_ve = df_plot["tymeventilation"].max() if "tymeventilation" in df_plot.columns else 0

    avg_br = df_plot["tymebreathrate"].mean() if "tymebreathrate" in df_plot.columns else 0
    min_br = df_plot["tymebreathrate"].min() if "tymebreathrate" in df_plot.columns else 0
    max_br = df_plot["tymebreathrate"].max() if "tymebreathrate" in df_plot.columns else 0

    est_vo2max = metrics.get("vo2_max_est", 0) if metrics else 0
    est_vlamax = metrics.get("vlamax_est", 0) if metrics else 0

    est_cp, est_w_prime = _estimate_cp_wprime(df_plot)

    st.markdown("### 📈 Metryki Treningowe")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Czas", f"{duration_min:.1f} min")
    c2.metric("⚡ AVG Power", f"{avg_power:.0f} W")
    c3.metric("📊 NP", f"{np_power:.0f} W")
    c4.metric("🔋 Praca", f"{work_kj:.0f} kJ")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("❤️ AVG HR", f"{avg_hr:.0f} bpm" if avg_hr else "--")
    c2.metric("❤️ MIN HR", f"{min_hr:.0f} bpm" if min_hr else "--")
    c3.metric("❤️ MAX HR", f"{max_hr:.0f} bpm" if max_hr else "--")
    c4.empty()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🩸 AVG SmO2", f"{avg_smo2:.1f}%" if avg_smo2 else "--")
    c2.metric("🩸 MIN SmO2", f"{min_smo2:.1f}%" if min_smo2 else "--")
    c3.metric("🩸 MAX SmO2", f"{max_smo2:.1f}%" if max_smo2 else "--")
    c4.empty()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🫁 AVG VE", f"{avg_ve:.1f} L/min" if avg_ve else "--")
    c2.metric("🫁 MIN VE", f"{min_ve:.1f} L/min" if min_ve else "--")
    c3.metric("🫁 MAX VE", f"{max_ve:.1f} L/min" if max_ve else "--")
    c4.empty()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💨 AVG BR", f"{avg_br:.0f} /min" if avg_br else "--")
    c2.metric("💨 MIN BR", f"{min_br:.0f} /min" if min_br else "--")
    c3.metric("💨 MAX BR", f"{max_br:.0f} /min" if max_br else "--")
    c4.empty()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Est. VO2max", f"{est_vo2max:.1f} ml/kg/min" if est_vo2max else "--")
    c2.metric("🧬 Est. VLamax", f"{est_vlamax:.2f} mmol/L/s" if est_vlamax else "--")
    c3.metric("⚡ Est. CP", f"{est_cp:.0f} W" if est_cp else "--")
    c4.metric("🔋 Est. W'", f"{est_w_prime:.0f} J" if est_w_prime else "--")


def _render_vo2max_uncertainty(df_plot: pd.DataFrame, rider_weight: float):
    """
    Estymacja VO2max z przedziałem ufności 95% (CI95%).

    Wzór Sitko et al. 2021: VO2max = 16.61 + 8.87 × 5' max power (W/kg)

    CI95% oparta na:
    - Zmienności mocy w ostatnich 5 minutach rampy (SD)
    - Stabilności odpowiedzi HR (CV)
    """
    if "watts" not in df_plot.columns:
        st.warning("⚠️ **Brak danych mocy** — nie można estymować VO2max.")
        return

    if rider_weight <= 0:
        st.warning("⚠️ **Nieprawidłowa waga zawodnika** — nie można estymować VO2max.")
        return

    if len(df_plot) < 300:
        st.warning("⚠️ **Za mało danych** (wymagane min. 5 minut) — nie można estymować VO2max.")
        return

    rolling_5min = pd.Series(df_plot["watts"].rolling(window=300, min_periods=300).mean())
    best_5min_idx_raw = rolling_5min.idxmax()
    best_5min_idx = (
        int(best_5min_idx_raw) if isinstance(best_5min_idx_raw, (int, np.integer)) else 299
    )
    mmp_5min = float(rolling_5min.max())

    best_5min_start = max(0, best_5min_idx - 299)
    df_best5 = df_plot.iloc[best_5min_start : best_5min_idx + 1]

    power_mean = mmp_5min
    power_sd = df_best5["watts"].std()
    power_cv = (power_sd / power_mean * 100) if power_mean > 0 else 0
    n = len(df_best5)

    power_per_kg = power_mean / rider_weight
    vo2max = 16.61 + 8.87 * power_per_kg

    se_power = power_sd / np.sqrt(n)
    se_vo2 = 8.87 * se_power / rider_weight
    ci95_vo2 = 1.96 * se_vo2

    hr_penalty = 0
    hr_col = resolve_hr_column(df_best5)
    hr_mean = 0.0
    hr_sd = 0.0
    hr_cv = 0.0

    if hr_col:
        hr_mean = df_best5[hr_col].mean()
        hr_sd = df_best5[hr_col].std()
        hr_cv = (hr_sd / hr_mean * 100) if hr_mean > 0 else 0
        if hr_cv > 5:
            hr_penalty = ci95_vo2 * 0.2

    ci95_total = ci95_vo2 + hr_penalty

    confidence_weight = 1 / (1 + ci95_total / vo2max) if vo2max > 0 else 0
    confidence_pct = confidence_weight * 100

    if confidence_pct >= 80:
        conf_color = "#00cc96"
        conf_label = "WYSOKA"
    elif confidence_pct >= 60:
        conf_color = "#ffa15a"
        conf_label = "UMIARKOWANA"
    else:
        conf_color = "#ef553b"
        conf_label = "NISKA"

    st.markdown(
        f"""
    <div style="padding:20px; border-radius:12px; border:3px solid #17a2b8; background-color: #1a1a1a; text-align:center;">
        <h2 style="margin:0; color: #17a2b8;">VO₂max = {vo2max:.1f} ± {ci95_total:.1f} ml/kg/min</h2>
        <p style="margin:10px 0 0 0; color:#888; font-size:0.85em;">
            (CI95%: {vo2max - ci95_total:.1f} – {vo2max + ci95_total:.1f} ml/kg/min)
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.caption(
        "📌 **Źródło:** Estymacja modelowa (Sitko et al. 2021), nie pomiar bezpośredni. Używać orientacyjnie."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
        <div style="padding:15px; border-radius:8px; border:2px solid {conf_color}; background-color: #222; text-align:center;">
            <p style="margin:0; color:#aaa; font-size:0.9em;">Waga Pewności (Confidence Weight)</p>
            <h3 style="margin:5px 0; color: {conf_color};">{confidence_pct:.0f}% — {conf_label}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with st.expander("📊 Szczegóły obliczeń", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("MMP5 (najlepsze 5 min)", f"{power_mean:.0f} W")
        c2.metric("SD mocy", f"{power_sd:.1f} W")
        c3.metric("CV mocy", f"{power_cv:.1f}%")

        if hr_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Średnie HR", f"{hr_mean:.0f} bpm")
            c2.metric("SD HR", f"{hr_sd:.1f} bpm")
            c3.metric("CV HR", f"{hr_cv:.1f}%")

        st.markdown(f"""
        | Parametr | Wartość |
        |----------|---------|
        | SE mocy | {se_power:.2f} W |
        | SE VO₂max | {se_vo2:.2f} ml/kg/min |
        | CI95% (moc) | ±{ci95_vo2:.2f} ml/kg/min |
        | Korekta HR | +{hr_penalty:.2f} ml/kg/min |
        | **CI95% całkowity** | **±{ci95_total:.2f} ml/kg/min** |
        """)

    with st.expander("📖 Metodologia estymacji VO2max", expanded=False):
        st.markdown("""
        ### Formuła Sitko et al. 2021

        ```
        VO₂max = 16.61 + 8.87 × 5' max power (W/kg)
        ```

        Gdzie:
        - `5' max power (W/kg)` = maksymalna moc 5-minutowa na kg masy ciała [W/kg]
        - `kg` = masa ciała zawodnika [kg]

        ---

        ### Przedział ufności (CI95%)

        CI95% jest obliczany na podstawie:

        1. **Zmienność mocy (SD):**
           - Wysoka zmienność = większa niepewność estymacji
           - SE = SD / √n
           - CI = 1.96 × SE × 8.87 / kg

        2. **Stabilność HR:**
           - CV HR > 5% → dodatkowa korekta +20% CI
           - Niestabilne HR może wskazywać na nieustalony stan metaboliczny

        ---

        ### Waga Pewności (Confidence Weight)

        ```
        Weight = 1 / (1 + CI/VO₂max)
        ```

        Używana do skalowania pewności wniosków centralnych:
        - **≥80%** = Wysoka pewność, wyniki wiarygodne
        - **60-80%** = Umiarkowana pewność, interpretować ostrożnie
        - **<60%** = Niska pewność, traktować orientacyjnie

        ---

        *Uwaga: Jest to estymacja modelowa, nie zastępuje bezpośredniego pomiaru VO₂max w laboratorium.*
        """)
