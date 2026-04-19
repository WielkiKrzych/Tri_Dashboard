"""Running Dynamics UI tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.calculations.running_dynamics import (
    RunningDynamicsMetrics,
    calculate_running_dynamics,
    classify_running_economics,
    get_ideal_ranges,
)
from modules.ui.shared import chart, metric


def render_running_dynamics_tab(*args, **kwargs) -> None:
    st.subheader("🏃 Dynamika Biegowa")
    st.caption("Analiza ekonomii biegu: czas kontaktu, oscylacja pionowa, sztywność nóg")

    _render_pace_lookup()
    _render_data_input()
    _render_theory()


def _render_pace_lookup() -> None:
    st.markdown("### 📊 Idealne zakresy dla tempa")
    pace = st.number_input(
        "Tempo (min/km)", min_value=3.0, max_value=10.0, value=5.0, step=0.1, key="rd_pace"
    )
    ranges = get_ideal_ranges(pace)

    col1, col2, col3 = st.columns(3)
    with col1:
        gct_lo, gct_hi = ranges["ground_contact_time_ms"]
        metric("Czas kontaktu", f"{gct_lo:.0f}–{gct_hi:.0f} ms")
        vo_lo, vo_hi = ranges["vertical_oscillation_mm"]
        metric("Oscylacja pionowa", f"{vo_lo:.1f}–{vo_hi:.1f} mm")
    with col2:
        lss_lo, lss_hi = ranges["leg_spring_stiffness_kn_m"]
        metric("Sztywność nóg", f"{lss_lo:.1f}–{lss_hi:.1f} kN/m")
        vr_lo, vr_hi = ranges["vertical_ratio_pct"]
        metric("Vertical ratio", f"{vr_lo:.1f}–{vr_hi:.1f} %")
    with col3:
        cad_lo, cad_hi = ranges["cadence_spm"]
        metric("Kadencja", f"{cad_lo:.0f}–{cad_hi:.0f} spm")


def _render_data_input() -> None:
    st.markdown("### 📈 Analiza z danych")

    has_accel = _check_accel_data()

    if not has_accel:
        st.info("""
        ℹ️ **Brak danych akcelerometru w tej sesji.**

        Aby korzystać z pełnej analizy dynamiki biegowej, wgraj plik z danymi akcelerometru
        zawierający kolumny: `accel_x`, `accel_y`, `accel_z`, `time`.

        Wspierane formaty: CSV z Garmin, Polar, Stryd, lub własne dane.

        Możesz też użyć **kalkulatora tempa** powyżej do sprawdzenia idealnych zakresów.
        """)

        _render_manual_input()
        return

    df = st.session_state.get("df_plot")
    if df is None:
        return

    result = calculate_running_dynamics(
        accel_x=df["accel_x"].values,
        accel_y=df["accel_y"].values,
        accel_z=df["accel_z"].values,
        time_arr=df["time"].values if "time" in df.columns else df.index.values,
        body_mass_kg=st.session_state.get("rider_weight", 75.0),
    )

    if result:
        _render_metrics(result)
        _render_gauge_charts(result)


def _check_accel_data() -> bool:
    df = st.session_state.get("df_plot")
    if df is None:
        return False
    required = {"accel_x", "accel_y", "accel_z"}
    return required.issubset(set(df.columns))


def _render_manual_input() -> None:
    with st.expander("🔧 Ręczne wprowadzenie metryk"):
        col1, col2 = st.columns(2)
        with col1:
            gct = st.number_input("Czas kontaktu (ms)", 100, 500, 250, key="rd_gct")
            vo = st.number_input("Oscylacja pionowa (mm)", 1.0, 25.0, 8.0, 0.5, key="rd_vo")
            lss = st.number_input("Sztywność nóg (kN/m)", 1.0, 30.0, 10.0, 0.5, key="rd_lss")
        with col2:
            cad = st.number_input("Kadencja (spm)", 120, 220, 175, key="rd_cad")
            sl = st.number_input("Długość kroku (m)", 0.5, 2.5, 1.2, 0.05, key="rd_sl")

        vr = (vo / 10.0 / sl * 100) if sl > 0 else 0.0
        metrics = RunningDynamicsMetrics(
            ground_contact_time_ms=gct,
            vertical_oscillation_mm=vo,
            leg_spring_stiffness_kn_m=lss,
            vertical_ratio_pct=round(vr, 1),
            cadence_spm=cad,
            stride_length_m=sl,
        )
        _render_metrics(metrics)
        _render_gauge_charts(metrics)


def _render_metrics(m: RunningDynamicsMetrics) -> None:
    col1, col2, col3 = st.columns(3)
    metric("Czas kontaktu", f"{m.ground_contact_time_ms:.0f} ms", column=col1)
    metric("Oscylacja pionowa", f"{m.vertical_oscillation_mm:.1f} mm", column=col2)
    metric("Sztywność nóg", f"{m.leg_spring_stiffness_kn_m:.1f} kN/m", column=col3)

    col4, col5, col6 = st.columns(3)
    metric("Vertical ratio", f"{m.vertical_ratio_pct:.1f} %", column=col4)
    metric("Kadencja", f"{m.cadence_spm:.0f} spm", column=col5)
    metric("Długość kroku", f"{m.stride_length_m:.2f} m", column=col6)

    classification = classify_running_economics(m)
    st.markdown(f"**Klasyfikacja:** {classification}")


def _render_gauge_charts(m: RunningDynamicsMetrics) -> None:
    gauges = [
        ("Czas kontaktu (ms)", m.ground_contact_time_ms, 100, 400, 250),
        ("Oscylacja pionowa (mm)", m.vertical_oscillation_mm, 2, 20, 8),
        ("Sztywność nóg (kN/m)", m.leg_spring_stiffness_kn_m, 2, 25, 10),
    ]

    cols = st.columns(len(gauges))
    for i, (title, value, vmin, vmax, optimal) in enumerate(gauges):
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title},
                gauge={
                    "axis": {"range": [vmin, vmax]},
                    "bar": {"color": "#3498db"},
                    "steps": [
                        {"range": [vmin, optimal], "color": "#27ae6022"},
                        {"range": [optimal, vmax], "color": "#e74c3c22"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": optimal,
                    },
                },
            )
        )
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
        with cols[i]:
            chart(fig, key=f"rd_gauge_{i}")


def _render_theory() -> None:
    with st.expander("📖 Teoria — Dynamika biegowa"):
        st.markdown("""
        **Dynamika biegowa** opisuje biomechaniczne parametry kroku biegowego:

        | Metryka | Optymalna | Znaczenie |
        |---------|-----------|-----------|
        | Czas kontaktu | 200-260 ms | Krótszy = lepsza ekonomia |
        | Oscylacja pionowa | 5-10 mm | Mniejsza = mniej zmarnowanej energii |
        | Sztywność nóg | 8-14 kN/m | Wyższa = lepsze wykorzystanie elastyczności |
        | Vertical ratio | 5-9% | Oscylacja / długość kroku — im mniej tym lepiej |
        | Kadencja | 170-185 spm | Zbyt niska = overstriding |

        **Praktyka:**
        - Kadencję zwiększaj stopniowo (5% na tydzień)
        - Oscylację pionową redukuj poprzez aktywację core
        - Czas kontaktu poprawiaj poprzez treningi plyometryczne
        """)
