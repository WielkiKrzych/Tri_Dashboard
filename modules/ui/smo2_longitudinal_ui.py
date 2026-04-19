"""SmO2 NIRS Longitudinal Threshold Tracker tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.calculations.smo2_longitudinal import (
    extract_session_thresholds,
    calculate_longitudinal_trend,
    interpret_trend,
    SmO2SessionThreshold,
    SmO2LongitudinalTrend,
)
from modules.ui.shared import chart, metric


def render_smo2_longitudinal_tab(df_plot, cp_input, *args, **kwargs) -> None:
    """Render the SmO2 Longitudinal Tracker tab."""
    st.subheader("📈 Longitudinalne Śledzenie Progów SmO2")
    st.caption("Śledzenie zmian progów tlenowych wykrywanych z NIRS (SmO2) w czasie")

    if df_plot is None or (hasattr(df_plot, "empty") and df_plot.empty):
        st.info("ℹ️ Wgraj plik z danymi SmO2, aby zobaczyć analizę longitudinalną.")
        return

    cp = cp_input if cp_input and cp_input > 0 else None

    current = _extract_current_threshold(df_plot, cp)
    if current:
        _render_current_threshold(current)

    historical = _get_historical_thresholds()
    if current:
        historical.append(current)

    if len(historical) >= 1:
        _render_trend(historical)
    else:
        st.warning("Brak danych progowych SmO2 do analizy.")

    _render_theory()


def _extract_current_threshold(df, cp) -> SmO2SessionThreshold | None:
    session_data = {"date": "Aktualna sesja", "df": df, "cp": cp}
    try:
        return extract_session_thresholds(session_data)
    except Exception:
        return None


def _get_historical_thresholds() -> list:
    try:
        from modules.cache_utils import get_session_store

        store = get_session_store()
        sessions = store.get_sessions()
    except Exception:
        return []

    thresholds = []
    for session in sessions or []:
        try:
            date = session.get("date", "")
            metrics_json = session.get("metrics", {})
            if isinstance(metrics_json, str):
                import json

                metrics_json = json.loads(metrics_json)

            t1_w = metrics_json.get("smo2_t1_watts")
            t1_s = metrics_json.get("smo2_t1_smo2")
            t2_w = metrics_json.get("smo2_t2_watts")
            t2_s = metrics_json.get("smo2_t2_smo2")

            if t1_w is not None:
                thresholds.append(
                    SmO2SessionThreshold(
                        date=date,
                        t1_power=float(t1_w),
                        t1_smo2=float(t1_s) if t1_s else None,
                        t2_power=float(t2_w) if t2_w else None,
                        t2_smo2=float(t2_s) if t2_s else None,
                        quality_grade="B",
                    )
                )
        except Exception:
            continue

    return thresholds


def _render_current_threshold(t: SmO2SessionThreshold) -> None:
    col1, col2, col3, col4 = st.columns(4)
    metric("T1 Moc", t.t1_power, column=col1, suffix=" W")
    metric("T1 SmO2", t.t1_smo2, column=col2, suffix=" %")
    metric("T2 Moc", t.t2_power, column=col3, suffix=" W")
    metric("T2 SmO2", t.t2_smo2, column=col4, suffix=" %")

    grade_colors = {"A": "#27ae60", "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c"}
    color = grade_colors.get(t.quality_grade, "#95a5a6")
    st.markdown(
        f'<div style="background: {color}22; padding: 8px 15px; border-radius: 8px; '
        f'border-left: 4px solid {color}; margin: 10px 0;">'
        f"<b>Jakość detekcji:</b> {t.quality_grade}</div>",
        unsafe_allow_html=True,
    )


def _render_trend(thresholds: list) -> None:
    min_grade = st.selectbox(
        "Minimalna jakość sesji",
        ["A", "B", "C"],
        index=1,
        key="smo2_long_grade",
    )

    trend = calculate_longitudinal_trend(thresholds, min_grade=min_grade)

    direction_icons = {
        "improving": "⬆️ Poprawa progów",
        "stable": "➡️ Stabilne progi",
        "declining": "⬇️ Spadek progów",
    }
    st.markdown(
        f'<div style="font-size: 1.3em; padding: 10px;">'
        f"{direction_icons.get(trend.direction, '➡️ Brak trendu')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    filtered = [
        t for t in trend.sessions if t.t1_power is not None and t.quality_grade <= min_grade
    ]

    if len(filtered) >= 2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[t.date for t in filtered],
                y=[t.t1_power for t in filtered],
                mode="lines+markers",
                name="T1 Power",
                line=dict(color="#3498db", width=2),
                marker=dict(size=8),
            )
        )

        if any(t.t2_power for t in filtered):
            t2_dates = [t.date for t in filtered if t.t2_power]
            t2_powers = [t.t2_power for t in filtered if t.t2_power]
            if t2_dates:
                fig.add_trace(
                    go.Scatter(
                        x=t2_dates,
                        y=t2_powers,
                        mode="lines+markers",
                        name="T2 Power",
                        line=dict(color="#e74c3c", width=2, dash="dash"),
                        marker=dict(size=8),
                    )
                )

        fig.update_layout(
            title="Trend Progów SmO2 w Czasie",
            xaxis_title="Data",
            yaxis_title="Moc (W)",
            height=400,
            hovermode="x unified",
        )
        chart(fig, key="smo2_longitudinal_trend")

    interpretation = interpret_trend(trend)
    st.info(f"📊 {interpretation}")


def _render_theory() -> None:
    with st.expander("📖 Teoria — Longitudinalne Śledzenie SmO2"):
        st.markdown("""
        **SmO2 NIRS** pozwala na ni inwazyjne śledzenie progów metabolicznych:

        | Próg | Fizjologia | Znaczenie |
        |------|------------|-----------|
        | **T1** | LT1 analog — początek desaturacji | Próg tlenowy, strefa Z2/Z3 |
        | **T2** | LT2/RCP analog — szybka desaturacja | Próg beztlenowy, strefa Z4 |

        **Analiza longitudinalna** śledzi zmiany mocy przy progu w czasie:
        - **⬆️ Poprawa**: Progi przesuwają się w górę — adaptacja do treningu
        - **➡️ Stabilny**: Utrzymanie poziomu — podstawa do budowania
        - **⬇️ Spadek**: Może wskazywać na przepracowanie lub detrenning

        **Jakość detekcji:**
        - **A**: Obie progi wykryte (T1 + T2)
        - **B**: Tylko T1 wykryte
        - **C**: Tylko T2 wykryte
        - **D**: Brak wyraźnych progów
        """)
