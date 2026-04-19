"""HRV Readiness Score tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.calculations.hrv_readiness import (
    calculate_readiness_score,
    get_readiness_level,
    ReadinessScore,
)
from modules.ui.shared import chart, metric


def render_hrv_readiness_tab(df_plot, *args, **kwargs) -> None:
    """Render the HRV Readiness tab."""
    st.subheader("🌅 Gotowość Treningowa (HRV Readiness)")
    st.caption("Ocena gotowości do treningu na podstawie zmienności rytmu zatokowego")

    rmssd_history = _extract_rmssd_history(df_plot)

    if not rmssd_history:
        st.info("ℹ️ Brak danych HRV. Wgraj plik z danymi RR lub HRV, aby zobaczyć ocenę gotowości.")
        return

    score = calculate_readiness_score(rmssd_history)
    if not score:
        st.warning("Niewystarczające dane HRV do oceny gotowości.")
        return

    _render_traffic_light(score)
    _render_gauge(score)
    _render_rmssd_trend(rmssd_history, score)
    _render_recommendation(score)
    _render_theory()


def _extract_rmssd_history(df) -> list:
    history = []

    rmssd_col = None
    for col in ["rmssd", "RMSSD", "hrv_rmssd"]:
        if col in df.columns:
            rmssd_col = col
            break

    if rmssd_col is None:
        return history

    values = df[rmssd_col].dropna().values
    if len(values) == 0:
        return history

    date_col = None
    for col in ["date", "datetime", "time", "timestamp"]:
        if col in df.columns:
            date_col = col
            break

    for i, val in enumerate(values):
        date_str = ""
        if date_col and i < len(df):
            date_str = str(df[date_col].iloc[i])
        elif date_col is None:
            date_str = f"Day {i + 1}"

        try:
            rmssd_val = float(val)
            if rmssd_val > 0:
                history.append({"date": date_str, "rmssd": rmssd_val})
        except (ValueError, TypeError):
            continue

    return history


def _render_traffic_light(score: ReadinessScore) -> None:
    icons = {
        "Wysoka gotowość": "🟢",
        "Gotowość umiarkowana": "🟡",
        "Obniżona gotowość": "🟠",
        "Niska gotowość": "🔴",
        "Krytycznie niska": "⛔",
    }
    icon = icons.get(score.level, "⚪")
    st.markdown(
        f'<div style="text-align: center; font-size: 4em; padding: 20px;">{icon}</div>'
        f'<div style="text-align: center; font-size: 1.5em; color: {score.color};">'
        f"<b>{score.level}</b></div>",
        unsafe_allow_html=True,
    )


def _render_gauge(score: ReadinessScore) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score.score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Gotowość Treningowa"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": score.color},
                "steps": [
                    {"range": [0, 20], "color": "#8e44ad22"},
                    {"range": [20, 40], "color": "#e74c3c22"},
                    {"range": [40, 60], "color": "#e67e2222"},
                    {"range": [60, 80], "color": "#f39c1222"},
                    {"range": [80, 100], "color": "#27ae6022"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": score.score,
                },
            },
        )
    )
    fig.update_layout(height=300)
    chart(fig, key="hrv_readiness_gauge")


def _render_rmssd_trend(rmssd_history: list, score: ReadinessScore) -> None:
    dates = [h["date"] for h in rmssd_history[-14:]]
    values = [h["rmssd"] for h in rmssd_history[-14:]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            name="RMSSD",
            line=dict(color="#2ecc71", width=2),
        )
    )

    avg = score.rmssd_7day_avg
    if avg > 0:
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color="#f39c12",
            annotation_text=f"Średnia 7d: {avg:.1f} ms",
        )

    fig.update_layout(
        title="Trend RMSSD (14 dni)",
        xaxis_title="Data",
        yaxis_title="RMSSD (ms)",
        height=350,
    )
    chart(fig, key="hrv_readiness_trend")


def _render_recommendation(score: ReadinessScore) -> None:
    st.markdown(
        f'<div style="background: {score.color}15; padding: 15px; border-radius: 8px; '
        f'border-left: 4px solid {score.color}; margin: 15px 0;">'
        f"<b>📊 Metryki:</b> RMSSD 7d avg = {score.rmssd_7day_avg:.1f} ms | "
        f"CV = {score.rmssd_cv:.1f}%<br>"
        f"<b>💡 Rekomendacja:</b> {score.recommendation}</div>",
        unsafe_allow_html=True,
    )


def _render_theory() -> None:
    with st.expander("📖 Teoria — HRV Readiness"):
        st.markdown("""
        **Gotowość treningowa** bazuje na wskaźniku RMSSD z ostatnich 7 dni:

        | Poziom | Wynik | Znaczenie |
        |--------|-------|-----------|
        | 🟢 Wysoka | 80-100 | Organizm gotowy na intensywny trening |
        | 🟡 Umiarkowana | 60-80 | Dobry stan, trening w strefie tempa |
        | 🟠 Obniżona | 40-60 | Lekkie zmęczenie, trening Z1-Z2 |
        | 🔴 Niska | 20-40 | Znaczne zmęczenie, regeneracja |
        | ⛔ Krytyczna | 0-20 | Poważne przepracowanie, odpoczynek |

        **Metoda:** Obliczamy współczynnik zmienności (CV) RMSSD z 7 dni.
        Niskie CV = stabilny HRV = wysoka gotowość.
        """)
