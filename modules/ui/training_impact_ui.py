"""Aerobic vs Anaerobic Training Impact Decomposition tab."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from modules.calculations.training_impact import (
    calculate_session_impact,
    classify_session_intensity,
    calculate_rolling_impact,
    TrainingImpact,
)
from modules.ui.shared import chart, metric


_INTENSITY_LABELS = {
    "recovery": ("🟢 Regeneracja", "#27ae60"),
    "endurance": ("🔵 Wytrzymałość", "#3498db"),
    "tempo": ("🟡 Tempo", "#f1c40f"),
    "threshold": ("🟠 Próg", "#e67e22"),
    "vo2max": ("🔴 VO2max", "#e74c3c"),
    "anaerobic": ("⛔ Anaerobowy", "#8e44ad"),
}


def render_training_impact_tab(df_plot, cp_input, w_prime_input, *args, **kwargs) -> None:
    """Render the Training Impact Decomposition tab."""
    st.subheader("🎯 Dekompozycja Treningu — Aerobowy vs Anaerobowy")
    st.caption("Podział obciążenia treningowego na komponenty tlenowy i beztlenowy")

    if df_plot is None or (hasattr(df_plot, "empty") and df_plot.empty):
        st.info("ℹ️ Wgraj plik z danymi, aby zobaczyć dekompozycję treningu.")
        return

    cp = cp_input if cp_input and cp_input > 0 else 250
    w_prime = w_prime_input if w_prime_input and w_prime_input > 0 else 20000

    impact = calculate_session_impact(df_plot, cp, w_prime)
    if not impact:
        st.warning("Nie udało się obliczyć dekompozycji treningu.")
        return

    _render_impact_cards(impact)
    _render_impact_chart(impact)
    _render_distribution_chart(impact)
    _render_theory()


def _render_impact_cards(impact: TrainingImpact) -> None:
    col1, col2, col3, col4 = st.columns(4)
    metric("TSS Aerobowy", impact.aerobic_tss, column=col1, suffix=" TSS")
    metric("TSS Anaerobowy", impact.anaerobic_tss, column=col2, suffix=" TSS")
    metric("TSS Całkowity", impact.total_tss, column=col3, suffix=" TSS")
    metric("Frakcja Aerobowa", f"{impact.aerobic_fraction:.0%}", column=col4)

    label, color = _INTENSITY_LABELS.get(impact.intensity_type, ("❓ Nieznany", "#95a5a6"))
    st.markdown(
        f'<div style="background: {color}22; padding: 10px 15px; border-radius: 8px; '
        f'border-left: 4px solid {color}; margin: 10px 0;">'
        f"<b>Klasyfikacja sesji:</b> {label} (aerobowy {impact.aerobic_fraction:.0%})"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_impact_chart(impact: TrainingImpact) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Aerobowy",
            x=["Sesja"],
            y=[impact.aerobic_tss],
            marker_color="#3498db",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Anaerobowy",
            x=["Sesja"],
            y=[impact.anaerobic_tss],
            marker_color="#e74c3c",
        )
    )
    fig.update_layout(
        barmode="stack",
        title="Dekompozycja TSS — Sesja",
        yaxis_title="TSS",
        height=350,
    )
    chart(fig, key="training_impact_stacked")


def _render_distribution_chart(impact: TrainingImpact) -> None:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Aerobowy", "Anaerobowy"],
                values=[impact.aerobic_tss, impact.anaerobic_tss],
                marker=dict(colors=["#3498db", "#e74c3c"]),
                hole=0.4,
            )
        ]
    )
    fig.update_layout(
        title="Rozkład Treningu",
        height=350,
    )
    chart(fig, key="training_impact_pie")


def _render_theory() -> None:
    with st.expander("📖 Teoria — Dekompozycja Treningu"):
        st.markdown("""
        **Dekompozycja TSS** dzieli obciążenie na:

        | Strefa | Frakcja aerobowa | Opis |
        |--------|------------------|------|
        | **Regeneracja** | > 90% | Bardzo lekki wysiłek |
        | **Wytrzymałość** | 75-90% | Strefa Z2, budowanie bazy |
        | **Tempo** | 60-75% | Strefa Z3, trening endurance |
        | **Próg** | 45-60% | Z4, trening przy CP/FTP |
        | **VO2max** | 30-45% | Z5, interwały powyżej CP |
        | **Anaerobowy** | < 30% | Z6+, sprinty, ataki |
        """)
