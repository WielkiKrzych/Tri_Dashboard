"""PMC (Performance Management Chart) Dashboard tab."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from modules.calculations.pmc import (
    calculate_pmc_history,
    predict_future_pmc,
    get_current_pmc_summary,
    get_form_interpretation,
)
from modules.ui.shared import chart, metric, require_data
from modules.plots import CHART_CONFIG


def render_pmc_tab(*args, **kwargs) -> None:
    """Render the PMC Dashboard tab."""
    st.subheader("📊 Performance Management Chart (PMC)")
    st.caption("Śledzenie formy, zmęczenia i gotowości treningowej na podstawie TSS")

    from modules.cache_utils import get_session_store

    store = get_session_store()

    _render_kpi_cards(store)
    _render_pmc_chart(store)
    _render_prediction_planner(store)
    _render_theory()


def _render_kpi_cards(store) -> None:
    summary = get_current_pmc_summary(store)
    if not summary:
        st.info("ℹ️ Brak danych treningowych. Wgraj sesje, aby zobaczyć PMC.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)
    metric("CTL (Fitness)", summary["ctl"], column=col1, suffix=" TSS/d")
    metric("ATL (Fatigue)", summary["atl"], column=col2, suffix=" TSS/d")
    metric("TSB (Form)", summary["tsb"], column=col3, suffix=" TSS/d", delta=summary["form_status"])
    metric("Ramp Rate", summary["ramp_rate"], column=col4, suffix=" %/tyg")
    metric(
        "Rekomendacja TSS",
        f"{summary['recommended_tss_min']:.0f}–{summary['recommended_tss_max']:.0f}",
        column=col5,
    )

    tsb = summary["tsb"]
    _, color = _tsb_color(tsb)
    st.markdown(
        f'<div style="background: {color}22; padding: 10px 15px; border-radius: 8px; '
        f'border-left: 4px solid {color}; margin: 10px 0;">'
        f"<b>Status formy:</b> {summary['form_status']}</div>",
        unsafe_allow_html=True,
    )


def _tsb_color(tsb: float) -> tuple:
    if tsb > 25:
        return "🟢", "#27ae60"
    elif tsb > 5:
        return "🟡", "#f39c12"
    elif tsb > -10:
        return "🟠", "#e67e22"
    elif tsb > -30:
        return "🔴", "#e74c3c"
    return "⛔", "#8e44ad"


def _render_pmc_chart(store) -> None:
    days = st.selectbox("Zakres danych", [30, 60, 90, 180], index=2, key="pmc_days")
    history = calculate_pmc_history(store, days=days)

    if not history:
        st.warning("Brak danych PMC do wyświetlenia.")
        return

    dates = [h.date for h in history]
    ctl = [h.ctl for h in history]
    atl = [h.atl for h in history]
    tsb = [h.tsb for h in history]
    tss = [h.tss for h in history]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dates,
            y=tss,
            name="TSS",
            marker_color="#6c757d",
            opacity=0.3,
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ctl,
            name="CTL (Fitness)",
            line=dict(color="#3498db", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=atl,
            name="ATL (Fatigue)",
            line=dict(color="#e74c3c", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=tsb,
            name="TSB (Form)",
            line=dict(color="#27ae60", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="PMC — Performance Management Chart",
        xaxis_title="Data",
        yaxis_title="TSS / obciążenie",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    chart(fig, key="pmc_main_chart")


def _render_prediction_planner(store) -> None:
    st.markdown("### 🔮 Planer — Prognoza PMC")
    st.caption("Wpisz planowane TSS na najbliższe 7 dni, aby zobaczyć prognozę formy.")

    summary = get_current_pmc_summary(store)
    if not summary:
        st.info("Brak aktualnych danych PMC do prognozy.")
        return

    cols = st.columns(7)
    planned = []
    today = datetime.now().date()
    day_labels = []

    for i, col in enumerate(cols):
        day = today + timedelta(days=i + 1)
        day_labels.append(day.strftime("%Y-%m-%d"))
        val = col.number_input(
            day.strftime("%a\n%m/%d"),
            min_value=0,
            max_value=400,
            value=int(summary["ctl"]),
            key=f"pmc_plan_{i}",
        )
        planned.append(float(val))

    predictions = predict_future_pmc(
        summary["ctl"],
        summary["atl"],
        planned,
        day_labels,
    )

    if predictions:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[p.date for p in predictions],
                y=[p.ctl for p in predictions],
                name="CTL (prognoza)",
                line=dict(color="#3498db", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[p.date for p in predictions],
                y=[p.atl for p in predictions],
                name="ATL (prognoza)",
                line=dict(color="#e74c3c", width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[p.date for p in predictions],
                y=[p.tsb for p in predictions],
                name="TSB (prognoza)",
                line=dict(color="#27ae60", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title="Prognoza PMC — 7 dni",
            xaxis_title="Data",
            yaxis_title="TSS/d",
            height=350,
            hovermode="x unified",
        )
        chart(fig, key="pmc_prediction_chart")


def _render_theory() -> None:
    with st.expander("📖 Teoria — PMC (Performance Management Chart)"):
        st.markdown("""
        **PMC** to model zarządzania formą oparty na koncepcji Coggana:

        | Metryka | Pełna nazwa | Okno | Znaczenie |
        |---------|------------|------|-----------|
        | **CTL** | Chronic Training Load | 42 dni (EWMA) | Fitness — długoterminowa adaptacja |
        | **ATL** | Acute Training Load | 7 dni (EWMA) | Fatigue — aktualne zmęczenie |
        | **TSB** | Training Stress Balance | CTL − ATL | Form — gotowość do wysiłku |

        **Interpretacja TSB:**
        - **TSB > 25**: Świeży — peak form, idealny na zawody
        - **TSB 5-25**: Gotowy — dobry dzień na mocny trening
        - **TSB -10 do 5**: Optymalne obciążenie — budowanie formy
        - **TSB -30 do -10**: Zmęczony — regeneracja
        - **TSB < -30**: Przepracowany — ryzyko kontuzji/overtraining

        **Ramp Rate**: tygodniowa zmiana CTL. Bezpieczny: 3-7%/tyg.
        """)
