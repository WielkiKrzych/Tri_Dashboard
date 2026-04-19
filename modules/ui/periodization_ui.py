"""Periodization Planner UI tab."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.calculations.periodization import (
    TrainingBlock,
    PeriodizationPlan,
    create_periodization_plan,
    validate_plan,
    get_weekly_schedule,
)
from modules.ui.shared import chart, metric


def render_periodization_tab(*args, **kwargs) -> None:
    st.subheader("📅 Planer Periodyzacji")
    st.caption("Planuj sezon z blokami Base/Build/Peak/Race i prognozą PMC")

    from modules.cache_utils import get_session_store

    store = get_session_store()

    plan = _render_plan_config(store)
    if plan:
        _render_gantt_chart(plan)
        _render_weekly_targets(plan)
        _render_pmc_overlay(plan, store)
        _render_validation(plan)
        _render_export(plan)
    _render_theory()


def _render_plan_config(store) -> PeriodizationPlan | None:
    from modules.calculations.pmc import get_current_pmc_summary

    pmc = get_current_pmc_summary(store)

    st.markdown("### ⚙️ Konfiguracja planu")
    col1, col2, col3 = st.columns(3)

    with col1:
        default_race = datetime.now() + timedelta(weeks=12)
        race_date = st.date_input("Data zawodów", default_race, key="per_race_date")
    with col2:
        total_weeks = st.slider("Czas trwania (tygodnie)", 8, 24, 12, key="per_weeks")
    with col3:
        ctl = pmc["ctl"] if pmc else 50.0
        atl = pmc["atl"] if pmc else 40.0
        ctl = st.number_input("Aktualny CTL", 10.0, 200.0, float(ctl), key="per_ctl")
        atl = st.number_input("Aktualny ATL", 5.0, 200.0, float(atl), key="per_atl")

    if race_date is None:
        return None

    return create_periodization_plan(
        race_date=datetime.combine(race_date, datetime.min.time()),
        current_ctl=ctl,
        current_atl=atl,
        total_weeks=total_weeks,
    )


def _render_gantt_chart(plan: PeriodizationPlan) -> None:
    st.markdown("### 📊 Gantt — Bloki treningowe")

    colors = {
        "Base": "#3498db",
        "Build": "#e74c3c",
        "Peak": "#f39c12",
        "Race": "#27ae60",
    }

    fig = go.Figure()

    for block in plan.blocks:
        fig.add_trace(
            go.Bar(
                x=[
                    (
                        datetime.strptime(block.end_date, "%Y-%m-%d")
                        - datetime.strptime(block.start_date, "%Y-%m-%d")
                    ).days
                    + 1
                ],
                y=[block.block_type],
                base=[datetime.strptime(block.start_date, "%Y-%m-%d").strftime("%Y-%m-%d")],
                orientation="h",
                name=block.block_type,
                marker_color=colors.get(block.block_type, "#95a5a6"),
                text=f"TSS: {block.target_tss_low:.0f}–{block.target_tss_high:.0f}",
                textposition="inside",
            )
        )

    fig.update_layout(
        title="Periodyzacja — bloki treningowe",
        xaxis_title="Dni",
        yaxis_title="Blok",
        height=250,
        barmode="stack",
        showlegend=True,
    )
    chart(fig, key="per_gantt_chart")


def _render_weekly_targets(plan: PeriodizationPlan) -> None:
    st.markdown("### 📋 Tygodniowy rozkład TSS")

    selected_block_type = st.selectbox(
        "Wybierz blok",
        [b.block_type for b in plan.blocks],
        key="per_block_select",
    )

    block = next((b for b in plan.blocks if b.block_type == selected_block_type), plan.blocks[0])
    schedule = get_weekly_schedule(block)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"**{block.block_type}** ({block.start_date} → {block.end_date})")
        st.caption(block.focus)
        metric("TSS min/dzień", f"{block.target_tss_low:.0f}")
        metric("TSS max/dzień", f"{block.target_tss_high:.0f}")

    with col2:
        fig = go.Figure()
        days = [s["day"] for s in schedule]
        tss_vals = [s["tss"] for s in schedule]
        colors_bar = [
            "#e74c3c"
            if s["intensity"] == "Wysoka"
            else "#f39c12"
            if s["intensity"] == "Umiarkowana"
            else "#27ae60"
            if s["intensity"] == "Niska"
            else "#95a5a6"
            for s in schedule
        ]

        fig.add_trace(go.Bar(x=days, y=tss_vals, marker_color=colors_bar, name="TSS"))
        fig.add_hline(
            y=block.target_tss_low,
            line_dash="dash",
            line_color="#3498db",
            annotation_text=f"Min: {block.target_tss_low:.0f}",
        )
        fig.add_hline(
            y=block.target_tss_high,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Max: {block.target_tss_high:.0f}",
        )

        fig.update_layout(
            title=f"Tygodniowy szablon — {block.block_type}",
            xaxis_title="Dzień",
            yaxis_title="TSS",
            height=300,
            showlegend=False,
        )
        chart(fig, key="per_weekly_chart")


def _render_pmc_overlay(plan: PeriodizationPlan, store) -> None:
    st.markdown("### 🔮 Prognoza PMC na planie")

    from modules.calculations.pmc import get_current_pmc_summary, predict_future_pmc

    pmc = get_current_pmc_summary(store)
    if not pmc:
        st.info("Brak danych PMC do prognozy.")
        return

    total_days = plan.total_weeks * 7
    planned_tss = []

    for i in range(total_days):
        day_date = datetime.strptime(plan.start_date, "%Y-%m-%d") + timedelta(days=i)
        day_str = day_date.strftime("%Y-%m-%d")

        tss = pmc["ctl"]
        for block in plan.blocks:
            if block.start_date <= day_str <= block.end_date:
                tss = (block.target_tss_low + block.target_tss_high) / 2
                break

        planned_tss.append(tss)

    dates = [
        (datetime.strptime(plan.start_date, "%Y-%m-%d") + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(total_days)
    ]

    preds = predict_future_pmc(pmc["ctl"], pmc["atl"], planned_tss, dates)

    if not preds:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p.date for p in preds],
            y=[p.ctl for p in preds],
            name="CTL (prognoza)",
            line=dict(color="#3498db", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p.date for p in preds],
            y=[p.atl for p in preds],
            name="ATL (prognoza)",
            line=dict(color="#e74c3c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p.date for p in preds],
            y=[p.tsb for p in preds],
            name="TSB (prognoza)",
            line=dict(color="#27ae60", width=2, dash="dash"),
        )
    )

    for block in plan.blocks:
        fig.add_vrect(
            x0=block.start_date,
            x1=block.end_date,
            fillcolor={
                "Base": "#3498db",
                "Build": "#e74c3c",
                "Peak": "#f39c12",
                "Race": "#27ae60",
            }.get(block.block_type, "#95a5a6"),
            opacity=0.1,
            annotation_text=block.block_type,
        )

    fig.update_layout(
        title="Prognoza PMC na planie periodyzacji",
        xaxis_title="Data",
        yaxis_title="TSS/d",
        height=400,
        hovermode="x unified",
    )
    chart(fig, key="per_pmc_overlay")


def _render_validation(plan: PeriodizationPlan) -> None:
    warnings = validate_plan(plan)
    if warnings:
        st.markdown("### ⚠️ Ostrzeżenia")
        for w in warnings:
            st.warning(w)
    else:
        st.success("✅ Plan wygląda rozsądnie — brak ostrzeżeń.")


def _render_export(plan: PeriodizationPlan) -> None:
    st.markdown("### 📥 Eksport planu")

    rows = []
    for block in plan.blocks:
        schedule = get_weekly_schedule(block)
        for s in schedule:
            rows.append(
                {
                    "Blok": block.block_type,
                    "Od": block.start_date,
                    "Do": block.end_date,
                    "Dzień": s["day"],
                    "TSS": s["tss"],
                    "Intensywność": s["intensity"],
                    "Cel": block.focus,
                }
            )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True, height=min(400, len(df) * 35 + 50))


def _render_theory() -> None:
    with st.expander("📖 Teoria — Periodyzacja treningowa"):
        st.markdown("""
        **Periodyzacja** to systematyczne planowanie obciążeń treningowych:

        | Blok | Proporcja | TSS | Cel |
        |------|-----------|-----|-----|
        | **Base** | 60% | CTL × 0.8-1.0 | Baza tlenowa, wytrzymałość |
        | **Build** | 25% | CTL × 1.0-1.3 | Intensyfikacja, progowe |
        | **Peak** | 10% | CTL × 0.6-0.8 | Taper — ostryLoading przed szczytem |
        | **Race** | 5% | CTL × 0.3-0.5 | Aktywacja i zawody |

        **Kluczowe zasady:**
        - Ramp rate CTL: 3-7%/tyg (bezpieczny)
        - Taper: redukcja objętości 40-60% przez 1-2 tyg
        - Utrzymuj intensywność podczas taperu
        - Każdy 3-4 tydzień: tydzień regeneracyjny (−30% TSS)
        """)
