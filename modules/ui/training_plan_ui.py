"""Training Plan Builder UI tab."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

import plotly.graph_objects as go
import streamlit as st

from modules.training_plan import (
    PERIODIZATION_TEMPLATES,
    TrainingPlan,
    TrainingPlanStore,
    WorkoutType,
    build_plan,
    apply_template_to_plan,
)
from modules.training_plan.periodization import (
    template_display_name,
    validate_plan_tss_ramp,
)
from modules.training_plan.plan_builder import _DEFAULT_WEEKLY_TEMPLATE
from modules.export.plan_exporter import export_plan_csv, export_plan_weekly_summary
from modules.plots import CHART_CONFIG


DAY_NAMES = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Ndz"]

_WORKOUT_TYPE_OPTIONS = [wt for wt in WorkoutType]
_WORKOUT_TYPE_LABELS = {wt: f"{wt.emoji} {wt.value}" for wt in WorkoutType}


def _get_store() -> TrainingPlanStore:
    if "_tp_store" not in st.session_state:
        st.session_state["_tp_store"] = TrainingPlanStore()
    return st.session_state["_tp_store"]


def _get_template_from_state() -> Dict[int, WorkoutType]:
    template: Dict[int, WorkoutType] = {}
    for dow in range(7):
        key = f"_tp_day_{dow}"
        if key in st.session_state:
            template[dow] = st.session_state[key]
        else:
            template[dow] = _DEFAULT_WEEKLY_TEMPLATE.get(dow, WorkoutType.REST)
    return template


def _predict_pmc(plan: TrainingPlan) -> go.Figure:
    """Predict CTL/ATL/TSB over the plan duration using EWMA."""
    atl_decay = 2 / 8
    ctl_decay = 2 / 43

    atl = plan.weekly_tss_baseline / 7 * 0.7
    ctl = plan.weekly_tss_baseline / 7 * 0.6

    dates: list[str] = []
    daily_tss_list: list[float] = []
    atl_list: list[float] = []
    ctl_list: list[float] = []
    tsb_list: list[float] = []

    for week in plan.weeks:
        for day in week.days:
            tss = day.tss
            atl = atl * (1 - atl_decay) + tss * atl_decay
            ctl = ctl * (1 - ctl_decay) + tss * ctl_decay
            tsb = ctl - atl

            dates.append(day.date.isoformat())
            daily_tss_list.append(tss)
            atl_list.append(round(atl, 1))
            ctl_list.append(round(ctl, 1))
            tsb_list.append(round(tsb, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=ctl_list, name="CTL (Fitness)", line=dict(color="#00cc96", width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=atl_list, name="ATL (Zmęczenie)", line=dict(color="#ef553b", width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=tsb_list,
            name="TSB (Forma)",
            line=dict(color="#ab63fa", width=2, dash="dash"),
        )
    )
    fig.add_bar(
        x=dates,
        y=daily_tss_list,
        name="TSS dzienny",
        marker_color="rgba(255,255,255,0.15)",
        yaxis="y2",
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Predykcja PMC", font=dict(size=20, color="#f0f6fc")),
        yaxis=dict(title="CTL / ATL / TSB", gridcolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(title="TSS", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def _render_calendar(plan: TrainingPlan) -> None:
    """Render a navigable weekly calendar view."""
    total_weeks = plan.total_weeks
    if total_weeks == 0:
        return

    if "_tp_cal_week" not in st.session_state:
        st.session_state["_tp_cal_week"] = 0

    col_prev, col_info, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀", disabled=st.session_state["_tp_cal_week"] <= 0):
            st.session_state["_tp_cal_week"] -= 1
            st.rerun()
    with col_next:
        if st.button("▶", disabled=st.session_state["_tp_cal_week"] >= total_weeks - 1):
            st.session_state["_tp_cal_week"] += 1
            st.rerun()

    week_idx = st.session_state["_tp_cal_week"]
    week = plan.weeks[week_idx]

    with col_info:
        st.markdown(
            f"**Tydzień {week.week_number}/{total_weeks}** — "
            f"{week.phase.emoji} {week.phase.value} — "
            f"{week.start_date.strftime('%d.%m')} – "
            f"{(week.start_date + timedelta(days=6)).strftime('%d.%m.%Y')}"
        )

    cols = st.columns(7)
    for dow in range(7):
        day = week.days[dow]
        with cols[dow]:
            if day.is_rest:
                st.markdown(
                    f"**{DAY_NAMES[dow]}**\n\n😴\n*Odpoczynek*",
                    unsafe_allow_html=False,
                )
            elif day.workout is not None:
                w = day.workout
                st.markdown(
                    f"**{DAY_NAMES[dow]}**\n\n"
                    f"{w.workout_type.emoji} **{w.workout_type.value}**\n\n"
                    f"TSS: **{w.tss_target:.0f}**\n"
                    f"⏱ {w.duration_min} min",
                    unsafe_allow_html=False,
                )
            st.divider()

    st.caption(
        f"Suma tygodniowa TSS: **{week.total_tss:.0f}** / {week.weekly_tss_target:.0f} cel  •  {week.training_days} dni treningowych"
    )


def _render_setup_section() -> Optional[TrainingPlan]:
    """Render plan setup form and return generated plan (or None)."""
    st.subheader("📋 Konfiguracja planu")

    c1, c2 = st.columns(2)
    with c1:
        plan_name = st.text_input("Nazwa planu", value="Plan treningowy", key="_tp_name")
        start_date = st.date_input("Data rozpoczęcia", value=date.today(), key="_tp_start")
    with c2:
        total_weeks = st.slider("Liczba tygodni", 1, 24, 13, key="_tp_weeks")
        tss_baseline = st.number_input(
            "Bazowe TSS / tydzień", min_value=50, max_value=1500, value=300, step=10, key="_tp_tss"
        )

    template_key = st.selectbox(
        "Szablon periodyzacji",
        options=list(PERIODIZATION_TEMPLATES.keys()),
        format_func=template_display_name,
        key="_tp_template",
    )

    st.divider()
    st.subheader("🗓️ Szablon tygodniowy")

    template_cols = st.columns(7)
    for dow in range(7):
        with template_cols[dow]:
            default_wt = _DEFAULT_WEEKLY_TEMPLATE.get(dow, WorkoutType.REST)
            st.selectbox(
                DAY_NAMES[dow],
                options=_WORKOUT_TYPE_OPTIONS,
                index=_WORKOUT_TYPE_OPTIONS.index(default_wt),
                format_func=lambda wt: _WORKOUT_TYPE_LABELS[wt],
                key=f"_tp_day_{dow}",
            )

    if st.button("🚀 Generuj plan", type="primary", use_container_width=True):
        phase_dist = PERIODIZATION_TEMPLATES[template_key]
        template = _get_template_from_state()
        plan = build_plan(
            name=plan_name,
            start_date=start_date,
            total_weeks=total_weeks,
            weekly_tss_baseline=tss_baseline,
            phase_distribution=phase_dist,
            template=template,
            athlete_id=st.session_state.get("selected_athlete_id", "default"),
        )
        st.session_state["_tp_plan"] = plan
        st.session_state["_tp_cal_week"] = 0
        st.rerun()

    return st.session_state.get("_tp_plan")


def _render_pmc_and_validation(plan: TrainingPlan) -> None:
    """Render PMC prediction chart and TSS ramp validation."""
    st.subheader("📈 Predykcja PMC")
    fig = _predict_pmc(plan)
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)

    weekly_tss_values = [w.total_tss for w in plan.weeks]
    warnings = validate_plan_tss_ramp(weekly_tss_values)
    if warnings:
        st.warning("⚠️ **Ostrzeżenia o tempie narastania TSS:**")
        for w in warnings:
            st.caption(f"• {w}")
    else:
        st.success("✅ Tempa narastania TSS są w bezpiecznych granicach (< 7%/tydzień).")


def _render_export_buttons(plan: TrainingPlan) -> None:
    """Render CSV export download buttons."""
    st.subheader("📤 Eksport")

    daily_csv = export_plan_csv(plan)
    summary_csv = export_plan_weekly_summary(plan)

    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "📊 CSV dzienny",
            data=daily_csv.encode("utf-8"),
            file_name=f"{plan.name.replace(' ', '_')}_daily.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        st.download_button(
            "📋 CSV podsumowanie",
            data=summary_csv.encode("utf-8"),
            file_name=f"{plan.name.replace(' ', '_')}_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_saved_plans() -> None:
    """Render saved plans section with load/delete buttons."""
    st.subheader("💾 Zapisane plany")

    store = _get_store()
    athlete_id = st.session_state.get("selected_athlete_id", "default")
    plans = store.list_plans(athlete_id=athlete_id)

    if not plans:
        st.info("Brak zapisanych planów.")
        return

    for plan in plans:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(
                f"**{plan.name}** — {plan.total_weeks} tyg — "
                f"{plan.start_date.strftime('%d.%m.%Y')} — "
                f"TSS bazowe: {plan.weekly_tss_baseline:.0f}"
            )
        with c2:
            if st.button("Wczytaj", key=f"_tp_load_{plan.id}"):
                st.session_state["_tp_plan"] = plan
                st.session_state["_tp_cal_week"] = 0
                st.rerun()
        with c3:
            if st.button("Usuń", key=f"_tp_del_{plan.id}"):
                store.delete_plan(plan.id)
                st.rerun()


def render_training_plan_tab() -> None:
    """Main entry point for the Training Plan Builder tab."""
    st.header("📅 Planer Treningowy")

    plan = _render_setup_section()

    if plan is None:
        _render_saved_plans()
        return

    st.divider()

    tab_cal, tab_pmc, tab_saved, tab_export = st.tabs(
        [
            "🗓️ Kalendarz",
            "📈 PMC",
            "💾 Zapisane",
            "📤 Eksport",
        ]
    )

    with tab_cal:
        _render_calendar(plan)

    with tab_pmc:
        _render_pmc_and_validation(plan)

    with tab_saved:
        _render_saved_plans()

    with tab_export:
        _render_export_buttons(plan)

    save_col, _ = st.columns([1, 3])
    with save_col:
        if st.button("💾 Zapisz plan", use_container_width=True):
            _get_store().save_plan(plan)
            st.success(f"Plan '{plan.name}' zapisany!")
