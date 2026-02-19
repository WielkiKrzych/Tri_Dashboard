"""
Summary Charts — Plotly chart builders for the Summary tab.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional

from modules.config import Config
from .summary_calculations import _estimate_cp_wprime


@st.cache_data(ttl=3600, show_spinner=False)
def _build_training_timeline_chart(df_plot: pd.DataFrame) -> Optional[go.Figure]:
    """Build training timeline chart with power, HR, SmO2, VE (cached)."""
    fig = go.Figure()
    time_x = (
        df_plot["time_min"]
        if "time_min" in df_plot.columns
        else df_plot["time"] / 60
        if "time" in df_plot.columns
        else None
    )

    if time_x is None:
        return None

    if "watts_smooth" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["watts_smooth"],
                name="Moc",
                fill="tozeroy",
                line=dict(color=Config.COLOR_POWER, width=1),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>",
            )
        )
    elif "watts" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["watts"].rolling(5, center=True).mean(),
                name="Moc",
                fill="tozeroy",
                line=dict(color=Config.COLOR_POWER, width=1),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>",
            )
        )

    if "heartrate_smooth" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["heartrate_smooth"],
                name="HR",
                line=dict(color=Config.COLOR_HR, width=2),
                yaxis="y2",
                hovertemplate="HR: %{y:.0f} bpm<extra></extra>",
            )
        )
    elif "heartrate" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["heartrate"],
                name="HR",
                line=dict(color=Config.COLOR_HR, width=2),
                yaxis="y2",
                hovertemplate="HR: %{y:.0f} bpm<extra></extra>",
            )
        )

    if "smo2_smooth" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["smo2_smooth"],
                name="SmO2",
                line=dict(color=Config.COLOR_SMO2, width=2, dash="dot"),
                yaxis="y3",
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>",
            )
        )
    elif "smo2" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["smo2"].rolling(5, center=True).mean(),
                name="SmO2",
                line=dict(color=Config.COLOR_SMO2, width=2, dash="dot"),
                yaxis="y3",
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>",
            )
        )

    if "tymeventilation_smooth" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["tymeventilation_smooth"],
                name="VE",
                line=dict(color=Config.COLOR_VE, width=2, dash="dash"),
                yaxis="y4",
                hovertemplate="VE: %{y:.1f} L/min<extra></extra>",
            )
        )
    elif "tymeventilation" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["tymeventilation"].rolling(10, center=True).mean(),
                name="VE",
                line=dict(color=Config.COLOR_VE, width=2, dash="dash"),
                yaxis="y4",
                hovertemplate="VE: %{y:.1f} L/min<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Przebieg Treningu (Moc, HR, SmO2, VE)",
        hovermode="x unified",
        xaxis=dict(title="Czas [min]"),
        yaxis=dict(title="Moc [W]", side="left"),
        yaxis2=dict(title="HR [bpm]", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="SmO2 [%]", overlaying="y", side="right", position=0.95, showgrid=False),
        yaxis4=dict(title="VE [L/min]", overlaying="y", side="right", position=0.98, showgrid=False),
        height=500,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _render_cp_model_chart(df_plot: pd.DataFrame, cp_input: int, w_prime_input: int):
    """Renderowanie wykresu modelu CP."""
    if "watts" not in df_plot.columns or len(df_plot) < 1200:
        st.info("Za mało danych (wymagane min. 20 minut) do wyświetlenia modelu CP.")
        return

    est_cp, est_w_prime = _estimate_cp_wprime(df_plot)

    if est_cp <= 0:
        st.info("Nie można wyestymować CP z danych.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Est. CP", f"{est_cp:.0f} W", delta=f"{est_cp - cp_input:.0f} W vs ustawienia")
    c2.metric(
        "Est. W'",
        f"{est_w_prime:.0f} J",
        delta=f"{est_w_prime - w_prime_input:.0f} J vs ustawienia",
    )

    durations = [180, 300, 600, 900, 1200]
    valid_durations = [d for d in durations if d < len(df_plot)]
    work_values = []
    for d in valid_durations:
        p = df_plot["watts"].rolling(window=d).mean().max()
        if not pd.isna(p):
            work_values.append(p * d)

    if len(work_values) == len(valid_durations):
        _, _, r_value, _, _ = stats.linregress(valid_durations, work_values)
        c3.metric("R² (dopasowanie)", f"{r_value**2:.4f}")

    x_theory = np.arange(60, 1800, 60)
    y_theory = [est_cp + (est_w_prime / t) for t in x_theory]

    y_actual = []
    x_actual = []
    for t in x_theory:
        if t < len(df_plot):
            val = df_plot["watts"].rolling(t).mean().max()
            y_actual.append(val)
            x_actual.append(t)

    fig_model = go.Figure()

    fig_model.add_trace(
        go.Scatter(
            x=np.array(x_actual) / 60,
            y=y_actual,
            mode="markers",
            name="Twoje MMP",
            marker=dict(color="#00cc96", size=8),
        )
    )

    fig_model.add_trace(
        go.Scatter(
            x=x_theory / 60,
            y=y_theory,
            mode="lines",
            name=f"Model CP ({est_cp:.0f}W)",
            line=dict(color="#ef553b", dash="dash"),
        )
    )

    fig_model.update_layout(
        template="plotly_dark",
        title="Power Duration Curve vs Model CP",
        xaxis=dict(title="Czas [min]", tickformat=".0f"),
        yaxis=dict(title="Moc [W]", tickformat=".0f"),
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_model, use_container_width=True)


def _render_smo2_thb_chart(df_plot: pd.DataFrame):
    """Renderowanie wykresu SmO2 vs THb w czasie."""
    if "smo2" not in df_plot.columns:
        st.info("Brak danych SmO2 w tym pliku.")
        return

    fig_smo2_thb = make_subplots(specs=[[{"secondary_y": True}]])

    time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

    smo2_smooth = (
        df_plot["smo2"].rolling(5, center=True).mean() if "smo2" in df_plot.columns else None
    )
    if smo2_smooth is not None:
        fig_smo2_thb.add_trace(
            go.Scatter(
                x=time_x,
                y=smo2_smooth,
                name="SmO2 (%)",
                line=dict(color="#2ca02c", width=2),
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    if "thb" in df_plot.columns:
        thb_smooth = df_plot["thb"].rolling(5, center=True).mean()
        fig_smo2_thb.add_trace(
            go.Scatter(
                x=time_x,
                y=thb_smooth,
                name="THb (g/dL)",
                line=dict(color="#9467bd", width=2),
                hovertemplate="THb: %{y:.2f} g/dL<extra></extra>",
            ),
            secondary_y=True,
        )
    else:
        st.caption("ℹ️ Brak danych THb w pliku.")

    fig_smo2_thb.update_layout(
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig_smo2_thb.update_yaxes(title_text="SmO2 (%)", secondary_y=False)
    fig_smo2_thb.update_yaxes(title_text="THb (g/dL)", secondary_y=True)
    st.plotly_chart(fig_smo2_thb, use_container_width=True)

    if "smo2" in df_plot.columns:
        smo2_min = df_plot["smo2"].min()
        smo2_max = df_plot["smo2"].max()
        smo2_mean = df_plot["smo2"].mean()

        thb_min = df_plot["thb"].min() if "thb" in df_plot.columns else None
        thb_max = df_plot["thb"].max() if "thb" in df_plot.columns else None
        thb_mean = df_plot["thb"].mean() if "thb" in df_plot.columns else None

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #2ca02c; background-color: #222;">
                <h3 style="margin:0; color: #2ca02c;">🩸 SmO2</h3>
                <p style="margin:5px 0; color:#aaa;"><b>Min:</b> {smo2_min:.1f}%</p>
                <p style="margin:5px 0; color:#aaa;"><b>Max:</b> {smo2_max:.1f}%</p>
                <p style="margin:5px 0; color:#aaa;"><b>Śr:</b> {smo2_mean:.1f}%</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            if thb_min is not None:
                st.markdown(
                    f"""
                <div style="padding:15px; border-radius:8px; border:2px solid #9467bd; background-color: #222;">
                    <h3 style="margin:0; color: #9467bd;">💉 THb</h3>
                    <p style="margin:5px 0; color:#aaa;"><b>Min:</b> {thb_min:.2f} g/dL</p>
                    <p style="margin:5px 0; color:#aaa;"><b>Max:</b> {thb_max:.2f} g/dL</p>
                    <p style="margin:5px 0; color:#aaa;"><b>Śr:</b> {thb_mean:.2f} g/dL</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
