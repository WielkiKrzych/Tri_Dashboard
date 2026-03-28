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
from modules.plots import CHART_CONFIG, CHART_HEIGHT_MAIN, CHART_HEIGHT_SUB
from .summary_calculations import _estimate_cp_wprime


def _get_smooth(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Return pre-smoothed column if available, otherwise raw (no extra rolling)."""
    smooth_col = f"{col}_smooth"
    if smooth_col in df.columns:
        return df[smooth_col]
    if col in df.columns:
        return df[col]
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _build_training_timeline_chart(
    df_plot: pd.DataFrame,
    cp_input: int = 0,
    vt1_watts: int = 0,
    vt2_watts: int = 0,
) -> Optional[go.Figure]:
    """Build training timeline as two stacked subplots: Power+HR (top) / SmO2+VE (bottom)."""
    time_x = (
        df_plot["time_min"]
        if "time_min" in df_plot.columns
        else df_plot["time"] / 60
        if "time" in df_plot.columns
        else None
    )
    if time_x is None:
        return None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # ── Row 1: Power (fill) ──────────────────────────────────────────────────
    watts = _get_smooth(df_plot, "watts")
    if watts is not None:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=watts,
                name="Moc",
                fill="tozeroy",
                line=dict(color=Config.COLOR_POWER, width=1),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>",
            ),
            row=1, col=1, secondary_y=False,
        )

    # ── Training zone bands in background (row 1) ───────────────────────────
    if cp_input and cp_input > 0:
        zone_bands = [
            (0,            cp_input * 0.55, "rgba(140,140,140,0.06)", "Z1"),
            (cp_input * 0.55, cp_input * 0.75, "rgba(50,205,50,0.06)",  "Z2"),
            (cp_input * 0.75, cp_input * 0.90, "rgba(255,215,0,0.06)",  "Z3"),
            (cp_input * 0.90, cp_input * 1.05, "rgba(255,140,0,0.07)",  "Z4"),
            (cp_input * 1.05, cp_input * 1.20, "rgba(255,69,0,0.07)",   "Z5"),
            (cp_input * 1.20, cp_input * 2.00, "rgba(139,0,0,0.07)",    "Z6"),
        ]
        for y0, y1, color, label in zone_bands:
            fig.add_hrect(
                y0=y0, y1=y1,
                fillcolor=color,
                line_width=0,
                layer="below",
                row=1, col=1,
                annotation_text=label,
                annotation_font_size=9,
                annotation_font_color="rgba(200,200,200,0.5)",
                annotation_position="right",
            )

    # ── CP / VT1 / VT2 threshold lines (row 1) ──────────────────────────────
    if cp_input and cp_input > 0:
        fig.add_hline(
            y=cp_input,
            line_dash="dash", line_color="#ef553b", opacity=0.6,
            annotation_text=f"CP {cp_input}W",
            annotation_font_size=9,
            annotation_position="right",
            row=1, col=1,
        )
    if vt2_watts and vt2_watts > 0:
        fig.add_hline(
            y=vt2_watts,
            line_dash="dot", line_color="#ffa15a", opacity=0.6,
            annotation_text=f"VT2 {vt2_watts}W",
            annotation_font_size=9,
            annotation_position="right",
            row=1, col=1,
        )
    if vt1_watts and vt1_watts > 0:
        fig.add_hline(
            y=vt1_watts,
            line_dash="dot", line_color="#00cc96", opacity=0.6,
            annotation_text=f"VT1 {vt1_watts}W",
            annotation_font_size=9,
            annotation_position="right",
            row=1, col=1,
        )

    # ── Row 1: HR (secondary Y) ──────────────────────────────────────────────
    hr = _get_smooth(df_plot, "heartrate")
    if hr is not None:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=hr,
                name="HR",
                line=dict(color=Config.COLOR_HR, width=2),
                hovertemplate="HR: %{y:.0f} bpm<extra></extra>",
            ),
            row=1, col=1, secondary_y=True,
        )

    # ── Row 2: SmO2 (primary Y) ──────────────────────────────────────────────
    smo2 = _get_smooth(df_plot, "smo2")
    if smo2 is not None:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=smo2,
                name="SmO2",
                line=dict(color=Config.COLOR_SMO2, width=2, dash="dot"),
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>",
            ),
            row=2, col=1, secondary_y=False,
        )

    # ── Row 2: VE (secondary Y) ──────────────────────────────────────────────
    ve = _get_smooth(df_plot, "tymeventilation")
    if ve is not None:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=ve,
                name="VE",
                line=dict(color=Config.COLOR_VE, width=2, dash="dash"),
                hovertemplate="VE: %{y:.1f} L/min<extra></extra>",
            ),
            row=2, col=1, secondary_y=True,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        title="Przebieg Treningu",
        hovermode="x unified",
        height=CHART_HEIGHT_MAIN,
        legend=dict(orientation="h", y=1.03, x=0, yanchor="bottom"),
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Axes labels
    fig.update_yaxes(title_text="Moc [W]",  row=1, col=1, secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="HR [bpm]", row=1, col=1, secondary_y=True,  showgrid=False)
    fig.update_yaxes(title_text="SmO2 [%]", row=2, col=1, secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="VE [L/min]", row=2, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(
        title_text="Czas [min]",
        row=2, col=1,
        rangeslider=dict(visible=True, thickness=0.04, bgcolor="#1c2128"),
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)

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
        height=CHART_HEIGHT_SUB,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_model, use_container_width=True, config=CHART_CONFIG)


def _render_smo2_thb_chart(df_plot: pd.DataFrame):
    """Renderowanie wykresu SmO2 vs THb w czasie."""
    if "smo2" not in df_plot.columns:
        st.info("Brak danych SmO2 w tym pliku.")
        return

    fig_smo2_thb = make_subplots(specs=[[{"secondary_y": True}]])

    time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

    smo2_col = _get_smooth(df_plot, "smo2")
    if smo2_col is not None:
        fig_smo2_thb.add_trace(
            go.Scatter(
                x=time_x,
                y=smo2_col,
                name="SmO2 (%)",
                line=dict(color=Config.COLOR_SMO2, width=2),
                hovertemplate="SmO2: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=False,
        )

    if "thb" in df_plot.columns:
        thb_col = _get_smooth(df_plot, "thb")
        fig_smo2_thb.add_trace(
            go.Scatter(
                x=time_x,
                y=thb_col,
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
        height=CHART_HEIGHT_SUB,
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_smo2_thb.update_yaxes(title_text="SmO2 (%)", secondary_y=False)
    fig_smo2_thb.update_yaxes(title_text="THb (g/dL)", secondary_y=True)
    st.plotly_chart(fig_smo2_thb, use_container_width=True, config=CHART_CONFIG)

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
            <div style="padding:15px; border-radius:8px; border:2px solid {Config.COLOR_SMO2}; background-color: #222;">
                <h3 style="margin:0; color: {Config.COLOR_SMO2};">🩸 SmO2</h3>
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
