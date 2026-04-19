"""
MPA (Maximum Power Available) tab — W'bal envelope visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from modules.ui.shared import chart, metric, require_data
from modules.plots import CHART_HEIGHT_MAIN, CHART_HEIGHT_SUB


def render_mpa_tab(df_plot, cp_input, w_prime_input):
    st.header("⚡ MPA — Maximum Power Available")

    if not require_data(df_plot, column="watts"):
        return

    watts = df_plot["watts"].to_numpy(dtype=np.float64)
    time_arr = (
        df_plot["time"].to_numpy(dtype=np.float64)
        if "time" in df_plot.columns
        else np.arange(len(watts), dtype=np.float64)
    )

    c1, c2 = st.columns(2)
    sport = c1.selectbox("Sport", ["Kolarstwo", "Bieg", "Pływanie"], index=0)
    model = c2.selectbox(
        "Model rekonstytucji W'", ["Bi-exponential (Caen 2021)", "Skiba (2015)"], index=0
    )

    sport_map = {"Kolarstwo": 0, "Bieg": 1, "Pływanie": 2}
    model_map = {"Bi-exponential (Caen 2021)": "biexp", "Skiba (2015)": "skiba"}

    from modules.calculations.mpa import calculate_mpa, calculate_time_to_exhaustion

    profile = calculate_mpa(
        watts,
        time_arr,
        cp=float(cp_input),
        w_prime_cap=float(w_prime_input),
        model=model_map[model],
        sport=sport_map[sport],
    )

    # Key metrics
    min_mpa = float(np.min(profile.mpa_array))
    time_above_95 = float(np.sum(watts > 0.95 * profile.mpa_array))
    total_active = float(np.sum(watts > 0))
    pct_above_95 = (time_above_95 / max(total_active, 1)) * 100

    m1, m2, m3, m4 = st.columns(4)
    metric("Min MPA", f"{min_mpa:.0f}", suffix=" W", column=m1)
    metric("Czas >95% MPA", f"{pct_above_95:.1f}", suffix=" %", column=m2)
    metric(
        "TTE @ Pmax",
        f"{profile.time_to_exhaustion_at_peak:.0f}"
        if profile.time_to_exhaustion_at_peak and profile.time_to_exhaustion_at_peak < 1e6
        else "∞",
        suffix=" s",
        column=m3,
    )
    metric("CP", f"{profile.cp:.0f}", suffix=" W", column=m4)

    time_min = profile.time_array / 60.0

    # Power + MPA overlay
    fig_power = go.Figure()
    fig_power.add_trace(
        go.Scatter(
            x=time_min,
            y=watts,
            name="Moc",
            line=dict(color="#636efa", width=1),
            opacity=0.6,
            hovertemplate="Czas: %{x:.1f} min<br>Moc: %{y:.0f} W<extra></extra>",
        )
    )
    fig_power.add_trace(
        go.Scatter(
            x=time_min,
            y=profile.mpa_array,
            name="MPA",
            line=dict(color="#ef553b", width=2, dash="dash"),
            hovertemplate="Czas: %{x:.1f} min<br>MPA: %{y:.0f} W<extra></extra>",
        )
    )
    fig_power.add_hline(
        y=profile.cp,
        line_dash="dot",
        line_color="#00cc96",
        annotation_text=f"CP = {profile.cp:.0f} W",
        annotation_position="top left",
    )
    fig_power.update_layout(
        template="plotly_dark",
        title="Moc vs MPA",
        xaxis_title="Czas [min]",
        yaxis_title="Moc [W]",
        hovermode="x unified",
        height=CHART_HEIGHT_MAIN,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    chart(fig_power, key="mpa_power_chart")

    # W' balance chart
    wbal_pct = (profile.wbal_array / max(profile.w_prime, 1)) * 100
    fig_wbal = go.Figure()
    fig_wbal.add_trace(
        go.Scatter(
            x=time_min,
            y=wbal_pct,
            name="W' Balance",
            line=dict(color="#ab63fa", width=2),
            fill="tozeroy",
            hovertemplate="Czas: %{x:.1f} min<br>W'bal: %{y:.1f}%<extra></extra>",
        )
    )
    fig_wbal.add_hline(
        y=25,
        line_dash="dash",
        line_color="red",
        annotation_text="Strefa krytyczna 25%",
        annotation_position="bottom right",
    )
    fig_wbal.update_layout(
        template="plotly_dark",
        title="W' Balance [%]",
        xaxis_title="Czas [min]",
        yaxis_title="W' Balance [% W']",
        hovermode="x unified",
        height=CHART_HEIGHT_SUB,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    chart(fig_wbal, key="mpa_wbal_chart")

    with st.expander("📖 Teoria — MPA (Maximum Power Available)"):
        st.markdown("""
        **MPA** (Maximum Power Available) to maksymalna moc, którą atlet może wygenerować
        w danej chwili, bazując na aktualnym stanie W' balance.

        **Model:** MPA = W'bal / τ_recovery + CP

        - Gdy moc < CP: W' się odnawia, MPA rośnie w kierunku Pmax
        - Gdy moc > CP: W' się wyczerpuje, MPA spada
        - TTE = W'bal / (P - CP) — czas do wyczerpania przy obecnej mocy

        **Znaczenie treningowe:**
        - Czas >95% MPA = intensywna praca nad progami beztlenowymi
        - Spadek poniżej 25% W' = strefa krytyczna, ryzyko „odcięcia"
        - MPA pokazuje moment, gdy atlet nie jest w stanie utrzymać danej mocy
        """)
