"""
Multi-sport thresholds tab — Cycling / Running / Swimming zone comparison.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from modules.ui.shared import chart, metric
from modules.plots import CHART_HEIGHT_SUB
from modules.calculations.multisport_zones import (
    calculate_cycling_zones,
    calculate_running_zones,
    calculate_swim_zones,
    estimate_critical_pace,
    pace_to_str,
    swim_pace_to_str,
)


def render_multisport_zones_tab(cp_input, w_prime_input, rider_weight):
    st.header("🏃‍♂️ Wielosportowe Progi")

    c1, c2, c3 = st.columns(3)

    # Cycling
    with c1:
        st.subheader("🚴 Kolarstwo")
        cycling_zones = calculate_cycling_zones(float(cp_input))
        _render_zone_table(cycling_zones, unit="W")
        _render_zone_bar_chart(cycling_zones, unit="W", color="#636efa")

    # Running
    with c2:
        st.subheader("🏃 Bieg")
        run_pace = st.number_input(
            "Tempo progowe [min/km]",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            key="run_threshold_pace",
        )
        run_pace_sec = run_pace * 60.0
        running_zones = calculate_running_zones(run_pace_sec)
        _render_zone_table(running_zones, unit="pace", is_run=True)
        _render_zone_bar_chart(running_zones, unit="pace", color="#00cc96", is_run=True)

    # Swimming
    with c3:
        st.subheader("🏊 Pływanie")
        swim_pace = st.number_input(
            "CSS [min/100m]",
            min_value=0.8,
            max_value=3.0,
            value=1.5,
            step=0.05,
            key="swim_css",
        )
        swim_pace_sec = swim_pace * 60.0
        swim_zones = calculate_swim_zones(swim_pace_sec)
        _render_zone_table(swim_zones, unit="pace", is_swim=True)
        _render_zone_bar_chart(swim_zones, unit="pace", color="#ab63fa", is_swim=True)

    # Critical Pace estimator
    st.markdown("---")
    st.subheader("📏 Estymator Critical Pace")
    st.caption("Wprowadź 2+ pary dystans-czas, aby wyestymować CP i D'.")

    num_pairs = st.number_input("Liczba par", min_value=2, max_value=6, value=2, key="cp_pairs")
    pairs_data = []
    c_left, c_right = st.columns(2)
    for i in range(int(num_pairs)):
        with c_left if i % 2 == 0 else c_right:
            d = st.number_input(
                f"Dystans {i + 1} [m]", min_value=100, value=1000 + i * 1000, key=f"cp_dist_{i}"
            )
            t = st.number_input(
                f"Czas {i + 1} [s]", min_value=10, value=180 + i * 120, key=f"cp_time_{i}"
            )
            pairs_data.append((float(d), float(t)))

    if st.button("Oblicz Critical Pace", key="calc_cp_btn"):
        distances = [p[0] for p in pairs_data]
        times = [p[1] for p in pairs_data]
        cp_pace, d_prime = estimate_critical_pace(distances, times)
        m1, m2 = st.columns(2)
        metric("Critical Pace", pace_to_str(cp_pace), column=m1)
        metric("D'", f"{d_prime:.0f}", suffix=" m", column=m2)


def _render_zone_table(zones, unit="W", is_run=False, is_swim=False):
    rows = []
    for name, lower, upper in zones.zones:
        if is_run:
            rows.append({"Strefa": name, "Od": pace_to_str(lower), "Do": pace_to_str(upper)})
        elif is_swim:
            rows.append(
                {"Strefa": name, "Od": swim_pace_to_str(lower), "Do": swim_pace_to_str(upper)}
            )
        else:
            rows.append({"Strefa": name, "Od": f"{lower:.0f} {unit}", "Do": f"{upper:.0f} {unit}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_zone_bar_chart(zones, unit="W", color="#636efa", is_run=False, is_swim=False):
    fig = go.Figure()
    y_pos = list(range(len(zones.zones)))
    names = [z[0] for z in zones.zones]
    widths = []
    for _, lower, upper in zones.zones:
        if is_run or is_swim:
            widths.append(abs(upper - lower) if upper > 0 and lower > 0 else 0)
        else:
            widths.append(upper - lower)

    fig.add_trace(
        go.Bar(
            y=y_pos,
            x=widths,
            orientation="h",
            text=names,
            marker_color=color,
            opacity=0.8,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=unit,
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    chart(fig, key=f"zone_bar_{zones.sport}")
