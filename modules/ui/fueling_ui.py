"""
Fueling Engine tab — event nutrition and hydration planning.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from modules.ui.shared import chart, metric
from modules.plots import CHART_HEIGHT_MAIN, CHART_HEIGHT_SUB
from modules.calculations.fueling import (
    calculate_fueling_plan,
    estimate_carb_burn_rate,
)


def render_fueling_tab(df_plot_resampled, cp_input):
    st.header("🍕 Plan Odżywiania — Fueling Engine")

    c1, c2, c3 = st.columns(3)
    duration_h = c1.number_input(
        "Czas trwania [h]", min_value=0.5, max_value=24.0, value=3.0, step=0.5
    )
    target_pct = (
        c2.number_input("Intensywność [%FTP]", min_value=40, max_value=130, value=70, step=5)
        / 100.0
    )
    weight = c3.number_input("Waga [kg]", min_value=40.0, max_value=120.0, value=75.0, step=1.0)

    c4, c5 = st.columns(2)
    heat = c4.checkbox("Warunki upalne (>30°C)", value=False)
    altitude = c5.checkbox("Wysokość (>1500m)", value=False)

    target_power = float(cp_input) * target_pct

    plan = calculate_fueling_plan(
        duration_hours=duration_h,
        target_power_w=target_power,
        weight_kg=weight,
        intensity_pct_ftp=target_pct,
        heat=heat,
        altitude=altitude,
    )

    m1, m2, m3, m4 = st.columns(4)
    metric(
        "Spalanie CHO",
        f"{estimate_carb_burn_rate(target_power, weight, target_pct):.0f}",
        suffix=" g/h",
        column=m1,
    )
    metric("Polecane spożycie CHO", f"{plan.carb_intake_g_h:.0f}", suffix=" g/h", column=m2)
    metric("Płyny", f"{plan.fluid_ml_h:.0f}", suffix=" ml/h", column=m3)
    metric("Sód", f"{plan.sodium_mg_h:.0f}", suffix=" mg/h", column=m4)

    # Glycogen balance chart
    times = [s.time_min for s in plan.plan_steps]
    glycogen = [s.glycogen_balance for s in plan.plan_steps]

    fig_glyc = go.Figure()
    fig_glyc.add_trace(
        go.Scatter(
            x=times,
            y=glycogen,
            name="Glikogen",
            line=dict(color="#00cc96", width=2),
            fill="tozeroy",
            hovertemplate="Czas: %{x:.0f} min<br>Glikogen: %{y:.0f} g<extra></extra>",
        )
    )
    fig_glyc.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Bonk!",
        annotation_position="bottom right",
    )
    fig_glyc.add_hline(
        y=100,
        line_dash="dot",
        line_color="yellow",
        opacity=0.5,
        annotation_text="Strefa ostrzegawcza",
        annotation_position="top left",
    )
    fig_glyc.update_layout(
        template="plotly_dark",
        title="Bilans Glikogenu",
        xaxis_title="Czas [min]",
        yaxis_title="Glikogen [g]",
        hovermode="x unified",
        height=CHART_HEIGHT_MAIN,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    chart(fig_glyc, key="fueling_glycogen_chart")

    # Plan table
    st.subheader("📋 Plan Godzinowy")
    rows = []
    for s in plan.plan_steps:
        rows.append(
            {
                "Czas [min]": f"{s.time_min:.0f}",
                "Akcja": s.action,
                "CHO [g]": f"{s.carb_g:.0f}",
                "Płyny [ml]": f"{s.fluid_ml:.0f}",
                "Sód [mg]": f"{s.sodium_mg:.0f}",
                "Glikogen [g]": f"{s.glycogen_balance:.0f}",
                "Uwagi": s.notes,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Summary totals
    total_carb = sum(s.carb_g for s in plan.plan_steps)
    total_fluid = sum(s.fluid_ml for s in plan.plan_steps)
    total_sodium = sum(s.sodium_mg for s in plan.plan_steps)
    st.subheader("📦 Podsumowanie")
    m5, m6, m7 = st.columns(3)
    metric("Węglowodany łącznie", f"{total_carb:.0f}", suffix=" g", column=m5)
    metric("Płyny łącznie", f"{total_fluid:.0f}", suffix=" ml", column=m6)
    metric("Sód łącznie", f"{total_sodium:.0f}", suffix=" mg", column=m7)

    # Carb burn by zone chart
    st.subheader("🔥 Spalanie CHO wg Strefy")
    zones_pct = [0.50, 0.65, 0.82, 0.95, 1.10]
    zone_names = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    burn_rates = [estimate_carb_burn_rate(float(cp_input) * z, weight, z) for z in zones_pct]

    fig_zone = go.Figure()
    fig_zone.add_trace(
        go.Bar(
            x=zone_names,
            y=burn_rates,
            marker_color=["#636efa", "#00cc96", "#ab63fa", "#ff7f0e", "#ef553b"],
            hovertemplate="%{x}: %{y:.0f} g/h<extra></extra>",
        )
    )
    fig_zone.add_hline(
        y=90,
        line_dash="dash",
        line_color="yellow",
        opacity=0.5,
        annotation_text="Limit jelitowy ~90g/h",
    )
    fig_zone.update_layout(
        template="plotly_dark",
        title="Spalanie CHO wg Strefy Intensywności",
        xaxis_title="Strefa",
        yaxis_title="Spalanie CHO [g/h]",
        height=CHART_HEIGHT_SUB,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    chart(fig_zone, key="fueling_zone_chart")

    with st.expander("📖 Teoria — Fueling Engine"):
        st.markdown("""
        **Plan Odżywiania** oparty na modelu spalania węglowodanów wg intensywności
        (%FTP) i fizykalnej konwersji mocy → energia.

        **Kluczowe zasady:**
        - Spożycie CHO: 30-60 g/h (zwykłe), do 90 g/h (maltodekstryna+fruktoza 2:1)
        - Płyny: 500-800 ml/h (chłodne), do 1000 ml/h (upał)
        - Sód: 500-1200 mg/L potu (500-1000 mg/h typowo)
        - Zapas glikogenu startowy: ~400-500g (wytrenowani atleci)

        **Bibliografia:**
        - Jeukendrup AE (2014). Carbohydrate intake during exercise. Sports Med.
        - Sawka MN et al. (2007). ACSM position stand: Exercise and fluid replacement.
        """)
