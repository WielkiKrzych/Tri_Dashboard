"""
Aerobic Efficiency tab — Power/HR ratio analysis and trends.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from modules.ui.shared import chart, metric, require_data
from modules.plots import CHART_HEIGHT_MAIN, CHART_HEIGHT_SUB
from modules.calculations.aerobic_efficiency import (
    calculate_session_efficiency,
    interpret_trend,
)


def render_aerobic_efficiency_tab(df_plot, cp_input):
    st.header("📊 Efektywność Tlenowa — Power/HR Ratio")

    if not require_data(df_plot, column="watts"):
        return

    hr_col = (
        "heart_rate"
        if "heart_rate" in df_plot.columns
        else ("hr" if "hr" in df_plot.columns else None)
    )
    if not hr_col:
        st.info("ℹ️ Brak danych HR w tym pliku.")
        return

    watts = df_plot["watts"].to_numpy(dtype=np.float64)
    hr = df_plot[hr_col].to_numpy(dtype=np.float64)
    time_arr = (
        df_plot["time"].to_numpy(dtype=np.float64)
        if "time" in df_plot.columns
        else np.arange(len(watts), dtype=np.float64)
    )
    time_min = time_arr / 60.0

    eff = calculate_session_efficiency(watts, hr, time_arr, float(cp_input))

    if eff.overall_ef <= 0:
        st.warning("Brak wystarczających danych do obliczenia EF.")
        return

    m1, m2, m3, m4 = st.columns(4)
    metric("EF Ogólny", f"{eff.overall_ef:.3f}", suffix=" W/bpm", column=m1)
    metric("EF Start", f"{eff.ef_start:.3f}", suffix=" W/bpm", column=m2)
    metric("EF Koniec", f"{eff.ef_end:.3f}", suffix=" W/bpm", column=m3)
    delta_color = "inverse"
    metric("ΔEF", f"{eff.ef_delta_pct:+.1f}", suffix=" %", column=m4, delta_color=delta_color)

    # EF trend over session
    with np.errstate(divide="ignore", invalid="ignore"):
        ef_array = np.where(hr > 50, watts / hr, np.nan)

    window = max(30, len(ef_array) // 60)
    ef_smooth = (
        pd.Series(ef_array).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    )

    fig_ef = go.Figure()
    fig_ef.add_trace(
        go.Scatter(
            x=time_min,
            y=ef_smooth,
            name="EF (rolling)",
            line=dict(color="#00cc96", width=2),
            hovertemplate="Czas: %{x:.1f} min<br>EF: %{y:.3f}<extra></extra>",
        )
    )
    fig_ef.add_hline(
        y=eff.overall_ef,
        line_dash="dash",
        line_color="#636efa",
        annotation_text=f"Średnia: {eff.overall_ef:.3f}",
    )
    fig_ef.update_layout(
        template="plotly_dark",
        title="Efficiency Factor w czasie",
        xaxis_title="Czas [min]",
        yaxis_title="EF [W/bpm]",
        hovermode="x unified",
        height=CHART_HEIGHT_MAIN,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    chart(fig_ef, key="aerobic_ef_trend")

    # Per-zone EF comparison
    zone_names = ["Z1 (<55%CP)", "Z2 (55-75%)", "Z3 (75-90%)", "Z4 (90-105%)", "Z5 (>105%CP)"]
    zone_values = [eff.zone1_ef, eff.zone2_ef, eff.zone3_ef, eff.zone4_ef, eff.zone5_ef]
    zone_colors = ["#636efa", "#00cc96", "#ab63fa", "#ff7f0e", "#ef553b"]

    fig_zones = go.Figure()
    fig_zones.add_trace(
        go.Bar(
            x=zone_names,
            y=zone_values,
            marker_color=zone_colors,
            hovertemplate="%{x}: %{y:.3f} W/bpm<extra></extra>",
        )
    )
    fig_zones.update_layout(
        template="plotly_dark",
        title="EF wg Strefy Mocy",
        xaxis_title="Strefa",
        yaxis_title="EF [W/bpm]",
        height=CHART_HEIGHT_SUB,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    chart(fig_zones, key="aerobic_zone_ef")

    # Cardiac drift indicator
    st.subheader("💊 Wskaźnik Dryfu Sercowego")
    drift_pct = eff.ef_delta_pct
    if drift_pct < -10:
        st.error(
            f"🔴 WYSOKI DRYF: EF spadł o {abs(drift_pct):.1f}% — możliwe przegrzanie lub zmęczenie centralne"
        )
    elif drift_pct < -5:
        st.warning(f"🟡 Umiarkowany dryf: EF spadł o {abs(drift_pct):.1f}%")
    else:
        st.success(f"🟢 Stabilny EF: zmiana {drift_pct:+.1f}%")

    st.caption(
        interpret_trend(
            "stable" if abs(drift_pct) < 5 else ("declining" if drift_pct < -5 else "improving")
        )
    )

    with st.expander("📖 Teoria — Efficiency Factor"):
        st.markdown("""
        **Efficiency Factor (EF)** = Moc / Tętno [W/bpm]

        EF mierzy ile watów generujesz na każde uderzenie serca.
        Wyższy EF = lepsza wydajność tlenowa.

        **Interpretacja:**
        - EF rośnie w sezonie: adaptacja tlenowa, poprawa stroke volume
        - EF maleje: zmęczenie, przeładowanie, dehydration
        - Spadek EF >10% w trakcie sesji = istotny dryf sercowy

        **Strefy EF:**
        - Z1-Z2: EF powinien być stabilny i relatywnie wysoki
        - Z3-Z4: EF nieznacznie niższy ze względu na wyższe HR
        - Z5: EF może być zmienny (anaerobowy udział)

        **Bibliografia:**
        - Coggan AR (2003). Training zones and EF.
        - Sanders D et al. (2022). Cardiac drift quantification. IJSPP.
        """)
