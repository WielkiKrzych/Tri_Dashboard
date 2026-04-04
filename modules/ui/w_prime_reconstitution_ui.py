"""
W' Reconstitution Map tab — visualize W' depletion and recovery cycles.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Optional

from modules.config import Config
from modules.plots import CHART_CONFIG, CHART_HEIGHT_LARGE
from modules.ui.shared import chart, metric, require_data
from modules.calculations.w_prime_reconstitution import (
    compute_w_prime_reconstitution_map,
    build_reconstitution_table,
    get_reconstitution_interpretation,
)


def render_w_prime_reconstitution_tab(
    df: Optional[pd.DataFrame] = None,
    df_resampled: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    rider_weight: float = 75.0,
    cp_input: int = 280,
    w_prime_input: int = 20000,
    **kwargs,
) -> None:
    """Render the W' Reconstitution Map tab."""

    if not require_data(df):
        return

    st.header("🔋 Mapa Rekonstytucji W'")
    st.markdown(
        "Wizualizacja wyczerpywania i odbudowywania zasobów beztlenowych (W') "
        "podczas treningu. Pokazuje kiedy W' było krytycznie niskie, jak szybko "
        "następowała regeneracja i jakie były warunki każdego cyklu."
    )

    # Model selection
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model = st.selectbox(
            "Model rekonstytucji",
            options=["biexp", "skiba"],
            format_func=lambda x: "Bi-eksponencjalny (Caen 2021)" if x == "biexp" else "Mono-eksponencjalny (Skiba 2015)",
            index=0,
        )
    with col_m2:
        sport = st.selectbox(
            "Dyscyplina",
            options=[0, 1, 2],
            format_func=lambda x: {0: "🚴 Kolarstwo", 1: "🏃 Bieganie", 2: "🏊 Pływanie"}.get(x, ""),
            index=0,
        )

    # Compute
    result_df, summary = compute_w_prime_reconstitution_map(
        df=df,
        cp=cp_input,
        w_prime_cap=w_prime_input,
        model=model,
        sport=sport,
    )

    if result_df is None or "w_prime_balance" not in result_df.columns:
        st.error("Nie udało się obliczyć W' Balance. Sprawdź dane mocy.")
        return

    # --- Summary metrics ---
    st.subheader("📊 Podsumowanie Sesji")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Liczba cykli wyczerpania",
            summary.total_depletions,
            help="Ile razy W' spadło poniżej 50% pojemności"
        )
    with col2:
        st.metric(
            "Najgłębsze wyczerpanie",
            f"{summary.deepest_depletion_pct:.1f}%",
            f"{summary.deepest_depletion_j:.0f} J",
            help="Minimalny poziom W' podczas treningu"
        )
    with col3:
        st.metric(
            "Śr. tempo regeneracji",
            f"{summary.avg_recovery_rate_j_s:.1f} J/s",
            help="Średnia szybkość odbudowy W' poniżej CP"
        )
    with col4:
        st.metric(
            "Czas <20% W'",
            f"{summary.time_below_20pct_pct:.1f}%",
            f"{summary.time_below_20pct_s:.0f} s",
            help="Procent czasu w strefie krytycznej"
        )

    # Interpretation
    interp = get_reconstitution_interpretation(summary)
    st.info(f"**Interpretacja:**\n\n{interp}")

    # --- W' Balance timeline ---
    st.subheader("📈 Przebieg W' Balance w Czasie")

    time_col = "time" if "time" in result_df.columns else result_df.index
    x_vals = result_df[time_col] if isinstance(time_col, str) else result_df.index

    fig = go.Figure()

    # W' balance area
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=result_df["w_prime_balance"],
        mode="lines",
        name="W' Balance",
        fill="tozeroy",
        line=dict(color="#FF6B35", width=1.5),
        hovertemplate="Czas: %{x:.0f}s<br>W': %{y:.0f} J (%{customdata:.1f}%)<extra></extra>",
        customdata=result_df["w_prime_pct"],
    ))

    # CP line (W' = 100%)
    fig.add_hline(
        y=w_prime_input,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text=f"W' = 100% ({w_prime_input} J)",
        annotation_position="top right",
    )

    # 50% threshold
    fig.add_hline(
        y=w_prime_input * 0.50,
        line_dash="dot",
        line_color="#FFC107",
        annotation_text="50%",
        annotation_position="bottom right",
    )

    # 20% critical zone
    fig.add_hline(
        y=w_prime_input * 0.20,
        line_dash="dot",
        line_color="#F44336",
        annotation_text="20% (krytyczna)",
        annotation_position="bottom right",
    )

    # Shade critical zone
    fig.add_hrect(
        y0=0, y1=w_prime_input * 0.20,
        fillcolor="rgba(244, 67, 54, 0.1)",
        line_width=0,
    )

    fig.update_layout(
        title="W' Balance — wyczerpywanie i regeneracja zasobów beztlenowych",
        xaxis_title="Czas [s]",
        yaxis_title="W' Balance [J]",
        yaxis=dict(range=[0, w_prime_input * 1.1]),
        hovermode="x unified",
        **CHART_CONFIG,
    )

    chart(fig, key="wbal_timeline")

    # --- Power + W' overlay ---
    st.subheader("🔌 Moc + W' Balance")

    fig2 = go.Figure()

    # Power trace
    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=result_df["watts"],
        mode="lines",
        name="Moc [W]",
        line=dict(color=Config.COLOR_POWER if hasattr(Config, "COLOR_POWER") else "#2196F3", width=1),
        yaxis="y",
        hovertemplate="Czas: %{x:.0f}s<br>Moc: %{y:.0f} W<extra></extra>",
    ))

    # W' balance trace (secondary axis)
    fig2.add_trace(go.Scatter(
        x=x_vals,
        y=result_df["w_prime_pct"],
        mode="lines",
        name="W' [%]",
        line=dict(color="#FF6B35", width=2),
        yaxis="y2",
        hovertemplate="Czas: %{x:.0f}s<br>W': %{y:.1f}%<extra></extra>",
    ))

    # CP line
    fig2.add_hline(
        y=cp_input,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text=f"CP ({cp_input} W)",
        annotation_position="top right",
        yaxis="y",
    )

    fig2.update_layout(
        title="Moc i W' Balance na wspólnym wykresie",
        xaxis_title="Czas [s]",
        yaxis=dict(title="Moc [W]", side="left"),
        yaxis2=dict(title="W' [%]", side="right", overlaying="y", range=[0, 110]),
        hovermode="x unified",
        **CHART_CONFIG,
    )

    chart(fig2, key="power_wbal_overlay")

    # --- Reconstitution events table ---
    if summary.events:
        st.subheader("📋 Cykle Wyczerpania i Regeneracji")

        table = build_reconstitution_table(summary.events)
        st.dataframe(table, hide_index=True, width="stretch")

        # Visual timeline of events
        st.subheader("🗺️ Mapa Cykli")

        fig3 = go.Figure()

        for i, ev in enumerate(summary.events, 1):
            color = "#F44336" if ev.min_w_prime_pct < 15 else "#FF9800" if ev.min_w_prime_pct < 30 else "#4CAF50"

            # Depletion bar
            fig3.add_trace(go.Bar(
                name=f"Cykl {i} — wyczerpanie",
                x=[f"Cykl {i}"],
                y=[ev.recovery_duration_s / 60],
                marker_color=color,
                hovertext=(
                    f"Min W': {ev.min_w_prime_pct:.1f}% ({ev.min_w_prime_j:.0f} J)<br>"
                    f"Regeneracja: {ev.recovery_duration_s:.0f}s<br>"
                    f"Tempo: {ev.recovery_rate_j_per_s:.2f} J/s<br>"
                    f"Moc (wyczerpanie): {ev.intensity_during_depletion:.0f} W<br>"
                    f"Moc (regeneracja): {ev.intensity_during_recovery:.0f} W"
                ),
                hoverinfo="text",
            ))

        fig3.update_layout(
            title="Czas regeneracji dla każdego cyklu wyczerpania W'",
            yaxis_title="Czas regeneracji [min]",
            barmode="group",
            **CHART_CONFIG,
        )

        chart(fig3, key="wbal_cycles_map")

    # --- Theory ---
    with st.expander("📖 Teoria i Fizjologia Rekonstytucji W'", expanded=False):
        st.markdown("""
        ### Czym jest W'?

        W' (W prime) to skończony zasób pracy beztlenowej wyrażony w dżulach [J].
        Reprezentuje całkowitą ilość pracy, którą można wykonać powyżej Critical Power (CP)
        przed wyczerpaniem.

        W' składa się z:
        - **Fosfokreatyny (PCr)**: ~40% W', regeneracja w ~120s (faza szybka)
        - **Buforowanie H+**: ~25% W', regeneracja w ~300s
        - **Glikogen mięśniowy**: ~35% W', regeneracja w ~600s+ (faza wolna)

        ### Model Mono-eksponencjalny (Skiba 2015)

        Podczas regeneracji (P < CP):
        **W'(t) = W'cap - (W'cap - W'(t₀)) × e^(-Δt/τ)**

        Gdzie τ (tau) zależy od intensywności regeneracji:
        **τ = 546 × e^(-0.01 × DCP) + 316**
        DCP = CP - P (różnica między CP a mocą regeneracji)

        ### Model Bi-eksponencjalny (Caen 2021)

        nowszy model z dwoma fazami regeneracji:
        - **Faza szybka** (τ ≈ 120s): Resynteza PCr
        - **Faza wolna** (τ ≈ 600s): Metaboliczna regeneracja

        Welburn et al. (2025) potwierdzili lepsze dopasowanie modelu bi-eksponencjalnego
        ale wskazali na potrzebę indywidualnej kalibracji tau.

        ### Praktyczne Zastosowanie

        1. **Planowanie interwałów**: Ile czasu regeneracji potrzeba między powtórzeniami?
        2. **Strategia wyścigowa**: Kiedy można bezpiecznie zaatakować?
        3. **Monitorowanie zmęczenia**: Wolniejsza regeneracja = akumulacja zmęczenia
        4. **Trening specyficzny**: Poprawa szybkości regeneracji W' przez interwały

        ### Bibliografia (2020-2026)

        1. Caen, K. et al. (2021). "Bi-exponential W' reconstitution following depletion." *European Journal of Applied Physiology*, 121(9): 2487-2498.

        2. Chorley, A. (2022). "Practical application of the W' balance model in competitive cycling." *IJSPP*, 17(3): 445-454.

        3. Welburn, J. et al. (2025). "W' reconstitution modelling: A comparison of mono- and bi-exponential models." *European Journal of Applied Physiology*, 125(2): 234-247.

        4. Skiba, P.F. et al. (2015). "Validation of a novel intermittent W' model for cycling." *Medicine & Science in Sports & Exercise*, 47(3): 638-644.

        5. Goulding, R.P. & Marwood, S. (2023). "The W' balance model: Current understanding and future directions." *Sports Medicine*, 53(1): 1-15.

        6. Ferguson, C. et al. (2022). "W' reconstitution kinetics in trained and untrained individuals." *Journal of Applied Physiology*, 132(4): 987-996.
        """)
