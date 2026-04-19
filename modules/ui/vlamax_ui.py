"""VLamax Metabolic Profile tab."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from modules.calculations.vlamax_profile import (
    build_vlamax_profile,
    get_metabolic_profile_chart_data,
    compare_vlamax_longitudinal,
    VLamaxProfile,
)
from modules.ui.shared import chart, metric


_RIDER_COLORS = {
    "Sprinter": "#e74c3c",
    "Puncheur": "#f39c12",
    "All-rounder": "#3498db",
    "Climber": "#27ae60",
}


def render_vlamax_tab(df_plot, cp_input, w_prime_input, rider_weight, *args, **kwargs) -> None:
    """Render the VLamax Metabolic Profile tab."""
    st.subheader("🧬 Profil Metaboliczny — VLamax")
    st.caption("Analiza profilu metabolicznego na podstawie krzywej mocy-czas")

    if df_plot is None or (hasattr(df_plot, "empty") and df_plot.empty):
        st.info("ℹ️ Wgraj plik z danymi, aby zobaczyć profil VLamax.")
        return

    pdc = _extract_pdc(df_plot)
    if not pdc:
        st.warning("Nie udało się wyznaczyć krzywej PDC z danych.")
        return

    weight = rider_weight if rider_weight and rider_weight > 0 else 70.0
    profile = build_vlamax_profile(pdc, weight)
    if not profile:
        st.warning("Nie udało się oszacować VLamax.")
        return

    _render_current_profile(profile)
    _render_metabolic_chart(profile, pdc, weight)
    _render_rider_classification(profile)
    _render_theory()


def _extract_pdc(df) -> dict:
    from modules.calculations.power import calculate_power_duration_curve

    try:
        result = calculate_power_duration_curve(df)
        if result and isinstance(result, dict):
            return result
    except Exception:
        pass

    power_col = None
    for col in ["watts", "power", "np_power"]:
        if col in df.columns:
            power_col = col
            break

    if not power_col:
        return {}

    pdc = {}
    durations = [5, 15, 30, 60, 120, 240, 480, 960, 1920]
    power = df[power_col].dropna().values
    for d in durations:
        if len(power) >= d:
            from modules.calculations.data_processing import ensure_pandas

            pdc[float(d)] = float(np.mean(np.sort(power)[-d:]))
    return pdc


def _render_current_profile(profile: VLamaxProfile) -> None:
    col1, col2, col3 = st.columns(3)
    metric("VLamax", f"{profile.vlamax:.3f}", column=col1, suffix=" mmol/l/s")
    metric("Pewność", f"{profile.confidence:.0%}", column=col2)
    metric("Typ kolarza", profile.rider_type, column=col3)


def _render_metabolic_chart(profile: VLamaxProfile, pdc: dict, weight: float) -> None:
    chart_data = get_metabolic_profile_chart_data(profile)

    if chart_data["durations"]:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Aerobowy",
                x=chart_data["durations"],
                y=chart_data["aerobic_pct"],
                marker_color="#3498db",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Anaerobowy",
                x=chart_data["durations"],
                y=chart_data["anaerobic_pct"],
                marker_color="#e74c3c",
            )
        )
        fig.update_layout(
            barmode="stack",
            title="Profil Metaboliczny — Udział Aerobowy vs Anaerobowy",
            xaxis_title="Czas trwania",
            yaxis_title="Udział (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
        )
        chart(fig, key="vlamax_metabolic_chart")
    else:
        st.info("Podaj VO2max w parametrach, aby zobaczyć dekompozycję metaboliczną.")


def _render_rider_classification(profile: VLamaxProfile) -> None:
    color = _RIDER_COLORS.get(profile.rider_type, "#95a5a6")
    st.markdown(
        f'<div style="background: {color}22; padding: 12px 15px; border-radius: 8px; '
        f'border-left: 4px solid {color}; margin: 10px 0;">'
        f"<b>Typ kolarza:</b> {profile.rider_type} (VLamax = {profile.vlamax:.3f} mmol/l/s)"
        f"</div>",
        unsafe_allow_html=True,
    )

    classifications = {
        "Sprinter": "≥ 1.0 mmol/l/s — Dominacja anaerobowa, wysoka moc szczytowa",
        "Puncheur": "0.7–1.0 mmol/l/s — Mieszany profil, mocny na krótkich podjazdach",
        "All-rounder": "0.45–0.7 mmol/l/s — Zrównoważony profil wszechstronny",
        "Climber": "< 0.45 mmol/l/s — Dominacja tlenowa, wytrzymałość na podjazdach",
    }
    st.info(f"ℹ️ {classifications.get(profile.rider_type, 'Brak opisu.')}")


def _render_theory() -> None:
    with st.expander("📖 Teoria — VLamax i Profil Metaboliczny"):
        st.markdown("""
        **VLamax** (maksymalna szybkość produkcji mleczanu) określa profil metaboliczny kolarza:

        | Typ | VLamax (mmol/l/s) | Charakterystyka |
        |-----|-------------------|-----------------|
        | **Sprinter** | ≥ 1.0 | Wysoka moc szczytowa, szybkie wyczerpanie W' |
        | **Puncheur** | 0.7–1.0 | Krótkie ataki, moc na 1-5 min |
        | **All-rounder** | 0.45–0.7 | Zrównoważony, wszechstronny |
        | **Climber** | < 0.45 | Wysoki VO2max, wytrzymałość sub-CV |

        **Ograniczenia:** Estymacja z PDC ma niepewność ±15-20%.
        """)
