"""Sleep and Recovery UI tab."""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st

from modules.calculations.sleep_recovery import (
    SleepData,
    SleepScore,
    calculate_sleep_score,
    calculate_composite_recovery,
    parse_garmin_sleep_csv,
    parse_oura_sleep_csv,
)
from modules.ui.shared import chart, metric


def render_sleep_recovery_tab(*args, **kwargs) -> None:
    st.subheader("😴 Sen i Odnowa")
    st.caption("Integracja danych o śnie z Garmin/Oura — wskaźnik odnowy i rekomendacje")

    sleep_data = _render_file_uploader()
    _render_last_night(sleep_data)
    _render_composite_recovery()
    _render_trend_chart(sleep_data)
    _render_theory()


def _render_file_uploader() -> Optional[List[SleepData]]:
    st.markdown("### 📤 Wgraj dane snu")
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio("Źródło danych", ["Garmin Connect", "Oura Ring"], key="sleep_source")
    with col2:
        csv_file = st.file_uploader(
            "Wybierz plik CSV",
            type=["csv"],
            key="sleep_csv_upload",
        )

    if csv_file is None:
        st.info("ℹ️ Wgraj plik CSV z danymi snu (Garmin lub Oura), aby zobaczyć analizę.")
        return None

    import tempfile
    import os

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.getvalue())
            tmp_path = tmp.name

        if source == "Garmin Connect":
            data = parse_garmin_sleep_csv(tmp_path)
        else:
            data = parse_oura_sleep_csv(tmp_path)

        os.unlink(tmp_path)

        if not data:
            st.warning("Nie udało się sparsować pliku. Sprawdź format CSV.")
            return None

        st.success(f"✅ Wczytano {len(data)} nocy snu.")
        st.session_state["sleep_data_cache"] = data
        return data
    except Exception:
        st.error("Błąd wczytywania pliku CSV.")
        return None


def _get_cached_sleep() -> Optional[List[SleepData]]:
    if "sleep_data_cache" in st.session_state:
        return st.session_state["sleep_data_cache"]
    return None


def _render_last_night(sleep_data: Optional[List[SleepData]]) -> None:
    data = sleep_data or _get_cached_sleep()
    if not data:
        return

    last = data[-1]
    score = calculate_sleep_score(last)

    st.markdown("### 🌙 Ostatnia noc")
    col1, col2, col3, col4 = st.columns(4)
    metric("Czas snu", f"{last.total_sleep_hours:.1f} h", column=col1)
    metric("Głęboki", f"{last.deep_sleep_hours:.1f} h", column=col2)
    metric("REM", f"{last.rem_sleep_hours:.1f} h", column=col3)
    metric("Wynik snu", f"{score.score:.0f}/100", column=col4, delta=score.level)

    col5, col6, col7 = st.columns(3)
    metric("Efektywność", f"{last.sleep_efficiency_pct:.0f}%", column=col5)
    metric("Pobudki", f"{last.wake_count}", column=col6)
    if last.hr_during_sleep:
        metric("HR sen", f"{last.hr_during_sleep:.0f} bpm", column=col7)

    st.info(score.interpretation)


def _render_composite_recovery() -> None:
    st.markdown("### 🔄 Kompozytowy wskaźnik odnowy")

    from modules.calculations.pmc import get_current_pmc_summary
    from modules.cache_utils import get_session_store

    store = get_session_store()
    pmc = get_current_pmc_summary(store)

    sleep_score = st.session_state.get("_manual_sleep_score", 65.0)
    hrv_score = 65.0

    if pmc:
        tsb = pmc["tsb"]
    else:
        tsb = 0.0

    col1, col2 = st.columns(2)
    with col1:
        sleep_score = st.slider("Wynik snu (0-100)", 0, 100, int(sleep_score), key="comp_sleep")
    with col2:
        hrv_score = st.slider("Wynik HRV readiness (0-100)", 0, 100, int(hrv_score), key="comp_hrv")

    recovery = calculate_composite_recovery(sleep_score, hrv_score, tsb)

    col1, col2, col3, col4 = st.columns(4)
    metric("Sen", f"{recovery.sleep_score:.0f}", column=col1)
    metric("HRV", f"{recovery.hrv_readiness:.0f}", column=col2)
    metric("Obciążenie", recovery.training_load_status, column=col3)
    metric("Odnowa", f"{recovery.composite_score:.0f}/100", column=col4)

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=recovery.composite_score,
            title={"text": "Kompozytowa odnowa"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#27ae60" if recovery.composite_score > 60 else "#e74c3c"},
                "steps": [
                    {"range": [0, 40], "color": "#e74c3c22"},
                    {"range": [40, 70], "color": "#f39c1222"},
                    {"range": [70, 100], "color": "#27ae6022"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    chart(fig, key="sleep_composite_gauge")

    st.success(recovery.recommendation)


def _render_trend_chart(sleep_data: Optional[List[SleepData]]) -> None:
    data = sleep_data or _get_cached_sleep()
    if not data or len(data) < 2:
        return

    st.markdown("### 📈 Trend snu (7 dni)")

    recent = data[-7:]
    dates = [d.date for d in recent]
    total_h = [d.total_sleep_hours for d in recent]
    deep_h = [d.deep_sleep_hours for d in recent]
    rem_h = [d.rem_sleep_hours for d in recent]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=dates, y=total_h, name="Całkowity sen", marker_color="#3498db", opacity=0.7)
    )
    fig.add_trace(
        go.Bar(x=dates, y=deep_h, name="Sen głęboki", marker_color="#2c3e50", opacity=0.8)
    )
    fig.add_trace(go.Bar(x=dates, y=rem_h, name="REM", marker_color="#9b59b6", opacity=0.8))
    fig.add_hline(y=7, line_dash="dash", line_color="#27ae60", annotation_text="Optimum 7h")
    fig.add_hline(y=9, line_dash="dash", line_color="#27ae60", annotation_text="Optimum 9h")

    fig.update_layout(
        title="Czas snu — ostatnie 7 nocy",
        xaxis_title="Data",
        yaxis_title="Godziny",
        barmode="group",
        height=350,
        hovermode="x unified",
    )
    chart(fig, key="sleep_trend_chart")

    scores = [calculate_sleep_score(d) for d in recent]
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=dates,
            y=[s.score for s in scores],
            name="Wynik snu",
            line=dict(color="#27ae60", width=2.5),
            fill="tozeroy",
            fillcolor="#27ae6022",
        )
    )
    fig2.update_layout(
        title="Jakość snu — wynik 0-100",
        xaxis_title="Data",
        yaxis_title="Wynik",
        height=300,
    )
    chart(fig2, key="sleep_score_trend")


def _render_theory() -> None:
    with st.expander("📖 Teoria — Sen i odnowa"):
        st.markdown("""
        **Sen** jest najważniejszym czynnikiem odnowy po treningu:

        | Faza snu | Proporcja | Znaczenie dla sportowca |
        |----------|-----------|------------------------|
        | Głęboki (N3) | 15-25% | Regeneracja fizyczna, GH, naprawa tkanek |
        | REM | 20-25% | Konsolidacja pamięci, regeneracja nerwowa |
        | Płytki (N1/N2) | 50-60% | Przejście, konsolidacja |

        **Kompozytowy wskaźnik odnowy:**
        - Sen: 30% wagi
        - HRV readiness: 40% wagi
        - Obciążenie treningowe (TSB): 30% wagi

        **Zalecenia:**
        - Celuj w 7-9h snu każdej nocy
        - Efektywność snu >85% (>90% = optymalnie)
        - Minimum 1.5h snu głębokiego
        - Regularność: stała pora snu ±30 min
        """)
