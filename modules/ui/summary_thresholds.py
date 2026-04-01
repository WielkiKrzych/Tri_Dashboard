"""
Summary Thresholds — VT/LT/TDI threshold renderers for the Summary tab.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from .summary_calculations import _get_vent_metrics_for_power


def _render_vent_thresholds_summary(df_plot, cp_input, vt1_watts, vt2_watts, threshold_result):
    """Renderowanie wykresu progów wentylacyjnych VT1/VT2."""
    if "tymeventilation" not in df_plot.columns:
        st.info("Brak danych wentylacji do analizy progów VT.")
        return

    df_plot["ve_smooth"] = df_plot["tymeventilation"].rolling(window=10, center=True).mean()
    if "watts_smooth_5s" not in df_plot.columns and "watts" in df_plot.columns:
        df_plot["watts_smooth_5s"] = df_plot["watts"].rolling(window=5, center=True).mean()

    vt1_w = vt1_watts
    vt2_w = vt2_watts

    if threshold_result.vt1_watts and abs(threshold_result.vt1_watts - vt1_w) < 10:
        vt1_hr = threshold_result.vt1_hr or 0
        vt1_ve = threshold_result.vt1_ve or 0
        vt1_br = threshold_result.vt1_br or 0
    else:
        vt1_hr, vt1_ve, vt1_br = _get_vent_metrics_for_power(df_plot, vt1_w)

    if threshold_result.vt2_watts and abs(threshold_result.vt2_watts - vt2_w) < 10:
        vt2_hr = threshold_result.vt2_hr or 0
        vt2_ve = threshold_result.vt2_ve or 0
        vt2_br = threshold_result.vt2_br or 0
    else:
        vt2_hr, vt2_ve, vt2_br = _get_vent_metrics_for_power(df_plot, vt2_w)

    vt1_tv = (vt1_ve / vt1_br * 1000) if vt1_ve and vt1_br else 0
    vt2_tv = (vt2_ve / vt2_br * 1000) if vt2_ve and vt2_br else 0

    fig_vent = go.Figure()

    time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

    fig_vent.add_trace(
        go.Scatter(
            x=time_x,
            y=df_plot["ve_smooth"],
            mode="lines",
            name="VE (L/min)",
            line=dict(color="#ffa15a", width=2),
        )
    )

    if "watts_smooth_5s" in df_plot.columns:
        fig_vent.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["watts_smooth_5s"],
                mode="lines",
                name="Power",
                line=dict(color="#1f77b4", width=1),
                yaxis="y2",
                opacity=0.3,
            )
        )

    if vt1_w and threshold_result.step_ve_analysis:
        for step in threshold_result.step_ve_analysis:
            if step.get("is_vt1"):
                marker_time = step.get("end_time", 0)
                fig_vent.add_vline(x=marker_time, line=dict(color="#ffa15a", width=3, dash="dash"))

    if vt2_w and threshold_result.step_ve_analysis:
        for step in threshold_result.step_ve_analysis:
            if step.get("is_vt2"):
                marker_time = step.get("end_time", 0)
                fig_vent.add_vline(x=marker_time, line=dict(color="#ef553b", width=3, dash="dash"))

    fig_vent.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="VE (L/min)"),
        yaxis2=dict(title="Moc (W)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_vent, width="stretch")

    col_z1, col_z2 = st.columns(2)

    with col_z1:
        if vt1_w:
            st.markdown(
                f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt1_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt1_hr)} bpm</p>' if vt1_hr else ""}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt1_ve:.1f} L/min</p>' if vt1_ve else ""}
                {f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt1_br)} /min</p>' if vt1_br else ""}
                {f'<p style="margin:0; color:#aaa;"><b>TV:</b> {vt1_tv:.0f} mL</p>' if vt1_tv else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(vt1_w / cp_input) * 100:.0f}% CP")
        else:
            st.info("VT1: Nie wykryto")

    with col_z2:
        if vt2_w:
            st.markdown(
                f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt2_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt2_hr)} bpm</p>' if vt2_hr else ""}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt2_ve:.1f} L/min</p>' if vt2_ve else ""}
                {f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt2_br)} /min</p>' if vt2_br else ""}
                {f'<p style="margin:0; color:#aaa;"><b>TV:</b> {vt2_tv:.0f} mL</p>' if vt2_tv else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(vt2_w / cp_input) * 100:.0f}% CP")
        else:
            st.info("VT2: Nie wykryto")


def _render_smo2_thresholds_summary(df_plot, cp_input, lt1_watts, lt2_watts, smo2_result):
    """Renderowanie wykresu progów SmO2 LT1/LT2."""
    if "smo2" not in df_plot.columns:
        st.info("Brak danych SmO2 do analizy progów LT.")
        return

    df_plot["smo2_smooth"] = df_plot["smo2"].rolling(window=10, center=True).mean()
    if "watts_smooth_5s" not in df_plot.columns and "watts" in df_plot.columns:
        df_plot["watts_smooth_5s"] = df_plot["watts"].rolling(window=5, center=True).mean()

    lt1_w = lt1_watts
    lt2_w = lt2_watts
    lt1_hr = smo2_result.t1_hr if smo2_result and smo2_result.t1_hr else 0
    lt2_hr = smo2_result.t2_onset_hr if smo2_result and smo2_result.t2_onset_hr else 0
    lt1_smo2 = smo2_result.t1_smo2 if smo2_result and smo2_result.t1_smo2 else 0
    lt2_smo2 = smo2_result.t2_onset_smo2 if smo2_result and smo2_result.t2_onset_smo2 else 0

    fig_smo2 = go.Figure()

    time_x = df_plot["time"] if "time" in df_plot.columns else range(len(df_plot))

    fig_smo2.add_trace(
        go.Scatter(
            x=time_x,
            y=df_plot["smo2_smooth"],
            mode="lines",
            name="SmO2 (%)",
            line=dict(color="#2ca02c", width=2),
        )
    )

    if "watts_smooth_5s" in df_plot.columns:
        fig_smo2.add_trace(
            go.Scatter(
                x=time_x,
                y=df_plot["watts_smooth_5s"],
                mode="lines",
                name="Power",
                line=dict(color="#1f77b4", width=1),
                yaxis="y2",
                opacity=0.3,
            )
        )

    def find_time_for_power(power):
        if power <= 0:
            return None
        if "watts_smooth_5s" in df_plot.columns:
            idx = (df_plot["watts_smooth_5s"] - power).abs().idxmin()
            return df_plot.loc[idx, "time"] if "time" in df_plot.columns else idx
        elif "watts" in df_plot.columns:
            idx = (df_plot["watts"] - power).abs().idxmin()
            return df_plot.loc[idx, "time"] if "time" in df_plot.columns else idx
        return None

    lt1_time = find_time_for_power(lt1_w) if lt1_w else None
    lt2_time = find_time_for_power(lt2_w) if lt2_w else None

    if lt1_time is not None:
        fig_smo2.add_vline(x=lt1_time, line=dict(color="#2ca02c", width=3, dash="dash"))

    if lt2_time is not None:
        fig_smo2.add_vline(x=lt2_time, line=dict(color="#d62728", width=3, dash="dash"))

    fig_smo2.update_layout(
        template="plotly_dark",
        height=350,
        yaxis=dict(title="SmO2 (%)"),
        yaxis2=dict(title="Moc (W)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_smo2, width="stretch")

    col_z1, col_z2 = st.columns(2)

    with col_z1:
        if lt1_w:
            st.markdown(
                f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #2ca02c; background-color: #222;">
                <h3 style="margin:0; color: #2ca02c;">LT1 (SteadyState)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(lt1_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(lt1_hr)} bpm</p>' if lt1_hr else ""}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {lt1_smo2:.1f}%</p>' if lt1_smo2 else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(lt1_w / cp_input) * 100:.0f}% CP")
        else:
            st.info("LT1: Nie wykryto")

    with col_z2:
        if lt2_w:
            st.markdown(
                f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #d62728; background-color: #222;">
                <h3 style="margin:0; color: #d62728;">LT2 (Próg)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(lt2_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(lt2_hr)} bpm</p>' if lt2_hr else ""}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {lt2_smo2:.1f}%</p>' if lt2_smo2 else ""}
            </div>
            """,
                unsafe_allow_html=True,
            )
            if cp_input > 0:
                st.caption(f"~{(lt2_w / cp_input) * 100:.0f}% CP")
        else:
            st.info("LT2: Nie wykryto")


def _render_tdi_analysis(vt1_watts: int, lt1_watts: int):
    """
    Renderowanie analizy TDI porównującej VT1 (wentylacyjny) z LT1 (SmO2).

    TDI = |VT1_VE - LT1_SmO2| / VT1_VE * 100 [%]

    Klasyfikacja:
    - <5% = system zgodny
    - 5-10% = heterogeniczna adaptacja
    - >10% = konflikt centralno-obwodowy / okluzja / perfuzja
    """
    if not vt1_watts or vt1_watts <= 0:
        st.warning("⚠️ **Brak danych VT1 (wentylacyjny)** — nie można obliczyć TDI.")
        return

    if not lt1_watts or lt1_watts <= 0:
        st.warning("⚠️ **Brak danych LT1 (SmO2)** — nie można obliczyć TDI.")
        return

    tdi = abs(vt1_watts - lt1_watts) / vt1_watts * 100
    delta = lt1_watts - vt1_watts

    if tdi < 5:
        classification = "ZGODNY"
        color = "#00cc96"
        alert_type = "success"
        interpretation = "System tlenowy i obwodowy są zsynchronizowane. Optymalna koordynacja między wentylacją a perfuzją mięśniową."
        recommendation = "✅ **Trening:** Możesz trenować w pełnym zakresie intensywności. System transportu tlenu działa harmonijnie."
    elif tdi <= 10:
        classification = "HETEROGENICZNY"
        color = "#ffa15a"
        alert_type = "warning"
        interpretation = "Wykryto niewielką rozbieżność między progiem wentylacyjnym a progiem SmO2. Może wskazywać na różne tempo adaptacji systemów centralnego i obwodowego."
        if delta > 0:
            recommendation = "⚡ **Trening:** Skup się na treningach tempo (Sweet Spot) aby wyrównać adaptację obwodową. SmO2 wskazuje wyższy próg niż VE — mięśnie adaptują się szybciej niż układ oddechowy."
        else:
            recommendation = "🫁 **Trening:** Zwiększ udział treningów Z2 i długich wyjazdów. VE wskazuje wyższy próg niż SmO2 — układ oddechowy wyprzedza adaptację mięśniową."
    else:
        classification = "KONFLIKT"
        color = "#ef553b"
        alert_type = "error"
        interpretation = "Znacząca rozbieżność między systemem centralnym (wentylacja) a obwodowym (perfuzja mięśniowa). Możliwe przyczyny: okluzja naczyniowa, zaburzenia mikrokrążenia, lub błąd pomiaru sensora NIRS."
        if delta > 0:
            recommendation = "🔴 **Uwaga:** SmO2 znacząco wyżej niż VE. Sprawdź: (1) pozycję sensora NIRS, (2) grubość tkanki tłuszczowej, (3) okluzję podczas pedałowania. Rozważ konsultację z fizjologiem."
        else:
            recommendation = "🔴 **Uwaga:** VE znacząco wyżej niż SmO2. Sprawdź: (1) kalibrację sensora wentylacyjnego, (2) możliwą hiperperfuzję centralną, (3) ograniczenia mikrokrążenia obwodowego."

    st.markdown(
        f"""
    <div style="padding:20px; border-radius:12px; border:3px solid {color}; background-color: #1a1a1a; text-align:center;">
        <h2 style="margin:0; color: {color};">TDI: {tdi:.1f}%</h2>
        <p style="margin:5px 0; font-size:1.2em; color: {color}; font-weight:bold;">{classification}</p>
        <p style="margin:10px 0 0 0; color:#888; font-size:0.85em;">
            VT1 (VE): <b>{vt1_watts:.0f} W</b> | LT1 (SmO2): <b>{lt1_watts:.0f} W</b> | Δ = {delta:+.0f} W
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if alert_type == "success":
        st.success(f"✅ **Interpretacja:** {interpretation}")
    elif alert_type == "warning":
        st.warning(f"⚠️ **Interpretacja:** {interpretation}")
    else:
        st.error(f"🔴 **Interpretacja:** {interpretation}")

    st.info(recommendation)

    with st.expander("📖 Co to jest TDI?", expanded=False):
        st.markdown("""
        ### Threshold Discordance Index (TDI)

        TDI mierzy **rozbieżność** między dwoma kluczowymi progami metabolicznymi:

        | Próg | Źródło | Mechanizm |
        |------|--------|-----------|
        | **VT1 (Ventilatory)** | Wentylacja (VE) | Punkt, w którym układ oddechowy zaczyna kompensować narastającą kwasicę |
        | **LT1 (SmO2)** | Oksygenacja mięśniowa | Punkt, w którym ekstrakcja tlenu w mięśniu zaczyna przewyższać dostawę |

        ---

        #### Wzór
        ```
        TDI = |VT1 - LT1| / VT1 × 100%
        ```

        #### Interpretacja kliniczna

        | TDI | Stan | Znaczenie |
        |-----|------|-----------|
        | **< 5%** | Zgodny | Systemy centralny i obwodowy doskonale zsynchronizowane |
        | **5–10%** | Heterogeniczny | Różny tempo adaptacji — centralny vs obwodowy |
        | **> 10%** | Konflikt | Potencjalny problem z transportem O2 lub błąd pomiaru |

        #### Przyczyny rozbieżności

        1. **LT1 > VT1** (SmO2 wyżej):
           - Mięśnie dobrze ukrwione, ale wentylacja za wolna
           - Częste u osób z wysokim VO2max ale niską wydolnością oddechową

        2. **VT1 > LT1** (VE wyżej):
           - Układ oddechowy sprawny, ale perfuzja mięśniowa ograniczona
           - Może wskazywać na problemy z mikrokrążeniem lub niedopasowanie kadencji

        ---

        *Źródło: Adaptacja modelu NIRS-CPET integration (Feldmann et al., 2020)*
        """)
