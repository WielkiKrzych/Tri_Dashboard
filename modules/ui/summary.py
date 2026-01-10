"""
Moduł UI: Zakładka Podsumowanie (Summary)

Agreguje kluczowe wykresy i metryki z całego dashboardu w jednym miejscu.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from modules.config import Config
from modules.calculations.thresholds import analyze_step_test


def render_summary_tab(
    df_plot: pd.DataFrame,
    df_plot_resampled: pd.DataFrame,
    metrics: dict,
    training_notes,
    uploaded_file_name: str,
    cp_input: int,
    w_prime_input: int,
    rider_weight: float,
    vt1_watts: int = 0,
    vt2_watts: int = 0,
    lt1_watts: int = 0,
    lt2_watts: int = 0,
):
    """Renderowanie zakładki Podsumowanie z kluczowymi wykresami i metrykami."""
    st.header("📊 Podsumowanie Treningu")
    st.markdown("Wszystkie kluczowe wykresy i metryki w jednym miejscu.")

    # Normalize columns
    df_plot.columns = df_plot.columns.str.lower().str.strip()

    # --- SHARED THRESHOLD DETECTION ---
    # We perform detection once here to be used across multiple sections (5, 6, 7)
    hr_col = None
    for alias in ['hr', 'heartrate', 'heart_rate', 'bpm']:
        if alias in df_plot.columns:
            hr_col = alias
            break

    threshold_result = analyze_step_test(
        df_plot,
        power_column='watts',
        ve_column='tymeventilation' if 'tymeventilation' in df_plot.columns else None,
        smo2_column='smo2' if 'smo2' in df_plot.columns else None,
        hr_column=hr_col,
        time_column='time'
    )

    # Use detected values if parameters are 0
    eff_vt1 = vt1_watts if vt1_watts > 0 else (threshold_result.vt1_watts if threshold_result.vt1_watts else 0)
    eff_vt2 = vt2_watts if vt2_watts > 0 else (threshold_result.vt2_watts if threshold_result.vt2_watts else 0)
    eff_lt1 = lt1_watts if lt1_watts > 0 else (threshold_result.smo2_1_watts if threshold_result.smo2_1_watts else 0)
    eff_lt2 = lt2_watts if lt2_watts > 0 else (threshold_result.smo2_2_watts if threshold_result.smo2_2_watts else 0)

    # =========================================================================
    # 1. WYKRES PRZEBIEG TRENINGU
    # =========================================================================
    st.subheader("1️⃣ Przebieg Treningu")
    
    fig_training = go.Figure()
    time_x = df_plot['time_min'] if 'time_min' in df_plot.columns else df_plot['time'] / 60 if 'time' in df_plot.columns else None
    
    if time_x is not None:
        if 'watts_smooth' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['watts_smooth'],
                name='Moc', fill='tozeroy',
                line=dict(color=Config.COLOR_POWER, width=1),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>"
            ))
        elif 'watts' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['watts'].rolling(5, center=True).mean(),
                name='Moc', fill='tozeroy',
                line=dict(color=Config.COLOR_POWER, width=1),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>"
            ))

        if 'heartrate_smooth' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['heartrate_smooth'],
                name='HR', line=dict(color=Config.COLOR_HR, width=2),
                yaxis='y2', hovertemplate="HR: %{y:.0f} bpm<extra></extra>"
            ))
        elif 'heartrate' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['heartrate'],
                name='HR', line=dict(color=Config.COLOR_HR, width=2),
                yaxis='y2', hovertemplate="HR: %{y:.0f} bpm<extra></extra>"
            ))

        if 'smo2_smooth' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['smo2_smooth'],
                name='SmO2', line=dict(color=Config.COLOR_SMO2, width=2, dash='dot'),
                yaxis='y3', hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
            ))
        elif 'smo2' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['smo2'].rolling(5, center=True).mean(),
                name='SmO2', line=dict(color=Config.COLOR_SMO2, width=2, dash='dot'),
                yaxis='y3', hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
            ))

        if 'tymeventilation_smooth' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['tymeventilation_smooth'],
                name='VE', line=dict(color=Config.COLOR_VE, width=2, dash='dash'),
                yaxis='y4', hovertemplate="VE: %{y:.1f} L/min<extra></extra>"
            ))
        elif 'tymeventilation' in df_plot.columns:
            fig_training.add_trace(go.Scatter(
                x=time_x, y=df_plot['tymeventilation'].rolling(10, center=True).mean(),
                name='VE', line=dict(color=Config.COLOR_VE, width=2, dash='dash'),
                yaxis='y4', hovertemplate="VE: %{y:.1f} L/min<extra></extra>"
            ))

    fig_training.update_layout(
        template="plotly_dark", height=450,
        yaxis=dict(title="Moc [W]"),
        yaxis2=dict(title="HR", overlaying='y', side='right', showgrid=False),
        yaxis3=dict(title="SmO2", overlaying='y', side='right', showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis4=dict(title="VE", overlaying='y', side='right', showgrid=False, showticklabels=False),
        legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_training, use_container_width=True)

    # =========================================================================
    # 1a. METRYKI POD WYKRESEM
    # =========================================================================
    _render_metrics_panel(df_plot, metrics, cp_input, w_prime_input, rider_weight)

    st.markdown("---")

    # =========================================================================
    # 2. WYKRES WENTYLACJA (VE) I ODDECHY (BR)
    # =========================================================================
    st.subheader("2️⃣ Wentylacja (VE) i Oddechy (BR)")
    
    if 'tymeventilation' in df_plot.columns:
        fig_ve_br = make_subplots(specs=[[{"secondary_y": True}]])
        
        time_x_s = df_plot['time'] if 'time' in df_plot.columns else range(len(df_plot))
        
        # VE
        ve_data = df_plot['tymeventilation'].rolling(10, center=True).mean() if 'tymeventilation' in df_plot.columns else None
        if ve_data is not None:
            fig_ve_br.add_trace(go.Scatter(
                x=time_x_s, y=ve_data,
                name='VE (L/min)', line=dict(color='#ffa15a', width=2),
                hovertemplate="VE: %{y:.1f} L/min<extra></extra>"
            ), secondary_y=False)
        
        # BR
        if 'tymebreathrate' in df_plot.columns:
            br_data = df_plot['tymebreathrate'].rolling(10, center=True).mean()
            fig_ve_br.add_trace(go.Scatter(
                x=time_x_s, y=br_data,
                name='BR (oddech/min)', line=dict(color='#00cc96', width=2),
                hovertemplate="BR: %{y:.0f} /min<extra></extra>"
            ), secondary_y=True)
        
        fig_ve_br.update_layout(
            template="plotly_dark", height=350,
            legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        fig_ve_br.update_yaxes(title_text="VE (L/min)", secondary_y=False)
        fig_ve_br.update_yaxes(title_text="BR (/min)", secondary_y=True)
        st.plotly_chart(fig_ve_br, use_container_width=True)
    else:
        st.info("Brak danych wentylacji (VE/BR) w tym pliku.")

    st.markdown("---")

    # =========================================================================
    # 3. WYKRES MATEMATYCZNY MODEL CP
    # =========================================================================
    st.subheader("3️⃣ Model Matematyczny CP")
    _render_cp_model_chart(df_plot, cp_input, w_prime_input)

    st.markdown("---")

    # =========================================================================
    # 4. WYKRES SmO2 vs THb W CZASIE
    # =========================================================================
    st.subheader("4️⃣ SmO2 vs THb w czasie")
    _render_smo2_thb_chart(df_plot)

    st.markdown("---")

    # =========================================================================
    # 5. PROGI WENTYLACYJNE VT1/VT2
    # =========================================================================
    st.subheader("5️⃣ Progi Wentylacyjne (VT1/VT2)")
    _render_vent_thresholds_summary(df_plot, cp_input, eff_vt1, eff_vt2, threshold_result)

    st.markdown("---")

    # =========================================================================
    # 6. PROGI SmO2 LT1/LT2
    # =========================================================================
    st.subheader("6️⃣ Progi SmO2 (LT1/LT2)")
    _render_smo2_thresholds_summary(df_plot, cp_input, eff_lt1, eff_lt2, threshold_result)

    st.markdown("---")

    # =========================================================================
    # 7. THRESHOLD DISCORDANCE INDEX (TDI)
    # =========================================================================
    st.subheader("7️⃣ Threshold Discordance Index (TDI)")
    _render_tdi_analysis(eff_vt1, eff_lt1)

    st.markdown("---")

    # =========================================================================
    # 8. VO2max UNCERTAINTY ESTIMATION (CI95%)
    # =========================================================================
    st.subheader("8️⃣ Estymacja VO2max z Niepewnością (CI95%)")
    _render_vo2max_uncertainty(df_plot, rider_weight)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _render_metrics_panel(df_plot, metrics, cp_input, w_prime_input, rider_weight):
    """Renderowanie panelu z metrykami pod wykresem przebiegu treningu."""
    
    # Oblicz metryki z danych
    duration_min = len(df_plot) / 60 if len(df_plot) > 0 else 0
    
    # Power
    avg_power = df_plot['watts'].mean() if 'watts' in df_plot.columns else 0
    np_power = _calculate_np(df_plot['watts']) if 'watts' in df_plot.columns else 0
    work_kj = df_plot['watts'].sum() / 1000 if 'watts' in df_plot.columns else 0
    
    # HR
    hr_col = 'heartrate' if 'heartrate' in df_plot.columns else 'hr' if 'hr' in df_plot.columns else None
    avg_hr = df_plot[hr_col].mean() if hr_col else 0
    min_hr = df_plot[hr_col].min() if hr_col else 0
    max_hr = df_plot[hr_col].max() if hr_col else 0
    
    # SmO2
    avg_smo2 = df_plot['smo2'].mean() if 'smo2' in df_plot.columns else 0
    min_smo2 = df_plot['smo2'].min() if 'smo2' in df_plot.columns else 0
    max_smo2 = df_plot['smo2'].max() if 'smo2' in df_plot.columns else 0
    
    # VE
    avg_ve = df_plot['tymeventilation'].mean() if 'tymeventilation' in df_plot.columns else 0
    min_ve = df_plot['tymeventilation'].min() if 'tymeventilation' in df_plot.columns else 0
    max_ve = df_plot['tymeventilation'].max() if 'tymeventilation' in df_plot.columns else 0
    
    # BR
    avg_br = df_plot['tymebreathrate'].mean() if 'tymebreathrate' in df_plot.columns else 0
    min_br = df_plot['tymebreathrate'].min() if 'tymebreathrate' in df_plot.columns else 0
    max_br = df_plot['tymebreathrate'].max() if 'tymebreathrate' in df_plot.columns else 0
    
    # Estymacje
    est_vo2max = metrics.get('vo2_max_est', 0) if metrics else 0
    est_vlamax = metrics.get('vlamax_est', 0) if metrics else 0
    
    # Estymacja CP/W' z danych
    est_cp, est_w_prime = _estimate_cp_wprime(df_plot)

    # Wyświetlanie w 4 kolumnach
    st.markdown("### 📈 Metryki Treningowe")
    
    # Wiersz 1: Czas, Moc, NP, Praca
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⏱️ Czas", f"{duration_min:.1f} min")
    c2.metric("⚡ AVG Power", f"{avg_power:.0f} W")
    c3.metric("📊 NP", f"{np_power:.0f} W")
    c4.metric("🔋 Praca", f"{work_kj:.0f} kJ")
    
    # Wiersz 2: HR
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("❤️ AVG HR", f"{avg_hr:.0f} bpm" if avg_hr else "--")
    c2.metric("❤️ MIN HR", f"{min_hr:.0f} bpm" if min_hr else "--")
    c3.metric("❤️ MAX HR", f"{max_hr:.0f} bpm" if max_hr else "--")
    c4.empty()
    
    # Wiersz 3: SmO2
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🩸 AVG SmO2", f"{avg_smo2:.1f}%" if avg_smo2 else "--")
    c2.metric("🩸 MIN SmO2", f"{min_smo2:.1f}%" if min_smo2 else "--")
    c3.metric("🩸 MAX SmO2", f"{max_smo2:.1f}%" if max_smo2 else "--")
    c4.empty()
    
    # Wiersz 4: VE
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🫁 AVG VE", f"{avg_ve:.1f} L/min" if avg_ve else "--")
    c2.metric("🫁 MIN VE", f"{min_ve:.1f} L/min" if min_ve else "--")
    c3.metric("🫁 MAX VE", f"{max_ve:.1f} L/min" if max_ve else "--")
    c4.empty()
    
    # Wiersz 5: BR
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💨 AVG BR", f"{avg_br:.0f} /min" if avg_br else "--")
    c2.metric("💨 MIN BR", f"{min_br:.0f} /min" if min_br else "--")
    c3.metric("💨 MAX BR", f"{max_br:.0f} /min" if max_br else "--")
    c4.empty()
    
    # Wiersz 6: Estymacje
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Est. VO2max", f"{est_vo2max:.1f} ml/kg/min" if est_vo2max else "--")
    c2.metric("🧬 Est. VLamax", f"{est_vlamax:.2f} mmol/L/s" if est_vlamax else "--")
    c3.metric("⚡ Est. CP", f"{est_cp:.0f} W" if est_cp else "--")
    c4.metric("🔋 Est. W'", f"{est_w_prime:.0f} J" if est_w_prime else "--")


def _calculate_np(watts_series):
    """Obliczenie Normalized Power."""
    if len(watts_series) < 30:
        return watts_series.mean()
    rolling_avg = watts_series.rolling(30, min_periods=1).mean()
    fourth_power = rolling_avg ** 4
    return fourth_power.mean() ** 0.25


def _estimate_cp_wprime(df_plot):
    """Estymacja CP i W' z danych MMP."""
    if 'watts' not in df_plot.columns or len(df_plot) < 1200:
        return 0, 0
    
    durations = [180, 300, 600, 900, 1200]
    valid_durations = [d for d in durations if d < len(df_plot)]
    
    if len(valid_durations) < 3:
        return 0, 0
    
    work_values = []
    for d in valid_durations:
        p = df_plot['watts'].rolling(window=d).mean().max()
        if not pd.isna(p):
            work_values.append(p * d)
        else:
            return 0, 0
    
    try:
        slope, intercept, _, _, _ = stats.linregress(valid_durations, work_values)
        return slope, intercept
    except Exception:
        return 0, 0


def _render_cp_model_chart(df_plot, cp_input, w_prime_input):
    """Renderowanie wykresu modelu CP."""
    if 'watts' not in df_plot.columns or len(df_plot) < 1200:
        st.info("Za mało danych (wymagane min. 20 minut) do wyświetlenia modelu CP.")
        return
    
    est_cp, est_w_prime = _estimate_cp_wprime(df_plot)
    
    if est_cp <= 0:
        st.info("Nie można wyestymować CP z danych.")
        return
    
    # Metryki estymowane
    c1, c2, c3 = st.columns(3)
    c1.metric("Est. CP", f"{est_cp:.0f} W", delta=f"{est_cp - cp_input:.0f} W vs ustawienia")
    c2.metric("Est. W'", f"{est_w_prime:.0f} J", delta=f"{est_w_prime - w_prime_input:.0f} J vs ustawienia")
    
    # R² dopasowania
    durations = [180, 300, 600, 900, 1200]
    valid_durations = [d for d in durations if d < len(df_plot)]
    work_values = []
    for d in valid_durations:
        p = df_plot['watts'].rolling(window=d).mean().max()
        if not pd.isna(p):
            work_values.append(p * d)
    
    if len(work_values) == len(valid_durations):
        _, _, r_value, _, _ = stats.linregress(valid_durations, work_values)
        c3.metric("R² (dopasowanie)", f"{r_value**2:.4f}")
    
    # Wykres
    x_theory = np.arange(60, 1800, 60)
    y_theory = [est_cp + (est_w_prime / t) for t in x_theory]
    
    y_actual = []
    x_actual = []
    for t in x_theory:
        if t < len(df_plot):
            val = df_plot['watts'].rolling(t).mean().max()
            y_actual.append(val)
            x_actual.append(t)
    
    fig_model = go.Figure()
    
    fig_model.add_trace(go.Scatter(
        x=np.array(x_actual)/60, y=y_actual,
        mode='markers', name='Twoje MMP',
        marker=dict(color='#00cc96', size=8)
    ))
    
    fig_model.add_trace(go.Scatter(
        x=x_theory/60, y=y_theory,
        mode='lines', name=f'Model CP ({est_cp:.0f}W)',
        line=dict(color='#ef553b', dash='dash')
    ))
    
    fig_model.update_layout(
        template="plotly_dark",
        title="Power Duration Curve vs Model CP",
        xaxis=dict(
            title="Czas [min]",
            tickformat='.0f'
        ),
        yaxis=dict(
            title="Moc [W]",
            tickformat='.0f'
        ),
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_model, use_container_width=True)


def _render_smo2_thb_chart(df_plot):
    """Renderowanie wykresu SmO2 vs THb w czasie."""
    if 'smo2' not in df_plot.columns:
        st.info("Brak danych SmO2 w tym pliku.")
        return
    
    fig_smo2_thb = make_subplots(specs=[[{"secondary_y": True}]])
    
    time_x = df_plot['time'] if 'time' in df_plot.columns else range(len(df_plot))
    
    # SmO2
    smo2_smooth = df_plot['smo2'].rolling(5, center=True).mean() if 'smo2' in df_plot.columns else None
    if smo2_smooth is not None:
        fig_smo2_thb.add_trace(go.Scatter(
            x=time_x, y=smo2_smooth,
            name='SmO2 (%)', line=dict(color='#2ca02c', width=2),
            hovertemplate="SmO2: %{y:.1f}%<extra></extra>"
        ), secondary_y=False)
    
    # THb
    if 'thb' in df_plot.columns:
        thb_smooth = df_plot['thb'].rolling(5, center=True).mean()
        fig_smo2_thb.add_trace(go.Scatter(
            x=time_x, y=thb_smooth,
            name='THb (g/dL)', line=dict(color='#9467bd', width=2),
            hovertemplate="THb: %{y:.2f} g/dL<extra></extra>"
        ), secondary_y=True)
    else:
        st.caption("ℹ️ Brak danych THb w pliku.")
    
    fig_smo2_thb.update_layout(
        template="plotly_dark", height=350,
        legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    fig_smo2_thb.update_yaxes(title_text="SmO2 (%)", secondary_y=False)
    fig_smo2_thb.update_yaxes(title_text="THb (g/dL)", secondary_y=True)
    st.plotly_chart(fig_smo2_thb, use_container_width=True)


def _render_vent_thresholds_summary(df_plot, cp_input, vt1_watts, vt2_watts, threshold_result):
    """Renderowanie wykresu progów wentylacyjnych VT1/VT2."""
    if 'tymeventilation' not in df_plot.columns:
        st.info("Brak danych wentylacji do analizy progów VT.")
        return
    
    # Wygładzanie
    df_plot['ve_smooth'] = df_plot['tymeventilation'].rolling(window=10, center=True).mean()
    if 'watts_smooth_5s' not in df_plot.columns and 'watts' in df_plot.columns:
        df_plot['watts_smooth_5s'] = df_plot['watts'].rolling(window=5, center=True).mean()
    
    # Użyj wykrytych lub przekazanych wartości (już obsłużone w render_summary_tab, ale zachowujemy spójność z metrykami)
    vt1_w = vt1_watts
    vt2_w = vt2_watts
    
    # Pobierz HR i VE dla wyświetlenia, jeśli dostępne w threshold_result
    # Jeśli vt1_watts przekazane jako param (manual), HR i VE mogą być nieznane bez ponownej detekcji,
    # ale threshold_result zawiera dane z auto-detekcji.
    vt1_hr = threshold_result.vt1_hr if threshold_result.vt1_watts == vt1_w else 0
    vt2_hr = threshold_result.vt2_hr if threshold_result.vt2_watts == vt2_w else 0
    vt1_ve = threshold_result.vt1_ve if threshold_result.vt1_watts == vt1_w else 0
    vt2_ve = threshold_result.vt2_ve if threshold_result.vt2_watts == vt2_w else 0
    
    # Wykres
    fig_vent = go.Figure()
    
    time_x = df_plot['time'] if 'time' in df_plot.columns else range(len(df_plot))
    
    fig_vent.add_trace(go.Scatter(
        x=time_x, y=df_plot['ve_smooth'],
        mode='lines', name='VE (L/min)',
        line=dict(color='#ffa15a', width=2)
    ))
    
    if 'watts_smooth_5s' in df_plot.columns:
        fig_vent.add_trace(go.Scatter(
            x=time_x, y=df_plot['watts_smooth_5s'],
            mode='lines', name='Power',
            line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3
        ))
    
    # Markery VT1/VT2
    if vt1_w and threshold_result.step_ve_analysis:
        for step in threshold_result.step_ve_analysis:
            if step.get('is_vt1'):
                marker_time = step.get('end_time', 0)
                fig_vent.add_vline(x=marker_time, line=dict(color="#ffa15a", width=3, dash="dash"))
    
    if vt2_w and threshold_result.step_ve_analysis:
        for step in threshold_result.step_ve_analysis:
            if step.get('is_vt2'):
                marker_time = step.get('end_time', 0)
                fig_vent.add_vline(x=marker_time, line=dict(color="#ef553b", width=3, dash="dash"))
    
    fig_vent.update_layout(
        template="plotly_dark", height=350,
        yaxis=dict(title="VE (L/min)"),
        yaxis2=dict(title="Moc (W)", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_vent, use_container_width=True)
    
    # Panel VT1/VT2
    col_z1, col_z2 = st.columns(2)
    
    with col_z1:
        if vt1_w:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt1_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt1_hr)} bpm</p>' if vt1_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt1_ve:.1f} L/min</p>' if vt1_ve else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(vt1_w/cp_input)*100:.0f}% CP")
        else:
            st.info("VT1: Nie wykryto")
    
    with col_z2:
        if vt2_w:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt2_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt2_hr)} bpm</p>' if vt2_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt2_ve:.1f} L/min</p>' if vt2_ve else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(vt2_w/cp_input)*100:.0f}% CP")
        else:
            st.info("VT2: Nie wykryto")


def _render_smo2_thresholds_summary(df_plot, cp_input, lt1_watts, lt2_watts, threshold_result):
    """Renderowanie wykresu progów SmO2 LT1/LT2."""
    if 'smo2' not in df_plot.columns:
        st.info("Brak danych SmO2 do analizy progów LT.")
        return
    
    # Wygładzanie
    df_plot['smo2_smooth'] = df_plot['smo2'].rolling(window=10, center=True).mean()
    if 'watts_smooth_5s' not in df_plot.columns and 'watts' in df_plot.columns:
        df_plot['watts_smooth_5s'] = df_plot['watts'].rolling(window=5, center=True).mean()
    
    # Użyj wykrytych lub przekazanych wartości
    lt1_w = lt1_watts
    lt2_w = lt2_watts
    lt1_hr = threshold_result.smo2_1_hr if threshold_result.smo2_1_watts == lt1_w else 0
    lt2_hr = threshold_result.smo2_2_hr if threshold_result.smo2_2_watts == lt2_w else 0
    lt1_smo2 = threshold_result.smo2_1_value if threshold_result.smo2_1_watts == lt1_w else 0
    lt2_smo2 = threshold_result.smo2_2_value if threshold_result.smo2_2_watts == lt2_w else 0
    
    # Wykres
    fig_smo2 = go.Figure()
    
    time_x = df_plot['time'] if 'time' in df_plot.columns else range(len(df_plot))
    
    fig_smo2.add_trace(go.Scatter(
        x=time_x, y=df_plot['smo2_smooth'],
        mode='lines', name='SmO2 (%)',
        line=dict(color='#2ca02c', width=2)
    ))
    
    if 'watts_smooth_5s' in df_plot.columns:
        fig_smo2.add_trace(go.Scatter(
            x=time_x, y=df_plot['watts_smooth_5s'],
            mode='lines', name='Power',
            line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3
        ))
    
    # Znajdź czas dla LT1/LT2 (na podstawie mocy)
    def find_time_for_power(power):
        if power <= 0:
            return None
        if 'watts_smooth_5s' in df_plot.columns:
            idx = (df_plot['watts_smooth_5s'] - power).abs().idxmin()
            return df_plot.loc[idx, 'time'] if 'time' in df_plot.columns else idx
        elif 'watts' in df_plot.columns:
            idx = (df_plot['watts'] - power).abs().idxmin()
            return df_plot.loc[idx, 'time'] if 'time' in df_plot.columns else idx
        return None
    
    lt1_time = find_time_for_power(lt1_w) if lt1_w else None
    lt2_time = find_time_for_power(lt2_w) if lt2_w else None
    
    if lt1_time is not None:
        fig_smo2.add_vline(x=lt1_time, line=dict(color="#2ca02c", width=3, dash="dash"))
    
    if lt2_time is not None:
        fig_smo2.add_vline(x=lt2_time, line=dict(color="#d62728", width=3, dash="dash"))
    
    fig_smo2.update_layout(
        template="plotly_dark", height=350,
        yaxis=dict(title="SmO2 (%)"),
        yaxis2=dict(title="Moc (W)", overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", y=1.05, x=0), hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_smo2, use_container_width=True)
    
    # Panel LT1/LT2
    col_z1, col_z2 = st.columns(2)
    
    with col_z1:
        if lt1_w:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #2ca02c; background-color: #222;">
                <h3 style="margin:0; color: #2ca02c;">LT1 (SteadyState)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(lt1_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(lt1_hr)} bpm</p>' if lt1_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {lt1_smo2:.1f}%</p>' if lt1_smo2 else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(lt1_w/cp_input)*100:.0f}% CP")
        else:
            st.info("LT1: Nie wykryto")
    
    with col_z2:
        if lt2_w:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #d62728; background-color: #222;">
                <h3 style="margin:0; color: #d62728;">LT2 (Próg)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(lt2_w)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(lt2_hr)} bpm</p>' if lt2_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {lt2_smo2:.1f}%</p>' if lt2_smo2 else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(lt2_w/cp_input)*100:.0f}% CP")
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
    
    # Walidacja danych
    if not vt1_watts or vt1_watts <= 0:
        st.warning("⚠️ **Brak danych VT1 (wentylacyjny)** — nie można obliczyć TDI.")
        return
    
    if not lt1_watts or lt1_watts <= 0:
        st.warning("⚠️ **Brak danych LT1 (SmO2)** — nie można obliczyć TDI.")
        return
    
    # Obliczenie TDI
    tdi = abs(vt1_watts - lt1_watts) / vt1_watts * 100
    delta = lt1_watts - vt1_watts  # Dodatnia = SmO2 wyżej niż VE
    
    # Klasyfikacja
    if tdi < 5:
        classification = "ZGODNY"
        color = "#00cc96"  # Green
        alert_type = "success"
        interpretation = "System tlenowy i obwodowy są zsynchronizowane. Optymalna koordynacja między wentylacją a perfuzją mięśniową."
        recommendation = "✅ **Trening:** Możesz trenować w pełnym zakresie intensywności. System transportu tlenu działa harmonijnie."
    elif tdi <= 10:
        classification = "HETEROGENICZNY"
        color = "#ffa15a"  # Orange
        alert_type = "warning"
        interpretation = "Wykryto niewielką rozbieżność między progiem wentylacyjnym a progiem SmO2. Może wskazywać na różne tempo adaptacji systemów centralnego i obwodowego."
        if delta > 0:
            recommendation = "⚡ **Trening:** Skup się na treningach tempo (Sweet Spot) aby wyrównać adaptację obwodową. SmO2 wskazuje wyższy próg niż VE — mięśnie adaptują się szybciej niż układ oddechowy."
        else:
            recommendation = "🫁 **Trening:** Zwiększ udział treningów Z2 i długich wyjazdów. VE wskazuje wyższy próg niż SmO2 — układ oddechowy wyprzedza adaptację mięśniową."
    else:
        classification = "KONFLIKT"
        color = "#ef553b"  # Red
        alert_type = "error"
        interpretation = "Znacząca rozbieżność między systemem centralnym (wentylacja) a obwodowym (perfuzja mięśniowa). Możliwe przyczyny: okluzja naczyniowa, zaburzenia mikrokrążenia, lub błąd pomiaru sensora NIRS."
        if delta > 0:
            recommendation = "🔴 **Uwaga:** SmO2 znacząco wyżej niż VE. Sprawdź: (1) pozycję sensora NIRS, (2) grubość tkanki tłuszczowej, (3) okluzję podczas pedałowania. Rozważ konsultację z fizjologiem."
        else:
            recommendation = "🔴 **Uwaga:** VE znacząco wyżej niż SmO2. Sprawdź: (1) kalibrację sensora wentylacyjnego, (2) możliwą hiperperfuzję centralną, (3) ograniczenia mikrokrążenia obwodowego."
    
    # Wyświetlanie
    st.markdown(f"""
    <div style="padding:20px; border-radius:12px; border:3px solid {color}; background-color: #1a1a1a; text-align:center;">
        <h2 style="margin:0; color: {color};">TDI: {tdi:.1f}%</h2>
        <p style="margin:5px 0; font-size:1.2em; color: {color}; font-weight:bold;">{classification}</p>
        <p style="margin:10px 0 0 0; color:#888; font-size:0.85em;">
            VT1 (VE): <b>{vt1_watts:.0f} W</b> | LT1 (SmO2): <b>{lt1_watts:.0f} W</b> | Δ = {delta:+.0f} W
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alert
    if alert_type == "success":
        st.success(f"✅ **Interpretacja:** {interpretation}")
    elif alert_type == "warning":
        st.warning(f"⚠️ **Interpretacja:** {interpretation}")
    else:
        st.error(f"🔴 **Interpretacja:** {interpretation}")
    
    # Rekomendacja treningowa
    st.info(recommendation)
    
    # Teoria
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


def _render_vo2max_uncertainty(df_plot: pd.DataFrame, rider_weight: float):
    """
    Estymacja VO2max z przedziałem ufności 95% (CI95%).
    
    Wzór ACSM: VO2max = (10.8 * P_5min / kg) + 7
    
    CI95% oparta na:
    - Zmienności mocy w ostatnich 5 minutach rampy (SD)
    - Stabilności odpowiedzi HR (CV)
    """
    
    # Walidacja danych
    if 'watts' not in df_plot.columns:
        st.warning("⚠️ **Brak danych mocy** — nie można estymować VO2max.")
        return
    
    if rider_weight <= 0:
        st.warning("⚠️ **Nieprawidłowa waga zawodnika** — nie można estymować VO2max.")
        return
    
    # Pobierz ostatnie 5 minut danych (300 sekund)
    last_5min_samples = min(300, len(df_plot))
    df_last5 = df_plot.tail(last_5min_samples)
    
    if len(df_last5) < 60:
        st.warning("⚠️ **Za mało danych** (wymagane min. 1 minuta) — nie można estymować VO2max.")
        return
    
    # Obliczenia mocy
    power_mean = df_last5['watts'].mean()
    power_sd = df_last5['watts'].std()
    power_cv = (power_sd / power_mean * 100) if power_mean > 0 else 0
    n = len(df_last5)
    
    # Estymacja VO2max (ACSM)
    vo2max = (10.8 * power_mean / rider_weight) + 7
    
    # Obliczenie SE i CI95% dla VO2max
    # Propagacja błędu: SE_vo2 = 10.8 / kg * SE_power
    se_power = power_sd / np.sqrt(n)
    se_vo2 = 10.8 * se_power / rider_weight
    ci95_vo2 = 1.96 * se_vo2
    
    # Dodatkowa niepewność z HR response (jeśli dostępne)
    hr_penalty = 0
    hr_col = None
    for alias in ['hr', 'heartrate', 'heart_rate', 'bpm']:
        if alias in df_last5.columns:
            hr_col = alias
            break
    
    if hr_col:
        hr_mean = df_last5[hr_col].mean()
        hr_sd = df_last5[hr_col].std()
        hr_cv = (hr_sd / hr_mean * 100) if hr_mean > 0 else 0
        # Wysoki CV HR = większa niepewność
        if hr_cv > 5:
            hr_penalty = ci95_vo2 * 0.2  # +20% CI za niestabilne HR
    
    ci95_total = ci95_vo2 + hr_penalty
    
    # Confidence Weight: im mniejszy CI względem VO2max, tym wyższa waga
    confidence_weight = 1 / (1 + ci95_total / vo2max) if vo2max > 0 else 0
    confidence_pct = confidence_weight * 100
    
    # Klasyfikacja pewności
    if confidence_pct >= 80:
        conf_color = "#00cc96"
        conf_label = "WYSOKA"
    elif confidence_pct >= 60:
        conf_color = "#ffa15a"
        conf_label = "UMIARKOWANA"
    else:
        conf_color = "#ef553b"
        conf_label = "NISKA"
    
    # Wyświetlanie głównego wyniku
    st.markdown(f"""
    <div style="padding:20px; border-radius:12px; border:3px solid #17a2b8; background-color: #1a1a1a; text-align:center;">
        <h2 style="margin:0; color: #17a2b8;">VO₂max = {vo2max:.1f} ± {ci95_total:.1f} ml/kg/min</h2>
        <p style="margin:10px 0 0 0; color:#888; font-size:0.85em;">
            (CI95%: {vo2max - ci95_total:.1f} – {vo2max + ci95_total:.1f} ml/kg/min)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Źródło disclaimer
    st.caption("📌 **Źródło:** Estymacja modelowa (ACSM power-time), nie pomiar bezpośredni. Używać orientacyjnie.")
    
    # Confidence Weight
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="padding:15px; border-radius:8px; border:2px solid {conf_color}; background-color: #222; text-align:center;">
            <p style="margin:0; color:#aaa; font-size:0.9em;">Waga Pewności (Confidence Weight)</p>
            <h3 style="margin:5px 0; color: {conf_color};">{confidence_pct:.0f}% — {conf_label}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Szczegóły obliczeń
    with st.expander("📊 Szczegóły obliczeń", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Średnia moc (5 min)", f"{power_mean:.0f} W")
        c2.metric("SD mocy", f"{power_sd:.1f} W")
        c3.metric("CV mocy", f"{power_cv:.1f}%")
        
        if hr_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Średnie HR", f"{hr_mean:.0f} bpm")
            c2.metric("SD HR", f"{hr_sd:.1f} bpm")
            c3.metric("CV HR", f"{hr_cv:.1f}%")
        
        st.markdown(f"""
        | Parametr | Wartość |
        |----------|---------|
        | SE mocy | {se_power:.2f} W |
        | SE VO₂max | {se_vo2:.2f} ml/kg/min |
        | CI95% (moc) | ±{ci95_vo2:.2f} ml/kg/min |
        | Korekta HR | +{hr_penalty:.2f} ml/kg/min |
        | **CI95% całkowity** | **±{ci95_total:.2f} ml/kg/min** |
        """)
    
    # Teoria
    with st.expander("📖 Metodologia estymacji VO2max", expanded=False):
        st.markdown("""
        ### Formuła ACSM
        
        ```
        VO₂max = (10.8 × P / kg) + 7
        ```
        
        Gdzie:
        - `P` = średnia moc w ostatnich 5 minutach rampy [W]
        - `kg` = masa ciała zawodnika [kg]
        
        ---
        
        ### Przedział ufności (CI95%)
        
        CI95% jest obliczany na podstawie:
        
        1. **Zmienność mocy (SD):**
           - Wysoka zmienność = większa niepewność estymacji
           - SE = SD / √n
           - CI = 1.96 × SE × 10.8 / kg
        
        2. **Stabilność HR:**
           - CV HR > 5% → dodatkowa korekta +20% CI
           - Niestabilne HR może wskazywać na nieustalony stan metaboliczny
        
        ---
        
        ### Waga Pewności (Confidence Weight)
        
        ```
        Weight = 1 / (1 + CI/VO₂max)
        ```
        
        Używana do skalowania pewności wniosków centralnych:
        - **≥80%** = Wysoka pewność, wyniki wiarygodne
        - **60-80%** = Umiarkowana pewność, interpretować ostrożnie
        - **<60%** = Niska pewność, traktować orientacyjnie
        
        ---
        
        *Uwaga: Jest to estymacja modelowa, nie zastępuje bezpośredniego pomiaru VO₂max w laboratorium.*
        """)
