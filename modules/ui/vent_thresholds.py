import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.quality import check_step_test_protocol

def render_vent_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input):
    """Detekcja progów wentylacyjnych VT1/VT2 - wymaga Ramp Test."""
    st.header("🎯 Detekcja Progów Wentylacyjnych (VT1 / VT2)")
    st.markdown("Automatyczna detekcja progów wentylacyjnych. **Wymaga testu stopniowanego (Ramp Test).**")

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    if 'time' not in target_df.columns or 'tymeventilation' not in target_df.columns:
        st.error("Brak danych wentylacji (tymeventilation) lub czasu!")
        st.stop()

    # Wygładzanie
    # Normalize columns first
    target_df.columns = target_df.columns.str.lower().str.strip()
    
    # Handle HR aliases
    if 'hr' not in target_df.columns:
        for alias in ['heart_rate', 'heart rate', 'bpm', 'tętno', 'heartrate', 'heart_rate_bpm']:
            if alias in target_df.columns:
                target_df.rename(columns={alias: 'hr'}, inplace=True)
                break

    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    target_df['ve_smooth'] = target_df['tymeventilation'].rolling(window=10, center=True).mean()
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # --- Quality Check: Protocol Compliance ---
    st.subheader("📋 Weryfikacja Protokołu")
    
    proto_check = check_step_test_protocol(target_df)
    
    if not proto_check['is_valid']:
        st.error("⚠️ **Wykryto Problemy z Protokołem Testu**")
        for issue in proto_check['issues']:
            st.warning(issue)
        st.markdown("""
        **Dlaczego to ważne?**
        
        Detekcja progów wentylacyjnych wymaga **testu stopniowanego (Ramp Test)** z liniowym wzrostem obciążenia.
        Dla normalnych treningów użyj zakładki **"🫁 Ventilation"** do analizy manualnej.
        """)
        
        if not st.checkbox("⚠️ Wymuś analizę mimo błędów protokołu (wyniki mogą być niewiarygodne)"):
            st.stop()
    else:
        st.success("✅ Protokół Testu Stopniowanego: Poprawny (Liniowy Wzrost Obciążenia)")

    st.markdown("---")

    # 2. DETEKCJA AUTOMATYCZNA
    with st.spinner("Analizowanie progów wentylacyjnych..."):
        result = analyze_step_test(
            target_df, 
            power_column='watts',
            ve_column='tymeventilation',
            hr_column='hr' if 'hr' in target_df.columns else None,
            time_column='time'
        )
    
    vt1_zone = result.vt1_zone
    vt2_zone = result.vt2_zone
    hysteresis = result.hysteresis
    sensitivity = result.sensitivity
    
    # Display step detection info
    if result.analysis_notes:
        with st.expander("📋 Szczegóły Analizy", expanded=True):
            for note in result.analysis_notes:
                if note.startswith("✅"):
                    st.success(note)
                elif note.startswith("⚠️"):
                    st.warning(note)
                elif note.startswith("  •"):
                    st.caption(note)
                else:
                    st.info(note)
            
            if result.steps_analyzed > 0:
                st.metric("Liczba wykrytych stopni", result.steps_analyzed)
            
            # Show VE slope table for each step
            if result.step_ve_analysis:
                st.markdown("### 📊 VE Slope per step (debug)")
                step_df = pd.DataFrame(result.step_ve_analysis)
                
                # Format columns
                if 've_slope' in step_df.columns:
                    step_df['ve_slope'] = step_df['ve_slope'].apply(lambda x: f"{x:.4f}")
                # Safe formatting for numeric columns
                if 'avg_power' in step_df.columns:
                    step_df['avg_power'] = pd.to_numeric(step_df['avg_power'], errors='coerce').apply(lambda x: f"{x:.0f}W" if pd.notna(x) else "-")
                if 'avg_hr' in step_df.columns:
                    step_df['avg_hr'] = pd.to_numeric(step_df['avg_hr'], errors='coerce').apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
                if 'avg_ve' in step_df.columns:
                    step_df['avg_ve'] = pd.to_numeric(step_df['avg_ve'], errors='coerce').apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                if 'avg_br' in step_df.columns:
                    step_df['avg_br'] = pd.to_numeric(step_df['avg_br'], errors='coerce').apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
                if 've_slope' in step_df.columns:
                    step_df['ve_slope'] = pd.to_numeric(step_df['ve_slope'], errors='coerce').apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                
                # Add VT markers
                if 'is_vt1' in step_df.columns:
                    step_df['VT'] = step_df.apply(
                        lambda r: '🟠 VT1' if r.get('is_vt1') else ('🔴 VT2' if r.get('is_vt2') else ''), 
                        axis=1
                    )
                
                # Rename columns for display
                step_df = step_df.rename(columns={
                    'step_number': 'Step',
                    'avg_power': 'Power',
                    'avg_hr': 'HR',
                    'avg_ve': 'VE',
                    'avg_br': 'BR',
                    've_slope': 'Trend Slope'
                })
                
                # Select columns to show
                cols_to_show = ['Step', 'Power', 'HR', 'VE', 'BR', 'Trend Slope', 'VT']
                cols_available = [c for c in cols_to_show if c in step_df.columns]
                
                st.dataframe(step_df[cols_available], hide_index=True)

    # Wyświetlenie wyników automatycznych
    st.subheader("🤖 Wykryte Progi Wentylacyjne")
    
    col_z1, col_z2 = st.columns(2)
    
    # --- VT1 CARD ---
    with col_z1:
        if result.vt1_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(result.vt1_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(result.vt1_hr)} bpm</p>' if result.vt1_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {result.vt1_ve} L/min</p>' if result.vt1_ve else ''}
                {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(result.vt1_br)} BR/min</p>' if result.vt1_br else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt1_pct = (result.vt1_watts / cp_input) * 100
                st.caption(f"~{vt1_pct:.0f}% CP")
        else:
            st.info("VT1: Nie wykryto (brak slope >= 0.05)")

    # --- VT2 CARD ---
    with col_z2:
        if result.vt2_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(result.vt2_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(result.vt2_hr)} bpm</p>' if result.vt2_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {result.vt2_ve} L/min</p>' if result.vt2_ve else ''}
                 {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(result.vt2_br)} BR/min</p>' if result.vt2_br else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt2_pct = (result.vt2_watts / cp_input) * 100
                st.caption(f"~{vt2_pct:.0f}% CP")
        else:
            st.info("VT2: Nie wykryto (brak slope >= 0.05 po VT1)")

    st.markdown("---")

    # Hysteresis Information
    if hysteresis and (hysteresis.vt1_dec_zone or hysteresis.vt2_dec_zone):
        with st.expander("📉 Analiza Histerezy (Ramp Down vs Up)", expanded=False):
            st.markdown("""
            **Histereza** to różnica między progami wykrytymi podczas wzrostu obciążenia (Ramp Up) 
            a progami wykrytymi podczas spadku (Ramp Down / Recovery).
            
            - **Duża histereza (>15W)**: Może wskazywać na zmęczenie lub niedobór tlenowy
            - **Mała histereza (<10W)**: Dobra reprodukowalność progów
            """)
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                if hysteresis.vt1_dec_zone:
                    st.markdown(f"**VT1 (Down):** {hysteresis.vt1_dec_zone.range_watts[0]:.0f} - {hysteresis.vt1_dec_zone.range_watts[1]:.0f} W")
                    if hysteresis.vt1_shift_watts is not None:
                         shift = hysteresis.vt1_shift_watts
                         color = "red" if shift < -15 else "green"
                         st.markdown(f"Shift: **:{color}[{shift:+.0f} W]**")
            with h_col2:
                 if hysteresis.vt2_dec_zone:
                    st.markdown(f"**VT2 (Down):** {hysteresis.vt2_dec_zone.range_watts[0]:.0f} - {hysteresis.vt2_dec_zone.range_watts[1]:.0f} W")
                    if hysteresis.vt2_shift_watts is not None:
                         shift = hysteresis.vt2_shift_watts
                         color = "red" if shift < -15 else "green"
                         st.markdown(f"Shift: **:{color}[{shift:+.0f} W]**")
            
            if hysteresis.warnings:
                for w in hysteresis.warnings:
                    st.warning(f"⚠️ {w}")

    # 3. Wykres z zaznaczonymi strefami
    st.subheader("📈 Wizualizacja Progów")
    
    fig_thresh = go.Figure()

    # VE (Primary)
    fig_thresh.add_trace(go.Scatter(
        x=target_df['time'], y=target_df['ve_smooth'],
        customdata=target_df['time_str'],
        mode='lines', name='VE (L/min)',
        line=dict(color='#ffa15a', width=2),
        hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>"
    ))

    # Power (Secondary)
    if 'watts_smooth_5s' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['watts_smooth_5s'],
            customdata=target_df['time_str'],
            mode='lines', name='Power',
            line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3,
            hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
        ))

    # HR Trace (Red, Dotted, Secondary Axis)
    if 'hr' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['hr'],
            customdata=target_df['time_str'],
            mode='lines', name='Heart Rate',
            line=dict(color='#d62728', width=1, dash='dot'),
            yaxis='y2', opacity=0.5,
            hovertemplate="<b>Czas:</b> %{customdata}<br><b>HR:</b> %{y:.0f} bpm<extra></extra>"
        ))

    # VT Markers based on identified steps
    if result.step_ve_analysis:
        for step in result.step_ve_analysis:
            # Check if this step is VT1 or VT2
            is_vt1 = step.get('is_vt1', False)
            is_vt2 = step.get('is_vt2', False)
            
            if is_vt1 or is_vt2:
                # Get step info
                power = step.get('avg_power', 0)
                if power is None: power = 0
                hr = step.get('avg_hr')
                end_time = step.get('end_time', 0)  # Use end of step for marker
                
                label = "VT1" if is_vt1 else "VT2"
                color = "#ffa15a" if is_vt1 else "#ef553b"
                
                hr_str = f"{int(hr)}" if hr is not None else "--"
                
                # Add Vertical Line Marker at End of Step
                fig_thresh.add_vline(
                    x=end_time,
                    line=dict(color=color, width=2, dash="dash"),
                    annotation_text=f"<b>{label}</b><br>{int(power)}W @ {hr_str} bpm",
                    annotation_position="top left",
                    annotation_font=dict(color=color, size=12),
                    layer="above"
                )
    
    # Hysteresis Zones (Dashed, if available from legacy detection)
    if hysteresis:
        if hysteresis.vt1_dec_zone:
            fig_thresh.add_hrect(
               y0=hysteresis.vt1_dec_zone.range_watts[0], y1=hysteresis.vt1_dec_zone.range_watts[1],
               fillcolor="blue", opacity=0.05,
               layer="below", line_width=1, line_dash="dot", line_color="blue",
               yref="y2",
               annotation_text="VT1 (Down)", annotation_position="bottom right"
            )

        if hysteresis.vt2_dec_zone:
            fig_thresh.add_hrect(
               y0=hysteresis.vt2_dec_zone.range_watts[0], y1=hysteresis.vt2_dec_zone.range_watts[1],
               fillcolor="purple", opacity=0.05,
               layer="below", line_width=1, line_dash="dot", line_color="purple",
               yref="y2",
               annotation_text="VT2 (Down)", annotation_position="bottom right"
            )

    fig_thresh.update_layout(
        title="Dynamika Wentylacji z Progami VT1/VT2",
        xaxis_title="Czas",
        yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
        yaxis2=dict(title=dict(text="Moc (W)", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_thresh, use_container_width=True)

    # Sensitivity Analysis
    if sensitivity:
        with st.expander("📊 Analiza Wrażliwości (Sensitivity)", expanded=False):
            st.markdown("""
            **Sensitivity Analysis** sprawdza, jak stabilne są wykryte progi przy różnych parametrach algorytmu.
            Wysoka stabilność = wynik jest wiarygodny niezależnie od ustawień.
            """)
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.metric("VT1 Stability Score", f"{sensitivity.vt1_stability_score:.2f}", 
                          help="0-1, wyżej = lepiej")
                st.caption(f"Variability: ±{sensitivity.vt1_variability_watts:.1f} W")
            with s_col2:
                st.metric("VT2 Stability Score", f"{sensitivity.vt2_stability_score:.2f}",
                          help="0-1, wyżej = lepiej")
                st.caption(f"Variability: ±{sensitivity.vt2_variability_watts:.1f} W")
                
            if sensitivity.details:
                for w in sensitivity.details:
                    st.caption(f"ℹ️ {w}")

    # ===== TEORIA =====
    with st.expander("🫁 TEORIA: Progi Wentylacyjne (VT1 / VT2)", expanded=False):
        st.markdown("""
        ## Co to są progi wentylacyjne?
        
        **Progi wentylacyjne** to punkty, w których wentylacja (VE) zaczyna rosnąć nieliniowo względem mocy.
        
        | Próg | Inna nazwa | Fizjologia | % VO2max |
        |------|-----------|------------|----------|
        | **VT1** | Próg tlenowy, LT1 | Początek akumulacji mleczanu | ~50-60% |
        | **VT2** | Próg beztlenowy, LT2, OBLA | Maksymalny laktat steady-state | ~75-85% |
        
        ---
        
        ## Jak działa detekcja?
        
        System stosuje:
        1. **Sliding Window Analysis**: Skanuje okno po oknie, żeby znaleźć przejścia w nachyleniu (slope)
        2. **Breakpoint Detection**: Szuka punktów załamania krzywej VE vs Power
        3. **Sensitivity Analysis**: Uruchamia algorytm kilkukrotnie z różnymi parametrami
        
        ---
        
        ## Zastosowanie progów
        
        | Strefa | Zakres | Cel treningowy |
        |--------|--------|----------------|
        | **Z1 (Recovery)** | < VT1 | Regeneracja, rozgrzewka |
        | **Z2 (Endurance)** | VT1 - środek | Baza tlenowa |
        | **Z3 (Tempo)** | środek - VT2 | Sweet Spot |
        | **Z4 (Threshold)** | VT2 ± 5% | FTP, próg |
        | **Z5+ (VO2max)** | > VT2 | Interwały, moc szczytowa |
        
        ---
        
        ## Reliability Score (Niezawodność)
        
        * **HIGH**: Wynik jest stabilny niezależnie od wygładzania
        * **MEDIUM**: Wynik zależy nieco od parametrów
        * **LOW**: Duża zmienność (>15W różnicy) - sugeruje "szumiący" sygnał
        
        ---
        
        ## Wymagania testu
        
        ⚠️ **Dla wiarygodnych wyników potrzebujesz:**
        - Test stopniowany (Ramp Test) z liniowym wzrostem mocy
        - Minimum 10-15 minut narastającego obciążenia
        - Czysty sygnał wentylacji (stabilny sensor)
        - Brak przerw i wahań mocy
        """)
