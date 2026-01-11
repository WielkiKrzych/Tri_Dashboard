import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.ventilatory import detect_vt_vslope_savgol
from modules.calculations.quality import check_step_test_protocol

def render_manual_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input, max_hr_input):
    """Ręczna edycja progów VT1/VT2 i wizualizacja na wykresie wentylacji."""
    st.header("🛠️ Manualna Edycja Progów (VT1 / VT2)")
    st.markdown("Wprowadź własne wartości mocy dla progów VT1 i VT2, aby zobaczyć je na wykresie.")

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

    # 2. EDYCJA MANUALNA
    st.subheader("✍️ Parametry Manualne")
    
    # Próba pobrania domyślnych wartości z automatycznej detekcji
    with st.spinner("Analizowanie progów dla sugestii..."):
        result = analyze_step_test(
            target_df, 
            power_column='watts',
            ve_column='tymeventilation',
            hr_column='hr' if 'hr' in target_df.columns else None,
            time_column='time'
        )

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        manual_vt1 = st.number_input("Manualny VT1 (Moc W)", min_value=0, max_value=1000, value=int(result.vt1_watts) if result.vt1_watts else 0, step=5, key="manual_vt1_watts")
    with col_inp2:
        manual_vt2 = st.number_input("Manualny VT2 (Moc W)", min_value=0, max_value=1000, value=int(result.vt2_watts) if result.vt2_watts else 0, step=5, key="manual_vt2_watts")

    hysteresis = result.hysteresis
    sensitivity = result.sensitivity

    # Obliczanie HR dla podanych mocy manualnych
    def find_hr_for_power(power):
        if power <= 0: return None
        # Znajdź najbliższy punkt w wygładzonej mocy
        if 'watts_smooth_5s' in target_df.columns:
            idx = (target_df['watts_smooth_5s'] - power).abs().idxmin()
            return target_df.loc[idx, 'hr'] if 'hr' in target_df.columns else None
        elif 'watts' in target_df.columns:
            idx = (target_df['watts'] - power).abs().idxmin()
            return target_df.loc[idx, 'hr'] if 'hr' in target_df.columns else None
        return None

    def find_time_for_power(power):
        if power <= 0: return None
        if 'watts_smooth_5s' in target_df.columns:
            idx = (target_df['watts_smooth_5s'] - power).abs().idxmin()
            return target_df.loc[idx, 'time']
        elif 'watts' in target_df.columns:
            idx = (target_df['watts'] - power).abs().idxmin()
            return target_df.loc[idx, 'time']
        return None

    vt1_hr_est = find_hr_for_power(manual_vt1)
    vt2_hr_est = find_hr_for_power(manual_vt2)
    vt1_time_manual = find_time_for_power(manual_vt1)
    vt2_time_manual = find_time_for_power(manual_vt2)

    # Now add secondary manual inputs
    st.markdown("---")
    col_inpa, col_inpb = st.columns(2)
    
    with col_inpa:
        st.caption("Dodatkowe parametry VT1")
        manual_vt1_hr = st.number_input("VT1 HR (bpm)", min_value=0, max_value=250, value=int(vt1_hr_est) if vt1_hr_est else 0, step=1, key="vt1_hr")
        manual_vt1_ve = st.number_input("VT1 VE (L/min)", min_value=0.0, max_value=300.0, value=float(result.vt1_ve) if result.vt1_ve else 0.0, step=1.0, key="vt1_ve")
        manual_vt1_br = st.number_input("VT1 Oddechy (BR/min)", min_value=0, max_value=120, value=int(result.vt1_br) if result.vt1_br else 0, step=1, key="vt1_br")

    with col_inpb:
        st.caption("Dodatkowe parametry VT2")
        manual_vt2_hr = st.number_input("VT2 HR (bpm)", min_value=0, max_value=250, value=int(vt2_hr_est) if vt2_hr_est else 0, step=1, key="vt2_hr")
        manual_vt2_ve = st.number_input("VT2 VE (L/min)", min_value=0.0, max_value=300.0, value=float(result.vt2_ve) if result.vt2_ve else 0.0, step=1.0, key="vt2_ve")
        manual_vt2_br = st.number_input("VT2 Oddechy (BR/min)", min_value=0, max_value=120, value=int(result.vt2_br) if result.vt2_br else 0, step=1, key="vt2_br")

    # VE Breakpoint - manual input for PDF report
    st.markdown("---")
    st.caption("📈 VE Breakpoint dla raportu PDF (punkt załamania wentylacji):")
    ve_breakpoint_manual = st.number_input(
        "VE Breakpoint (W)", 
        min_value=0, 
        max_value=600, 
        value=0, 
        step=5, 
        key="ve_breakpoint_manual",
        help="Moc przy której wentylacja zaczyna rosnąć nieliniowo. Wartość 0 = użyj automatycznie wykrytego."
    )
    
    # === TEST PROTOCOL PARAMETERS FOR PDF REPORT ===
    st.markdown("---")
    st.subheader("📝 Protokół Testu (dla raportu PDF)")
    st.caption("Wprowadź parametry testu, które pojawią się w sekcji 1.2 Przebieg Testu:")
    
    col_proto1, col_proto2, col_proto3 = st.columns(3)
    with col_proto1:
        test_start_power = st.number_input(
            "Początek testu (W)",
            min_value=0,
            max_value=500,
            value=st.session_state.get("test_start_power", 100),
            step=10,
            key="test_start_power",
            help="Moc początkowa testu"
        )
    with col_proto2:
        test_end_power = st.number_input(
            "Koniec testu (W)",
            min_value=0,
            max_value=800,
            value=st.session_state.get("test_end_power", 400),
            step=10,
            key="test_end_power",
            help="Moc przy której nastąpiła odmowa"
        )
    with col_proto3:
        test_duration = st.text_input(
            "Czas trwania (mm:ss)",
            value=st.session_state.get("test_duration", "45:00"),
            key="test_duration",
            help="Całkowity czas testu w formacie mm:ss"
        )

    st.markdown("---")
    st.subheader("🎯 Wybrane Progi (Manualne)")
    
    # --- V-SLOPE DIAGNOSTICS FOR MANUAL OVERRIDE ASSISTANCE ---
    with st.expander("🔬 Metoda V-Slope (Scientific Diagnostics)", expanded=False):
        st.markdown("### Zaawansowana Analiza Gradientu dVE/dP")
        st.info("""
        Ta metoda wykorzystuje filtr **Savitzky-Golay** do wygładzenia trendów. 
        Pomaga precyzyjnie wskazać moment, w którym wentylacja "ucieka" liniowemu wzrostowi.
        """)
        
        with st.spinner("Przeliczanie gradientów..."):
            vslope_res = detect_vt_vslope_savgol(target_df, result.step_range, 'watts', 'tymeventilation', 'time')
            
        if 'error' in vslope_res:
            st.error(f"Błąd analizy V-Slope: {vslope_res['error']}")
        else:
            v1_w = vslope_res['vt1_watts']
            v2_w = vslope_res['vt2_watts']
            df_s = vslope_res['df_steps']
            
            # Diagnostic Plot
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(10, 4))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0E1117')
            ax1.set_facecolor('#0E1117')
            
            ax1.plot(df_s['watts'], df_s['ve_smooth'], 'b-', label='VE (L/min)', alpha=0.8)
            ax1.set_xlabel('Moc [W]')
            ax1.set_ylabel('VE [L/min]', color='#5da5da')
            
            ax2 = ax1.twinx()
            ax2.plot(df_s['watts'], df_s['slope'], 'g--', label='Slope', alpha=0.5)
            ax2.set_ylabel('Slope (dVE/dP)', color='#60bd68')
            
            if v1_w: ax1.axvline(v1_w, color='#ffa15a', linestyle='--', alpha=0.7, label=f'VT1 Sug: {v1_w}W')
            if v2_w: ax1.axvline(v2_w, color='#ef553b', linestyle='--', alpha=0.7, label=f'VT2 Sug: {v2_w}W')
            
            ax1.legend(loc='upper left', fontsize='x-small')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"💡 Sugestia V-Slope: **VT1: {v1_w}W**, **VT2: {v2_w}W**")
            ma_c1, ma_c2 = st.columns(2)
            with ma_c1:
                if st.button("Aplikuj V1", key="m_apply_v1"):
                    st.session_state['manual_vt1_watts'] = v1_w
                    st.rerun()
            with ma_c2:
                if st.button("Aplikuj V2", key="m_apply_v2"):
                    st.session_state['manual_vt2_watts'] = v2_w
                    st.rerun()
    
    col_z1, col_z2 = st.columns(2)
    
    # --- VT1 CARD ---
    with col_z1:
        if manual_vt1 > 0:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(manual_vt1)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(manual_vt1_hr)} bpm</p>' if manual_vt1_hr > 0 else (f'<p style="margin:0; color:#aaa;"><b>HR (est):</b> {int(vt1_hr_est)} bpm</p>' if vt1_hr_est else '')}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {manual_vt1_ve:.1f} L/min</p>' if manual_vt1_ve > 0 else ''}
                {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(manual_vt1_br)} BR/min</p>' if manual_vt1_br > 0 else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("VT1: Nie ustawiono")

    # --- VT2 CARD ---
    with col_z2:
        if manual_vt2 > 0:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(manual_vt2)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(manual_vt2_hr)} bpm</p>' if manual_vt2_hr > 0 else (f'<p style="margin:0; color:#aaa;"><b>HR (est):</b> {int(vt2_hr_est)} bpm</p>' if vt2_hr_est else '')}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {manual_vt2_ve:.1f} L/min</p>' if manual_vt2_ve > 0 else ''}
                {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(manual_vt2_br)} BR/min</p>' if manual_vt2_br > 0 else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("VT2: Nie ustawiono")

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

    # VT Markers Manual
    if manual_vt1 > 0 and vt1_time_manual is not None:
        fig_thresh.add_vline(x=vt1_time_manual, line=dict(color="#ffa15a", width=3, dash="dash"), layer="above")
        fig_thresh.add_annotation(
            x=vt1_time_manual, y=1, yref="paper",
            text=f"<b>VT1 (Próg Tlenowy)</b><br>{int(manual_vt1)}W",
            showarrow=False, font=dict(color="white", size=11),
            bgcolor="rgba(255, 161, 90, 0.8)", bordercolor="#ffa15a",
            borderwidth=2, borderpad=4, align="center", xanchor="center", yanchor="top"
        )
    
    if manual_vt2 > 0 and vt2_time_manual is not None:
        fig_thresh.add_vline(x=vt2_time_manual, line=dict(color="#ef553b", width=3, dash="dash"), layer="above")
        fig_thresh.add_annotation(
            x=vt2_time_manual, y=1, yref="paper",
            text=f"<b>VT2 (Próg Beztlenowy)</b><br>{int(manual_vt2)}W",
            showarrow=False, font=dict(color="white", size=11),
            bgcolor="rgba(239, 85, 59, 0.8)", bordercolor="#ef553b",
            borderwidth=2, borderpad=4, align="center", xanchor="center", yanchor="bottom",
            yshift=-40
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

    # 4. STREFY TRENINGOWE (NOWE)
    st.divider()
    st.subheader("🎨 Strefy Treningowe")
    st.markdown("Ustaw parametry bazowe do wyliczenia stref:")

    col_z1, col_z2, col_z3 = st.columns(3)
    with col_z1:
        zone_vt2 = st.number_input("VT2 Power (W)", min_value=0, max_value=1000, value=int(manual_vt2) if manual_vt2 > 0 else 250, step=5, help="100% mocy dla wyliczenia stref")
    with col_z2:
        zone_lthr = st.number_input("LTHR (HR na VT2)", min_value=0, max_value=250, value=int(manual_vt2_hr) if manual_vt2_hr > 0 else 170, step=1, help="100% tętna progowego (VT2)")
    with col_z3:
        zone_max_hr = st.number_input("Max Heart Rate (bpm)", min_value=0, max_value=250, value=int(max_hr_input) if max_hr_input > 0 else 190, step=1)

    if zone_vt2 > 0:
        zones = _calculate_zones(manual_vt1, zone_vt2, cp_input, zone_max_hr, zone_lthr)
        
        zone_data = []
        # power_zones_list = list(zones['power'].items())
        # hr_zones_list = list(zones['hr'].items())
        
        power_keys = ['Z1_Recovery', 'Z2_Endurance', 'Z3_Tempo', 'Z4_Threshold', 'Z5_VO2max', 'Z6_Anaerobic']
        
        for i, key in enumerate(power_keys, 1):
            p_low, p_high = zones['power'][key]
            hr_range = zones['hr'].get(key, (None, None))
            
            # Use bpm only, no "S" labels
            hr_val = f"{hr_range[0]} - {hr_range[1]}" if hr_range[0] is not None else "—"
            
            zone_data.append({
                "Strefa": key.split('_')[0], # Z1, Z2, etc.
                "Moc (wat)": f"{p_low} - {p_high} W",
                "Tętno (bpm)": hr_val,
                "Fizjologia": zones['description'].get(key, "")
            })
        
        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)
        _render_zones_bar(zones['power'])
    else:
        st.info("Ustaw manualnie VT1 i VT2, aby wygenerować strefy treningowe.")

def _calculate_zones(vt1: int, vt2: int, cp: int, max_hr: int, lthr: int) -> dict:
    """Obliczanie stref treningowych na podstawie progów."""
    return {
        'power': {
            'Z1_Recovery': (0, int(vt2 * 0.55)),
            'Z2_Endurance': (int(vt2 * 0.55) + 1, int(vt2 * 0.75)),
            'Z3_Tempo': (int(vt2 * 0.75) + 1, int(vt2 * 0.90)),
            'Z4_Threshold': (int(vt2 * 0.90) + 1, int(vt2 * 1.05)),
            'Z5_VO2max': (int(vt2 * 1.05) + 1, int(vt2 * 1.20)),
            'Z6_Anaerobic': (int(vt2 * 1.20) + 1, int(vt2 * 1.50))
        },
        'hr': {
            'Z1_Recovery': (0, int(lthr * 0.72)),
            'Z2_Endurance': (int(lthr * 0.72) + 1, int(lthr * 0.81)),
            'Z3_Tempo': (int(lthr * 0.81) + 1, int(lthr * 0.92)),
            'Z4_Threshold': (int(lthr * 0.92) + 1, int(lthr * 1.00)),
            'Z5_VO2max': (int(lthr * 1.01), max_hr),
            'Z6_Anaerobic': (None, None)
        },
        'description': {
            'Z1_Recovery': 'Regeneracja',
            'Z2_Endurance': 'Baza tlenowa',
            'Z3_Tempo': 'Tempo / Sweet Spot',
            'Z4_Threshold': 'Próg FTP',
            'Z5_VO2max': 'VO2max',
            'Z6_Anaerobic': 'Beztlenowa'
        }
    }

def _render_zones_bar(power_zones: dict):
    """Renderowanie kolorowego paska stref mocy."""
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
    fig = go.Figure()
    
    for i, (zone, (low, high)) in enumerate(power_zones.items()):
        fig.add_trace(go.Bar(
            y=['Strefy Mocy'], x=[high - low], name=zone.replace('_', ' '),
            orientation='h', marker_color=colors[i],
            text=f"{zone.split('_')[0]}<br>{low}-{high}W", textposition='inside',
            hovertemplate=f"<b>{zone.replace('_', ' ')}</b><br>{low}-{high}W<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack', height=100, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False, xaxis=dict(title='Moc (W)', showgrid=False),
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)
