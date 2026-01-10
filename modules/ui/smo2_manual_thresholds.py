import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.quality import check_step_test_protocol

def render_smo2_manual_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input):
    """Ręczna edycja progów SmO2 (LT1/LT2) i wizualizacja na wykresie saturacji."""
    st.header("🛠️ Manualna Edycja Progów SmO2 (LT1 / LT2)")
    st.markdown("Wprowadź własne wartości mocy dla progów metabolicznych SmO2, aby zobaczyć je na wykresie.")

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    # Normalize columns first
    target_df.columns = target_df.columns.str.lower().str.strip()

    if 'smo2' not in target_df.columns:
        st.error("Brak danych SmO2 w pliku!")
        st.stop()

    if 'time' not in target_df.columns:
        st.error("Brak kolumny czasu!")
        st.stop()

    # Handle HR aliases
    if 'hr' not in target_df.columns:
        for alias in ['heart_rate', 'heart rate', 'bpm', 'tętno', 'heartrate', 'heart_rate_bpm']:
            if alias in target_df.columns:
                target_df.rename(columns={alias: 'hr'}, inplace=True)
                break

    # Wygładzanie
    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    target_df['smo2_smooth'] = target_df['smo2'].rolling(window=10, center=True).mean()
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # --- Quality Check: Protocol Compliance ---
    st.subheader("📋 Weryfikacja Protokołu")
    
    proto_check = check_step_test_protocol(target_df)
    
    if not proto_check['is_valid']:
        st.warning("⚠️ Protokół może nie być idealnym testem schodkowym. Wyniki mogą być przybliżone.")
    else:
        st.success("✅ Protokół Testu Stopniowanego: Poprawny")

    st.markdown("---")

    # 2. EDYCJA MANUALNA
    st.subheader("✍️ Parametry Manualne")
    
    # Próba pobrania domyślnych wartości z automatycznej detekcji
    with st.spinner("Analizowanie progów SmO2 dla sugestii..."):
        result = analyze_step_test(
            target_df, 
            power_column='watts',
            ve_column='tymeventilation' if 'tymeventilation' in target_df.columns else None,
            smo2_column='smo2',
            hr_column='hr' if 'hr' in target_df.columns else None,
            time_column='time'
        )

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        manual_lt1 = st.number_input("Manualny LT1 (Moc W)", min_value=0, max_value=1000, value=int(result.smo2_1_watts) if result.smo2_1_watts else 0, step=5, key="smo2_lt1_m")
    with col_inp2:
        manual_lt2 = st.number_input("Manualny LT2 (Moc W)", min_value=0, max_value=1000, value=int(result.smo2_2_watts) if result.smo2_2_watts else 0, step=5, key="smo2_lt2_m")

    # Obliczanie HR i SmO2 dla podanych mocy manualnych
    def find_values_for_power(power):
        if power <= 0: return None, None
        if 'watts_smooth_5s' in target_df.columns:
            idx = (target_df['watts_smooth_5s'] - power).abs().idxmin()
        elif 'watts' in target_df.columns:
            idx = (target_df['watts'] - power).abs().idxmin()
        else:
            return None, None
        hr = target_df.loc[idx, 'hr'] if 'hr' in target_df.columns else None
        smo2 = target_df.loc[idx, 'smo2_smooth'] if 'smo2_smooth' in target_df.columns else None
        return hr, smo2

    def find_time_for_power(power):
        if power <= 0: return None
        if 'watts_smooth_5s' in target_df.columns:
            idx = (target_df['watts_smooth_5s'] - power).abs().idxmin()
            return target_df.loc[idx, 'time']
        elif 'watts' in target_df.columns:
            idx = (target_df['watts'] - power).abs().idxmin()
            return target_df.loc[idx, 'time']
        return None

    lt1_hr_est, lt1_smo2_est = find_values_for_power(manual_lt1)
    lt2_hr_est, lt2_smo2_est = find_values_for_power(manual_lt2)
    lt1_time_manual = find_time_for_power(manual_lt1)
    lt2_time_manual = find_time_for_power(manual_lt2)

    # Additional manual inputs
    st.markdown("---")
    col_inpa, col_inpb = st.columns(2)
    
    with col_inpa:
        st.caption("Dodatkowe parametry LT1")
        manual_lt1_hr = st.number_input("LT1 HR (bpm)", min_value=0, max_value=250, value=int(lt1_hr_est) if lt1_hr_est else 0, step=1, key="smo2_lt1_hr_m")
        manual_lt1_smo2 = st.number_input("LT1 SmO2 (%)", min_value=0.0, max_value=100.0, value=float(lt1_smo2_est) if lt1_smo2_est else 0.0, step=0.5, key="smo2_lt1_smo2_m")

    with col_inpb:
        st.caption("Dodatkowe parametry LT2")
        manual_lt2_hr = st.number_input("LT2 HR (bpm)", min_value=0, max_value=250, value=int(lt2_hr_est) if lt2_hr_est else 0, step=1, key="smo2_lt2_hr_m")
        manual_lt2_smo2 = st.number_input("LT2 SmO2 (%)", min_value=0.0, max_value=100.0, value=float(lt2_smo2_est) if lt2_smo2_est else 0.0, step=0.5, key="smo2_lt2_smo2_m")

    # Reoxy Half-Time - manual input for PDF report
    st.markdown("---")
    st.caption("⏱️ Reoxy Half-Time dla raportu PDF (czas półodnowy reoxygenacji):")
    reoxy_halftime_manual = st.number_input(
        "Reoxy Half-Time (s)", 
        min_value=0, 
        max_value=300, 
        value=0, 
        step=5, 
        key="reoxy_halftime_manual",
        help="Czas w sekundach do połowy reoxygenacji po wysiłku. Wartość 0 = użyj automatycznie wykrytego."
    )

    st.markdown("---")
    st.subheader("🎯 Wybrane Progi SmO2 (Manualne)")
    
    col_z1, col_z2 = st.columns(2)
    
    # --- LT1 CARD ---
    with col_z1:
        if manual_lt1 > 0:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #2ca02c; background-color: #222;">
                <h3 style="margin:0; color: #2ca02c;">LT1 (SteadyState)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(manual_lt1)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(manual_lt1_hr)} bpm</p>' if manual_lt1_hr > 0 else (f'<p style="margin:0; color:#aaa;"><b>HR (est):</b> {int(lt1_hr_est)} bpm</p>' if lt1_hr_est else '')}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {manual_lt1_smo2:.1f}%</p>' if manual_lt1_smo2 > 0 else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(manual_lt1/cp_input)*100:.0f}% CP")
        else:
            st.info("LT1: Nie ustawiono")

    # --- LT2 CARD ---
    with col_z2:
        if manual_lt2 > 0:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #d62728; background-color: #222;">
                <h3 style="margin:0; color: #d62728;">LT2 (Próg)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(manual_lt2)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(manual_lt2_hr)} bpm</p>' if manual_lt2_hr > 0 else (f'<p style="margin:0; color:#aaa;"><b>HR (est):</b> {int(lt2_hr_est)} bpm</p>' if lt2_hr_est else '')}
                {f'<p style="margin:0; color:#aaa;"><b>SmO2:</b> {manual_lt2_smo2:.1f}%</p>' if manual_lt2_smo2 > 0 else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                st.caption(f"~{(manual_lt2/cp_input)*100:.0f}% CP")
        else:
            st.info("LT2: Nie ustawiono")

    st.markdown("---")

    # 3. Wykres z zaznaczonymi strefami
    st.subheader("📈 Wizualizacja Progów SmO2")
    
    fig_thresh = go.Figure()

    # SmO2 (Primary)
    fig_thresh.add_trace(go.Scatter(
        x=target_df['time'], y=target_df['smo2_smooth'],
        customdata=target_df['time_str'],
        mode='lines', name='SmO2 (%)',
        line=dict(color='#2ca02c', width=2),
        hovertemplate="<b>Czas:</b> %{customdata}<br><b>SmO2:</b> %{y:.1f}%<extra></extra>"
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

    # LT Markers Manual
    if manual_lt1 > 0 and lt1_time_manual is not None:
        fig_thresh.add_vline(x=lt1_time_manual, line=dict(color="#2ca02c", width=3, dash="dash"), layer="above")
        fig_thresh.add_annotation(
            x=lt1_time_manual, y=1, yref="paper",
            text=f"<b>LT1 (SteadyState)</b><br>{int(manual_lt1)}W",
            showarrow=False, font=dict(color="white", size=11),
            bgcolor="rgba(44, 160, 44, 0.8)", bordercolor="#2ca02c",
            borderwidth=2, borderpad=4, align="center", xanchor="center", yanchor="top"
        )
    
    if manual_lt2 > 0 and lt2_time_manual is not None:
        fig_thresh.add_vline(x=lt2_time_manual, line=dict(color="#d62728", width=3, dash="dash"), layer="above")
        fig_thresh.add_annotation(
            x=lt2_time_manual, y=1, yref="paper",
            text=f"<b>LT2 (Próg)</b><br>{int(manual_lt2)}W",
            showarrow=False, font=dict(color="white", size=11),
            bgcolor="rgba(214, 39, 40, 0.8)", bordercolor="#d62728",
            borderwidth=2, borderpad=4, align="center", xanchor="center", yanchor="bottom",
            yshift=-40
        )

    fig_thresh.update_layout(
        title="Dynamika SmO2 z Progami LT1/LT2",
        xaxis_title="Czas",
        yaxis=dict(title=dict(text="SmO2 (%)", font=dict(color="#2ca02c"))),
        yaxis2=dict(title=dict(text="Moc (W)", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_thresh, use_container_width=True)

    # ===== TEORIA =====
    with st.expander("🩸 TEORIA: Progi SmO2 (LT1 / LT2)", expanded=False):
        st.markdown("""
        ## Co to są progi SmO2?
        
        **Progi SmO2** to punkty, w których saturacja tlenowa mięśni (SmO2) zaczyna spadać w charakterystyczny sposób względem obciążenia.
        
        | Próg | Inna nazwa | Fizjologia | Typowy SmO2 |
        |------|-----------|------------|-------------|
        | **LT1** | Próg tlenowy, AeT | Początek desaturacji | ~60-70% |
        | **LT2** | Próg beztlenowy, AnT | Szybka desaturacja | ~40-50% |
        
        ---
        
        ## Jak działa detekcja?
        
        System analizuje:
        1. **Trend SmO2**: Spadek nachylenia (slope) wskazuje na rosnące zużycie tlenu
        2. **Slope < -0.01**: Wskazuje na początek desaturacji (LT1)
        3. **Slope < -0.02**: Wskazuje na szybką desaturację (LT2)
        
        ---
        
        ## Zastosowanie progów
        
        | Strefa | Zakres | Cel treningowy |
        |--------|--------|----------------|
        | **Z1 (Recovery)** | < LT1 | Regeneracja, rozgrzewka |
        | **Z2 (Endurance)** | LT1 - środek | Baza tlenowa |
        | **Z3 (Tempo)** | środek - LT2 | Sweet Spot |
        | **Z4 (Threshold)** | LT2 ± 5% | FTP, próg |
        | **Z5+ (VO2max)** | > LT2 | Interwały, moc szczytowa |
        
        ---
        
        ## Wymagania testu
        
        ⚠️ **Dla wiarygodnych wyników potrzebujesz:**
        - Test stopniowany (Ramp Test) z liniowym wzrostem mocy
        - Minimum 10-15 minut narastającego obciążenia
        - Poprawnie założony sensor NIRS (np. Moxy, TrainRed)
        - Brak przerw i wahań mocy
        """)
