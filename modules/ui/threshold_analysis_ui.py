"""
Threshold Analysis and Training Plan Generator UI.

Provides:
- Step test analysis with VT1/VT2/LT1/LT2 detection
- Time range selection for step test portion
- Personalized training zones
- 4-week microcycle plan generator
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def render_threshold_analysis_tab(target_df, training_notes, uploaded_file_name, 
                                   cp_input, ftp_input, max_hr_input):
    """Render the threshold analysis and training plan tab."""
    st.header("🎯 Analiza Progów & Plan Treningowy")
    st.markdown("Automatyczna detekcja progów wentylacyjnych i metabolicznych z testu schodkowego oraz generowanie planu treningowego.")
    
    if target_df is None or target_df.empty:
        st.error("Brak danych. Wgraj plik CSV z testem schodkowym.")
        st.stop()
    
    # Check for required columns
    has_ve = 'tymeventilation' in target_df.columns
    has_smo2 = 'smo2' in target_df.columns
    has_watts = 'watts' in target_df.columns
    has_hr = 'hr' in target_df.columns or 'heartrate' in target_df.columns
    hr_col = 'hr' if 'hr' in target_df.columns else 'heartrate'
    
    if not has_watts:
        st.error("Brak danych mocy (kolumna 'watts'). Analiza niemożliwa.")
        st.stop()
    
    # Calculate time in minutes for easier UI
    if 'time' in target_df.columns:
        total_duration_sec = int(target_df['time'].max() - target_df['time'].min())
        total_duration_min = max(1, total_duration_sec // 60)  # Avoid division by zero
    else:
        total_duration_sec = len(target_df)
        total_duration_min = max(1, total_duration_sec // 60)
    
    # =================================================================
    # SECTION 1: Power Preview & Time Range Selection
    # =================================================================
    st.subheader("📊 Wybór Zakresu Testu Schodkowego")
    st.markdown("**Zaznacz zakres czasowy samego testu schodkowego** (bez rozgrzewki i schłodzenia)")
    
    # Create power preview chart
    fig = go.Figure()
    
    # Power trace
    if 'time' in target_df.columns:
        x_data = target_df['time'] / 60  # Convert to minutes
    else:
        x_data = list(range(len(target_df)))
        x_data = [x / 60 for x in x_data]
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=target_df['watts'],
        name='Moc',
        fill='tozeroy',
        line=dict(color='#00d4aa', width=1),
        fillcolor='rgba(0, 212, 170, 0.3)'
    ))
    
    if has_hr and hr_col in target_df.columns:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=target_df[hr_col],
            name='HR',
            yaxis='y2',
            line=dict(color='#ff6b6b', width=1, dash='dot')
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title='Czas (min)', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title='Moc (W)', showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis2=dict(title='HR (bpm)', overlaying='y', side='right'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='top', y=1.15),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time range selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        test_start_min = st.number_input(
            "⏱️ Start testu (min)", 
            min_value=0, 
            max_value=total_duration_min - 1,
            value=min(10, total_duration_min // 4),  # Default: 10 min or 1/4 of workout
            step=1,
            help="Minuta rozpoczęcia testu schodkowego (pomiń rozgrzewkę)"
        )
    with col2:
        test_end_min = st.number_input(
            "🏁 Koniec testu (min)",
            min_value=test_start_min + 1,
            max_value=total_duration_min,
            value=min(test_start_min + 30, total_duration_min),  # Default: +30 min
            step=1,
            help="Minuta zakończenia testu schodkowego (przed schłodzeniem)"
        )
    with col3:
        test_duration = test_end_min - test_start_min
        st.metric("Czas testu", f"{test_duration} min")
    
    # Convert to seconds for analysis
    test_start_sec = test_start_min * 60
    test_end_sec = test_end_min * 60
    
    st.divider()
    
    # =================================================================
    # SECTION 2: Test Configuration
    # =================================================================
    st.subheader("⚙️ Konfiguracja Testu")
    
    col1, col2 = st.columns(2)
    with col1:
        step_duration = st.slider(
            "Czas trwania stopnia (min)", 
            min_value=1, max_value=5, value=3, step=1,
            help="Standardowy ramp test to 3 minuty na stopień"
        )
        expected_steps = test_duration // step_duration
        st.caption(f"Oczekiwana liczba stopni: ~{expected_steps}")
    
    with col2:
        available = []
        if has_ve: available.append("✅ Wentylacja (VE)")
        else: available.append("❌ Wentylacja (VE)")
        if has_smo2: available.append("✅ SmO2")
        else: available.append("❌ SmO2")
        if has_hr: available.append("✅ Tętno (HR)")
        else: available.append("❌ Tętno (HR)")
        st.markdown("**Dostępne dane:**")
        st.markdown(" | ".join(available))
    
    st.divider()
    
    # =================================================================
    # SECTION 3: Threshold Detection
    # =================================================================
    st.subheader("📈 Detekcja Progów")
    
    if st.button("🔍 Analizuj Test", type="primary"):
        with st.spinner("Analizuję test schodkowy..."):
            # Filter data to selected range
            if 'time' in target_df.columns:
                min_time = target_df['time'].min()
                mask = (target_df['time'] >= min_time + test_start_sec) & (target_df['time'] <= min_time + test_end_sec)
            else:
                mask = (target_df.index >= test_start_sec) & (target_df.index <= test_end_sec)
            
            test_df = target_df[mask].copy()
            
            if len(test_df) < 100:
                st.error(f"Za mało danych w wybranym zakresie ({len(test_df)} rekordów). Rozszerz zakres.")
            else:
                result = _analyze_step_test_internal(
                    df=test_df,
                    step_duration_sec=step_duration * 60,
                    has_ve=has_ve,
                    has_smo2=has_smo2,
                    has_hr=has_hr
                )
                st.session_state['threshold_result'] = result
    
    # Display results
    if 'threshold_result' in st.session_state:
        result = st.session_state['threshold_result']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vt1 = result.get('vt1_watts')
            if vt1:
                st.metric("🟢 VT1 (Próg Tlenowy)", f"{vt1:.0f} W")
                if result.get('vt1_hr'):
                    st.caption(f"@ {result['vt1_hr']:.0f} bpm")
            else:
                st.metric("🟢 VT1", "—")
        
        with col2:
            vt2 = result.get('vt2_watts')
            if vt2:
                st.metric("🔴 VT2 (Próg Beztlenowy)", f"{vt2:.0f} W")
                if result.get('vt2_hr'):
                    st.caption(f"@ {result['vt2_hr']:.0f} bpm")
            else:
                st.metric("🔴 VT2", "—")
        
        with col3:
            lt1 = result.get('lt1_watts')
            if lt1:
                st.metric("🟡 LT1 (SmO2)", f"{lt1:.0f} W")
            else:
                st.metric("🟡 LT1", "—")
        
        with col4:
            lt2 = result.get('lt2_watts')
            if lt2:
                st.metric("🟠 LT2 (SmO2)", f"{lt2:.0f} W")
            else:
                st.metric("🟠 LT2", "—")
        
        if result.get('notes'):
            with st.expander("📋 Notatki z analizy"):
                for note in result['notes']:
                    st.info(note)
        
        detected_vt1 = result.get('vt1_watts') or result.get('lt1_watts') or int(ftp_input * 0.75)
        detected_vt2 = result.get('vt2_watts') or result.get('lt2_watts') or ftp_input
        st.session_state['detected_vt1'] = detected_vt1
        st.session_state['detected_vt2'] = detected_vt2
    
    st.divider()
    
    # =================================================================
    # SECTION 4: Training Zones
    # =================================================================
    st.subheader("🎨 Strefy Treningowe")
    
    use_detected = st.checkbox("Użyj wykrytych progów", value=True, 
                                disabled='threshold_result' not in st.session_state)
    
    if use_detected and 'detected_vt1' in st.session_state:
        vt1_for_zones = int(st.session_state['detected_vt1'])
        vt2_for_zones = int(st.session_state['detected_vt2'])
    else:
        col1, col2 = st.columns(2)
        with col1:
            vt1_for_zones = st.number_input("VT1 (W)", min_value=50, max_value=500, value=int(ftp_input * 0.75))
        with col2:
            vt2_for_zones = st.number_input("VT2 (W)", min_value=50, max_value=600, value=ftp_input)
    
    zones = _calculate_zones(vt1_for_zones, vt2_for_zones, cp_input, max_hr_input)
    
    zone_data = []
    for zone_name, (low, high) in zones['power'].items():
        hr_range = zones['hr'].get(zone_name, (None, None))
        zone_data.append({
            "Strefa": zone_name.replace("_", " "),
            "Moc (W)": f"{low} - {high}",
            "Tętno (bpm)": f"{hr_range[0]} - {hr_range[1]}" if hr_range[0] else "—",
            "Opis": zones['description'].get(zone_name, "")
        })
    
    st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)
    _render_zones_bar(zones['power'])
    


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def _analyze_step_test_internal(df: pd.DataFrame, step_duration_sec: int, 
                                 has_ve: bool, has_smo2: bool, has_hr: bool) -> dict:
    """
    Analyze step test and detect thresholds using step averages and delta ratios.
    
    Algorithm:
    1. Calculate average VE/SmO2 for each step (last 60s)
    2. Compute delta (change) between consecutive steps
    3. Detect VT1/VT2 when delta ratio > threshold
    4. Detect LT1/LT2 when SmO2 delta changes sign or drops significantly
    """
    result = {'vt1_watts': None, 'vt2_watts': None, 'lt1_watts': None, 'lt2_watts': None,
              'vt1_hr': None, 'vt2_hr': None, 'notes': [], 'steps': []}
    
    df.columns = df.columns.str.lower().str.strip()
    
    if 'time' not in df.columns:
        result['notes'].append("Brak kolumny czasu")
        return result
    
    df = df.copy()
    df['time'] = df['time'] - df['time'].min()
    
    total_duration = df['time'].max()
    num_steps = int(total_duration / step_duration_sec)
    
    if num_steps < 3:
        result['notes'].append(f"Za mało stopni ({num_steps}). Minimum 3 wymagane.")
        return result
    
    result['notes'].append(f"Wykryto {num_steps} stopni po {step_duration_sec//60} min")
    
    steps = []
    hr_col = 'hr' if 'hr' in df.columns else ('heartrate' if 'heartrate' in df.columns else None)
    
    for step in range(num_steps):
        step_start = step * step_duration_sec
        step_end = step_start + step_duration_sec
        stable_start = step_end - 60
        
        mask = (df['time'] >= stable_start) & (df['time'] < step_end)
        step_df = df[mask]
        
        if len(step_df) < 10:
            continue
        
        step_info = {
            'step_num': step + 1,
            'power': step_df['watts'].mean() if 'watts' in step_df.columns else None,
            'hr': step_df[hr_col].mean() if hr_col and hr_col in step_df.columns else None,
            've_avg': step_df['tymeventilation'].mean() if has_ve and 'tymeventilation' in step_df.columns else None,
            'smo2_avg': step_df['smo2'].mean() if has_smo2 and 'smo2' in step_df.columns else None,
        }
        steps.append(step_info)
    
    if len(steps) < 3:
        result['notes'].append(f"Za mało ważnych stopni ({len(steps)})")
        return result
    
    result['notes'].append(f"Przeanalizowano {len(steps)} stopni")
    result['steps'] = steps
    
    powers = [f"{s['power']:.0f}W" for s in steps if s['power']]
    if powers:
        result['notes'].append(f"Moce: {', '.join(powers)}")
    
    # Compute deltas
    for i in range(1, len(steps)):
        if steps[i]['ve_avg'] and steps[i-1]['ve_avg']:
            steps[i]['ve_delta'] = steps[i]['ve_avg'] - steps[i-1]['ve_avg']
        else:
            steps[i]['ve_delta'] = None
        
        if steps[i]['smo2_avg'] and steps[i-1]['smo2_avg']:
            steps[i]['smo2_delta'] = steps[i]['smo2_avg'] - steps[i-1]['smo2_avg']
        else:
            steps[i]['smo2_delta'] = None
    
    # Detect VT1/VT2 from VE delta ratios
    if has_ve:
        ve_steps = [s for s in steps if s.get('ve_delta') is not None]
        for i in range(1, len(ve_steps)):
            curr_delta = ve_steps[i]['ve_delta']
            prev_delta = ve_steps[i-1].get('ve_delta')
            
            if prev_delta and prev_delta > 0:
                delta_ratio = curr_delta / prev_delta
                
                if result['vt1_watts'] is None and delta_ratio > 1.5:
                    result['vt1_watts'] = ve_steps[i-1]['power']
                    result['vt1_hr'] = ve_steps[i-1]['hr']
                    result['notes'].append(f"VT1: delta ratio = {delta_ratio:.2f}")
                
                if result['vt2_watts'] is None and delta_ratio > 2.0:
                    result['vt2_watts'] = ve_steps[i-1]['power']
                    result['vt2_hr'] = ve_steps[i-1]['hr']
                    result['notes'].append(f"VT2: delta ratio = {delta_ratio:.2f}")
    
    # Detect LT1/LT2 from SmO2 delta
    if has_smo2:
        smo2_steps = [s for s in steps if s.get('smo2_delta') is not None]
        for i, s in enumerate(smo2_steps):
            delta = s['smo2_delta']
            
            if result['lt1_watts'] is None and delta <= 0:
                prev = smo2_steps[max(0, i-1)]
                result['lt1_watts'] = prev['power']
                result['notes'].append(f"LT1: SmO2 delta = {delta:.2f}%")
            
            if result['lt2_watts'] is None and delta < -2.0:
                prev = smo2_steps[max(0, i-1)]
                result['lt2_watts'] = prev['power']
                result['notes'].append(f"LT2: SmO2 delta = {delta:.2f}%")
    
    if result['vt1_watts']:
        result['notes'].append(f"✓ VT1: {result['vt1_watts']:.0f}W")
    if result['vt2_watts']:
        result['notes'].append(f"✓ VT2: {result['vt2_watts']:.0f}W")
    if result['lt1_watts']:
        result['notes'].append(f"✓ LT1: {result['lt1_watts']:.0f}W")
    if result['lt2_watts']:
        result['notes'].append(f"✓ LT2: {result['lt2_watts']:.0f}W")
    
    return result





def _calculate_zones(vt1: int, vt2: int, cp: int, max_hr: int) -> dict:
    """Calculate training zones from thresholds."""
    return {
        'power': {
            'Z1_Recovery': (0, int(vt1 * 0.75)),
            'Z2_Endurance': (int(vt1 * 0.75), vt1),
            'Z3_Tempo': (vt1, int((vt1 + vt2) / 2)),
            'Z4_Threshold': (int((vt1 + vt2) / 2), vt2),
            'Z5_VO2max': (vt2, int(cp * 1.2)),
            'Z6_Anaerobic': (int(cp * 1.2), int(cp * 1.5))
        },
        'hr': {
            'Z1_Recovery': (0, int(max_hr * 0.6)),
            'Z2_Endurance': (int(max_hr * 0.6), int(max_hr * 0.7)),
            'Z3_Tempo': (int(max_hr * 0.7), int(max_hr * 0.8)),
            'Z4_Threshold': (int(max_hr * 0.8), int(max_hr * 0.9)),
            'Z5_VO2max': (int(max_hr * 0.9), max_hr),
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
    """Render a colored bar showing power zones."""
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


