"""
Threshold Analysis and Training Plan Generator UI.

Provides:
- Step test analysis with VT1/VT2/LT1/LT2 detection
- Personalized training zones
- 4-week microcycle plan generator
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional


def render_threshold_analysis_tab(target_df, training_notes, uploaded_file_name, 
                                   cp_input, ftp_input, max_hr_input):
    """
    Render the threshold analysis and training plan tab.
    
    Args:
        target_df: DataFrame with training data
        training_notes: TrainingNotes instance
        uploaded_file_name: Name of uploaded file
        cp_input: Critical Power from sidebar
        ftp_input: FTP from sidebar
        max_hr_input: Max HR from sidebar
    """
    st.header("🎯 Analiza Progów & Plan Treningowy")
    st.markdown("Automatyczna detekcja progów wentylacyjnych i metabolicznych z testu schodkowego oraz generowanie planu treningowego.")
    
    if target_df is None or target_df.empty:
        st.error("Brak danych. Wgraj plik CSV z testem schodkowym.")
        st.stop()
    
    # Check for required columns
    has_ve = 'tymeventilation' in target_df.columns
    has_smo2 = 'smo2' in target_df.columns
    has_watts = 'watts' in target_df.columns
    has_hr = 'hr' in target_df.columns
    
    if not has_watts:
        st.error("Brak danych mocy (kolumna 'watts'). Analiza niemożliwa.")
        st.stop()
    
    # =================================================================
    # SECTION 1: Test Configuration
    # =================================================================
    st.subheader("⚙️ Konfiguracja Testu Schodkowego")
    
    col1, col2 = st.columns(2)
    with col1:
        step_duration = st.slider(
            "Czas trwania stopnia (min)", 
            min_value=1, max_value=5, value=3, step=1,
            help="Standardowy ramp test to 3 minuty na stopień"
        )
    with col2:
        # Info about available data
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
    # SECTION 2: Threshold Detection
    # =================================================================
    st.subheader("📈 Detekcja Progów")
    
    # Run analysis when button clicked or auto-run
    if st.button("🔍 Analizuj Test", type="primary"):
        with st.spinner("Analizuję test schodkowy..."):
            result = _analyze_step_test_internal(
                df=target_df.copy(),
                step_duration_sec=step_duration * 60,
                has_ve=has_ve,
                has_smo2=has_smo2,
                has_hr=has_hr
            )
            st.session_state['threshold_result'] = result
    
    # Display results if available
    if 'threshold_result' in st.session_state:
        result = st.session_state['threshold_result']
        
        # Display detected thresholds
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vt1 = result.get('vt1_watts')
            if vt1:
                st.metric("🟢 VT1 (Próg Tlenowy)", f"{vt1:.0f} W")
                if result.get('vt1_hr'):
                    st.caption(f"@ {result['vt1_hr']:.0f} bpm")
            else:
                st.metric("🟢 VT1", "—")
                st.caption("Nie wykryto")
        
        with col2:
            vt2 = result.get('vt2_watts')
            if vt2:
                st.metric("🔴 VT2 (Próg Beztlenowy)", f"{vt2:.0f} W")
                if result.get('vt2_hr'):
                    st.caption(f"@ {result['vt2_hr']:.0f} bpm")
            else:
                st.metric("🔴 VT2", "—")
                st.caption("Nie wykryto")
        
        with col3:
            lt1 = result.get('lt1_watts')
            if lt1:
                st.metric("🟡 LT1 (SmO2)", f"{lt1:.0f} W")
            else:
                st.metric("🟡 LT1 (SmO2)", "—")
                st.caption("Brak danych SmO2" if not has_smo2 else "Nie wykryto")
        
        with col4:
            lt2 = result.get('lt2_watts')
            if lt2:
                st.metric("🟠 LT2 (SmO2)", f"{lt2:.0f} W")
            else:
                st.metric("🟠 LT2 (SmO2)", "—")
                st.caption("Brak danych SmO2" if not has_smo2 else "Nie wykryto")
        
        # Show analysis notes
        if result.get('notes'):
            with st.expander("📋 Notatki z analizy"):
                for note in result['notes']:
                    st.info(note)
        
        # Store detected values for zones
        detected_vt1 = result.get('vt1_watts') or result.get('lt1_watts') or int(ftp_input * 0.75)
        detected_vt2 = result.get('vt2_watts') or result.get('lt2_watts') or ftp_input
        
        st.session_state['detected_vt1'] = detected_vt1
        st.session_state['detected_vt2'] = detected_vt2
    
    st.divider()
    
    # =================================================================
    # SECTION 3: Training Zones
    # =================================================================
    st.subheader("🎨 Strefy Treningowe")
    
    # Use detected or manual values
    col1, col2 = st.columns(2)
    with col1:
        use_detected = st.checkbox("Użyj wykrytych progów", value=True, disabled='threshold_result' not in st.session_state)
    
    if use_detected and 'detected_vt1' in st.session_state:
        vt1_for_zones = int(st.session_state['detected_vt1'])
        vt2_for_zones = int(st.session_state['detected_vt2'])
    else:
        col1, col2 = st.columns(2)
        with col1:
            vt1_for_zones = st.number_input("VT1 (W)", min_value=50, max_value=500, value=int(ftp_input * 0.75))
        with col2:
            vt2_for_zones = st.number_input("VT2 (W)", min_value=50, max_value=600, value=ftp_input)
    
    # Calculate and display zones
    zones = _calculate_zones(vt1_for_zones, vt2_for_zones, cp_input, max_hr_input)
    
    # Display as table
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
    
    # Visualization bar
    _render_zones_bar(zones['power'])
    
    st.divider()
    
    # =================================================================
    # SECTION 4: Training Plan Generator
    # =================================================================
    st.subheader("📅 Generator Planu 4-Tygodniowego")
    
    col1, col2 = st.columns(2)
    with col1:
        goal = st.selectbox(
            "Cel treningowy",
            options=["general_fitness", "endurance", "threshold", "vo2max"],
            format_func=lambda x: {
                "general_fitness": "🏃 Ogólna forma",
                "endurance": "🚴 Wytrzymałość",
                "threshold": "⚡ Próg (FTP)",
                "vo2max": "🔥 VO2max"
            }.get(x, x)
        )
    with col2:
        weekly_hours = st.slider(
            "Dostępne godziny tygodniowo",
            min_value=4.0, max_value=20.0, value=8.0, step=0.5
        )
    
    if st.button("🚀 Generuj Plan Treningowy", type="primary"):
        plan = _generate_plan(vt1_for_zones, vt2_for_zones, goal, weekly_hours)
        st.session_state['training_plan'] = plan
    
    # Display plan if available
    if 'training_plan' in st.session_state:
        plan = st.session_state['training_plan']
        
        st.success(f"**Plan wygenerowany!** Cel: {plan['goal_name']} | VT1={plan['vt1']}W, VT2={plan['vt2']}W")
        
        # Display each week
        for week in plan['weeks']:
            with st.expander(f"📆 Tydzień {week['week_number']} - {week['phase_name']} ({week['total_hours']}h)", expanded=week['week_number'] == 1):
                
                # Sessions table
                sessions_df = pd.DataFrame(week['sessions'])
                sessions_df.columns = ["Dzień", "Typ", "Czas (min)", "Cel mocy", "RPE"]
                st.dataframe(sessions_df, use_container_width=True, hide_index=True)
        
        st.info(f"**Struktura:** {plan['structure']}")


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def _analyze_step_test_internal(df: pd.DataFrame, step_duration_sec: int, 
                                 has_ve: bool, has_smo2: bool, has_hr: bool) -> dict:
    """Analyze step test and detect thresholds."""
    from scipy import stats
    
    result = {
        'vt1_watts': None, 'vt2_watts': None,
        'lt1_watts': None, 'lt2_watts': None,
        'vt1_hr': None, 'vt2_hr': None,
        'notes': []
    }
    
    # Normalize columns
    df.columns = df.columns.str.lower().str.strip()
    
    if 'time' not in df.columns:
        result['notes'].append("Brak kolumny czasu")
        return result
    
    # Calculate number of steps
    total_duration = df['time'].max() - df['time'].min()
    num_steps = int(total_duration / step_duration_sec)
    
    if num_steps < 3:
        result['notes'].append(f"Za mało stopni ({num_steps}). Minimum 3 wymagane.")
        return result
    
    result['notes'].append(f"Wykryto {num_steps} stopni po {step_duration_sec//60} min")
    
    # Analyze each step
    step_data = []
    start_time = df['time'].min()
    
    for step in range(num_steps):
        step_start = start_time + step * step_duration_sec
        step_end = step_start + step_duration_sec
        
        # Use last 60s of step for stable values
        stable_start = step_end - 60
        mask = (df['time'] >= stable_start) & (df['time'] < step_end)
        step_df = df[mask]
        
        if len(step_df) < 10:
            continue
        
        info = {'step': step + 1}
        
        if 'watts' in step_df.columns:
            info['power'] = step_df['watts'].mean()
        
        if 'hr' in step_df.columns:
            info['hr'] = step_df['hr'].mean()
        
        if has_ve and 'tymeventilation' in step_df.columns:
            if len(step_df) > 1:
                slope, _, _, _, _ = stats.linregress(step_df['time'], step_df['tymeventilation'])
                info['ve_slope'] = slope
        
        if has_smo2 and 'smo2' in step_df.columns:
            if len(step_df) > 1:
                slope, _, _, _, _ = stats.linregress(step_df['time'], step_df['smo2'])
                info['smo2_slope'] = slope
        
        step_data.append(info)
    
    # Detect VT1/VT2 from ventilation slopes
    if has_ve:
        ve_steps = [s for s in step_data if 've_slope' in s]
        for i, s in enumerate(ve_steps):
            slope = s['ve_slope']
            if result['vt1_watts'] is None and slope > 0.05:
                prev = ve_steps[max(0, i-1)]
                result['vt1_watts'] = prev.get('power')
                result['vt1_hr'] = prev.get('hr')
            if result['vt2_watts'] is None and slope > 0.15:
                prev = ve_steps[max(0, i-1)]
                result['vt2_watts'] = prev.get('power')
                result['vt2_hr'] = prev.get('hr')
    
    # Detect LT1/LT2 from SmO2 slopes
    if has_smo2:
        smo2_steps = [s for s in step_data if 'smo2_slope' in s]
        for i, s in enumerate(smo2_steps):
            slope = s['smo2_slope']
            if result['lt1_watts'] is None and slope <= 0.005:
                prev = smo2_steps[max(0, i-1)]
                result['lt1_watts'] = prev.get('power')
            if result['lt2_watts'] is None and slope < -0.01:
                prev = smo2_steps[max(0, i-1)]
                result['lt2_watts'] = prev.get('power')
    
    if result['vt1_watts']:
        result['notes'].append(f"VT1 wykryty z wentylacji: {result['vt1_watts']:.0f}W")
    if result['vt2_watts']:
        result['notes'].append(f"VT2 wykryty z wentylacji: {result['vt2_watts']:.0f}W")
    if result['lt1_watts']:
        result['notes'].append(f"LT1 wykryty z SmO2: {result['lt1_watts']:.0f}W")
    if result['lt2_watts']:
        result['notes'].append(f"LT2 wykryty z SmO2: {result['lt2_watts']:.0f}W")
    
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
            y=['Strefy Mocy'],
            x=[high - low],
            name=zone.replace('_', ' '),
            orientation='h',
            marker_color=colors[i],
            text=f"{zone.split('_')[0]}<br>{low}-{high}W",
            textposition='inside',
            hovertemplate=f"<b>{zone.replace('_', ' ')}</b><br>{low}-{high}W<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='stack',
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(title='Moc (W)', showgrid=False),
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _generate_plan(vt1: int, vt2: int, goal: str, weekly_hours: float) -> dict:
    """Generate 4-week training microcycle."""
    
    def create_week(week_num: int, phase: str, hours: float) -> dict:
        templates = {
            "build": [
                ["Pon", "Z2 Endurance", 60, f"< {vt1}W", 3],
                ["Śr", "Tempo", 75, f"{vt1}-{(vt1+vt2)//2}W", 5],
                ["Pt", "Sweet Spot", 60, f"2x15min @ {int(vt2*0.9)}W", 6],
                ["Nd", "Long Z2", 120, f"< {vt1}W", 3]
            ],
            "peak": [
                ["Pon", "Recovery", 45, f"< {int(vt1*0.8)}W", 2],
                ["Śr", "VO2max", 60, f"5x4min @ {int(vt2*1.1)}W", 8],
                ["Pt", "Threshold", 75, f"2x20min @ {vt2}W", 7],
                ["Nd", "Long Tempo", 150, f"{vt1}-{(vt1+vt2)//2}W", 5]
            ],
            "recovery": [
                ["Pon", "Rest", 0, "Off", 0],
                ["Śr", "Easy Z1", 45, f"< {int(vt1*0.65)}W", 2],
                ["Pt", "Light", 45, f"< {vt1}W", 4],
                ["Nd", "Z2", 60, f"< {vt1}W", 3]
            ]
        }
        
        sessions = templates.get(phase, templates["build"])
        total_min = sum(s[2] for s in sessions)
        scale = (hours * 60) / total_min if total_min > 0 else 1
        
        scaled = [[s[0], s[1], max(0, int(s[2] * scale)), s[3], s[4]] for s in sessions]
        
        return {
            "week_number": week_num,
            "phase": phase,
            "phase_name": {"build": "Budowanie", "peak": "Szczyt", "recovery": "Regeneracja"}[phase],
            "total_hours": round(hours, 1),
            "sessions": scaled
        }
    
    return {
        "goal": goal,
        "goal_name": {"general_fitness": "Ogólna forma", "endurance": "Wytrzymałość",
                      "threshold": "Próg FTP", "vo2max": "VO2max"}.get(goal, goal),
        "vt1": vt1,
        "vt2": vt2,
        "structure": "Build → Build → Peak → Recovery",
        "weeks": [
            create_week(1, "build", weekly_hours * 0.9),
            create_week(2, "build", weekly_hours * 1.0),
            create_week(3, "peak", weekly_hours * 1.1),
            create_week(4, "recovery", weekly_hours * 0.6)
        ]
    }
