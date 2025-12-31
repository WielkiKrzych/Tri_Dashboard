import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.quality import check_step_test_protocol

def render_smo2_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input):
    """Detekcja progów SmO2 (AeT/AnT) - wymaga Ramp Test."""
    st.header("🎯 Detekcja Progów SmO2")
    st.markdown("Automatyczna detekcja progów metabolicznych na podstawie saturacji mięśniowej (SmO2). **Wymaga testu stopniowanego (Ramp Test).**")

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()
    
    # Normalize
    target_df.columns = target_df.columns.str.lower().str.strip()

    # Aliases
    if 'hr' not in target_df.columns:
        for alias in ['heart_rate', 'heart rate', 'bpm', 'tętno', 'heartrate', 'heart_rate_bpm']:
            if alias in target_df.columns:
                target_df.rename(columns={alias: 'hr'}, inplace=True)
                break
    
    if 'smo2' not in target_df.columns:
        st.error("Brak danych SmO2 w pliku!")
        st.stop()

    if 'time' not in target_df.columns:
         st.error("Brak kolumny czasu!")
         st.stop()

    # Wygładzanie
    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    
    # SmO2 smoothing
    target_df['smo2_smooth'] = target_df['smo2'].rolling(window=10, center=True).mean()
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # --- Quality Check ---
    st.subheader("📋 Weryfikacja Protokołu")
    proto_check = check_step_test_protocol(target_df)
    
    if not proto_check['is_valid']:
        st.warning("⚠️ Protokół może nie być idealnym testem schodkowym. Wyniki mogą być przybliżone.")
    else:
        st.success("✅ Protokół Testu Stopniowanego: Poprawny")

    st.markdown("---")

    # 2. DETEKCJA AUTOMATYCZNA
    with st.spinner("Analizowanie trendów SmO2..."):
        result = analyze_step_test(
            target_df, 
            power_column='watts',
            ve_column='tymeventilation', # Still passed but we focus on SmO2 results
            smo2_column='smo2',
            hr_column='hr' if 'hr' in target_df.columns else None,
            time_column='time'
        )
    
    # SmO2 Results
    t1_watts = result.smo2_1_watts
    t2_watts = result.smo2_2_watts
    t1_hr = result.smo2_1_hr
    t2_hr = result.smo2_2_hr
    
    # Display step detection info
    if result.analysis_notes:
        with st.expander("📋 Szczegóły Analizy", expanded=True):
            for note in result.analysis_notes:
                if "SmO2" in note or "wykryto" in note.lower() or "step" in note.lower():  
                     st.info(note)

            if result.steps_analyzed > 0:
                st.metric("Liczba wykrytych stopni", result.steps_analyzed)
            
            # Show SmO2 slope table
            if result.step_smo2_analysis:
                st.markdown("### 📊 SmO2 Slope per step (debug)")
                step_df = pd.DataFrame(result.step_smo2_analysis)
                
                # Format
                if 'slope' in step_df.columns:
                    step_df['slope'] = step_df['slope'].apply(lambda x: f"{x:.4f}")
                if 'avg_power' in step_df.columns:
                    step_df['avg_power'] = step_df['avg_power'].apply(lambda x: f"{x:.0f}W")
                if 'avg_hr' in step_df.columns:
                    step_df['avg_hr'] = step_df['avg_hr'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
                if 'avg_smo2' in step_df.columns:
                    step_df['avg_smo2'] = step_df['avg_smo2'].apply(lambda x: f"{x:.1f}%")
                
                # Markers
                if 'is_t1' in step_df.columns:
                     step_df['Marker'] = step_df.apply(
                        lambda r: '🟢 SmO2 T1' if r.get('is_t1') else ('🔴 SmO2 T2' if r.get('is_t2') else ''), 
                        axis=1
                    )
                
                # Rename
                step_df = step_df.rename(columns={
                    'step_number': 'Step',
                    'avg_power': 'Power',
                    'avg_hr': 'HR',
                    'avg_smo2': 'SmO2',
                    'slope': 'Slope (%/s)'
                })
                
                cols = ['Step', 'Power', 'HR', 'SmO2', 'Slope (%/s)', 'Marker']
                cols_av = [c for c in cols if c in step_df.columns]
                st.dataframe(step_df[cols_av], hide_index=True)

    # Cards
    st.subheader("🤖 Wykryte Progi SmO2")
    col_z1, col_z2 = st.columns(2)
    
    with col_z1:
        if t1_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #2ca02c; background-color: #222;">
                <h3 style="margin:0; color: #2ca02c;">SmO2 T1 (Początek Desaturacji)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(t1_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(t1_hr)} bpm</p>' if t1_hr else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                 st.caption(f"~{(t1_watts/cp_input)*100:.0f}% CP")
        else:
            st.info("T1: Nie wykryto")

    with col_z2:
        if t2_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #d62728; background-color: #222;">
                <h3 style="margin:0; color: #d62728;">SmO2 T2 (Szybka Desaturacja)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(t2_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(t2_hr)} bpm</p>' if t2_hr else ''}
            </div>
            """, unsafe_allow_html=True)
            if cp_input > 0:
                 st.caption(f"~{(t2_watts/cp_input)*100:.0f}% CP")
        else:
            st.info("T2: Nie wykryto")

    st.markdown("---")

    # 3. Chart
    st.subheader("📈 Wizualizacja SmO2")
    fig_thresh = go.Figure()

    # SmO2 Trace
    fig_thresh.add_trace(go.Scatter(
        x=target_df['time'], y=target_df['smo2_smooth'],
        customdata=target_df['time_str'],
        mode='lines', name='SmO2 (%)',
        line=dict(color='#2ca02c', width=2),
        hovertemplate="<b>Czas:</b> %{customdata}<br><b>SmO2:</b> %{y:.1f}%<extra></extra>"
    ))

    # Power Trace
    if 'watts_smooth_5s' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['watts_smooth_5s'],
            customdata=target_df['time_str'],
            mode='lines', name='Power',
            line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3,
            hovertemplate="<b>Moc:</b> %{y:.0f} W<extra></extra>"
        ))

    # HR Trace
    if 'hr' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['hr'],
            customdata=target_df['time_str'],
            mode='lines', name='HR',
            line=dict(color='#d62728', width=1, dash='dot'),
            yaxis='y2', opacity=0.5,
            hovertemplate="<b>HR:</b> %{y:.0f} bpm<extra></extra>"
        ))

    # Markers
    if result.step_smo2_analysis:
        for step in result.step_smo2_analysis:
            is_t1 = step.get('is_t1', False)
            is_t2 = step.get('is_t2', False)
            
            if is_t1 or is_t2:
                power = step.get('avg_power', 0)
                hr = step.get('avg_hr')
                end_time = step.get('end_time', 0)
                
                label = "T1" if is_t1 else "T2"
                color = "#2ca02c" if is_t1 else "#d62728"
                hr_str = f"{int(hr)}" if hr is not None else "--"
                
                fig_thresh.add_vline(
                    x=end_time,
                    line=dict(color=color, width=2, dash="dash"),
                    annotation_text=f"<b>SmO2 {label}</b><br>{int(power)}W @ {hr_str} bpm",
                    annotation_position="top left",
                    annotation_font=dict(color=color, size=12),
                    layer="above"
                )

    fig_thresh.update_layout(
        title="Dynamika SmO2 z Progami",
        xaxis_title="Czas",
        yaxis=dict(title=dict(text="SmO2 (%)", font=dict(color="#2ca02c"))),
        yaxis2=dict(title=dict(text="Moc (W) / HR", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.01),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_thresh, use_container_width=True)
