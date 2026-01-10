import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.calculations.smo2_advanced import detect_smo2_thresholds_moxy


def render_smo2_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input):
    """Ramp Test SmO₂ threshold detection (T1 + T2_onset only)."""
    st.header("🎯 SmO₂ Ramp Test Thresholds")
    st.markdown("""
    **Ramp Test Protocol:** Only **T1** and **T2_onset** are valid.  
    T2_steady (MLSS) is **not detectable** in ramp tests.
    """)

    if target_df is None or target_df.empty:
        st.error("Brak danych.")
        st.stop()
    
    target_df = target_df.copy()
    target_df.columns = target_df.columns.str.lower().str.strip()

    if 'hr' not in target_df.columns:
        for alias in ['heart_rate', 'bpm', 'heartrate']:
            if alias in target_df.columns:
                target_df.rename(columns={alias: 'hr'}, inplace=True)
                break
    
    if 'smo2' not in target_df.columns:
        st.error("Brak SmO2!")
        st.stop()

    if 'time' not in target_df.columns:
        st.error("Brak czasu!")
        st.stop()

    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    
    target_df['smo2_smooth'] = target_df['smo2'].rolling(window=30, center=True).median()
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # =========================================================================
    # DETEKCJA
    # =========================================================================
    
    cpet_result = st.session_state.get('cpet_vt_result', {})
    vt1_watts = cpet_result.get('vt1_onset_watts') or cpet_result.get('vt1_watts')
    rcp_onset = cpet_result.get('rcp_onset_watts') or cpet_result.get('vt2_watts')
    hr_max = int(target_df['hr'].max()) if 'hr' in target_df.columns else None
    
    with st.spinner("Analiza SmO₂..."):
        result = detect_smo2_thresholds_moxy(
            df=target_df,
            step_duration_sec=180,
            smo2_col='smo2',
            power_col='watts',
            hr_col='hr' if 'hr' in target_df.columns else None,
            time_col='time',
            cp_watts=cp_input if cp_input > 0 else None,
            hr_max=hr_max,
            vt1_watts=vt1_watts,
            rcp_onset_watts=rcp_onset
        )
    
    st.markdown("---")

    # =========================================================================
    # KARTY PROGÓW (tylko T1 i T2_onset)
    # =========================================================================
    
    st.subheader("🎯 Detected Thresholds")
    col1, col2 = st.columns(2)
    
    with col1:
        if result.t1_watts:
            st.markdown(f"""
            <div style="padding:12px; border-radius:8px; border:2px solid #2ca02c; background:#1a1a1a;">
                <h4 style="margin:0; color:#2ca02c;">🟢 SmO₂ T1 (LT1)</h4>
                <h2 style="margin:5px 0;">{int(result.t1_watts)} W</h2>
                <p style="margin:0; color:#aaa;">SmO₂: {result.t1_smo2}% | HR: {result.t1_hr or '--'}</p>
                <p style="margin:0; color:#666;">slope: {result.t1_trend}%/min</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("T1: Not detected")

    with col2:
        if result.t2_onset_watts:
            st.markdown(f"""
            <div style="padding:12px; border-radius:8px; border:2px solid #d62728; background:#1a1a1a;">
                <h4 style="margin:0; color:#d62728;">🔴 SmO₂ T2_onset (RCP)</h4>
                <h2 style="margin:5px 0;">{int(result.t2_onset_watts)} W</h2>
                <p style="margin:0; color:#aaa;">SmO₂: {result.t2_onset_smo2}% | HR: {result.t2_onset_hr or '--'}</p>
                <p style="margin:0; color:#666;">curv: {result.t2_onset_curvature}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("T2_onset: Not detected")

    # T2_steady info
    st.info("ℹ️ **T2_steady (MLSS):** N/A in ramp test (requires ≥3min plateau)")

    # =========================================================================
    # CONFIDENCE + AGREEMENT
    # =========================================================================
    
    st.markdown("---")
    agreement = result.physiological_agreement
    
    # Extract confidence from notes
    confidence = 0
    for note in result.analysis_notes:
        if "confidence:" in note.lower():
            try:
                confidence = int(note.split("confidence:")[1].split("%")[0].strip())
            except:
                pass
    
    if agreement == "high":
        st.success(f"🟢 **High systemic agreement** — Confidence: {confidence}%")
    elif agreement == "moderate":
        st.warning(f"🟡 **Moderate agreement** — Confidence: {confidence}%")
    else:
        st.error(f"🔴 **Local Perfusion Limitation** — Confidence: {confidence}%")

    # =========================================================================
    # PANEL DIAGNOSTYCZNY
    # =========================================================================
    
    if result.step_data:
        with st.expander("📊 Diagnostic Panel", expanded=False):
            diag_df = pd.DataFrame(result.step_data)
            
            cols = ['step', 'power', 'smo2', 'trend', 'cv', 'osc_amp', 'curvature']
            if 'hr' in diag_df.columns:
                cols.insert(3, 'hr')
            
            available = [c for c in cols if c in diag_df.columns]
            diag = diag_df[available].round({'power': 0, 'smo2': 1, 'trend': 2, 'cv': 1, 'osc_amp': 1, 'curvature': 5})
            
            rename = {
                'step': 'Step', 'power': 'Power', 'smo2': 'SmO₂', 'hr': 'HR',
                'trend': 'Slope', 'cv': 'CV%', 'osc_amp': 'Osc', 'curvature': 'Curv'
            }
            
            st.dataframe(diag.rename(columns=rename), use_container_width=True, hide_index=True)

    # =========================================================================
    # STREFY
    # =========================================================================
    
    if result.zones:
        st.subheader("🎯 SmO₂ Zones")
        zone_data = [{
            "Zone": f"Z{z['zone']}",
            "Name": z['name'],
            "Power": f"{z['power_min']}–{z['power_max']}W",
            "Description": z['description']
        } for z in result.zones]
        st.dataframe(pd.DataFrame(zone_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # ANALIZA
    # =========================================================================
    
    if result.analysis_notes:
        with st.expander("📋 Analysis Log", expanded=False):
            for note in result.analysis_notes:
                if "✓" in note or "🟢" in note:
                    st.success(note)
                elif "❌" in note or "🔴" in note:
                    st.error(note)
                elif "⚠" in note or "🟡" in note:
                    st.warning(note)
                else:
                    st.info(note)

    st.markdown("---")

    # =========================================================================
    # WYKRES
    # =========================================================================
    
    st.subheader("📈 Visualization")
    fig = go.Figure()

    def get_time_at_power(power_val):
        if power_val and 'watts' in target_df.columns:
            mask = target_df['watts'] >= power_val
            if mask.any():
                return target_df.loc[mask, 'time'].iloc[0]
        return None
    
    t1_time = get_time_at_power(result.t1_watts)
    t2o_time = get_time_at_power(result.t2_onset_watts)
    min_time = target_df['time'].min()
    max_time = target_df['time'].max()

    # Domain shading
    if t1_time:
        fig.add_vrect(x0=min_time, x1=t1_time, fillcolor="rgba(44,160,44,0.12)", line_width=0,
                      annotation_text="Z1: Stable", annotation_position="top left")
    if t1_time and t2o_time:
        fig.add_vrect(x0=t1_time, x1=t2o_time, fillcolor="rgba(255,200,0,0.12)", line_width=0,
                      annotation_text="Z2: Heavy", annotation_position="top left")
    if t2o_time:
        fig.add_vrect(x0=t2o_time, x1=max_time, fillcolor="rgba(214,39,40,0.12)", line_width=0,
                      annotation_text="Z3: Severe", annotation_position="top left")

    # Traces
    fig.add_trace(go.Scatter(
        x=target_df['time'], y=target_df['smo2_smooth'],
        customdata=target_df['time_str'],
        mode='lines', name='SmO₂',
        line=dict(color='#2ca02c', width=2),
        hovertemplate="<b>%{customdata}</b><br>SmO₂: %{y:.1f}%<extra></extra>"
    ))

    if 'watts_smooth_5s' in target_df.columns:
        fig.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['watts_smooth_5s'],
            mode='lines', name='Power', line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3
        ))

    if 'hr' in target_df.columns:
        fig.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['hr'],
            mode='lines', name='HR', line=dict(color='#d62728', width=1, dash='dot'),
            yaxis='y2', opacity=0.5
        ))
    
    # Threshold lines
    if t1_time and result.t1_watts:
        fig.add_vline(x=t1_time, line=dict(color="#2ca02c", width=2, dash="dash"))
        fig.add_annotation(x=t1_time, y=1, yref="paper",
            text=f"<b>T1</b> {result.t1_watts}W",
            showarrow=False, font=dict(color="white", size=9),
            bgcolor="rgba(44,160,44,0.8)", borderpad=2)

    if t2o_time and result.t2_onset_watts:
        fig.add_vline(x=t2o_time, line=dict(color="#d62728", width=2, dash="dash"))
        fig.add_annotation(x=t2o_time, y=0.9, yref="paper",
            text=f"<b>T2</b> {result.t2_onset_watts}W",
            showarrow=False, font=dict(color="white", size=9),
            bgcolor="rgba(214,39,40,0.8)", borderpad=2)

    fig.update_layout(
        title="SmO₂ with Domain Shading",
        xaxis_title="Time",
        yaxis=dict(title="SmO₂ (%)"),
        yaxis2=dict(title="Power/HR", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.01),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # VALIDATION SUMMARY
    # =========================================================================
    
    st.subheader("📄 Validation Summary")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Thresholds:**")
        if result.t1_watts:
            st.write(f"T1: {result.t1_watts}W @ {result.t1_hr or '--'}bpm")
        if result.t2_onset_watts:
            st.write(f"T2: {result.t2_onset_watts}W @ {result.t2_onset_hr or '--'}bpm")
    
    with c2:
        st.markdown("**% CP:**")
        if cp_input > 0:
            if result.t1_watts:
                st.write(f"T1: {(result.t1_watts/cp_input)*100:.0f}% CP")
            if result.t2_onset_watts:
                st.write(f"T2: {(result.t2_onset_watts/cp_input)*100:.0f}% CP")
    
    with c3:
        st.markdown("**CPET Validation:**")
        if result.vt1_correlation_watts is not None:
            st.write(f"T1 vs VT1: ±{result.vt1_correlation_watts}W")
        if result.rcp_onset_correlation_watts is not None:
            st.write(f"T2 vs VT2: ±{result.rcp_onset_correlation_watts}W")
