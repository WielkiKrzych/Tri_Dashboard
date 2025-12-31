import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats
from modules.calculations.kinetics import (
    normalize_smo2_series, 
    detect_smo2_trend, 
    classify_smo2_context, 
    calculate_resaturation_metrics,
    analyze_temporal_sequence,
    generate_state_timeline
)

def render_smo2_tab(target_df, training_notes, uploaded_file_name):
    st.header("Analiza Kinetyki SmO2 (State & Lag)")
    st.markdown("Analiza stanów fizjologicznych, opóźnień (lag) oraz kontekstu obciążenia.")

    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    if 'time' not in target_df.columns:
        st.error("Brak kolumny 'time' w danych!")
        st.stop()

    # Ensure smoothed columns exist if not already present
    from modules.calculations.quality import check_signal_quality
    
    if 'smo2' in target_df.columns:
        # Check Quality
        qual_res = check_signal_quality(target_df['smo2'], "SmO2", (0, 100))
        if not qual_res['is_valid']:
            st.warning(f"⚠️ **Niska Jakość Sygnału SmO2 (Score: {qual_res['score']})**")
            for issue in qual_res['issues']:
                st.caption(f"❌ {issue}")
            # Could mask metrics here

    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    if 'smo2_smooth' not in target_df.columns and 'smo2' in target_df.columns:
        target_df['smo2_smooth'] = target_df['smo2'].rolling(window=5, center=True).mean()
        
    # NORMALIZE SmO2 for Session
    if 'smo2_norm' not in target_df.columns and 'smo2_smooth' in target_df.columns:
        target_df['smo2_norm'] = normalize_smo2_series(target_df['smo2_smooth']) * 100.0 # Scale to 0-100% relative
        
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')
    
    col_inp1, col_inp2 = st.columns(2)
    
    # Inicjalizacja session_state
    if 'smo2_start_sec' not in st.session_state:
        st.session_state.smo2_start_sec = 600
    if 'smo2_end_sec' not in st.session_state:
        st.session_state.smo2_end_sec = 1200
        
    # ===== NOTATKI SmO2 =====
    with st.expander("📝 Dodaj Notatkę do tej Analizy", expanded=False):
        note_col1, note_col2 = st.columns([1, 2])
        with note_col1:
            note_time = st.number_input(
                "Czas (min)", 
                min_value=0.0, 
                max_value=float(len(target_df)/60) if len(target_df) > 0 else 60.0,
                value=float(len(target_df)/120) if len(target_df) > 0 else 15.0,
                step=0.5,
                key="smo2_note_time"
            )
        with note_col2:
            note_text = st.text_input(
                "Notatka",
                key="smo2_note_text",
                placeholder="Np. 'Atak 500W', 'Próg beztlenowy', 'Błąd sensoryka'"
            )
        
        if st.button("➕ Dodaj Notatkę", key="smo2_add_note"):
            if note_text:
                training_notes.add_note(uploaded_file_name, note_time, "smo2", note_text)
                st.success(f"✅ Notatka: {note_text} @ {note_time:.1f} min")
            else:
                st.warning("Wpisz tekst notatki!")

    # Wyświetl istniejące notatki SmO2
    existing_notes_smo2 = training_notes.get_notes_for_metric(uploaded_file_name, "smo2")
    if existing_notes_smo2:
        st.subheader("📋 Notatki SmO2")
        for idx, note in enumerate(existing_notes_smo2):
            col_note, col_del = st.columns([4, 1])
            with col_note:
                st.info(f"⏱️ **{note['time_minute']:.1f} min** | {note['text']}")
            with col_del:
                if st.button("🗑️", key=f"del_smo2_note_{idx}"):
                    training_notes.delete_note(uploaded_file_name, idx)
                    st.rerun()

    st.markdown("---")
    # ===== KONIEC NOTATEK SmO2 =====

    st.info("Algorytm wykrywa stany fizjologiczne (Gantt) oraz analizuje zaznaczony fragment.")
    
    # ===== STATE TIMELINE GANTT =====
    # Generate timeline if not in cache (could cache later) or just compute fast
    if 'smo2' in target_df.columns:
        with st.expander("📊 Oś Czasu Stanów Fizjologicznych (State Timeline)", expanded=False):
            # Compute timeline
            timeline = generate_state_timeline(target_df, window_size_sec=30, step_sec=10)
            
            if timeline:
                fig_gantt = go.Figure()
                
                # Colors based on state
                color_map = {
                    "RECOVERY": "green",
                    "STEADY_STATE": "blue",
                    "NON_STEADY": "orange",
                    "FATIGUE": "red",
                    "RAMP_UP": "purple",
                    "UNKNOWN": "grey"
                }

                for segment in timeline:
                    # Parse start/end to datetime for Plotly Gantt usually, or use linear x
                    # Using Bar chart (horizontal) or Shapes
                    
                    fig_gantt.add_trace(go.Bar(
                        x=[segment['end'] - segment['start']],
                        y=["State"],
                        base=[segment['start']],
                        orientation='h',
                        marker=dict(color=color_map.get(segment['state'], 'grey')),
                        name=segment['state'],
                        hovertemplate=f"<b>{segment['state']}</b><br>Conf: {segment['confidence']:.2f}<extra></extra>",
                        showlegend=False
                    ))
                
                # Deduplicate legend items manually if desired, but here we hide legend for clutter
                # Add Legend manually?
                for state, color in color_map.items():
                     fig_gantt.add_trace(go.Bar(x=[0], y=["State"], marker=dict(color=color), name=state, visible='legendonly'))

                fig_gantt.update_layout(
                    title="Przebieg Treningu (Physiological States)",
                    xaxis_title="Czas (s)",
                    barmode='stack', # Stack keeps them in single line
                    height=150,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=True
                )
                st.plotly_chart(fig_gantt, use_container_width=True)


    # Użyj wartości z session_state
    startsec = st.session_state.smo2_start_sec
    endsec = st.session_state.smo2_end_sec
    
    # Parsowanie czasu
    def format_time(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h}:{m:02d}:{sec:02d}"

    if startsec is not None and endsec is not None:
        if endsec > startsec:
            duration_sec = endsec - startsec
            
            mask = (target_df['time'] >= startsec) & (target_df['time'] <= endsec)
            interval_data = target_df.loc[mask]

            if not interval_data.empty:
                avg_watts = interval_data['watts'].mean() if 'watts' in interval_data.columns else 0
                avg_norm = interval_data['smo2_norm'].mean() if 'smo2_norm' in interval_data.columns else 0
                
                # Trend Classification & Context
                is_recovery = False
                resat_result = {}
                lag_results = {}
                
                if len(interval_data) > 30 and 'smo2_norm' in interval_data.columns:
                    # 1. Detect Base SmO2 Trend
                    trend_res = detect_smo2_trend(interval_data['time'], interval_data['smo2_norm'])
                    slope = trend_res['slope']
                    trend_cat = trend_res['category']
                    
                    # 2. Check for Recovery Mode
                    if "Reoxygenation" in trend_cat:
                        is_recovery = True
                        resat_result = calculate_resaturation_metrics(interval_data['time'], interval_data['smo2_norm'])
                        context_res = classify_smo2_context(interval_data, trend_res)
                    else:
                        # 3. Classify Context (Why?) if not recovery
                        context_res = classify_smo2_context(interval_data, trend_res)
                    
                    cause = context_res.get('cause', 'Unknown')
                    explanation = context_res.get('explanation', '')
                    c_type = context_res.get('type', 'normal')
                    
                    # 4. Lag Analysis (Cross-Correlation)
                    lag_results = analyze_temporal_sequence(interval_data)
                    
                else:
                    trend_res = {}
                    slope = 0
                    trend_cat = "N/A"
                    cause = "N/A"
                    explanation = ""
                    c_type = "normal"

                st.subheader(f"Metryki: {format_time(startsec)} - {format_time(endsec)}")
                
                # Dynamic Metric Layout based on context
                if is_recovery:
                    # RECOVERY MODE UI
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Kinetic State", "RESATURATION", delta="Recovery Mode", delta_color="inverse")
                    
                    c2.metric("T½ (Half-Recovery)", f"{resat_result.get('t_half', 0):.1f} s")
                    
                    tau = resat_result.get('tau_est', 0)
                    score = resat_result.get('recovery_score', 0)
                    c3.metric("Est. Tau", f"{tau:.1f} s", delta=f"Score: {score:.0f}/100")
                    
                    rate = resat_result.get('resat_rate', 0) * 100 
                    c4.metric("Resat Rate", f"{rate:.2f} %/s")
                    
                    st.success(f"**Analiza Regeneracji:** Szybkość odbudowy (Tau={tau:.1f}s) wskazuje na tempo resyntezy PCr.")
                    
                else:
                    # STANDARD CONTEXT UI
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Śr. Moc", f"{avg_watts:.0f} W")
                    c1.caption(f"Śr. Kadencja: {interval_data['cadence'].mean():.0f} rpm")
                    
                    c2.metric("Kinetic State", trend_cat, delta=f"{slope:.4f}/s")
                    
                    # Context Metric
                    if c_type == 'mechanical':
                        c3.metric("Przyczyna (Inferred)", f"⚙️ {cause}", delta="Sprawdź Kadencję", delta_color="inverse")
                    elif c_type == 'limit':
                        c3.metric("Przyczyna (Inferred)", f"🛑 {cause}", delta="Limit Dostaw", delta_color="inverse")
                    elif c_type == 'warning':
                        c3.metric("Przyczyna (Inferred)", f"⚠️ {cause}", delta="FATIGUE ALERT", delta_color="inverse")
                    else:
                        c3.metric("Przyczyna (Inferred)", f"✅ {cause}")
                        
                    c4.metric("Rel SmO2", f"{avg_norm:.1f} %")

                    if explanation:
                        if c_type in ['mechanical', 'limit', 'warning']:
                            st.warning(f"**Wyjaśnienie Algorytmu:** {explanation}")
                        else:
                            st.info(f"**Wyjaśnienie Algorytmu:** {explanation}")
                            
                # ===== LAG ANALYSIS VISUALIZATION =====
                if lag_results:
                    with st.expander("⏱️ Analiza Opóźnień (Kinetyska Sekwencja)", expanded=False):
                        st.markdown("Opóźnienie reakcji względem zmiany obciążenia (Moc = 0s).")
                        
                        lag_hr = lag_results.get('hr_lag', 0)
                        lag_smo2 = lag_results.get('smo2_lag', 0)
                        
                        lag_col1, lag_col2 = st.columns([1, 1])
                        
                        # Visualization of timeline
                        # Simple ASCII or metrics
                        lag_col1.metric("SmO2 Lag", f"{lag_smo2:.1f} s", 
                                        delta="OK" if lag_smo2 < 40 else "Slow", 
                                        delta_color="inverse" if lag_smo2 > 40 else "normal")
                        lag_col2.metric("HR Lag", f"{lag_hr:.1f} s",
                                        delta="Lagging Indicator" if lag_hr > 60 else "OK",
                                        delta_color="inverse" if lag_hr > 60 else "normal")
                        
                        # Timeline visualization
                        fig_lag = go.Figure()
                        
                        metrics = ['Power (Trigger)', 'SmO2 Response', 'HR Response']
                        lags = [0, lag_smo2, lag_hr]
                        colors = ['blue', 'red', 'lightgreen']
                        
                        fig_lag.add_trace(go.Bar(
                            y=metrics,
                            x=lags,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{l:.1f}s" for l in lags],
                            textposition='auto'
                        ))
                        
                        fig_lag.update_layout(
                            title="Sekwencja Czasowa Reakcji (Time to Response)",
                            xaxis_title="Opóźnienie (s)",
                            height=200,
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig_lag, use_container_width=True)

                fig_smo2 = go.Figure()

                # Add Normalized Trace (Primary)
                if 'smo2_norm' in target_df.columns:
                    fig_smo2.add_trace(go.Scatter(
                        x=target_df['time'], 
                        y=target_df['smo2_norm'],
                        customdata=target_df['time_str'],
                        mode='lines', 
                        name='Relative SmO2 (Norm)',
                        line=dict(color='#FF4B4B', width=3), # Bold Red
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Rel SmO2:</b> %{y:.1f}%<extra></extra>"
                    ))

                fig_smo2.add_trace(go.Scatter(
                    x=target_df['time'], 
                    y=target_df['watts_smooth_5s'],
                    customdata=target_df['time_str'],
                    mode='lines', 
                    name='Power',
                    line=dict(color='#1f77b4', width=1),
                    yaxis='y2',
                    opacity=0.3,
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
                ))
                
                # Add Cadence trace to see grinding
                if 'cadence' in target_df.columns:
                     fig_smo2.add_trace(go.Scatter(
                        x=target_df['time'], 
                        y=target_df['cadence'],
                        customdata=target_df['time_str'],
                        mode='lines', 
                        name='Cadence',
                        line=dict(color='orange', width=1, dash='dot'),
                        yaxis='y3',
                        visible='legendonly',
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>RPM:</b> %{y:.0f}<extra></extra>"
                    ))

                fig_smo2.add_vrect(
                    x0=startsec, x1=endsec,
                    fillcolor="blue" if is_recovery else "green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="RECOVERY" if is_recovery else "LOADING", annotation_position="top left"
                )

                fig_smo2.update_layout(
                    title="Analiza Relatywna SmO2 (Context & Recovery)",
                    xaxis_title="Czas",
                    yaxis=dict(title="Relative SmO2 (%)", range=[0, 100]),
                    yaxis2=dict(title="Power (W)", overlaying='y', side='right', showgrid=False),
                    yaxis3=dict(title="RPM", overlaying='y', side='right', anchor='free', position=1.0, showgrid=False, range=[0, 150]),
                    legend=dict(x=0.01, y=0.99),
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )
                
                # Wykres z interaktywnym zaznaczaniem
                selected = st.plotly_chart(fig_smo2, use_container_width=True, key="smo2_chart", on_select="rerun", selection_mode="box")

                # Obsługa zaznaczenia
                if selected and 'selection' in selected and 'box' in selected['selection']:
                    box_data = selected['selection']['box']
                    if box_data and len(box_data) > 0:
                        # Pobierz zakres X (czas) z zaznaczenia
                        x_range = box_data[0].get('x', [])
                        if len(x_range) == 2:
                            new_start = min(x_range)
                            new_end = max(x_range)
                            
                            # Aktualizuj session_state
                            if new_start != st.session_state.smo2_start_sec or new_end != st.session_state.smo2_end_sec:
                                st.session_state.smo2_start_sec = new_start
                                st.session_state.smo2_end_sec = new_end
                                st.rerun()
                                
            else:
                st.warning("Brak danych w wybranym zakresie.")
