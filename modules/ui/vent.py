import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats
from modules.calculations.thresholds import analyze_step_test

def render_vent_tab(target_df, training_notes, uploaded_file_name):
    st.header("Analiza Progu Wentylacyjnego (VT1 / VT2 Detection)")
    st.markdown("Analiza dynamiki oddechu. Szukamy nieliniowych przyrostów wentylacji (VE) względem mocy.")

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych.")
        st.stop()

    if 'time' not in target_df.columns or 'tymeventilation' not in target_df.columns:
        st.error("Brak danych wentylacji (tymeventilation) lub czasu!")
        st.stop()

    # Wygładzanie (VE jest szumiące, dajemy 10s smooth)
    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    target_df['ve_smooth'] = target_df['tymeventilation'].rolling(window=10, center=True).mean()
    target_df['rr_smooth'] = target_df['tymebreathrate'].rolling(window=10, center=True).mean() if 'tymebreathrate' in target_df else 0
    
    # Format czasu
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # 2. DETEKCJA AUTOMATYCZNA (Full Analysis with Sensitivity)
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

    # Wyświetlenie wyników automatycznych (Primary - Increasing Load)
    st.subheader("🤖 Automatyczna Detekcja Stref (Ramp Up)")
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        if vt1_zone:
            st.success(f"**VT1 Zone (Up):** {vt1_zone.range_watts[0]:.0f} - {vt1_zone.range_watts[1]:.0f} W")
            st.caption(f"Confidence: {vt1_zone.confidence:.0%}")
            
            # Reliability Badge
            if sensitivity and sensitivity.vt1_stability_score > 0.8:
                st.caption(f"🛡️ Reliability: :green[HIGH] (Var: {sensitivity.vt1_variability_watts:.1f}W)")
            elif sensitivity and sensitivity.vt1_stability_score > 0.5:
                 st.caption(f"🛡️ Reliability: :orange[MEDIUM] (Var: {sensitivity.vt1_variability_watts:.1f}W)")
            elif sensitivity:
                 st.caption(f"🛡️ Reliability: :red[LOW] (Var: {sensitivity.vt1_variability_watts:.1f}W)")
            
            if vt1_zone.range_hr:
                st.caption(f"HR: {vt1_zone.range_hr[0]:.0f}-{vt1_zone.range_hr[1]:.0f} bpm")
        else:
            st.info("VT1 Zone: Nie wykryto (brak wyraźnego przejścia slope 0.05)")

    with col_z2:
        if vt2_zone:
            st.error(f"**VT2 Zone (Up):** {vt2_zone.range_watts[0]:.0f} - {vt2_zone.range_watts[1]:.0f} W")
            st.caption(f"Confidence: {vt2_zone.confidence:.0%}")
            
            if sensitivity and sensitivity.vt2_stability_score > 0.8:
                st.caption(f"🛡️ Reliability: :green[HIGH] (Var: {sensitivity.vt2_variability_watts:.1f}W)")
            elif sensitivity and sensitivity.vt2_stability_score > 0.5:
                 st.caption(f"🛡️ Reliability: :orange[MEDIUM] (Var: {sensitivity.vt2_variability_watts:.1f}W)")
            elif sensitivity:
                 st.caption(f"🛡️ Reliability: :red[LOW] (Var: {sensitivity.vt2_variability_watts:.1f}W)")
                 
            if vt2_zone.range_hr:
                st.caption(f"HR: {vt2_zone.range_hr[0]:.0f}-{vt2_zone.range_hr[1]:.0f} bpm")
        else:
            st.info("VT2 Zone: Nie wykryto (brak wyraźnego przejścia slope 0.15)")
            
    # Hysteresis Information
    if hysteresis and (hysteresis.vt1_dec_zone or hysteresis.vt2_dec_zone):
        with st.expander("📉 Analiza Histerezy (Ramp Down vs Up)", expanded=False):
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

    st.markdown("---")

    # 3. Interfejs Manualny (START -> KONIEC)
    # Inicjalizacja session_state dla zaznaczenia
    if 'vent_start_sec' not in st.session_state:
            st.session_state.vent_start_sec = 600  # 10 minut domyślnie
    if 'vent_end_sec' not in st.session_state:
            st.session_state.vent_end_sec = 1200  # 20 minut domyślnie
            
    # ===== NOTATKI VENTILATION =====
    with st.expander("📝 Dodaj Notatkę do tej Analizy", expanded=False):
        note_col1, note_col2 = st.columns([1, 2])
        with note_col1:
            note_time_vent = st.number_input(
                "Czas (min)", 
                min_value=0.0, 
                max_value=float(len(target_df)/60) if len(target_df) > 0 else 60.0,
                value=float(len(target_df)/120) if len(target_df) > 0 else 15.0,
                step=0.5,
                key="vent_note_time"
            )
        with note_col2:
            note_text_vent = st.text_input(
                "Notatka",
                key="vent_note_text",
                placeholder="Np. 'Próg beztlenowy', 'VE jump', 'Spłycenie oddechu'"
            )
        
        if st.button("➕ Dodaj Notatkę", key="vent_add_note"):
            if note_text_vent:
                training_notes.add_note(uploaded_file_name, note_time_vent, "ventilation", note_text_vent)
                st.success(f"✅ Notatka: {note_text_vent} @ {note_time_vent:.1f} min")
            else:
                st.warning("Wpisz tekst notatki!")

    # Wyświetl istniejące notatki Ventilation
    existing_notes_vent = training_notes.get_notes_for_metric(uploaded_file_name, "ventilation")
    if existing_notes_vent:
        st.subheader("📋 Notatki Wentylacji")
        for idx, note in enumerate(existing_notes_vent):
            col_note, col_del = st.columns([4, 1])
            with col_note:
                st.info(f"⏱️ **{note['time_minute']:.1f} min** | {note['text']}")
            with col_del:
                if st.button("🗑️", key=f"del_vent_note_{idx}"):
                    training_notes.delete_note(uploaded_file_name, idx)
                    st.rerun()

    st.markdown("---")
    # ===== KONIEC NOTATEK VENTILATION =====

    st.info("💡 **ANALIZA MANUALNA:** Zaznacz obszar na wykresie poniżej (kliknij i przeciągnij), aby sprawdzić nachylenie lokalne.")


    def parse_time_to_seconds(t_str):
        try:
            parts = list(map(int, t_str.split(':')))
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
            if len(parts) == 2: return parts[0]*60 + parts[1]
            if len(parts) == 1: return parts[0]
        except (ValueError, AttributeError):
            return None
        return None

    with st.expander("🔧 Ręczne wprowadzenie zakresu czasowego (opcjonalne)", expanded=False):
            col_inp_1, col_inp_2 = st.columns(2)
            with col_inp_1:
                manual_start = st.text_input("Start Interwału (hh:mm:ss)", value="01:00:00", key="vent_manual_start")
            with col_inp_2:
                manual_end = st.text_input("Koniec Interwału (hh:mm:ss)", value="01:20:00", key="vent_manual_end")

            if st.button("Zastosuj ręczny zakres", key="btn_vent_manual"):
                manual_start_sec = parse_time_to_seconds(manual_start)
                manual_end_sec = parse_time_to_seconds(manual_end)
                if manual_start_sec is not None and manual_end_sec is not None:
                    st.session_state.vent_start_sec = manual_start_sec
                    st.session_state.vent_end_sec = manual_end_sec
                    st.success(f"✅ Zaktualizowano zakres: {manual_start} - {manual_end}")

        # Użyj wartości z session_state
    startsec = st.session_state.vent_start_sec
    endsec = st.session_state.vent_end_sec

        
        # 3. Wycinanie
    mask_v = (target_df['time'] >= startsec) & (target_df['time'] <= endsec)
    interval_v = target_df.loc[mask_v]

    if not interval_v.empty:
            # 4. Obliczenia
            avg_w = interval_v['watts'].mean()
            avg_ve = interval_v['tymeventilation'].mean()
            avg_rr = interval_v['tymebreathrate'].mean() if 'tymebreathrate' in interval_v else 0
            
            # Trend (Slope) dla VE
            if len(interval_v) > 1:
                slope_ve, intercept_ve, _, _, _ = stats.linregress(interval_v['time'], interval_v['tymeventilation'])
                trend_desc_ve = f"{slope_ve:.4f} L/s"
            else:
                slope_ve = 0; intercept_ve = 0; trend_desc_ve = "N/A"

            # Formatowanie czasu dla wyświetlania
            def fmt_time_v(seconds):
                try:
                    seconds = int(seconds)
                    h = seconds // 3600
                    m = (seconds % 3600) // 60
                    s = seconds % 60
                    if h > 0:
                        return f"{h:02d}:{m:02d}:{s:02d}"
                    else:
                        return f"{m:02d}:{s:02d}"
                except (ValueError, TypeError):
                    return "-"
            start_time_v = fmt_time_v(startsec)
            end_time_v = fmt_time_v(endsec)
            duration_v = int(endsec - startsec) if (endsec is not None and startsec is not None) else 0

            # Metryki
            st.subheader(f"Metryki Manualne: {start_time_v} - {end_time_v} ({duration_v}s)")
            mv1, mv2, mv3, mv4 = st.columns(4)
            mv1.metric("Śr. Moc", f"{avg_w:.0f} W")
            mv2.metric("Śr. Wentylacja (VE)", f"{avg_ve:.1f} L/min")
            mv3.metric("Częstość (RR)", f"{avg_rr:.1f} /min")
            
            # Kolorowanie trendu
            trend_color = "inverse" if slope_ve > 0.1 else "normal"
            mv4.metric("Trend VE (Slope)", trend_desc_ve, delta=trend_desc_ve, delta_color=trend_color)

            # 5. Wykres
            fig_vent = go.Figure()

            # Lewa Oś: Wentylacja
            fig_vent.add_trace(go.Scatter(
                x=target_df['time'], y=target_df['ve_smooth'],
                customdata=target_df['time_str'],
                mode='lines', name='VE (L/min)',
                line=dict(color='#ffa15a', width=2),
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>"
            ))

            # Prawa Oś: Moc
            fig_vent.add_trace(go.Scatter(
                x=target_df['time'], y=target_df['watts_smooth_5s'],
                customdata=target_df['time_str'],
                mode='lines', name='Power',
                line=dict(color='#1f77b4', width=1),
                yaxis='y2', opacity=0.3,
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
            ))

            # === WIZUALIZACJA STREF (ZONES) ===
            
            # VT1 Primary (Green)
            if vt1_zone:
                fig_vent.add_hrect(
                    y0=vt1_zone.range_watts[0], y1=vt1_zone.range_watts[1],
                    fillcolor="green", opacity=0.15,
                    layer="below", line_width=0,
                    yref="y2",
                    annotation_text=f"VT1 Up ({vt1_zone.range_watts[0]:.0f}-{vt1_zone.range_watts[1]:.0f}W)", annotation_position="top left"
                )
            
            # VT2 Primary (Red)
            if vt2_zone:
                fig_vent.add_hrect(
                    y0=vt2_zone.range_watts[0], y1=vt2_zone.range_watts[1],
                    fillcolor="red", opacity=0.15,
                    layer="below", line_width=0,
                    yref="y2",
                    annotation_text=f"VT2 Up ({vt2_zone.range_watts[0]:.0f}-{vt2_zone.range_watts[1]:.0f}W)", annotation_position="top left"
                )
            
            # Hysteresis Zones (Decreasing) - Dashed/Ghost
            if hysteresis:
                if hysteresis.vt1_dec_zone:
                    fig_vent.add_hrect(
                       y0=hysteresis.vt1_dec_zone.range_watts[0], y1=hysteresis.vt1_dec_zone.range_watts[1],
                       fillcolor="blue", opacity=0.05, # Very faint
                       layer="below", line_width=1, line_dash="dot", line_color="blue",
                       yref="y2",
                       annotation_text="VT1 Down", annotation_position="bottom right"
                    )

                if hysteresis.vt2_dec_zone:
                    fig_vent.add_hrect(
                       y0=hysteresis.vt2_dec_zone.range_watts[0], y1=hysteresis.vt2_dec_zone.range_watts[1],
                       fillcolor="purple", opacity=0.05,
                       layer="below", line_width=1, line_dash="dot", line_color="purple",
                       yref="y2",
                       annotation_text="VT2 Down", annotation_position="bottom right"
                    )

            # Zaznaczenie manualne
            fig_vent.add_vrect(x0=startsec, x1=endsec, fillcolor="orange", opacity=0.1, layer="below", annotation_text="MANUAL", annotation_position="top left")

            # Linia trendu VE (dla manualnego)
            if len(interval_v) > 1:
                trend_line_ve = intercept_ve + slope_ve * interval_v['time']
                fig_vent.add_trace(go.Scatter(
                    x=interval_v['time'], y=trend_line_ve,
                    customdata=interval_v['time_str'],
                    mode='lines', name='Trend VE (Man)',
                    line=dict(color='white', width=2, dash='dash'),
                    hovertemplate="<b>Trend:</b> %{y:.2f} L/min<extra></extra>"
                ))

            fig_vent.update_layout(
                title="Dynamika Wentylacji vs Moc (z analiża histerezy)",
                xaxis_title="Czas",
                yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
                yaxis2=dict(title=dict(text="Moc (W)", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
                legend=dict(x=0.01, y=0.99),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            # Wykres z interaktywnym zaznaczaniem
            selected = st.plotly_chart(fig_vent, use_container_width=True, key="vent_chart", on_select="rerun", selection_mode="box")

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
                        if new_start != st.session_state.vent_start_sec or new_end != st.session_state.vent_end_sec:
                            st.session_state.vent_start_sec = new_start
                            st.session_state.vent_end_sec = new_end
                            st.rerun()

            # 6. TEORIA ODDECHOWA
            with st.expander("🫁 TEORIA: Płynne Strefy Przejścia vs Pojedynczy Punkt", expanded=False):
                st.markdown("""
                ### Dynamiczna Analiza
                
                System stosuje:
                1. **Sliding Window Analysis**: Skanuje okno po oknie, żeby znaleźć przejścia w nachyleniu (slope).
                2. **Histereza**: Porównuje fazę narastania ("Up") z fazą opadania/odpoczynku ("Down").
                3. **Sensitivity Analysis**: Uruchamia algorytm kilkukrotnie z różnymi oknami czasowymi (30s, 60s, 90s), aby sprawdzić stabilność wyniku.
                
                #### 🛡️ Reliability Score (Niezawodność)
                * **HIGH**: Wynik jest stabilny niezależnie od wygładzania.
                * **MEDIUM**: Wynik zależy nieco od parametrów.
                * **LOW**: Duża zmienność (>15W różnicy) w zależności od okna. Sugeruje "szumiący" sygnał lub nietypową fizjologię.
                """)
    else:
        st.warning("Brak danych w tym zakresie.")
