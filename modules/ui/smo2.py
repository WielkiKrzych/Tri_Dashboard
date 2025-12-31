import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats
from modules.calculations.kinetics import normalize_smo2_series, detect_smo2_trend

def render_smo2_tab(target_df, training_notes, uploaded_file_name):
    st.header("Analiza Kinetyki SmO2 (Relative Trends)")
    st.markdown("Analiza relatywna zmian saturacji mięśniowej. Ignorujemy wartości absolutne na rzecz trendów i normalizacji.")

    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    if 'time' not in target_df.columns:
        st.error("Brak kolumny 'time' w danych!")
        st.stop()

    # Ensure smoothed columns exist if not already present
    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    if 'smo2_smooth' not in target_df.columns and 'smo2' in target_df.columns:
        target_df['smo2_smooth'] = target_df['smo2'].rolling(window=5, center=True).mean()
        
    # NORMALIZE SmO2 for Session (This is the new feature!)
    if 'smo2_norm' not in target_df.columns and 'smo2_smooth' in target_df.columns:
        target_df['smo2_norm'] = normalize_smo2_series(target_df['smo2_smooth']) * 100.0 # Scale to 0-100% relative
        
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')
    
    col_inp1, col_inp2 = st.columns(2)
    
    # Inicjalizacja session_state dla zaznaczenia
    if 'smo2_start_sec' not in st.session_state:
        st.session_state.smo2_start_sec = 600  # 10 minut domyślnie
    if 'smo2_end_sec' not in st.session_state:
        st.session_state.smo2_end_sec = 1200  # 20 minut domyślnie
        
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

    st.info("💡 **ANALIZA MANUALNA:** Zaznacz obszar na wykresie poniżej, aby obliczyć trend względny.")

    # Opcjonalne: ręczne wprowadzenie czasu (dla precyzji)
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
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            manual_start = st.text_input("Start Interwału (hh:mm:ss)", value="01:00:00", key="smo2_manual_start")
        with col_inp2:
            manual_end = st.text_input("Koniec Interwału (hh:mm:ss)", value="01:20:00", key="smo2_manual_end")
        
        if st.button("Zastosuj ręczny zakres"):
            manual_start_sec = parse_time_to_seconds(manual_start)
            manual_end_sec = parse_time_to_seconds(manual_end)
            if manual_start_sec is not None and manual_end_sec is not None:
                st.session_state.smo2_start_sec = manual_start_sec
                st.session_state.smo2_end_sec = manual_end_sec
                st.success(f"✅ Zaktualizowano zakres: {manual_start} - {manual_end}")

    # Użyj wartości z session_state
    startsec = st.session_state.smo2_start_sec
    endsec = st.session_state.smo2_end_sec

    start_time_str = st.session_state.get('smo2_manual_start', "01:00:00")
    nd_time_str = st.session_state.get('smo2_manual_end', "01:20:00")
    
    if startsec is not None and endsec is not None:
        if endsec > startsec:
            duration_sec = endsec - startsec
            
            mask = (target_df['time'] >= startsec) & (target_df['time'] <= endsec)
            interval_data = target_df.loc[mask]

            if not interval_data.empty:
                avg_watts = interval_data['watts'].mean() if 'watts' in interval_data.columns else 0
                avg_smo2 = interval_data['smo2'].mean() if 'smo2' in interval_data.columns else 0
                avg_norm = interval_data['smo2_norm'].mean() if 'smo2_norm' in interval_data.columns else 0
                
                # Trend Detection (Relative)
                if len(interval_data) > 10 and 'smo2_norm' in interval_data.columns:
                    trend_res = detect_smo2_trend(interval_data['time'], interval_data['smo2_norm'])
                    slope = trend_res['slope']
                    trend_cat = trend_res['category']
                    trend_desc = trend_res['description']
                else:
                    trend_res = {}
                    slope = 0
                    trend_cat = "N/A"
                    trend_desc = "Za mało danych"

                st.subheader(f"Metryki dla odcinka: {start_time_str} - {nd_time_str} (Czas trwania: {duration_sec}s)")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Śr. Moc", f"{avg_watts:.0f} W")
                m2.metric("Raw SmO2", f"{avg_smo2:.1f} %")
                m3.metric("Rel SmO2", f"{avg_norm:.1f} %", help="Normalized to session Min-Max range (0-100%)")
                
                trend_color = "normal"
                if "Deox" in trend_cat: trend_color = "inverse"
                
                m4.metric("Kinetics Trend", trend_cat, delta=None) # String metric
                m5.metric("Slope (Rel)", f"{slope:.4f} /s", delta=trend_desc, delta_color=trend_color)

                # Wyświetl wykrytą strefę
                st.markdown("---")
                st.markdown("### 🎯 Interpretacja Kinetyczna")
                
                if "Reox" in trend_cat or "Stable" in trend_cat:
                     st.success(f"**{trend_cat}**\n\n{trend_desc}")
                elif "Rapid" in trend_cat:
                     st.error(f"**{trend_cat}**\n\n{trend_desc}")
                else:
                     st.warning(f"**{trend_cat}**\n\n{trend_desc}")

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

                # Add Absolute Trace (Secondary/Muted)
                if 'smo2_smooth' in target_df.columns:
                    fig_smo2.add_trace(go.Scatter(
                        x=target_df['time'], 
                        y=target_df['smo2_smooth'],
                        customdata=target_df['time_str'],
                        mode='lines', 
                        name='Raw SmO2 (Abs)',
                        line=dict(color='gray', width=1, dash='dot'), # Faint
                        visible='legendonly', # Hide by default to reduce clutter
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Raw SmO2:</b> %{y:.1f}%<extra></extra>"
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

                fig_smo2.add_vrect(
                    x0=startsec, x1=endsec,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="ANALIZA", annotation_position="top left"
                )
                
                # Draw Linear Slope on Normalized Data
                if len(interval_data) > 10 and 'smo2_norm' in interval_data.columns:
                     # Calculate linear fit for normalized data to show slope visually
                     slope_n, intercept_n, _, _, _ = stats.linregress(interval_data['time'], interval_data['smo2_norm'])
                     trend_line = intercept_n + slope_n * interval_data['time']
                     
                     if slope_n < -0.01: color = "red"
                     elif slope_n > 0.01: color = "green"
                     else: color = "white"
                     
                     fig_smo2.add_trace(go.Scatter(
                        x=interval_data['time'], 
                        y=trend_line,
                        customdata=interval_data['time_str'],
                        mode='lines', 
                        name='Trend (Rel)',
                        line=dict(color=color, width=3, dash='dash'),
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Trend:</b> %{y:.1f}%<extra></extra>"
                    ))

                fig_smo2.update_layout(
                    title="Analiza Relatywna SmO2 (Normalized 0-100%)",
                    xaxis_title="Czas",
                    yaxis=dict(title="Relative SmO2 (%)", range=[0, 100]),
                    yaxis2=dict(title="Power (W)", overlaying='y', side='right', showgrid=False),
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

                # --- PĘTLA HISTEREZY (SmO2 vs WATTS) ---
                st.divider()
                st.subheader("🔄 Pętla Histerezy (Relatywna)")
            
                if 'watts_smooth_5s' in interval_data.columns and 'smo2_norm' in interval_data.columns:
                    
                    fig_hyst = go.Figure()

                    fig_hyst.add_trace(go.Scatter(
                        x=interval_data['watts_smooth_5s'],
                        y=interval_data['smo2_norm'],
                        mode='markers+lines',
                        name='Histereza',
                        marker=dict(
                            size=6,
                            color=interval_data['time'], 
                            colorscale='Plasma',
                            showscale=True,
                            colorbar=dict(title="Upływ Czasu", tickmode="array", ticktext=["Start", "Koniec"], tickvals=[interval_data['time'].min(), interval_data['time'].max()])
                        ),
                        line=dict(color='rgba(255,255,255,0.3)', width=1), # Cienka linia łącząca
                        hovertemplate="<b>Moc:</b> %{x:.0f} W<br><b>Rel SmO2:</b> %{y:.1f}%<extra></extra>"
                    ))

                    start_pt = interval_data.iloc[0]
                    end_pt = interval_data.iloc[-1]

                    fig_hyst.add_annotation(
                        x=start_pt['watts_smooth_5s'], y=start_pt['smo2_norm'],
                        text="START", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="green"
                    )
                    fig_hyst.add_annotation(
                        x=end_pt['watts_smooth_5s'], y=end_pt['smo2_norm'],
                        text="META", showarrow=True, arrowhead=2, ax=0, ay=-40, bgcolor="red"
                    )

                    fig_hyst.update_layout(
                        template="plotly_dark",
                        title="Kinetyka Tlenowa: Relacja Moc vs Znormalizowane SmO2",
                        xaxis_title="Moc [W]",
                        yaxis_title="Rel SmO2 [%]",
                        height=600,
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="closest"
                    )

                    c_h1, c_h2 = st.columns([3, 1])
                    with c_h1:
                        st.plotly_chart(fig_hyst, use_container_width=True)
                    
                    with c_h2:
                        st.info("""
                        **📚 Zmiana Paradygmatu!**
                        
                        Używamy teraz **relatywnego SmO2** (0-100% dla tej sesji).
                        
                        * **Dlaczego?** Sensory MOXY/Humon mają różną kalibrację. To, czy masz 60% czy 40% absolutnie, nie ma znaczenia.
                        * **Co się liczy?** Jak bardzo spalasz tlen względem swojego **minimum sesyjnego**.
                        
                        **Interpretacja:**
                        * **Spadek krzywej:** Desaturacja (Przewaga zużycia)
                        * **Płasko:** Równowaga (Steady State)
                        * **Wzrost:** Regeneracja (Resynteza)
                        """)
                else:
                    st.warning("Brakuje wygładzonych danych mocy lub SmO2 dla tego interwału.")
                    
            else:
                st.warning("Brak danych w wybranym zakresie. Sprawdź poprawność wpisanego czasu.")
        else:
            st.error("Czas zakończenia musi być późniejszy niż czas rozpoczęcia!")
    else:
        st.warning("Wprowadź poprawne czasy w formacie h:mm:ss (np. 0:10:00).")
