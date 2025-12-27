import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats

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

    # 2. Interfejs (START -> KONIEC)
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

    st.info("💡 **NOWA FUNKCJA:** Zaznacz obszar na wykresie poniżej (kliknij i przeciągnij), aby automatycznie obliczyć metryki!")

        # Opcjonalne: ręczne wprowadzenie czasu (dla precyzji)
    def parse_time_to_seconds(t_str):
        try:
            parts = list(map(int, t_str.split(':')))
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
            if len(parts) == 2: return parts[0]*60 + parts[1]
            if len(parts) == 1: return parts[0]
        except:
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
            max_ve = interval_v['tymeventilation'].max()
            
            # Ve/Power Ratio (Efektywność)
            ve_power_ratio = avg_ve / avg_w if avg_w > 0 else 0
            
            # Trend (Slope) dla VE
            if len(interval_v) > 1:
                slope_ve, intercept_ve, _, _, _ = stats.linregress(interval_v['time'], interval_v['tymeventilation'])
                trend_desc_ve = f"{slope_ve:.4f} L/s"
            else:
                slope_ve = 0; intercept_ve = 0; trend_desc_ve = "N/A"

            # === AUTOMATYCZNA DETEKCJA STREFY (VT1/VT2) ===
            def detect_vent_zone(slope_val):
                """Wykrywa strefę wentylacyjną na podstawie slope VE."""
                if slope_val < 0.02:
                    return "⚪ Bardzo niska intensywność", "info", "Wentylacja stabilna. Strefa regeneracji."
                elif slope_val <= 0.05:
                    return "🟢 Poniżej VT1 (Strefa tlenowa)", "success", "Liniowy wzrost VE. Komfortowa intensywność tlenowa."
                elif slope_val <= 0.15:
                    return "🟡 VT1-VT2 (Strefa progowa)", "warning", "Pierwsze przełamanie wentylacyjne. Buforowanie kwasu mlekowego."
                else:
                    return "🔴 Powyżej VT2 (Hiperwentylacja)", "error", "Wykładniczy wzrost VE. Organizm nie nadąża z usuwaniem CO2."
            
            zone_name, zone_type, zone_desc = detect_vent_zone(slope_ve)

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
                except:
                    return "-"
            start_time_v = fmt_time_v(startsec)
            end_time_v = fmt_time_v(endsec)
            duration_v = int(endsec - startsec) if (endsec is not None and startsec is not None) else 0

            # Metryki
            st.subheader(f"Metryki Oddechowe: {start_time_v} - {end_time_v} ({duration_v}s)")
            mv1, mv2, mv3, mv4, mv5 = st.columns(5)
            mv1.metric("Śr. Moc", f"{avg_w:.0f} W")
            mv2.metric("Śr. Wentylacja (VE)", f"{avg_ve:.1f} L/min")
            mv3.metric("Częstość (RR)", f"{avg_rr:.1f} /min")
            mv4.metric("Wydajność (VE/W)", f"{ve_power_ratio:.3f}", help="Ile litrów powietrza na 1 Wat mocy. Niżej = lepiej (do pewnego momentu).")
            
            # Kolorowanie trendu (Tu odwrotnie niż w SmO2: Duży wzrost = Czerwony/Ostrzegawczy)
            trend_color = "inverse" if slope_ve > 0.1 else "normal"
            mv5.metric("Trend VE (Slope)", trend_desc_ve, delta=trend_desc_ve, delta_color=trend_color)

            # Wyświetl wykrytą strefę
            st.markdown("---")
            st.markdown("### 🎯 Automatyczna Detekcja Progu Wentylacyjnego")
            if zone_type == "info":
                st.info(f"**{zone_name}**\n\n{zone_desc}")
            elif zone_type == "success":
                st.success(f"**{zone_name}**\n\n{zone_desc}")
            elif zone_type == "warning":
                st.warning(f"**{zone_name}**\n\n{zone_desc}")
            else:
                st.error(f"**{zone_name}**\n\n{zone_desc}")

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

            # Zaznaczenie
            fig_vent.add_vrect(x0=startsec, x1=endsec, fillcolor="orange", opacity=0.1, layer="below", annotation_text="ANALIZA", annotation_position="top left")

            # Linia trendu VE
            if len(interval_v) > 1:
                trend_line_ve = intercept_ve + slope_ve * interval_v['time']
                fig_vent.add_trace(go.Scatter(
                    x=interval_v['time'], y=trend_line_ve,
                    customdata=interval_v['time_str'],
                    mode='lines', name='Trend VE',
                    line=dict(color='white', width=2, dash='dash'),
                    hovertemplate="<b>Trend:</b> %{y:.2f} L/min<extra></extra>"
                ))

            fig_vent.update_layout(
                title="Dynamika Wentylacji vs Moc",
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
            with st.expander("🫁 TEORIA: Jak znaleźć VT1 i VT2 na podstawie Slope?", expanded=False):
                st.markdown("""
                ### Interpretacja Slope (Nachylenia VE)
                Wentylacja rośnie nieliniowo. Szukamy punktów załamania krzywej ("Kinks").

                #### 🟢 1. Strefa Tlenowa (Poniżej VT1)
                * **Zachowanie:** VE rośnie proporcjonalnie do mocy (liniowo).
                * **Slope:** Stabilny, umiarkowanie dodatni (np. 0.02 - 0.05 L/s).
                * **RR (Oddechy):** Stabilne, wolne pogłębianie oddechu.

                #### 🟡 2. Próg VT1 (Aerobic Threshold) - "Pierwsze Przełamanie"
                * **Co szukać:** Pierwszy moment, gdzie Slope wyraźnie wzrasta, mimo że moc rośnie liniowo.
                * **Fizjologia:** Buforowanie kwasu mlekowego wodorowęglanami -> powstaje ekstra CO2 -> musisz go wydychać.
                * **Test mowy:** Tutaj zaczynasz urywać zdania.

                #### 🔴 3. Próg VT2 (Respiratory Compensation Point) - "Drugie Przełamanie"
                * **Co szukać:** Slope wystrzeliwuje w górę ("Vertical spike"). VE rośnie wykładniczo.
                * **Wartości Slope:** Bardzo wysokie (np. > 0.15 L/s).
                * **RR (Oddechy):** Gwałtowny wzrost częstości (tachypnoe).
                * **Fizjologia:** Hiperwentylacja. Organizm nie nadąża z usuwaniem CO2. Koniec równowagi.
                ---
                **Pro Tip:** Porównaj Slope VE ze Slope Mocy. Jeśli Moc rośnie o 5%, a VE o 15% -> właśnie przekroczyłeś próg.
                """)
    else:
        st.warning("Brak danych w tym zakresie.")
