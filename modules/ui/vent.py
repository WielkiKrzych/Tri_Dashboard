import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from scipy import stats
from modules.calculations.quality import check_signal_quality


def render_vent_tab(target_df, training_notes, uploaded_file_name):
    """Analiza wentylacji dla dowolnego treningu - struktura jak SmO2."""
    st.header("Analiza Wentylacji (VE & Breathing Rate)")
    st.markdown(
        "Analiza dynamiki oddechu dla dowolnego treningu. Szukaj anomalii w wentylacji i częstości oddechów."
    )

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    if "time" not in target_df.columns:
        st.error("Brak kolumny 'time' w danych!")
        st.stop()

    if "tymeventilation" not in target_df.columns:
        st.error("Brak danych wentylacji (tymeventilation)!")
        st.stop()

    # Wygładzanie
    if "watts_smooth_5s" not in target_df.columns and "watts" in target_df.columns:
        target_df["watts_smooth_5s"] = target_df["watts"].rolling(window=5, center=True).mean()
    if "ve_smooth" not in target_df.columns:
        target_df["ve_smooth"] = target_df["tymeventilation"].rolling(window=10, center=True).mean()
    if "tymebreathrate" in target_df.columns and "rr_smooth" not in target_df.columns:
        target_df["rr_smooth"] = target_df["tymebreathrate"].rolling(window=10, center=True).mean()

    target_df["time_str"] = pd.to_datetime(target_df["time"], unit="s").dt.strftime("%H:%M:%S")

    # Check Quality
    qual_res = check_signal_quality(target_df["tymeventilation"], "VE", (0, 300))
    if not qual_res["is_valid"]:
        st.warning(f"⚠️ **Niska Jakość Sygnału VE (Score: {qual_res['score']})**")
        for issue in qual_res["issues"]:
            st.caption(f"❌ {issue}")

    # Inicjalizacja session_state
    if "vent_start_sec" not in st.session_state:
        st.session_state.vent_start_sec = 600
    if "vent_end_sec" not in st.session_state:
        st.session_state.vent_end_sec = 1200

    # ===== NOTATKI VENTILATION =====
    with st.expander("📝 Dodaj Notatkę do tej Analizy", expanded=False):
        note_col1, note_col2 = st.columns([1, 2])
        with note_col1:
            note_time = st.number_input(
                "Czas (min)",
                min_value=0.0,
                max_value=float(len(target_df) / 60) if len(target_df) > 0 else 60.0,
                value=float(len(target_df) / 120) if len(target_df) > 0 else 15.0,
                step=0.5,
                key="vent_note_time",
            )
        with note_col2:
            note_text = st.text_input(
                "Notatka",
                key="vent_note_text",
                placeholder="Np. 'VE jump', 'Spłycenie oddechu', 'Hiperwentylacja'",
            )

        if st.button("➕ Dodaj Notatkę", key="vent_add_note"):
            if note_text:
                training_notes.add_note(uploaded_file_name, note_time, "ventilation", note_text)
                st.success(f"✅ Notatka: {note_text} @ {note_time:.1f} min")
            else:
                st.warning("Wpisz tekst notatki!")

    # Wyświetl istniejące notatki
    existing_notes = training_notes.get_notes_for_metric(uploaded_file_name, "ventilation")
    if existing_notes:
        st.subheader("📋 Notatki Wentylacji")
        for idx, note in enumerate(existing_notes):
            col_note, col_del = st.columns([4, 1])
            with col_note:
                st.info(f"⏱️ **{note['time_minute']:.1f} min** | {note['text']}")
            with col_del:
                if st.button("🗑️", key=f"del_vent_note_{idx}"):
                    training_notes.delete_note(uploaded_file_name, idx)
                    st.rerun()

    st.markdown("---")

    # ===== ANALIZA MANUALNA =====
    st.info(
        "💡 **ANALIZA MANUALNA:** Zaznacz obszar na wykresie poniżej (kliknij i przeciągnij), aby sprawdzić nachylenie lokalne."
    )

    def parse_time_to_seconds(t_str):
        try:
            parts = list(map(int, t_str.split(":")))
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 1:
                return parts[0]
        except (ValueError, AttributeError):
            return None
        return None

    with st.expander("🔧 Ręczne wprowadzenie zakresu czasowego (opcjonalne)", expanded=False):
        col_inp_1, col_inp_2 = st.columns(2)
        with col_inp_1:
            manual_start = st.text_input(
                "Start Interwału (hh:mm:ss)", value="00:10:00", key="vent_manual_start"
            )
        with col_inp_2:
            manual_end = st.text_input(
                "Koniec Interwału (hh:mm:ss)", value="00:20:00", key="vent_manual_end"
            )

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

    def format_time(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    # Wycinanie danych
    mask = (target_df["time"] >= startsec) & (target_df["time"] <= endsec)
    interval_data = target_df.loc[mask]

    if not interval_data.empty and endsec > startsec:
        duration_sec = int(endsec - startsec)

        # Obliczenia
        avg_watts = interval_data["watts"].mean() if "watts" in interval_data.columns else 0
        avg_ve = interval_data["tymeventilation"].mean()
        avg_rr = (
            interval_data["tymebreathrate"].mean()
            if "tymebreathrate" in interval_data.columns
            else 0
        )

        # Trend (Slope) dla VE
        if len(interval_data) > 1:
            slope_ve, intercept_ve, _, _, _ = stats.linregress(
                interval_data["time"], interval_data["tymeventilation"]
            )
            trend_desc = f"{slope_ve:.4f} L/s"
        else:
            slope_ve = 0
            intercept_ve = 0
            trend_desc = "N/A"

        # Metryki Manualne
        st.subheader(
            f"METRYKI MANUALNE: {format_time(startsec)} - {format_time(endsec)} ({duration_sec}s)"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Śr. Moc", f"{avg_watts:.0f} W")
        m2.metric("Śr. VE", f"{avg_ve:.1f} L/min")
        m3.metric("Śr. BR", f"{avg_rr:.1f} /min")

        # Kolorowanie trendu (pozytywny = wzrost VE = potencjalnie próg)
        trend_color = "inverse" if slope_ve > 0.05 else "normal"
        m4.metric("Trend VE (Slope)", trend_desc, delta=trend_desc, delta_color=trend_color)

        # ===== WYKRES GŁÓWNY (VE + Power) =====
        fig_vent = go.Figure()

        # VE (Primary)
        fig_vent.add_trace(
            go.Scatter(
                x=target_df["time"],
                y=target_df["ve_smooth"],
                customdata=target_df["time_str"],
                mode="lines",
                name="VE (L/min)",
                line=dict(color="#ffa15a", width=2),
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>",
            )
        )

        # Power (Secondary)
        if "watts_smooth_5s" in target_df.columns:
            fig_vent.add_trace(
                go.Scatter(
                    x=target_df["time"],
                    y=target_df["watts_smooth_5s"],
                    customdata=target_df["time_str"],
                    mode="lines",
                    name="Power",
                    line=dict(color="#1f77b4", width=1),
                    yaxis="y2",
                    opacity=0.3,
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>",
                )
            )

        # Zaznaczenie manualne
        fig_vent.add_vrect(
            x0=startsec,
            x1=endsec,
            fillcolor="orange",
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text="MANUAL",
            annotation_position="top left",
        )

        # Linia trendu VE (dla manualnego)
        if len(interval_data) > 1:
            trend_line = intercept_ve + slope_ve * interval_data["time"]
            fig_vent.add_trace(
                go.Scatter(
                    x=interval_data["time"],
                    y=trend_line,
                    mode="lines",
                    name="Trend VE (Man)",
                    line=dict(color="white", width=2, dash="dash"),
                    hovertemplate="<b>Trend:</b> %{y:.2f} L/min<extra></extra>",
                )
            )

        fig_vent.update_layout(
            title="Dynamika Wentylacji vs Moc",
            xaxis_title="Czas",
            yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
            yaxis2=dict(
                title=dict(text="Moc (W)", font=dict(color="#1f77b4")),
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.01, y=0.99),
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified",
        )

        # Wykres z interaktywnym zaznaczaniem
        selected = st.plotly_chart(
            fig_vent,
            use_container_width=True,
            key="vent_chart",
            on_select="rerun",
            selection_mode="box",
        )

        # Obsługa zaznaczenia
        if selected and "selection" in selected and "box" in selected["selection"]:
            box_data = selected["selection"]["box"]
            if box_data and len(box_data) > 0:
                x_range = box_data[0].get("x", [])
                if len(x_range) == 2:
                    new_start = min(x_range)
                    new_end = max(x_range)
                    if (
                        new_start != st.session_state.vent_start_sec
                        or new_end != st.session_state.vent_end_sec
                    ):
                        st.session_state.vent_start_sec = new_start
                        st.session_state.vent_end_sec = new_end
                        st.rerun()

        # ===== DYNAMIKA TIDAL VOLUME (VT) vs MOC =====
        if "tymebreathrate" in target_df.columns:
            st.markdown("---")
            st.subheader("Dynamika Tidal Volume vs Moc")

            # Oblicz VT (Tidal Volume) = VE / BR
            target_df["vt"] = target_df["tymeventilation"] / target_df["tymebreathrate"]
            target_df["vt_smooth"] = target_df["vt"].rolling(window=10, center=True).mean()

            fig_vt = go.Figure()

            # VT (Primary)
            fig_vt.add_trace(
                go.Scatter(
                    x=target_df["time"],
                    y=target_df["vt_smooth"],
                    customdata=target_df["time_str"],
                    mode="lines",
                    name="Tidal Volume (L)",
                    line=dict(color="#00cc96", width=2),
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>VT:</b> %{y:.2f} L<extra></extra>",
                )
            )

            # Power (Secondary)
            if "watts_smooth_5s" in target_df.columns:
                fig_vt.add_trace(
                    go.Scatter(
                        x=target_df["time"],
                        y=target_df["watts_smooth_5s"],
                        customdata=target_df["time_str"],
                        mode="lines",
                        name="Power",
                        line=dict(color="#1f77b4", width=1),
                        yaxis="y2",
                        opacity=0.3,
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>",
                    )
                )

            # Zaznaczenie manualne
            fig_vt.add_vrect(
                x0=startsec,
                x1=endsec,
                fillcolor="orange",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="MANUAL",
                annotation_position="top left",
            )

            # Linia trendu VT (dla manualnego)
            if len(interval_data) > 1 and "vt" in interval_data.columns:
                slope_vt, intercept_vt, _, _, _ = stats.linregress(
                    interval_data["time"], interval_data["vt"]
                )
                trend_line_vt = intercept_vt + slope_vt * interval_data["time"]
                fig_vt.add_trace(
                    go.Scatter(
                        x=interval_data["time"],
                        y=trend_line_vt,
                        mode="lines",
                        name="Trend VT (Man)",
                        line=dict(color="white", width=2, dash="dash"),
                        hovertemplate="<b>Trend VT:</b> %{y:.2f} L<extra></extra>",
                    )
                )

            fig_vt.update_layout(
                title="Dynamika Tidal Volume (VT) vs Moc",
                xaxis_title="Czas",
                yaxis=dict(title=dict(text="Tidal Volume (L)", font=dict(color="#00cc96"))),
                yaxis2=dict(
                    title=dict(text="Moc (W)", font=dict(color="#1f77b4")),
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(x=0.01, y=0.99),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
            )

            st.plotly_chart(fig_vt, use_container_width=True)

        # ===== DYNAMIKA BREATHING RATE (BR) vs MOC =====
        if "tymebreathrate" in target_df.columns:
            st.markdown("---")
            st.subheader("Dynamika Breathing Rate vs Moc")

            fig_br_power = go.Figure()

            # BR (Primary)
            fig_br_power.add_trace(
                go.Scatter(
                    x=target_df["time"],
                    y=target_df["rr_smooth"]
                    if "rr_smooth" in target_df.columns
                    else target_df["tymebreathrate"],
                    customdata=target_df["time_str"],
                    mode="lines",
                    name="Breathing Rate (/min)",
                    line=dict(color="#ab63fa", width=2),
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>BR:</b> %{y:.1f} /min<extra></extra>",
                )
            )

            # Power (Secondary)
            if "watts_smooth_5s" in target_df.columns:
                fig_br_power.add_trace(
                    go.Scatter(
                        x=target_df["time"],
                        y=target_df["watts_smooth_5s"],
                        customdata=target_df["time_str"],
                        mode="lines",
                        name="Power",
                        line=dict(color="#1f77b4", width=1),
                        yaxis="y2",
                        opacity=0.3,
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>",
                    )
                )

            # Zaznaczenie manualne
            fig_br_power.add_vrect(
                x0=startsec,
                x1=endsec,
                fillcolor="orange",
                opacity=0.1,
                layer="below",
                line_width=0,
                annotation_text="MANUAL",
                annotation_position="top left",
            )

            # Linia trendu BR (dla manualnego)
            if len(interval_data) > 1 and "tymebreathrate" in interval_data.columns:
                slope_br, intercept_br, _, _, _ = stats.linregress(
                    interval_data["time"], interval_data["tymebreathrate"]
                )
                trend_line_br = intercept_br + slope_br * interval_data["time"]
                fig_br_power.add_trace(
                    go.Scatter(
                        x=interval_data["time"],
                        y=trend_line_br,
                        mode="lines",
                        name="Trend BR (Man)",
                        line=dict(color="white", width=2, dash="dash"),
                        hovertemplate="<b>Trend BR:</b> %{y:.1f} /min<extra></extra>",
                    )
                )

            fig_br_power.update_layout(
                title="Dynamika Breathing Rate (BR) vs Moc",
                xaxis_title="Czas",
                yaxis=dict(title=dict(text="Breathing Rate (/min)", font=dict(color="#ab63fa"))),
                yaxis2=dict(
                    title=dict(text="Moc (W)", font=dict(color="#1f77b4")),
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(x=0.01, y=0.99),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
            )

            st.plotly_chart(fig_br_power, use_container_width=True)

        # ===== LEGACY TOOLS (Surowe Dane) =====
        with st.expander("🔧 Szczegółowa Analiza (Surowe Dane)", expanded=False):
            st.markdown("### Surowe Dane i Korelacje")

            # Scatter Plot: VE vs Watts
            if "watts" in interval_data.columns:
                interval_time_str = pd.to_datetime(interval_data["time"], unit="s").dt.strftime(
                    "%H:%M:%S"
                )

                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(
                        x=interval_data["watts"],
                        y=interval_data["tymeventilation"],
                        customdata=interval_time_str,
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=interval_data["time"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Czas (s)"),
                        ),
                        name="VE vs Power",
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{x:.0f} W<br><b>VE:</b> %{y:.1f} L/min<extra></extra>",
                    )
                )
                fig_scatter.update_layout(
                    title="Korelacja: VE vs Moc",
                    xaxis_title="Power (W)",
                    yaxis_title="VE (L/min)",
                    height=400,
                    hovermode="closest",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Breathing Rate Visualization
            if "tymebreathrate" in interval_data.columns:
                st.subheader("Częstość Oddechów (Breathing Rate)")

                interval_time_str = pd.to_datetime(interval_data["time"], unit="s").dt.strftime(
                    "%H:%M:%S"
                )

                fig_br = go.Figure()
                fig_br.add_trace(
                    go.Scatter(
                        x=interval_data["time"],
                        y=interval_data["tymebreathrate"],
                        customdata=interval_time_str,
                        mode="lines",
                        name="BR",
                        line=dict(color="#00cc96", width=2),
                        hovertemplate="<b>Czas:</b> %{customdata}<br><b>BR:</b> %{y:.1f} /min<extra></extra>",
                    )
                )
                fig_br.update_layout(
                    title="Breathing Rate",
                    xaxis_title="Czas",
                    yaxis_title="BR (/min)",
                    height=300,
                    hovermode="x unified",
                )
                st.plotly_chart(fig_br, use_container_width=True)

            # Minute Ventilation Chart
            st.subheader("Wentylacja Minutowa (VE)")

            interval_time_str = pd.to_datetime(interval_data["time"], unit="s").dt.strftime(
                "%H:%M:%S"
            )

            fig_ve = go.Figure()
            fig_ve.add_trace(
                go.Scatter(
                    x=interval_data["time"],
                    y=interval_data["tymeventilation"],
                    customdata=interval_time_str,
                    mode="lines",
                    name="VE",
                    line=dict(color="#ffa15a", width=2),
                    hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>",
                )
            )
            fig_ve.update_layout(
                title="Minute Ventilation (VE)",
                xaxis_title="Czas",
                yaxis_title="VE (L/min)",
                height=300,
                hovermode="x unified",
            )
            st.plotly_chart(fig_ve, use_container_width=True)

    else:
        st.warning("Brak danych w wybranym zakresie.")

    # ===== TEORIA =====
    with st.expander("🫁 TEORIA: Interpretacja Wentylacji", expanded=False):
        st.markdown("""
        ## Co oznacza Wentylacja (VE)?
        
        **VE (Minute Ventilation)** to objętość powietrza wdychanego/wydychanego na minutę.
        Mierzona przez sensory oddechowe np. **CORE, Tyme Wear, Garmin HRM-Pro (estymacja)**.
        
        | Parametr | Opis | Jednostka |
        |----------|------|-----------|
        | **VE** | Wentylacja minutowa | L/min |
        | **BR / RR** | Częstość oddechów | oddechy/min |
        | **VT** | Objętość oddechowa (VE/BR) | L |
        
        ---
        
        ## Strefy VE i ich znaczenie
        
        | VE (L/min) | Interpretacja | Typ wysiłku |
        |------------|---------------|-------------|
        | **20-40** | Spokojny oddech | Recovery, rozgrzewka |
        | **40-80** | Umiarkowany wysiłek | Tempo, Sweet Spot |
        | **80-120** | Intensywny wysiłek | Threshold, VO2max |
        | **> 120** | Maksymalny wysiłek | Sprint, test wyczerpania |
        
        ---
        
        ## Trend VE (Slope) - Co oznacza nachylenie?
        
        | Trend | Wartość | Interpretacja |
        |-------|---------|---------------|
        | 🟢 **Stabilny** | ~ 0 | Steady state, VE odpowiada obciążeniu |
        | 🟡 **Łagodny wzrost** | 0.01-0.05 | Normalna adaptacja do wysiłku |
        | 🔴 **Gwałtowny wzrost** | > 0.05 | Możliwy próg wentylacyjny (VT1/VT2) |
        
        ---
        
        ## BR (Breathing Rate) - Częstość oddechów
        
        **BR** odzwierciedla strategię oddechową:
        
        - **⬆️ Wzrost BR przy stałej VE**: Płytszy oddech, możliwe zmęczenie przepony
        - **⬇️ Spadek BR przy stałej VE**: Głębszy oddech, lepsza efektywność
        - **➡️ Stabilny BR**: Optymalna strategia oddechowa
        
        ### Praktyczny przykład:
        - **VE=100, BR=30**: Objętość oddechowa = 3.3L (głęboki oddech)
        - **VE=100, BR=50**: Objętość oddechowa = 2.0L (płytki oddech - nieefektywne!)
        
        ---
        
        ## Zastosowania Treningowe VE
        
        ### 1️⃣ Detekcja Progów (VT1, VT2)
        - **VT1 (Próg tlenowy)**: Pierwszy nieliniowy skok VE względem mocy
        - **VT2 (Próg beztlenowy)**: Drugi, gwałtowniejszy skok VE
        - 🔗 Użyj zakładki **"Ventilation - Progi"** do automatycznej detekcji
        
        ### 2️⃣ Kontrola Intensywności
        - Jeśli VE rośnie szybciej niż moc → zbliżasz się do progu
        - Stabilna VE przy stałej mocy → jesteś w strefie tlenowej
        
        ### 3️⃣ Efektywność Oddechowa
        - Optymalna częstość BR: 20-40 oddechów/min
        - Powyżej 50/min: możliwe zmęczenie, stres, lub panika
        
        ### 4️⃣ Detekcja Zmęczenia
        - **BR rośnie przy spadku VE**: Zmęczenie przepony
        - **VE fluktuuje chaotycznie**: Możliwe odwodnienie lub hipoglikemia
        
        ---
        
        ## Korelacja VE vs Moc
        
        Wykres scatter pokazuje zależność między mocą a wentylacją:
        
        - **Liniowa zależność**: Normalna odpowiedź fizjologiczna
        - **Punkt załamania**: Próg wentylacyjny (VT)
        - **Stroma krzywa**: Niska wydolność, szybkie zadyszenie
        
        ### Kolor punktów (czas):
        - **Wczesne punkty (ciemne)**: Początek treningu
        - **Późne punkty (jasne)**: Koniec treningu, kumulacja zmęczenia
        
        ---
        
        ## Limitacje Pomiaru VE
        
        ⚠️ **Czynniki wpływające na dokładność:**
        - Pozycja sensora na klatce piersiowej
        - Oddychanie ustami vs nosem
        - Warunki atmosferyczne (wysokość, wilgotność)
        - Intensywność mowy podczas jazdy
        
        💡 **Wskazówka**: Dla dokładnej detekcji progów wykonaj Test Stopniowany (Ramp Test)!
        """)
