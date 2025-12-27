import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.calculations import calculate_dynamic_dfa

def render_hrv_tab(df_clean_pl):
    st.header("Analiza Zmienności Rytmu Serca (HRV)")

    # 1. Inicjalizacja "Pamięci" (Session State)
    if 'df_dfa' not in st.session_state:
        st.session_state.df_dfa = None
    if 'dfa_error' not in st.session_state:
        st.session_state.dfa_error = None

    # 2. Obsługa Przycisku
    if st.session_state.df_dfa is None and st.session_state.dfa_error is None:
        st.info("💡 Analiza DFA Alpha-1 wymaga zaawansowanych obliczeń fraktalnych.")
        st.markdown("Kliknij przycisk poniżej, aby uruchomić algorytm. Może to zająć od kilku do kilkunastu sekund.")
        
        if st.button("🚀 Oblicz HRV i DFA Alpha-1"):
            with st.spinner("Analiza geometrii rytmu serca... Proszę czekać..."):
                try:
                    result_df, error_msg = calculate_dynamic_dfa(df_clean_pl)
                    
                    st.session_state.df_dfa = result_df
                    st.session_state.dfa_error = error_msg
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Wystąpił błąd krytyczny algorytmu: {e}")

    # 3. Pobranie danych z pamięci do zmiennych lokalnych
    df_dfa = st.session_state.df_dfa
    dfa_error = st.session_state.dfa_error
   
    if df_dfa is not None and not df_dfa.empty:
        
        df_dfa = df_dfa.sort_values('time')
        # Konwersja df_clean_pl na Pandas/Numpy dla interpolacji
        df_clean = df_clean_pl.to_pandas() if hasattr(df_clean_pl, "to_pandas") else df_clean_pl
        
        orig_times = df_clean['time'].values
        orig_watts = df_clean['watts_smooth'].values if 'watts_smooth' in df_clean.columns else np.zeros(len(orig_times))
        # orig_hr nieużywany do interpolacji tutaj? W app.py była interpolacja HR, ale w metrykach jest mean_rr przeliczone na HR.
        # Sprawdzam app.py: row 1080: df_dfa['hr'] = np.interp(..., orig_hr)
        orig_hr = df_clean['heartrate_smooth'].values if 'heartrate_smooth' in df_clean.columns else np.zeros(len(orig_times))
        
        df_dfa['watts'] = np.interp(df_dfa['time'], orig_times, orig_watts)
        df_dfa['hr'] = np.interp(df_dfa['time'], orig_times, orig_hr)
        df_dfa['time_min'] = df_dfa['time'] / 60.0

        # Metryki podsumowujące
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Śr. RMSSD", f"{df_dfa['rmssd'].mean():.1f} ms" if 'rmssd' in df_dfa.columns else "N/A")
        col2.metric("Śr. SDNN", f"{df_dfa['sdnn'].mean():.1f} ms" if 'sdnn' in df_dfa.columns else "N/A")
        col3.metric("Śr. RR", f"{df_dfa['mean_rr'].mean():.0f} ms" if 'mean_rr' in df_dfa.columns else "N/A")
        col4.metric("Śr. HR (z RR)", f"{60000/df_dfa['mean_rr'].mean():.0f} bpm" if 'mean_rr' in df_dfa.columns else "N/A")

        st.subheader("Indeks Zmienności HRV (Pseudo-Alpha)")
        st.caption("Wyższe wartości = większa zmienność = lepszy stan regeneracji. Niższe = stres metaboliczny.")
        
        fig_dfa = go.Figure()
        fig_dfa.add_trace(go.Scatter(
            x=df_dfa['time_min'], 
            y=df_dfa['alpha1'],
            name='Indeks HRV',
            mode='lines',
            line=dict(color='#00cc96', width=2),
            hovertemplate="Indeks: %{y:.2f}<extra></extra>"
        ))

        fig_dfa.add_trace(go.Scatter(
            x=df_dfa['time_min'], 
            y=df_dfa['watts'],
            name='Moc',
            yaxis='y2',
            fill='tozeroy',
            line=dict(width=0.5, color='rgba(255,255,255,0.1)'),
            hovertemplate="Moc: %{y:.0f} W<extra></extra>"
        ))

        fig_dfa.add_hline(y=0.75, line_dash="solid", line_color="#ef553b", line_width=2, 
                        annotation_text="Próg stresu (0.75)", annotation_position="top left")
        
        fig_dfa.add_hline(y=0.50, line_dash="dot", line_color="#ab63fa", line_width=1, 
                        annotation_text="Wysoki stres (0.50)", annotation_position="bottom left")

        fig_dfa.update_layout(
            template="plotly_dark",
            title="Indeks Zmienności HRV vs Czas",
            hovermode="x unified",
            xaxis=dict(title="Czas [min]"),
            yaxis=dict(title="Indeks HRV", range=[0.2, 1.6]),
            yaxis2=dict(title="Moc [W]", overlaying='y', side='right', showgrid=False),
            height=500,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.05, x=0)
        )

        st.plotly_chart(fig_dfa, use_container_width=True)

        # Wykres RMSSD jeśli dostępny
        if 'rmssd' in df_dfa.columns:
            st.subheader("RMSSD w czasie")
            fig_rmssd = go.Figure()
            fig_rmssd.add_trace(go.Scatter(
                x=df_dfa['time_min'], 
                y=df_dfa['rmssd'],
                name='RMSSD',
                mode='lines',
                line=dict(color='#636efa', width=2),
                hovertemplate="RMSSD: %{y:.1f} ms<extra></extra>"
            ))
            fig_rmssd.add_trace(go.Scatter(
                x=df_dfa['time_min'], 
                y=df_dfa['watts'],
                name='Moc',
                yaxis='y2',
                fill='tozeroy',
                line=dict(width=0.5, color='rgba(255,255,255,0.1)'),
                hovertemplate="Moc: %{y:.0f} W<extra></extra>"
            ))
            fig_rmssd.update_layout(
                template="plotly_dark",
                title="RMSSD (Root Mean Square of Successive Differences)",
                hovermode="x unified",
                xaxis=dict(title="Czas [min]"),
                yaxis=dict(title="RMSSD [ms]"),
                yaxis2=dict(title="Moc [W]", overlaying='y', side='right', showgrid=False),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=1.05, x=0)
            )
            st.plotly_chart(fig_rmssd, use_container_width=True)

        # --- WYKRES POINCARE (Lorenz Plot) ---
        st.markdown("---")
        st.subheader("Wykres Poincaré (Geometria Rytmu)")
        
        # Upewnij się że masz dostęp do rr_col_raw
        # W app.py: rr_col_raw = next(...) from df_clean_pl.columns
        rr_col_raw = next((c for c in df_clean.columns if any(x in c.lower() for x in ['rr', 'hrv', 'ibi', 'r-r'])), None)
        
        if rr_col_raw:
            raw_rr_series = df_clean[rr_col_raw].dropna().values
            if raw_rr_series.mean() < 2.0: raw_rr_series = raw_rr_series * 1000
            raw_rr_series = raw_rr_series[(raw_rr_series > 300) & (raw_rr_series < 2000)]
            if len(raw_rr_series) > 10:
                rr_n = raw_rr_series[:-1]
                rr_n1 = raw_rr_series[1:]
                
                diff_rr = rr_n1 - rr_n
                sd1 = np.std(diff_rr) / np.sqrt(2)
                sd2 = np.sqrt(2 * np.std(raw_rr_series)**2 - 0.5 * np.std(diff_rr)**2)
                ratio_sd = sd2 / sd1 if sd1 > 0 else 0

                fig_poincare = go.Figure()
                
                fig_poincare.add_trace(go.Scatter(
                    x=rr_n, y=rr_n1,
                    mode='markers',
                    name='Interwały R-R',
                    marker=dict(
                        size=3,
                        color='rgba(0, 204, 150, 0.5)', 
                        line=dict(width=0)
                    ),
                    hovertemplate="RR(n): %{x:.0f} ms<br>RR(n+1): %{y:.0f} ms<extra></extra>"
                ))
                
                min_rr, max_rr = min(raw_rr_series), max(raw_rr_series)
                fig_poincare.add_trace(go.Scatter(
                    x=[min_rr, max_rr], y=[min_rr, max_rr],
                    mode='lines',
                    name='Linia tożsamości',
                    line=dict(color='white', width=1, dash='dash'),
                    hoverinfo='skip'
                ))

                fig_poincare.update_layout(
                    template="plotly_dark",
                    title=f"Poincaré Plot (SD1: {sd1:.1f}ms, SD2: {sd2:.1f}ms, Ratio: {ratio_sd:.2f})",
                    xaxis=dict(title="RR [n] (ms)", scaleanchor="y", scaleratio=1),
                    yaxis=dict(title="RR [n+1] (ms)"),
                    width=600, height=600, # Kwadratowy wykres
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                c_p1, c_p2 = st.columns([2, 1])
                with c_p1:
                    st.plotly_chart(fig_poincare, use_container_width=True)
                with c_p2:
                    st.info(f"""
                    **📊 Interpretacja Kliniczna:**
                    
                    * **Kształt "Komety" / "Rakiety":** Fizjologiczna norma u sportowca. Długa oś (SD2) to ogólna zmienność, krótka oś (SD1) to nagłe zmiany (parasympatyka).
                    * **Kształt "Kulisty":** Wysoki stres, dominacja współczulna (Fight or Flight) lub... bardzo równe tempo (metronom).
                    * **SD1 ({sd1:.1f} ms):** Czysta aktywność nerwu błędnego (regeneracja). Im więcej, tym lepiej.
                    * **SD2 ({sd2:.1f} ms):** Długoterminowa zmienność (rytm dobowy + termoregulacja).
                    
                    *Punkty daleko od głównej chmury to zazwyczaj ektopie (dodatkowe skurcze) lub błędy pomiaru.*
                    """)
            else:
                st.warning("Za mało danych R-R po filtracji artefaktów.")
        else:
            st.warning("Brak surowych danych R-R do wygenerowania wykresu Poincaré.")    

        mask_threshold = (df_dfa['time_min'] > 5) & (df_dfa['alpha1'] < 0.75)
        
        if mask_threshold.any():
            row = df_dfa[mask_threshold].iloc[0]
            vt1_est_power = row['watts']
            vt1_est_hr = row['hr']
            vt1_time = row['time_min']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Estymowane VT1 (Moc)", f"{vt1_est_power:.0f} W", help="Moc w momencie przecięcia linii 0.75")
            c2.metric("Estymowane VT1 (HR)", f"{vt1_est_hr:.0f} bpm", help="Tętno w momencie przecięcia linii 0.75")
            c3.metric("Czas przecięcia", f"{vt1_time:.0f} min")
            
            if vt1_est_power < 100:
                st.warning("⚠️ Wykryto bardzo niskie VT1. Sprawdź jakość danych HRV (artefakty mogą zaniżać wynik).")
        else:
            st.info("Nie przekroczono progu 0.75 w trakcie tego treningu (cały czas praca tlenowa lub krótkie dane).")

        # --- TEORIA ---
        with st.expander("🧠 O co chodzi z DFA Alpha-1?", expanded=True):
            st.markdown(r"""
            **Detrended Fluctuation Analysis ($\alpha_1$)** mierzy tzw. korelacje fraktalne w odstępach między uderzeniami serca.
            
            * **$\alpha_1 \approx 1.0$ (Szum Różowy):** Stan zdrowy, wypoczęty. Serce bije w sposób złożony, elastyczny. Organizuje się samo.
            * **$\alpha_1 \approx 0.5$ (Szum Biały/Losowy):** Silny stres metaboliczny. Układ nerwowy "bombarduje" węzeł zatokowy, rytm staje się nieskorelowany.
            
            **Dlaczego 0.75?**
            Badania (m.in. Rogers et al.) wykazały, że przejście przez wartość **0.75** idealnie pokrywa się z **Pierwszym Progiem Wentylacyjnym (VT1)**. Jest to punkt, w którym zaczynasz tracić "luz tlenowy", a organizm zaczyna rekrutować więcej włókien szybkokurczliwych.
            """)

    else:
        # Debugowanie - pokaż dostępne kolumny
        # Wymagamy df_clean_pl
        df_clean = df_clean_pl.to_pandas() if hasattr(df_clean_pl, "to_pandas") else df_clean_pl
        hrv_cols = [c for c in df_clean.columns if any(x in c.lower() for x in ['rr', 'hrv', 'ibi', 'r-r'])]
        if hrv_cols:
            st.info(f"🔍 Znaleziono kolumny HRV: {hrv_cols}")
            for col in hrv_cols:
                col_data = df_clean[col].dropna()
                valid_count = (col_data > 0).sum()
                st.write(f"  - {col}: {valid_count} wartości > 0, średnia: {col_data.mean():.2f}, zakres: {col_data.min():.2f} - {col_data.max():.2f}")
        else:
            st.info(f"🔍 Dostępne kolumny: {list(df_clean.columns)}")
        
        if dfa_error:
            st.error(f"❌ Błąd DFA: {dfa_error}")
        
        st.warning("⚠️ **Brak wystarczających danych R-R (Inter-Beat Intervals).**")
        st.markdown("""
        Aby analiza DFA zadziałała, plik musi zawierać surowe dane o każdym uderzeniu serca, a nie tylko uśrednione tętno.
        * Sprawdź, czy Twój pas HR obsługuje HRV (np. Polar H10, Garmin HRM-Pro).
        * Upewnij się, że włączyłeś zapis zmienności tętna w zegarku/komputerze (często opcja "Log HRV").
        * Wymagane jest minimum 300 próbek z interwałami R-R > 0.
        """)
