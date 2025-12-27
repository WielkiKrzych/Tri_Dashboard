import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_limiters_tab(df_plot, cp_input, vt2_vent):
    st.header("Analiza Limiterów Fizjologicznych (Radar)")
    st.markdown("Sprawdzamy, który układ (Serce, Płuca, Mięśnie) był 'wąskim gardłem' podczas najcięższych momentów treningu.")

    # Sprawdzamy dostępność danych
    has_hr = 'heartrate' in df_plot.columns
    has_ve = any(c in df_plot.columns for c in ['tymeventilation', 've', 'ventilation'])
    has_smo2 = 'smo2' in df_plot.columns
    has_watts = 'watts' in df_plot.columns

    if has_watts and (has_hr or has_ve or has_smo2):
        
        # 1. Wybór okna czasowego (Peak Power)
        window_options = {
            "1 min (Anaerobic)": 60, 
            "5 min (VO2max)": 300, 
            "20 min (FTP)": 1200,
            "60 min (Endurance)": 3600
        }
        selected_window_name = st.selectbox("Wybierz okno analizy (MMP):", list(window_options.keys()), index=1)
        window_sec = window_options[selected_window_name]

        # Znajdujemy indeks startu dla najlepszej średniej mocy w tym oknie
        # Rolling musi mieć min_periods=window_sec, żeby nie liczyć "połówek" na początku
        df_plot['rolling_watts'] = df_plot['watts'].rolling(window=window_sec, min_periods=window_sec).mean()

        if df_plot['rolling_watts'].isna().all():
            st.warning(f"Trening jest krótszy niż {window_sec/60:.0f} min. Wybierz krótsze okno.")
            st.stop()

        peak_idx = df_plot['rolling_watts'].idxmax()

        # Sprawdzamy, czy znaleziono peak (czy trening był wystarczająco długi)
        if not pd.isna(peak_idx):
            # Wycinamy ten fragment danych
            start_idx = max(0, peak_idx - window_sec + 1)
            df_peak = df_plot.iloc[start_idx:peak_idx+1]
            
            # 2. Obliczamy % wykorzystania potencjału (Estymacja Maxów)
            
            # HR (Centralny)
            peak_hr_avg = df_peak['heartrate'].mean() if has_hr else 0
            max_hr_user = df_plot['heartrate'].max() 
            pct_hr = (peak_hr_avg / max_hr_user * 100) if max_hr_user > 0 else 0
            
            # VE (Oddechowy)
            col_ve_nm = next((c for c in ['tymeventilation', 've', 'ventilation'] if c in df_plot.columns), None)
            peak_ve_avg = df_peak[col_ve_nm].mean() if col_ve_nm else 0
            # Estymujemy Max VE jako 110% VT2 (bezpieczny margines dla RCP)
            max_ve_user = vt2_vent * 1.1 
            pct_ve = (peak_ve_avg / max_ve_user * 100) if max_ve_user > 0 else 0
            
            # SmO2 (Lokalny) - Odwrócona logika (im mniej tym "więcej" pracy)
            peak_smo2_avg = df_peak['smo2'].mean() if has_smo2 else 100
            # Używamy 100 - SmO2 jako "stopnia ekstrakcji tlenu"
            pct_smo2_util = 100 - peak_smo2_avg
            
            # Power (Mechaniczny) vs CP
            peak_w_avg = df_peak['watts'].mean()
            pct_power = (peak_w_avg / cp_input * 100) if cp_input > 0 else 0

            # 3. Rysujemy Radar
            categories = ['Serce (% HRmax)', 'Płuca (% VEmax)', 'Mięśnie (% Desat)', 'Moc (% CP)']
            values = [pct_hr, pct_ve, pct_smo2_util, pct_power]
            
            # Zamykamy koło dla wykresu radarowego
            values += [values[0]]
            categories += [categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=selected_window_name,
                line=dict(color='#00cc96'),
                fillcolor='rgba(0, 204, 150, 0.3)',
                hovertemplate="%{theta}: <b>%{r:.1f}%</b><extra></extra>"
            ))

            # Dynamiczna skala - jeśli moc wyskoczy poza 120% (np. przy 1 min), zwiększamy zakres
            max_val = max(values)
            range_max = 100 if max_val < 100 else (max_val + 10)

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, range_max] 
                    )
                ),
                template="plotly_dark",
                title=f"Profil Obciążenia: {selected_window_name} ({peak_w_avg:.0f} W)",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # 4. Interpretacja
            st.info(f"""
            **🔍 Diagnoza dla odcinka {selected_window_name}:**
            
            * **Serce (Central):** {pct_hr:.1f}% Maxa. (Wysokie tętno = koszt transportu).
            * **Płuca (Oddech):** {pct_ve:.1f}% Szacowanego Maxa. (Wysokie VE = koszt usunięcia CO2).
            * **Mięśnie (Lokalne):** {pct_smo2_util:.1f}% Wykorzystania tlenu (Średnie SmO2: {peak_smo2_avg:.1f}%).
            * **Moc:** {pct_power:.0f}% Twojego CP/FTP.
            
            **Co Cię zatrzymało?**
            Patrz, który "wierzchołek" jest najdalej od środka.
            * Jeśli **Serce > Mięśnie**: Ograniczenie centralne (układ krążenia nie nadąża z dostawą).
            * Jeśli **Mięśnie > Serce**: Ograniczenie peryferyjne (mięśnie zużywają wszystko, co dostają, albo jest okluzja mechaniczna).
            """)
        else:
            st.warning(f"Twój trening jest krótszy niż {window_sec/60:.0f} min, więc nie możemy wyznaczyć tego okna.")
    else:
        st.error("Brakuje kluczowych danych (Watts + HR/VE/SmO2) do stworzenia radaru.")
