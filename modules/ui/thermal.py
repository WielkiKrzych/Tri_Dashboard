import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_thermal_tab(df_plot):
    st.header("Wydajność Chłodzenia")
    
    fig_t = go.Figure()
    
    # 1. CORE TEMP (Oś Lewa)
    # Kolor pomarańczowy - symbolizuje ciepło
    if 'core_temperature_smooth' in df_plot.columns:
        fig_t.add_trace(go.Scatter(
            x=df_plot['time_min'], 
            y=df_plot['core_temperature_smooth'], 
            name='Core Temp', 
            line=dict(color='#ff7f0e', width=2), 
            hovertemplate="Temp: %{y:.2f}°C<extra></extra>"
        ))
    
    # 2. HSI - HEAT STRAIN INDEX (Oś Prawa)
    # Kolor czerwony przerywany - symbolizuje ryzyko/alarm
    if 'hsi' in df_plot.columns:
        fig_t.add_trace(go.Scatter(
            x=df_plot['time_min'], 
            y=df_plot['hsi'], 
            name='HSI', 
            yaxis="y2", # Druga oś
            line=dict(color='#d62728', width=2, dash='dot'), 
            hovertemplate="HSI: %{y:.1f}<extra></extra>"
        ))
    
    # Linie referencyjne dla temperatury (Strefy)
    fig_t.add_hline(y=38.5, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Krytyczna (38.5°C)", annotation_position="top left")
    fig_t.add_hline(y=37.5, line_dash="dot", line_color="green", opacity=0.5, annotation_text="Optymalna (37.5°C)", annotation_position="bottom left")

    # LAYOUT (Unified Hover)
    fig_t.update_layout(
        template="plotly_dark",
        title="Termoregulacja: Temperatura Głęboka vs Indeks Zmęczenia (HSI)",
        hovermode="x unified",
        
        # Oś Lewa
        yaxis=dict(title="Core Temp [°C]"),
        
        # Oś Prawa
        yaxis2=dict(
            title="HSI [0-10]", 
            overlaying="y", 
            side="right", 
            showgrid=False,
            range=[0, 12] # Lekki zapas na skali, żeby wykres nie dotykał sufitu
        ),
        
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450
    )
    
    st.plotly_chart(fig_t, use_container_width=True)
    
    st.info("""
    **🌡️ Kompendium Termoregulacji: Fizjologia i Strategia**

    **1. Fizjologiczny Koszt Ciepła (Konkurencja o Krew)**
    Twój układ krążenia to system zamknięty o ograniczonej pojemności (ok. 5L krwi). Podczas wysiłku w upale serce musi obsłużyć dwa konkurencyjne cele:
    * **Mięśnie:** Dostarczenie tlenu i paliwa (priorytet wysiłkowy).
    * **Skóra:** Oddanie ciepła przez pot i konwekcję (priorytet przeżycia).
    * **Efekt:** Mniej krwi trafia do mięśni -> Spadek VO2max -> Wzrost tętna przy tej samej mocy (Cardiac Drift). Dodatkowo, utrata osocza (pot) zagęszcza krew, zmuszając serce do cięższej pracy.

    **2. Strefy Temperaturowe (Core Temp):**
    * **36.5°C - 37.5°C:** Homeostaza. Strefa komfortu i rozgrzewki.
    * **37.5°C - 38.4°C:** **Strefa Wydajności.** Optymalna temperatura pracy mięśni (enzymy działają najszybciej). Tutaj chcesz być podczas wyścigu.
    * **> 38.5°C:** **Strefa Krytyczna ("The Meltdown").** Ośrodkowy Układ Nerwowy (mózg) zaczyna "zaciągać hamulec ręczny", redukując rekrutację jednostek motorycznych, by chronić organy przed ugotowaniem. Odczuwasz to jako nagły brak mocy ("odcięcie").

    **3. HSI (Heat Strain Index 0-10):**
    * **0-3 (Niski):** Pełen komfort. Możesz cisnąć maxa.
    * **4-6 (Umiarkowany):** Fizjologiczny koszt rośnie. Wymagane nawadnianie.
    * **7-9 (Wysoki):** Znaczący spadek wydajności. Skup się na chłodzeniu, nie na watach.
    * **10 (Ekstremalny):** Ryzyko udaru. Zwolnij natychmiast.

    **4. Protokół Chłodzenia (Strategia):**
    * **Internal (Wewnętrzne):** Pij zimne napoje (tzw. ice slurry). Obniża to temp. żołądka i core temp.
    * **External (Zewnętrzne):** Polewaj wodą głowę, kark i **nadgarstki** (duże naczynia krwionośne blisko skóry). Lód w stroju startowym (na karku/klatce) to game-changer.

    **5. Czerwone Flagi (Kiedy przerwać):**
    * Gęsia skórka lub dreszcze w upale (paradoksalna reakcja - mózg "wariuje").
    * Nagły spadek tętna przy utrzymaniu wysiłku.
    * Zaburzenia widzenia lub koordynacji.
    """)

    st.header("Koszt Termiczny Wydajności (Cardiac Drift)")
    
    # Sprawdzamy czy mamy potrzebne kolumny
    temp_col = 'core_temperature_smooth' if 'core_temperature_smooth' in df_plot.columns else 'core_temperature'
    
    if 'watts' in df_plot.columns and temp_col in df_plot.columns and 'heartrate' in df_plot.columns:
        
        # 1. FILTROWANIE DANYCH
        # Wywalamy zera i postoje
        mask = (df_plot['watts'] > 10) & (df_plot['heartrate'] > 60)
        df_clean = df_plot[mask].copy()
        
        # 2. OBLICZENIE EFEKTYWNOŚCI (EF)
        df_clean['eff_raw'] = df_clean['watts'] / df_clean['heartrate']
        
        # 3. USUWANIE OUTLIERÓW
        df_clean = df_clean[df_clean['eff_raw'] < 6.0]

        if not df_clean.empty:
            # Tworzymy wykres z linią trendu (Lowess - lokalna regresja)
            fig_te = px.scatter(
                df_clean, 
                x=temp_col, 
                y='eff_raw', 
                trendline="lowess", 
                trendline_options=dict(frac=0.3), 
                trendline_color_override="#FF4B4B", 
                template="plotly_dark",
                opacity=0.3 # Przezroczyste punkty, żeby widzieć gęstość
            )
            
            # Formatowanie punktów (Scatter)
            fig_te.update_traces(
                selector=dict(mode='markers'),
                marker=dict(size=5, color='#1f77b4'),
                hovertemplate="<b>Temp:</b> %{x:.2f}°C<br><b>EF:</b> %{y:.2f} W/bpm<extra></extra>"
            )
            
            # Formatowanie linii trendu
            fig_te.update_traces(
                selector=dict(mode='lines'),
                line=dict(width=4),
                hovertemplate="<b>Trend:</b> %{y:.2f} W/bpm<extra></extra>"
            )
            
            # LAYOUT (Unified Hover)
            fig_te.update_layout(
                title="Spadek Efektywności (W/HR) vs Temperatura",
                hovermode="x unified",
                
                xaxis=dict(title="Temperatura Głęboka [°C]"),
                yaxis=dict(title="Efficiency Factor [W/bpm]"),
                
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                height=450
            )

            st.plotly_chart(fig_te, use_container_width=True, config={'scrollZoom': False}, key="thermal_eff")
            
            st.info("""
            ℹ️ **Jak to czytać?**
            Ten wykres pokazuje **Cardiac Drift** w funkcji temperatury.
            * **Oś Y (W/HR):** Ile watów generujesz z jednego uderzenia serca. Wyższa wartość = lepsza efektywność.
            * **Oś X (Core Temp):** Twoja temperatura wewnętrzna. Wyższa wartość = większy stres cieplny.
            * **Trend spadkowy:** Oznacza, że wraz ze wzrostem temperatury Twoje serce musi bić szybciej dla tej samej mocy (krew idzie do skóry na chłodzenie = mniejszy rzut serca dla mięśni).
            * **Filtracja:** Usunąłem momenty, gdy nie pedałujesz (Moc < 10W), żeby nie zaburzać wyniku.
            """)
        else:
            st.warning("Zbyt mało danych po przefiltrowaniu (sprawdź czy masz odczyty mocy i tętna).")
    else:
        st.error("Brak wymaganych kolumn (watts, heartrate, core_temperature).")
        
        st.info("""
        **💡 Interpretacja: Koszt Fizjologiczny Ciepła (Decoupling Termiczny)**

        Ten wykres pokazuje, jak Twoje "serce płaci" za każdy wat mocy w miarę wzrostu temperatury ciała.
        * **Oś X:** Temperatura Centralna (Core Temp).
        * **Oś Y:** Efektywność (Waty na 1 uderzenie serca).
        * **Czerwona Linia:** Trend zmian.

        **🔍 Scenariusze:**
        1.  **Linia Płaska (Idealnie):** Twoja termoregulacja działa świetnie. Mimo wzrostu temperatury, serce pracuje tak samo wydajnie. Jesteś dobrze nawodniony i zaadaptowany do ciepła.
        2.  **Linia Opadająca (Typowe):** Wraz ze wzrostem temp. serce musi bić szybciej, by utrzymać tę samą moc (Dryf). Krew ucieka do skóry, by Cię chłodzić, zamiast napędzać mięśnie.
        3.  **Gwałtowny Spadek:** "Zawał termiczny" wydajności. Zazwyczaj powyżej 38.5°C. W tym momencie walczysz o przetrwanie, a nie o wynik.

        **Wniosek:** Jeśli linia leci mocno w dół, musisz poprawić chłodzenie (polewanie wodą, lód) lub strategię nawadniania przed startem.
        """)
