"""
Biomechanical Stress tab — cadence, torque, and neuro-muscular load.
"""
import streamlit as st
import plotly.graph_objects as go

def render_biomech_tab(df_plot, df_plot_resampled):
    st.header("Biomechaniczny Stres")
    
    if 'torque_smooth' in df_plot_resampled.columns:
        fig_b = go.Figure()
        
        # 1. MOMENT OBROTOWY (Oś Lewa)
        # Kolor różowy/magenta - symbolizuje napięcie/siłę
        fig_b.add_trace(go.Scatter(
            x=df_plot_resampled['time_min'], 
            y=df_plot_resampled['torque_smooth'], 
            name='Moment (Torque)', 
            line=dict(color='#e377c2', width=1.5), 
            hovertemplate="Moment: %{y:.1f} Nm<extra></extra>"
        ))
        
        # 2. KADENCJA (Oś Prawa)
        # Kolor cyan/turkus - symbolizuje szybkość/obroty
        if 'cadence_smooth' in df_plot_resampled.columns:
            fig_b.add_trace(go.Scatter(
                x=df_plot_resampled['time_min'], 
                y=df_plot_resampled['cadence_smooth'], 
                name='Kadencja', 
                yaxis="y2", # Druga oś
                line=dict(color='#19d3f3', width=1.5), 
                hovertemplate="Kadencja: %{y:.0f} RPM<extra></extra>"
            ))
        
        # LAYOUT (Unified Hover)
        fig_b.update_layout(
            template="plotly_dark",
            title="Analiza Generowania Mocy (Siła vs Szybkość)",
            hovermode="x unified",
            
            # Oś X - Czas
            xaxis=dict(
                title="Czas [min]",
                tickformat=".0f",
                hoverformat=".0f"
            ),
            
            # Oś Lewa
            yaxis=dict(title="Moment [Nm]"),
            
            # Oś Prawa
            yaxis2=dict(
                title="Kadencja [RPM]", 
                overlaying="y", 
                side="right", 
                showgrid=False
            ),
            
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            height=450
        )
        
        st.plotly_chart(fig_b, use_container_width=True)
        
        st.info("""
        **💡 Kompendium: Moment Obrotowy (Siła) vs Kadencja (Szybkość)**

        Wykres pokazuje, w jaki sposób generujesz moc.
        Pamiętaj: `Moc = Moment x Kadencja`. Tę samą moc (np. 200W) możesz uzyskać "siłowo" (50 RPM) lub "szybkościowo" (100 RPM).

        **1. Interpretacja Stylu Jazdy:**
        * **Grinding (Niska Kadencja < 70, Wysoki Moment):**
            * **Fizjologia:** Dominacja włókien szybkokurczliwych (beztlenowych). Szybkie zużycie glikogenu.
            * **Skutek:** "Betonowe nogi" na biegu.
            * **Ryzyko:** Przeciążenie stawu rzepkowo-udowego (ból kolan) i odcinka lędźwiowego.
        * **Spinning (Wysoka Kadencja > 90, Niski Moment):**
            * **Fizjologia:** Przeniesienie obciążenia na układ krążenia (serce i płuca). Lepsze ukrwienie mięśni (pompa mięśniowa).
            * **Skutek:** Świeższe nogi do biegu (T2).
            * **Wyzwanie:** Wymaga dobrej koordynacji nerwowo-mięśniowej (żeby nie podskakiwać na siodełku).

        **2. Praktyczne Przykłady (Kiedy co stosować?):**
        * **Podjazd:** Naturalna tendencja do spadku kadencji. **Błąd:** "Przepychanie" na twardym biegu. **Korekta:** Zredukuj bieg, utrzymaj 80+ RPM, nawet jeśli prędkość spadnie. Oszczędzisz mięśnie.
        * **Płaski odcinek (TT):** Utrzymuj "Sweet Spot" kadencji (zazwyczaj 85-95 RPM). To balans między zmęczeniem mięśniowym a sercowym.
        * **Finisz / Atak:** Chwilowe wejście w wysoki moment I wysoką kadencję. Kosztowne energetycznie, ale daje max prędkość.

        **3. Możliwe Komplikacje i Sygnały Ostrzegawcze:**
        * **Ból przodu kolana:** Zbyt duży moment obrotowy (za twarde przełożenia). -> Zwiększ kadencję.
        * **Ból bioder / "skakanie":** Zbyt wysoka kadencja przy słabej stabilizacji (core). -> Wzmocnij brzuch lub nieco zwolnij obroty.
        * **Drętwienie stóp:** Często wynik ciągłego nacisku przy niskiej kadencji. Wyższa kadencja poprawia krążenie (faza luzu w obrocie).
        """)
    
    st.divider()
    st.subheader("Wpływ Momentu na Oksydację (Torque vs SmO2)")
    
    if 'torque' in df_plot.columns and 'smo2' in df_plot.columns:
        # Przygotowanie danych (Binning)
        df_bins = df_plot.copy()
        # Grupujemy moment co 2 Nm
        df_bins['Torque_Bin'] = (df_bins['torque'] // 2 * 2).astype(int)
        
        # Liczymy statystyki dla każdego koszyka
        bin_stats = df_bins.groupby('Torque_Bin')['smo2'].agg(['mean', 'std', 'count']).reset_index()
        # Filtrujemy szum (musi być min. 10 próbek dla danej siły)
        bin_stats = bin_stats[bin_stats['count'] > 10]
        
        fig_ts = go.Figure()
        
        # 1. GÓRNA GRANICA (Mean + STD) - Niewidoczna linia, potrzebna do cieniowania
        fig_ts.add_trace(go.Scatter(
            x=bin_stats['Torque_Bin'], 
            y=bin_stats['mean'] + bin_stats['std'], 
            mode='lines', 
            line=dict(width=0), 
            showlegend=False, 
            name='Górny zakres (+1SD)',
            hovertemplate="Max (zakres): %{y:.1f}%<extra></extra>"
        ))
        
        # 2. DOLNA GRANICA (Mean - STD) - Wypełnienie
        fig_ts.add_trace(go.Scatter(
            x=bin_stats['Torque_Bin'], 
            y=bin_stats['mean'] - bin_stats['std'], 
            mode='lines', 
            line=dict(width=0), 
            fill='tonexty', # Wypełnia do poprzedniej ścieżki (Górnej granicy)
            fillcolor='rgba(255, 75, 75, 0.15)', # Lekka czerwień
            showlegend=False, 
            name='Dolny zakres (-1SD)',
            hovertemplate="Min (zakres): %{y:.1f}%<extra></extra>"
        ))
        
        # 3. ŚREDNIA (Główna Linia)
        fig_ts.add_trace(go.Scatter(
            x=bin_stats['Torque_Bin'], 
            y=bin_stats['mean'], 
            mode='lines+markers', 
            name='Średnie SmO2', 
            line=dict(color='#FF4B4B', width=3), 
            marker=dict(size=6, color='#FF4B4B', line=dict(width=1, color='white')),
            hovertemplate="<b>Śr. SmO2:</b> %{y:.1f}%<extra></extra>"
        ))
        
        # LAYOUT (Unified Hover)
        fig_ts.update_layout(
            template="plotly_dark",
            title="Agregacja: Jak Siła (Moment) wpływa na Tlen (SmO2)?",
            hovermode="x unified",
            xaxis=dict(title="Moment Obrotowy [Nm]"),
            yaxis=dict(title="SmO2 [%]"),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            height=450
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        st.info("""
        **💡 Fizjologia Okluzji (Analiza Koszykowa):**
        
        **Mechanizm Okluzji:** Kiedy mocno napinasz mięsień (wysoki moment), ciśnienie wewnątrzmięśniowe przewyższa ciśnienie w naczyniach włosowatych. Krew przestaje płynąć, tlen nie dociera, a metabolity (kwas mlekowy) nie są usuwane. To "duszenie" mięśnia od środka.
        
        **Punkt Krytyczny:** Szukaj momentu (na osi X), gdzie czerwona linia gwałtownie opada w dół. To Twój limit siłowy. Powyżej tej wartości generujesz waty 'na kredyt' beztlenowy.
        
        **Praktyczny Wniosek (Scenario):** * Masz do wygenerowania 300W. Możesz to zrobić siłowo (70 RPM, wysoki moment) lub kadencyjnie (90 RPM, niższy moment).
        * Spójrz na wykres: Jeśli przy momencie odpowiadającym 70 RPM Twoje SmO2 spada do 30%, a przy momencie dla 90 RPM wynosi 50% -> **Wybierz wyższą kadencję!** Oszczędzasz nogi (glikogen) kosztem nieco wyższego tętna.
        """)

    # =========================================================================
    # NOWA SEKCJA: ANALIZA RYZYKA OKLUZJI (KADENCJA MINIMALNA)
    # =========================================================================
    st.divider()
    st.subheader("🚨 Analiza Ryzyka Okluzji (Kadencja Minimalna)")
    
    _render_occlusion_cadence_analysis(df_plot)


def _render_occlusion_cadence_analysis(df_plot):
    """
    Renderuje analizę minimalnej bezpiecznej kadencji dla uniknięcia okluzji.
    
    Na podstawie relacji: Torque = Power / (2π × Cadence)
    i progów okluzji SmO2 (-10%, -20%)
    """
    import numpy as np
    import pandas as pd
    from modules.calculations.biomech_occlusion import (
        analyze_biomech_occlusion, 
        minimal_safe_cadence,
        calculate_power_zone_cadences
    )
    
    # Walidacja danych
    if 'torque' not in df_plot.columns or 'smo2' not in df_plot.columns:
        st.warning("⚠️ **Brak danych momentu obrotowego lub SmO2** — nie można przeprowadzić analizy okluzji kadencji.")
        return
    
    # Przeprowadź analizę okluzji
    torque = df_plot['torque'].values
    smo2 = df_plot['smo2'].values
    cadence = df_plot['cadence'].values if 'cadence' in df_plot.columns else None
    
    profile = analyze_biomech_occlusion(torque, smo2, cadence)
    
    if profile.data_points < 30:
        st.warning("⚠️ **Za mało danych** do analizy okluzji kadencji.")
        return
    
    # Definicja stref mocy (uproszczona, można rozszerzyć)
    # Zakładamy typowe strefy dla kolarza ~280W CP
    power_zones = {
        "Z1 Recovery": (0, 100),
        "Z2 Endurance": (100, 180),
        "Z3 Tempo": (180, 220),
        "Z4 Threshold": (220, 260),
        "Z5 VO2max": (260, 320),
        "Z6 Anaerobic": (320, 400),
    }
    
    # Oblicz minimalne kadencje dla każdej strefy
    zone_cadences = calculate_power_zone_cadences(
        power_zones,
        profile.torque_at_minus_10,
        profile.torque_at_minus_20
    )
    
    # === TABELA MINIMALNYCH KADENCJI ===
    st.markdown("### 📊 Minimalna Kadencja dla Uniknięcia Okluzji")
    
    # Przygotowanie danych do tabeli
    table_data = []
    for zc in zone_cadences:
        table_data.append({
            "Strefa": zc["zone"],
            "Zakres Mocy": zc["power_range"],
            "Kadencja Bezpieczna (SmO2 -10%)": f"{zc['cadence_safe']:.0f} rpm" if isinstance(zc['cadence_safe'], float) else zc['cadence_safe'],
            "Kadencja Krytyczna (SmO2 -20%)": f"{zc['cadence_critical']:.0f} rpm" if isinstance(zc['cadence_critical'], float) else zc['cadence_critical'],
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # === MAPA RYZYKA (Power × Cadence) ===
    st.markdown("### 🌡️ Mapa Ryzyka Okluzji (Power × Cadence)")
    
    # Generuj heatmap
    powers = np.arange(100, 401, 20)  # 100W to 400W
    cadences = np.arange(50, 121, 5)   # 50 to 120 RPM
    
    risk_matrix = np.zeros((len(cadences), len(powers)))
    
    for i, cad in enumerate(cadences):
        for j, pwr in enumerate(powers):
            # Oblicz moment dla danej mocy i kadencji
            if cad > 0:
                torque_at_point = pwr / (2 * np.pi * (cad / 60))
            else:
                torque_at_point = 999
            
            # Klasyfikacja ryzyka na podstawie progów
            if profile.torque_at_minus_20 > 0 and torque_at_point >= profile.torque_at_minus_20:
                risk_matrix[i, j] = 3  # Krytyczne
            elif profile.torque_at_minus_10 > 0 and torque_at_point >= profile.torque_at_minus_10:
                risk_matrix[i, j] = 2  # Umiarkowane
            else:
                risk_matrix[i, j] = 1  # Niskie
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=risk_matrix,
        x=powers,
        y=cadences,
        colorscale=[
            [0, "#27AE60"],      # Niskie - zielony
            [0.5, "#F39C12"],    # Umiarkowane - pomarańczowy
            [1, "#E74C3C"]       # Krytyczne - czerwony
        ],
        showscale=False,
        hovertemplate="Moc: %{x}W<br>Kadencja: %{y} rpm<br>Ryzyko: %{z}<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        template="plotly_dark",
        title="Strefy Ryzyka Okluzji",
        xaxis=dict(title="Moc [W]"),
        yaxis=dict(title="Kadencja [RPM]"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    # Dodaj legendę jako adnotacje
    fig_heatmap.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="🟢 Niskie | 🟠 Umiarkowane | 🔴 Krytyczne",
        showarrow=False, font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.5)", borderpad=4
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # === WNIOSKI ===
    st.markdown("### 📋 Wnioski")
    
    # Znajdź przykładowe punkty krytyczne
    critical_examples = []
    for zc in zone_cadences:
        if isinstance(zc['cadence_critical'], float) and zc['cadence_critical'] > 0:
            critical_examples.append(
                f"**{zc['zone']}** ({zc['power_mid']:.0f}W): poniżej **{zc['cadence_critical']:.0f} rpm**"
            )
    
    if critical_examples:
        st.error(f"""
        🔴 **Krytyczna okluzja występuje przy:**
        
        {chr(10).join(['- ' + ex for ex in critical_examples[:3]])}
        
        **Rekomendacja:** Utrzymuj kadencję powyżej progów krytycznych, szczególnie przy wysokich mocach.
        """)
    
    # Próg ogólny
    if profile.torque_at_minus_10 > 0:
        st.warning(f"""
        ⚠️ **Próg okluzji umiarkowanej (SmO2 -10%):** {profile.torque_at_minus_10:.0f} Nm
        
        Przy tym momencie obrotowym SmO2 spada o 10% poniżej baseline. Przekroczenie tego progu oznacza wejście w strefę ograniczonej perfuzji mięśniowej.
        """)
    
    if profile.torque_at_minus_20 > 0:
        st.error(f"""
        🔴 **Próg okluzji krytycznej (SmO2 -20%):** {profile.torque_at_minus_20:.0f} Nm
        
        Poniżej tej kadencji następuje znacząca hipoksja mięśniowa. Praca w tej strefie prowadzi do szybkiej akumulacji mleczanu i przedwczesnego zmęczenia.
        """)
    
    # Zielone światło jeśli niska okluzja
    if profile.classification == "low":
        st.success("""
        ✅ **Profil niskiej okluzji:** Twoje mięśnie dobrze tolerują wysokie momenty obrotowe.
        Możesz stosować zarówno styl siłowy jak i kadencyjny w zależności od terenu i taktyki.
        """)
    
    # Teoria
    with st.expander("📖 Wzory i metodologia", expanded=False):
        st.markdown("""
        ### Relacja Moment-Moc-Kadencja
        
        ```
        Torque [Nm] = Power [W] / (2π × Cadence [RPS])
        
        gdzie RPS = RPM / 60
        ```
        
        Przekształcając:
        ```
        Cadence_min [RPM] = Power [W] / (2π × Torque_threshold [Nm]) × 60
        ```
        
        ---
        
        ### Progi Okluzji SmO2
        
        | Próg | Znaczenie |
        |------|-----------|
        | **-10% SmO2** | Początek ograniczonej perfuzji |
        | **-20% SmO2** | Krytyczna hipoksja mięśniowa |
        
        ---
        
        ### Interpretacja Mapy Ryzyka
        
        - **🟢 Zielona strefa:** Bezpieczna kombinacja power × cadence
        - **🟠 Pomarańczowa strefa:** Umiarkowane ryzyko okluzji
        - **🔴 Czerwona strefa:** Krytyczne ryzyko okluzji — unikaj!
        """)
