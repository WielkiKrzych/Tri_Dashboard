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
