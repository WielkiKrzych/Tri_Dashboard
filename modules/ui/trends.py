import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_trends_tab(df_plot):
    st.header("Trendy")
    
    if 'watts_smooth' in df_plot.columns and 'heartrate_smooth' in df_plot.columns:
        # Przygotowanie danych do ścieżki (Rolling Average 5 min)
        df_trend = df_plot.copy()
        df_trend['w_trend'] = df_trend['watts'].rolling(window=300, min_periods=60).mean()
        df_trend['hr_trend'] = df_trend['heartrate'].rolling(window=300, min_periods=60).mean()
        
        # Próbkowanie co 60 wierszy (co minutę), żeby nie zamulić wykresu tysiącami kropek
        df_path = df_trend.iloc[::60, :]
        
        fig_d = go.Figure()
        
        fig_d.add_trace(go.Scatter(
            x=df_path['w_trend'], 
            y=df_path['hr_trend'], 
            mode='markers+lines', 
            name='Ścieżka',
            # Kolorowanie wg czasu (Gradient)
            marker=dict(
                size=8, 
                color=df_path['time_min'], 
                colorscale='Viridis', 
                showscale=True, 
                colorbar=dict(title="Czas [min]"),
                line=dict(width=1, color='white')
            ),
            line=dict(color='rgba(255,255,255,0.3)', width=1), # Cienka linia łącząca
            
            # Bogaty Tooltip (Stylizowany jak w innych zakładkach)
            hovertemplate="<b>Czas: %{marker.color:.0f} min</b><br>" +
                          "Moc (5min): %{x:.0f} W<br>" +
                          "HR (5min): %{y:.0f} BPM<extra></extra>"
        ))
        
        fig_d.update_layout(
            template="plotly_dark",
            title="Ścieżka Dryfu: Relacja Moc vs Tętno w Czasie",
            
            # Tutaj używamy 'closest', bo oś X to Moc, a nie czas. 
            # 'x unified' zrobiłoby bałagan pokazując wszystkie momenty z tą samą mocą na raz.
            hovermode="closest", 
            
            xaxis=dict(title="Moc (Średnia 5 min) [W]"),
            yaxis=dict(title="Tętno (Średnia 5 min) [BPM]"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=500
        )
        
        st.plotly_chart(fig_d, use_container_width=True)
        
        st.info("""
        **💡 Interpretacja Ścieżki:**
        * **Pionowo w górę:** Czysty dryf tętna (rosnące zmęczenie przy stałej mocy). Związane jest to z odwodnieniem lub nagromadzeniem ciepła. Zazwyczaj obserwowane w długotrwałych wysiłkach (>60 min) w ciepłych warunkach. Protip: nawadniaj się regularnie i stosuj chłodzenie.
        * **Poziomo w prawo:** Zwiększenie mocy bez wzrostu tętna. Oznacza poprawę efektywności (np. zjazd, lepsza aerodynamika, wiatr w plecy).
        * **Poziomo w lewo:** Spadek mocy przy stałym tętnie. Może wskazywać na zmęczenie mięśniowe lub pogorszenie warunków (podjazd pod wiatr).
        * **W lewo i w dół:** Niekorzystna reakcja organizmu (spadek mocy i tętna) - możliwe początki wyczerpania energetycznego lub przegrzania.
        * **W prawo i w górę:** Zdrowa reakcja na zwiększenie intensywności. Twoje ciało efektywnie dostosowuje się do rosnącego wysiłku. Oznaka odpowiedniego poziomu wytrenowania.
        """)
    else:
        st.warning("Brak danych mocy i tętna do wygenerowania ścieżki dryfu.")

    st.divider()
    st.subheader("Analiza Kwadrantowa 3D")
    if 'torque' in df_plot.columns and 'cadence' in df_plot.columns and 'watts' in df_plot.columns:
        df_q = df_plot.sample(min(len(df_plot), 5000))
        color_col = 'smo2_smooth' if 'smo2_smooth' in df_q.columns else 'watts'
        title_col = 'SmO2' if 'smo2_smooth' in df_q.columns else 'Moc'
        scale = 'Spectral' if 'smo2_smooth' in df_q.columns else 'Viridis'
        
        fig_3d = px.scatter_3d(df_q, x='cadence', y='torque', z='watts', color=color_col, title=f"3D Quadrant Analysis (Kolor: {title_col})", labels={'cadence': 'Kadencja', 'torque': 'Moment', 'watts': 'Moc'}, color_continuous_scale=scale, template='plotly_dark')
        fig_3d.update_traces(marker=dict(size=3, opacity=0.6), hovertemplate="Kadencja: %{x:.0f}<br>Moment: %{y:.1f}<br>Moc: %{z:.0f}<br>Val: %{marker.color:.1f}<extra></extra>")
        # W 3D używamy wbudowanego w px template, więc tylko update layout dla wysokości
        fig_3d.update_layout(height=700) 
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("""
        **💡 Jak czytać ten wykres 3D? (Instrukcja i Przykłady)**

        Ten wykres to "mapa Twojego silnika". Każdy punkt to jedna sekunda jazdy.
        * **Oś X (Kadencja):** Szybkość obrotu korbą.
        * **Oś Y (Moment):** Siła nacisku na pedał.
        * **Oś Z (Wysokość - Moc):** Wynik końcowy (Siła x Szybkość).
        * **Kolor (SmO2):** Poziom tlenu w mięśniu (Czerwony = Niedotlenienie, Niebieski = Komfort).

        **🔍 Przykłady z Życia (Szukaj tych obszarów na wykresie):**
        1.  **"Młynek" (Prawa Strona, Nisko):** Wysoka kadencja, niski moment. To jazda ekonomiczna (np. na płaskim). Punkty powinny być **niebieskie/zielone** (dobre ukrwienie, "pompa mięśniowa" działa).
        2.  **"Przepychanie" (Lewa Strona, Wysoko):** Niska kadencja, duża siła (np. sztywny podjazd na twardym przełożeniu). Mięśnie są napięte, krew nie dopływa. Punkty mogą być **czerwone** (hipoksja/okluzja). To męczy mięśnie szybciej niż serce.
        3.  **Sprint (Prawy Górny Róg, Wysoko w górę):** Max kadencja i max siła. Generujesz szczytową moc (Oś Z). To stan beztlenowy, punkty szybko zmienią się na **czerwone**.
        4.  **Jazda w Grupie (Środek):** Umiarkowana kadencja i siła. To Twój "Sweet Spot" biomechaniczny.

        **Wniosek:** Jeśli widzisz dużo czerwonych punktów przy niskiej kadencji, zredukuj bieg i kręć szybciej, aby dotlenić nogi!
        """)
    else:
        st.warning("Brak wymaganych danych (moment, kadencja, moc) do analizy 3D.")
