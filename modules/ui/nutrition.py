import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_nutrition_tab(df_plot, cp_input, vt1_watts, vt2_watts):
    st.header("⚡ Kalkulator Spalania Glikogenu (The Bonk Prediction)")
    
    # Interaktywne suwaki
    c1, c2, c3 = st.columns(3)
    carb_intake = c1.number_input("Spożycie Węglowodanów [g/h]", min_value=0, max_value=200, value=60, step=10)
    initial_glycogen = c2.number_input("Początkowy Zapas Glikogenu [g]", min_value=200, max_value=800, value=450, step=50, help="Standardowo: 400-500g dla wytrenowanego sportowca.")
    efficiency_input = c3.number_input("Sprawność Mechaniczna [%]", min_value=18.0, max_value=26.0, value=22.0, step=0.5, help="Amator: 18-21%, Pro: 23%+")
    
    # --- ZMIANA: "MENU KOLARSKIE" (CHEAT SHEET) ---
    with st.expander("🍬 Menu Kolarskie (Ile to węglowodanów?)", expanded=False):
        st.markdown("""
        Aby dostarczyć 90g węgli na godzinę, potrzebujesz np.:
        * **3 x Żel Energetyczny** (standardowo ~25-30g CHO / sztukę)
        * **1.5 Bidonu Izotonika** (standardowo ~40g CHO / 500ml)
        * **3 x Banan** (~25-30g CHO / sztukę)
        * **2 x Baton Energetyczny** (~40-50g CHO / sztukę)
        * **Garść Żelków (100g)** (~75g CHO)
        
        *Pamiętaj: Trening jelita jest równie ważny jak trening nóg! Nie testuj 90g/h pierwszy raz na zawodach.*
        """)
    
    if 'watts' in df_plot.columns:
        intensity_factor = df_plot['watts'] / cp_input
        
        # Model metaboliczny (Logika bez zmian)
        conditions = [
            (df_plot['watts'] < vt1_watts),
            (df_plot['watts'] >= vt1_watts) & (df_plot['watts'] < vt2_watts),
            (df_plot['watts'] >= vt2_watts)
        ]
        choices = [0.3, 0.8, 1.1] 
        carb_fraction = np.select(conditions, choices, default=1.0)
        
        # Obliczenia energii
        energy_kcal_sec = df_plot['watts'] / (efficiency_input/100.0) / 4184.0
        carbs_burned_per_sec = (energy_kcal_sec * carb_fraction) / 4.0
        cumulative_burn = carbs_burned_per_sec.cumsum()
        
        intake_per_sec = carb_intake / 3600.0
        cumulative_intake = np.cumsum(np.full(len(df_plot), intake_per_sec))
        
        glycogen_balance = initial_glycogen - cumulative_burn + cumulative_intake
        
        df_nutri = pd.DataFrame({
            'Czas [min]': df_plot['time_min'],
            'Bilans Glikogenu [g]': glycogen_balance,
            'Spalone [g]': cumulative_burn,
            'Spożyte [g]': cumulative_intake,
            'Burn Rate [g/h]': carbs_burned_per_sec * 3600
        })
        
        # --- WYKRES 1: BILANS GLIKOGENU ---
        fig_nutri = go.Figure()
        
        # Linia Balansu
        line_color = '#00cc96' if df_nutri['Bilans Glikogenu [g]'].min() > 0 else '#ef553b'
        
        fig_nutri.add_trace(go.Scatter(
            x=df_nutri['Czas [min]'], 
            y=df_nutri['Bilans Glikogenu [g]'], 
            name='Zapas Glikogenu', 
            fill='tozeroy', 
            line=dict(color=line_color, width=2), 
            hovertemplate="<b>Czas: %{x:.0f} min</b><br>Zapas: %{y:.0f} g<extra></extra>"
        ))
        
        # Linia "Ściana" (Bonk)
        fig_nutri.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Ściana (Bonk)", annotation_position="bottom right")
        
        fig_nutri.update_layout(
            template="plotly_dark",
            title=f"Symulacja Baku Paliwa (Start: {initial_glycogen}g, Intake: {carb_intake}g/h)",
            hovermode="x unified",
            yaxis=dict(title="Glikogen [g]"),
            xaxis=dict(title="Czas [min]", tickformat=".0f"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_nutri, use_container_width=True)
        
        # --- WYKRES 2: TEMPO SPALANIA (BURN RATE) ---
        st.subheader("🔥 Tempo Spalania (Burn Rate)")
        fig_burn = go.Figure()
        
        burn_rate_smooth = df_nutri['Burn Rate [g/h]'].rolling(window=60, center=True, min_periods=1).mean()
        
        fig_burn.add_trace(go.Scatter(
            x=df_nutri['Czas [min]'], 
            y=burn_rate_smooth, 
            name='Spalanie', 
            line=dict(color='#ff7f0e', width=2), 
            fill='tozeroy', 
            hovertemplate="<b>Czas: %{x:.0f} min</b><br>Spalanie: %{y:.0f} g/h<extra></extra>"
        ))
        
        # Linia Spożycia (Intake)
        fig_burn.add_hline(y=carb_intake, line_dash="dot", line_color="#00cc96", annotation_text=f"Intake: {carb_intake}g/h", annotation_position="top right")
        
        fig_burn.update_layout(
            template="plotly_dark",
            title="Zapotrzebowanie na Węglowodany",
            hovermode="x unified",
            yaxis=dict(title="Burn Rate [g/h]"),
            xaxis=dict(title="Czas [min]", tickformat=".0f"),
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_burn, use_container_width=True)

        # PODSUMOWANIE LICZBOWE
        total_burn = cumulative_burn.iloc[-1]
        total_intake = cumulative_intake[-1]
        final_balance = glycogen_balance.iloc[-1]
        
        n1, n2, n3 = st.columns(3)
        n1.metric("Spalone Węgle", f"{total_burn:.0f} g", help="Suma węglowodanów zużytych na wysiłek")
        n2.metric("Spożyte Węgle", f"{total_intake:.0f} g", help="Suma węglowodanów dostarczonych z jedzenia/napojów")
        n3.metric("Wynik Końcowy", f"{final_balance:.0f} g", delta=f"{final_balance - initial_glycogen:.0f} g", delta_color="inverse" if final_balance < 0 else "normal")
        
        if final_balance < 0:
            st.error(f"⚠️ **UWAGA:** Według symulacji, Twoje zapasy glikogenu wyczerpały się w okolicach {df_nutri[df_nutri['Bilans Glikogenu [g]'] < 0]['Czas [min]'].iloc[0]:.0f} minuty! To oznacza ryzyko 'odcięcia' (bonk).")
        else:
            st.success(f"✅ **OK:** Zakończyłeś trening z zapasem {final_balance:.0f}g glikogenu. Strategia żywieniowa wystarczająca dla tej intensywności.")
        
        st.info("""
        **💡 Fizjologia Spalania (Model VT1/VT2):**
        
        * **Strefa Tłuszczowa (< VT1):** Spalasz ok. **20-40g węgli/h**. Reszta to tłuszcz. Tutaj możesz jechać godzinami na samej wodzie.
        * **Strefa Mieszana (VT1 - VT2):** Spalanie węgli skacze do **60-90g/h**. Musisz zacząć jeść (żele/izotonik), żeby nie opróżniać baku.
        * **Strefa Cukrowa (> VT2):** "Turbo". Spalasz **120g/h i więcej**. Twoje jelita nie są w stanie tyle wchłonąć (max ~90g/h). Każda minuta tutaj to "pożyczka", której nie spłacisz w trakcie jazdy.
        
        *Model uwzględnia Twoją wagę, sprawność (Efficiency) oraz progi mocy.*
        """)
    else:
        st.warning("Brak danych mocy (Watts) do obliczenia wydatku energetycznego.")
