"""
Nutrition tab — carbohydrate oxidation, fat oxidation, and fuelling strategy.
"""
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
    
    # --- MENU KOLARSKIE (ROZBUDOWANE) ---
    with st.expander("🍬 Menu Kolarskie (Ile to węglowodanów?)", expanded=False):
        st.markdown("""
        ### Produkty Energetyczne na Rower
        
        | Produkt | CHO [g] | Szybkość wchłaniania | Uwagi |
        |---------|---------|---------------------|-------|
        | **Żel energetyczny** (1 szt.) | 25-30 | ⚡ Bardzo szybka | Glukoza/maltodekstryna, łatwy do spożycia |
        | **Baton energetyczny** | 40-50 | 🔵 Średnia | Orzech/płatki, dłuższe żucie |
        | **Banan** | 25-30 | 🟢 Średnia | Naturalny cukier + potas |
        | **Izotonik (500ml)** | 30-40 | ⚡ Szybka | Płynne, łatwe do spożycia w ruchu |
        | **Żelki (100g)** | ~75 | ⚡ Szybka | Glukoza/fruktoza mix, idealne na interwały |
        | **Rodzynki (50g)** | 35 | 🔵 Średnia | Naturalne, ale wolniejsze wchłanianie |
        | **Miód (1 łyżka)** | 20 | ⚡ Szybka | Może podrażnić żołądek |
        | **Cola (330ml)** | 35 | ⚡ Szybka | Kofeina + cukier, "emergency boost" |
        | **Daktyle (3 szt.)** | 45 | 🟢 Średnia | Naturalne, wysokie w błonnik |
        | **Ryż kleisty (100g)** | 80 | 🔵 Średnia-wolna | "Rice cakes", popularne w peletonie |
        | **Syrop klonowy (50ml)** | 50 | ⚡ Szybka | Alternatywa dla żeli |
        
        ---
        
        **💡 Pro Tip: Glukoza + Fruktoza (2:1)**
        
        Jelita mają oddzielne transportery dla glukozy (SGLT1) i fruktozy (GLUT5). 
        Łącząc oba cukry w proporcji 2:1, możesz wchłonąć nawet **90-120g/h** zamiast standardowych 60g/h samej glukozy.
        
        *Pamiętaj: Trening jelita jest równie ważny jak trening nóg! Nie testuj 90g/h pierwszy raz na zawodach.*
        """)
    
    if 'watts' in df_plot.columns:
        # --- CORRECTED MODEL: Physics-based CHO consumption ---
        # Based on XERT/INSCYD research and intervals.icu forum discussions
        
        # Step 1: Mechanical work to total energy expenditure
        # 1 kJ mechanical work ≈ 1 kcal total energy burned
        # (The ~24% efficiency factor roughly cancels the 4.184 kJ/kcal conversion)
        # Power [W] = J/s. For 1 hour: W * 3600 / 1000 = kJ/h ≈ kcal/h
        energy_kcal_per_hour = df_plot['watts'] * 3.6  # kJ/h ≈ kcal/h
        
        # Step 2: CHO fraction based on %FTP (INSCYD/XERT model)
        # Research shows:
        # - FatMax occurs around 55-65% FTP
        # - Above threshold, almost all energy from CHO
        intensity = df_plot['watts'] / cp_input if cp_input > 0 else df_plot['watts'] / 280
        
        # Piecewise linear CHO fraction model
        # Z1 (<55% FTP): ~30% CHO (fat dominant - recovery)
        # Z2 (55-75% FTP): 30-60% CHO (endurance - mixed)
        # Z3-Z4 (75-100% FTP): 60-90% CHO (tempo/threshold)
        # Z5+ (>100% FTP): 90-100% CHO (VO2max - almost all CHO)
        cho_fraction = np.where(intensity < 0.55, 0.30,
                      np.where(intensity < 0.75, 0.30 + (intensity - 0.55) * 1.5,   # 30→60%
                      np.where(intensity < 1.00, 0.60 + (intensity - 0.75) * 1.2,   # 60→90%
                      np.clip(0.90 + (intensity - 1.0) * 0.5, 0.90, 1.0))))         # 90-100%
        
        # Step 3: Calculate CHO burn rate
        # 1g CHO = 4 kcal
        cho_kcal_per_hour = energy_kcal_per_hour * cho_fraction
        carb_rate_per_sec = cho_kcal_per_hour / 4.0 / 3600.0  # Convert to g/s
        
        cumulative_burn = carb_rate_per_sec.cumsum()
        
        intake_per_sec = carb_intake / 3600.0
        cumulative_intake = np.cumsum(np.full(len(df_plot), intake_per_sec))
        
        glycogen_balance = initial_glycogen - cumulative_burn + cumulative_intake
        
        df_nutri = pd.DataFrame({
            'Czas [min]': df_plot['time_min'],
            'Bilans Glikogenu [g]': glycogen_balance,
            'Spalone [g]': cumulative_burn,
            'Spożyte [g]': cumulative_intake,
            'Burn Rate [g/h]': carb_rate_per_sec * 3600
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
        st.plotly_chart(fig_nutri, width="stretch")
        
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
        
        # Linia limitu jelitowego
        fig_burn.add_hline(y=90, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text="Limit jelitowy ~90g/h", annotation_position="bottom left")
        
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
        st.plotly_chart(fig_burn, width="stretch")

        # PODSUMOWANIE LICZBOWE
        total_burn = cumulative_burn.iloc[-1]
        total_intake = cumulative_intake[-1]
        final_balance = glycogen_balance.iloc[-1]
        avg_burn_rate = df_nutri['Burn Rate [g/h]'].mean()
        
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Spalone Węgle", f"{total_burn:.0f} g", help="Suma węglowodanów zużytych na wysiłek")
        n2.metric("Spożyte Węgle", f"{total_intake:.0f} g", help="Suma węglowodanów dostarczonych z jedzenia/napojów")
        n3.metric("Wynik Końcowy", f"{final_balance:.0f} g", delta=f"{final_balance - initial_glycogen:.0f} g", delta_color="inverse" if final_balance < 0 else "normal")
        n4.metric("Śr. Spalanie", f"{avg_burn_rate:.0f} g/h", help="Średnie tempo spalania węgli podczas treningu")
        
        if final_balance < 0:
            bonk_time = df_nutri[df_nutri['Bilans Glikogenu [g]'] < 0]['Czas [min]'].iloc[0]
            st.error(f"⚠️ **UWAGA:** Według symulacji, Twoje zapasy glikogenu wyczerpały się w okolicach {bonk_time:.0f} minuty! To oznacza ryzyko 'odcięcia' (bonk).")
        else:
            st.success(f"✅ **OK:** Zakończyłeś trening z zapasem {final_balance:.0f}g glikogenu. Strategia żywieniowa wystarczająca dla tej intensywności.")
        
        # --- TEORIA FIZJOLOGII SPALANIA (ROZBUDOWANA) ---
        with st.expander("🔬 Fizjologia Spalania Węglowodanów (Model INSCYD)", expanded=False):
            st.markdown("""
            ## Model Metaboliczny: VO2max, VLaMax, i Spalanie Węglowodanów
            
            INSCYD i WKO5 używają zaawansowanych modeli metabolicznych, które uwzględniają dwa kluczowe parametry:
            
            ### 1. VO2max (Maksymalny Pobór Tlenu)
            * Określa Twoją maksymalną zdolność aerobową (tlenową)
            * Im wyższy VO2max, tym więcej energii możesz wytworzyć z tłuszczu i węglowodanów przy udziale tlenu
            
            ### 2. VLaMax (Maksymalna Produkcja Mleczanu)
            * Określa Twoją zdolność glikolityczną (beztlenową)
            * **Wysoki VLaMax** (>0.6 mmol/L/s): Sprintery, szybkie spalanie węgli, słabsza wytrzymałość
            * **Niski VLaMax** (<0.4 mmol/L/s): Climbers, oszczędne spalanie, lepsza ekonomia tłuszczowa
            
            ---
            
            ## Strefy Spalania Paliwa (dla FTP ~280W)
            
            | Intensywność | %FTP | Moc [W] | Udział CHO | Spalanie CHO [g/h] |
            |--------------|------|---------|------------|-------------------|
            | Z1 (Recovery) | <55% | <155 | ~30% | 30-50 |
            | Z2 (Endurance) | 55-75% | 155-210 | 30-60% | 50-100 |
            | Z3 (Tempo) | 76-90% | 210-250 | 60-80% | 100-180 |
            | Z4 (Threshold) | 91-105% | 250-295 | 80-95% | 180-250 |
            | Z5/Z6 (VO2max) | >105% | >295 | 95-100% | 250-350+ |
            
            ---
            
            ## Kluczowe Koncepcje
            
            ### FatMax (Maksymalne Spalanie Tłuszczu)
            * Intensywność, przy której spalasz najwięcej tłuszczu (zwykle 55-65% FTP)
            * Powyżej tego punktu, spalanie tłuszczu spada, a węgla rośnie
            
            ### CarbMax (Maksymalne Spalanie Węgli)
            * Maksymalne tempo, w jakim Twój organizm może spalać węglowodany
            * Limitowane przez VLaMax i enzymy glikolityczne
            * Typowo: 150-250 g/h dla elitarnych sportowców
            
            ### Limity Jelitowe
            * **Sama glukoza**: max ~60 g/h absorpcji
            * **Glukoza + Fruktoza (2:1)**: max ~90-120 g/h
            * Dlatego przy intensywnych wysiłkach (>Z4) zawsze "pożyczasz" z rezerw glikogenu
            
            ---
            
            ## Strategie Żywieniowe
            
            | Strategia | Kiedy stosować | Cel |
            |-----------|----------------|-----|
            | **Train Low** | Treningi Z2, długie bazy | Poprawa adaptacji tłuszczowej |
            | **Train High** | Interwały, tempo, wyścigi | Maksymalna wydajność |
            | **Periodyzacja** | Cykl tygodniowy | Łączenie obu strategii |
            | **Sleep Low** | Po treningu wieczorem | Wzmocnienie odpowiedzi adaptacyjnej |
            
            *Ten kalkulator używa modelu opartego na fizyce i badaniach XERT/INSCYD: praca mechaniczna (kJ) → energia całkowita (kcal) × udział węglowodanów (zależny od %FTP) / 4 kcal/g.*
            """)
    else:
        st.warning("Brak danych mocy (Watts) do obliczenia wydatku energetycznego.")

    # =========================================================================
    # MULTI-FACTOR GLYCOGEN MODEL (Cadence Comparison)
    # =========================================================================
    st.divider()
    st.header("🧬 Model Zużycia Glikogenu (Multi-Factor)")
    
    _render_glycogen_model_section(df_plot, cp_input)


def _render_glycogen_model_section(df_plot, cp_input):
    """
    Renderuje sekcję modelu zużycia glikogenu z wieloma czynnikami.
    Porównuje zużycie przy różnych kadencjach (koszt metaboliczny okluzji).
    """
    import numpy as np
    from modules.calculations.nutrition import (
        calculate_glycogen_consumption,
        compare_cadence_glycogen
    )
    
    st.markdown("""
    Model zużycia glikogenu uwzględniający wiele czynników fizjologicznych:
    - **Moc (%CP)** — intensywność wysiłku
    - **Temperatura rdzenia** — hiperthermia zwiększa zużycie CHO
    - **VLaMax** — zdolność glikolotyczna
    - **Indeks okluzji** — ograniczona perfuzja = więcej beztlenowo
    - **Nachylenie SmO2** — desaturacja mięśniowa
    """)
    
    # === PARAMETRY WEJŚCIOWE ===
    st.subheader("📊 Parametry Symulacji")
    
    c1, c2, c3 = st.columns(3)
    power_sim = c1.number_input("Moc [W]", min_value=100, max_value=1000, value=300, step=10)
    cp_sim = c2.number_input("CP [W]", min_value=150, max_value=500, value=min(int(cp_input), 500) if cp_input > 0 else 280, step=10)
    core_temp = c3.number_input("Temperatura rdzenia [°C]", min_value=36.0, max_value=41.0, value=37.5, step=0.1)
    
    c4, c5, c6 = st.columns(3)
    vlamax = c4.number_input("VLaMax [mmol/L/s]", min_value=0.2, max_value=1.0, value=0.5, step=0.05)
    occlusion_idx = c5.slider("Indeks okluzji", min_value=0.0, max_value=0.5, value=0.15, step=0.05)
    smo2_slope = c6.slider("SmO2 slope [%/Nm]", min_value=-0.10, max_value=0.0, value=-0.02, step=0.01)
    
    # === OBLICZENIE DLA WYBRANEJ KADENCJI ===
    st.subheader("📈 Wynik dla Pojedynczej Kadencji")
    
    cadence_single = st.slider("Kadencja [RPM]", min_value=50, max_value=120, value=90, step=5)
    
    result = calculate_glycogen_consumption(
        power=power_sim,
        cp=cp_sim,
        core_temp=core_temp,
        vlamax=vlamax,
        occlusion_index=occlusion_idx,
        smo2_slope=smo2_slope,
        cadence=cadence_single
    )
    
    # Główny wynik
    st.markdown(f"""
    <div style="padding:20px; border-radius:12px; border:3px solid #ff7f0e; background-color: #1a1a1a; text-align:center;">
        <h2 style="margin:0; color: #ff7f0e;">CHO: {result['cho_g_per_hour']} g/h</h2>
        <p style="margin:10px 0 0 0; color:#888; font-size:0.85em;">
            Moc: {power_sim}W ({result['intensity_pct']:.0f}% CP) | Kadencja: {cadence_single} RPM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Ostrzeżenia
    for warning in result.get("warnings", []):
        st.warning(warning)
    
    # Rozbicie modyfikatorów
    with st.expander("📊 Rozbicie Modyfikatorów", expanded=False):
        breakdown = result.get("breakdown", {})
        c1, c2 = st.columns(2)
        c1.metric("CHO bazowy", f"{result['cho_base']} g/h")
        c2.metric("Modyfikator całkowity", f"×{result['total_modifier']:.3f}")
        
        st.markdown("| Czynnik | Modyfikator |")
        st.markdown("|---------|-------------|")
        for factor, value in breakdown.items():
            delta = (value - 1) * 100
            sign = "+" if delta >= 0 else ""
            st.markdown(f"| {factor.capitalize()} | ×{value:.3f} ({sign}{delta:.1f}%) |")
    
    # === PORÓWNANIE KADENCJI (Koszt Okluzji) ===
    st.divider()
    st.subheader("⚔️ Porównanie: 300W @ 60 RPM vs 300W @ 95 RPM")
    st.markdown("**Koszt metaboliczny okluzji** — jak kadencja wpływa na zużycie glikogenu?")
    
    c1, c2 = st.columns(2)
    cad_low = c1.number_input("Kadencja niska [RPM]", min_value=40, max_value=80, value=60, step=5)
    cad_high = c2.number_input("Kadencja wysoka [RPM]", min_value=80, max_value=120, value=95, step=5)
    
    # Zakładamy wyższą okluzję przy niskiej kadencji
    occlusion_low = min(0.35, occlusion_idx + 0.20)  # +20% więcej okluzji
    occlusion_high = max(0.05, occlusion_idx - 0.05)  # -5% mniej okluzji
    
    comparison = compare_cadence_glycogen(
        power=power_sim,
        cp=cp_sim,
        cadence_low=cad_low,
        cadence_high=cad_high,
        core_temp=core_temp,
        vlamax=vlamax,
        occlusion_index_low=occlusion_low,
        occlusion_index_high=occlusion_high,
        smo2_slope=smo2_slope
    )
    
    # Wyświetlanie porównania
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222; text-align:center;">
            <h3 style="margin:0; color: #ef553b;">🔴 Grinding ({cad_low:.0f} RPM)</h3>
            <h2 style="margin:5px 0;">{comparison['low_cadence']['cho_g_per_hour']} g/h</h2>
            <p style="margin:0; color:#aaa;">Moment: {comparison['low_cadence']['torque']:.0f} Nm</p>
            <p style="margin:0; color:#aaa;">Okluzja: {comparison['low_cadence']['occlusion_index']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding:15px; border-radius:8px; border:2px solid #00cc96; background-color: #222; text-align:center;">
            <h3 style="margin:0; color: #00cc96;">🟢 Spinning ({cad_high:.0f} RPM)</h3>
            <h2 style="margin:5px 0;">{comparison['high_cadence']['cho_g_per_hour']} g/h</h2>
            <p style="margin:0; color:#aaa;">Moment: {comparison['high_cadence']['torque']:.0f} Nm</p>
            <p style="margin:0; color:#aaa;">Okluzja: {comparison['high_cadence']['occlusion_index']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Delta
    delta = comparison['delta_cho_g_per_hour']
    delta_pct = comparison['delta_pct']
    
    if delta > 0:
        st.error(f"""
        🔴 **Koszt metaboliczny okluzji:** +{delta:.1f} g/h (+{delta_pct:.1f}%)
        
        Przy {cad_low:.0f} RPM zużywasz **{delta:.1f} g/h więcej węglowodanów** niż przy {cad_high:.0f} RPM dla tej samej mocy {power_sim}W.
        
        Na 3-godzinny wyścig to dodatkowe **{delta * 3:.0f} g glikogenu** — różnica między ukończeniem a "ścianą"!
        """)
    else:
        st.success(f"""
        ✅ Różnica w zużyciu CHO jest minimalna ({delta:.1f} g/h).
        """)
    
    # Wykres porównawczy
    st.markdown("### 📊 Δ Glikogen vs Kadencja")
    
    cadences = np.arange(50, 121, 5)
    cho_values = []
    
    for cad in cadences:
        # Occlusion scales with torque (inversely with cadence)
        torque = power_sim / (2 * np.pi * (cad / 60))
        occ = min(0.5, 0.05 + (torque / 100) * 0.3)  # Scale occlusion with torque
        
        res = calculate_glycogen_consumption(
            power=power_sim,
            cp=cp_sim,
            core_temp=core_temp,
            vlamax=vlamax,
            occlusion_index=occ,
            smo2_slope=smo2_slope,
            cadence=cad
        )
        cho_values.append(res['cho_g_per_hour'])
    
    fig_delta = go.Figure()
    
    fig_delta.add_trace(go.Scatter(
        x=cadences,
        y=cho_values,
        mode='lines+markers',
        name='CHO [g/h]',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=6),
        hovertemplate="Kadencja: %{x} RPM<br>CHO: %{y:.1f} g/h<extra></extra>"
    ))
    
    # Linia referencyjna dla 90 RPM
    fig_delta.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="90 RPM", annotation_position="top")
    
    fig_delta.update_layout(
        template="plotly_dark",
        title=f"Zużycie CHO vs Kadencja przy {power_sim}W",
        xaxis=dict(title="Kadencja [RPM]"),
        yaxis=dict(title="CHO [g/h]"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    st.plotly_chart(fig_delta, width="stretch")
    
    # Teoria
    with st.expander("📖 Model i Metodologia", expanded=False):
        st.markdown("""
        ### Wzory Modelu
        
        **Bazowe zużycie CHO:**
        ```
        CHO_base = f(power / CP) — krzywizna wykładnicza
        ```
        
        **Modyfikatory:**
        | Czynnik | Wpływ |
        |---------|-------|
        | Temperatura | +10% / °C powyżej 37.5°C |
        | VLaMax | ±25% przy VLaMax ±0.5 od 0.5 |
        | Okluzja | do +24% przy wysokiej okluzji |
        | SmO2 slope | skalowanie z desaturacją |
        | Kadencja | +1% / RPM poniżej 70 |
        
        **Wzór końcowy:**
        ```
        CHO_final = CHO_base × Π(modyfikatory)
        ```
        
        ---
        
        ### Mechanizm Okluzji → CHO
        
        Przy niskiej kadencji (wysokim momencie obrotowym):
        1. Ciśnienie wewnątrzmięśniowe rośnie
        2. Perfuzja kapilarna jest ograniczona
        3. Mięsień przechodzi w tryb beztlenowy
        4. Glikoliza przyspiesza → więcej CHO zużyte
        
        **Praktyczna wskazówka:** Przy tej samej mocy, wyższa kadencja oszczędza glikogen!
        """)
