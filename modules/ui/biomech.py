"""
Biomechanical Stress tab — cadence, torque, and neuro-muscular load.
"""
import streamlit as st
import plotly.graph_objects as go
from modules.plots import CHART_CONFIG

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
        
        st.plotly_chart(fig_b, width="stretch", config=CHART_CONFIG)
        
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
        
        st.plotly_chart(fig_ts, width="stretch", config=CHART_CONFIG)
        
        with st.expander("📖 Moment vs Oksydacja (SmO2) — Teoria i Fizjologia", expanded=False):
            st.markdown("""
### Definicja: SmO2 a Moment Obrotowy

**SmO2** (muscle oxygen saturation) mierzony przez NIRS na vastus lateralis odzwierciedla balans między dostawą a zużyciem O₂ w mikrokrażeniu mięśniowym (Yogev et al., 2023).

**Mechanizm:** wzrost momentu obrotowego → wzrost siły skurczu → wzrost ciśnienia wewnątrzmięśniowego (intramuscular pressure, IMP) → kompresja naczyń włosowatych → redukcja przepływu krwi → spadek SmO2.

Jak zauważyli Kilgas et al. (2022): *"blood flow would increase only if the intramuscular pressure generated during the muscle contraction exceeds the cuff pressure"* — analogicznie, wysoki moment działa jak **wewnętrzna mankieta BFR**, ograniczając perfuzję od środka.

---

### Tabela Interpretacji Trendu SmO2

| Trend SmO2 | Zakres | Interpretacja |
|---|---|---|
| **Stabilne** | ±5% | Równowaga O₂ delivery/consumption — praca tlenowa |
| **Spadek łagodny** | -5 do -15% | Wzrost ekstrakcji O₂, typowy powyżej VT1 (Feldmann et al., 2022) |
| **Spadek umiarkowany** | -15 do -25% | Znacząca deoksygenacja — zbliżenie do progu krytycznego (Iskra & Paravlić, 2025) |
| **Spadek gwałtowny** | >25% | Prawie całkowita okluzja włośniczkowa — praca beztlenowa |

---

### 4 Mechanizmy Fizjologiczne

**1. Intramuscular Pressure (IMP)**
Podczas skurczu izometrycznego i koncentrycznego, IMP może przekroczyć 200-300 mmHg w vastus lateralis (Kilgas et al., 2022), znacznie przewyższając ciśnienie skurczowe (~120 mmHg) → mechaniczna okluzja naczyń.

**2. 4-Fazowy Model Deoksygenacji (Feldmann, 2022)**
(I) stabilna bazowa → (II) pierwszy breakpoint BP1 ≈ VT1 → (III) strome spadki → (IV) plateau/minimum. Każda faza odpowiada innemu reżimowi metabolicznemu.

**3. Wpływ Kadencji na IMP**
Wyższa kadencja = krótszy czas skurczu = niższe szczytowe IMP = lepsza perfuzja między pedałowaniami. Kilgas et al. (2022) pokazał że BFR 80% AOP powoduje ~40% spadek torque — analogicznie wysoki moment powoduje autookluzję.

**4. Reoksygenacja po Wysiłku**
Arnold et al. (2024) pokazał że half-recovery time (HRT) SmO2 zależy od intensywności: 8s (50% Wpeak) → 12s (75%) → 17s (100%). Wolniejsza reoksygenacja = większy deficyt O₂.

---

### Wskazówki Praktyczne

- Szukaj **"punktu załamania"** na wykresie — moment, od którego SmO2 zaczyna gwałtownie spadać
- Porównaj ten punkt z kadencją: ten sam watt przy niższej kadencji = wyższy moment = niższe SmO2
- **Zalecenie:** utrzymuj kadencję powyżej progu krytycznego (patrz sekcja Analiza Ryzyka Okluzji poniżej)

---

### Bibliografia

- Yogev et al. (2023). The effect of severe intensity bouts on muscle oxygen saturation responses in trained cyclists. *Frontiers in Sports and Active Living*, 5, 1086227.
- Feldmann et al. (2022). Muscle oxygen saturation breakpoints reflect ventilatory thresholds in both cycling and running. *Journal of Human Kinetics*, 83, 87–97.
- Iskra & Paravlić (2025). Assessing the feasibility of NIRS for evaluating physiological exercise thresholds. *Scientific Reports*, 15, 14920.
- Kilgas et al. (2022). Physiological responses to acute cycling with blood flow restriction. *Frontiers in Physiology*, 13, 800155.
- Arnold et al. (2024). Muscle reoxygenation is slower after higher cycling intensity. *Frontiers in Physiology*, 15, 1449384.
- Sendra-Pérez et al. (2024). Profiles of muscle-specific oxygenation responses and thresholds during graded cycling incremental test. *European Journal of Applied Physiology*.
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
    st.dataframe(df_table, width="stretch", hide_index=True)
    
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
    
    st.plotly_chart(fig_heatmap, width="stretch", config=CHART_CONFIG)
    
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
    with st.expander("📖 Mapa Ryzyka Okluzji — Teoria i Fizjologia", expanded=False):
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

Ta relacja jest fundamentem strategii kadencyjnej — przy danej mocy, jedyną drogą do obniżenia momentu (i IMP) jest zwiększenie kadencji.

---

### Fizjologia Okluzji

Podczas cyklu pedałowania, **faza Propulsive (push-down)** generuje skurcz koncentryczny z ciśnieniem wewnątrzmięśniowym (IMP) dochodzącym do 200+ mmHg (Kilgas et al., 2022).

**Faza Recovery** pozwala na reperfuzję — czas trwania tej fazy zależy od kadencji:
- Wyższa kadencja = krótszy czas pod IMP > ciśnieniem tętniczym = lepsza okluzyjna "szczelina" na przepływ krwi
- Niższa kadencja = dłuższa kompresja naczyń = większa akumulacja metabolitów

NIRS (Moxy Monitor) mierzy SmO2 na głębokość ~12.5mm w vastus lateralis — odzwierciedla mikrokrążenie w głównym mięśniu napędowym (Arnold et al., 2024).

---

### Progi Okluzji SmO2

| Próg SmO2 | Zmiana vs Baseline | Interpretacja Fizjologiczna |
|---|---|---|
| **Baseline** | 0% (ok. 60-70%) | Pełna perfuzja, praca tlenowa poniżej VT1 |
| **-10% (SmO2)** | Spadek ~10 pkt | Początek znaczącej deoksygenacji — odpowiada okolicom VT1 (Feldmann et al., 2022). Wzrost ekstrakcji O₂, perfuzja nadal wystarczająca |
| **-20% (SmO2)** | Spadek ~20 pkt | Krytyczna hipoksja — zbliżenie do VT2/RCP (Iskra & Paravlić, 2025). Balans O₂ delivery/consumption silnie zaburzony |
| **>25% spadek** | >25 pkt poniżej baseline | Prawie całkowita okluzja — praca czysto beztlenowa. Reoksygenacja po wysiłku bardzo wolna (HRT >17s wg Arnold et al., 2024) |

---

### Interpretacja Mapy Ryzyka

- 🟢 **Zielona strefa (niskie ryzyko):** Kadencja powyżej wartości bezpiecznej dla danej strefy mocy. Pełna reperfuzja między skurczami. Można utrzymać długo.
- 🟠 **Pomarańczowa strefa (umiarkowane ryzyko):** Kadencja w okolicach progu -10% SmO2. Ograniczona perfuzja, kumulacja metabolitów. Można utrzymać przez kilka minut — typowe dla interwałów VO2max.
- 🔴 **Czerwona strefa (krytyczne ryzyko):** Kadencja poniżej progu krytycznego (-20% SmO2). Częściowa/całkowita okluzja naczyń. Gwałtowna kumulacja H⁺, Pi, spadek siły. Typowe dla sprintów i ataków — krótkotrwałe zjawisko.

---

### Kluczowe Badania

- **Kilgas et al. (2022)** pokazał że BFR 60% AOP vs 80% AOP daje dramatycznie różne odpowiedzi neuromuskularne (18% vs 40% spadek torque) — analogicznie, różnica kadencji może przesunąć Cię ze strefy zielonej do czerwonej.
- **Yogev et al. (2023)** udowodnił dobrą repeatability SmO2 między sesjami (ICC > 0.75) przy intensywności severe — mapa ryzyka jest stabilna między treningami.
- **Sendra-Pérez et al. (2024)** pokazał różnice w profilach deoksygenacji między mięśniami — vastus lateralis vs rectus femoris mają różne progi. Warto pamiętać że nasza mapa dotyczy VL.

---

### Bibliografia

- Yogev et al. (2023). *Frontiers in Sports and Active Living*, 5, 1086227.
- Feldmann et al. (2022). *Journal of Human Kinetics*, 83, 87–97.
- Iskra & Paravlić (2025). *Scientific Reports*, 15, 14920.
- Kilgas et al. (2022). *Frontiers in Physiology*, 13, 800155.
- Arnold et al. (2024). *Frontiers in Physiology*, 15, 1449384.
- Sendra-Pérez et al. (2024). *European Journal of Applied Physiology*.
        """)
