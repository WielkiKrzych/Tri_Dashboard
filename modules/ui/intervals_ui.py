"""
Intervals tab — Pulse Power and Gross Efficiency analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render_intervals_tab(df_plot, df_plot_resampled, cp_input, rider_weight, rider_age, is_male):
    from scipy import stats

    # --- PULSE POWER (EFICIENCY) ---
    st.subheader("🫀 Pulse Power (Moc na Uderzenie Serca)")

    if (
        "watts_smooth" in df_plot_resampled.columns
        and "heartrate_smooth" in df_plot_resampled.columns
    ):
        mask_pp = (df_plot_resampled["watts_smooth"] > 50) & (
            df_plot_resampled["heartrate_smooth"] > 90
        )
        df_pp = df_plot_resampled[mask_pp].copy()

        if not df_pp.empty:
            df_pp["pulse_power"] = df_pp["watts_smooth"] / df_pp["heartrate_smooth"]

            df_pp["pp_smooth"] = df_pp["pulse_power"].rolling(window=12, center=True).mean()
            x_pp = df_pp["time_min"]
            y_pp = df_pp["pulse_power"]
            valid_idx = np.isfinite(x_pp) & np.isfinite(y_pp)

            if valid_idx.sum() > 100:
                slope_pp, intercept_pp, _, _, _ = stats.linregress(x_pp[valid_idx], y_pp[valid_idx])
                trend_line_pp = intercept_pp + slope_pp * x_pp
                total_drop = (
                    (trend_line_pp.iloc[-1] - trend_line_pp.iloc[0]) / trend_line_pp.iloc[0] * 100
                )
            else:
                slope_pp = 0
                total_drop = 0
                trend_line_pp = None

            avg_pp = df_pp["pulse_power"].mean()

            c_pp1, c_pp2, c_pp3 = st.columns(3)
            c_pp1.metric(
                "Średnie Pulse Power",
                f"{avg_pp:.2f} W/bpm",
                help="Ile watów generuje jedno uderzenie serca.",
            )

            drift_color = "normal"
            if total_drop < -5:
                drift_color = "inverse"

            c_pp2.metric(
                "Zmiana Efektywności (Trend)", f"{total_drop:.1f}%", delta_color=drift_color
            )
            c_pp3.metric(
                "Interpretacja", "Stabilna Wydolność" if total_drop > -5 else "Dryf / Zmęczenie"
            )

            # Manual CCI Breakpoint input for PDF report
            st.caption("📊 Manualny próg CCI dla raportu PDF:")
            cci_breakpoint_manual = st.number_input(
                "CCI Breakpoint (W)",
                min_value=0,
                max_value=600,
                value=0,
                step=5,
                key="cci_breakpoint_manual",
                help="Moc przy której załamuje się Pulse Power. Wartość 0 = użyj automatycznie wykrytego.",
            )

            fig_pp = go.Figure()

            fig_pp.add_trace(
                go.Scatter(
                    x=df_pp["time_min"],
                    y=df_pp["pp_smooth"],
                    customdata=df_pp["watts_smooth"],
                    name="Pulse Power (W/bpm)",
                    mode="lines",
                    line=dict(color="#FFD700", width=2),  # Złoty kolor
                    hovertemplate="Pulse Power: %{y:.2f} W/bpm<br>Moc: %{customdata:.0f} W<extra></extra>",
                )
            )

            if trend_line_pp is not None:
                fig_pp.add_trace(
                    go.Scatter(
                        x=x_pp,
                        y=trend_line_pp,
                        name="Trend",
                        mode="lines",
                        line=dict(color="white", width=1.5, dash="dash"),
                        hoverinfo="skip",
                    )
                )

            fig_pp.add_trace(
                go.Scatter(
                    x=df_pp["time_min"],
                    y=df_pp["watts_smooth"],
                    name="Moc (tło)",
                    yaxis="y2",
                    line=dict(width=0),
                    fill="tozeroy",
                    fillcolor="rgba(255,255,255,0.05)",
                    hoverinfo="skip",
                )
            )

            fig_pp.update_layout(
                template="plotly_dark",
                title="Pulse Power: Koszt Energetyczny Serca",
                hovermode="x unified",
                xaxis=dict(title="Czas [min]", tickformat=".0f", hoverformat=".0f"),
                yaxis=dict(title="Pulse Power [W / bpm]"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, visible=False),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=1.05, x=0),
                height=450,
            )

            st.plotly_chart(fig_pp, width="stretch")

            with st.expander("📖 Jak czytać Pulse Power? — Teoria i Fizjologia", expanded=False):
                st.markdown("""
### 🔬 Czym jest Pulse Power?

**Pulse Power (PP)** to stosunek mocy mechanicznej do tętna — **W/bpm**. Mierzy, ile watów generujesz 
na każde uderzenie serca. To wskaźnik **efektywności krążeniowo-oddechowej**, który łączy w sobie:

- **Objętość wyrzutową serca (SV)** — ile krwi serce tłoczy na uderzenie
- **Ekstrakcję tlenu (a-vO₂ diff)** — ile tlenu mięśnie pobierają z krwi
- **Efektywność mechaniczną (GE)** — ile energii metabolicznej zamieniasz na waty

Z fizjologii: **Moc = HR × SV × a-vO₂ diff × GE**, więc PP = **SV × a-vO₂ diff × GE**. 
Im wyższe PP, tym bardziej wydajny Twój „silnik" tlenowy.

---

### 📊 Jak interpretować wykres?

| Trend PP | Interpretacja | Co się dzieje |
|---|---|---|
| **Stabilny (±3%)** | ✅ Optymalny stan | Serce i mięśnie pracują stabilnie. Dobry stan nawodnienia i termoregulacji. |
| **Spadek 3–5%** | ⚠️ Fizjologiczna norma | Typowy **cardiovascular drift** — wzrost HR z powodu wzrostu temperatury ciała i utraty osocza (Coates & Burr, 2018). |
| **Spadek 5–10%** | 🔴 Wczesne zmęczenie | Odwodnienie ≥2% masy ciała, wyczerpanie glikogenu lub kumulacja ciepła. Czas na nawodnienie i węglowodany. |
| **Spadek >10%** | 🛑 Znaczące zmęczenie | Poważny dryf kardiowaskularny. Ryzyko „zawału mocy" — drastyczny spadek wydolności. |

---

### 🧠 Fizjologia: Dlaczego PP spada w czasie?

**1. Cardiovascular Drift (CV Drift)**
Podczas długotrwałego wysiłku tętno rośnie mimo stałej mocy. To nie jest „poprawa wydolności" — 
to **kompensacja spadku objętości wyrzutowej serca**. Gdy tracisz potem osocze (nawet 10–15% objętości krwi), 
mniej krwi wraca do serca (↓ preload), więc serce musi bić szybciej, żeby utrzymać rzut serka 
(Walters & Cox, 2022; Barsumyan et al., 2025).

**2. Temperatura ciała (Core Temp)**
Wzrost temperatury rdzeniowej powyżej 38.5°C aktywuje termoregulację — krew kierowana jest do skóry 
w celu chłodzenia, co „krzywdzi" mięśnie. To tzw. **cardiovascular strain** — serce bije szybciej, 
ale dostarcza mniej O₂ do pracujących mięśni (Frontiers in Physiology, 2025).

**3. Deplecja glikogenu**
Gdy zapasy glikogenu w mięśniach i wątrobie maleją, organizm musi korzystać z mniej wydajnych 
źródeł energii (tłuszcze), co wymaga więcej O₂ na jednostkę mocy (↓ GE). 
Dodatkowo spadek glikogenu upośledza zdolność buforowania kwasu mlekowego, 
co przyspiesza rekrutację włókien typu II (mniej wydajnych tlenowo).

**4. Rekrutacja włókien mięśniowych typu II**
Gdy włókna typu I (wolnokurczliwe, wysoce wydajne tlenowo) się męczą, 
organizm rekrutuje włókna typu II (szybkokurczliwe), które zużywają więcej energii 
na jednostkę mocy → **GE spada** → **PP spada** (Dunst et al., 2023).

---

### 📈 Pulse Power jako wskaźnik zmęczenia

Badanie Nuuttila et al. (2024, *European Journal of Sport Science*) pokazało, że 
**HR-power index** (odpowiednik PP) jest jednym z najwcześniejszych wskaźników overreachingu — 
różnice między grupami pojawiały się już po **6 dniach** overloadu, 
wcześniej niż subiektywne odczucia sportowców (gotowość do treningu, ból nóg).

Barsumyan et al. (2025, *Frontiers in AI*) zastosowali machine learning do detekcji 
cardiovascular drift w oparciu o dane z power metera i HR — model trafnie 
klasyfikował zmęczenie na podstawie trendów podobnych do PP.

---

### 💡 Praktyczne wskazówki

- **Porównuj PP między treningami**, nie tylko wewnątrz jednej jazdy. 
  Trend PP w podobnych treningach z tygodnia na tydzień = najlepszy wskaźnik adaptacji.
- **PP w Z2/Z3 jest najbardziej miarodajne.** Przy mocy >FTP HR lag zniekształca wynik.
- **Spadek PP >5% + wysoki HR spoczynkowy** = silny sygnał niewystarczającej regeneracji.
- **Wzrost PP w kolejnych tygodniach** = pozytywna adaptacja: wzrost SV i/lub GE.

---

**Bibliografia:**
- Nuuttila et al. (2024). Monitoring fatigue state with HR-based and subjective methods. *Eur J Sport Sci*, 24(7), 857–869.
- Barsumyan et al. (2025). Quantifying training response in cycling based on cardiovascular drift using ML. *Frontiers in AI*, 8, 1623384.
- Dunst et al. (2023). Time- and Fatigue-Dependent Efficiency during Maximal Cycling Sprints. *Sports*, 11, 29.
- Walters & Cox (2022). Prolonged cycling reduces power output at the moderate-to-heavy intensity transition. *Eur J Appl Physiol*, 122, 2673–2682.
- Coates, Millar & Burr (2018). Blunted Cardiac Output from Overtraining Is Related to Increased Arterial Stiffness. *Med Sci Sports Exerc*, 50(12), 2459–2464.
- Gejl et al. (2024). Substrate utilization and durability during prolonged intermittent exercise in elite road cyclists. *Eur J Appl Physiol*, 124, 1801–1815.
""")
        else:
            st.warning(
                "Zbyt mało danych (jazda poniżej 50W lub HR poniżej 90bpm), aby obliczyć wiarygodne Pulse Power."
            )
    else:
        st.error("Brak danych mocy lub tętna.")

    # --- GROSS EFFICIENCY ---
    st.divider()
    st.subheader("⚙️ Gross Efficiency (GE%) - Estymacja")
    st.caption("Stosunek mocy generowanej (Waty) do spalanej energii (Metabolizm). Typowo: 18-23%.")

    # 1. Sprawdzamy, czy mamy potrzebne dane
    if (
        "watts_smooth" in df_plot_resampled.columns
        and "heartrate_smooth" in df_plot_resampled.columns
    ):
        # 2. Obliczamy Moc Metaboliczną (Wzór Keytela na podstawie HR)
        # Wzór zwraca kJ/min. Zamieniamy to na Waty (J/s).
        # P_met [W] = (kJ/min * 1000) / 60

        # Współczynniki Keytela
        gender_factor = -55.0969 if is_male else -20.4022

        # Obliczenie wydatku energetycznego (EE) w kJ/min dla każdej sekundy
        # Używamy wygładzonego HR, żeby uniknąć skoków
        ee_kj_min = (
            gender_factor
            + (0.6309 * df_plot_resampled["heartrate_smooth"])
            + (0.1988 * rider_weight)
            + (0.2017 * rider_age)
        )

        # Konwersja na Waty Metaboliczne (P_met)
        # Uwaga: EE nie może być ujemne ani zerowe (serce bije)
        p_metabolic = (ee_kj_min * 1000) / 60
        p_metabolic = p_metabolic.replace(0, np.nan)  # Unikamy dzielenia przez zero

        # 3. Obliczamy Gross Efficiency (GE)
        # GE = (Moc Mechaniczna / Moc Metaboliczna) * 100
        # Filtrujemy momenty, gdzie nie pedałujesz (Moc < 10W), bo wtedy GE=0

        ge_series = (df_plot_resampled["watts_smooth"] / p_metabolic) * 100

        # Filtrujemy dane nierealistyczne i "zimny start"
        # 1. Watts > 40 (żeby nie dzielić przez zero na postojach)
        # 2. GE między 5% a 30% (wszystko powyżej 30% to błąd pomiaru lub HR Lag)
        # 3. HR > 100 bpm (Wzór Keytela bardzo słabo działa dla niskiego tętna!)

        mask_ge = (
            (df_plot_resampled["watts_smooth"] > 100)
            & (ge_series > 5)
            & (ge_series < 30)
            & (df_plot_resampled["heartrate_smooth"] > 110)
        )

        # Zerujemy błędne wartości (zamieniamy na NaN, żeby nie rysowały się na wykresie)
        df_ge = pd.DataFrame(
            {
                "time_min": df_plot_resampled["time_min"],
                "ge": ge_series,
                "watts": df_plot_resampled["watts_smooth"],
            }
        )
        df_ge.loc[~mask_ge, "ge"] = np.nan

        # 4. Czyszczenie danych (Realistyczne ramy fizjologiczne)
        # GE rzadko przekracza 30% (chyba że zjeżdżasz z góry i HR spada szybciej niż waty)
        # GE poniżej 0% to błąd.
        mask_ge = (df_plot_resampled["watts_smooth"] > 40) & (ge_series > 5) & (ge_series < 35)

        df_ge = pd.DataFrame(
            {
                "time_min": df_plot_resampled["time_min"],
                "ge": ge_series,
                "watts": df_plot_resampled["watts_smooth"],
            }
        )
        # Zerujemy nierealistyczne wartości do wykresu
        df_ge.loc[~mask_ge, "ge"] = np.nan

        if not df_ge["ge"].isna().all():
            avg_ge = df_ge["ge"].mean()

            # KOLUMNY Z WYNIKAMI
            cg1, cg2, cg3 = st.columns(3)
            cg1.metric("Średnie GE", f"{avg_ge:.1f}%", help="Pro: 23%+, Amator: 18-21%")

            # Trend GE (czy spada w czasie?)
            valid_ge = df_ge.dropna(subset=["ge"])
            if len(valid_ge) > 100:
                slope_ge, _, _, _, _ = stats.linregress(valid_ge["time_min"], valid_ge["ge"])
                total_drift_ge = slope_ge * (
                    valid_ge["time_min"].iloc[-1] - valid_ge["time_min"].iloc[0]
                )
                cg2.metric(
                    "Zmiana GE (Trend)",
                    f"{total_drift_ge:.1f}%",
                    delta_color="inverse" if total_drift_ge < 0 else "normal",
                )
            else:
                cg2.metric("Zmiana GE", "-")

            cg3.info(
                "Wartości powyżej 25% mogą wynikać z opóźnienia tętna względem mocy (np. krótkie interwały). Analizuj trendy na długich odcinkach."
            )

            # WYKRES GE
            fig_ge = go.Figure()

            # Linia GE
            fig_ge.add_trace(
                go.Scatter(
                    x=df_ge["time_min"],
                    y=df_ge["ge"],
                    customdata=df_ge["watts"],
                    mode="lines",
                    name="Gross Efficiency (%)",
                    line=dict(color="#00cc96", width=1.5),
                    connectgaps=False,  # Nie łączymy przerw (postojów)
                    hovertemplate="GE: %{y:.1f}%<br>Moc: %{customdata:.0f} W<extra></extra>",
                )
            )

            # Tło (Moc)
            fig_ge.add_trace(
                go.Scatter(
                    x=df_ge["time_min"],
                    y=df_ge["watts"],
                    mode="lines",
                    name="Moc (Tło)",
                    yaxis="y2",
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(255,255,255,0.05)",
                    hoverinfo="skip",
                )
            )

            # Linia Trendu GE
            if len(valid_ge) > 100:
                trend_line = np.poly1d(np.polyfit(valid_ge["time_min"], valid_ge["ge"], 1))(
                    valid_ge["time_min"]
                )
                fig_ge.add_trace(
                    go.Scatter(
                        x=valid_ge["time_min"],
                        y=trend_line,
                        mode="lines",
                        name="Trend GE",
                        line=dict(color="white", width=2, dash="dash"),
                    )
                )

            fig_ge.update_layout(
                template="plotly_dark",
                title="Efektywność Brutto (GE%) w Czasie",
                hovermode="x unified",
                xaxis=dict(title="Czas [min]", tickformat=".0f", hoverformat=".0f"),
                yaxis=dict(title="GE [%]", range=[10, 30]),
                yaxis2=dict(title="Moc [W]", overlaying="y", side="right", showgrid=False),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=1.1, x=0),
            )

            st.plotly_chart(fig_ge, width="stretch")

            with st.expander(
                "📖 Jak interpretować Gross Efficiency? — Teoria i Fizjologia", expanded=False
            ):
                st.markdown("""
### 🔬 Czym jest Gross Efficiency?

**Gross Efficiency (GE)** to stosunek mocy mechanicznej (watów na pedałach) do całkowitej mocy 
metabolicznej (energii z utleniania) — wyrażony w procentach. Mierzy, **ile procent energii z 
jedzenia i tlenu zamieniasz na użyteczną pracę**, a ile tracisz na ciepło.

Wzór: **GE (%) = (Moc Mechaniczna / Moc Metaboliczna) × 100**

Moc metaboliczną estymujemy z tętna za pomocą wzoru Keytela (Keytel et al., 1972), 
który jest standardem w monitorowaniu wydatku energetycznego w warunkach polowych.

> ⚠️ **Uwaga:** Estymacja GE z HR ma ograniczenia. Wzór Keytela najlepiej działa 
> przy HR > 110 bpm i mocy > 100 W. Poniżej tych wartości wyniki są zawyżone.

---

### 📊 Tabela interpretacji GE

| Zakres GE | Poziom | Co oznacza |
|---|---|---|
| **< 17%** | 🔴 Niska wydajność | Typowe dla początkujących. Dużo energii traconej na koordynację, napięcie izometryczne i produkcję ciepła. |
| **17–19%** | 🟡 Poniżej średniej | Osoby rekreacyjnie aktywne. Rozwijalna technika pedałowania może podnieść GE o 2-3%. |
| **19–21%** | ✅ Standard amatorski | Dobrze wytrenowany kolarz klubowy. Solidna baza tlenowa. |
| **21–23%** | 🟢 Dobry poziom | Zaawansowany amator / zawodnik regionalny. Wysoki udział włókien typu I. |
| **23–25%** | 🔵 ELITE / PRO | Kolarze zawodowi. Wedle badań Joyner & Coyle (2008), topowi endurance athletes osiągają GE ~23-25%. |
| **> 25%** | ⚠️ Podejrzane | Prawdopodobnie błąd pomiaru: HR lag (tętno nie nadąża za mocą), jazda w dół, lub miernik mocy zawyża. Teoretyczny limit GE to ~27% (Mogensen et al., 2006). |

---

### 🧠 Czynniki determinujące GE

**1. Typ włókien mięśniowych**
GE jest silnie skorelowane z procentowym udziałem włókien typu I (wolnokurczliwych). 
Coyle et al. (1992) wykazali, że kolarze z >55% włókien I mieli GE ~23%, 
podczas gdy osoby z <45% — zaledwie ~19%. Włókna typu I zużywają mniej ATP 
na jednostkę siły (niższy ATPCOST) i mają wyższą efektywność mitochondrialną 
(Mogensen et al., 2006; Zoladz et al., 2025).

**2. Kadens (Częstotliwość pedałowania)**
Kamba et al. (2023) wykazali, że GE zmienia się z kadensem — **optymalny kadens dla GE 
wynosi zazwyczaj 70–90 rpm** przy umiarkowanej intensywności. Zbyt wysoki kadens (>100 rpm) 
zwiększa koszt energetyczny rekrutacji mięśni, Przekrocznie niski (<60 rpm) wymaga większej siły 
na pedał, co rekrutuje więcej włókien II (mniej wydajnych).

**3. Trening i adaptacja**
Wieloletni trening endurance zwiększa GE przez:
- Hipertrofię włókien typu I i zwiększenie ich udziału
- Wzrost gęstości mitochondrialnej (lepsze wykorzystanie O₂)
- Poprawę koordynacji nerwowo-mięśniowej (mniej „przecieków" motorycznych)
- Zmniejszenie niepotrzebnego napięcia izometrycznego (np. kołysanie biodrami)
Zoladz et al. (2012) wykazali, że trening endurance zmniejsza nieliniowość relacji V̇O₂–moc.

**4. Masa mięśniowa**
Zoladz et al. (2025, *Scientific Reports*) wykazali paradoksalnie, że **większa masa mięśni nóg 
obniża GE** podczas jazdy o umiarkowanej intensywności. Powód: większa masa = wyższy 
„internal work" (koszt utrzymania i ruchu samej masy mięśniowej). To częściowo tłumaczy, 
dlaczego lekko zbudowani kolarze mogą mieć lepsze GE niż silnie umięśnieni.

---

### 📉 Dlaczego GE spada w czasie?

Spadek GE w trakcie długotrwałego wysiłku to jeden z najdokładniejszych wskaźników **zmęczenia 
metabolicznego**. Mechanizmy:

**1. Rekrutacja włókien typu II**
Gdy włókna I są wyczerpane (deplekcja glikogenu), organizm rekrutuje włókna II 
(szybkokurczliwe), które zużywają ~2x więcej ATP na jednostkę mocy → GE spada 
(Zoladz et al., 2008; Cannon et al., 2014).

**2. Deplekcja glikogenu**
Gejl et al. (2024, *Eur J Appl Physiol*) wykazali w badaniu na elitarnych kolarzach, 
że mimo wysokiego spożycia węglowodanów (~90g/h), kumulacyjna oksydacja CHO wyniosła ~630g 
podczas 4h jazdy — z czego ~270g pochodziło z glikogenu mięśniowego i wątrobowego. 
Gdy glikogen maleje, organizm przełącza się na tłuszcze (niższa wydajność tlenowa/ATP) → GE spada.

**3. Termoregulacja**
Wzrost temperatury rdzeniowej >38.5°C wymaga przesunięcia krwi do skóry w celu chłodzenia. 
To „okrada" mięśnie z O₂ i zwiększa koszt sercowo-naczyniowy, który nie przekłada się na moc 
mechaniczną → GE spada (tzw. cardiovascular strain).

**4. Zmęczenie nerwowo-mięśniowe**
Prolongowany wysiłek upośledza wzorzec rekrutacji jednostek motorycznych — mniej 
ekonomiczna sekwencja aktywacji mięśni, więcej „przecieków" siły, gorsza koordynacja 
mięśni agonistów/antagonistów (Fares et al., 2025).

---

### 📈 GE jako wskaźnik adaptacji do treningu

- **Wzrost GE w kolejnych tygodniach** = jedna z najważniejszych adaptacji do treningu endurance. 
  Zwiększenie GE z 20% na 22% oznacza, że przy tej samej mocy metabolicznej generujesz **10% więcej watów**.
- **Porównuj GE w podobnych treningach** (ta sama intensywność, podobny kadens). 
  GE jest czułe na kadens — porównywanie GE przy 70 rpm z GE przy 100 rpm jest mylące.
- **GE w Z2/Z3 jest najbardziej miarodajne.** Przy mocy >FTP, anaerobic contribution zniekształca wynik.
- **Najlepszy trend:** GE stabilne lub rosnące w Z2 przy rosnącej mocy = adaptacja.

---

**Bibliografia:**
- Zoladz et al. (2025). Higher legs muscle mass reduces gross mechanical efficiency. *Scientific Reports*, 15, 16469.
- Zoladz et al. (2012). Endurance training decreases the non-linearity in the VO₂-power output relationship. *Exp Physiol*, 97(3), 386–399.
- Gejl et al. (2024). Substrate utilization and durability during prolonged intermittent exercise in elite road cyclists. *Eur J Appl Physiol*, 124, 1801–1815.
- Kamba et al. (2023). Effect of Gear Ratio and Cadence on Gross Efficiency. *Sports*, 11, 5.
- Carlsson et al. (2020). Gross and delta efficiencies during uphill running and cycling among elite triathletes. *Eur J Appl Physiol*, 120, 1749–1761.
- Coyle et al. (1992). Cycling efficiency is related to the percentage of type I muscle fibers. *Med Sci Sports Exerc*, 24(7), 782–788.
- Joyner & Coyle (2008). Endurance exercise performance: the physiology of champions. *J Physiol*, 586(1), 35–44.
- Mogensen et al. (2006). Cycling efficiency is related to low UCP3 content and to type I fibres. *J Physiol*, 571(3), 669–681.
- Fares et al. (2025). Cycling exercise efficiency and economy: Exploring the role of phase angle. *J Elec Bioimpedance*, 16(1), 129–134.
""")
        else:
            st.warning(
                "Brak wystarczających danych do obliczenia GE (zbyt krótkie odcinki stabilnej jazdy)."
            )
    else:
        st.error("Do obliczenia GE potrzebujesz danych Mocy (Watts) oraz Tętna (HR).")
