"""
TTE (Time-to-Exhaustion) UI Module.

Displays TTE analysis for the current session and historical trends.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from modules.plots import CHART_CONFIG
from modules.tte import (
    compute_tte_result,
    format_tte,
    export_tte_json,
    TTEResult,
)


def render_tte_tab(df_plot: pd.DataFrame, ftp: float, uploaded_file_name: str = "manual_upload") -> None:
    """Render the TTE analysis tab.
    
    Args:
        df_plot: Session data with 'watts' column
        ftp: Functional Threshold Power
        uploaded_file_name: Original filename for record matching
    """
    st.header("⏱️ Time-to-Exhaustion (TTE)")
    st.markdown("""
    Analiza maksymalnego czasu, przez który utrzymałeś zadaną moc.
    TTE mierzy Twoją zdolność do utrzymania intensywności na poziomie progu (FTP).
    """)
    
    # Check for power data
    if 'watts' not in df_plot.columns:
        st.error("Brak danych mocy (watts) w pliku.")
        return
    
    # Configuration
    st.subheader("⚙️ Konfiguracja")
    col1, col2 = st.columns(2)
    
    with col1:
        target_pct = st.slider(
            "Docelowy % FTP",
            min_value=70,
            max_value=120,
            value=100,
            step=5,
            help="Procent FTP, który chcesz analizować"
        )
    
    with col2:
        tol_pct = st.slider(
            "Tolerancja [%]",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Dopuszczalne odchylenie od docelowej mocy"
        )
    
    # Compute TTE for current session
    power_series = df_plot['watts']
    result = compute_tte_result(
        power_series,
        target_pct=target_pct,
        ftp=ftp,
        tol_pct=tol_pct
    )
    
    # Display results
    st.subheader("📊 Wyniki Sesji")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Maksymalny TTE",
            format_tte(result.tte_seconds),
            help="Najdłuższy ciągły czas utrzymania mocy w zadanym zakresie"
        )
    
    with col2:
        st.metric(
            "Zakres Mocy",
            f"{result.target_power_min:.0f} - {result.target_power_max:.0f} W",
            help=f"{target_pct}% FTP ± {tol_pct}%"
        )
    
    with col3:
        # Interpretation
        if result.tte_seconds >= 3600:
            status = "🏆 Elitarny"
            color = "🟢"
        elif result.tte_seconds >= 1800:
            status = "💪 Dobry"
            color = "🟢"
        elif result.tte_seconds >= 600:
            status = "📈 Rozwijający się"
            color = "🟡"
        else:
            status = "🔄 Do poprawy"
            color = "🟠"
        
        st.metric("Ocena", f"{color} {status}")
    
    # Power distribution chart
    _render_power_distribution_chart(df_plot, result)
    
    # Export section
    st.divider()
    with st.expander("📥 Eksport JSON"):
        json_data = export_tte_json(result)
        st.code(json_data, language="json")
        st.download_button(
            "Pobierz JSON",
            data=json_data,
            file_name=f"tte_{result.session_id}.json",
            mime="application/json"
        )
    
    # Theory section
    with st.expander("📖 Time-to-Exhaustion (TTE) — Teoria i Fizjologia", expanded=False):
        st.markdown("""
### Definicja: Time-to-Exhaustion

**TTE** (Time-to-Exhaustion) to maksymalny czas, przez który sportowiec może utrzymać zadaną intensywność wysiłku przed wyczerpaniem. W kontekście kolarstwa, TTE przy 100% FTP mierzy **wytrzymałość progową** — jak długo potrafisz pracować na granicy metabolizmu tlenowego i beztlenowego (Wilber et al., 2022).

---

### Tabela Interpretacji TTE przy 100% FTP

| Poziom | TTE | Interpretacja |
|---|---|---|
| **Początkujący** | 20-40 min | Słaba wytrzymałość progowa. Dominacja metabolizmu beztlenowego, szybka akumulacja metabolitów |
| **Amator** | 40-60 min | Typowy zakres dla rekreatyjnych kolarzy. Baza aerobowa w budowie |
| **Zaawansowany** | 60-75 min | Dobra wytrzymałość progowa. Efektywny klirens mleczanu, wysoka ekonomia |
| **Elitarny** | 75+ min | Wysoka tolerancja na intensywność progową. Typowe dla zawodników krajowych/elitarnych |
| **Światowy** | 90+ min | Ekstremalna wytrzymałość. TTE >90 min przy 100% FTP to poziom WorldTour |

---

### 4 Mechanizmy Fizjologiczne Ograniczające TTE

**1. Wyczerpanie W' (Work Capacity above CP)**
TTE jest bezpośrednio powiązany z modelem CP/W'. Każdy wysiłek powyżej CP zużywa W' — skończoną pojemność pracy beztlenowej. Gdy W' się wyczerpie, organizm nie jest w stanie utrzymać zadanej mocy. Klimstra (2024) pokazał że W' jest kluczowym predyktorem TTE przy intensywnościach >CP.

**2. Akumulacja Metabolitów i Zakwaszenie**
Powyżej progu, produkcja H⁺ przekracza zdolność buforową mięśnia. Spadek pH hamuje enzymy glikolityczne (PFK) i interferuje z wiązaniem Ca²⁺ w sarkomerze → spadek siły skurczu. Lipková et al. (2022) wykazali że TTE koreluje z progiem mleczanowym (LT2) — im wyżej LT2 względem FTP, tym dłuższe TTE.

**3. Zaburzenie Homeostazy Tlenowej**
Goulding & Marwood (2023) pokazali że powyżej CP organizm nie utrzymuje homeostazy metabolicznej — VO2 rośnie do VO2max (VO2 slow component), a SmO2 spada do minimum. Punkt, w którym VO2 osiąga VO2max, wyznacza fizjologiczny limit TTE.

**4. Zmęczenie Centralne (CNS Fatigue)**
Przedłużony wysiłek na wysokim procencie FTP powoduje spadek rekrutacji jednostek motorycznych z powodu zmian w neurotransmisji (serotonina, dopamina, amoniak). Mózg "odcina" mięśnie aby chronić organizm przed uszkodzeniem.

---

### TTE a Critical Power vs FTP

**FTP** (Functional Threshold Power) to estymacja mocy przy LT2 — zwykle ~95-100% CP. **CP** (Critical Power) to fizjologiczny próg homeostazy. TTE jest bardziej precyzyjny gdy odniesiony do CP niż FTP, ponieważ:
- CP ma silniejszą podstawę fizjologiczną (Goulding & Marwood, 2023)
- FTP jest estymacją subiektywną (test 20-minutowy × 0.95)
- TTE @ 100% CP ≈ 20-30 min, podczas gdy TTE @ 100% FTP ≈ 40-75 min

---

### Wpływ Treningu na TTE

Badania pokazują że TTE można poprawić o 15-30% w ciągu 8-12 tygodni:

| Typ Treningu | Mechanizm | Efekt na TTE |
|---|---|---|
| **Interwały VO2max** (4-8 × 3-5min @ 110-120% FTP) | Zwiększenie VO2max, poprawa kinetyki VO2 | +10-20% |
| **Tempo/SST** (2 × 20min @ 88-94% FTP) | Podniesienie progu mleczanowego, poprawa ekonomii | +15-25% |
| **Jazda Z2** (3-5h @ 60-75% FTP) | Kapilaryzacja, mitochondria, oksydacja tłuszczów | +5-15% |
| **Interwały powyżej CP** (5 × 3min @ 120% FTP) | Zwiększenie W', poprawa tolerancji na metabolity | +10-20% |

---

### TTE w Praktyce Wyścigowej

- **Jazda indywidualna na czas:** TTE @ 105-110% FTP determinuje czy utrzymasz tempo do mety
- **Podjazdy:** TTE @ 110-130% FTP decyduje czy "odjedziesz" czy "odpadniesz"
- **Ataki:** Po każdym ataku powyżej CP, TTE mówi ile masz "paliwa" w baku W'
- **Wyścigi kryterialne:** Wielokrotne wyczerpania i częściowe regeneracje W' — kluczowa jest szybkość rekonstytucji (Chorley et al., 2022)

---

### Bibliografia

- Wilber et al. (2022). Time to exhaustion at estimated functional threshold power in road cyclists of different performance levels. *Journal of Science and Medicine in Sport*, 25(9), 740-745.
- Goulding & Marwood (2023). Interaction of factors determining critical power. *Sports Medicine*, 53, 595–613.
- Klimstra (2024). Estimate anaerobic work capacity and critical power with constant-power all-out test. *J. Funct. Morphol. Kinesiol.*, 9(4), 202.
- Lipková et al. (2022). Determination of critical power using different possible approaches among endurance athletes: A review. *Int. J. Environ. Res. Public Health*, 19, 7589.
- Chorley et al. (2022). Bi-exponential modelling of W′ reconstitution kinetics in trained cyclists. *European Journal of Applied Physiology*, 122, 677–689.
        """)


def _render_power_distribution_chart(df_plot: pd.DataFrame, result: TTEResult) -> None:
    """Render power distribution chart with TTE range highlighted."""
    fig = go.Figure()
    
    # Power trace
    fig.add_trace(go.Scatter(
        x=df_plot['time_min'],
        y=df_plot['watts'],
        name='Moc',
        line=dict(color='#1f77b4', width=1),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate='Moc: %{y:.0f} W<extra></extra>'
    ))
    
    # Target range shading
    fig.add_hrect(
        y0=result.target_power_min,
        y1=result.target_power_max,
        fillcolor="rgba(0, 255, 0, 0.1)",
        line=dict(color="green", width=1, dash="dash"),
        annotation_text=f"Zakres TTE ({result.target_pct}% ± {result.tolerance_pct}%)",
        annotation_position="top left"
    )
    
    # Layout
    fig.update_layout(
        template="plotly_dark",
        title="Rozkład Mocy z Zakresem TTE",
        hovermode="x unified",
        xaxis=dict(
            title="Czas [min]",
            tickformat=".0f",
            hoverformat=".0f"
        ),
        yaxis=dict(
            title="Moc [W]",
            tickformat=".0f"
        ),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1, x=0)
    )
    
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
