"""
Drift Maps UI Module.

Displays Power-HR-SmO2 scatter plots and drift analysis at constant power.
"""
import streamlit as st
import pandas as pd
import json

from modules.plots import CHART_CONFIG
from modules.physio_maps import (
    scatter_power_hr,
    scatter_power_smo2,
    detect_constant_power_segments,
    trend_at_constant_power,
    calculate_drift_metrics,
)


def _format_min_to_mmss(decimal_min: float) -> str:
    """Helper to convert decimal minutes to mm:ss string."""
    total_seconds = int(decimal_min * 60)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def render_drift_maps_tab(df_plot: pd.DataFrame) -> None:
    """Render the Drift Maps tab in Performance section.
    
    Args:
        df_plot: Session DataFrame with power, hr, and optionally smo2 data
    """
    st.header("📊 Drift Maps: Power-HR-SmO₂")
    
    st.markdown("""
    Analiza relacji między mocą, tętnem i saturacją mięśniową (SmO₂).
    **Drift HR** wskazuje na zmęczenie sercowo-naczyniowe, 
    **spadek SmO₂** sugeruje narastający deficyt tlenowy.
    """)
    
    # Check data availability
    has_hr = any(col in df_plot.columns for col in ['heartrate', 'hr', 'heart_rate', 'HeartRate'])
    has_smo2 = any(col in df_plot.columns for col in ['smo2', 'SmO2', 'muscle_oxygen'])
    
    if not has_hr:
        st.warning("Brak danych HR - nie można wygenerować wykresów.")
        return
    
    # ===== SCATTER PLOTS =====
    st.subheader("🔵 Scatter Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_power_hr = scatter_power_hr(df_plot, title="Power vs HR")
        if fig_power_hr:
            st.plotly_chart(fig_power_hr, width="stretch", config=CHART_CONFIG)
        else:
            st.info("Za mało danych do wygenerowania wykresu Power vs HR.")
    
    with col2:
        if has_smo2:
            fig_power_smo2 = scatter_power_smo2(df_plot, title="Power vs SmO₂")
            if fig_power_smo2:
                st.plotly_chart(fig_power_smo2, width="stretch", config=CHART_CONFIG)
            else:
                st.info("Za mało danych SmO₂ do wygenerowania wykresu.")
        else:
            st.info("📉 Brak danych SmO₂ - wykres niedostępny.")
    
    st.divider()
    
    # ===== CONSTANT POWER SEGMENT ANALYSIS =====
    st.subheader("📏 Analiza Dryfu przy Stałej Mocy")
    
    # Detect segments
    segments = detect_constant_power_segments(df_plot, tolerance_pct=10, min_duration_sec=120)
    
    if not segments:
        st.info("Nie wykryto segmentów stałej mocy (min. 2 minuty, ±10%).")
        
        # Manual power input fallback
        st.markdown("**Ręczny wybór mocy:**")
        col_manual1, col_manual2 = st.columns(2)
        with col_manual1:
            power_target = st.number_input(
                "Docelowa moc [W]",
                min_value=50,
                max_value=500,
                value=int(df_plot['watts'].median()) if 'watts' in df_plot.columns else 200,
                step=10,
                key="drift_power_target"
            )
        with col_manual2:
            tolerance = st.slider(
                "Tolerancja [%]",
                min_value=5,
                max_value=20,
                value=10,
                key="drift_tolerance"
            )
        
        fig_drift, drift_metrics = trend_at_constant_power(
            df_plot, power_target, tolerance_pct=tolerance
        )
        
        if fig_drift:
            st.plotly_chart(fig_drift, width="stretch", config=CHART_CONFIG)
            _display_drift_metrics(drift_metrics)
        else:
            st.warning(f"Brak danych w zakresie {power_target}W ±{tolerance}%.")
    else:
        # Segment selector
        segment_options = [
            f"{i+1}. {seg[2]:.0f}W ({_format_min_to_mmss((seg[1]-seg[0])/60)})"
            for i, seg in enumerate(segments)
        ]
        
        selected_idx = st.selectbox(
            "Wybierz segment stałej mocy:",
            range(len(segments)),
            format_func=lambda x: segment_options[x],
            key="segment_selector"
        )
        
        selected_segment = segments[selected_idx]
        power_target = selected_segment[2]
        
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            tolerance = st.slider(
                "Tolerancja [%]",
                min_value=5,
                max_value=20,
                value=10,
                key="drift_tolerance_seg"
            )
        
        fig_drift, drift_metrics = trend_at_constant_power(
            df_plot, power_target, tolerance_pct=tolerance
        )
        
        if fig_drift:
            st.plotly_chart(fig_drift, width="stretch", config=CHART_CONFIG)
            _display_drift_metrics(drift_metrics)
        else:
            st.warning("Nie można obliczyć dryfu dla wybranego segmentu.")
    
    st.divider()
    
    # ===== OVERALL METRICS JSON (Hidden in Expander) =====
    with st.expander("📋 Metryki Sesji (JSON)"):
        overall_metrics = calculate_drift_metrics(df_plot)
        
        col_json1, col_json2 = st.columns([2, 1])
        
        with col_json1:
            st.json(overall_metrics)
        
        with col_json2:
            st.download_button(
                "📥 Pobierz JSON",
                data=json.dumps(overall_metrics, indent=2),
                file_name="drift_metrics.json",
                mime="application/json",
                key="download_drift_json"
            )
    
    # Interpretation
    with st.expander("📖 Drift Maps — Teoria i Fizjologia", expanded=False):
        st.markdown("""
### Definicja: Cardiovascular Drift

**Cardiovascular drift (CV Drift)** to zjawisko stopniowego wzrostu tętna przy stałej mocy wysiłku, spowodowane spadkiem objętości wyrzutowej serca (stroke volume, SV). Po 5-10 minutach ciągłego wysiłku, HR zaczyna "dryfować" w górę mimo braku wzrostu obciążenia (Souissi et al., 2021).

**Mechanizm:** SV spada z powodu (1) utraty objętości osocza przez potenie, (2) redystrybucji krwi do skóry w termoregulacji, (3) wzrostu aktywności współczulnej. Aby utrzymać rzut serca (CO = HR × SV), serce przyspiesza (Ganio et al., 2021).

---

### Tabela Interpretacji HR Drift

| HR Drift Slope | Zakres | Interpretacja |
|---|---|---|
| **Minimalny** | < 0.2 bpm/min | Doskonała stabilność sercowo-naczyniowa. Typowe dla dobrze wytrenowanych kolarzy w chłodnych warunkach |
| **Umiarkowany** | 0.2 - 0.5 bpm/min | Normalne zmęczenie fizjologiczne. Oczekiwany przy wysiłku >30 min w strefie Z2-Z3 |
| **Znaczący** | 0.5 - 1.0 bpm/min | Pogarszająca się wydolność. Możliwe: odwodnienie, przegrzanie, zbyt wysoka intensywność |
| **Krytyczny** | > 1.0 bpm/min | Silny stres sercowo-naczyniowy. Ryzyko przedwczesnego zmęczenia. Rozważ nawodnienie i schłodzenie |

---

### Tabela Interpretacji SmO₂ Drift

| SmO₂ Slope | Zakres | Interpretacja |
|---|---|---|
| **Stabilny** | > -0.1 %/min | Praca w strefie tlenowej — dostawa O₂ nadąża za zużyciem |
| **Umiarkowany spadek** | -0.1 do -0.3 %/min | Praca na granicy wydolności — typowe dla strefy Z3-Z4 |
| **Postępujący deficyt** | < -0.3 %/min | Przekroczenie progu — narastający deficyt tlenowy, typowe dla Z5+ |

---

### 4 Mechanizmy Cardiovascular Drift

**1. Utrata Objętości Osocza (Plasma Volume Loss)**
Podczas wysiłku, potenie powoduje utratę płynów → spadek objętości osocza o 5-15% w ciągu godziny → zmniejszenie SV → kompensacyjny wzrost HR. Barsumyan et al. (2025) pokazali że ML może predykować stan zmęczenia na podstawie wzorca CV drift z danych power meter + HR.

**2. Redystrybucja Krwi do Skóry (Skin Blood Flow)**
Wzrost temperatury ciała → rozszerzenie naczyń skórnych → krew "ucieka" z centralnego krążenia do obwodu → dalszy spadek SV. Souissi et al. (2021) zaproponowali że CV drift może być strategią ochronną — wzrost HR z NO (nitric oxide) chroni mięsień sercowy przed przeciążeniem.

**3. Wzrost Aktywności Współczulnej (Sympathetic Drive)**
Postępująca aktywacja układu współczulnego → przyspieszenie HR niezależnie od SV. Ganio et al. (2021) wykazali że siła skurczu serca rośnie z częstotliwością (force-frequency relationship), co częściowo kompensuje spadek SV.

**4. Spadek VO₂max Skutkiem Driftu**
CV drift redukuje VO₂max o 5-10% podczas długich wysiłków — serce nie jest w stanie dostarczyć wystarczającej ilości O₂ do mięśni mimo stałej mocy.

---

### Scatter Plots: Jak Czytać?

**Power vs HR (Viridis):**
- Gradient koloru (ciemny → jasny) = czas. Jeśli punkty o tej samej mocy są jaśniejsze (później w czasie) i mają wyższe HR → drift.
- **Linia trendu (czerwona przerywana):** Nachylenie = ogólna relacja moc-tętno. Im większe nachylenie, tym mniej ekonomiczny jesteś.
- **Korelacja r w tytule:** r > 0.7 = silna zależność (typowa), r < 0.4 = słaba (może wskazywać na problemy z danymi lub interwały).

**Power vs SmO₂ (Plasma):**
- Gradient koloru = czas. Jeśli punkty o tej samej mocy są ciemniejsze (później) i mają niższe SmO₂ → narastający deficyt tlenowy.
- **Linia trendu (cyjan przerywana):** Ujemne nachylenie = SmO₂ spada z mocą (oczekiwane). Dodatnie nachylenie = anomalia.

---

### Analiza przy Stałej Mocy: Złoty Standard

Najczujsza metoda detekcji driftu — izoluje efekt czasu od efektu zmiany mocy.
- **Segment automatyczny:** Algorytm szuka odcinków ≥2 min z mocą ±10%.
- **Ręczny wybór:** Jeśli nie znaleziono segmentów — wpisz moc i tolerancję.
- **HR Drift Slope** w bpm/min: bezpośrednia miara CV drift.
- **SmO₂ Slope** w %/min: miara narastającego deficytu tlenowego.

---

### Wskazówki Praktyczne

- **HR Drift > 0.5 bpm/min:** Sprawdź nawodnienie (cel: 500-750 ml/h), rozważ chłodzenie (oblania, lód)
- **SmO₂ Drift < -0.3 %/min:** Moc jest zbyt wysoka dla danej kondycji — rozważ obniżenie o 5-10W
- **Aerobic Decoupling (Pwr:HR):** Jeśli stosunek mocy do HR spada >5% w drugiej połowie jazdy — sygnał że baza tlenowa wymaga pracy
- **Barsumyan et al. (2025)** pokazali że wzorce CV drift są unikalne dla każdego kolarza — porównuj drift między sesjami, nie między zawodnikami

---

### Bibliografia

- Souissi et al. (2021). A new perspective on cardiovascular drift during prolonged exercise. *Life Sciences*, 287, 120109.
- Barsumyan et al. (2025). Quantifying training response in cycling based on cardiovascular drift using machine learning. *Frontiers in Artificial Intelligence*, 8, 1623384.
- Ganio et al. (2021). Cardiovascular drift during prolonged exercise: New perspectives. *Exercise and Sport Sciences Reviews*.
- Nuuttila et al. (2024). Monitoring fatigue state with HR-based and subjective methods. *Eur J Sport Sci*, 24(7), 857–869.
- Arnold et al. (2024). Muscle reoxygenation is slower after higher cycling intensity. *Frontiers in Physiology*, 15, 1449384.
        """)


def _display_drift_metrics(metrics) -> None:
    """Display drift metrics in a formatted way."""
    if metrics is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hr_drift = metrics.hr_drift_slope
        if hr_drift is not None:
            delta_color = "inverse" if hr_drift > 0.5 else "normal"
            st.metric(
                "HR Drift",
                f"{hr_drift:.2f} bpm/min",
                delta="znaczący" if hr_drift > 0.5 else "normalny",
                delta_color=delta_color
            )
        else:
            st.metric("HR Drift", "—")
    
    with col2:
        smo2_slope = metrics.smo2_slope
        if smo2_slope is not None:
            delta_color = "inverse" if smo2_slope < -0.3 else "normal"
            st.metric(
                "SmO₂ Slope",
                f"{smo2_slope:.2f} %/min",
                delta="spadek" if smo2_slope < -0.1 else "stabilny",
                delta_color=delta_color
            )
        else:
            st.metric("SmO₂ Slope", "—")
    
    with col3:
        st.metric(
            "Czas segmentu",
            _format_min_to_mmss(metrics.segment_duration_min)
        )
    
    with col4:
        st.metric(
            "Śr. moc",
            f"{metrics.avg_power:.0f} W"
        )
