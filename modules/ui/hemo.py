"""
Haemodynamics tab — cardiac output, stroke volume, and SpO2 trends.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from modules.plots import CHART_CONFIG
from modules.cache_utils import cached_rolling_mean


def render_hemo_tab(target_df):
    st.header("Profil Hemodynamiczny (Mechanika vs Metabolizm)")
    st.markdown(
        "Analiza relacji objętości krwi (THb) do saturacji (SmO2). Pozwala wykryć okluzję (ucisk) i limitery przepływu."
    )

    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik.")
        return

    col_thb = next(
        (c for c in ["thb", "total_hemoglobin", "total_hgb"] if c in target_df.columns), None
    )
    col_smo2 = (
        "smo2_smooth" if "smo2_smooth" in target_df else ("smo2" if "smo2" in target_df else None)
    )

    if col_thb and col_smo2:
        if f"{col_thb}_smooth" not in target_df.columns:
            target_df[f"{col_thb}_smooth"] = cached_rolling_mean(
                target_df[col_thb], window=10, center=True
            )

        thb_val = f"{col_thb}_smooth"

        if "smo2" in target_df.columns:
            target_df["smo2_smooth_10s_hemo_trend"] = cached_rolling_mean(
                target_df["smo2"], window=10, center=True
            )
            col_smo2_hemo_trend = "smo2_smooth_10s_hemo_trend"
        else:
            col_smo2_hemo_trend = col_smo2

        # 2. Wykres XY (Scatter) - SmO2 vs THb
        # Kolorujemy punktami Mocy, żeby widzieć co się dzieje na wysokich watach

        # Próbkowanie dla szybkości (oryginalne zachowanie)
        df_hemo = target_df.sample(min(len(target_df), 5000))

        fig_hemo = px.scatter(
            df_hemo,
            x=col_smo2,
            y=thb_val,
            color="watts" if "watts" in df_hemo.columns else None,
            title="Hemo-Scatter: SmO2 (Oś X) vs THb (Oś Y)",
            labels={
                col_smo2: "SmO2 (Saturacja) [%]",
                thb_val: "THb (Objętość Krwi) [a.u.]",
                "watts": "Moc [W]",
            },
            hover_data={
                col_smo2: ":.1f",
                thb_val: ":.1f",
                "watts": ":.0f" if "watts" in df_hemo.columns else False,
            },
            template="plotly_dark",
            color_continuous_scale="Turbo",
        )

        # Odwracamy oś X dla SmO2 (zwyczajowo w fizjologii wykresy czyta się od prawej do lewej dla desaturacji)
        fig_hemo.update_xaxes(autorange="reversed")

        fig_hemo.update_traces(marker=dict(size=5, opacity=0.6))
        fig_hemo.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))

        # Dodajemy adnotacje "ćwiartek" (Uproszczona interpretacja)
        # To wymagałoby znania średnich, ale damy opisy w rogach
        fig_hemo.add_annotation(
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            text="<b>Stres Metaboliczny</b><br>(Wazodylatacja)",
            showarrow=False,
            font=dict(color="#00cc96"),
        )
        fig_hemo.add_annotation(
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.05,
            text="<b>OKLUZJA / UCISK</b><br>(Limit Przepływu)",
            showarrow=False,
            font=dict(color="#ef553b"),
        )
        fig_hemo.add_annotation(
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.95,
            text="<b>Regeneracja</b><br>(Napływ)",
            showarrow=False,
            font=dict(color="#ffa15a"),
        )

        st.plotly_chart(fig_hemo, width="stretch", config=CHART_CONFIG)

        # 3. Wykres Liniowy w czasie (Dual Axis)
        st.subheader("Trendy w Czasie (Szukanie Rozjazdu)")

        # Prepare time formatting for hover
        if "time" in target_df.columns:
            time_str_trend = pd.to_datetime(target_df["time"], unit="s").dt.strftime("%H:%M:%S")
        else:
            time_str_trend = target_df["time_min"].apply(
                lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}:00"
            )

        fig_trend = go.Figure()

        # SmO2 (Oś Lewa)
        fig_trend.add_trace(
            go.Scatter(
                x=target_df["time_min"],
                y=target_df[col_smo2_hemo_trend],
                customdata=time_str_trend,
                name="SmO2",
                line=dict(color="#ab63fa", width=2),
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>SmO2:</b> %{y:.1f}%<extra></extra>",
            )
        )

        # THb (Oś Prawa)
        fig_trend.add_trace(
            go.Scatter(
                x=target_df["time_min"],
                y=target_df[thb_val],
                customdata=time_str_trend,
                name="THb",
                line=dict(color="#ffa15a", width=2),
                yaxis="y2",
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>THb:</b> %{y:.2f} g/dL<extra></extra>",
            )
        )

        # Tło - Moc (dla kontekstu)
        if "watts_smooth_30s" in target_df:
            fig_trend.add_trace(
                go.Scatter(
                    x=target_df["time_min"],
                    y=target_df["watts_smooth_30s"],
                    name="Moc",
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(255,255,255,0.05)",
                    yaxis="y3",
                    hoverinfo="skip",
                )
            )

        # Poprawiony Layout dla fig_trend (bez titlefont)
        fig_trend.update_layout(
            template="plotly_dark",
            title="SmO2 vs THb w Czasie",
            xaxis=dict(title="Czas [min]", tickformat=".0f", hoverformat=".0f"),
            hovermode="x unified",
            yaxis=dict(title=dict(text="SmO2 [%]", font=dict(color="#ab63fa"))),
            yaxis2=dict(
                title=dict(text="THb [g/dL]", font=dict(color="#ffa15a")),
                overlaying="y",
                side="right",
            ),
            yaxis3=dict(
                title="Moc", overlaying="y", side="right", showgrid=False, showticklabels=False
            ),
            height=450,
        )
        st.plotly_chart(fig_trend, width="stretch", config=CHART_CONFIG)

        # 4. Teoria dla Fizjologii
        with st.expander("📖 Hemodynamika (THb + SmO2) — Teoria i Fizjologia", expanded=False):
            st.markdown("""
### Definicja: THb i SmO2 w Kontekście Hemodynamicznym

**THb (Total Hemoglobin)** to suma hemoglobiny utlenowanej i odtlenowanej w polu pomiaru NIRS. Odzwierciedla **objętość krwi** w mikrokrążeniu mięśniowym — "pojemność baku paliwowego" (Dennis et al., 2021).

**SmO2 (Muscle Oxygen Saturation)** to procent utlenowanej hemoglobiny względem całkowitej. Odzwierciedla **balans między dostawą a zużyciem O₂** — "wskaźnik spalania paliwa" (Cherouveim et al., 2023).

Razem tworzą **mapę hemodynamiczną** — relację między pompą (THb) a metabolizmem (SmO2).

---

### Mapa Ćwiartek: 4 Scenariusze Hemodynamiczne

| SmO2 ↓ / THb → | **THb ROŚNIE 📈** | **THb SPADA 📉** |
|---|---|---|
| **SmO2 SPADA 📉** | **ĆWIARTKA 1: Wazodylatacja** ✅<br>Zdrowa odpowiedź — mięsień zużywa O₂, naczynia się rozszerzają, napływa więcej krwi | **ĆWIARTKA 2: Okluzja** 🔴<br>Ciśnienie wewnątrzmięśniowe > ciśnienie perfuzji. Krew nie dopływa. "Wyżymanie gąbki" |
| **SmO2 ROŚNIE 📈** | **ĆWIARTKA 3: Venous Pooling** ⚠️<br>Krew napływa, ale nie jest odprowadzana. Zastój żylny — typowe po nagłym zatrzymaniu | **ĆWIARTKA 4: Regeneracja** ✅<br>Krew odpływa z metabolitami, świeża krew dostarcza O₂. Typowe w fazie recovery |

---

### Mechanizmy Fizjologiczne

**1. Wazodylatacja Ćwiczeniowa (ĆWIARTKA 1)**
Podczas wysiłku, metabolity (CO₂, H⁺, adenozyna, NO) powodują rozszerzenie naczyń — THb rośnie. Jednocześnie rosnące zapotrzebowanie na O₂ obniża SmO2. To **zdrowa odpowiedź** — układ krążenia nadąża za metabolizmem. Cherouveim et al. (2023) pokazali że restrykcja przepływu krwi (thigh cuffs 120 mmHg) obniża VO₂max o 17% i moc szczytową o 28% — dowód na krytyczną rolę perfuzji.

**2. Okluzja Mechaniczna (ĆWIARTKA 2)**
Gdy ciśnienie wewnątrzmięśniowe (IMP) przekracza ciśnienie perfuzji włosowatej (~25-35 mmHg), przepływ krwi zostaje zablokowany. THb spada (krew jest "wyciskana" z naczyń), a SmO2 gwałtownie leci w dół (brak nowej dostawy O₂). Dennis et al. (2021) wykazali że NIRS jest walidowanym narzędziem do pomiaru przepływu krwi w mięśniach — spadek THb koreluje z redukcją perfuzji.

**3. Venous Pooling (ĆWIARTKA 3)**
Po nagłym zatrzymaniu wysiłku, pompa mięśniowa (skurcze mięśni nóg pompujące krew żylną) przestaje działać. Krew napływa tętniczo (THb rośnie) ale nie jest efektywnie odprowadzana żylnie. SmO2 rośnie (brak zużycia), ale krew "stoi" — może to prowadzić do zawrotów głowy i omdleń.

**4. Regeneracja Aktywna (ĆWIARTKA 4)**
Podczas lekkiego pedałowania recovery, pompa mięśniowa działa — krew jest efektywnie pompowana przez mięsień. THb spada (krew "przepływa" a nie "stoi"), SmO2 rośnie (dostawa > zużycie). To optymalny stan regeneracji.

---

### Hemo-Scatter: Jak Czytać Wykres Punktowy?

Wykres SmO2 vs THb z kolorowaniem według mocy pokazuje **całą historię hemodynamiczną** treningu:
- **Punkty czerwone (wysoka moc)** w lewym dolnym rogu = okluzja przy dużym wysiłku
- **Punkty zielone (niska moc)** w prawym górnym rogu = regeneracja z pełną perfuzją
- **Gradient kolorów** pokazuje płynne przejście między stanami metabolicznymi

**Oś X odwrócona** (SmO2 od prawej do lewej) — konwencja fizjologiczna: desaturacja = ruch w prawo = "gorzej".

---

### Trendy w Czasie: Czego Szukać?

- **Rozjazd SmO2 i THb:** Jeśli SmO2 spada a THb pozostaje płaskie — sygnał ostrzegawczy. Układ krążenia nie reaguje wazodylatacją na rosnące zapotrzebowanie.
- **Nagłe skoki THb:** Mogą wskazywać na zmianę pozycji, uderzenie w siodełko, lub artefakt ruchu.
- **Powolna reoksygenacja po wysiłku:** Jeśli SmO2 wolno wraca do baseline po interwale — może wskazywać na zmęczenie mitochondrialne lub ograniczoną perfuzję.

---

### Bibliografia

- Cherouveim et al. (2023). The effect of skeletal muscle oxygenation on hemodynamics, cerebral oxygenation and activation, and exercise performance during incremental exercise to exhaustion in male cyclists. *Biology*, 12(7), 981.
- Dennis et al. (2021). Measurement of muscle blood flow and O₂ uptake via near-infrared spectroscopy using a novel occlusion protocol. *Scientific Reports*, 11, 918.
- Cross et al. (2021). Muscle oximetry in sports science: An updated systematic review. *Sports Medicine*.
- Kilgas et al. (2022). Physiological responses to acute cycling with blood flow restriction. *Frontiers in Physiology*, 13, 800155.
- Arnold et al. (2024). Muscle reoxygenation is slower after higher cycling intensity. *Frontiers in Physiology*, 15, 1449384.
            """)

    else:
        st.warning(
            "⚠️ Brak danych THb (Total Hemoglobin). Sensor Moxy/Train.Red powinien dostarczać tę kolumnę (często jako 'thb' lub 'total_hemoglobin'). Bez tego analiza hemodynamiczna jest niemożliwa."
        )
        st.markdown("Dostępne kolumny w pliku: " + ", ".join(target_df.columns))
