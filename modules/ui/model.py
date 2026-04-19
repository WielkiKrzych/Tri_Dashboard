"""
Performance Model tab — CP/W' model visualisation and fatigue curve.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from modules.plots import CHART_CONFIG


def render_model_tab(df_plot, cp_input, w_prime_input):
    from scipy import stats

    st.header("Matematyczny Model CP (Critical Power Estimation)")
    st.markdown(
        "Estymacja Twojego CP i W' na podstawie krzywej mocy (MMP) z tego treningu. Używamy modelu liniowego: `Praca = CP * t + W'`."
    )

    if "watts" in df_plot.columns and len(df_plot) > 1200:  # Minimum 20 minut danych
        # 1. Wybieramy punkty czasowe do modelu (standardowe dla modelu 2-parametrowego)
        # Unikamy bardzo krótkich czasów (< 2-3 min), bo tam dominuje Pmax/AC
        durations = [180, 300, 600, 900, 1200]  # 3min, 5min, 10min, 15min, 20min

        # Filtrujemy czasy dłuższe niż długość treningu
        valid_durations = [d for d in durations if d < len(df_plot)]

        if len(valid_durations) >= 3:  # Potrzebujemy min. 3 punktów do sensownej regresji
            mmp_values = []
            work_values = []

            # Liczymy MMP i Pracę dla każdego punktu
            for d in valid_durations:
                # Rolling mean max
                p = df_plot["watts"].rolling(window=d).mean().max()
                if not pd.isna(p):
                    mmp_values.append(p)
                    # Praca [J] = Moc [W] * Czas [s]
                    work_values.append(p * d)

            # 2. Regresja Liniowa (Work vs Time)
            # Y = Work, X = Time
            # Slope = CP, Intercept = W'
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_durations, work_values
            )

            modeled_cp = slope
            modeled_w_prime = intercept
            r_squared = r_value**2

            # 3. Wyświetlenie Wyników
            c_res1, c_res2, c_res3 = st.columns(3)

            c_res1.metric(
                "Estymowane CP (z pliku)",
                f"{modeled_cp:.0f} W",
                delta=f"{modeled_cp - cp_input:.0f} W vs Ustawienia",
                help="Moc Krytyczna wyliczona z Twoich najmocniejszych odcinków w tym pliku.",
            )

            c_res2.metric(
                "Estymowane W'",
                f"{modeled_w_prime:.0f} J",
                delta=f"{modeled_w_prime - w_prime_input:.0f} J vs Ustawienia",
                help="Pojemność beztlenowa wyliczona z modelu.",
            )

            c_res3.metric(
                "Jakość Dopasowania (R²)",
                f"{r_squared:.4f}",
                delta_color="normal" if r_squared > 0.98 else "inverse",
                help="Jak bardzo Twoje wyniki pasują do teoretycznej krzywej. >0.98 = Bardzo wiarygodne.",
            )

            st.markdown("---")

            # 4. Wizualizacja: Krzywa MMP vs Krzywa Modelowa
            # Generujemy punkty teoretyczne dla zakresu 1 min - 30 min
            x_theory = np.arange(60, 1800, 60)  # co minutę
            y_theory = [modeled_cp + (modeled_w_prime / t) for t in x_theory]

            # Rzeczywiste MMP z pliku dla tych samych czasów
            y_actual = []
            x_actual = []
            for t in x_theory:
                if t < len(df_plot):
                    val = df_plot["watts"].rolling(t).mean().max()
                    y_actual.append(val)
                    x_actual.append(t)

            fig_model = go.Figure()

            # Rzeczywiste MMP
            fig_model.add_trace(
                go.Scatter(
                    x=np.array(x_actual) / 60,
                    y=y_actual,
                    mode="markers",
                    name="MMP (Plik)",
                    marker=dict(color="#00cc96", size=8),
                    hovertemplate="<b>MMP:</b> %{y:.0f} W<br><b>Czas:</b> %{x:.1f} min<extra></extra>",
                )
            )

            # Model Teoretyczny
            fig_model.add_trace(
                go.Scatter(
                    x=x_theory / 60,
                    y=y_theory,
                    mode="lines",
                    name=f"Model: {modeled_cp:.0f}W",
                    line=dict(color="#ef553b", dash="dash"),
                    hovertemplate="<b>Model:</b> %{y:.0f} W<br><b>Czas:</b> %{x:.1f} min<extra></extra>",
                )
            )

            fig_model.update_layout(
                template="plotly_dark",
                title="Power Duration Curve: Rzeczywistość vs Model",
                xaxis_title="Czas trwania [min]",
                yaxis_title="Moc [W]",
                yaxis=dict(tickformat=".0f"),
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig_model, width="stretch", config=CHART_CONFIG)

            # 5. Interpretacja
            with st.expander("📖 Model CP/W' — Teoria i Fizjologia", expanded=False):
                st.markdown("""
### Definicja: Critical Power (CP) i W'

**Critical Power (CP)** to asymptota hiperbolicznej relacji między mocą a czasem do wyczerpania. Reprezentuje górną granicę intensywności, przy której organizm utrzymuje homeostazę metaboliczną — powyżej CP nie ma stanu ustalonego (steady-state) (Goulding & Marwood, 2023).

**W'** (Work Capacity above CP) to skończona pojemność pracy beztlenowej wyrażona w dżulach. Działa jak "bateria" — każdy wysiłek powyżej CP zużywa W', a odpoczynek poniżej CP ją regeneruje (Chorley et al., 2022).

---

### Tabela Interpretacji Wyników

| Parametr | Zakres | Interpretacja |
|---|---|---|
| **CP** | >95% ustawionego | Doskonała zgodność — Twoje ustawienia odzwierciedlają aktualną formę |
| **CP** | 85-95% ustawionego | Lekki spadek lub trening nie był maksymalny — typowe dla jazdy grupowej |
| **CP** | <85% ustawionego | Znaczna różnica — albo forma spadła, albo brakowało wysiłków all-out |
| **W'** | 15-25 kJ | Typowy zakres dla wytrenowanych kolarzy (Lamb et al., 2020) |
| **W'** | >25 kJ | Wysoka pojemność beztlenowa — predyspozycje do sprintów/ataków |
| **W'** | <15 kJ | Niska pojemność beztlenowa — typowe dla specjalistów endurance |
| **R²** | >0.98 | Bardzo wiarygodne dopasowanie modelu |
| **R²** | 0.95-0.98 | Dobre dopasowanie, ale rozważ powtórzenie testu |
| **R²** | <0.95 | Niska wiarygodność — jazda nieregularna, za mało punktów MMP |

---

### 4 Mechanizmy Fizjologiczne CP

**1. Dostawa i Wykorzystanie Tlenu (Goulding & Marwood, 2023)**
CP jest parametrem funkcji tlenowej — zależy od każdego ogniwa łańcucha transportu O₂: dostawa konwekcyjna (serce → naczynia → krew), dyfuzyjna (naczynia włosowate → mitochondria) i wykorzystanie wewnątrzkomórkowe. Trening wytrzymałościowy poprawia każde z tych ogniw.

**2. Rekrutacja Jednostek Motorycznych**
Powyżej CP organizm musi rekrutować włókna typu II (szybkokurczliwe), które mają niższą wydajność tlenową i szybciej generują metabolity. Punkt, w którym dzieje się to masowo, wyznacza CP.

**3. Związek z Progami Wentylacyjnymi**
CP koreluje silnie z VT2/RCP (r ≈ 0.85-0.95). W praktyce CP ≈ moc na VT2 ± 5-10W. To sprawia, że CP jest fizjologicznym odpowiednikiem "progu mleczanowego" w świecie mocy.

**4. Regeneracja W' — Model Bi-Wykładniczy (Chorley et al., 2022)**
Regeneracja W' ma dwie fazy:
- **Szybka faza (FC):** τ ≈ 20-30s — regeneracja PCr (fosfokreatyny)
- **Wolna faza (SC):** τ ≈ 150-300s — usuwanie metabolitów, przywrócenie homeostazy

Pełna regeneracja W' po maksymalnym wysiłku zajmuje 10-15 minut jazdy poniżej CP.

---

### Metody Wyznaczania CP

Istnieje kilka podejść (Lipková et al., 2022):
- **Model liniowy (Work-Time):** Używany tutaj — regresja Praca vs Czas. Slope = CP, Intercept = W'.
- **Model 3-min All-Out:** Test do wyczerpania — końcowa moc = CP (Klimstra, 2024).
- **Model MMP:** Dopasowanie krzywej mocy-czas z danych z pliku — najwygodniejsze, ale wymaga wysiłków all-out w zakresie 3-20 min.

**⚠️ Ważne:** Model z pliku jest ważny TYLKO jeśli zawierałeś wysiłki maksymalne (all-out) w zakresie 3-20 minut. Bez nich CP będzie zaniżone — model pokaże to, co *zademonstrowałeś*, a nie Twój absolutny potencjał.

---

### Wskazówki Praktyczne

- **CP > ustawione:** Rozważ aktualizację w sidebarze — jazda na starych wartościach zaniża Twoje strefy treningowe
- **CP < ustawione:** To normalne w dni "pod górkę" lub przy jeździe grupowej. Powtórz test po odpoczynku
- **W' niskie:** Rozważ trening interwałowy powyżej CP (np. 4x4min na 110% CP)
- **W' wysokie:** Wykorzystaj w wyścigach — masz duży "bufor" na ataki i sprinty

---

### Bibliografia

- Goulding & Marwood (2023). Interaction of factors determining critical power. *Sports Medicine*, 53, 595–613.
- Chorley et al. (2022). Bi-exponential modelling of W′ reconstitution kinetics in trained cyclists. *European Journal of Applied Physiology*, 122, 677–689.
- Lipková et al. (2022). Determination of critical power using different possible approaches among endurance athletes: A review. *Int. J. Environ. Res. Public Health*, 19, 7589.
- Lamb et al. (2020). The application of critical power, W′, and its reconstitution: A narrative review. *Sports*, 8(9), 123.
- Klimstra (2024). Estimate anaerobic work capacity and critical power with constant-power all-out test. *J. Funct. Morphol. Kinesiol.*, 9(4), 202.
                """)

        else:
            st.warning(
                "Trening jest zbyt krótki lub brakuje mocnych odcinków, by zbudować wiarygodny model CP (wymagane wysiłki > 3 min i > 10 min)."
            )
    else:
        st.warning("Za mało danych (wymagane min. 20 minut jazdy z pomiarem mocy).")
