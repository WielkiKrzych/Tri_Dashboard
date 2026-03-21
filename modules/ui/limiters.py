"""
Limiters tab — identifies the primary performance limiter (central vs local vs metabolic).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_limiters_tab(df_plot, cp_input, vt2_vent):
    st.header("Analiza Limiterów Fizjologicznych (INSCYD-style)")
    st.markdown("Identyfikujemy Twoje ograniczenia metaboliczne i typ zawodniczy na podstawie danych treningowych.")

    # Work on a copy to avoid mutating the caller's DataFrame
    df_plot = df_plot.copy()
    df_plot.columns = df_plot.columns.str.lower().str.strip()

    # Handle HR aliases
    if 'hr' not in df_plot.columns:
        for alias in ['heartrate', 'heart_rate', 'bpm']:
            if alias in df_plot.columns:
                df_plot = df_plot.rename(columns={alias: 'hr'})
                break
    
    has_hr = 'hr' in df_plot.columns
    has_ve = any(c in df_plot.columns for c in ['tymeventilation', 've', 'ventilation'])
    has_smo2 = 'smo2' in df_plot.columns
    has_watts = 'watts' in df_plot.columns

    if has_watts:
        # --- SEKCJA 1: PROFIL METABOLICZNY (INSCYD-style) ---
        st.subheader("🧬 Profil Metaboliczny (Szacunkowy)")
        
        # Oblicz MMP dla różnych okien
        df_plot['mmp_1min'] = df_plot['watts'].rolling(window=60, min_periods=60).mean()
        df_plot['mmp_5min'] = df_plot['watts'].rolling(window=300, min_periods=300).mean()
        df_plot['mmp_20min'] = df_plot['watts'].rolling(window=1200, min_periods=1200).mean()
        
        mmp_1min = df_plot['mmp_1min'].max() if not df_plot['mmp_1min'].isna().all() else 0
        mmp_5min = df_plot['mmp_5min'].max() if not df_plot['mmp_5min'].isna().all() else 0
        mmp_20min = df_plot['mmp_20min'].max() if not df_plot['mmp_20min'].isna().all() else 0
        
        # Klasyfikacja typu zawodnika
        if mmp_20min > 0:
            anaerobic_ratio = mmp_5min / mmp_20min
            sprint_ratio = mmp_1min / mmp_5min if mmp_5min > 0 else 1.0
            
            if anaerobic_ratio > 1.08:
                profile = "🏃 Sprinter / Puncheur"
                vlamax_est = "Wysoki (>0.5 mmol/L/s)"
                profile_color = "#ff6b6b"
                strength = "Krótkie, dynamiczne ataki i sprinty"
                weakness = "Dłuższe wysiłki powyżej progu"
            elif anaerobic_ratio < 0.95:
                profile = "🚴 Climber / TT Specialist"
                vlamax_est = "Niski (<0.4 mmol/L/s)"
                profile_color = "#4ecdc4"
                strength = "Długie, równe tempo, wspinaczki"
                weakness = "Reaktywność na ataki, sprint finiszowy"
            else:
                profile = "⚖️ All-Rounder"
                vlamax_est = "Średni (0.4-0.5 mmol/L/s)"
                profile_color = "#ffd93d"
                strength = "Wszechstronność"
                weakness = "Brak dominującej cechy"
            
            # Wyświetl profil
            col1, col2, col3 = st.columns(3)
            col1.metric("Typ Zawodnika", profile)
            col2.metric("Est. VLaMax", vlamax_est)
            col3.metric("Ratio 5min/20min", f"{anaerobic_ratio:.2f}")
            
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid {profile_color}; background-color: #222;">
                <p style="margin:0;"><b>💪 Mocna strona:</b> {strength}</p>
                <p style="margin:5px 0 0 0;"><b>⚠️ Do poprawy:</b> {weakness}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Trening zbyt krótki dla analizy profilu metabolicznego (min. 20 min).")
            anaerobic_ratio = None
        
        st.divider()

        # --- SEKCJA 2: RADAR LIMITERÓW ---
        if has_hr or has_ve or has_smo2:
            st.subheader("📊 Radar Obciążenia Systemów")
            
            window_options = {
                "1 min (Anaerobic)": 60, 
                "5 min (VO2max)": 300, 
                "20 min (FTP)": 1200,
                "60 min (Endurance)": 3600
            }
            selected_window_name = st.selectbox("Wybierz okno analizy (MMP):", list(window_options.keys()), index=1)
            window_sec = window_options[selected_window_name]

            df_plot['rolling_watts'] = df_plot['watts'].rolling(window=window_sec, min_periods=window_sec).mean()

            if df_plot['rolling_watts'].isna().all():
                st.warning(f"Trening jest krótszy niż {window_sec/60:.0f} min. Wybierz krótsze okno.")
            else:
                peak_idx = df_plot['rolling_watts'].idxmax()

                if not pd.isna(peak_idx):
                    start_idx = max(0, peak_idx - window_sec + 1)
                    df_peak = df_plot.iloc[start_idx:peak_idx+1]
                    
                    # Obliczenia % with z-score normalization
                    # Raw percentages are on different physiological scales,
                    # so we normalize to expected ranges at given intensity
                    # to make cross-system comparison meaningful.

                    peak_hr_avg = df_peak['hr'].mean() if has_hr else 0
                    max_hr_user = df_plot['hr'].max() if has_hr else 1
                    pct_hr = (peak_hr_avg / max_hr_user * 100) if max_hr_user > 0 else 0

                    col_ve_nm = next((c for c in ['tymeventilation', 've', 'ventilation'] if c in df_plot.columns), None)
                    peak_ve_avg = df_peak[col_ve_nm].mean() if col_ve_nm else 0
                    max_ve_user = vt2_vent * 1.1 if vt2_vent > 0 else 1
                    pct_ve = (peak_ve_avg / max_ve_user * 100) if max_ve_user > 0 else 0

                    peak_smo2_avg = df_peak['smo2'].mean() if has_smo2 else 100
                    pct_smo2_util = 100 - peak_smo2_avg

                    peak_w_avg = df_peak['watts'].mean()
                    pct_power = (peak_w_avg / cp_input * 100) if cp_input > 0 else 0

                    # Normalize to comparable 0-100 scale using expected physiological ranges
                    # at maximal effort. Raw % are NOT comparable across systems:
                    # - HR 95% HRmax is normal at VO2max
                    # - SmO2 desaturation >60% is extremely rare
                    # - VE 95% VEmax suggests ventilatory limitation
                    # Expected maximal values (population norms for trained cyclists):
                    expected_max_hr_pct = 98.0     # HR rarely exceeds ~98% HRmax sustained
                    expected_max_ve_pct = 95.0     # VE at max effort ~95% of capacity
                    expected_max_desat = 55.0      # SmO2 desaturation rarely >55% in VL
                    expected_max_power_pct = 120.0 # Can exceed CP in short efforts

                    norm_hr = min(100.0, pct_hr / expected_max_hr_pct * 100)
                    norm_ve = min(100.0, pct_ve / expected_max_ve_pct * 100)
                    norm_smo2 = min(100.0, pct_smo2_util / expected_max_desat * 100)
                    norm_power = min(100.0, pct_power / expected_max_power_pct * 100)

                    # Radar uses normalized values for cross-system comparison
                    categories = ['Serce (norm.)', 'Płuca (norm.)', 'Mięśnie (norm.)', 'Moc (norm.)']
                    values = [norm_hr, norm_ve, norm_smo2, norm_power]
                    values += [values[0]]
                    categories += [categories[0]]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=selected_window_name,
                        line=dict(color='#00cc96'),
                        fillcolor='rgba(0, 204, 150, 0.3)',
                        hovertemplate="%{theta}: <b>%{r:.1f}%</b><extra></extra>"
                    ))

                    max_val = max(values)
                    range_max = 100 if max_val < 100 else (max_val + 10)

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, range_max])),
                        template="plotly_dark",
                        title=f"Profil Obciążenia: {selected_window_name} ({peak_w_avg:.0f} W)",
                        height=450
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Diagnoza — use normalized values for fair comparison
                    limiting_factor = "Serce" if norm_hr >= max(norm_ve, norm_smo2) else ("Płuca" if norm_ve >= norm_smo2 else "Mięśnie")

                    st.markdown(f"""
                    ### 🔍 Diagnoza: {selected_window_name}

                    | System | Surowa | Znormalizowana | Interpretacja |
                    |--------|--------|----------------|---------------|
                    | **Serce** | {pct_hr:.1f}% HRmax | {norm_hr:.0f}/100 | {"🔴 Limiter" if limiting_factor == "Serce" else "🟢 OK"} |
                    | **Płuca** | {pct_ve:.1f}% VEmax | {norm_ve:.0f}/100 | {"🔴 Limiter" if limiting_factor == "Płuca" else "🟢 OK"} |
                    | **Mięśnie** | {pct_smo2_util:.1f}% Desat | {norm_smo2:.0f}/100 | {"🔴 Limiter" if limiting_factor == "Mięśnie" else "🟢 OK"} |
                    | **Moc** | {pct_power:.0f}% CP | {norm_power:.0f}/100 | — |

                    **Główny Limiter: {limiting_factor}**

                    *Normalizacja: surowe % przeliczone na wspólną skalę 0-100 względem oczekiwanych wartości maksymalnych w populacji wytrenowanych kolarzy.*
                    """)
                    
                    # Rekomendacje
                    if limiting_factor == "Serce":
                        st.warning("""
                        **🫀 Ograniczenie Centralne (Serce)**

                        Twoje serce pracuje na maksymalnych obrotach, ale mięśnie mogłyby więcej.

                        **Uwaga terminologiczna:** VT1 (Próg Wentylacyjny) ≠ LT1 (Próg Mleczanowy).
                        Powiązane mechanizmy, ale różne pomiary (Gronwald et al. 2024 meta-analiza).

                        **Sugestie zależne od kontekstu:**
                        - Jeśli objętość Z2 < 6h/tydz: Dodaj więcej Z2 (podniesienie SV)
                        - Jeśli objętość Z2 ≥ 6h/tydz: Dodaj interwały VO₂max 2×/tydz
                        - Interwały 4x8 min @ 88-94% HRmax (Garcia-Tabar 2024)
                        - Sprawdź cardiac drift — jeśli >10%, odwodnienie/upał, nie fitness
                        """)
                    elif limiting_factor == "Płuca":
                        st.warning("""
                        **🫁 Ograniczenie Oddechowe (Płuca)**

                        Wentylacja jest na limicie.

                        **Kontekst:** Wysoki VE/VO₂ slope sugeruje nieefektywną wentylację.
                        Norma: 25-30 L/min na L/min VO₂ (wytrenowani).

                        **Sugestie:**
                        - Sprawdź technikę oddychania (przepona, nie klatka)
                        - Ćwiczenia oddechowe (pranayama, kontrola RR)
                        - Wykluczenie astmy wysiłkowej (EIB) — konsultacja lekarska
                        - Jeśli zimne powietrze: rozgrzej drogi oddechowe
                        """)
                    else:
                        st.warning("""
                        **💪 Ograniczenie Peryferyjne (Mięśnie)**

                        Mięśnie zużywają cały dostarczany tlen (SmO₂ desaturacja wysoka).

                        **Kontekst:** Interpretacja zależy od typu wysiłku:
                        - Sprint: SmO₂ <25% = normalne (anaerobic)
                        - Próg: SmO₂ 40-60% = optymalne
                        - Z2: SmO₂ >65% = oczekiwane

                        **Sugestie:**
                        - Interwały "over-under" (93-97% / 103-107% FTP)
                        - Praca siłowa (squat, single-leg press)
                        - Sprawdź pozycję — okluzja mechaniczna? (SmO₂ zależy od ATT i pozycji sensora)
                        - Monitoruj reoxygenation half-time (<15s = dobra kapilaryzacja)
                        """)
        
        st.divider()
        
        # --- SEKCJA 3: TEORIA INSCYD ---
        with st.expander("📚 Teoria: Model INSCYD i Typy Zawodników", expanded=False):
            st.markdown("""
            ## Model Metaboliczny INSCYD
            
            INSCYD (Power Performance Decoder) to zaawansowany system profilowania metabolicznego, który analizuje interakcję między:
            
            ### 1. VO2max (Zdolność Aerobowa)
            * Maksymalny pobór tlenu [ml/min/kg]
            * Im wyższy, tym więcej energii możesz wytworzyć tlenowo
            * Typowe wartości: Amator 40-50, Pro 70-85+ ml/min/kg
            
            ### 2. VLaMax (Zdolność Glikolityczna)
            * Maksymalna produkcja mleczanu [mmol/L/s]
            * Wysoki VLaMax = mocny sprint, ale szybsze zużycie glikogenu
            * Niski VLaMax = lepsza ekonomia tłuszczowa, wyższy próg
            
            ---
            
            ## Typy Zawodników (Profiling)
            
            | Typ | VO2max | VLaMax | Charakterystyka | Przykłady |
            |-----|--------|--------|-----------------|-----------|
            | **Sprinter** | Średni | Wysoki | Dynamika, punch, sprinty | Sagan, Cavendish, Philipsen |
            | **Climber** | Wysoki | Niski | Długie wspinaczki, tempo | Pogačar, Vingegaard, Yates |
            | **Puncheur** | Wysoki | Średni | Ataki, stromie, krótkie górki | Van Aert, Evenepoel |
            | **Time Trialist** | Wysoki | Niski | Równe tempo, aerodynamika | Ganna, Dennis, Küng |
            | **Rouleur** | Średni | Średni | Klasyki, bruk, wszechstronność | Van der Poel, Pidcock |
            
            ---
            
            ## Interakcja VO2max ↔ VLaMax
            
            ```
            Wysoki VO2max + Niski VLaMax = Wysoki FTP, dobre spalanie tłuszczu
            Wysoki VO2max + Wysoki VLaMax = Mocny sprint, ale niższy próg
            Niski VO2max + Niski VLaMax = Słaba wydolność ogólna
            ```
            
            ---
            
            ## Jak Zmienić Swój Profil?
            
            | Cel | Strategia | Przykładowe Treningi |
            |-----|-----------|---------------------|
            | ⬇️ **Obniżyć VLaMax** | Więcej Z2, mniej sprintów | 3-5h Z2 bez żadnych interwałów |
            | ⬆️ **Podnieść VO2max** | Interwały w Z5 | 5x5 min @ 105-110% FTP |
            | ⬆️ **Podnieść FatMax** | Train Low, Z2 na czczo | Poranki Z2 bez śniadania |
            | ⬆️ **Podnieść FTP** | Sweet Spot, Threshold | 2x20 min @ 88-94% FTP |
            
            ---
            
            ## Krzywa FatMax
            
            FatMax to intensywność, przy której spalasz najwięcej tłuszczu (zwykle 55-65% FTP).
            
            * Poniżej FatMax: Spalasz mniej energii ogółem
            * Powyżej FatMax: Spalanie tłuszczu spada, węgle dominują
            * Cel treningu Z2: Przesunąć FatMax w prawo (wyższa moc przy max spalaniu tłuszczu)
            
            *Ten kalkulator szacuje Twój profil na podstawie stosunku mocy 5min/20min.*
            """)
    else:
        st.error("Brakuje danych mocy (Watts) do analizy limiterów.")
