"""
AI Coach tab — GPT-powered personalised training feedback.

Renders training load analysis, recovery status, and AI-generated
coaching recommendations based on session metrics.
"""
import logging
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from modules.ml_logic import MLX_AVAILABLE, train_cycling_brain, HISTORY_FILE
from modules.ui.callbacks import StreamlitCallback

logger = logging.getLogger(__name__)

def render_ai_coach_tab(df_plot_resampled, cp_watts: float = 0.0):
    """Render AI Coach tab with dynamic power targets based on CP.

    Args:
        df_plot_resampled: Resampled workout DataFrame
        cp_watts: Critical Power in watts (used for dynamic base/threshold targets)
    """
    # Dynamic power targets based on athlete's CP (not hardcoded)
    base_power = int(cp_watts * 0.75) if cp_watts > 0 else 280
    threshold_power = int(cp_watts * 0.95) if cp_watts > 0 else 360

    st.header("🧠 AI Neural Coach (Powered by Apple MLX)")
    source_label = "CP" if cp_watts > 0 else "domyślne"
    st.caption(f"Analiza 'Bazy Tlenowej' ({base_power}W, 75% CP) oraz 'Silnika' ({threshold_power}W, 95% CP) [{source_label}]")

    if MLX_AVAILABLE:
        col_ai_1, col_ai_2 = st.columns([1, 2])
        
        with col_ai_1:
            st.info("System Neuralny Gotowy.")
            if st.button("🚀 Trenuj Mózg (Aktualizuj)", type="primary"):
                with st.spinner("Trening sieci neuronowej (200 epok)..."):
                    try:
                        # SOLID (DIP): Używamy StreamlitCallback zamiast hardkodowanych st.*
                        callback = StreamlitCallback()
                        y_p, b_val, t_val, was_loaded, history = train_cycling_brain(
                            df_plot_resampled, 
                            epochs=200,
                            callback=callback
                        )
                        
                        b_str = f"{b_val:.1f}" if b_val is not None else "N/A"
                        t_str = f"{t_val:.1f}" if t_val is not None else "N/A"
                        st.success(f"✅ Trening Zakończony! Baza: {b_str}, Próg: {t_str}")
                        st.toast("Odświeżanie...", icon="🔄")
                        st.rerun()
                    except Exception as e:
                        logger.error("AI training failed: %s", e, exc_info=True)
                        st.error("Blad treningu modelu. Sprawdz logi.")
            
            last_base, last_thresh = "-", "-"
            
            if os.path.exists(HISTORY_FILE):
                try:
                    with open(HISTORY_FILE, 'r') as f:
                        h_data = json.load(f)
                        
                        if h_data:
                            for entry in reversed(h_data):
                                val = entry.get('hr_base')
                                if val is not None and val != "None":
                                    last_base = f"{float(val):.1f}"
                                    break
                            
                            for entry in reversed(h_data):
                                val = entry.get('hr_thresh')
                                if val is not None and val != "None":
                                    last_thresh = f"{float(val):.1f}"
                                    break
                                    
                except Exception as e:
                    logger.warning("Błąd odczytu historii: %s", e)
            
            st.markdown("### Aktualna Forma")
            k1, k2 = st.columns(2)
            k1.metric(f"Baza ({base_power}W)", f"{last_base} bpm", help=f"Oczekiwane tętno przy {base_power}W (75% CP) @ 80rpm")
            k2.metric(f"Próg ({threshold_power}W)", f"{last_thresh} bpm", help=f"Oczekiwane tętno przy {threshold_power}W (95% CP) @ 80rpm")

        with col_ai_2:
            # --- NOWY WYKRES DWULINIOWY (POPRAWIONY - OŚ X TO NUMER SESJI) ---
            if os.path.exists(HISTORY_FILE):
                try:
                    with open(HISTORY_FILE, 'r') as f:
                        hist_data = json.load(f)
                    
                    if len(hist_data) > 0:
                        hist_df = pd.DataFrame(hist_data)
                        
                        hist_df = hist_df.reset_index()
                        hist_df['session_nr'] = hist_df.index + 1
                        
                        import html as _html
                        hover_text_base = hist_df.apply(
                            lambda row: f"Plik: {_html.escape(str(row.get('source_file', 'N/A')))}<br>Baza: {row['hr_base']:.1f} bpm"
                            if row['hr_base'] is not None and not pd.isna(row['hr_base']) else "N/A",
                            axis=1
                        )
                        hover_text_thresh = hist_df.apply(
                            lambda row: f"Plik: {_html.escape(str(row.get('source_file', 'N/A')))}<br>Próg: {row['hr_thresh']:.1f} bpm"
                            if row['hr_thresh'] is not None and not pd.isna(row['hr_thresh']) else "N/A",
                            axis=1
                        )

                        fig_evo = go.Figure()
                        
                        # Linia 1: Baza (280W)
                        fig_evo.add_trace(go.Scatter(
                            x=hist_df['session_nr'], 
                            y=hist_df['hr_base'], 
                            mode='lines+markers',
                            name=f'Baza ({base_power}W)',
                            line=dict(color='#00cc96', width=3), # Zielony
                            marker=dict(size=6),
                            hovertext=hover_text_base,
                            hoverinfo="text"
                        ))
                        
                        # Linia 2: Próg (360W)
                        fig_evo.add_trace(go.Scatter(
                            x=hist_df['session_nr'], 
                            y=hist_df['hr_thresh'], 
                            mode='lines+markers',
                            name=f'Próg ({threshold_power}W)',
                            line=dict(color='#ef553b', width=3), # Czerwony
                            marker=dict(size=6),
                            hovertext=hover_text_thresh,
                            hoverinfo="text"
                        ))
                        
                        fig_evo.update_layout(
                            template="plotly_dark",
                            title="Ewolucja Formy: Baza vs Próg",
                            xaxis_title="Kolejne Treningi (Sesja #)",
                            yaxis_title="HR [bpm] (Im niżej tym lepiej)",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=1.1, x=0),
                            height=350
                        )
                        st.plotly_chart(fig_evo, width="stretch")
                except Exception as e:
                    logger.error("AI history chart failed: %s", e, exc_info=True)
                    st.error("Blad wykresu historii. Sprawdz logi.")

        st.divider()
        
        if 'ai_hr' in df_plot_resampled.columns:
            st.subheader("Analiza: Rzeczywistość vs AI")
            fig_ai_comp = go.Figure()
            fig_ai_comp.add_trace(go.Scatter(
                x=df_plot_resampled['time_min'], 
                y=df_plot_resampled['heartrate_smooth'], 
                name='Rzeczywiste HR', 
                line=dict(color='#ef553b', width=2),
                hovertemplate="<b>Czas:</b> %{x:.0f} min<br><b>Rzeczywiste HR:</b> %{y:.1f} bpm<extra></extra>"
            ))
            fig_ai_comp.add_trace(go.Scatter(
                x=df_plot_resampled['time_min'], 
                y=df_plot_resampled['ai_hr'], 
                name='AI Model HR (Oczekiwane)', 
                line=dict(color='#00cc96', dash='dot', width=2),
                hovertemplate="<b>Czas:</b> %{x:.0f} min<br><b>AI Model HR (Oczekiwane):</b> %{y:.1f} bpm<extra></extra>"
            ))
            
            fig_ai_comp.update_layout(
                template="plotly_dark", 
                title="Czy serce reagowało zgodnie z planem?", 
                xaxis=dict(
                    title="Czas [min]",
                    tickformat=".0f",
                    hoverformat=".0f"
                ),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0)
            )
            st.plotly_chart(fig_ai_comp, width="stretch")
            
            diff = df_plot_resampled['heartrate_smooth'] - df_plot_resampled['ai_hr']
            avg_diff = diff.mean()
            
            if avg_diff > 3:
                st.warning(f"⚠️ **Wysoki Dryf Dnia (+{avg_diff:.1f} bpm):** Twoje tętno było wyższe niż model oczekiwał dla tej mocy. Możliwe zmęczenie, choroba lub upał.")
            elif avg_diff < -3:
                st.success(f"✅ **Dzień Konia ({avg_diff:.1f} bpm):** Tętno niższe niż zazwyczaj. Świetna dyspozycja!")
            else:
                st.info(f"🆗 **Norma ({avg_diff:.1f} bpm):** Reakcja serca zgodna z Twoim profilem historycznym.")

        # H2: Cardiac Drift Analysis (Sperlich et al. 2025, Papini et al. 2024)
        st.divider()
        st.subheader("📉 Cardiac Drift (HR:Power Decoupling)")

        if 'heartrate_smooth' in df_plot_resampled.columns and 'watts' in df_plot_resampled.columns:
            # Filter for active riding (>50W for >5min)
            df_active = df_plot_resampled[df_plot_resampled['watts'] > 50].copy()

            if len(df_active) > 600:  # Need >10 min of data
                mid_point = len(df_active) // 2
                first_half = df_active.iloc[:mid_point]
                second_half = df_active.iloc[mid_point:]

                # HR:Power ratio (efficiency factor)
                ef_first = first_half['watts'].mean() / max(1, first_half['heartrate_smooth'].mean())
                ef_second = second_half['watts'].mean() / max(1, second_half['heartrate_smooth'].mean())

                # Decoupling percentage (>5% = aerobic deficiency per coaching heuristic)
                if ef_first > 0:
                    decoupling_pct = ((ef_first - ef_second) / ef_first) * 100
                else:
                    decoupling_pct = 0.0

                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric("EF 1. połowa", f"{ef_first:.2f} W/bpm")
                col_d2.metric("EF 2. połowa", f"{ef_second:.2f} W/bpm")
                col_d3.metric("Decoupling", f"{decoupling_pct:.1f}%",
                             delta=f"{decoupling_pct:.1f}%" if decoupling_pct > 0 else None,
                             delta_color="inverse" if decoupling_pct > 5 else "normal")

                if decoupling_pct > 10:
                    st.error(
                        f"⚠️ **Wysoki cardiac drift ({decoupling_pct:.1f}%):** "
                        "Możliwe przyczyny: odwodnienie, stress termiczny, "
                        "wyczerpanie glikogenu lub zbyt wysoka intensywność. "
                        "(Sperlich et al. 2025: 93.1% accuracy CV drift → fitness prediction)"
                    )
                elif decoupling_pct > 5:
                    st.warning(
                        f"🟡 **Umiarkowany drift ({decoupling_pct:.1f}%):** "
                        "Decoupling >5% sugeruje, że baza aerobowa wymaga pracy. "
                        "Rozważ więcej Z2. (Papini et al. 2024)"
                    )
                elif decoupling_pct <= 2:
                    st.success(
                        f"✅ **Świetna stabilność ({decoupling_pct:.1f}%):** "
                        "Minimalne decoupling — dobra baza aerobowa."
                    )
                else:
                    st.info(
                        f"🆗 **Normalny drift ({decoupling_pct:.1f}%):** "
                        "W granicach normy dla sesji treningowej."
                    )
            else:
                st.info("Za mało danych aktywnych (min. 10 min >50W) dla analizy cardiac drift.")

    else:
        st.warning("⚠️ Moduł AI wymaga procesora Apple Silicon i biblioteki `mlx`. Zainstaluj: `pip install mlx`")
