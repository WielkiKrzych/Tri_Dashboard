import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time
from modules.ml_logic import MLX_AVAILABLE, train_cycling_brain, HISTORY_FILE

def render_ai_coach_tab(df_plot_resampled):
    st.header("🧠 AI Neural Coach (Powered by Apple MLX)")
    st.caption("Analiza 'Bazy Tlenowej' (280W) oraz 'Silnika' (360W)")

    if MLX_AVAILABLE:
        col_ai_1, col_ai_2 = st.columns([1, 2])
        
        with col_ai_1:
            st.info("System Neuralny Gotowy.")
            if st.button("🚀 Trenuj Mózg (Aktualizuj)", type="primary"):
                with st.spinner("Trening sieci neuronowej (200 epok)..."):
                    try:
                        # Trenujemy na resamplowanych danych (1 sekunda)
                        y_p, b_val, t_val, was_loaded, history = train_cycling_brain(df_plot_resampled, epochs=200)
                        
                        st.success(f"✅ Trening Zakończony! Baza: {b_val:.1f}, Próg: {t_val:.1f}")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Błąd treningu: {e}") 
            
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
                    print(f"Błąd odczytu historii: {e}")
            
            st.markdown("### Aktualna Forma")
            k1, k2 = st.columns(2)
            k1.metric("Baza (280W)", f"{last_base} bpm", help="Oczekiwane tętno przy 280W @ 80rpm")
            k2.metric("Próg (360W)", f"{last_thresh} bpm", help="Oczekiwane tętno przy 360W @ 80rpm")

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
                        
                        hover_text_base = hist_df.apply(lambda row: f"Plik: {row.get('source_file', 'N/A')}<br>Baza: {row['hr_base']:.1f} bpm", axis=1)
                        hover_text_thresh = hist_df.apply(lambda row: f"Plik: {row.get('source_file', 'N/A')}<br>Próg: {row['hr_thresh']:.1f} bpm", axis=1)

                        fig_evo = go.Figure()
                        
                        # Linia 1: Baza (280W)
                        fig_evo.add_trace(go.Scatter(
                            x=hist_df['session_nr'], 
                            y=hist_df['hr_base'], 
                            mode='lines+markers',
                            name='Baza (280W)',
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
                            name='Próg (360W)',
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
                        st.plotly_chart(fig_evo, use_container_width=True)
                except Exception as e:
                    st.error(f"Błąd wykresu historii: {e}")

        st.divider()
        
        if 'ai_hr' in df_plot_resampled.columns:
            st.subheader("Analiza: Rzeczywistość vs AI")
            fig_ai_comp = go.Figure()
            fig_ai_comp.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['heartrate_smooth'], 
                                         name='Rzeczywiste HR', line=dict(color='#ef553b', width=2)))
            fig_ai_comp.add_trace(go.Scatter(x=df_plot_resampled['time_min'], y=df_plot_resampled['ai_hr'], 
                                         name='AI Model HR (Oczekiwane)', line=dict(color='#00cc96', dash='dot', width=2)))
            
            fig_ai_comp.update_layout(template="plotly_dark", title="Czy serce reagowało zgodnie z planem?", hovermode="x unified")
            st.plotly_chart(fig_ai_comp, use_container_width=True)
            
            diff = df_plot_resampled['heartrate_smooth'] - df_plot_resampled['ai_hr']
            avg_diff = diff.mean()
            
            if avg_diff > 3:
                st.warning(f"⚠️ **Wysoki Dryf Dnia (+{avg_diff:.1f} bpm):** Twoje tętno było wyższe niż model oczekiwał dla tej mocy. Możliwe zmęczenie, choroba lub upał.")
            elif avg_diff < -3:
                st.success(f"✅ **Dzień Konia ({avg_diff:.1f} bpm):** Tętno niższe niż zazwyczaj. Świetna dyspozycja!")
            else:
                st.info(f"🆗 **Norma ({avg_diff:.1f} bpm):** Reakcja serca zgodna z Twoim profilem historycznym.")

    else:
        st.warning("⚠️ Moduł AI wymaga procesora Apple Silicon i biblioteki `mlx`. Zainstaluj: `pip install mlx`")
