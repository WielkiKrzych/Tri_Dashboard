import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modules.calculations.thresholds import analyze_step_test
from modules.calculations.ventilatory import detect_vt_vslope_savgol
from modules.calculations.quality import check_step_test_protocol
from modules.calculations.pipeline import run_ramp_test_pipeline
from modules.reporting.persistence import save_ramp_test_report
from modules.manual_overrides import get_manual_overrides, to_dict
from models.results import ValidityLevel

def render_vent_thresholds_tab(target_df, training_notes, uploaded_file_name, cp_input, w_prime_input=20000, rider_weight=75.0, max_hr=190.0):
    st.header("🎯 Detekcja Progów Wentylacyjnych (VT1 / VT2)")
    st.markdown("Automatyczna detekcja progów wentylacyjnych. **Wymaga testu stopniowanego (Ramp Test).**")

    # 1. Przygotowanie danych
    if target_df is None or target_df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        st.stop()

    if 'time' not in target_df.columns or 'tymeventilation' not in target_df.columns:
        st.error("Brak danych wentylacji (tymeventilation) lub czasu!")
        st.stop()

    # Wygładzanie
    # Normalize columns first
    target_df.columns = target_df.columns.str.lower().str.strip()
    
    # Handle HR aliases
    if 'hr' not in target_df.columns:
        for alias in ['heart_rate', 'heart rate', 'bpm', 'tętno', 'heartrate', 'heart_rate_bpm']:
            if alias in target_df.columns:
                target_df.rename(columns={alias: 'hr'}, inplace=True)
                break

    if 'watts_smooth_5s' not in target_df.columns and 'watts' in target_df.columns:
        target_df['watts_smooth_5s'] = target_df['watts'].rolling(window=5, center=True).mean()
    target_df['ve_smooth'] = target_df['tymeventilation'].rolling(window=10, center=True).mean()
    target_df['time_str'] = pd.to_datetime(target_df['time'], unit='s').dt.strftime('%H:%M:%S')

    # --- Quality Check: Protocol Compliance ---
    st.subheader("📋 Weryfikacja Protokołu")
    
    proto_check = check_step_test_protocol(target_df)
    
    if not proto_check['is_valid']:
        st.error("⚠️ **Wykryto Problemy z Protokołem Testu**")
        for issue in proto_check['issues']:
            st.warning(issue)
        st.markdown("""
        **Dlaczego to ważne?**
        
        Detekcja progów wentylacyjnych wymaga **testu stopniowanego (Ramp Test)** z liniowym wzrostem obciążenia.
        Dla normalnych treningów użyj zakładki **"🫁 Ventilation"** do analizy manualnej.
        """)
        
        if not st.checkbox("⚠️ Wymuś analizę mimo błędów protokołu (wyniki mogą być niewiarygodne)"):
            st.stop()
    else:
        st.success("✅ Protokół Testu Stopniowanego: Poprawny (Liniowy Wzrost Obciążenia)")

    st.markdown("---")

    # 2. DETEKCJA AUTOMATYCZNA
    with st.spinner("Analizowanie progów wentylacyjnych..."):
        result = analyze_step_test(
            target_df, 
            power_column='watts',
            ve_column='tymeventilation',
            hr_column='hr' if 'hr' in target_df.columns else None,
            time_column='time'
        )

        # 3. NOWA METODOLOGIA: Uruchom pełny pipeline (BEZ automatycznego zapisu)
        # Pipeline result jest przechowywany w session_state - zapis tylko po kliknięciu przycisku
        try:
            pipeline_result = run_ramp_test_pipeline(
                target_df,
                power_column='watts',
                ve_column='tymeventilation',
                hr_column='hr' if 'hr' in target_df.columns else None,
                smo2_column='smo2' if 'smo2' in target_df.columns else None,
                time_column='time',
                test_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                protocol="Ramp Test",
                cp_watts=float(cp_input),
                w_prime_joules=float(w_prime_input),
                smo2_manual_lt1=st.session_state.get('smo2_lt1_m'),
                smo2_manual_lt2=st.session_state.get('smo2_lt2_m'),
                rider_weight=float(rider_weight),
                max_hr=float(max_hr)
            )
            
            # Store pipeline result for manual report generation
            st.session_state['pending_pipeline_result'] = pipeline_result
            st.session_state['pending_source_df'] = target_df
            st.session_state['pending_uploaded_file_name'] = uploaded_file_name
            st.session_state['pending_cp_input'] = cp_input
            
            # Show validity status
            if pipeline_result.validity.validity in [ValidityLevel.VALID, ValidityLevel.CONDITIONAL]:
                st.success(f"✅ Test poprawny - gotowy do wygenerowania raportu")
            else:
                st.warning(f"⚠️ Test niepoprawny - raport może być niewiarygodny")
                
        except Exception as e:
            st.error(f"Błąd analizy pipeline: {e}")
            print(f"Pipeline failed: {e}")
    
    # ========== GENERUJ RAPORT BUTTON ==========
    st.markdown("---")
    st.subheader("📄 Generowanie Raportu")
    st.info("💡 Raport NIE generuje się automatycznie. Kliknij przycisk poniżej, aby wygenerować i zapisać raport.")
    
    # Check if we have pending pipeline result
    pending_result = st.session_state.get('pending_pipeline_result')
    
    if pending_result is not None:
        # ========== VALIDATION: Check for manual overrides ==========
        manual_overrides = to_dict(get_manual_overrides())
        
        # Count how many manual values are set
        manual_keys = ["manual_vt1_watts", "manual_vt2_watts", "smo2_lt1_m", "smo2_lt2_m", "cp_input"]
        manual_values_set = sum(1 for k in manual_keys if manual_overrides.get(k) and manual_overrides[k] > 0)
        
        # Show warning if no manual values
        can_generate = True
        if manual_values_set == 0:
            st.warning("⚠️ **Brak wartości manualnych!** Raport zostanie wygenerowany wyłącznie z wartościami automatycznymi (algorytm).")
            bypass_warning = st.checkbox(
                "✅ Generuj raport mimo braku wartości manualnych",
                key="bypass_manual_warning",
                help="Zaznacz, aby wygenerować raport z wartościami automatycznymi. Wartości manualne mają wyższy poziom zaufania."
            )
            if not bypass_warning:
                can_generate = False
                st.caption("Aby dodać wartości manualne, przejdź do zakładki **Vent - Progi Manuals** lub **SmO2 - Progi Manuals**.")
        else:
            st.success(f"✅ Wykryto {manual_values_set} wartości manualnych - zostaną użyte w raporcie")
        
        # Show button only if validation passed
        if can_generate and st.button("📄 GENERUJ RAPORT", type="primary", use_container_width=True):
            st.session_state['report_generation_requested'] = True
            with st.spinner("Generowanie raportu..."):
                try:
                    # Get data from session state
                    source_df = st.session_state.get('pending_source_df')
                    file_name = st.session_state.get('pending_uploaded_file_name', 'unknown')
                    cp = st.session_state.get('pending_cp_input', 0)
                    
                    # Get session type and confidence
                    session_type = st.session_state.get('session_type')
                    ramp_classification = st.session_state.get('ramp_classification')
                    ramp_confidence = ramp_classification.confidence if ramp_classification else 1.0
                    
                    # Get manual overrides from SINGLE SOURCE OF TRUTH
                    manual_overrides = to_dict(get_manual_overrides())
                    
                    # Call save with explicit user trigger
                    save_result = save_ramp_test_report(
                        pending_result,
                        notes=f"User-triggered save from UI. File: {file_name}",
                        session_type=session_type,
                        ramp_confidence=ramp_confidence,
                        source_file=file_name,
                        source_df=source_df,
                        manual_overrides=manual_overrides
                    )
                    
                    session_id = save_result.get('session_id', 'unknown')
                    saved_path = save_result.get('path', '')
                    pdf_path = save_result.get('pdf_path', '')
                    
                    if save_result.get('gated'):
                        reason = save_result.get('reason', 'unknown')
                        st.error(f"❌ Raport NIE zapisany: {reason}")
                    elif save_result.get('deduplicated'):
                        st.warning(f"⚠️ Raport dla tego pliku już istnieje")
                    elif saved_path:
                        st.success(f"✅ Raport wygenerowany pomyślnie!")
                        st.info(f"📁 JSON: `{saved_path}`")
                        if pdf_path:
                            st.info(f"📄 PDF: `{pdf_path}`")
                        st.balloons()
                        
                        # Clear pending state
                        st.session_state.pop('pending_pipeline_result', None)
                    else:
                        st.error(f"❌ Nieznany błąd zapisu")
                        
                except Exception as e:
                    st.error(f"❌ Błąd generowania raportu: {e}")
                    print(f"Report generation failed: {e}")
    else:
        st.warning("⚠️ Brak danych do wygenerowania raportu. Najpierw wgraj plik i poczekaj na analizę.")
    
    st.markdown("---")
    vt1_zone = result.vt1_zone
    vt2_zone = result.vt2_zone
    hysteresis = result.hysteresis
    sensitivity = result.sensitivity
    
    # Display step detection info
    if result.analysis_notes:
        with st.expander("📋 Szczegóły Analizy", expanded=True):
            for note in result.analysis_notes:
                if note.startswith("✅"):
                    st.success(note)
                elif note.startswith("⚠️"):
                    st.warning(note)
                elif note.startswith("  •"):
                    st.caption(note)
                else:
                    st.info(note)
            
            if result.steps_analyzed > 0:
                st.metric("Liczba wykrytych stopni", result.steps_analyzed)
            
            # Show VE slope table for each step
            if result.step_ve_analysis:
                st.markdown("### 📊 VE Slope per step (debug)")
                step_df = pd.DataFrame(result.step_ve_analysis)
                
                # Format columns
                if 've_slope' in step_df.columns:
                    step_df['ve_slope'] = step_df['ve_slope'].apply(lambda x: f"{x:.4f}")
                # Safe formatting for numeric columns
                if 'avg_power' in step_df.columns:
                    step_df['avg_power'] = pd.to_numeric(step_df['avg_power'], errors='coerce').apply(lambda x: f"{x:.0f}W" if pd.notna(x) else "-")
                if 'avg_hr' in step_df.columns:
                    step_df['avg_hr'] = pd.to_numeric(step_df['avg_hr'], errors='coerce').apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
                if 'avg_ve' in step_df.columns:
                    step_df['avg_ve'] = pd.to_numeric(step_df['avg_ve'], errors='coerce').apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                if 'avg_br' in step_df.columns:
                    step_df['avg_br'] = pd.to_numeric(step_df['avg_br'], errors='coerce').apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
                if 've_slope' in step_df.columns:
                    step_df['ve_slope'] = pd.to_numeric(step_df['ve_slope'], errors='coerce').apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                
                # Add VT markers
                if 'is_vt1' in step_df.columns:
                    step_df['VT'] = step_df.apply(
                        lambda r: '⚪ Skipped' if r.get('is_skipped') else ('🟠 VT1' if r.get('is_vt1') else ('🔴 VT2' if r.get('is_vt2') else '')), 
                        axis=1
                    )
                
                # Rename columns for display
                step_df = step_df.rename(columns={
                    'step_number': 'Step',
                    'avg_power': 'Power',
                    'avg_hr': 'HR',
                    'avg_ve': 'VE',
                    'avg_br': 'BR',
                    've_slope': 'Trend Slope'
                })
                
                # Select columns to show
                cols_to_show = ['Step', 'Power', 'HR', 'VE', 'BR', 'Trend Slope', 'VT']
                cols_available = [c for c in cols_to_show if c in step_df.columns]
                
                st.dataframe(step_df[cols_available], hide_index=True)

    # Wyświetlenie wyników automatycznych
    st.subheader("🤖 Wykryte Progi Wentylacyjne")
    
    col_z1, col_z2 = st.columns(2)
    
    # --- VT1 CARD ---
    with col_z1:
        if result.vt1_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(result.vt1_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(result.vt1_hr)} bpm</p>' if result.vt1_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {result.vt1_ve} L/min</p>' if result.vt1_ve else ''}
                {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(result.vt1_br)} BR/min</p>' if result.vt1_br else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt1_pct = (result.vt1_watts / cp_input) * 100
                st.caption(f"~{vt1_pct:.0f}% CP")
        else:
            st.info("VT1: Nie wykryto (brak slope >= 0.05)")

    # --- VT2 CARD ---
    with col_z2:
        if result.vt2_watts:
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(result.vt2_watts)} W</h1>
                {f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(result.vt2_hr)} bpm</p>' if result.vt2_hr else ''}
                {f'<p style="margin:0; color:#aaa;"><b>VE:</b> {result.vt2_ve} L/min</p>' if result.vt2_ve else ''}
                 {f'<p style="margin:0; color:#aaa;"><b>Oddechy:</b> {int(result.vt2_br)} BR/min</p>' if result.vt2_br else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt2_pct = (result.vt2_watts / cp_input) * 100
                st.caption(f"~{vt2_pct:.0f}% CP")
        else:
            st.info("VT2: Nie wykryto (brak slope >= 0.05 po VT1)")

    # =========================================================================
    # 🔬 METODA V-SLOPE (ADVANCED DIAGNOSTICS)
    # =========================================================================
    st.markdown("---")
    with st.expander("🔬 Metoda CPET (Scientific Diagnostics)", expanded=True):
        st.markdown("### Zaawansowana Analiza Ekwiwalentów Wentylacyjnych")
        st.info("""
        **CPET-Grade Algorithm** wykorzystuje:
        - **VE/VO2** (ekwiwalent tlenowy) - dla detekcji VT1
        - **VE/VCO2** (ekwiwalent węglowy) - dla detekcji VT2
        - **Segmented Regression** (breakpoint detection)
        - **RER ≈ 1.0** walidacja dla VT2
        """)
        
        with st.spinner("Przeliczanie CPET breakpoints..."):
            vslope_res = detect_vt_vslope_savgol(target_df, result.step_range, 'watts', 'tymeventilation', 'time')
            
        if 'error' in vslope_res:
            st.error(f"Błąd analizy CPET: {vslope_res['error']}")
        else:
            v1_w = vslope_res.get('vt1_watts')
            v2_w = vslope_res.get('vt2_watts')
            df_s = vslope_res.get('df_steps')
            has_gas = vslope_res.get('has_gas_exchange', False)
            method = vslope_res.get('method', 'unknown')
            notes = vslope_res.get('analysis_notes', [])
            validation = vslope_res.get('validation', {})
            
            # Method badge
            if has_gas:
                st.success("✅ **Tryb CPET**: Wykryto dane VO2/VCO2 - pełna analiza ekwiwalentów wentylacyjnych")
            else:
                st.warning("⚠️ **Tryb VE-only**: Brak danych VO2/VCO2 - uproszczona analiza gradientowa")
            
            # Analysis notes
            if notes:
                with st.expander("📋 Notatki z analizy", expanded=False):
                    for note in notes:
                        if note.startswith("⚠️"):
                            st.warning(note)
                        else:
                            st.info(note)
            
            # Results cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("VT1 (CPET)", f"{v1_w} W" if v1_w else "—", 
                          help="Próg tlenowy - punkt gdzie VE/VO2 zaczyna rosnąć")
                if vslope_res.get('vt1_pct_vo2max'):
                    st.caption(f"@ {vslope_res['vt1_pct_vo2max']:.0f}% VO2max")
                if st.button("Aplikuj VT1", key="apply_vt1_cpet"):
                    st.session_state['manual_vt1_watts'] = v1_w
                    if vslope_res.get('vt1_ve'): st.session_state['vt1_ve'] = vslope_res['vt1_ve']
                    if vslope_res.get('vt1_hr'): st.session_state['vt1_hr'] = vslope_res['vt1_hr']
                    st.success(f"Ustawiono VT1 = {v1_w}W")
                    st.rerun()
                    
            with c2:
                st.metric("VT2 (CPET)", f"{v2_w} W" if v2_w else "—",
                          help="Próg beztlenowy - punkt gdzie VE/VCO2 zaczyna rosnąć, RER ≈ 1.0")
                if vslope_res.get('vt2_pct_vo2max'):
                    st.caption(f"@ {vslope_res['vt2_pct_vo2max']:.0f}% VO2max")
                if st.button("Aplikuj VT2", key="apply_vt2_cpet"):
                    st.session_state['manual_vt2_watts'] = v2_w
                    if vslope_res.get('vt2_ve'): st.session_state['vt2_ve'] = vslope_res['vt2_ve']
                    if vslope_res.get('vt2_hr'): st.session_state['vt2_hr'] = vslope_res['vt2_hr']
                    st.success(f"Ustawiono VT2 = {v2_w}W")
                    st.rerun()
                    
            with c3:
                # Validation status
                if validation.get('vt1_lt_vt2'):
                    st.success("✅ VT1 < VT2")
                else:
                    st.error("❌ VT1 >= VT2")
                if validation.get('ve_vo2_rises_first'):
                    st.success("✅ VE/VO2 rośnie przed VE/VCO2")
            
            # =========================================================================
            # CPET CHARTS (VE/VO2 vs Power, VE/VCO2 vs Power)
            # =========================================================================
            if df_s is not None and len(df_s) > 0:
                import matplotlib.pyplot as plt
                
                # Check if we have gas exchange data
                has_ve_vo2 = 've_vo2' in df_s.columns and df_s['ve_vo2'].notna().any()
                has_ve_vco2 = 've_vco2' in df_s.columns and df_s['ve_vco2'].notna().any()
                
                if has_ve_vo2 or has_ve_vco2:
                    st.markdown("### 📊 Wykresy Ekwiwalentów Wentylacyjnych")
                    
                    # Create 2-panel figure
                    fig, axes = plt.subplots(1, 2 if (has_ve_vo2 and has_ve_vco2) else 1, figsize=(14, 5))
                    plt.style.use('dark_background')
                    fig.patch.set_facecolor('#0E1117')
                    
                    if not isinstance(axes, (list, tuple)):
                        axes = [axes]
                    ax_idx = 0
                    
                    # Panel 1: VE/VO2 vs Power (VT1 detection)
                    if has_ve_vo2:
                        ax1 = axes[ax_idx]
                        ax1.set_facecolor('#0E1117')
                        ax1.plot(df_s['power'], df_s['ve_vo2'], 'b-o', linewidth=2, markersize=6, label='VE/VO2')
                        ax1.set_xlabel('Moc [W]', color='white')
                        ax1.set_ylabel('VE/VO2', color='#5da5da')
                        ax1.set_title('VE/VO2 vs Power (VT1 Detection)', color='white', pad=10)
                        ax1.grid(True, alpha=0.2)
                        
                        # Mark VT1
                        if v1_w:
                            ax1.axvline(v1_w, color='#ffa15a', linestyle='--', linewidth=2, label=f'VT1: {v1_w}W')
                            # Find y value at VT1
                            vt1_row = df_s[df_s['power'] == v1_w]
                            if len(vt1_row) > 0:
                                y_vt1 = vt1_row['ve_vo2'].iloc[0]
                                ax1.scatter([v1_w], [y_vt1], color='#ffa15a', s=150, zorder=5, marker='*')
                        
                        ax1.legend(loc='upper left')
                        ax_idx += 1
                    
                    # Panel 2: VE/VCO2 vs Power (VT2 detection)
                    if has_ve_vco2:
                        ax2 = axes[ax_idx] if ax_idx < len(axes) else axes[0]
                        ax2.set_facecolor('#0E1117')
                        ax2.plot(df_s['power'], df_s['ve_vco2'], 'r-o', linewidth=2, markersize=6, label='VE/VCO2')
                        ax2.set_xlabel('Moc [W]', color='white')
                        ax2.set_ylabel('VE/VCO2', color='#ef553b')
                        ax2.set_title('VE/VCO2 vs Power (VT2 Detection)', color='white', pad=10)
                        ax2.grid(True, alpha=0.2)
                        
                        # Mark VT1 and VT2
                        if v1_w:
                            ax2.axvline(v1_w, color='#ffa15a', linestyle=':', linewidth=1.5, alpha=0.7, label=f'VT1: {v1_w}W')
                        if v2_w:
                            ax2.axvline(v2_w, color='#ef553b', linestyle='--', linewidth=2, label=f'VT2: {v2_w}W')
                            # Find y value at VT2
                            vt2_row = df_s[df_s['power'] == v2_w]
                            if len(vt2_row) > 0:
                                y_vt2 = vt2_row['ve_vco2'].iloc[0]
                                ax2.scatter([v2_w], [y_vt2], color='#ef553b', s=150, zorder=5, marker='*')
                        
                        ax2.legend(loc='upper left')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # RER chart if available
                    if 'rer' in df_s.columns and df_s['rer'].notna().any():
                        st.markdown("### 📈 RER (Respiratory Exchange Ratio)")
                        fig_rer, ax_rer = plt.subplots(figsize=(10, 4))
                        ax_rer.set_facecolor('#0E1117')
                        fig_rer.patch.set_facecolor('#0E1117')
                        
                        ax_rer.plot(df_s['power'], df_s['rer'], 'g-o', linewidth=2, markersize=6, label='RER')
                        ax_rer.axhline(1.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='RER = 1.0 (VT2 marker)')
                        ax_rer.set_xlabel('Moc [W]', color='white')
                        ax_rer.set_ylabel('RER (VCO2/VO2)', color='#60bd68')
                        ax_rer.set_title('Respiratory Exchange Ratio vs Power', color='white', pad=10)
                        ax_rer.grid(True, alpha=0.2)
                        
                        if v2_w:
                            ax_rer.axvline(v2_w, color='#ef553b', linestyle='--', linewidth=2, label=f'VT2: {v2_w}W')
                        
                        ax_rer.legend(loc='upper left')
                        plt.tight_layout()
                        st.pyplot(fig_rer)
                        plt.close(fig_rer)
                
                else:
                    # VE-only mode chart
                    st.markdown("### 📊 Wykres VE vs Power (VE-only mode)")
                    
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    plt.style.use('dark_background')
                    fig.patch.set_facecolor('#0E1117')
                    ax1.set_facecolor('#0E1117')
                    
                    # Plot VE
                    if 've' in df_s.columns:
                        ax1.plot(df_s['power'], df_s['ve'], 'b-o', linewidth=2, label='VE (L/min)')
                    elif 've_smooth' in df_s.columns:
                        ax1.plot(df_s['power'], df_s['ve_smooth'], 'b-o', linewidth=2, label='VE (L/min)')
                    
                    ax1.set_xlabel('Moc [W]', color='white')
                    ax1.set_ylabel('Wentylacja [L/min]', color='#5da5da')
                    
                    # Plot slope on secondary axis
                    if 've_slope' in df_s.columns:
                        ax2 = ax1.twinx()
                        ax2.plot(df_s['power'], df_s['ve_slope'], 'g--', alpha=0.5, label='Slope')
                        ax2.set_ylabel('dVE/dP', color='#60bd68')
                    
                    # Mark thresholds
                    if v1_w:
                        ax1.axvline(v1_w, color='#ffa15a', linestyle='--', linewidth=2, label=f'VT1: {v1_w}W')
                    if v2_w:
                        ax1.axvline(v2_w, color='#ef553b', linestyle='--', linewidth=2, label=f'VT2: {v2_w}W')
                    
                    ax1.set_title('VE vs Power z Progami VT1/VT2', color='white', pad=10)
                    ax1.grid(True, alpha=0.2)
                    ax1.legend(loc='upper left')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                # =========================================================================
                # METABOLIC ZONES TABLE
                # =========================================================================
                st.markdown("### 🎯 Strefy Metaboliczne")
                
                if v1_w and v2_w:
                    zones_data = [
                        {"Strefa": "Z1 (Recovery)", "Zakres": f"< {v1_w} W", "Opis": "Regeneracja, rozgrzewka", "Metabolizm": "100% Tlenowy"},
                        {"Strefa": "Z2 (Endurance)", "Zakres": f"{v1_w} - {int((v1_w + v2_w) / 2)} W", "Opis": "Baza tlenowa, spalanie tłuszczu", "Metabolizm": "Dominująco tlenowy"},
                        {"Strefa": "Z3 (Tempo)", "Zakres": f"{int((v1_w + v2_w) / 2)} - {v2_w} W", "Opis": "Sweet Spot, próg", "Metabolizm": "Mieszany"},
                        {"Strefa": "Z4 (Threshold)", "Zakres": f"{v2_w} - {int(v2_w * 1.05)} W", "Opis": "FTP, MLSS", "Metabolizm": "Wysoki udziału glikolizy"},
                        {"Strefa": "Z5 (VO2max)", "Zakres": f"> {int(v2_w * 1.05)} W", "Opis": "Interwały, moc szczytowa", "Metabolizm": "Anaerobowy"}
                    ]
                    zones_df = pd.DataFrame(zones_data)
                    st.dataframe(zones_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("Brak danych do wygenerowania stref metabolicznych")
                
                # Raw data expander
                with st.expander("📝 Dane schodków (raw)", expanded=False):
                    display_cols = ['step', 'power', 've']
                    if 'vo2' in df_s.columns:
                        display_cols.append('vo2')
                    if 'vco2' in df_s.columns:
                        display_cols.append('vco2')
                    if 've_vo2' in df_s.columns:
                        display_cols.append('ve_vo2')
                    if 've_vco2' in df_s.columns:
                        display_cols.append('ve_vco2')
                    if 'rer' in df_s.columns:
                        display_cols.append('rer')
                    if 'hr' in df_s.columns:
                        display_cols.append('hr')
                    
                    available_cols = [c for c in display_cols if c in df_s.columns]
                    st.dataframe(df_s[available_cols].round(2), use_container_width=True)

    st.markdown("---")

    # Hysteresis Information
    if hysteresis and (hysteresis.vt1_dec_zone or hysteresis.vt2_dec_zone):
        with st.expander("📉 Analiza Histerezy (Ramp Down vs Up)", expanded=False):
            st.markdown("""
            **Histereza** to różnica między progami wykrytymi podczas wzrostu obciążenia (Ramp Up) 
            a progami wykrytymi podczas spadku (Ramp Down / Recovery).
            
            - **Duża histereza (>15W)**: Może wskazywać na zmęczenie lub niedobór tlenowy
            - **Mała histereza (<10W)**: Dobra reprodukowalność progów
            """)
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                if hysteresis.vt1_dec_zone:
                    st.markdown(f"**VT1 (Down):** {hysteresis.vt1_dec_zone.range_watts[0]:.0f} - {hysteresis.vt1_dec_zone.range_watts[1]:.0f} W")
                    if hysteresis.vt1_shift_watts is not None:
                         shift = hysteresis.vt1_shift_watts
                         color = "red" if shift < -15 else "green"
                         st.markdown(f"Shift: **:{color}[{shift:+.0f} W]**")
            with h_col2:
                 if hysteresis.vt2_dec_zone:
                    st.markdown(f"**VT2 (Down):** {hysteresis.vt2_dec_zone.range_watts[0]:.0f} - {hysteresis.vt2_dec_zone.range_watts[1]:.0f} W")
                    if hysteresis.vt2_shift_watts is not None:
                         shift = hysteresis.vt2_shift_watts
                         color = "red" if shift < -15 else "green"
                         st.markdown(f"Shift: **:{color}[{shift:+.0f} W]**")
            
            if hysteresis.warnings:
                for w in hysteresis.warnings:
                    st.warning(f"⚠️ {w}")

    # 3. Wykres z zaznaczonymi strefami
    st.subheader("📈 Wizualizacja Progów")
    
    fig_thresh = go.Figure()

    # VE (Primary)
    fig_thresh.add_trace(go.Scatter(
        x=target_df['time'], y=target_df['ve_smooth'],
        customdata=target_df['time_str'],
        mode='lines', name='VE (L/min)',
        line=dict(color='#ffa15a', width=2),
        hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>"
    ))

    # Power (Secondary)
    if 'watts_smooth_5s' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['watts_smooth_5s'],
            customdata=target_df['time_str'],
            mode='lines', name='Power',
            line=dict(color='#1f77b4', width=1),
            yaxis='y2', opacity=0.3,
            hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>"
        ))

    # HR Trace (Red, Dotted, Secondary Axis)
    if 'hr' in target_df.columns:
        fig_thresh.add_trace(go.Scatter(
            x=target_df['time'], y=target_df['hr'],
            customdata=target_df['time_str'],
            mode='lines', name='Heart Rate',
            line=dict(color='#d62728', width=1, dash='dot'),
            yaxis='y2', opacity=0.5,
            hovertemplate="<b>Czas:</b> %{customdata}<br><b>HR:</b> %{y:.0f} bpm<extra></extra>"
        ))

    # VT Markers based on identified steps
    if result.step_ve_analysis:
        for step in result.step_ve_analysis:
            # Check if this step is VT1 or VT2
            is_vt1 = step.get('is_vt1', False)
            is_vt2 = step.get('is_vt2', False)
            
            if is_vt1 or is_vt2:
                # Get step info
                power = step.get('avg_power', 0)
                if power is None: power = 0
                hr = step.get('avg_hr')
                # Use end of step for marker. If end_time is missing, fallback to 0 (should be fixed now)
                marker_time = step.get('end_time', 0)
                
                label = "VT1 (Próg Aerobowy)" if is_vt1 else "VT2 (Próg Beztlenowy)"
                line_color = "#ffa15a" if is_vt1 else "#ef553b" # Orange for VT1, Red for VT2
                bg_color = "rgba(255, 161, 90, 0.8)" if is_vt1 else "rgba(239, 85, 59, 0.8)"
                
                hr_str = f"{int(hr)}" if hr is not None else "--"
                
                # Add Vertical Line Marker
                fig_thresh.add_vline(
                    x=marker_time,
                    line=dict(color=line_color, width=3, dash="dash"),
                    layer="above"
                )
                
                # Add Annotation with Box for prominence
                fig_thresh.add_annotation(
                    x=marker_time,
                    y=1,
                    yref="paper",
                    text=f"<b>{label}</b><br>{int(power)}W @ {hr_str} bpm",
                    showarrow=False,
                    font=dict(color="white", size=11),
                    bgcolor=bg_color,
                    bordercolor=line_color,
                    borderwidth=2,
                    borderpad=4,
                    align="center",
                    xanchor="center",
                    yanchor="top" if is_vt1 else "bottom", # Offset slightly to avoid overlap if close
                    yshift=0 if is_vt1 else -40 # VT1 at top, VT2 slightly below or vice versa
                )
    
    # Hysteresis Zones (Dashed, if available from legacy detection)
    if hysteresis:
        if hysteresis.vt1_dec_zone:
            fig_thresh.add_hrect(
               y0=hysteresis.vt1_dec_zone.range_watts[0], y1=hysteresis.vt1_dec_zone.range_watts[1],
               fillcolor="blue", opacity=0.05,
               layer="below", line_width=1, line_dash="dot", line_color="blue",
               yref="y2",
               annotation_text="VT1 (Down)", annotation_position="bottom right"
            )

        if hysteresis.vt2_dec_zone:
            fig_thresh.add_hrect(
               y0=hysteresis.vt2_dec_zone.range_watts[0], y1=hysteresis.vt2_dec_zone.range_watts[1],
               fillcolor="purple", opacity=0.05,
               layer="below", line_width=1, line_dash="dot", line_color="purple",
               yref="y2",
               annotation_text="VT2 (Down)", annotation_position="bottom right"
            )

    fig_thresh.update_layout(
        title="Dynamika Wentylacji z Progami VT1/VT2",
        xaxis_title="Czas",
        yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
        yaxis2=dict(title=dict(text="Moc (W)", font=dict(color="#1f77b4")), overlaying='y', side='right', showgrid=False),
        legend=dict(x=0.01, y=0.99),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_thresh, use_container_width=True)

    # Sensitivity Analysis
    if sensitivity:
        with st.expander("📊 Analiza Wrażliwości (Sensitivity)", expanded=False):
            st.markdown("""
            **Sensitivity Analysis** sprawdza, jak stabilne są wykryte progi przy różnych parametrach algorytmu.
            Wysoka stabilność = wynik jest wiarygodny niezależnie od ustawień.
            """)
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.metric("VT1 Stability Score", f"{sensitivity.vt1_stability_score:.2f}", 
                          help="0-1, wyżej = lepiej")
                st.caption(f"Variability: ±{sensitivity.vt1_variability_watts:.1f} W")
            with s_col2:
                st.metric("VT2 Stability Score", f"{sensitivity.vt2_stability_score:.2f}",
                          help="0-1, wyżej = lepiej")
                st.caption(f"Variability: ±{sensitivity.vt2_variability_watts:.1f} W")
                
            if sensitivity.details:
                for w in sensitivity.details:
                    st.caption(f"ℹ️ {w}")

    # ===== TEORIA =====
    with st.expander("🫁 TEORIA: Progi Wentylacyjne (VT1 / VT2)", expanded=False):
        st.markdown("""
        ## Co to są progi wentylacyjne?
        
        **Progi wentylacyjne** to punkty, w których wentylacja (VE) zaczyna rosnąć nieliniowo względem mocy.
        
        | Próg | Inna nazwa | Fizjologia | % VO2max |
        |------|-----------|------------|----------|
        | **VT1** | Próg tlenowy, LT1 | Początek akumulacji mleczanu | ~50-60% |
        | **VT2** | Próg beztlenowy, LT2, OBLA | Maksymalny laktat steady-state | ~75-85% |
        
        ---
        
        ## Jak działa detekcja?
        
        System stosuje:
        1. **Sliding Window Analysis**: Skanuje okno po oknie, żeby znaleźć przejścia w nachyleniu (slope)
        2. **Breakpoint Detection**: Szuka punktów załamania krzywej VE vs Power
        3. **Sensitivity Analysis**: Uruchamia algorytm kilkukrotnie z różnymi parametrami
        
        ---
        
        ## Zastosowanie progów
        
        | Strefa | Zakres | Cel treningowy |
        |--------|--------|----------------|
        | **Z1 (Recovery)** | < VT1 | Regeneracja, rozgrzewka |
        | **Z2 (Endurance)** | VT1 - środek | Baza tlenowa |
        | **Z3 (Tempo)** | środek - VT2 | Sweet Spot |
        | **Z4 (Threshold)** | VT2 ± 5% | FTP, próg |
        | **Z5+ (VO2max)** | > VT2 | Interwały, moc szczytowa |
        
        ---
        
        ## Reliability Score (Niezawodność)
        
        * **HIGH**: Wynik jest stabilny niezależnie od wygładzania
        * **MEDIUM**: Wynik zależy nieco od parametrów
        * **LOW**: Duża zmienność (>15W różnicy) - sugeruje "szumiący" sygnał
        
        ---
        
        ## Wymagania testu
        
        ⚠️ **Dla wiarygodnych wyników potrzebujesz:**
        - Test stopniowany (Ramp Test) z liniowym wzrostem mocy
        - Minimum 10-15 minut narastającego obciążenia
        - Czysty sygnał wentylacji (stabilny sensor)
        - Brak przerw i wahań mocy
        """)
