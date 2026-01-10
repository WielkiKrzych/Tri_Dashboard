import streamlit as st
import plotly.graph_objects as go
import pandas as pd
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
    
    # Manual override for ramp start power
    with st.expander("⚙️ Ustawienia detekcji progów", expanded=False):
        st.markdown("""
        **Manualne ustawienie początku Ramp Testu**
        
        Jeśli automatyczna detekcja rozgrzewki nie działa poprawnie, ustaw minimalną moc 
        od której algorytm zacznie szukać progów wentylacyjnych.
        """)
        min_power_input = st.number_input(
            "Minimalna moc do analizy [W]",
            min_value=0,
            max_value=500,
            value=st.session_state.get('vt_min_power_watts', 0),
            step=10,
            help="Ustaw 0 dla automatycznej detekcji. Ustaw np. 200W aby pominąć rozgrzewkę poniżej 200W."
        )
        st.session_state['vt_min_power_watts'] = min_power_input
        
        if min_power_input > 0:
            st.info(f"📌 Analiza progów rozpocznie się od mocy ≥ {min_power_input}W")
        else:
            st.caption("ℹ️ Automatyczna detekcja początku Ramp Testu")

    # 2. DETEKCJA AUTOMATYCZNA - CPET METHOD
    min_power_watts = st.session_state.get('vt_min_power_watts', 0) or None
    
    with st.spinner("Analizowanie progów wentylacyjnych (CPET)..."):
        # Use CPET-grade detection as primary method
        cpet_result = detect_vt_vslope_savgol(
            target_df, 
            step_range=None,  # Auto-detect steps
            power_column='watts',
            ve_column='tymeventilation',
            time_column='time',
            min_power_watts=min_power_watts
        )
        
        # Store CPET result for later use
        st.session_state['cpet_vt_result'] = cpet_result

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
    
    # =========================================================================
    # 🤖 WYKRYTE PROGI WENTYLACYJNE (FROM CPET METHOD)
    # =========================================================================
    st.subheader("🤖 Wykryte Progi Wentylacyjne (CPET)")
    
    # Get CPET result
    cpet_result = st.session_state.get('cpet_vt_result', {})
    has_gas = cpet_result.get('has_gas_exchange', False)
    method = cpet_result.get('method', 've_only_gradient')
    
    # Method badge
    if has_gas:
        st.success("✅ **Tryb CPET**: Analiza VE/VO2 i VE/VCO2")
    else:
        st.info("ℹ️ **Tryb VE-only**: Brak VO2/VCO2 - analiza gradientowa")
    
    col_z1, col_z2 = st.columns(2)
    
    # --- VT1 CARD (from CPET) ---
    with col_z1:
        vt1_watts = cpet_result.get('vt1_watts')
        vt1_hr = cpet_result.get('vt1_hr')
        vt1_ve = cpet_result.get('vt1_ve')
        vt1_br = cpet_result.get('vt1_br')
        vt1_pct_vo2max = cpet_result.get('vt1_pct_vo2max')
        
        # Fallback: Get BR from target_df if not in cpet_result
        if vt1_br is None and vt1_watts and 'tymebreathrate' in target_df.columns:
            # Find rows around VT1 power
            vt1_mask = (target_df['watts'] >= vt1_watts - 10) & (target_df['watts'] <= vt1_watts + 10)
            if vt1_mask.any():
                vt1_br = target_df.loc[vt1_mask, 'tymebreathrate'].mean()
                if pd.notna(vt1_br):
                    vt1_br = int(vt1_br)
        
        if vt1_watts:
            hr_line = f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt1_hr)} bpm</p>' if vt1_hr else ''
            ve_line = f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt1_ve} L/min</p>' if vt1_ve else ''
            br_line = f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt1_br)} oddech/min</p>' if vt1_br else ''
            vo2_line = f'<p style="margin:0; color:#aaa;"><b>%VO2max:</b> {vt1_pct_vo2max:.0f}%</p>' if vt1_pct_vo2max else ''
            
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ffa15a; background-color: #222;">
                <h3 style="margin:0; color: #ffa15a;">VT1 (Próg Tlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt1_watts)} W</h1>
                {hr_line}
                {ve_line}
                {br_line}
                {vo2_line}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt1_pct = (vt1_watts / cp_input) * 100
                st.caption(f"~{vt1_pct:.0f}% CP")
                
            # Apply button
            if st.button("✅ Aplikuj VT1", key="apply_vt1_main"):
                st.session_state['manual_vt1_watts'] = vt1_watts
                if vt1_ve: st.session_state['vt1_ve'] = vt1_ve
                if vt1_hr: st.session_state['vt1_hr'] = vt1_hr
                if vt1_br: st.session_state['vt1_br'] = vt1_br
                st.success(f"Ustawiono VT1 = {vt1_watts}W")
                st.rerun()
        else:
            st.warning("VT1: Nie wykryto automatycznie")

    # --- VT2 CARD (from CPET) ---
    with col_z2:
        vt2_watts = cpet_result.get('vt2_watts')
        vt2_hr = cpet_result.get('vt2_hr')
        vt2_ve = cpet_result.get('vt2_ve')
        vt2_br = cpet_result.get('vt2_br')
        vt2_pct_vo2max = cpet_result.get('vt2_pct_vo2max')
        
        # Fallback: Get BR from target_df if not in cpet_result
        if vt2_br is None and vt2_watts and 'tymebreathrate' in target_df.columns:
            vt2_mask = (target_df['watts'] >= vt2_watts - 10) & (target_df['watts'] <= vt2_watts + 10)
            if vt2_mask.any():
                vt2_br = target_df.loc[vt2_mask, 'tymebreathrate'].mean()
                if pd.notna(vt2_br):
                    vt2_br = int(vt2_br)
        
        if vt2_watts:
            hr_line = f'<p style="margin:0; color:#aaa;"><b>HR:</b> {int(vt2_hr)} bpm</p>' if vt2_hr else ''
            ve_line = f'<p style="margin:0; color:#aaa;"><b>VE:</b> {vt2_ve} L/min</p>' if vt2_ve else ''
            br_line = f'<p style="margin:0; color:#aaa;"><b>BR:</b> {int(vt2_br)} oddech/min</p>' if vt2_br else ''
            vo2_line = f'<p style="margin:0; color:#aaa;"><b>%VO2max:</b> {vt2_pct_vo2max:.0f}%</p>' if vt2_pct_vo2max else ''
            
            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; border:2px solid #ef553b; background-color: #222;">
                <h3 style="margin:0; color: #ef553b;">VT2 (Próg Beztlenowy)</h3>
                <h1 style="margin:5px 0; font-size:2.5em;">{int(vt2_watts)} W</h1>
                {hr_line}
                {ve_line}
                {br_line}
                {vo2_line}
            </div>
            """, unsafe_allow_html=True)
            
            # % of CP
            if cp_input > 0:
                vt2_pct = (vt2_watts / cp_input) * 100
                st.caption(f"~{vt2_pct:.0f}% CP")
                
            # Apply button
            if st.button("✅ Aplikuj VT2", key="apply_vt2_main"):
                st.session_state['manual_vt2_watts'] = vt2_watts
                if vt2_ve: st.session_state['vt2_ve'] = vt2_ve
                if vt2_hr: st.session_state['vt2_hr'] = vt2_hr
                if vt2_br: st.session_state['vt2_br'] = vt2_br
                st.success(f"Ustawiono VT2 = {vt2_watts}W")
                st.rerun()
        else:
            st.warning("VT2: Nie wykryto automatycznie")
    
    # Analysis notes
    analysis_notes = cpet_result.get('analysis_notes', [])
    if analysis_notes:
        with st.expander("📋 Notatki z analizy CPET", expanded=False):
            for note in analysis_notes:
                if note.startswith("⚠️"):
                    st.warning(note)
                else:
                    st.info(note)


    # =========================================================================
    # 🔬 SZCZEGÓŁY CPET (CHARTS)
    # =========================================================================
    st.markdown("---")
    with st.expander("📊 Wykresy CPET", expanded=True):
        df_s = cpet_result.get('df_steps')
        v1_w = cpet_result.get('vt1_watts')
        v2_w = cpet_result.get('vt2_watts')
        
        if df_s is not None and len(df_s) > 0:
            import matplotlib.pyplot as plt
            
            # Check if we have gas exchange data
            has_ve_vo2 = 've_vo2' in df_s.columns and df_s['ve_vo2'].notna().any()
            has_ve_vco2 = 've_vco2' in df_s.columns and df_s['ve_vco2'].notna().any()
            
            if has_ve_vo2 or has_ve_vco2:
                st.markdown("### Wykresy Ekwiwalentów Wentylacyjnych")
                
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
                    
                    if v1_w:
                        ax1.axvline(v1_w, color='#ffa15a', linestyle='--', linewidth=2, label=f'VT1: {v1_w}W')
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
                    
                    if v1_w:
                        ax2.axvline(v1_w, color='#ffa15a', linestyle=':', linewidth=1.5, alpha=0.7, label=f'VT1: {v1_w}W')
                    if v2_w:
                        ax2.axvline(v2_w, color='#ef553b', linestyle='--', linewidth=2, label=f'VT2: {v2_w}W')
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
                    st.markdown("### RER (Respiratory Exchange Ratio)")
                    fig_rer, ax_rer = plt.subplots(figsize=(10, 4))
                    ax_rer.set_facecolor('#0E1117')
                    fig_rer.patch.set_facecolor('#0E1117')
                    
                    ax_rer.plot(df_s['power'], df_s['rer'], 'g-o', linewidth=2, markersize=6, label='RER')
                    ax_rer.axhline(1.0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='RER = 1.0')
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
                st.markdown("### Wykres VE vs Power")
                
                fig, ax1 = plt.subplots(figsize=(10, 5))
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#0E1117')
                ax1.set_facecolor('#0E1117')
                
                if 've' in df_s.columns:
                    ax1.plot(df_s['power'], df_s['ve'], 'b-o', linewidth=2, label='VE (L/min)')
                elif 've_smooth' in df_s.columns:
                    ax1.plot(df_s['power'], df_s['ve_smooth'], 'b-o', linewidth=2, label='VE (L/min)')
                
                ax1.set_xlabel('Moc [W]', color='white')
                ax1.set_ylabel('Wentylacja [L/min]', color='#5da5da')
                
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
                    {"Strefa": "Z2 (Endurance)", "Zakres": f"{v1_w} - {int((v1_w + v2_w) / 2)} W", "Opis": "Baza tlenowa", "Metabolizm": "Dominująco tlenowy"},
                    {"Strefa": "Z3 (Tempo)", "Zakres": f"{int((v1_w + v2_w) / 2)} - {v2_w} W", "Opis": "Sweet Spot", "Metabolizm": "Mieszany"},
                    {"Strefa": "Z4 (Threshold)", "Zakres": f"{v2_w} - {int(v2_w * 1.05)} W", "Opis": "FTP, MLSS", "Metabolizm": "Glikolityczny"},
                    {"Strefa": "Z5 (VO2max)", "Zakres": f"> {int(v2_w * 1.05)} W", "Opis": "Interwały", "Metabolizm": "Anaerobowy"}
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
        else:
            st.warning("Brak danych schodków do wyświetlenia wykresów")


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

    # VT Markers from CPET result
    cpet_result = st.session_state.get('cpet_vt_result', {})
    vt1_watts = cpet_result.get('vt1_watts')
    vt2_watts = cpet_result.get('vt2_watts')
    vt1_hr = cpet_result.get('vt1_hr')
    vt2_hr = cpet_result.get('vt2_hr')
    
    # Find times at VT1/VT2 power levels
    if vt1_watts and 'watts' in target_df.columns:
        # Find first time when power crosses VT1
        vt1_mask = target_df['watts'] >= vt1_watts
        if vt1_mask.any():
            vt1_time = target_df.loc[vt1_mask, 'time'].iloc[0]
            hr_str = f"{int(vt1_hr)}" if vt1_hr else "--"
            
            fig_thresh.add_vline(
                x=vt1_time,
                line=dict(color="#ffa15a", width=3, dash="dash"),
                layer="above"
            )
            fig_thresh.add_annotation(
                x=vt1_time, y=1, yref="paper",
                text=f"<b>VT1</b><br>{int(vt1_watts)}W @ {hr_str} bpm",
                showarrow=False, font=dict(color="white", size=11),
                bgcolor="rgba(255, 161, 90, 0.8)", bordercolor="#ffa15a",
                borderwidth=2, borderpad=4, align="center",
                xanchor="center", yanchor="top"
            )
    
    if vt2_watts and 'watts' in target_df.columns:
        vt2_mask = target_df['watts'] >= vt2_watts
        if vt2_mask.any():
            vt2_time = target_df.loc[vt2_mask, 'time'].iloc[0]
            hr_str = f"{int(vt2_hr)}" if vt2_hr else "--"
            
            fig_thresh.add_vline(
                x=vt2_time,
                line=dict(color="#ef553b", width=3, dash="dash"),
                layer="above"
            )
            fig_thresh.add_annotation(
                x=vt2_time, y=1, yref="paper",
                text=f"<b>VT2</b><br>{int(vt2_watts)}W @ {hr_str} bpm",
                showarrow=False, font=dict(color="white", size=11),
                bgcolor="rgba(239, 85, 59, 0.8)", bordercolor="#ef553b",
                borderwidth=2, borderpad=4, align="center",
                xanchor="center", yanchor="bottom", yshift=-40
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
