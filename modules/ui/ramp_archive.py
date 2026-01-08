"""
Ramp Test Archive UI.

Displays a list of executed Ramp Tests from the CSV index.
Allows downloading reports in HTML format.
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

from modules.reporting.persistence import load_ramp_test_report
from modules.reporting.html_generator import generate_html_report

def render_ramp_archive():
    """Render the Ramp Test Archive view."""
    st.header("🗄️ Archiwum Raportów Ramp Test")
    
    # 1. Locate Index CSV
    base_dir = "reports/ramp_tests"
    index_path = os.path.join(base_dir, "index.csv")
    
    if not os.path.exists(index_path):
        st.info("Brak wykonanych testów. Przeprowadź analizę, aby zobaczyć raporty tutaj.")
        return
        
    # 2. Load Index
    try:
        df = pd.read_csv(index_path)
    except Exception as e:
        st.error(f"Błąd odczytu indeksu raportów: {e}")
        return
        
    if df.empty:
        st.info("Brak wykonanych testów.")
        return
        
    # 3. Display Table
    st.markdown("### Historia Analiz")
    
    # Sort by date descending
    if 'test_date' in df.columns:
        df['test_date'] = pd.to_datetime(df['test_date'])
        df = df.sort_values(by='test_date', ascending=False)
    
    # Ensure pdf_path is string
    if 'pdf_path' not in df.columns:
        df['pdf_path'] = ""
    else:
        df['pdf_path'] = df['pdf_path'].fillna("").astype(str)
        
    # Add visual indicator for PDF availability (single source of truth: pdf_path + file exists)
    def _check_pdf_exists(path: str) -> str:
        """Check if PDF file exists at given path."""
        if not path or not path.strip() or path.lower() == 'nan':
            return "❌"
        exists = os.path.exists(path)
        return "✅" if exists else "❌"
    
    df['PDF'] = df['pdf_path'].apply(_check_pdf_exists)
    
    # Select columns to display
    display_cols = ['test_date', 'session_id', 'athlete_id', 'PDF', 'method_version']
    # Filter only existing columns (PDF is synthetic, so it exists)
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Interactive dataframe
    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    st.caption("💡 Kliknij na wiersz, aby zobaczyć szczegóły i opcje PDF")
    
    # 4. Handle Selection
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        # Map back to original dataframe row using index if needed, 
        # but since we filtered/sorted "df" in place (mostly), we need to be careful.
        # Streamlit 1.35+ returns integer index relative to the displayed data?
        # Actually, st.dataframe returns keys relative to the original index if not hidden?
        # Let's rely on the fact that we sorted 'df' and passed it directly.
        # BUT: we modified 'df' above with fillna/columns.
        
        record = df.iloc[idx]
        
        st.divider()
        st.subheader(f"Szczegóły Testu: {record['test_date'].date()}")
        
        json_path = record.get('json_path')
        
        if not json_path or not os.path.exists(json_path):
            st.error(f"Plik raportu nie istnieje: {json_path}")
            return
            
        # Load canonical JSON
        try:
            report_data = load_ramp_test_report(json_path)
        except Exception as e:
            st.error(f"Błąd odczytu raportu: {e}")
            return
            
        # Display basic JSON info (optional preview)
        meta = report_data.get('metadata', {})
        col1, col2 = st.columns(2)
        with col1:
             st.write(f"**ID Sesji:** `{meta.get('session_id')}`")
             st.write(f"**Wersja metody:** `{meta.get('method_version')}`")
        with col2:
             st.write(f"**Timestamp:** `{meta.get('analysis_timestamp')}`")
             st.write(f"**Notatka:** {meta.get('notes', '-')}")

        # 5. Download buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # PDF Generation & Download
        with btn_col1:
            raw_path = record.get('pdf_path', '')
            pdf_path = str(raw_path).strip() if raw_path and str(raw_path).lower() != 'nan' else ""
            session_id = record.get('session_id', 'unknown')
            pdf_exists = os.path.exists(pdf_path) if pdf_path else False
            
            if pdf_path and pdf_exists:
                try:
                    with open(pdf_path, 'rb') as f:
                        pdf_data = f.read()
                    
                    st.download_button(
                        label="📕 Pobierz PDF",
                        data=pdf_data,
                        file_name=f"raport_ramp_{record['test_date'].date()}.pdf",
                        mime="application/pdf",
                        type="primary",
                        key=f"pdf_dl_{session_id}"
                    )
                    
                    # Simple regeneration (uses saved JSON values)
                    if st.button("🔄 Wygeneruj PDF ponownie", key=f"regen_pdf_{session_id}"):
                        from modules.reporting.persistence import generate_ramp_test_pdf
                        with st.spinner("Generowanie..."):
                            generate_ramp_test_pdf(session_id)
                            st.rerun()
                    
                    # Force regeneration with CURRENT MANUAL VALUES from session_state
                    if st.button("⚡ Generuj z wartościami manualnymi", key=f"regen_manual_{session_id}", type="secondary"):
                        from modules.reporting.persistence import generate_ramp_test_pdf
                        
                        # Collect manual overrides from session_state
                        manual_overrides = {
                            # VT1/VT2 from Manual Thresholds tab
                            "manual_vt1_watts": st.session_state.get("manual_vt1_watts", 0),
                            "manual_vt2_watts": st.session_state.get("manual_vt2_watts", 0),
                            "vt1_hr": st.session_state.get("vt1_hr", 0),
                            "vt2_hr": st.session_state.get("vt2_hr", 0),
                            "vt1_ve": st.session_state.get("vt1_ve", 0),
                            "vt2_ve": st.session_state.get("vt2_ve", 0),
                            "vt1_br": st.session_state.get("vt1_br", 0),
                            "vt2_br": st.session_state.get("vt2_br", 0),
                            # SmO2 from Manual SmO2 tab
                            "smo2_lt1_m": st.session_state.get("smo2_lt1_m", 0),
                            "smo2_lt2_m": st.session_state.get("smo2_lt2_m", 0),
                            # CP from Sidebar
                            "cp_input": st.session_state.get("cp_input", 0),
                            # CCI Breakpoint from Intervals tab
                            "cci_breakpoint_manual": st.session_state.get("cci_breakpoint_manual", 0),
                            # VE Breakpoint from Manual Thresholds tab
                            "ve_breakpoint_manual": st.session_state.get("ve_breakpoint_manual", 0),
                            # Reoxy Half-Time from SmO2 Manual tab
                            "reoxy_halftime_manual": st.session_state.get("reoxy_halftime_manual", 0),
                        }
                        
                        with st.spinner("Generowanie PDF z wartościami manualnymi..."):
                            generate_ramp_test_pdf(session_id, manual_overrides=manual_overrides)
                            st.success("✅ PDF wygenerowany z wartościami manualnymi!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Błąd PDF: {e}")
            else:
                if st.button("📕 Wygeneruj PDF", key=f"gen_pdf_{session_id}", type="primary"):
                    from modules.reporting.persistence import generate_ramp_test_pdf
                    with st.spinner("Generowanie..."):
                        generate_ramp_test_pdf(session_id)
                        st.rerun()

        # DOCX Download / Generate
        with btn_col2:
            docx_path = ""
            if pdf_path:
                docx_path = str(Path(pdf_path).with_suffix(".docx"))
            
            docx_exists = os.path.exists(docx_path) if docx_path else False
            
            if docx_path and docx_exists:
                try:
                    with open(docx_path, 'rb') as f:
                        docx_data = f.read()
                        
                    st.download_button(
                        label="📝 Pobierz DOCX",
                        data=docx_data,
                        file_name=f"raport_ramp_{record['test_date'].date()}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"docx_btn_{session_id}"
                    )
                except Exception as e:
                    st.error(f"Błąd DOCX: {e}")
            
            # Generate DOCX button (always visible)
            if st.button("📝 Generuj DOCX", key=f"gen_docx_{session_id}", type="secondary"):
                from modules.reporting.docx_builder import build_ramp_docx
                
                with st.spinner("Generowanie DOCX..."):
                    try:
                        # Generate DOCX path  
                        if pdf_path:
                            docx_output = str(Path(pdf_path).with_suffix(".docx"))
                        else:
                            docx_dir = f"reports/ramp_tests/{session_id}"
                            os.makedirs(docx_dir, exist_ok=True)
                            docx_output = f"{docx_dir}/raport.docx"
                        
                        # Get figure paths from session directory
                        session_dir = str(Path(json_path).parent)
                        figure_paths_docx = {}
                        for fig_name in ["ramp_profile", "ve_profile", "smo2_power", "pdc_curve", 
                                         "biomech_summary", "biomech_torque_smo2", "limiters_radar",
                                         "drift_heatmap_hr", "drift_heatmap_smo2", 
                                         "thermal_hsi", "thermal_efficiency"]:
                            fig_path = os.path.join(session_dir, f"{fig_name}.png")
                            if os.path.exists(fig_path):
                                figure_paths_docx[fig_name] = fig_path
                        
                        result = build_ramp_docx(
                            report_data=report_data,
                            figure_paths=figure_paths_docx,
                            output_path=docx_output
                        )
                        
                        if result:
                            st.success("✅ DOCX wygenerowany!")
                            st.rerun()
                        else:
                            st.error("Błąd generowania DOCX")
                    except Exception as e:
                        st.error(f"Błąd: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # HTML Generation
        with btn_col3:
            if st.button("📄 Wygeneruj HTML", key=f"html_btn_{session_id}"):
                with st.spinner("Generowanie..."):
                    html_content = generate_html_report(report_data)
                    
                    st.download_button(
                        label="⬇️ Pobierz HTML",
                        data=html_content,
                        file_name=f"raport_ramp_{record['test_date'].date()}.html",
                        mime="text/html",
                        key=f"html_dl_{session_id}"
                    )
        
        # Preview (in expander)
        with st.expander("📋 Podgląd danych JSON"):
            st.json(report_data)

