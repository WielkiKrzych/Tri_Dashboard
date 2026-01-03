"""
DOCX Builder Module.

Generates editable Word documents for Ramp Test reports.
Mirrors the structure of the PDF report.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from .pdf.builder import map_ramp_json_to_pdf_data

logger = logging.getLogger("Tri_Dashboard.DOCXBuilder")

def build_ramp_docx(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None
) -> Optional[str]:
    """Build Ramp Test DOCX report."""
    if not HAS_DOCX:
        logger.error("python-docx library not installed. Cannot generate DOCX.")
        return None

    try:
        doc = Document()
        
        # Styles setup (basic)
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)

        # 1. Map Data
        data = map_ramp_json_to_pdf_data(report_data)
        meta = data['metadata']
        thresholds = data['thresholds']
        cp = data['cp_model']
        smo2 = data['smo2']
        smo2_manual = data['smo2_manual']

        # === PAGE 1: TITLE & SUMMARY ===
        doc.add_heading("Raport z Testu Ramp", 0)
        
        p = doc.add_paragraph()
        p.add_run(f"Data: {meta['test_date']} | ID: {meta['session_id'][:8]}").bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_heading("Kluczowe Wyniki", 1)
        
        table = doc.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Parametr'
        hdr_cells[1].text = 'Wartość'
        
        # Helper to add row
        def add_row(k, v):
            row = table.add_row().cells
            row[0].text = k
            row[1].text = str(v)

        add_row("VT1 (W)", thresholds['vt1_watts'])
        add_row("VT2 (W)", thresholds['vt2_watts'])
        add_row("CP (W)", cp['cp_watts'])
        add_row("CP/kg", f"{float(cp['cp_watts'])/float(meta['athlete_weight_kg']):.2f} W/kg" if meta['athlete_weight_kg'] and cp['cp_watts'] != 'brak danych' else "brak danych")
        add_row("W' (kJ)", cp['w_prime_kj'])
        add_row("Pmax (W)", meta['pmax_watts'])
        
        # Chart: Ramp Profile
        if figure_paths and "ramp_profile" in figure_paths:
            doc.add_heading("Przebieg Testu", 2)
            doc.add_picture(figure_paths["ramp_profile"], width=Inches(6))

        doc.add_page_break()

        # === PAGE 2: THRESHOLDS ===
        doc.add_heading("Szczegóły Progów VT1 / VT2", 1)
        
        table_vt = doc.add_table(rows=1, cols=4)
        table_vt.style = 'Table Grid'
        hdr = table_vt.rows[0].cells
        hdr[0].text = "Próg"
        hdr[1].text = "Moc [W]"
        hdr[2].text = "HR [bpm]"
        hdr[3].text = "VE [L/min]"
        
        row1 = table_vt.add_row().cells
        row1[0].text = "VT1"
        row1[1].text = str(thresholds['vt1_watts'])
        row1[2].text = str(thresholds['vt1_hr'])
        row1[3].text = str(thresholds['vt1_ve'])
        
        row2 = table_vt.add_row().cells
        row2[0].text = "VT2"
        row2[1].text = str(thresholds['vt2_watts'])
        row2[2].text = str(thresholds['vt2_hr'])
        row2[3].text = str(thresholds['vt2_ve'])

        if figure_paths and "ve_profile" in figure_paths:
             doc.add_picture(figure_paths["ve_profile"], width=Inches(6))
             
        doc.add_page_break()
        
        # === PAGE 3: SmO2 ===
        doc.add_heading("Analiza SmO₂", 1)
        
        table_smo2 = doc.add_table(rows=1, cols=3)
        table_smo2.style = 'Table Grid'
        h = table_smo2.rows[0].cells
        h[0].text = "Próg"
        h[1].text = "Moc [W]"
        h[2].text = "HR [bpm]"
        
        r1 = table_smo2.add_row().cells
        r1[0].text = "Próg Tlenowy Mięśni (LT1)"
        r1[1].text = str(smo2_manual.get('lt1_watts'))
        r1[2].text = str(smo2_manual.get('lt1_hr', 'brak danych')) # Using updated key
        
        r2 = table_smo2.add_row().cells
        r2[0].text = "Próg Beztlenowy Mięśni (LT2)"
        r2[1].text = str(smo2_manual.get('lt2_watts'))
        r2[2].text = str(smo2_manual.get('lt2_hr', 'brak danych'))

        if figure_paths and "smo2_power" in figure_paths:
             doc.add_picture(figure_paths["smo2_power"], width=Inches(6))

        doc.add_page_break()

        # === PAGE 5: TEORIA ===
        doc.add_heading("Teoria Fizjologiczna (INSCYD/WKO5)", 1)
        doc.add_paragraph("VO2max vs VLaMax...")
        # (Short version for DOCX, user can edit)

        doc.add_page_break()

        # === PAGE 6/7: THERMAL & LIMITERS ===
        doc.add_heading("Analiza Termoregulacji", 1)
        if figure_paths and "thermal_hsi" in figure_paths:
             doc.add_picture(figure_paths["thermal_hsi"], width=Inches(6))
             
        doc.add_heading("Efektywność (Cardiac Drift)", 1)
        if figure_paths and "thermal_efficiency" in figure_paths:
             doc.add_picture(figure_paths["thermal_efficiency"], width=Inches(6))
             
        doc.add_heading("Profil Metaboliczny (Radar)", 1)
        if figure_paths and "limiters_radar" in figure_paths:
             doc.add_picture(figure_paths["limiters_radar"], width=Inches(6))

        doc.add_page_break()

        # === PAGE 8: EXTRA ===
        doc.add_heading("Analityka Zaawansowana", 1)
        if figure_paths and "vent_full" in figure_paths:
             doc.add_picture(figure_paths["vent_full"], width=Inches(6))
             
        if figure_paths and "drift_hr" in figure_paths:
             doc.add_picture(figure_paths["drift_hr"], width=Inches(6))
             
        if figure_paths and "drift_smo2" in figure_paths:
             doc.add_picture(figure_paths["drift_smo2"], width=Inches(6))

        # Save
        if output_path:
            doc.save(output_path)
            return output_path
            
    except Exception as e:
        logger.error(f"DOCX Generation failed: {e}")
        return None
    
    return None
