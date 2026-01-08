"""
DOCX Builder Module (v2.0 - Full PDF Parity).

Generates editable Word documents for Ramp Test reports.
Structure matches PDF 1:1 with COACH NOTES after each section.

Key Features:
- Single Source of Truth: uses map_ramp_json_to_pdf_data()
- 17 sections in exact PDF order
- COACH NOTES block after every section
- Pages-friendly styles (no custom fonts, no floating elements)
"""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from .pdf.builder import map_ramp_json_to_pdf_data

logger = logging.getLogger("Tri_Dashboard.DOCXBuilder")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _add_coach_notes(doc, title: str = "📝 NOTATKI TRENERA", lines: int = 6):
    """Add Coach Notes block after each section.
    
    This is the key differentiator from PDF - editable space for trainer comments.
    """
    # Separator line
    p_sep = doc.add_paragraph()
    p_sep.add_run("─" * 60)
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Title
    p_title = doc.add_paragraph()
    run_title = p_title.add_run(title)
    run_title.bold = True
    run_title.font.size = Pt(11)
    run_title.font.color.rgb = RGBColor(41, 128, 185)  # Blue
    
    # Instruction
    p_instr = doc.add_paragraph()
    run_instr = p_instr.add_run("[Miejsce na Twoje obserwacje i komentarze]")
    run_instr.italic = True
    run_instr.font.color.rgb = RGBColor(127, 140, 141)  # Gray
    
    # Empty lines for notes
    for _ in range(lines):
        doc.add_paragraph()
    
    # Closing separator
    p_sep2 = doc.add_paragraph()
    p_sep2.add_run("─" * 60)
    p_sep2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()  # Extra spacing


def _add_section_header(doc, title: str, level: int = 1):
    """Add section header with consistent styling."""
    doc.add_heading(title, level)


def _add_metric_table(doc, data: List[List[str]], has_header: bool = True):
    """Add a simple editable table."""
    if not data:
        return
    
    rows = len(data)
    cols = len(data[0]) if data else 0
    
    if rows == 0 or cols == 0:
        return
    
    table = doc.add_table(rows=rows, cols=cols)
    table.style = 'Table Grid'
    
    for i, row_data in enumerate(data):
        cells = table.rows[i].cells
        for j, val in enumerate(row_data):
            cells[j].text = str(val) if val is not None else "---"
            # Bold header row
            if i == 0 and has_header:
                for paragraph in cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
    
    doc.add_paragraph()  # Spacing after table


def _add_verdict_box(doc, text: str, color: str = "info"):
    """Add a styled verdict/callout box (as indented paragraph)."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.right_indent = Cm(0.5)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(8)
    
    run = p.add_run(text)
    run.font.size = Pt(10)
    
    if color == "warning":
        run.font.color.rgb = RGBColor(231, 76, 60)  # Red
    elif color == "success":
        run.font.color.rgb = RGBColor(39, 174, 96)  # Green
    else:
        run.font.color.rgb = RGBColor(52, 73, 94)  # Dark gray


# =============================================================================
# SECTION BUILDERS (mirror PDF layout.py structure)
# =============================================================================

def _build_section_toc(doc, section_titles: List[Dict[str, str]]):
    """Section 0: Table of Contents."""
    _add_section_header(doc, "SPIS TREŚCI", 0)
    
    toc_data = [["Sekcja", "Strona"]]
    for section in section_titles:
        toc_data.append([section.get("title", ""), section.get("page", "")])
    
    _add_metric_table(doc, toc_data)
    _add_coach_notes(doc, lines=3)
    doc.add_page_break()


def _build_section_cover(doc, data: Dict, meta: Dict):
    """Section 1: Cover Page (Badania Wydolnościowe - Raport Potestowy)."""
    _add_section_header(doc, "Badania Wydolnościowe - Raport Potestowy", 0)
    
    # Metadata
    p = doc.add_paragraph()
    p.add_run(f"Data: {meta.get('test_date', '---')} | ID: {meta.get('session_id', '')[:8]} | v{meta.get('method_version', '1.0.0')}").bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()
    
    # Key Results Table
    _add_section_header(doc, "Kluczowe Wyniki", 2)
    
    thresholds = data.get("thresholds", {})
    cp = data.get("cp_model", {})
    
    results = [
        ["Parametr", "Wartość"],
        ["VT1 (Próg tlenowy)", f"{thresholds.get('vt1_watts', '---')} W"],
        ["VT2 (Próg beztlenowy)", f"{thresholds.get('vt2_watts', '---')} W"],
        ["Critical Power (CP)", f"{cp.get('cp_watts', '---')} W"],
        ["W' (Rezerwa beztlenowa)", f"{cp.get('w_prime_kj', '---')} kJ"],
    ]
    _add_metric_table(doc, results)
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_zones(doc, thresholds: Dict):
    """Section 2: Training Zones."""
    _add_section_header(doc, "Rekomendowane Strefy Treningowe", 1)
    
    vt1_str = str(thresholds.get("vt1_watts", "0"))
    vt2_str = str(thresholds.get("vt2_watts", "0"))
    
    if vt1_str.isdigit() and vt2_str.isdigit() and int(vt1_str) > 0:
        vt1, vt2 = float(vt1_str), float(vt2_str)
        
        zones = [
            ["Strefa", "Zakres [W]", "Opis", "Cel"],
            ["Z1 Recovery", f"< {int(vt1 * 0.8)}", "Bardzo łatwy", "Regeneracja"],
            ["Z2 Endurance", f"{int(vt1 * 0.8)}–{int(vt1)}", "Komfortowy", "Baza tlenowa"],
            ["Z3 Tempo", f"{int(vt1)}–{int(vt2)}", "Umiarkowany", "Próg"],
            ["Z4 Threshold", f"{int(vt2)}–{int(vt2 * 1.05)}", "Ciężki", "Wytrzymałość"],
            ["Z5 VO₂max", f"> {int(vt2 * 1.05)}", "Maksymalny", "Kapacytacja"],
        ]
        _add_metric_table(doc, zones)
    else:
        doc.add_paragraph("Brak danych do wyznaczenia stref (wymagane VT1 i VT2).")
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_thresholds(doc, thresholds: Dict, figure_paths: Dict):
    """Section 3: VT1/VT2 Threshold Details."""
    _add_section_header(doc, "Szczegóły Progów VT1/VT2", 1)
    
    data = [
        ["Próg", "Moc [W]", "HR [bpm]", "VE [L/min]"],
        ["VT1", thresholds.get('vt1_watts', '---'), thresholds.get('vt1_hr', '---'), thresholds.get('vt1_ve', '---')],
        ["VT2", thresholds.get('vt2_watts', '---'), thresholds.get('vt2_hr', '---'), thresholds.get('vt2_ve', '---')],
    ]
    _add_metric_table(doc, data)
    
    # Chart
    if "ve_profile" in figure_paths and os.path.exists(figure_paths["ve_profile"]):
        doc.add_picture(figure_paths["ve_profile"], width=Inches(5.5))
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_interpretation(doc, thresholds: Dict, cp: Dict):
    """Section 4: What Do These Results Mean?"""
    _add_section_header(doc, "Co oznaczają te wyniki?", 1)
    
    doc.add_paragraph(
        "Progi wentylacyjne to Twoje najważniejsze drogowskazy w planowaniu obciążeń. "
        "VT1 wyznacza granicę komfortu tlenowego, VT2 to 'szklany sufit' powyżej którego "
        "kwas narasta szybciej niż organizm go utylizuje."
    )
    
    _add_section_header(doc, "Interpretacja VT1 (Próg tlenowy)", 2)
    vt1_w = thresholds.get('vt1_watts', '---')
    doc.add_paragraph(f"VT1 = {vt1_w} W — górna granica treningu bazowego, strefy Z1-Z2.")
    
    _add_section_header(doc, "Interpretacja VT2 (Próg beztlenowy)", 2)
    vt2_w = thresholds.get('vt2_watts', '---')
    doc.add_paragraph(f"VT2 = {vt2_w} W — próg mleczanowy, granica tempo/threshold.")
    
    _add_section_header(doc, "Interpretacja CP (Critical Power)", 2)
    cp_w = cp.get('cp_watts', '---')
    wprime = cp.get('w_prime_kj', '---')
    doc.add_paragraph(f"CP = {cp_w} W, W' = {wprime} kJ — Twoja moc krytyczna i rezerwa beztlenowa.")
    
    _add_coach_notes(doc, lines=8)  # Larger block for interpretation
    doc.add_page_break()


def _build_section_ventilation(doc, vent_data: Dict):
    """Section 5: Breathing & Metabolic Control."""
    _add_section_header(doc, "Kontrola Oddychania i Metabolizmu", 1)
    
    if not vent_data:
        doc.add_paragraph("Brak danych wentylacyjnych.")
        _add_coach_notes(doc)
        doc.add_page_break()
        return
    
    # VE Metrics
    metrics = vent_data.get("metrics", {})
    ve_data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["VE Peak", f"{metrics.get('ve_peak', '---')} L/min", "Szczytowa wentylacja"],
        ["VE/VCO2 @ VT1", f"{metrics.get('ve_vco2_vt1', '---')}", "Efektywność oddychania"],
        ["Breathing Reserve", f"{metrics.get('breathing_reserve', '---')}%", "Margines wentylacyjny"],
    ]
    _add_metric_table(doc, ve_data)
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_metabolic_engine(doc, metabolic_data: Dict):
    """Section 6: Metabolic Engine & Training Strategy."""
    _add_section_header(doc, "Silnik Metaboliczny i Strategia Treningowa", 1)
    
    if not metabolic_data:
        doc.add_paragraph("Brak danych o profilu metabolicznym.")
        _add_coach_notes(doc)
        doc.add_page_break()
        return
    
    profile = metabolic_data.get("profile", {})
    
    doc.add_paragraph(
        f"Profil metaboliczny determinuje Twoją strategię treningową. "
        f"Zidentyfikowany limiter: {profile.get('limiter', 'nieznany')}."
    )
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_pdc(doc, cp: Dict, figure_paths: Dict):
    """Section 7: Power Duration Curve."""
    _add_section_header(doc, "Krzywa Mocy (PDC) i Critical Power", 1)
    
    doc.add_paragraph(f"Critical Power (CP): {cp.get('cp_watts', '---')} W")
    doc.add_paragraph(f"W' (Pojemność Beztlenowa): {cp.get('w_prime_kj', '---')} kJ")
    
    if "pdc_curve" in figure_paths and os.path.exists(figure_paths["pdc_curve"]):
        doc.add_picture(figure_paths["pdc_curve"], width=Inches(5.5))
    
    doc.add_paragraph(
        "Model CP/W' to Twoja 'cyfrowa bateria'. CP to moc utrzymywana bez wyczerpania rezerw, "
        "W' to 'bak paliwa' na ataki i sprinty powyżej mocy progowej."
    )
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_smo2(doc, smo2_data: Dict, smo2_manual: Dict, figure_paths: Dict):
    """Section 8: Muscle Oxygenation Diagnostic."""
    _add_section_header(doc, "Diagnostyka Oksygenacji Mięśniowej (SmO₂)", 1)
    
    doc.add_paragraph(
        "SmO₂ dostarcza informacji o balansie między podażą a zapotrzebowaniem na tlen "
        "bezpośrednio w pracującym mięśniu. Jest to sygnał lokalny."
    )
    
    smo2_table = [
        ["Punkt", "Moc [W]", "HR [bpm]"],
        ["SmO₂ LT1 (Drop Point)", smo2_manual.get('lt1_watts', '---'), smo2_manual.get('lt1_hr', '---')],
        ["SmO₂ LT2 (Steady Limit)", smo2_manual.get('lt2_watts', '---'), smo2_manual.get('lt2_hr', '---')],
    ]
    _add_metric_table(doc, smo2_table)
    
    if "smo2_power" in figure_paths and os.path.exists(figure_paths["smo2_power"]):
        doc.add_picture(figure_paths["smo2_power"], width=Inches(5.5))
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_cardiovascular(doc, cardio_data: Dict):
    """Section 9: Cardiovascular Cost Diagnostic."""
    _add_section_header(doc, "Diagnostyka Kosztu Sercowo-Naczyniowego", 1)
    
    if not cardio_data:
        doc.add_paragraph("Brak danych sercowo-naczyniowych.")
        _add_coach_notes(doc)
        doc.add_page_break()
        return
    
    pp = cardio_data.get("pulse_power", 0)
    ef = cardio_data.get("efficiency_factor", 0)
    drift = cardio_data.get("hr_drift_pct", 0)
    cci = cardio_data.get("cci_avg", 0)
    
    cardio_table = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Moc Pulsowa", f"{pp:.2f} W/bpm", "Moc na uderzenie serca"],
        ["Wsp. Efektywności (EF)", f"{ef:.2f} W/bpm", "Wydajność krążenia"],
        ["Dryf HR", f"{drift:.1f}%", "Stabilność tętna"],
        ["CCI", f"{cci:.4f} bpm/W", "Koszt sercowy mocy"],
    ]
    _add_metric_table(doc, cardio_table)
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_limiter_radar(doc, limiter_data: Dict, figure_paths: Dict):
    """Section 10: System Load Radar."""
    _add_section_header(doc, "Radar Obciążenia Systemów", 1)
    
    if "limiters_radar" in figure_paths and os.path.exists(figure_paths["limiters_radar"]):
        doc.add_picture(figure_paths["limiters_radar"], width=Inches(5.5))
    
    doc.add_paragraph(
        "Radar pokazuje obciążenie poszczególnych systemów fizjologicznych podczas testu. "
        "System z najwyższym obciążeniem jest Twoim głównym limitatorem wydajności."
    )
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_biomech(doc, biomech_data: Dict, figure_paths: Dict):
    """Section 11: Biomechanical Analysis."""
    _add_section_header(doc, "Analiza Biomechaniczna", 1)
    
    doc.add_paragraph(
        "Analiza biomechaniczna skupia się na sposobie generowania mocy. "
        "Balans między kadencją a momentem obrotowym pozwala zidentyfikować optymalny styl jazdy."
    )
    
    if "biomech_summary" in figure_paths and os.path.exists(figure_paths["biomech_summary"]):
        doc.add_picture(figure_paths["biomech_summary"], width=Inches(5.5))
    
    if "biomech_torque_smo2" in figure_paths and os.path.exists(figure_paths["biomech_torque_smo2"]):
        doc.add_picture(figure_paths["biomech_torque_smo2"], width=Inches(5.5))
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_drift(doc, kpi: Dict, figure_paths: Dict):
    """Section 12: Physiological Drift."""
    _add_section_header(doc, "Dryf Fizjologiczny", 1)
    
    if "drift_heatmap_hr" in figure_paths and os.path.exists(figure_paths["drift_heatmap_hr"]):
        _add_section_header(doc, "Mapa Dryfu HR vs Power", 2)
        doc.add_picture(figure_paths["drift_heatmap_hr"], width=Inches(5.5))
    
    if "drift_heatmap_smo2" in figure_paths and os.path.exists(figure_paths["drift_heatmap_smo2"]):
        _add_section_header(doc, "Mapa Oksydacji SmO2 vs Power", 2)
        doc.add_picture(figure_paths["drift_heatmap_smo2"], width=Inches(5.5))
    
    doc.add_paragraph(
        "Dryf tętna to sygnał ostrzegawczy Twojego układu chłodzenia. "
        "Jeśli przy stałej mocy tętno systematycznie rośnie, serce musi pracować ciężej."
    )
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_kpi(doc, kpi: Dict):
    """Section 13: Key Performance Indicators."""
    _add_section_header(doc, "Kluczowe Wskaźniki Wydajności (KPI)", 1)
    
    kpi_table = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Efficiency Factor", kpi.get("ef", "---"), "Moc na uderzenie serca"],
        ["Pa:Hr (Decoupling)", f"{kpi.get('pa_hr', '---')}%", "Stabilność krążenia"],
        ["SmO2 Drift", f"{kpi.get('smo2_drift', '---')}%", "Zmęczenie lokalne"],
        ["VO2max Estimate", f"{kpi.get('vo2max_est', '---')} ml/kg", "Szacowany pułap"],
    ]
    _add_metric_table(doc, kpi_table)
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_thermal(doc, thermo_data: Dict, figure_paths: Dict):
    """Section 14: Thermoregulation Analysis."""
    _add_section_header(doc, "Analiza Termoregulacji", 1)
    
    if "thermal_hsi" in figure_paths and os.path.exists(figure_paths["thermal_hsi"]):
        _add_section_header(doc, "Heat Strain Index (HSI)", 2)
        doc.add_picture(figure_paths["thermal_hsi"], width=Inches(5.5))
    
    if "thermal_efficiency" in figure_paths and os.path.exists(figure_paths["thermal_efficiency"]):
        _add_section_header(doc, "Efektywność vs Temperatura", 2)
        doc.add_picture(figure_paths["thermal_efficiency"], width=Inches(5.5))
    
    metrics = thermo_data.get("metrics", {})
    if metrics:
        thermal_table = [
            ["Metryka", "Wartość"],
            ["Max Core Temp", f"{metrics.get('max_core_temp', '---')} °C"],
            ["Peak HSI", f"{metrics.get('peak_hsi', '---')}"],
            ["Delta Temp / 10 min", f"{metrics.get('delta_per_10min', '---')} °C"],
        ]
        _add_metric_table(doc, thermal_table)
    
    _add_coach_notes(doc)
    doc.add_page_break()


def _build_section_executive_summary(doc, exec_data: Dict, meta: Dict):
    """Section 15: Physiological Summary (Executive Summary + Verdict)."""
    _add_section_header(doc, "Podsumowanie Fizjologiczne", 1)
    
    limiter = exec_data.get("limiter", {})
    training_cards = exec_data.get("training_cards", [])
    confidence_panel = exec_data.get("confidence_panel", {})
    
    # Primary Limiter
    _add_section_header(doc, "Dominujący Limiter", 2)
    limiter_name = limiter.get("name", "NIEZNANY")
    verdict = limiter.get("verdict", "")
    
    p = doc.add_paragraph()
    p.add_run(f"{limiter.get('icon', '⚖️')} {limiter_name}").bold = True
    p.add_run(f"\n{verdict}")
    
    for line in limiter.get("interpretation", [])[:3]:
        doc.add_paragraph(f"• {line}")
    
    # Test Confidence
    _add_section_header(doc, "Pewność Testu", 2)
    overall = confidence_panel.get("overall_score", 0)
    label = confidence_panel.get("label", "---")
    doc.add_paragraph(f"Pewność: {overall}% ({label})")
    
    # Training Decisions (FULLY EDITABLE)
    _add_section_header(doc, "Decyzje Treningowe", 2)
    
    for i, card in enumerate(training_cards[:3], 1):
        p_card = doc.add_paragraph()
        p_card.add_run(f"{i}. {card.get('strategy_name', '---')}").bold = True
        
        doc.add_paragraph(f"Moc: {card.get('power_range', '---')} | Objętość: {card.get('volume', '---')}")
        doc.add_paragraph(f"Cel: {card.get('adaptation_goal', '---')}")
        
        risk = card.get("risk_level", "low")
        risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
        doc.add_paragraph(f"Spodziewany efekt: {card.get('expected_response', '---')} | Ryzyko: {risk_label}")
        doc.add_paragraph()  # Spacing between cards
    
    _add_coach_notes(doc, lines=10)  # Larger block for executive summary
    doc.add_page_break()


def _build_section_limitations(doc, conf: Dict):
    """Section 16: Interpretation Limitations."""
    _add_section_header(doc, "Ograniczenia Interpretacji", 1)
    
    limitations = [
        ("1. To nie jest badanie medyczne.", 
         "Wyniki są szacunkami algorytmicznymi, nie pomiarami laboratoryjnymi."),
        ("2. Dokładność zależy od jakości danych.", 
         "Niepoprawna kalibracja czujników może wpłynąć na wyniki."),
        ("3. Progi są przybliżeniami.", 
         "VT1/VT2 mogą się różnić od wyników spirometrycznych."),
        ("4. Wyniki są jednorazowe.", 
         "Wydolność zmienia się w czasie – powtarzaj testy co 6-8 tygodni."),
    ]
    
    for title, description in limitations:
        p = doc.add_paragraph()
        p.add_run(title).bold = True
        doc.add_paragraph(description)
    
    # Warnings
    warnings = conf.get("warnings", [])
    if warnings:
        _add_section_header(doc, "Ostrzeżenia", 2)
        for w in warnings:
            doc.add_paragraph(f"⚠️ {w}")
    
    # Notes
    notes = conf.get("notes", [])
    if notes:
        _add_section_header(doc, "Notatki z Analizy", 2)
        for n in notes:
            doc.add_paragraph(f"ℹ️ {n}")
    
    _add_coach_notes(doc, lines=5)


# =============================================================================
# MAIN BUILDER FUNCTION
# =============================================================================

def build_ramp_docx(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None
) -> Optional[str]:
    """Build complete Ramp Test DOCX report with PDF parity + COACH NOTES.
    
    Structure mirrors PDF exactly (17 sections) but adds editable COACH NOTES
    after each section for trainer annotations.
    
    Args:
        report_data: Canonical JSON report data
        figure_paths: Dict of chart name -> file path
        output_path: Where to save the DOCX file
        
    Returns:
        Path to saved DOCX file, or None on error
    """
    if not HAS_DOCX:
        logger.error("python-docx library not installed. Cannot generate DOCX.")
        return None

    try:
        figure_paths = figure_paths or {}
        doc = Document()
        
        # Map data using SAME function as PDF (Single Source of Truth)
        data = map_ramp_json_to_pdf_data(report_data)
        
        meta = data['metadata']
        thresholds = data['thresholds']
        cp = data['cp_model']
        smo2 = data['smo2']
        smo2_manual = data['smo2_manual']
        conf = data['confidence']
        kpi = data['kpi']
        exec_summary = data.get('executive_summary', {})
        vent_data = data.get('vent_advanced', {})
        metabolic_data = data.get('metabolic_strategy', {})
        cardio_data = data.get('cardio_advanced', {})
        limiter_data = data.get('limiter_analysis', {})
        biomech_data = data.get('biomech_occlusion', {})
        thermo_data = data.get('thermo_analysis', {})
        
        # Section titles (same as PDF)
        section_titles = [
            {"title": "1. Badania Wydolnościowe - Raport Potestowy", "page": "2"},
            {"title": "2. Rekomendowane Strefy Treningowe", "page": "3"},
            {"title": "3. Szczegóły Progów VT1/VT2", "page": "4"},
            {"title": "4. Co oznaczają te wyniki?", "page": "5"},
            {"title": "5. Kontrola Oddychania i Metabolizmu", "page": "6"},
            {"title": "6. Silnik Metaboliczny i Strategia", "page": "7"},
            {"title": "7. Krzywa Mocy (PDC)", "page": "8"},
            {"title": "8. Diagnostyka Oksygenacji Mięśniowej", "page": "9"},
            {"title": "9. Diagnostyka Sercowo-Naczyniowa", "page": "10"},
            {"title": "10. Radar Obciążenia Systemów", "page": "11"},
            {"title": "11. Analiza Biomechaniczna", "page": "12"},
            {"title": "12. Dryf Fizjologiczny", "page": "13"},
            {"title": "13. Kluczowe Wskaźniki Wydajności", "page": "14"},
            {"title": "14. Analiza Termoregulacji", "page": "15"},
            {"title": "15. Podsumowanie Fizjologiczne", "page": "16"},
            {"title": "16. Ograniczenia Interpretacji", "page": "17"},
        ]
        
        # =======================================================================
        # BUILD ALL SECTIONS (same order as PDF)
        # =======================================================================
        
        # 0. Table of Contents
        _build_section_toc(doc, section_titles)
        
        # 1. Cover Page
        _build_section_cover(doc, data, meta)
        
        # 2. Training Zones
        _build_section_zones(doc, thresholds)
        
        # 3. VT1/VT2 Thresholds
        _build_section_thresholds(doc, thresholds, figure_paths)
        
        # 4. Interpretation
        _build_section_interpretation(doc, thresholds, cp)
        
        # 5. Ventilation
        _build_section_ventilation(doc, vent_data)
        
        # 6. Metabolic Engine
        _build_section_metabolic_engine(doc, metabolic_data)
        
        # 7. PDC & CP
        _build_section_pdc(doc, cp, figure_paths)
        
        # 8. SmO2 Diagnostic
        _build_section_smo2(doc, smo2, smo2_manual, figure_paths)
        
        # 9. Cardiovascular
        _build_section_cardiovascular(doc, cardio_data)
        
        # 10. Limiter Radar
        _build_section_limiter_radar(doc, limiter_data, figure_paths)
        
        # 11. Biomechanical Analysis
        _build_section_biomech(doc, biomech_data, figure_paths)
        
        # 12. Physiological Drift
        _build_section_drift(doc, kpi, figure_paths)
        
        # 13. KPI Dashboard
        _build_section_kpi(doc, kpi)
        
        # 14. Thermoregulation
        _build_section_thermal(doc, thermo_data, figure_paths)
        
        # 15. Executive Summary + Verdict
        _build_section_executive_summary(doc, exec_summary, meta)
        
        # 16. Limitations
        _build_section_limitations(doc, conf)
        
        # =======================================================================
        # SAVE
        # =======================================================================
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_path)
            logger.info(f"DOCX saved to: {output_path}")
            return output_path
            
    except Exception as e:
        logger.error(f"DOCX Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    return None
