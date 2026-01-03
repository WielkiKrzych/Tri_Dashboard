"""
Ramp Test PDF Report Generator.

Generates athlete-facing PDF report from canonical JSON and figures.
No raw data, no algorithms - clean presentation only.

Per specification: methodology/ramp_test/10_pdf_report_spec.md
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, gray
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .figures import generate_all_ramp_figures, FigureConfig


# Colors from spec
COLOR_VT1 = HexColor("#FFA15A")
COLOR_VT2 = HexColor("#EF553B")
COLOR_PRIMARY = HexColor("#1F77B4")
COLOR_CONFIDENCE_OK = HexColor("#2ECC71")
COLOR_CONFIDENCE_WARN = HexColor("#F1C40F")
COLOR_GRAY = HexColor("#7F8C8D")


@dataclass
class PDFConfig:
    """Configuration for PDF generation."""
    page_size: tuple = A4
    margin: float = 15 * mm
    title: str = "Raport z testu Ramp"
    author: str = "Tri_Dashboard"
    include_charts: bool = True


def generate_ramp_pdf(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    config: Optional[PDFConfig] = None
) -> bytes:
    """Generate complete PDF report for athlete.
    
    Args:
        report_data: Canonical JSON report dictionary
        figure_paths: Dict mapping chart name to file path (optional, will generate if None)
        output_path: File path to save PDF (optional, returns bytes if None)
        config: PDF configuration
        
    Returns:
        PDF bytes if output_path is None
    """
    config = config or PDFConfig()
    
    # Generate figures if not provided
    if figure_paths is None and config.include_charts:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        figure_paths = generate_all_ramp_figures(report_data, temp_dir)
    
    # Setup document
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=config.page_size,
        leftMargin=config.margin,
        rightMargin=config.margin,
        topMargin=config.margin,
        bottomMargin=config.margin
    )
    
    # Build content
    styles = _create_styles()
    story = []
    
    # Extract data
    metadata = report_data.get("metadata", {})
    thresholds = report_data.get("thresholds", {})
    cp_model = report_data.get("cp_model", {})
    confidence = report_data.get("confidence", {})
    
    # === PAGE 1: Header + Summary ===
    story.extend(_build_header(metadata, styles))
    story.append(Spacer(1, 10 * mm))
    
    story.extend(_build_confidence_badge(confidence, styles))
    story.append(Spacer(1, 8 * mm))
    
    story.extend(_build_results_summary(thresholds, cp_model, metadata, styles))
    story.append(Spacer(1, 8 * mm))
    
    # Ramp profile chart
    if figure_paths and "ramp_profile" in figure_paths:
        story.extend(_build_chart_section(
            figure_paths["ramp_profile"], 
            "Przebieg Testu Ramp",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 2: Thresholds + SmO2 ===
    story.extend(_build_thresholds_section(thresholds, styles))
    story.append(Spacer(1, 8 * mm))
    
    if figure_paths and "smo2_power" in figure_paths:
        story.extend(_build_chart_section(
            figure_paths["smo2_power"],
            "SmO₂ vs Moc – Sygnał Wspierający",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 3: PDC + CP ===
    story.extend(_build_cp_section(cp_model, styles))
    story.append(Spacer(1, 8 * mm))
    
    if figure_paths and "pdc" in figure_paths:
        story.extend(_build_chart_section(
            figure_paths["pdc"],
            "Power-Duration Curve",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 4: Interpretation + Limitations ===
    story.extend(_build_interpretation_section(thresholds, cp_model, styles))
    story.append(Spacer(1, 10 * mm))
    
    story.extend(_build_limitations_section(styles))
    story.append(Spacer(1, 10 * mm))
    
    story.extend(_build_training_zones(thresholds, styles))
    
    # Build PDF
    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)
    
    # Return or save
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes


def _create_styles() -> Dict[str, ParagraphStyle]:
    """Create custom styles for the report."""
    base = getSampleStyleSheet()
    
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Heading1"],
            fontSize=24,
            textColor=COLOR_PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=5 * mm
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=COLOR_GRAY,
            alignment=TA_CENTER
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontSize=16,
            textColor=black,
            spaceBefore=8 * mm,
            spaceAfter=4 * mm
        ),
        "subheading": ParagraphStyle(
            "Subheading",
            parent=base["Heading3"],
            fontSize=13,
            textColor=COLOR_GRAY
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontSize=20,
            textColor=COLOR_PRIMARY,
            alignment=TA_CENTER
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontSize=10,
            textColor=COLOR_GRAY,
            alignment=TA_CENTER
        ),
        "info_box": ParagraphStyle(
            "InfoBox",
            parent=base["Normal"],
            fontSize=10,
            textColor=COLOR_GRAY,
            backColor=HexColor("#F8F9FA"),
            borderPadding=5
        ),
        "warning": ParagraphStyle(
            "Warning",
            parent=base["Normal"],
            fontSize=10,
            textColor=HexColor("#856404"),
            backColor=HexColor("#FFF3CD"),
            borderPadding=8
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=COLOR_GRAY,
            alignment=TA_RIGHT
        )
    }
    
    return styles


def _build_header(metadata: Dict, styles: Dict) -> List:
    """Build report header with title, date, and session ID."""
    elements = []
    
    test_date = metadata.get("test_date", datetime.now().strftime("%Y-%m-%d"))
    session_id = metadata.get("session_id", "N/A")[:8]
    method_version = metadata.get("method_version", "1.0.0")
    
    elements.append(Paragraph("Raport z Testu Ramp", styles["title"]))
    elements.append(Paragraph(f"Data testu: {test_date}", styles["subtitle"]))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        f"ID: {session_id} | Wersja metody: v{method_version}",
        styles["subtitle"]
    ))
    
    return elements


def _build_confidence_badge(confidence: Dict, styles: Dict) -> List:
    """Build confidence score display."""
    elements = []
    
    score = confidence.get("overall", 0)
    score_pct = int(score * 100)
    
    if score >= 0.75:
        badge_color = COLOR_CONFIDENCE_OK
        badge_text = "Wysoka pewność"
    elif score >= 0.5:
        badge_color = COLOR_CONFIDENCE_WARN
        badge_text = "Umiarkowana pewność"
    else:
        badge_color = HexColor("#E74C3C")
        badge_text = "Niska pewność"
    
    # Create badge table
    data = [[
        Paragraph(f"<b>Confidence Score: {score_pct}%</b>", styles["body"]),
        Paragraph(f"<b>{badge_text}</b>", styles["body"])
    ]]
    
    table = Table(data, colWidths=[80 * mm, 80 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F8F9FA")),
        ("TEXTCOLOR", (1, 0), (1, 0), badge_color),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, COLOR_GRAY),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]),
    ]))
    
    elements.append(table)
    
    return elements


def _build_results_summary(thresholds: Dict, cp_model: Dict, metadata: Dict, styles: Dict) -> List:
    """Build key results summary table."""
    elements = []
    
    elements.append(Paragraph("Kluczowe Wyniki", styles["heading"]))
    
    vt1 = thresholds.get("vt1_watts", 0)
    vt2 = thresholds.get("vt2_watts", 0)
    pmax = thresholds.get("pmax_watts", 0)
    vo2max = metadata.get("vo2max_est", 0)
    cp = cp_model.get("cp_watts", 0)
    
    data = [
        ["Parametr", "Wartość", "Interpretacja"],
        ["VT1 (Próg tlenowy)", f"{vt1} W", "Strefa komfortowa"],
        ["VT2 (Próg beztlenowy)", f"{vt2} W", "Strefa wysiłku"],
        ["Zakres VT1–VT2", f"{vt1}–{vt2} W", "Strefa tempo/threshold"],
        ["Moc maksymalna (Pmax)", f"{pmax} W", "Szczyt testu"],
        ["VO₂max (szacowany)", f"{vo2max:.1f} ml/kg/min", "Wydolność tlenowa"],
        ["CP (Critical Power)", f"{cp} W", "Moc progowa"],
    ]
    
    table = Table(data, colWidths=[55 * mm, 45 * mm, 60 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F8F9FA")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(table)
    
    return elements


def _build_thresholds_section(thresholds: Dict, styles: Dict) -> List:
    """Build detailed thresholds section."""
    elements = []
    
    elements.append(Paragraph("Szczegóły Progów", styles["heading"]))
    
    vt1 = thresholds.get("vt1_watts", 0)
    vt2 = thresholds.get("vt2_watts", 0)
    vt1_hr = thresholds.get("vt1_hr", 0)
    vt2_hr = thresholds.get("vt2_hr", 0)
    vt1_ve = thresholds.get("vt1_ve", 0)
    vt2_ve = thresholds.get("vt2_ve", 0)
    
    # Descriptive text
    elements.append(Paragraph(
        "Progi zostały wykryte na podstawie zmian w wentylacji (oddychaniu) podczas testu. "
        "<b>VT1</b> oznacza moment, gdy organizm zaczyna intensywniej pracować. "
        "<b>VT2</b> to punkt, powyżej którego wysiłek staje się bardzo ciężki.",
        styles["body"]
    ))
    elements.append(Spacer(1, 5 * mm))
    
    data = [
        ["Próg", "Moc [W]", "HR [bpm]", "VE [L/min]"],
        ["VT1", f"{vt1}", f"{vt1_hr}" if vt1_hr else "–", f"{vt1_ve:.1f}" if vt1_ve else "–"],
        ["VT2", f"{vt2}", f"{vt2_hr}" if vt2_hr else "–", f"{vt2_ve:.1f}" if vt2_ve else "–"],
    ]
    
    table = Table(data, colWidths=[40 * mm, 40 * mm, 40 * mm, 40 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (0, 1), COLOR_VT1),
        ("BACKGROUND", (0, 2), (0, 2), COLOR_VT2),
        ("TEXTCOLOR", (0, 1), (0, 2), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRAY),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(table)
    
    return elements


def _build_cp_section(cp_model: Dict, styles: Dict) -> List:
    """Build CP/W' section."""
    elements = []
    
    elements.append(Paragraph("Critical Power & W'", styles["heading"]))
    
    cp = cp_model.get("cp_watts", 0)
    w_prime = cp_model.get("w_prime_joules", 0)
    weight = cp_model.get("rider_weight", 70)
    
    cp_kg = cp / weight if weight > 0 else 0
    w_prime_kj = w_prime / 1000
    
    elements.append(Paragraph(
        "Krzywa mocy pokazuje, jak długo możesz utrzymać dany poziom wysiłku. "
        "<b>CP</b> to moc, którą teoretycznie możesz utrzymać bardzo długo. "
        "<b>W'</b> to Twoja rezerwa energetyczna powyżej CP.",
        styles["body"]
    ))
    elements.append(Spacer(1, 5 * mm))
    
    data = [
        ["Parametr", "Wartość", "Znaczenie"],
        ["CP", f"{cp} W", "Moc „długotrwała”"],
        ["CP/kg", f"{cp_kg:.2f} W/kg", "Względna wydolność"],
        ["W'", f"{w_prime_kj:.1f} kJ", "Rezerwa anaerobowa"],
    ]
    
    table = Table(data, colWidths=[50 * mm, 50 * mm, 60 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F8F9FA")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(table)
    
    return elements


def _build_chart_section(chart_path: str, title: str, styles: Dict) -> List:
    """Build section with embedded chart."""
    elements = []
    
    elements.append(Paragraph(title, styles["subheading"]))
    elements.append(Spacer(1, 3 * mm))
    
    if Path(chart_path).exists():
        img = Image(chart_path, width=160 * mm, height=95 * mm)
        elements.append(img)
    else:
        elements.append(Paragraph(f"[Wykres niedostępny: {chart_path}]", styles["info_box"]))
    
    elements.append(Spacer(1, 5 * mm))
    
    return elements


def _build_interpretation_section(thresholds: Dict, cp_model: Dict, styles: Dict) -> List:
    """Build 'What do these results mean' section."""
    elements = []
    
    elements.append(Paragraph("Co oznaczają te wyniki?", styles["heading"]))
    
    vt1 = thresholds.get("vt1_watts", 0)
    vt2 = thresholds.get("vt2_watts", 0)
    cp = cp_model.get("cp_watts", 0)
    
    text = f"""
    <b>Twój próg tlenowy (VT1) wynosi {vt1} W.</b> To moc, przy której możesz jechać 
    komfortowo przez wiele godzin. Oddychasz spokojnie, możesz rozmawiać.<br/><br/>
    
    <b>Twój próg beztlenowy (VT2) wynosi {vt2} W.</b> Powyżej tej mocy wysiłek staje się 
    bardzo wymagający. Oddychasz ciężko, nie możesz swobodnie mówić.<br/><br/>
    
    <b>Strefa między {vt1} a {vt2} W</b> to Twoja strefa „tempo" – idealna do treningu 
    wytrzymałościowego i poprawy progu.<br/><br/>
    
    <b>CP ({cp} W)</b> to matematyczne przybliżenie Twojej mocy progowej. Możesz używać 
    tej wartości do planowania interwałów i wyznaczania stref treningowych.
    """
    
    elements.append(Paragraph(text, styles["body"]))
    
    return elements


def _build_limitations_section(styles: Dict) -> List:
    """Build limitations and disclaimers section."""
    elements = []
    
    elements.append(Paragraph("⚠️ Ograniczenia interpretacji", styles["heading"]))
    
    limitations = [
        "<b>To nie jest badanie medyczne.</b> Wyniki są szacunkami algorytmicznymi, "
        "nie pomiarami laboratoryjnymi.",
        
        "<b>Dokładność zależy od jakości danych.</b> Niepoprawna kalibracja czujników "
        "może wpłynąć na wyniki.",
        
        "<b>Progi są przybliżeniami.</b> VT1/VT2 wykryte algorytmicznie mogą się różnić "
        "od wyników testu spirometrycznego.",
        
        "<b>Wyniki są jednorazowe.</b> Wydolność zmienia się w czasie – powtarzaj testy "
        "co 6-8 tygodni.",
        
        "<b>Skonsultuj się z trenerem.</b> Przed wprowadzeniem zmian w treningu skonsultuj "
        "wyniki z wykwalifikowanym specjalistą.",
    ]
    
    for i, limitation in enumerate(limitations, 1):
        elements.append(Paragraph(f"{i}. {limitation}", styles["body"]))
        elements.append(Spacer(1, 2 * mm))
    
    return elements


def _build_training_zones(thresholds: Dict, styles: Dict) -> List:
    """Build training zones table."""
    elements = []
    
    elements.append(Paragraph("Rekomendowane Strefy Treningowe", styles["heading"]))
    
    vt1 = thresholds.get("vt1_watts", 200)
    vt2 = thresholds.get("vt2_watts", 280)
    
    # Calculate zones
    z1_max = int(vt1 * 0.8)
    z2_max = int(vt1)
    z3_max = int(vt2)
    z4_max = int(vt2 * 1.05)
    
    data = [
        ["Strefa", "Zakres [W]", "Opis", "Cel"],
        ["Z1 Recovery", f"< {z1_max}", "Bardzo łatwy", "Regeneracja"],
        ["Z2 Endurance", f"{z1_max}–{z2_max}", "Komfortowy", "Baza tlenowa"],
        ["Z3 Tempo", f"{z2_max}–{z3_max}", "Umiarkowany", "Próg"],
        ["Z4 Threshold", f"{z3_max}–{z4_max}", "Ciężki", "Wytrzymałość"],
        ["Z5 VO₂max", f"> {z4_max}", "Maksymalny", "Kapacytacja"],
    ]
    
    table = Table(data, colWidths=[35 * mm, 35 * mm, 40 * mm, 50 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F8F9FA")]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    
    elements.append(table)
    
    return elements


def _add_footer(canvas, doc):
    """Add footer to each page."""
    canvas.saveState()
    
    page_num = canvas.getPageNumber()
    footer_text = f"Strona {page_num} | Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Tri_Dashboard"
    
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(COLOR_GRAY)
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, footer_text)
    
    canvas.restoreState()
