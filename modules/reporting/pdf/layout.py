"""
PDF Layout Module.

Defines page structure and content sections for Ramp Test PDF.
Each function builds a specific section of the report.
No physiological calculations - only layout and formatting.
"""
from reportlab.platypus import (
    Paragraph, Spacer, Table, Image, PageBreak, KeepTogether
)
from reportlab.lib.units import mm
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS, PAGE_WIDTH, MARGIN,
    create_styles, get_table_style, get_card_style,
)


def build_header(metadata: Dict[str, Any], styles: Dict) -> List:
    """Build report header section.
    
    Args:
        metadata: Report metadata from JSON
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for the header
    """
    elements = []
    
    test_date = metadata.get("test_date", "Unknown")
    session_id = metadata.get("session_id", "")[:8]
    method_version = metadata.get("method_version", "1.0.0")
    
    # Title
    elements.append(Paragraph("Raport z Testu Ramp", styles["title"]))
    
    # Metadata line
    meta_text = f"Data: {test_date} | ID: {session_id} | v{method_version}"
    elements.append(Paragraph(meta_text, styles["center"]))
    
    return elements


def build_confidence_badge(confidence: Dict[str, Any], styles: Dict) -> List:
    """Build confidence score badge.
    
    Args:
        confidence: Confidence data from JSON
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for the badge
    """
    elements = []
    
    score = confidence.get("overall_score", 0)
    level = confidence.get("level", "unknown")
    
    # Determine color based on score
    if score >= 0.75:
        badge_color = COLORS["success"]
        label = "Wysoka pewność"
    elif score >= 0.5:
        badge_color = COLORS["warning"]
        label = "Umiarkowana pewność"
    else:
        badge_color = COLORS["danger"]
        label = "Niska pewność"
    
    # Build badge as table
    badge_content = f"<b>Confidence:</b> {score:.0%} ({label})"
    badge_table = Table(
        [[Paragraph(badge_content, styles["body"])]],
        colWidths=[120 * mm]
    )
    badge_table.setStyle(get_card_style())
    
    elements.append(badge_table)
    
    return elements


def build_conditional_warning(styles: Dict) -> List:
    """Build warning block for conditional ramp test.
    
    Args:
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for the warning
    """
    elements = []
    
    warning_text = (
        "<b>⚠️ Test rozpoznany warunkowo</b><br/>"
        "Interpretacja obarczona zwiększoną niepewnością. "
        "Profil mocy lub czas kroków wykazują odchylenia od standardowego protokołu Ramp Test. "
        "Skonsultuj wyniki z trenerem lub powtórz test w bardziej kontrolowanych warunkach."
    )
    
    from reportlab.platypus import TableStyle
    warning_table = Table(
        [[Paragraph(warning_text, styles["body"])]],
        colWidths=[160 * mm]
    )
    warning_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLORS["warning"]),
        ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["text"]),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 2, COLORS["warning"]),
    ]))
    
    elements.append(warning_table)
    
    return elements


def build_results_summary(thresholds: Dict, cp_model: Dict, metadata: Dict, styles: Dict) -> List:
    """Build key results summary table.
    
    Args:
        thresholds: Threshold data from JSON
        cp_model: CP model data from JSON
        metadata: Metadata from JSON
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for the summary
    """
    elements = []
    
    elements.append(Paragraph("Kluczowe Wyniki", styles["heading"]))
    
    vt1_watts = thresholds.get("vt1_watts", "-")
    vt2_watts = thresholds.get("vt2_watts", "-")
    cp_watts = cp_model.get("cp_watts", "-")
    w_prime = cp_model.get("w_prime_joules", 0)
    w_prime_kj = f"{w_prime/1000:.1f}" if w_prime else "-"
    
    # Build summary table
    data = [
        ["Parametr", "Wartość", "Opis"],
        ["VT1 (Próg tlenowy)", f"{vt1_watts} W", "Strefa komfortowa"],
        ["VT2 (Próg beztlenowy)", f"{vt2_watts} W", "Strefa wysiłku"],
        ["Critical Power (CP)", f"{cp_watts} W", "Moc progowa"],
        ["W' (Rezerwa)", f"{w_prime_kj} kJ", "Rezerwa anaerobowa"],
    ]
    
    table = Table(data, colWidths=[50 * mm, 40 * mm, 60 * mm])
    table.setStyle(get_table_style())
    
    elements.append(table)
    
    return elements


def build_chart_section(chart_path: str, title: str, styles: Dict) -> List:
    """Build a section with a chart image.
    
    Args:
        chart_path: Path to chart PNG file
        title: Section title
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for the chart section
    """
    elements = []
    
    elements.append(Paragraph(title, styles["heading"]))
    
    try:
        # Calculate image dimensions to fit page
        available_width = PAGE_WIDTH - 2 * MARGIN
        img = Image(chart_path)
        
        # Scale to fit width while maintaining aspect ratio
        aspect = img.imageHeight / img.imageWidth
        img_width = min(available_width, 160 * mm)
        img_height = img_width * aspect
        
        # Limit height
        if img_height > 100 * mm:
            img_height = 100 * mm
            img_width = img_height / aspect
        
        img.drawWidth = img_width
        img.drawHeight = img_height
        
        elements.append(img)
    except Exception as e:
        elements.append(Paragraph(f"[Wykres niedostępny: {e}]", styles["small"]))
    
    return elements


def build_thresholds_section(thresholds: Dict, styles: Dict) -> List:
    """Build detailed thresholds section.
    
    Args:
        thresholds: Threshold data from JSON
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for thresholds section
    """
    elements = []
    
    elements.append(Paragraph("Szczegóły Progów VT1 / VT2", styles["heading"]))
    
    # Explanation text
    elements.append(Paragraph(
        "Progi zostały wykryte na podstawie zmian w wentylacji podczas testu. "
        "VT1 oznacza moment, gdy organizm zaczyna intensywniej pracować. "
        "VT2 to punkt, powyżej którego wysiłek staje się bardzo ciężki.",
        styles["body"]
    ))
    
    # Thresholds table
    vt1_watts = thresholds.get("vt1_watts", "-")
    vt1_hr = thresholds.get("vt1_hr", "-")
    vt2_watts = thresholds.get("vt2_watts", "-")
    vt2_hr = thresholds.get("vt2_hr", "-")
    
    data = [
        ["Próg", "Moc [W]", "HR [bpm]"],
        ["VT1 (Próg tlenowy)", f"{vt1_watts}", f"{vt1_hr}"],
        ["VT2 (Próg beztlenowy)", f"{vt2_watts}", f"{vt2_hr}"],
    ]
    
    table = Table(data, colWidths=[60 * mm, 40 * mm, 40 * mm])
    table.setStyle(get_table_style())
    
    elements.append(table)
    
    return elements


def build_limitations_section(styles: Dict) -> List:
    """Build interpretation limitations section.
    
    Args:
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for limitations section
    """
    elements = []
    
    elements.append(Paragraph("⚠️ Ograniczenia interpretacji", styles["heading"]))
    
    limitations = [
        "<b>1. To nie jest badanie medyczne.</b> Wyniki są szacunkami algorytmicznymi, nie pomiarami laboratoryjnymi.",
        "<b>2. Dokładność zależy od jakości danych.</b> Niepoprawna kalibracja czujników może wpłynąć na wyniki.",
        "<b>3. Progi są przybliżeniami.</b> VT1/VT2 wykryte algorytmicznie mogą różnić się od wyników laboratoryjnych.",
        "<b>4. Wyniki są jednorazowe.</b> Wydolność zmienia się w czasie – powtarzaj testy co 6-8 tygodni.",
        "<b>5. SmO₂ to sygnał wspierający.</b> LT1/LT2 z SmO₂ nie zastępują progów wentylacyjnych.",
        "<b>6. Skonsultuj się z trenerem.</b> Przed zmianami w treningu skonsultuj wyniki ze specjalistą.",
    ]
    
    for limitation in limitations:
        elements.append(Paragraph(limitation, styles["body"]))
        elements.append(Spacer(1, 2 * mm))
    
    return elements


def build_footer(metadata: Dict, styles: Dict) -> List:
    """Build page footer.
    
    Args:
        metadata: Report metadata from JSON
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables for footer
    """
    elements = []
    
    session_id = metadata.get("session_id", "")[:8]
    method_version = metadata.get("method_version", "1.0.0")
    
    footer_text = f"ID: {session_id} | v{method_version} | Tri_Dashboard"
    elements.append(Paragraph(footer_text, styles["footer"]))
    
    return elements
