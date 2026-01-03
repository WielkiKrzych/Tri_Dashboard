"""
PDF Layout Module.

Defines page structure and content sections for Ramp Test PDF.
Each page is a separate function.
No physiological calculations - only layout and formatting.

Per specification: methodology/ramp_test/10_pdf_layout.md
"""
from reportlab.platypus import (
    Paragraph, Spacer, Table, Image, PageBreak, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
import os
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS, PAGE_WIDTH, MARGIN,
    create_styles, get_table_style, get_card_style,
    FONT_FAMILY_BOLD, FONT_SIZE_BODY,
)


# ============================================================================
# PAGE 1: OKŁADKA / PODSUMOWANIE
# ============================================================================

def build_page_cover(
    metadata: Dict[str, Any],
    thresholds: Dict[str, Any],
    cp_model: Dict[str, Any],
    confidence: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict,
    is_conditional: bool = False
) -> List:
    """Build Page 1: Cover / Summary.
    
    Contains:
    - Title and metadata
    - Confidence badge
    - Conditional warning (if applicable)
    - Key results table
    - Ramp profile chart
    """
    elements = []
    
    # === HEADER ===
    test_date = metadata.get("test_date", "Unknown")
    session_id = metadata.get("session_id", "")[:8]
    method_version = metadata.get("method_version", "1.0.0")
    
    elements.append(Paragraph("Raport z Testu Ramp", styles["title"]))
    
    meta_text = f"Data: <b>{test_date}</b> | ID: {session_id} | v{method_version}"
    elements.append(Paragraph(meta_text, styles["center"]))
    elements.append(Spacer(1, 8 * mm))
    
    # === CONFIDENCE BADGE ===
    score = confidence.get("overall_confidence", 0)
    
    if score >= 0.75:
        badge_color = COLORS["success"]
        label = "Wysoka pewność"
    elif score >= 0.5:
        badge_color = COLORS["warning"]
        label = "Umiarkowana pewność"
    else:
        badge_color = COLORS["danger"]
        label = "Niska pewność"
    
    from reportlab.platypus import TableStyle
    badge_content = f"<b>Confidence Score:</b> {score:.0%} ({label})"
    badge_table = Table(
        [[Paragraph(badge_content, styles["body"])]],
        colWidths=[100 * mm]
    )
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["white"]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, badge_color),
    ]))
    elements.append(badge_table)
    elements.append(Spacer(1, 6 * mm))
    
    # === CONDITIONAL WARNING ===
    if is_conditional:
        warning_text = (
            "<b>⚠️ Test rozpoznany warunkowo</b><br/>"
            "Interpretacja obarczona zwiększoną niepewnością. "
            "Profil mocy lub czas kroków wykazują odchylenia od standardowego protokołu."
        )
        warning_table = Table(
            [[Paragraph(warning_text, styles["body"])]],
            colWidths=[160 * mm]
        )
        warning_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLORS["warning"]),
            ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["text"]),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 2, COLORS["warning"]),
        ]))
        elements.append(warning_table)
        elements.append(Spacer(1, 6 * mm))
    
    # === KEY RESULTS TABLE ===
    elements.append(Paragraph("Kluczowe Wyniki", styles["heading"]))
    
    vt1_watts = thresholds.get("vt1_watts", "brak danych")
    vt2_watts = thresholds.get("vt2_watts", "brak danych")
    cp_watts = cp_model.get("cp_watts", "brak danych")
    w_prime_kj = cp_model.get("w_prime_kj", "brak danych")
    pmax = metadata.get("pmax_watts", "brak danych")
    
    # Calculate VT1-VT2 range
    if vt1_watts != "brak danych" and vt2_watts != "brak danych":
        vt_range = f"{vt1_watts}–{vt2_watts}"
    else:
        vt_range = "brak danych"
    
    data = [
        ["Parametr", "Wartość", "Interpretacja"],
        ["VT1 (Próg tlenowy)", f"{vt1_watts} W", "Strefa komfortowa"],
        ["VT2 (Próg beztlenowy)", f"{vt2_watts} W", "Strefa wysiłku"],
        ["Zakres VT1–VT2", f"{vt_range} W", "Strefa tempo/threshold"],
        ["Moc maksymalna (Pmax)", f"{pmax} W", "Szczyt testu"],
        ["Critical Power (CP)", f"{cp_watts} W", "Moc progowa"],
        ["W' (Rezerwa)", f"{w_prime_kj} kJ", "Rezerwa anaerobowa"],
    ]
    
    table = Table(data, colWidths=[55 * mm, 35 * mm, 55 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === RAMP PROFILE CHART ===
    if figure_paths and "ramp_profile" in figure_paths:
        elements.extend(_build_chart(figure_paths["ramp_profile"], "Przebieg Testu", styles))
    
    return elements


# ============================================================================
# PAGE 2: SZCZEGÓŁY PROGÓW VT1/VT2
# ============================================================================

def build_page_thresholds(
    thresholds: Dict[str, Any],
    smo2: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 2: Threshold Details.
    
    Contains:
    - VT1/VT2 explanation
    - Thresholds table with HR/VE
    - SmO2 vs Power chart
    - SmO2 supporting signal note
    """
    elements = []
    
    elements.append(Paragraph("Szczegóły Progów VT1 / VT2", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # === EXPLANATION ===
    elements.append(Paragraph(
        "Progi zostały wykryte na podstawie zmian w wentylacji (oddychaniu) podczas testu.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>VT1 (Próg tlenowy):</b> Moment, gdy organizm zaczyna intensywniej pracować. "
        "Możesz jechać komfortowo przez wiele godzin.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>VT2 (Próg beztlenowy):</b> Punkt, powyżej którego wysiłek staje się bardzo ciężki. "
        "Oddychasz ciężko, nie możesz swobodnie mówić.",
        styles["body"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === THRESHOLDS TABLE ===
    elements.append(Paragraph("Tabela Progów", styles["heading"]))
    
    vt1_watts = thresholds.get("vt1_watts", "brak danych")
    vt1_range = thresholds.get("vt1_range_watts", "brak danych")
    vt1_hr = thresholds.get("vt1_hr", "brak danych")
    vt1_ve = thresholds.get("vt1_ve", "brak danych")
    
    vt2_watts = thresholds.get("vt2_watts", "brak danych")
    vt2_range = thresholds.get("vt2_range_watts", "brak danych")
    vt2_hr = thresholds.get("vt2_hr", "brak danych")
    vt2_ve = thresholds.get("vt2_ve", "brak danych")
    
    def format_thresh(mid, rng):
        if mid == "brak danych": return mid
        if rng == "brak danych": return f"~{mid}"
        return f"{rng} (środek: {mid})"

    data = [
        ["Próg", "Moc [W]", "HR [bpm]", "VE [L/min]"],
        ["VT1 (Próg tlenowy)", format_thresh(vt1_watts, vt1_range), f"{vt1_hr}", f"{vt1_ve}"],
        ["VT2 (Próg beztlenowy)", format_thresh(vt2_watts, vt2_range), f"{vt2_hr}", f"{vt2_ve}"],
    ]
    
    table = Table(data, colWidths=[50 * mm, 35 * mm, 35 * mm, 35 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === SMO2 CHART ===
    if figure_paths and "smo2_power" in figure_paths:
        elements.extend(_build_chart(figure_paths["smo2_power"], "SmO₂ vs Moc", styles))
    
    # === SMO2 ANALYSIS ===
    elements.append(Paragraph("Analiza SmO₂ (Lokalna)", styles["subheading"]))
    drop_point = smo2.get("drop_point_watts", "brak danych")
    interpretation = smo2.get("interpretation", "nie przeanalizowano")
    
    smo2_text = (
        f"<b>Punkt spadku SmO₂:</b> {drop_point} W<br/>"
        f"<b>Interpretacja:</b> {interpretation}<br/><br/>"
        "<i>ℹ️ SmO₂ LT1/LT2 są sygnałem wspierającym. "
        "Nie zastępują progów wentylacyjnych, ale pomagają je potwierdzić.</i>"
    )
    elements.append(Paragraph(smo2_text, styles["body"]))
    
    return elements


# ============================================================================
# PAGE 3: POWER-DURATION CURVE / CP
# ============================================================================

def build_page_pdc(
    cp_model: Dict[str, Any],
    metadata: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 3: Power-Duration Curve and Critical Power.
    
    Contains:
    - PDC explanation
    - CP/W' table
    - PDC chart
    """
    elements = []
    
    elements.append(Paragraph("Krzywa Mocy i Critical Power", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # === EXPLANATION ===
    elements.append(Paragraph(
        "Krzywa mocy pokazuje, jak długo możesz utrzymać dany poziom wysiłku.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>CP (Critical Power)</b> to moc, którą teoretycznie możesz utrzymać bardzo długo. "
        "W praktyce oznacza to maksymalny wysiłek przez 30-60 minut.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>W' (W-prime)</b> to Twoja rezerwa energetyczna powyżej CP. "
        "Możesz ją „spalić” na ataki, podjazdy lub sprint.",
        styles["body"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === CP/W' TABLE ===
    elements.append(Paragraph("Parametry CP", styles["heading"]))
    
    cp_watts = cp_model.get("cp_watts", "brak danych")
    w_prime_kj = cp_model.get("w_prime_kj", "brak danych")
    
    # Calculate CP/kg if weight available
    athlete_weight = metadata.get("athlete_weight_kg", 0)
    if athlete_weight and cp_watts != "brak danych":
        try:
            cp_per_kg = f"{float(cp_watts) / athlete_weight:.2f}"
        except (ValueError, TypeError):
            cp_per_kg = "brak danych"
    else:
        cp_per_kg = "brak danych"
    
    data = [
        ["Parametr", "Wartość", "Znaczenie"],
        ["CP", f"{cp_watts} W", "Moc „długotrwała”"],
        ["CP/kg", f"{cp_per_kg} W/kg", "Względna wydolność"],
        ["W'", f"{w_prime_kj} kJ", "Rezerwa anaerobowa"],
    ]
    
    table = Table(data, colWidths=[45 * mm, 45 * mm, 55 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === PDC CHART ===
    if figure_paths and "pdc" in figure_paths:
        elements.extend(_build_chart(figure_paths["pdc"], "Power-Duration Curve", styles))
    
    return elements


# ============================================================================
# PAGE 4: INTERPRETACJA WYNIKÓW
# ============================================================================

def build_page_interpretation(
    thresholds: Dict[str, Any],
    cp_model: Dict[str, Any],
    styles: Dict
) -> List:
    """Build Page 4: Results Interpretation.
    
    Contains:
    - VT1 explanation with values
    - VT2 explanation with values
    - Tempo zone explanation
    - CP practical usage
    """
    elements = []
    
    elements.append(Paragraph("Co oznaczają te wyniki?", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    vt1_watts = thresholds.get("vt1_watts", "brak danych")
    vt2_watts = thresholds.get("vt2_watts", "brak danych")
    cp_watts = cp_model.get("cp_watts", "brak danych")
    
    # === VT1 ===
    elements.append(Paragraph("Próg tlenowy (VT1)", styles["heading"]))
    elements.append(Paragraph(
        f"Twój próg tlenowy wynosi <b>{vt1_watts} W</b>. "
        "To moc, przy której możesz jechać komfortowo przez wiele godzin. "
        "Oddychasz spokojnie, możesz swobodnie rozmawiać. "
        "Treningi poniżej VT1 budują bazę tlenową i służą regeneracji.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # === VT2 ===
    elements.append(Paragraph("Próg beztlenowy (VT2)", styles["heading"]))
    elements.append(Paragraph(
        f"Twój próg beztlenowy wynosi <b>{vt2_watts} W</b>. "
        "Powyżej tej mocy wysiłek staje się bardzo wymagający. "
        "Oddychasz ciężko, nie możesz swobodnie mówić. "
        "Treningi powyżej VT2 rozwijają VO₂max, ale wymagają pełnej regeneracji.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # === TEMPO ZONE ===
    elements.append(Paragraph("Strefa Tempo", styles["heading"]))
    elements.append(Paragraph(
        f"Strefa między <b>{vt1_watts}</b> a <b>{vt2_watts} W</b> to Twoja strefa „tempo”. "
        "Jest idealna do treningu wytrzymałościowego i poprawy progu. "
        "W tej strefie możesz spędzać znaczną część czasu treningowego bez nadmiernego zmęczenia.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # === CP ===
    elements.append(Paragraph("Critical Power", styles["heading"]))
    elements.append(Paragraph(
        f"CP ({cp_watts} W) to matematyczne przybliżenie Twojej mocy progowej. "
        "Możesz używać tej wartości do planowania interwałów i wyznaczania stref treningowych. "
        "CP jest przydatne do pacing'u podczas zawodów i długich treningów.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# PAGE 5: STREFY TRENINGOWE
# ============================================================================

def build_page_zones(
    thresholds: Dict[str, Any],
    styles: Dict
) -> List:
    """Build Page 5: Training Zones.
    
    Contains:
    - Zones table based on VT1/VT2
    - Zone descriptions
    """
    elements = []
    
    elements.append(Paragraph("Rekomendowane Strefy Treningowe", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    vt1_raw = thresholds.get("vt1_watts", "brak danych")
    vt2_raw = thresholds.get("vt2_watts", "brak danych")
    
    # Parse numbers for zone calculation
    try:
        vt1 = float(vt1_raw) if vt1_raw != "brak danych" else 0
        vt2 = float(vt2_raw) if vt2_raw != "brak danych" else 0
    except (ValueError, TypeError):
        vt1 = 0
        vt2 = 0
    
    # Calculate zones
    if vt1 and vt2:
        z1_max = int(vt1 * 0.8)
        z2_min = z1_max
        z2_max = int(vt1)
        z3_min = z2_max
        z3_max = int(vt2)
        z4_min = z3_max
        z4_max = int(vt2 * 1.05)
        z5_min = z4_max
        
        data = [
            ["Strefa", "Zakres [W]", "Opis", "Cel treningowy"],
            ["Z1 Recovery", f"< {z1_max}", "Bardzo łatwy", "Regeneracja"],
            ["Z2 Endurance", f"{z2_min}–{z2_max}", "Komfortowy", "Baza tlenowa"],
            ["Z3 Tempo", f"{z3_min}–{z3_max}", "Umiarkowany", "Próg"],
            ["Z4 Threshold", f"{z4_min}–{z4_max}", "Ciężki", "Wytrzymałość"],
            ["Z5 VO₂max", f"> {z5_min}", "Maksymalny", "Kapacytacja"],
        ]
    else:
        data = [
            ["Strefa", "Zakres [W]", "Opis", "Cel treningowy"],
            ["Z1 Recovery", "-", "Bardzo łatwy", "Regeneracja"],
            ["Z2 Endurance", "-", "Komfortowy", "Baza tlenowa"],
            ["Z3 Tempo", "-", "Umiarkowany", "Próg"],
            ["Z4 Threshold", "-", "Ciężki", "Wytrzymałość"],
            ["Z5 VO₂max", "-", "Maksymalny", "Kapacytacja"],
        ]
    
    table = Table(data, colWidths=[35 * mm, 35 * mm, 35 * mm, 40 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === USAGE NOTE ===
    elements.append(Paragraph(
        "Powyższe strefy są obliczone automatycznie na podstawie wykrytych progów VT1 i VT2. "
        "Przed zastosowaniem skonsultuj je z trenerem, który może dostosować je do Twoich celów.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# PAGE 6: OGRANICZENIA INTERPRETACJI
# ============================================================================

def build_page_limitations(
    styles: Dict,
    is_conditional: bool = False
) -> List:
    """Build Page 6: Interpretation Limitations.
    
    Contains:
    - All mandatory disclaimers
    - Conditional warning (if applicable)
    """
    elements = []
    
    elements.append(Paragraph("⚠️ Ograniczenia interpretacji", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    limitations = [
        ("<b>1. To nie jest badanie medyczne.</b>", 
         "Wyniki są szacunkami algorytmicznymi, nie pomiarami laboratoryjnymi. "
         "Nie służą do diagnozowania stanów zdrowotnych."),
        
        ("<b>2. Dokładność zależy od jakości danych.</b>", 
         "Niepoprawna kalibracja czujników, artefakty ruchu lub niestabilność sygnału "
         "mogą wpłynąć na wyniki."),
        
        ("<b>3. Progi są przybliżeniami.</b>", 
         "VT1/VT2 wykryte algorytmicznie mogą się różnić od wyników testu "
         "spirometrycznego w laboratorium."),
        
        ("<b>4. Wyniki są jednorazowe.</b>", 
         "Wydolność zmienia się w czasie – powtarzaj testy co 6-8 tygodni, "
         "aby śledzić postępy."),
        
        ("<b>5. SmO₂ to sygnał wspierający.</b>", 
         "LT1/LT2 z SmO₂ nie zastępują progów wentylacyjnych. "
         "Służą do dodatkowej walidacji."),
        
        ("<b>6. Skonsultuj się z trenerem.</b>", 
         "Przed wprowadzeniem zmian w treningu skonsultuj wyniki "
         "z wykwalifikowanym specjalistą."),
    ]
    
    for title, description in limitations:
        elements.append(Paragraph(title, styles["body"]))
        elements.append(Paragraph(description, styles["body"]))
        elements.append(Spacer(1, 3 * mm))
    
    # === CONDITIONAL WARNING ===
    if is_conditional:
        elements.append(Spacer(1, 4 * mm))
        from reportlab.platypus import TableStyle
        warning_text = (
            "<b>⚠️ Ten raport został wygenerowany dla testu rozpoznanego warunkowo.</b><br/>"
            "Profil mocy lub czas kroków wykazują odchylenia od standardowego protokołu Ramp Test. "
            "Interpretacja jest obarczona zwiększoną niepewnością."
        )
        warning_table = Table(
            [[Paragraph(warning_text, styles["body"])]],
            colWidths=[160 * mm]
        )
        warning_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLORS["warning"]),
            ("TEXTCOLOR", (0, 0), (-1, -1), COLORS["text"]),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 2, COLORS["warning"]),
        ]))
        elements.append(warning_table)
    
    return elements


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _build_chart(chart_path: str, title: str, styles: Dict) -> List:
    """Build a chart section with image.
    
    Args:
        chart_path: Path to chart PNG file
        title: Section title
        styles: Paragraph styles dictionary
        
    Returns:
        List of flowables
    """
    elements = []
    
    elements.append(Paragraph(title, styles["subheading"]))
    
    try:
        available_width = PAGE_WIDTH - 2 * MARGIN
        
        if not chart_path or not os.path.exists(chart_path):
            raise FileNotFoundError(f"Plik wykresu nie istnieje: {chart_path}")
            
        img = Image(chart_path)
        
        # Scale to fit width
        aspect = img.imageHeight / img.imageWidth
        img_width = min(available_width, 150 * mm)
        img_height = img_width * aspect
        
        # Limit height
        if img_height > 90 * mm:
            img_height = 90 * mm
            img_width = img_height / aspect
        
        img.drawWidth = img_width
        img.drawHeight = img_height
        
        elements.append(img)
    except Exception as e:
        elements.append(Paragraph(f"[Wykres niedostępny: {e}]", styles["small"]))
    
    return elements
