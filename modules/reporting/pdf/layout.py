"""
PDF Layout Module.

Defines page structure and content sections for Ramp Test PDF.
Each page is a separate function.
No physiological calculations - only layout and formatting.

Per specification: methodology/ramp_test/10_pdf_layout.md
"""
from reportlab.platypus import (
    Paragraph, Spacer, Table, Image, PageBreak, KeepTogether, TableStyle
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
import os
import logging
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS, PAGE_WIDTH, MARGIN,
    create_styles, get_table_style, get_card_style,
    FONT_FAMILY_BOLD, FONT_SIZE_BODY,
)

# Setup logger
logger = logging.getLogger("Tri_Dashboard.PDFLayout")


# ============================================================================
# PAGE 0: EXECUTIVE PHYSIO SUMMARY (NEW FIRST PAGE)
# ============================================================================

def build_page_executive_summary(
    executive_data: Dict[str, Any],
    metadata: Dict[str, Any],
    styles: Dict
) -> List:
    """Build Page 0: PREMIUM Executive Physio Summary.
    
    Commercial-grade layout with:
    - Hero Header with status badge
    - Physiological Verdict Card
    - Signal Agreement Matrix
    - Test Confidence Panel
    - Training Decision Cards
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT
    
    elements = []
    
    limiter = executive_data.get("limiter", {})
    signal_matrix = executive_data.get("signal_matrix", {})
    confidence_panel = executive_data.get("confidence_panel", {})
    training_cards = executive_data.get("training_cards", [])
    
    test_date = metadata.get("test_date", "---")
    
    # ==========================================================================
    # 1. HERO HEADER
    # ==========================================================================
    
    limiter_color = HexColor(limiter.get("color", "#7F8C8D"))
    limiter_icon = limiter.get("icon", "⚖️")
    limiter_name = limiter.get("name", "NIEZNANY")
    
    # Title row
    elements.append(Paragraph(
        f"<font size='22'><b>EXECUTIVE PHYSIO SUMMARY</b></font>",
        styles["title"]
    ))
    
    # Status badge + date row
    status_text = f"{limiter_icon} {limiter_name}"
    header_table = Table([
        [
            Paragraph(f"<font color='{limiter.get('color', '#7F8C8D')}'><b>{status_text}</b></font>", styles["heading"]),
            Paragraph(f"<font size='10'>Data testu: <b>{test_date}</b></font>", styles["body"])
        ]
    ], colWidths=[100 * mm, 70 * mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 2. PHYSIOLOGICAL VERDICT CARD
    # ==========================================================================
    
    verdict = limiter.get("verdict", "Brak diagnozy")
    interpretation = limiter.get("interpretation", [])
    subtitle = limiter.get("subtitle", "")
    
    # Card content
    verdict_content = [
        Paragraph(f"<font size='14'><b>{limiter_icon} PRIMARY LIMITER: {limiter_name}</b></font>", styles["heading"]),
        Paragraph(f"<font size='10' color='#7F8C8D'>{subtitle}</font>", styles["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>{verdict}</b>", styles["body"]),
        Spacer(1, 2 * mm),
    ]
    
    for line in interpretation[:3]:
        verdict_content.append(Paragraph(f"• {line}", styles["body"]))
    
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
        ('BOX', (0, 0), (-1, -1), 2, limiter_color),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 3. SIGNAL AGREEMENT MATRIX
    # ==========================================================================
    
    elements.append(Paragraph("<b>SIGNAL AGREEMENT MATRIX</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    signals = signal_matrix.get("signals", [])
    agreement_idx = signal_matrix.get("agreement_index", 1.0)
    agreement_label = signal_matrix.get("agreement_label", "Wysoka")
    
    # Signal tiles
    signal_cells = []
    for sig in signals:
        status = sig.get("status", "ok")
        icon = sig.get("icon", "❓")
        name = sig.get("name", "?")
        note = sig.get("note", "")
        
        if status == "ok":
            bg_color = HexColor("#D5F5E3")
            status_label = "✓ OK"
        elif status == "warning":
            bg_color = HexColor("#FCF3CF")
            status_label = "⚠ WARNING"
        else:
            bg_color = HexColor("#FADBD8")
            status_label = "✗ CONFLICT"
        
        tile_content = [
            Paragraph(f"<font size='16'>{icon}</font>", styles["center"]),
            Paragraph(f"<b>{name}</b>", styles["center"]),
            Paragraph(f"<font size='8'>{status_label}</font>", styles["center"]),
        ]
        
        tile_table = Table([[tile_content]], colWidths=[52 * mm])
        tile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ]))
        signal_cells.append(tile_table)
    
    # Add conflict index tile
    idx_color = HexColor("#D5F5E3") if agreement_idx >= 0.8 else (HexColor("#FCF3CF") if agreement_idx >= 0.5 else HexColor("#FADBD8"))
    idx_content = [
        Paragraph(f"<font size='14'><b>{agreement_idx:.2f}</b></font>", styles["center"]),
        Paragraph("<font size='8'>Agreement Index</font>", styles["center"]),
        Paragraph(f"<font size='9'>{agreement_label}</font>", styles["center"]),
    ]
    idx_table = Table([[idx_content]], colWidths=[52 * mm])
    idx_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), idx_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
    ]))
    signal_cells.append(idx_table)
    
    # Horizontal layout for signal tiles
    if signal_cells:
        row_table = Table([signal_cells], colWidths=[55 * mm] * len(signal_cells))
        row_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(row_table)
    
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. TEST CONFIDENCE PANEL
    # ==========================================================================
    
    elements.append(Paragraph("<b>TEST CONFIDENCE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    overall_score = confidence_panel.get("overall_score", 0)
    breakdown = confidence_panel.get("breakdown", {})
    limiting_factor = confidence_panel.get("limiting_factor", "---")
    score_color = confidence_panel.get("color", "#7F8C8D")
    score_label = confidence_panel.get("label", "---")
    
    # Score display + breakdown
    score_para = Paragraph(
        f"<font size='28' color='{score_color}'><b>{overall_score}%</b></font> "
        f"<font size='12'>({score_label})</font>",
        styles["body"]
    )
    
    # Breakdown bars
    breakdown_rows = []
    for key, label in [("ve_stability", "VE Stability"), ("hr_lag", "HR Response"), ("smo2_noise", "SmO₂ Quality"), ("protocol_quality", "Protocol")]:
        val = breakdown.get(key, 50)
        bar_color = "#2ECC71" if val >= 70 else ("#F39C12" if val >= 50 else "#E74C3C")
        breakdown_rows.append([
            Paragraph(f"<font size='8'>{label}</font>", styles["body"]),
            Paragraph(f"<font size='9' color='{bar_color}'><b>{val}%</b></font>", styles["body"])
        ])
    
    breakdown_table = Table(breakdown_rows, colWidths=[35 * mm, 20 * mm])
    breakdown_table.setStyle(TableStyle([
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    confidence_row = Table([[score_para, breakdown_table]], colWidths=[60 * mm, 110 * mm])
    confidence_row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(confidence_row)
    elements.append(Paragraph(f"<font size='9' color='#7F8C8D'>Ogranicza: <b>{limiting_factor}</b></font>", styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 5. TRAINING DECISION CARDS
    # ==========================================================================
    
    elements.append(Paragraph("<b>TRAINING DECISIONS</b>", styles["subheading"]))
    elements.append(Spacer(1, 3 * mm))
    
    for i, card in enumerate(training_cards[:3], 1):
        strategy = card.get("strategy_name", "---")
        power = card.get("power_range", "---")
        volume = card.get("volume", "---")
        goal = card.get("adaptation_goal", "---")
        response = card.get("expected_response", "---")
        risk = card.get("risk_level", "low")
        
        risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
        risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
        
        card_content = [
            Paragraph(f"<font size='11'><b>{i}. {strategy}</b></font>", styles["heading"]),
            Paragraph(f"<b>Moc:</b> {power} | <b>Objętość:</b> {volume}", styles["body"]),
            Paragraph(f"<b>Cel:</b> {goal}", styles["body"]),
            Paragraph(f"<font size='9' color='#7F8C8D'>Expected: {response}</font>", styles["body"]),
            Paragraph(f"<font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
        ]
        
        card_table = Table([[card_content]], colWidths=[170 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
            ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(card_table)
        elements.append(Spacer(1, 2 * mm))
    
    return elements


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
    
    # === CONFIDENCE BADGE REMOVED per user request ===
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
        ["Zakres VT1–VT2", vt_range, "Strefa tempo/threshold"],
        ["Critical Power (CP)", f"{cp_watts} W", "Moc progowa"],
        ["W' (Rezerwa)", f"{w_prime_kj} kJ", "Rezerwa anaerobowa"]
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
        
    def fmt(val):
        if val == "brak danych": return val
        try:
            return f"{float(val):.0f}"
        except:
            return str(val)

    data = [
        ["Próg", "Moc [W]", "HR [bpm]", "VE [L/min]"],
        ["VT1 (Próg tlenowy)", format_thresh(vt1_watts, vt1_range), fmt(vt1_hr), fmt(vt1_ve)],
        ["VT2 (Próg beztlenowy)", format_thresh(vt2_watts, vt2_range), fmt(vt2_hr), fmt(vt2_ve)],
    ]
    
    table = Table(data, colWidths=[50 * mm, 50 * mm, 30 * mm, 30 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    # === EDUCATION BLOCK: VT1/VT2 ===
    elements.append(Spacer(1, 4 * mm))
    elements.extend(_build_education_block(
        "Dlaczego to ma znaczenie? (VT1 / VT2)",
        "Progi wentylacyjne to Twoje najważniejsze drogowskazy w planowaniu obciążeń. "
        "VT1 wyznacza granicę komfortu tlenowego i „przepalania” tłuszczy – to tu budujesz bazę na długie godziny. "
        "VT2 to Twój „szklany sufit” – powyżej niego kwas narasta szybciej niż organizm go utylizuje, "
        "co wymaga długiej regeneracji. Znajomość tych punktów pozwala unikać „strefy zgubnej” między progami, "
        "gdzie zmęczenie jest duże, a adaptacje nieoptymalne. Jako trener używam ich, by każda Twoja minuta "
        "na rowerze miała konkretny cel fizjologiczny. Dzięki temu nie trenujesz po prostu „ciężko”, "
        "ale trenujesz mądrze i precyzyjnie.",
        styles
    ))
    
    return elements


def build_page_smo2(smo2_data, smo2_manual, figure_paths, styles):
    """Build SmO2 analysis page - PREMIUM MUSCLE OXYGENATION DIAGNOSTIC."""
    from reportlab.lib.colors import HexColor
    
    elements = []
    smo2_advanced = smo2_data.get("advanced_metrics", {})
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='18'><b>MUSCLE OXYGENATION DIAGNOSTIC</b></font>",
        styles['title']
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Kliniczna analiza dostawy i wykorzystania tlenu</font>",
        styles['center']
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 1. METRIC CARDS ROW
    # ==========================================================================
    
    slope = smo2_advanced.get("slope_per_100w", 0) if smo2_advanced else 0
    halftime = smo2_advanced.get("halftime_reoxy_sec") if smo2_advanced else None
    coupling = smo2_advanced.get("hr_coupling_r", 0) if smo2_advanced else 0
    data_quality = smo2_advanced.get("data_quality", "unknown") if smo2_advanced else "unknown"
    
    def build_metric_card(title, value, unit, interpretation, color):
        card_content = [
            Paragraph(f"<font size='8' color='#7F8C8D'>{title}</font>", styles["center"]),
            Paragraph(f"<font size='16' color='{color}'><b>{value}</b></font>", styles["center"]),
            Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
            Spacer(1, 1 * mm),
            Paragraph(f"<font size='8'>{interpretation}</font>", styles["center"]),
        ]
        card_table = Table([[card_content]], colWidths=[55 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
            ('BOX', (0, 0), (-1, -1), 1, HexColor(color)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        return card_table
    
    slope_color = "#E74C3C" if slope < -6 else ("#F39C12" if slope < -3 else "#2ECC71")
    slope_interp = "Szybka desaturacja" if slope < -6 else ("Umiarkowana" if slope < -3 else "Stabilna")
    card1 = build_metric_card("DESATURATION RATE", f"{slope:.1f}", "%/100W", slope_interp, slope_color)
    
    if halftime:
        ht_color = "#E74C3C" if halftime > 60 else ("#F39C12" if halftime > 30 else "#2ECC71")
        ht_interp = "Wolna reoksygenacja" if halftime > 60 else ("Umiarkowana" if halftime > 30 else "Szybka")
        card2 = build_metric_card("REOXY HALF-TIME", f"{halftime:.0f}", "sekund", ht_interp, ht_color)
    else:
        card2 = build_metric_card("REOXY HALF-TIME", "---", "sekund", "Brak danych", "#7F8C8D")
    
    coup_color = "#3498DB" if abs(coupling) > 0.6 else ("#F39C12" if abs(coupling) > 0.3 else "#2ECC71")
    coup_interp = "Silna (centralna)" if abs(coupling) > 0.6 else ("Umiarkowana" if abs(coupling) > 0.3 else "Słaba (lokalna)")
    card3 = build_metric_card("HR COUPLING", f"{coupling:.2f}", "r-Pearson", coup_interp, coup_color)
    
    cards_row = Table([[card1, card2, card3]], colWidths=[58 * mm, 58 * mm, 58 * mm])
    cards_row.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(cards_row)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 2. OXYGEN DELIVERY MECHANISM PANEL
    # ==========================================================================
    
    limiter_type = smo2_advanced.get("limiter_type", "unknown") if smo2_advanced else "unknown"
    limiter_conf = smo2_advanced.get("limiter_confidence", 0) if smo2_advanced else 0
    interpretation_adv = smo2_advanced.get("interpretation", "") if smo2_advanced else ""
    
    mechanism_colors = {"local": "#3498DB", "central": "#E74C3C", "metabolic": "#F39C12", "unknown": "#7F8C8D"}
    mechanism_names = {"local": "PERIPHERAL", "central": "CENTRAL", "metabolic": "MIXED", "unknown": "UNDEFINED"}
    mechanism_icons = {"local": "💪", "central": "❤️", "metabolic": "🔥", "unknown": "❓"}
    
    mech_color = HexColor(mechanism_colors.get(limiter_type, "#7F8C8D"))
    mech_name = mechanism_names.get(limiter_type, "UNDEFINED")
    mech_icon = mechanism_icons.get(limiter_type, "❓")
    
    elements.append(Paragraph("<b>DOMINANT OXYGEN DELIVERY MECHANISM</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{mech_icon} {mech_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>{limiter_conf:.0%} confidence</font>", styles["center"]),
    ]
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), mech_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 3 * mm))
    
    if interpretation_adv:
        for line in interpretation_adv.split('\n')[:2]:
            elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 3. SmO2 THRESHOLDS (compact cards)
    # ==========================================================================
    
    lt1 = smo2_manual.get("lt1_watts", "---")
    lt2 = smo2_manual.get("lt2_watts", "---")
    lt1_hr = smo2_manual.get("lt1_hr", "---")
    lt2_hr = smo2_manual.get("lt2_hr", "---")
    
    def fmt(val):
        if val in ("brak danych", None, "---"): return "---"
        try: return f"{float(val):.0f}"
        except: return str(val)
    
    elements.append(Paragraph("<b>MUSCLE OXYGENATION THRESHOLDS</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    lt1_card = [Paragraph("<font size='9' color='#7F8C8D'>SmO₂ LT1</font>", styles["center"]),
                Paragraph(f"<font size='14'><b>{fmt(lt1)} W</b></font>", styles["center"]),
                Paragraph(f"<font size='9'>@ {fmt(lt1_hr)} bpm</font>", styles["center"])]
    lt2_card = [Paragraph("<font size='9' color='#7F8C8D'>SmO₂ LT2</font>", styles["center"]),
                Paragraph(f"<font size='14'><b>{fmt(lt2)} W</b></font>", styles["center"]),
                Paragraph(f"<font size='9'>@ {fmt(lt2_hr)} bpm</font>", styles["center"])]
    
    lt1_table = Table([[lt1_card]], colWidths=[85 * mm])
    lt1_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor("#E8F6F3")), ('BOX', (0, 0), (-1, -1), 1, HexColor("#1ABC9C")), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    lt2_table = Table([[lt2_card]], colWidths=[85 * mm])
    lt2_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), HexColor("#FDEDEC")), ('BOX', (0, 0), (-1, -1), 1, HexColor("#E74C3C")), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    
    thresh_row = Table([[lt1_table, lt2_table]], colWidths=[88 * mm, 88 * mm])
    elements.append(thresh_row)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. TRAINING DECISION CARDS
    # ==========================================================================
    
    recommendations = smo2_advanced.get("recommendations", []) if smo2_advanced else []
    if recommendations:
        elements.append(Paragraph("<b>TRAINING DECISIONS BASED ON O₂ KINETICS</b>", styles["subheading"]))
        elements.append(Spacer(1, 3 * mm))
        
        expected = {"local": ["Wzrost bazowego SmO₂ o 2-4%", "Szybsza reoksygenacja", "Zmniejszenie slope"],
                    "central": ["Wyższe SmO₂ przy tym samym HR", "Lepsza korelacja", "Stabilniejsza saturacja"],
                    "metabolic": ["Późniejszy drop point", "Mniejszy slope", "Lepsza tolerancja kwasu"]}
        exp_list = expected.get(limiter_type, ["Poprawa ogólna", "Stabilniejsza saturacja", "Lepszy klirens"])
        
        for i, rec in enumerate(recommendations[:3]):
            exp_resp = exp_list[i] if i < len(exp_list) else "Poprawa wydolności"
            card_content = [Paragraph(f"<font size='10'><b>{i+1}. {rec}</b></font>", styles["body"]),
                           Paragraph(f"<font size='8' color='#27AE60'>Expected: {exp_resp}</font>", styles["body"])]
            card_table = Table([[card_content]], colWidths=[170 * mm])
            card_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]), ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]), ('LEFTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
            elements.append(card_table)
            elements.append(Spacer(1, 2 * mm))
    
    # Chart
    if figure_paths and "smo2_power" in figure_paths:
        elements.append(Spacer(1, 4 * mm))
        elements.extend(_build_chart(figure_paths["smo2_power"], "SmO₂ vs Power Profile", styles))
    
    # Data quality
    quality_color = "#2ECC71" if data_quality == "good" else ("#F39C12" if data_quality == "low" else "#7F8C8D")
    quality_label = "Wysoka" if data_quality == "good" else ("Niska" if data_quality == "low" else "Brak danych")
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(f"<font size='8' color='#7F8C8D'>Data Quality: </font><font size='8' color='{quality_color}'><b>{quality_label}</b></font>", styles["body"]))
    
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
    if figure_paths and "pdc_curve" in figure_paths:
        elements.extend(_build_chart(figure_paths["pdc_curve"], "Power-Duration Curve", styles))
    
    # === EDUCATION BLOCK: CP/W' ===
    elements.append(Spacer(1, 6 * mm))
    elements.extend(_build_education_block(
        "Dlaczego to ma znaczenie? (CP / W')",
        "Model CP/W' to Twoja cyfrowa bateria, która mówi na co Cię stać w decydującym momencie wyścigu. "
        "Critical Power (CP) to Twoja najwyższa moc „długodystansowa”, utrzymywana bez wyczerpania rezerw. "
        "W' to Twój „bak paliwa” na ataki, krótkie podjazdy i sprinty powyżej mocy progowej. "
        "Każdy skok powyżej CP kosztuje konkretną ilość dżuli, a regeneracja następuje dopiero poniżej tego progu. "
        "Rozumienie tego balansu pozwala decydować, czy odpowiedzieć na atak, czy czekać na swoją szansę. "
        "To serce Twojej strategii, które mówi nam, jak optymalnie zarządzać Twoimi siłami.",
        styles
    ))

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
# PAGE: CARDIOVASCULAR COST DIAGNOSTIC (PREMIUM)
# ============================================================================

def build_page_cardiovascular(cardio_data: Dict[str, Any], styles: Dict) -> List:
    """Build Cardiovascular Cost Diagnostic page - PREMIUM."""
    from reportlab.lib.colors import HexColor
    
    elements = []
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='18'><b>CARDIOVASCULAR COST DIAGNOSTIC</b></font>",
        styles['title']
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Diagnostyka kosztu sercowego generowania mocy</font>",
        styles['center']
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # Extract metrics
    pp = cardio_data.get("pulse_power", 0)
    ef = cardio_data.get("efficiency_factor", 0)
    drift = cardio_data.get("hr_drift_pct", 0)
    recovery = cardio_data.get("hr_recovery_1min")
    cci = cardio_data.get("cci_avg", 0)
    cci_bp = cardio_data.get("cci_breakpoint_watts")
    status = cardio_data.get("efficiency_status", "unknown")
    confidence = cardio_data.get("efficiency_confidence", 0)
    interpretation = cardio_data.get("interpretation", "")
    recommendations = cardio_data.get("recommendations", [])
    
    # ==========================================================================
    # 1. METRIC CARDS
    # ==========================================================================
    
    def build_card(title, value, unit, interp, color):
        card_content = [
            Paragraph(f"<font size='8' color='#7F8C8D'>{title}</font>", styles["center"]),
            Paragraph(f"<font size='16' color='{color}'><b>{value}</b></font>", styles["center"]),
            Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
            Spacer(1, 1 * mm),
            Paragraph(f"<font size='8'>{interp}</font>", styles["center"]),
        ]
        card_table = Table([[card_content]], colWidths=[42 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
            ('BOX', (0, 0), (-1, -1), 1, HexColor(color)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        return card_table
    
    # Pulse Power
    pp_color = "#2ECC71" if pp > 2.0 else ("#F39C12" if pp > 1.5 else "#E74C3C")
    pp_interp = "Efektywny" if pp > 2.0 else ("Umiarkowany" if pp > 1.5 else "Niski")
    card1 = build_card("PULSE POWER", f"{pp:.2f}", "W/bpm", pp_interp, pp_color)
    
    # Efficiency Factor
    ef_color = "#2ECC71" if ef > 1.8 else ("#F39C12" if ef > 1.4 else "#E74C3C")
    ef_interp = "Wysoki" if ef > 1.8 else ("Średni" if ef > 1.4 else "Niski")
    card2 = build_card("EFFICIENCY FACTOR", f"{ef:.2f}", "W/bpm", ef_interp, ef_color)
    
    # HR Drift
    drift_color = "#2ECC71" if drift < 3 else ("#F39C12" if drift < 6 else "#E74C3C")
    drift_interp = "Stabilny" if drift < 3 else ("Drift" if drift < 6 else "Wysoki Drift")
    card3 = build_card("HR DRIFT", f"{drift:.1f}", "%", drift_interp, drift_color)
    
    # HR Recovery or CCI
    if recovery:
        rec_color = "#2ECC71" if recovery > 25 else ("#F39C12" if recovery > 15 else "#E74C3C")
        rec_interp = "Szybki" if recovery > 25 else ("Średni" if recovery > 15 else "Wolny")
        card4 = build_card("HR RECOVERY", f"{recovery:.0f}", "bpm/min", rec_interp, rec_color)
    else:
        cci_color = "#2ECC71" if cci < 0.15 else ("#F39C12" if cci < 0.25 else "#E74C3C")
        cci_interp = "Efektywny" if cci < 0.15 else ("Średni" if cci < 0.25 else "Wysoki koszt")
        card4 = build_card("CCI (avg)", f"{cci:.3f}", "bpm/W", cci_interp, cci_color)
    
    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(cards_row)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 2. CCI METRIC PANEL
    # ==========================================================================
    
    elements.append(Paragraph("<b>CARDIOVASCULAR COST INDEX (CCI)</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    cci_text = f"<b>CCI = {cci:.4f}</b> bpm/W – koszt tętna na jednostkę mocy."
    if cci_bp:
        cci_text += f" <b>Breakpoint</b> przy {cci_bp:.0f}W – punkt załamania efektywności."
    elements.append(Paragraph(cci_text, styles["body"]))
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # 3. EFFICIENCY VERDICT PANEL
    # ==========================================================================
    
    elements.append(Paragraph("<b>CARDIOVASCULAR EFFICIENCY VERDICT</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    status_colors = {"efficient": "#2ECC71", "compensating": "#F39C12", "decompensating": "#E74C3C", "unknown": "#7F8C8D"}
    status_names = {"efficient": "EFFICIENT", "compensating": "COMPENSATING", "decompensating": "DECOMPENSATING", "unknown": "UNDEFINED"}
    status_icons = {"efficient": "✓", "compensating": "⚠", "decompensating": "✗", "unknown": "?"}
    
    st_color = HexColor(status_colors.get(status, "#7F8C8D"))
    st_name = status_names.get(status, "UNDEFINED")
    st_icon = status_icons.get(status, "?")
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{st_icon} {st_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>{confidence:.0%} confidence</font>", styles["center"]),
    ]
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), st_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 3 * mm))
    
    # Interpretation
    if interpretation:
        for line in interpretation.split('\n')[:3]:
            elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. DECISION CARDS
    # ==========================================================================
    
    if recommendations:
        elements.append(Paragraph("<b>TRAINING & ENVIRONMENTAL DECISIONS</b>", styles["subheading"]))
        elements.append(Spacer(1, 3 * mm))
        
        type_colors = {"TRENINGOWA": "#3498DB", "ŚRODOWISKOWA": "#9B59B6", "RECOVERY": "#1ABC9C", "PERFORMANCE": "#2ECC71", "DIAGNOSTYCZNA": "#E74C3C"}
        
        for rec in recommendations[:3]:
            rec_type = rec.get("type", "TRENINGOWA")
            action = rec.get("action", "---")
            expected = rec.get("expected", "---")
            risk = rec.get("risk", "low")
            
            type_color = type_colors.get(rec_type, "#7F8C8D")
            risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
            risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
            
            card_content = [
                Paragraph(f"<font size='9' color='{type_color}'><b>[{rec_type}]</b></font> {action}", styles["body"]),
                Paragraph(f"<font size='8' color='#27AE60'>Expected: {expected}</font> | <font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
            ]
            card_table = Table([[card_content]], colWidths=[170 * mm])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
                ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(card_table)
            elements.append(Spacer(1, 2 * mm))
    
    return elements


# ============================================================================
# PAGE: BREATHING & METABOLIC CONTROL DIAGNOSTIC (PREMIUM)
# ============================================================================

def build_page_ventilation(vent_data: Dict[str, Any], styles: Dict) -> List:
    """Build Breathing & Metabolic Control Diagnostic page - PREMIUM."""
    from reportlab.lib.colors import HexColor
    
    elements = []
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='18'><b>BREATHING & METABOLIC CONTROL</b></font>",
        styles['title']
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Diagnostyka wentylacji i kontroli metabolicznej</font>",
        styles['center']
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # Extract metrics
    ve_avg = vent_data.get("ve_avg", 0)
    ve_max = vent_data.get("ve_max", 0)
    rr_avg = vent_data.get("rr_avg", 0)
    rr_max = vent_data.get("rr_max", 0)
    ve_rr = vent_data.get("ve_rr_ratio", 0)
    ve_slope = vent_data.get("ve_slope", 0)
    ve_bp = vent_data.get("ve_breakpoint_watts")
    pattern = vent_data.get("breathing_pattern", "unknown")
    status = vent_data.get("control_status", "unknown")
    confidence = vent_data.get("control_confidence", 0)
    interpretation = vent_data.get("interpretation", "")
    recommendations = vent_data.get("recommendations", [])
    
    # ==========================================================================
    # 1. METRIC CARDS
    # ==========================================================================
    
    def build_card(title, value, unit, interp, color):
        card_content = [
            Paragraph(f"<font size='8' color='#7F8C8D'>{title}</font>", styles["center"]),
            Paragraph(f"<font size='14' color='{color}'><b>{value}</b></font>", styles["center"]),
            Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
            Paragraph(f"<font size='7'>{interp}</font>", styles["center"]),
        ]
        card_table = Table([[card_content]], colWidths=[42 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
            ('BOX', (0, 0), (-1, -1), 1, HexColor(color)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        return card_table
    
    # VE
    ve_color = "#2ECC71" if ve_max < 120 else ("#F39C12" if ve_max < 150 else "#E74C3C")
    card1 = build_card("VE MAX", f"{ve_max:.0f}", "L/min", f"avg: {ve_avg:.0f}", ve_color)
    
    # RR
    rr_color = "#2ECC71" if rr_max < 45 else ("#F39C12" if rr_max < 55 else "#E74C3C")
    rr_interp = "Ekonomiczny" if rr_max < 45 else ("Podwyższony" if rr_max < 55 else "Wysoki")
    card2 = build_card("RR MAX", f"{rr_max:.0f}", "/min", rr_interp, rr_color)
    
    # VE/RR
    verr_color = "#2ECC71" if ve_rr > 2.5 else ("#F39C12" if ve_rr > 1.5 else "#E74C3C")
    verr_interp = "Głęboki oddech" if ve_rr > 2.5 else ("Średni" if ve_rr > 1.5 else "Płytki")
    card3 = build_card("VE/RR RATIO", f"{ve_rr:.2f}", "L/breath", verr_interp, verr_color)
    
    # VE Slope
    slope_color = "#2ECC71" if ve_slope < 0.25 else ("#F39C12" if ve_slope < 0.4 else "#E74C3C")
    slope_interp = "Stabilny" if ve_slope < 0.25 else ("Rosnący" if ve_slope < 0.4 else "Stromy")
    card4 = build_card("VE SLOPE", f"{ve_slope:.2f}", "L/min/100W", slope_interp, slope_color)
    
    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(cards_row)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 2. BREATHING PATTERN
    # ==========================================================================
    
    pattern_colors = {"efficient": "#2ECC71", "shallow": "#E74C3C", "hyperventilation": "#F39C12", "mixed": "#7F8C8D", "unknown": "#7F8C8D"}
    pattern_names = {"efficient": "EFFICIENT BREATHING", "shallow": "SHALLOW/PANIC", "hyperventilation": "HYPERVENTILATION", "mixed": "MIXED PATTERN", "unknown": "UNDEFINED"}
    
    elements.append(Paragraph("<b>BREATHING PATTERN DETECTION</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    pattern_badge = Paragraph(
        f"<font color='white'><b>{pattern_names.get(pattern, 'UNDEFINED')}</b></font>",
        styles["center"]
    )
    pattern_table = Table([[pattern_badge]], colWidths=[170 * mm])
    pattern_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(pattern_colors.get(pattern, "#7F8C8D"))),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(pattern_table)
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # 3. CONTROL VERDICT
    # ==========================================================================
    
    elements.append(Paragraph("<b>VENTILATORY CONTROL VERDICT</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    status_colors = {"controlled": "#2ECC71", "compensatory": "#F39C12", "unstable": "#E74C3C", "unknown": "#7F8C8D"}
    status_names = {"controlled": "CONTROLLED", "compensatory": "COMPENSATORY", "unstable": "UNSTABLE", "unknown": "UNDEFINED"}
    status_icons = {"controlled": "✓", "compensatory": "⚠", "unstable": "✗", "unknown": "?"}
    
    st_color = HexColor(status_colors.get(status, "#7F8C8D"))
    st_name = status_names.get(status, "UNDEFINED")
    st_icon = status_icons.get(status, "?")
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{st_icon} {st_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>{confidence:.0%} confidence</font>", styles["center"]),
    ]
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), st_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 3 * mm))
    
    if interpretation:
        for line in interpretation.split('\n')[:2]:
            elements.append(Paragraph(line, styles["body"]))
    
    if ve_bp:
        elements.append(Paragraph(f"<b>VE Breakpoint:</b> {ve_bp:.0f}W – punkt załamania kontroli wentylacyjnej", styles["body"]))
    
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. DECISION CARDS
    # ==========================================================================
    
    if recommendations:
        elements.append(Paragraph("<b>VENTILATORY TRAINING DECISIONS</b>", styles["subheading"]))
        elements.append(Spacer(1, 3 * mm))
        
        type_colors = {"TRENINGOWA": "#3498DB", "TECHNICZNA": "#9B59B6", "PERFORMANCE": "#2ECC71", "INTENSYWNOŚĆ": "#1ABC9C", "PILNA": "#E74C3C", "DIAGNOSTYKA": "#F39C12", "MEDYCZNA": "#E74C3C"}
        
        for rec in recommendations[:3]:
            rec_type = rec.get("type", "TRENINGOWA")
            action = rec.get("action", "---")
            expected = rec.get("expected", "---")
            risk = rec.get("risk", "low")
            
            type_color = type_colors.get(rec_type, "#7F8C8D")
            risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
            risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
            
            card_content = [
                Paragraph(f"<font size='9' color='{type_color}'><b>[{rec_type}]</b></font> {action}", styles["body"]),
                Paragraph(f"<font size='8' color='#27AE60'>Expected: {expected}</font> | <font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
            ]
            card_table = Table([[card_content]], colWidths=[170 * mm])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
                ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(card_table)
            elements.append(Spacer(1, 2 * mm))
    
    return elements


# ============================================================================
# PAGE: METABOLIC ENGINE & TRAINING STRATEGY (PREMIUM)
# ============================================================================

def build_page_metabolic_engine(metabolic_data: Dict[str, Any], styles: Dict) -> List:
    """Build Metabolic Engine & Training Strategy page - PREMIUM."""
    from reportlab.lib.colors import HexColor
    
    elements = []
    
    profile = metabolic_data.get("profile", {})
    block = metabolic_data.get("training_block", {})
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='18'><b>METABOLIC ENGINE & TRAINING STRATEGY</b></font>",
        styles['title']
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Profil metaboliczny i strategia 6-8 tygodni</font>",
        styles['center']
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 1. METABOLIC PROFILE PANEL
    # ==========================================================================
    
    vo2max = profile.get("vo2max", 0)
    vlamax = profile.get("vlamax", 0)
    cp = profile.get("cp_watts", 0)
    ratio = profile.get("vo2max_vlamax_ratio", 0)
    phenotype = profile.get("phenotype", "unknown")
    vo2max_source = profile.get("vo2max_source", "unknown")
    data_quality = profile.get("data_quality", "unknown")
    
    elements.append(Paragraph("<b>METABOLIC PROFILE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    def build_metric_card(title, value, unit, color, subtitle=""):
        card_content = [
            Paragraph(f"<font size='8' color='#7F8C8D'>{title}</font>", styles["center"]),
            Paragraph(f"<font size='14' color='{color}'><b>{value}</b></font>", styles["center"]),
            Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
        ]
        if subtitle:
            card_content.append(Paragraph(f"<font size='7' color='#95A5A6'>{subtitle}</font>", styles["center"]))
        card_table = Table([[card_content]], colWidths=[42 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
            ('BOX', (0, 0), (-1, -1), 1, HexColor(color)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        return card_table
    
    # VO2max card - handle n/a and show source
    if vo2max and vo2max > 0:
        vo2_color = "#2ECC71" if vo2max >= 60 else ("#F39C12" if vo2max >= 50 else "#E74C3C")
        vo2_val = f"{vo2max:.0f}"
        vo2_source_label = {"acsm_5min": "ACSM", "acsm_cp": "~CP", "metrics_direct": "test", "ramp_test_peak": "test", "intervals_api": "API"}.get(vo2max_source, "")
    else:
        vo2_color = "#7F8C8D"
        vo2_val = "n/a"
        vo2_source_label = "brak danych"
    
    card1 = build_metric_card("VO₂max", vo2_val, "ml/kg/min", vo2_color, vo2_source_label)
    
    vla_color = "#2ECC71" if vlamax < 0.4 else ("#F39C12" if vlamax < 0.6 else "#E74C3C")
    card2 = build_metric_card("VLaMax", f"{vlamax:.2f}", "mmol/L/s", vla_color, "estymowany")
    card3 = build_metric_card("CP / FTP", f"{cp:.0f}", "W", "#3498DB")
    
    # Ratio card - show n/a if insufficient data
    if ratio and ratio > 0 and vo2max > 0:
        ratio_color = "#2ECC71" if ratio > 130 else ("#F39C12" if ratio > 90 else "#E74C3C")
        ratio_val = f"{ratio:.0f}"
    else:
        ratio_color = "#7F8C8D"
        ratio_val = "n/a"
    card4 = build_metric_card("VO₂/VLa RATIO", ratio_val, "", ratio_color)
    
    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    elements.append(cards_row)
    elements.append(Spacer(1, 3 * mm))
    
    # Phenotype badge
    phenotype_colors = {"diesel": "#27AE60", "allrounder": "#3498DB", "puncher": "#F39C12", "sprinter": "#E74C3C"}
    phenotype_names = {"diesel": "DIESEL (Aerobic Dominant)", "allrounder": "ALLROUNDER", "puncher": "PUNCHER", "sprinter": "SPRINTER (Glycolytic)"}
    
    phenotype_badge = Paragraph(
        f"<font color='white'><b>PHENOTYPE: {phenotype_names.get(phenotype, phenotype.upper())}</b></font>",
        styles["center"]
    )
    phenotype_table = Table([[phenotype_badge]], colWidths=[170 * mm])
    phenotype_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(phenotype_colors.get(phenotype, "#7F8C8D"))),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(phenotype_table)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 2. LIMITER DIAGNOSIS
    # ==========================================================================
    
    limiter = profile.get("limiter", "unknown")
    limiter_conf = profile.get("limiter_confidence", 0)
    
    elements.append(Paragraph("<b>LIMITER DIAGNOSIS</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    limiter_colors = {"aerobic": "#E74C3C", "glycolytic": "#F39C12", "mixed": "#3498DB", "unknown": "#7F8C8D"}
    limiter_names = {"aerobic": "AEROBIC CAPACITY", "glycolytic": "GLYCOLYTIC DOMINANCE", "mixed": "MIXED / BALANCED", "unknown": "UNDEFINED"}
    
    limiter_content = [
        Paragraph(f"<font color='white'><b>{limiter_names.get(limiter, 'UNDEFINED')}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>{limiter_conf:.0%} confidence</font>", styles["center"]),
    ]
    limiter_table = Table([[limiter_content]], colWidths=[170 * mm])
    limiter_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(limiter_colors.get(limiter, "#7F8C8D"))),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(limiter_table)
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # 3. STRATEGY VERDICT
    # ==========================================================================
    
    target = profile.get("adaptation_target", "unknown")
    strategy_interp = profile.get("strategy_interpretation", "")
    
    elements.append(Paragraph("<b>PRIMARY ADAPTATION TARGET</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    target_colors = {"increase_vo2max": "#E74C3C", "lower_vlamax": "#27AE60", "maintain_balance": "#3498DB"}
    target_names = {"increase_vo2max": "🔺 ZWIĘKSZ VO₂max", "lower_vlamax": "🔻 OBNIŻ VLaMax", "maintain_balance": "⚖️ UTRZYMAJ BALANS"}
    
    target_badge = Paragraph(
        f"<font color='white'><b>{target_names.get(target, target.upper())}</b></font>",
        styles["center"]
    )
    target_table = Table([[target_badge]], colWidths=[170 * mm])
    target_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(target_colors.get(target, "#7F8C8D"))),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(target_table)
    elements.append(Spacer(1, 3 * mm))
    
    if strategy_interp:
        for line in strategy_interp.split('\n')[:3]:
            elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. TRAINING BLOCK
    # ==========================================================================
    
    sessions = block.get("sessions", [])
    block_name = block.get("name", "Training Block")
    weeks = block.get("duration_weeks", 6)
    
    elements.append(Paragraph(f"<b>TRAINING BLOCK: {block_name}</b>", styles["subheading"]))
    elements.append(Paragraph(f"<font size='9' color='#7F8C8D'>{weeks} tygodni | {block.get('primary_focus', '')}</font>", styles["body"]))
    elements.append(Spacer(1, 3 * mm))
    
    for i, session in enumerate(sessions[:4]):
        name = session.get("name", "---")
        power = session.get("power_range", "---")
        duration = session.get("duration", "---")
        goal = session.get("adaptation_goal", "---")
        freq = session.get("frequency", "---")
        failure = session.get("failure_criteria", "---")
        exp_smo2 = session.get("expected_smo2", "---")
        exp_hr = session.get("expected_hr", "---")
        
        session_content = [
            Paragraph(f"<font size='10'><b>{i+1}. {name}</b></font> <font size='9' color='#7F8C8D'>({freq})</font>", styles["body"]),
            Paragraph(f"<font size='9'><b>Moc:</b> {power} | <b>Czas:</b> {duration}</font>", styles["body"]),
            Paragraph(f"<font size='8'><b>Cel:</b> {goal}</font>", styles["body"]),
            Paragraph(f"<font size='8' color='#27AE60'>Expected: SmO₂ {exp_smo2}, HR {exp_hr}</font>", styles["body"]),
            Paragraph(f"<font size='8' color='#E74C3C'>⚠ Failure: {failure}</font>", styles["body"]),
        ]
        
        session_table = Table([[session_content]], colWidths=[170 * mm])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
            ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(session_table)
        elements.append(Spacer(1, 2 * mm))
    
    # ==========================================================================
    # 5. KPI MONITORING
    # ==========================================================================
    
    kpi_progress = block.get("kpi_progress", [])
    kpi_regress = block.get("kpi_regress", [])
    
    if kpi_progress or kpi_regress:
        elements.append(Spacer(1, 4 * mm))
        elements.append(Paragraph("<b>KPI MONITORING</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))
        
        # Progress KPIs
        if kpi_progress:
            elements.append(Paragraph("<font color='#27AE60'><b>✓ Sygnały postępu:</b></font>", styles["body"]))
            for kpi in kpi_progress[:3]:
                elements.append(Paragraph(f"<font size='9'>• {kpi}</font>", styles["body"]))
        
        # Regress KPIs
        if kpi_regress:
            elements.append(Paragraph("<font color='#E74C3C'><b>✗ Sygnały regresu / overreaching:</b></font>", styles["body"]))
            for kpi in kpi_regress[:3]:
                elements.append(Paragraph(f"<font size='9'>• {kpi}</font>", styles["body"]))
    
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
    
    # 1. Check if file exists
    if not chart_path or not os.path.exists(chart_path):
        logger.warning(f"PDF Layout: Chart file missing for '{title}' at path: {chart_path}")
        elements.append(Paragraph("Wykres niedostępny", styles["small"]))
        return elements
    
    # 2. Embed image
    try:
        available_width = PAGE_WIDTH - 2 * MARGIN
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
        logger.error(f"PDF Layout: Error embedding chart '{title}' from {chart_path}: {e}")
    return elements


def _build_education_block(title: str, content: str, styles: Dict) -> List:
    """Helper to build a consistent education block with 'Dlaczego to ma znaczenie?'."""
    elements = []
    
    # Label
    label = Paragraph("<b>Część edukacyjna – do zrozumienia wyników</b>", styles["small"])
    elements.append(label)
    
    # Title & Text in a subtle box
    inner_story = [
        Paragraph(f"<b>{title}</b>", styles["heading"]),
        Spacer(1, 1 * mm),
        Paragraph(content, styles["body_italic"])
    ]
    
    table = Table([[inner_story]], colWidths=[170 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLORS["light_grey"]),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, COLORS["grey"]),
        ('BOX', (0, 0), (-1, -1), 0.25, COLORS["grey"]),
        ('LEFTPADDING', (0, 0), (-1, -1), 4 * mm),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4 * mm),
        ('TOPPADDING', (0, 0), (-1, -1), 4 * mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4 * mm),
    ]))
    
    elements.append(table)
    return elements


# ============================================================================
# PAGE 5: TEORIA FIZJOLOGICZNA (INSCYD/WKO5)
# ============================================================================

def build_page_theory(styles: Dict) -> List:
    """Build Page 5: Advanced Physiological Theory."""
    elements = []
    
    elements.append(Paragraph("Model Metaboliczny (INSCYD/WKO5)", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # Intro
    elements.append(Paragraph(
        "Twoja wydajność zależy od interakcji trzech systemów energetycznych. "
        "Zrozumienie ich pozwala na precyzyjne dopasowanie treningu.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # VO2max
    elements.append(Paragraph("1. VO₂max (System Tlenowy)", styles["heading"]))
    elements.append(Paragraph(
        "Maksymalna ilość tlenu, jaką Twój organizm może przyswoić. "
        "Jest to Twój „silnik diesla” – odpowiada za moc na długim dystansie. "
        "Wysokie VO₂max jest kluczowe dla każdego kolarza wytrzymałościowego.",
        styles["body"]
    ))
    
    # VLaMax
    elements.append(Paragraph("2. VLaMax (System Glikolityczny)", styles["heading"]))
    elements.append(Paragraph(
        "Maksymalne tempo produkcji mleczanu. To Twój „dopalacz turbo”. "
        "Wysokie VLaMax daje świetny sprint i ataki, ale powoduje szybkie zużycie węglowodanów (niskie FatMax) i obniża próg FTP. "
        "Niskie VLaMax zwiększa próg FTP i oszczędza glikogen (dobre dla Ironman/GC), ale ogranicza dynamikę.",
        styles["body"]
    ))
    
    # Interaction
    elements.append(Paragraph("Interakcja VO₂max ↔ VLaMax", styles["subheading"]))
    elements.append(Paragraph(
        "Twój próg FTP to wynik walki między tymi dwoma systemami. "
        "Aby podnieść FTP, możesz albo zwiększyć VO₂max (powiększyć silnik), albo obniżyć VLaMax (zmniejszyć spalanie). "
        "Wybór strategii zależy od Twojego typu zawodnika.",
        styles["body"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # FatMax explain
    elements.append(Paragraph("FatMax (Metabolizm Tłuszczowy)", styles["heading"]))
    elements.append(Paragraph(
        "Moc, przy której spalasz najwięcej tłuszczu (zwykle strefa Z2). "
        "Trening w tej strefie uczy organizm oszczędzania glikogenu.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # Advanced INSCYD Theory Table
    elements.append(Paragraph("Typy Zawodników i Strategie", styles["heading"]))
    data = [
        ["Typ", "VO2max", "VLaMax", "Charakterystyka"],
        ["Sprinter", "Średni", "Wysoki", "Dynamika, punch, sprinty"],
        ["Climber", "Wysoki", "Niski", "Długie wspinaczki, tempo"],
        ["Time Trialist", "Wysoki", "Niski", "Równe tempo, aerodynamika"],
        ["Puncheur", "Wysoki", "Średni", "Ataki, krótkie górki"]
    ]
    t = Table(data, colWidths=[30*mm, 30*mm, 30*mm, 80*mm])
    t.setStyle(get_table_style())
    elements.append(t)
    
    elements.append(Spacer(1, 4 * mm))
    # Hierarchy & Protocol (New Content)
    elements.append(Paragraph("3. Hierarchia Sygnałów i Protokół", styles["heading"]))
    elements.append(Paragraph(
        "Nie wszystkie dane są równe. W naszej metodologii najważniejsza jest Wentylacja (VE), "
        "ponieważ najdokładniej odzwierciedla stan metaboliczny całego ciała. HR i SmO₂ to sygnały wspierające. "
        "Długość kroku (np. 1-2 minuty) jest krytyczna, by sygnały zdążyły się ustabilizować. "
        "Zrozumienie opóźnień (HR reaguje najwolniej, SmO₂ najszybciej) pozwala na precyzyjną detekcję progów.",
        styles["body"]
    ))

    return elements


# ============================================================================
# PAGE 6: TERMOREGULACJA
# ============================================================================

def build_page_thermal(
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 6: Thermal Analysis."""
    elements = []
    
    elements.append(Paragraph("Analiza Termoregulacji", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    elements.append(Paragraph(
        "Ciepło jest „cichym zabójcą” wydajności. Wzrost temperatury głębokiej (Core Temp) "
        "powoduje przekierowanie krwi do skóry (chłodzenie), co zabiera tlen pracującym mięśniom.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # Chart 1: Core Temp vs HSI
    if figure_paths and "thermal_hsi" in figure_paths:
        elements.extend(_build_chart(figure_paths["thermal_hsi"], "Temp. Głęboka vs Indeks Zmęczenia (HSI)", styles))
        elements.append(Spacer(1, 4 * mm))
    else:
        elements.append(Paragraph("Brak wykresu Temp vs HSI (brak danych)", styles["small"]))
        
    elements.append(Paragraph(
        "<b>Heat Strain Index (HSI):</b> Skumulowane obciążenie cieplne. "
        "Powyżej 38.5°C organizm wchodzi w strefę krytyczną.",
        styles["body"]
    ))
    
    # FORCE PAGE BREAK BEFORE EFFICIENCY
    elements.append(PageBreak())
    
    # Chart 2: Efficiency (Start of new page)
    elements.append(Paragraph("Spadek Efektywności (Cardiac Drift)", styles["title"])) # Use title style for new page
    elements.append(Spacer(1, 6 * mm))
    
    if figure_paths and "thermal_efficiency" in figure_paths:
        elements.extend(_build_chart(figure_paths["thermal_efficiency"], "Efektywność vs Temperatura", styles))
        elements.append(Spacer(1, 4 * mm))
    else:
        elements.append(Paragraph("Brak wykresu Efektywności (brak danych)", styles["small"]))
        
    elements.append(Paragraph(
        "<b>Efficiency Factor (W/bpm):</b> Wykres pokazuje, jak spada generowana moc na jedno uderzenie serca w miarę wzrostu temperatury. "
        "Stromy spadek oznacza słabą termoregulację.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# PAGE 7: PROFIL METABOLICZNY (LIMITERY)
# ============================================================================

# ============================================================================
# NEW PAGE: BIOMECHANIKA
# ============================================================================

def build_page_biomech(
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Biomechanics page."""
    elements = []
    
    elements.append(Paragraph("Analiza Biomechaniczna (Transfer Mocy)", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    elements.append(Paragraph(
        "Biomechanika kolarstwa analizuje sposób, w jaki generujesz moc. "
        "Kluczowym elementem jest balans między kadencją (szybkością) a momentem obrotowym (siłą).",
        styles["body"]
    ))
    
    # Chart 1: Torque vs Cadence
    if figure_paths and "biomech_summary" in figure_paths:
        elements.extend(_build_chart(figure_paths["biomech_summary"], "Moment Obrotowy vs Kadencja", styles))
        elements.append(Spacer(1, 4 * mm))
        
    # Chart 2: Torque vs SmO2
    if figure_paths and "biomech_torque_smo2" in figure_paths:
        elements.extend(_build_chart(figure_paths["biomech_torque_smo2"], "Fizjologia Okluzji (Siła vs Tlen)", styles))
        elements.append(Spacer(1, 4 * mm))
        
    elements.append(Paragraph(
        "<b>Interpretacja:</b> Spadek saturacji (SmO₂) przy wysokich momentach obrotowych może świadczyć o okluzji mechanicznej "
        "lub niskiej efektywności układu krążenia w warunkach wysokiego napięcia mięśniowego.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# NEW PAGE: DRYF FIZJOLOGICZNY I KPI
# ============================================================================

def build_page_drift_kpi(
    kpi: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Physiological Drift and KPI page."""
    elements = []
    
    elements.append(Paragraph("Dryf Fizjologiczny i Wskaźniki KPI", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # 1. Heatmaps (Side by side or sequential)
    if figure_paths and "drift_heatmap_hr" in figure_paths:
        elements.extend(_build_chart(figure_paths["drift_heatmap_hr"], "Mapa Dryfu (HR vs Power)", styles))
        elements.append(Spacer(1, 4 * mm))
        
    if figure_paths and "drift_heatmap_smo2" in figure_paths:
        elements.extend(_build_chart(figure_paths["drift_heatmap_smo2"], "Mapa Oksydacji (SmO2 vs Power)", styles))
        elements.append(Spacer(1, 4 * mm))

    # 2. KPI Table
    elements.append(Paragraph("Kluczowe Wskaźniki Wydajności (KPI)", styles["heading"]))
    
    def fmt(val, unit=""):
        if val is None or val == "brak danych": return "---"
        try:
            return f"{float(val):.2f}{unit}"
        except:
            return f"{val}{unit}"

    # =========================================================================
    # CRITICAL: VO2max MUST come from canonical source via kpi["vo2max_est"]
    # DO NOT calculate VO2max here - it is READ-ONLY display
    # =========================================================================
    vo2max_val = kpi.get("vo2max_est")
    vo2max_source = kpi.get("vo2max_source", "")
    
    # Format VO2max label - NO "Estimate" in name
    if vo2max_val and vo2max_val != "brak danych":
        vo2max_display = fmt(vo2max_val, " ml/kg")
        vo2max_label = "VO₂max"
        if vo2max_source:
            source_short = {"acsm_5min": "(ACSM)", "acsm_cp": "(~CP)", "metrics_fallback": ""}.get(vo2max_source, "")
            vo2max_label = f"VO₂max {source_short}".strip()
    else:
        vo2max_display = "n/a"
        vo2max_label = "VO₂max"

    data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Efficiency Factor (EF)", fmt(kpi.get("ef")), "Moc na uderzenie serca (im wyżej, tym lepiej)"],
        ["Pa:Hr (Decoupling)", fmt(kpi.get("pa_hr"), "%"), "Stabilność układu krążenia"],
        ["% SmO2 Drift", fmt(kpi.get("smo2_drift"), "%"), "Zmęczenie lokalne mięśni"],
        [vo2max_label, vo2max_display, "Pułap tlenowy (canonical)"]
    ]
    
    table = Table(data, colWidths=[50 * mm, 30 * mm, 85 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph(
        "<b>Dryf (Pa:Hr):</b> Wartość powyżej 5% sugeruje niepełną adaptację do danego obciążenia "
        "lub wpływ czynników zewnętrznych (upał, odwodnienie, chroniczne zmęczenie).",
        styles["body"]
    ))
    
    # === EDUCATION BLOCK: DRIFT ===
    elements.append(Spacer(1, 6 * mm))
    elements.extend(_build_education_block(
        "Dlaczego to ma znaczenie? (Cardiac Drift)",
        "Dryf tętna to sygnał ostrzegawczy Twojego układu chłodzenia, którego nie wolno ignorować. "
        "Jeśli przy stałej mocy tętno systematycznie rośnie, serce musi pracować ciężej, "
        "by przetłoczyć krew nie tylko do mięśni, ale i do skóry w celu ochłodzenia organizmu. "
        "Oznacza to spadek efektywności (EF) i nieproporcjonalnie wysoki koszt energetyczny ruchu. "
        "Śledząc ten parametr, wiemy kiedy warto zainwestować w trening w cieple lub poprawić picie. "
        "To klucz do utrzymania stabilnego tempa w drugiej połowie długodystansowych startów.",
        styles
    ))

    return elements


# ============================================================================
# UPDATED PAGE 7: METABOLIC MODEL
# ============================================================================

def build_page_limiters(
    metadata: Dict[str, Any],
    cp_model: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Metabolic Profile page (Radar + VLaMax Balance)."""
    elements = []
    
    elements.append(Paragraph("Model Metaboliczny (INSCYD-style)", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # 1. VLaMax Balance Schema
    if figure_paths and "vlamax_balance" in figure_paths:
        elements.extend(_build_chart(figure_paths["vlamax_balance"], "Balans VO2max vs VLaMax", styles))
        elements.append(Spacer(1, 4 * mm))
    else:
        elements.append(Paragraph("Brak danych do analizy VLaMax (wymagane min. 20 min MMP)", styles["small"]))

    # 2. Radar Chart
    if figure_paths and "limiters_radar" in figure_paths:
        elements.extend(_build_chart(figure_paths["limiters_radar"], "Radar Obciążenia (5 min Peak)", styles))
        elements.append(Spacer(1, 4 * mm))
        
    elements.append(Paragraph(
        "<b>Profil Zawodnika:</b> Twój profil wynika ze stosunku między zdolnością tlenową (VO₂max) "
        "a beztlenową (VLaMax). Wykryty limiter wskazuje, nad czym pracować w następnym bloku treningowym.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# PAGE 8: DODATKOWE ANALIZY
# ============================================================================

def build_page_extra(
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 8: Extra Analytics (Ventilation & Drift)."""
    elements = []
    
    elements.append(Paragraph("Zaawansowana Analityka", styles["title"]))
    elements.append(Spacer(1, 6 * mm))
    
    # Vent Full
    if figure_paths and "vent_full" in figure_paths:
        elements.extend(_build_chart(figure_paths["vent_full"], "Dynamika Wentylacji (VE) vs Moc", styles))
        elements.append(Spacer(1, 6 * mm))
        
    # Drift Maps
    elements.append(Paragraph("Mapy Dryfu i Decoupling", styles["heading"]))
    
    if figure_paths and "drift_hr" in figure_paths:
        elements.extend(_build_chart(figure_paths["drift_hr"], "Moc vs Tętno", styles))
        elements.append(Spacer(1, 4 * mm))
        
    # PAGE BREAK FOR SECOND DRIFT MAP
    elements.append(PageBreak())
        
    if figure_paths and "drift_smo2" in figure_paths:
        # Title moved to new page
        elements.append(Paragraph("Moc vs Saturacja Mięśniowa", styles["title"]))
        elements.append(Spacer(1, 6 * mm))
        elements.extend(_build_chart(figure_paths["drift_smo2"], "Decoupling Mięśniowy", styles))
        
    return elements
