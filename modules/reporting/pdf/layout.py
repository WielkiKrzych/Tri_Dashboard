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
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
import os
import logging
from typing import Dict, Any, List, Optional

from .styles import (
    COLORS, PAGE_WIDTH, MARGIN,
    get_table_style, FONT_FAMILY, FONT_FAMILY_BOLD, FONT_FAMILY_ITALIC,
)
from ...calculations.version import RAMP_METHOD_VERSION

# Setup logger
logger = logging.getLogger("Tri_Dashboard.PDFLayout")

# Premium color constants
PREMIUM_COLORS = {
    "navy": HexColor("#1A5276"),      # Recommendations/training
    "dark_glass": HexColor("#17252A"), # Title page background
    "red": HexColor("#C0392B"),        # Warnings/limitations
    "green": HexColor("#27AE60"),      # Positives/strengths
    "white": HexColor("#FFFFFF"),
    "light_gray": HexColor("#BDC3C7"),
}

# ==============================================================================
# SIGNAL QUALITY LANGUAGE HELPERS
# ==============================================================================

def get_confidence_prefix(confidence: float) -> str:
    """Return empty prefix - interpretation text is self-sufficient.

    Previously added qualifiers like 'Analiza wskazuje:' but these looked
    mechanical when repeated before every line. Kept for API compatibility.
    """
    return ""


def get_confidence_suffix(confidence: float) -> str:
    """Get methodology note based on signal quality level.

    Professional phrasing - no numeric confidence exposed to end users.
    """
    return ""


def _signal_quality_label(confidence: float) -> str:
    """Convert numeric confidence to professional signal quality label."""
    if confidence >= 0.85:
        return "bardzo dobra"
    elif confidence >= 0.7:
        return "dobra"
    elif confidence >= 0.5:
        return "wystarczająca"
    else:
        return "podstawowa"


def _signal_quality_stars(confidence: float) -> str:
    """Convert numeric confidence to star rating (1-5)."""
    if confidence >= 0.9:
        stars = 5
    elif confidence >= 0.75:
        stars = 4
    elif confidence >= 0.6:
        stars = 3
    elif confidence >= 0.4:
        stars = 2
    else:
        stars = 1
    return "★" * stars + "☆" * (5 - stars)


# PREMIUM HELPER FUNCTIONS
# ============================================================================

def build_colored_box(text: str, styles: Dict, bg_color: str = "navy") -> List:
    """Create a colored box with text for recommendations/warnings/positives.
    
    Args:
        text: Text content
        styles: PDF styles dict
        bg_color: "navy", "red", or "green"
        
    Returns:
        List of flowables
    """
    color_map = {
        "navy": PREMIUM_COLORS["navy"],
        "red": PREMIUM_COLORS["red"],
        "green": PREMIUM_COLORS["green"],
    }
    bg = color_map.get(bg_color, PREMIUM_COLORS["navy"])
    
    table_data = [[Paragraph(
        f"<font color='white'><b>{text}</b></font>",
        styles["center"]
    )]]
    
    box_table = Table(table_data, colWidths=[170 * mm])
    box_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    return [box_table, Spacer(1, 4 * mm)]


def build_section_description(text: str, styles: Dict) -> List:
    """Add 10pt italic description under section header.
    
    Args:
        text: Description text (1-2 sentences)
        styles: PDF styles dict
        
    Returns:
        List of flowables
    """
    desc_style = ParagraphStyle(
        "SectionDescription",
        fontName="DejaVuSans-Oblique" if "DejaVuSans" in str(styles.get("body", {})) else "Helvetica-Oblique",
        fontSize=10,
        textColor=HexColor("#7F8C8D"),
        leading=12,
        spaceAfter=4 * mm,
    )
    return [Paragraph(text, desc_style)]


def build_title_page(metadata: Dict[str, Any], styles: Dict) -> List:
    """Build premium title page with background image.
    
    Layout similar to KNF CSIRT document:
    - Background image banner with title overlay
    - Document metadata at bottom
    
    Args:
        metadata: Report metadata (test_date, session_id, etc.)
        styles: PDF styles dict
        
    Returns:
        List of flowables for title page
    """
    from datetime import datetime
    import os
    
    elements = []
    
    # Get path to background image
    # Try multiple locations for the background image
    bg_paths = [
        "assets/title_background.jpg",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "title_background.jpg"),
    ]
    
    bg_image_path = None
    for path in bg_paths:
        if os.path.exists(path):
            bg_image_path = path
            break
    
    # Title banner with background image
    if bg_image_path and os.path.exists(bg_image_path):
        # Use image as banner
        try:
            from reportlab.platypus import Image as RLImage
            
            # Add banner image (full width)
            banner = RLImage(bg_image_path, width=180 * mm, height=100 * mm)
            elements.append(banner)
            
            # Overlay title - use negative spacer to position on image
            elements.append(Spacer(1, -70 * mm))  # Move up into image area
            
            # Title with proper spacing between lines
            title_content = [
                [Paragraph(
                    "<font color='white' size='28'><b>BADANIA WYDOLNOŚCIOWE</b></font>",
                    styles["center"]
                )],
                [Spacer(1, 8 * mm)],  # Space between title and subtitle
                [Paragraph(
                    "<font color='#E0E0E0' size='12'>w oparciu o Wentylację Minutową (VE)</font>",
                    styles["center"]
                )],
                [Paragraph(
                    "<font color='#E0E0E0' size='12'>i Natlenienie Mięśniowe (SmO₂)</font>",
                    styles["center"]
                )],
            ]
            
            title_table = Table(title_content, colWidths=[170 * mm])
            title_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(title_table)
            
            elements.append(Spacer(1, 50 * mm))  # Move back down
            
        except Exception as e:
            logger.warning(f"Could not load background image: {e}")
            # Fallback to solid color
            elements.append(Spacer(1, 30 * mm))
            _add_fallback_title(elements, styles)
    else:
        # Fallback: solid color background
        elements.append(Spacer(1, 30 * mm))
        _add_fallback_title(elements, styles)
    
    # Spacer before metadata (balanced to show watermark and fit on one page)
    elements.append(Spacer(1, 15 * mm))
    
    # === METRYKA DOKUMENTU (CENTERED) ===
    elements.append(Paragraph(
        "<font size='14'><b>Metryka dokumentu:</b></font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # Test info
    test_date = metadata.get('test_date', '---')
    session_id = metadata.get('session_id', '')[:8] if metadata.get('session_id') else ''
    method_version = metadata.get('method_version', '1.0.0')
    gen_date = datetime.now().strftime("%d.%m.%Y, %H:%M")
    subject_name = metadata.get('subject_name', '')
    subject_anthropometry = metadata.get('subject_anthropometry', '')
    
    meta_data = [
        [Paragraph("<b>Data testu:</b>", styles["center"]), 
         Paragraph(str(test_date), styles["center"])],
        [Paragraph("<b>ID sesji:</b>", styles["center"]), 
         Paragraph(session_id, styles["center"])],
        [Paragraph("<b>Wersja metody:</b>", styles["center"]), 
         Paragraph(method_version, styles["center"])],
        [Paragraph("<b>Data generowania:</b>", styles["center"]), 
         Paragraph(gen_date, styles["center"])],
    ]
    
    # Add subject name row if provided
    if subject_name:
        meta_data.append([
            Paragraph("<b>Osoba badana:</b>", styles["center"]),
            Paragraph(subject_name, styles["center"])
        ])
    
    # Add anthropometry row if provided
    if subject_anthropometry:
        meta_data.append([
            Paragraph("<b>Wiek / Wzrost / Waga:</b>", styles["center"]),
            Paragraph(subject_anthropometry, styles["center"])
        ])
    
    meta_table = Table(meta_data, colWidths=[60 * mm, 80 * mm])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    
    # Spacer before author section (reduced to fit on one page)
    elements.append(Spacer(1, 20 * mm))
    
    # === OPRACOWANIE SECTION WITH BACKGROUND IMAGE (like title banner) ===
    # Use the same background image for premium styling
    if bg_image_path and os.path.exists(bg_image_path):
        try:
            from reportlab.platypus import Image as RLImage
            
            # Add smaller banner for author section (compact to fit on one page)
            author_banner = RLImage(bg_image_path, width=180 * mm, height=25 * mm)
            elements.append(author_banner)
            
            # Overlay author text - use negative spacer to position on image
            elements.append(Spacer(1, -18 * mm))  # Move up into image area
            
            author_content = [[Paragraph(
                "<font color='white' size='20'><b>Opracowanie: Krzysztof Kubicz</b></font>",
                styles["center"]
            )]]
            
            author_table = Table(author_content, colWidths=[170 * mm])
            author_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(author_table)
            elements.append(Spacer(1, 5 * mm))  # Move back down
            
        except Exception as e:
            logger.warning(f"Could not create author banner: {e}")
            # Fallback to simple larger bold text
            elements.append(Paragraph(
                "<font size='20'><b>Opracowanie: Krzysztof Kubicz</b></font>",
                styles["center"]
            ))
    else:
        # Fallback: larger bold text with solid background
        author_content = [[Paragraph(
            "<font color='white' size='20'><b>Opracowanie: Krzysztof Kubicz</b></font>",
            styles["center"]
        )]]
        author_table = Table(author_content, colWidths=[170 * mm])
        author_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PREMIUM_COLORS["navy"]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(author_table)
    
    return elements


def _add_fallback_title(elements: List, styles: Dict):
    """Add title block with solid color background (fallback when no image)."""
    title_content = [
        [Paragraph(
            "<font color='white' size='24'><b>BADANIA WYDOLNOŚCIOWE</b></font>",
            styles["center"]
        )],
        [Paragraph(
            "<font color='#BDC3C7' size='14'>w oparciu o Wentylację Minutową (VE)</font>",
            styles["center"]
        )],
        [Paragraph(
            "<font color='#BDC3C7' size='14'>i Natlenienie Mięśniowe (SmO₂)</font>",
            styles["center"]
        )],
    ]
    
    title_table = Table(title_content, colWidths=[170 * mm])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PREMIUM_COLORS["dark_glass"]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (0, 0), 25),
        ('BOTTOMPADDING', (-1, -1), (-1, -1), 25),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -2), 5),
    ]))
    elements.append(title_table)


def build_contact_footer(styles: Dict) -> List:
    """Build contact info footer for last page.
    
    Args:
        styles: PDF styles dict
        
    Returns:
        List of flowables
    """
    elements = []
    
    # Separator
    elements.append(Spacer(1, 10 * mm))
    sep_table = Table([[""]], colWidths=[170 * mm])
    sep_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, -1), 1, HexColor("#DEE2E6")),
    ]))
    elements.append(sep_table)
    elements.append(Spacer(1, 5 * mm))
    
    # Contact info
    elements.append(Paragraph(
        "<font size='12'><b>Krzysztof Kubicz</b></font>",
        styles["center"]
    ))
    elements.append(Paragraph(
        "<font size='11' color='#1A5276'>kubiczk@icloud.com</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        "<font size='9' color='#7F8C8D'>Kontakt, pytania, konsultacje oraz umówienie się na ponowne badanie - "
        "wiadomość mailowa lub tekstowa na nr tel.: 453 330 419</font>",
        styles["center"]
    ))
    
    return elements


# ============================================================================
# TABLE OF CONTENTS (SPIS TREŚCI) - HIERARCHICAL
# ============================================================================

def build_table_of_contents(styles: Dict, section_titles: List[Dict[str, Any]]) -> List:
    """Build hierarchical Table of Contents page with clickable hyperlinks.
    
    Args:
        styles: PDF styles dictionary
        section_titles: List of dicts with 'title', 'page', 'level' keys
                        level=0 for main chapters, level=1 for subchapters
        
    Returns:
        List of reportlab flowables
    """
    
    elements = []
    
    # Header
    elements.append(Paragraph(
        "<font size='22'><b>SPIS TREŚCI</b></font>",
        styles["title"]
    ))
    elements.append(Spacer(1, 8 * mm))
    
    # Table of Contents entries with hierarchy and hyperlinks
    # We need to track row heights to add spacing before chapters
    toc_data = []
    row_heights = []  # Track heights for each row
    
    for i, section in enumerate(section_titles):
        title = section.get("title", "---")
        page = section.get("page", "---")
        level = section.get("level", 1)  # 0=chapter, 1=subchapter
        
        # Create anchor name from page number for internal linking
        # ReportLab uses <a href="#page_X"> format for internal links
        anchor_name = f"page_{page}"
        
        # Check if this is a chapter (level=0) and not the first entry
        # If so, check if the previous entry was a subchapter (level=1)
        is_chapter_after_subchapter = False
        if level == 0 and i > 0:
            prev_level = section_titles[i - 1].get("level", 1)
            if prev_level == 1:
                is_chapter_after_subchapter = True
        
        if level == 0:
            # Main chapter - bold, larger, dark blue background with hyperlink
            title_para = Paragraph(
                f"<a href='#{anchor_name}' color='#1A5276'><font size='11'><b>{title}</b></font></a>",
                styles["body"]
            )
            page_para = Paragraph(
                f"<a href='#{anchor_name}' color='#1A5276'><font size='11'><b>{page}</b></font></a>",
                styles["body"]
            )
        else:
            # Subchapter - indented with bullet and hyperlink
            title_para = Paragraph(
                f"<a href='#{anchor_name}' color='#555555'><font size='9'>    • {title}</font></a>",
                styles["body"]
            )
            page_para = Paragraph(
                f"<a href='#{anchor_name}' color='#7F8C8D'><font size='9'>{page}</font></a>",
                styles["body"]
            )
        
        toc_data.append([title_para, page_para])
        # Add extra height for chapters that follow subchapters (half-line, ~6mm extra)
        if is_chapter_after_subchapter:
            row_heights.append(14 * mm)  # Normal row ~8mm + extra 6mm
        else:
            row_heights.append(None)  # Auto height
    
    if toc_data:
        toc_table = Table(toc_data, colWidths=[150 * mm, 20 * mm], rowHeights=row_heights)
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(toc_table)
    else:
        elements.append(Paragraph(
            "<font color='#7F8C8D'>Brak sekcji do wyświetlenia</font>",
            styles["body"]
        ))
    
    return elements


def build_chapter_header(chapter_num: str, chapter_title: str, styles: Dict) -> List:
    """Build a prominent chapter header with Roman numeral.
    
    Args:
        chapter_num: Roman numeral (I, II, III, IV, V)
        chapter_title: Chapter title text
        styles: PDF styles dict
        
    Returns:
        List of flowables
    """
    elements = []
    
    # Chapter header with navy background
    header_content = [[Paragraph(
        f"<font color='white' size='16'><b>{chapter_num}. {chapter_title}</b></font>",
        styles["center"]
    )]]
    
    header_table = Table(header_content, colWidths=[170 * mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PREMIUM_COLORS["navy"]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6 * mm))
    
    return elements


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
    
    # Title row — collect with first content block to prevent orphan heading
    intro_elements = []
    intro_elements.append(Paragraph(
        "<font size='14'>5.2 PODSUMOWANIE FIZJOLOGICZNE</font>",
        styles["center"]
    ))

    # Status badge + date row
    status_text = f"{limiter_icon} {limiter_name}"
    header_table = Table([
        [
            Paragraph("", styles["body"]),  # Removed duplicate CENTRALNY header
            Paragraph("", styles["body"])  # Removed Data testu line
        ]
    ], colWidths=[100 * mm, 70 * mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    intro_elements.append(header_table)
    intro_elements.append(Spacer(1, 2 * mm))
    
    # ==========================================================================
    # 2. PHYSIOLOGICAL VERDICT CARD
    # ==========================================================================
    
    verdict = limiter.get("verdict", "Brak diagnozy")
    interpretation = limiter.get("interpretation", [])
    subtitle = limiter.get("subtitle", "")
    
    # Get confidence from confidence_panel (0-100 -> 0.0-1.0)
    overall_confidence = confidence_panel.get("overall_score", 100) / 100.0
    
    # Apply confidence-aware language to verdict
    verdict_with_confidence = get_confidence_prefix(overall_confidence) + verdict + get_confidence_suffix(overall_confidence)
    
    # Card content
    verdict_content = [
        Paragraph(f"<font size='14'><b>{limiter_icon} DOMINUJĄCY LIMITER: {limiter_name}</b></font>", styles["heading"]),
        Paragraph(f"<font size='10' color='#7F8C8D'>{subtitle}</font>", styles["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>{verdict_with_confidence}</b>", styles["body"]),
        Spacer(1, 2 * mm),
    ]
    
    for line in interpretation[:3]:
        # Apply confidence-aware language to each interpretation line
        line_with_confidence = get_confidence_prefix(overall_confidence) + line + get_confidence_suffix(overall_confidence)
        verdict_content.append(Paragraph(f"• {line_with_confidence}", styles["body"]))
    
    verdict_table = Table([[verdict_content]], colWidths=[170 * mm])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#F8F9FA")),
        ('BOX', (0, 0), (-1, -1), 2, limiter_color),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    # Only keep heading + a minimal anchor to prevent orphan heading
    # The verdict_table is too large to keep together with the heading
    elements.append(KeepTogether(intro_elements + [verdict_table]))
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 3. SIGNAL AGREEMENT MATRIX
    # ==========================================================================
    
    elements.append(Paragraph("<b>MACIERZ SYGNAŁÓW</b>", styles["subheading"]))
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
        
        # Use simple pictogram symbols for PDF compatibility
        icon_map = {
            "🫁": "~",      # VE - wave for ventilation
            "🩸": "O₂",     # O2 - oxygen symbol  
            "♥": "♥",       # HR - heart (standard character)
            "💪": "O₂",     # SmO2 - oxygen symbol
            "❓": "?"
        }
        display_icon = icon_map.get(icon, "•")
        
        tile_content = [
            Paragraph(f"<font size='14'>{display_icon}</font>", styles["center"]),
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
        Paragraph("<font size='8'>Indeks zgodności</font>", styles["center"]),
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
    
    elements.append(Paragraph("<b>JAKOŚĆ SYGNAŁU</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    overall_score = confidence_panel.get("overall_score", 0)
    breakdown = confidence_panel.get("breakdown", {})
    limiting_factor = confidence_panel.get("limiting_factor", "---")
    score_color = confidence_panel.get("color", "#7F8C8D")

    overall_conf = overall_score / 100.0
    quality_label = _signal_quality_label(overall_conf)
    quality_stars = _signal_quality_stars(overall_conf)

    score_para = Paragraph(
        f"<font size='20' color='{score_color}'><b>{quality_stars}</b></font><br/>"
        f"<font size='11'>Jakość: <b>{quality_label}</b></font>",
        styles["body"]
    )

    # Breakdown bars - professional labels
    signal_labels = {
        "ve_stability": "Stabilność VE",
        "hr_lag": "Odpowiedź HR",
        "smo2_noise": "Jakość SmO₂",
        "protocol_quality": "Protokół",
    }
    breakdown_rows = []
    for key, label in signal_labels.items():
        val = breakdown.get(key, 50)
        val_conf = val / 100.0
        stars = _signal_quality_stars(val_conf)
        breakdown_rows.append([
            Paragraph(f"<font size='8'>{label}</font>", styles["body"]),
            Paragraph(f"<font size='9'>{stars}</font>", styles["body"]),
        ])

    breakdown_table = Table(breakdown_rows, colWidths=[35 * mm, 30 * mm])
    breakdown_table.setStyle(TableStyle([
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    confidence_row = Table([[score_para, breakdown_table]], colWidths=[60 * mm, 110 * mm])
    confidence_row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(confidence_row)
    if limiting_factor and limiting_factor != "---":
        elements.append(Paragraph(f"<font size='9' color='#7F8C8D'>Uwaga: jakość ograniczona przez <b>{limiting_factor}</b></font>", styles["body"]))
    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # 5. TRAINING DECISION CARDS
    # ==========================================================================

    training_card_elements = []
    training_card_elements.append(Paragraph("<b>DECYZJE TRENINGOWE</b>", styles["subheading"]))
    training_card_elements.append(Spacer(1, 3 * mm))

    first_card_added = False
    for i, card in enumerate(training_cards[:3], 1):
        strategy = card.get("strategy_name", "---")
        power = card.get("power_range", "---")
        volume = card.get("volume", "---")
        goal = card.get("adaptation_goal", "---")
        response = card.get("expected_response", "---")
        risk = card.get("risk_level", "low")
        constraint = card.get("constraint", "")  # OCCLUSION CONSTRAINT
        
        risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
        risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
        
        card_content = [
            Paragraph(f"<font size='11'><b>{i}. {strategy}</b></font>", styles["heading"]),
            Paragraph(f"<b>Moc:</b> {power} | <b>Objętość:</b> {volume}", styles["body"]),
            Paragraph(f"<b>Cel:</b> {goal}", styles["body"]),
            Paragraph(f"<font size='9' color='#7F8C8D'>Spodziewany efekt: {response}</font>", styles["body"]),
            Paragraph(f"<font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
        ]
        
        # Add occlusion constraint if present (CRITICAL for athlete safety)
        if constraint:
            card_content.append(Paragraph(
                f"<font size='8' color='#E67E22'><b>{constraint}</b></font>",
                styles["body"]
            ))
        
        card_table = Table([[card_content]], colWidths=[170 * mm])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
            ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        if not first_card_added:
            # Keep heading + first card together to prevent orphan heading
            training_card_elements.append(card_table)
            training_card_elements.append(Spacer(1, 2 * mm))
            elements.append(KeepTogether(training_card_elements))
            first_card_added = True
        else:
            elements.append(card_table)
            elements.append(Spacer(1, 2 * mm))

    if not first_card_added and training_card_elements:
        # No cards were added, still add the heading
        elements.extend(training_card_elements)

    return elements


# ============================================================================
# PAGE 2: EXECUTIVE VERDICT (1-PAGE DECISION SUMMARY)
# ============================================================================

def build_page_executive_verdict(
    canonical_physio: Dict[str, Any],
    smo2_advanced: Dict[str, Any],
    biomech_occlusion: Dict[str, Any],
    thermo_analysis: Dict[str, Any],
    cardio_advanced: Dict[str, Any],
    metadata: Dict[str, Any],
    styles: Dict
) -> List:
    """Build Page 2: EXECUTIVE VERDICT - 1-page decision summary.
    
    Contains ONLY decision boxes, numbers and conclusions. NO CHARTS.
    
    Sections:
    A. HERO BOX - Main verdict with profile
    B. GREEN BOX - Strengths (VO2max, coupling, reoxy)
    C. RED/AMBER BOX - Limiters (occlusion, thermoregulation)
    D. 3 STARTUP RISKS
    E. 3 TRAINING PRIORITIES
    F. TAGLINE
    G. TECHNICAL FOOTER
    """

    
    elements = []
    
    # ==========================================================================
    # EXTRACT DATA with fallbacks
    # ==========================================================================
    
    # Canonical physiology
    summary = canonical_physio.get("summary", {})
    vo2max = summary.get("vo2max")
    vo2max_source = summary.get("vo2max_source", "unknown")
    
    # SmO2 advanced
    hr_coupling = smo2_advanced.get("hr_coupling_r", 0)
    halftime = smo2_advanced.get("halftime_reoxy_sec")
    smo2_slope = smo2_advanced.get("slope_per_100w", 0)
    limiter_type = smo2_advanced.get("limiter_type", "unknown")
    smo2_drift = smo2_advanced.get("drift_pct", 0)
    
    # Biomech occlusion
    biomech_metrics = biomech_occlusion.get("metrics", {})
    occlusion_index = biomech_metrics.get("occlusion_index", 0)
    torque_10 = biomech_metrics.get("torque_at_minus_10")
    torque_20 = biomech_metrics.get("torque_at_minus_20")
    occlusion_level = (biomech_occlusion.get("classification") or {}).get("level", "unknown")
    
    # Thermoregulation
    thermo_metrics = thermo_analysis.get("metrics", {})
    max_core_temp = thermo_metrics.get("max_core_temp", 0)
    peak_hsi = thermo_metrics.get("peak_hsi", 0)
    
    # Cardiac drift
    cardiac_drift = thermo_analysis.get("cardiac_drift", {})
    cardiac_drift_metrics = cardiac_drift.get("metrics", {})
    ef_delta_pct = cardiac_drift_metrics.get("delta_ef_pct", cardiac_drift.get("delta_pct", 0))
    drift_classification = cardiac_drift.get("classification", "unknown")
    
    # Cardio advanced
    ef = cardio_advanced.get("efficiency_factor", 0)
    hr_drift_pct = cardio_advanced.get("hr_drift_pct", 0)
    
    # Metadata
    test_date = metadata.get("test_date", "---")
    
    # ==========================================================================
    # GENERATE PROFILE DESCRIPTION
    # ==========================================================================
    
    profile_parts = []
    
    # Central capacity
    if vo2max and vo2max > 50:
        profile_parts.append("STABILNY CENTRALNIE")
    elif vo2max:
        profile_parts.append("UMIARKOWANY CENTRALNIE")
    else:
        profile_parts.append("PROFIL NIEZNANY")
    
    # Limiters
    limiters = []
    if occlusion_level in ["high", "moderate"]:
        limiters.append("OGRANICZANY MECHANICZNIE")
    if max_core_temp > 38.0 or peak_hsi > 6:
        limiters.append("OGRANICZANY TERMICZNIE")
    if abs(smo2_drift) > 8:
        limiters.append("DRYF OBWODOWY")
    
    if limiters:
        profile_parts.extend(limiters)
    
    profile_description = ", ".join(profile_parts)
    
    # Confidence score - get it first to apply confidence-aware language
    confidence_score = smo2_advanced.get("limiter_confidence", 0.5)
    
    # Generate main interpretation with confidence-aware language
    if limiter_type == "central":
        base_interp = (
            "Wydajność VO₂max jest wysoka, układ krążenia dyktuje tempo. "
            "Priorytet: rozbudowa pojemności minutowej serca."
        )
    elif limiter_type == "local":
        base_interp = (
            "Potencjał VO₂max jest wysoki, ale jego wykorzystanie ogranicza okluzja mięśniowa "
            "przy wysokim momencie obrotowym oraz narastający koszt termoregulacyjny."
        )
    else:
        base_interp = (
            "Profil mieszany: zarówno zdolność centralna jak i obwodowa wymagają równoczesnej pracy. "
            "Treningi zrównoważone dadzą najlepsze efekty."
        )
    
    # Apply confidence-aware language modifier
    main_interpretation = get_confidence_prefix(confidence_score) + base_interp + get_confidence_suffix(confidence_score)
    
    # ==========================================================================
    # A. HERO BOX - MAIN VERDICT
    # ==========================================================================
    
    elements.append(Paragraph("<font size='14'>5.3 WERDYKT FIZJOLOGICZNY</font>", styles["center"]))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Decyzyjne podsumowanie całego raportu fizjologicznego</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    hero_content = [
        [Paragraph(
            f"<font size='12' color='#FFFFFF'><b>WERDYKT GŁÓWNY</b></font>",
            styles["center"]
        )],
        [Paragraph(
            f"<font size='11' color='#F1C40F'><b>Profil wydolnościowy: {profile_description}</b></font>",
            styles["center"]
        )],
        [Spacer(1, 2 * mm)],
        [Paragraph(
            f"<font size='10' color='#FFFFFF'>{main_interpretation}</font>",
            styles["center"]
        )],
        [Spacer(1, 2 * mm)],
        [Paragraph(
            f"<font size='8' color='#BDC3C7'>"
            f"Źródła danych: VO₂max ({vo2max_source}), SmO₂, korelacja HR, temp. głęboka</font>",
            styles["center"]
        )],
        [Paragraph(
            f"<font size='9' color='{'#E74C3C' if confidence_score < 0.5 else '#F1C40F'}'>"
            f"<b>Pewność werdyktu: {confidence_score:.0%} "
            f"{'— NISKA: wynik orientacyjny, zalecana weryfikacja' if confidence_score < 0.5 else '— umiarkowana'}"
            f"</b></font>",
            styles["center"]
        )],
    ]

    hero_table = Table(hero_content, colWidths=[170 * mm])
    hero_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#1a1a2e")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(hero_table)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # A2. DECISION MATRIX (WHY / WHAT / HOW)
    # ==========================================================================
    
    elements.append(Paragraph("<b>MATRYCA DECYZJI (DLACZEGO / CO / JAK)</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    # Determine PRIMARY BOTTLENECK based on logic
    bottleneck = "MIESZANY"
    bottleneck_color = "#7F8C8D"
    
    # Logic: mechanical if torque_at_-20 < 65 Nm
    if torque_20 and torque_20 < 65:
        bottleneck = "MECHANICZNE (Okluzja)"
        bottleneck_color = "#E74C3C"
    # Logic: thermal if cardiac_drift > 15% AND core_temp > 38.0
    elif hr_drift_pct > 15 and max_core_temp > 38.0:
        bottleneck = "TERMICZNE (Obciążenie Cieplne)"
        bottleneck_color = "#F39C12"
    # Logic: central if HR-SmO2 r < -0.75 (strong negative = preserved delivery)
    elif hr_coupling < -0.75:
        bottleneck = "CENTRALNY (Pojemność Minutowa)"
        bottleneck_color = "#3498DB"
    elif limiter_type == "central":
        bottleneck = "CENTRALNY (Pojemność Minutowa)"
        bottleneck_color = "#3498DB"
    elif limiter_type == "local":
        bottleneck = "OBWODOWY (Ekstrakcja O₂)"
        bottleneck_color = "#9B59B6"
    
    # Generate WHY text based on bottleneck
    why_text = ""
    what_text = ""
    how_text = ""
    
    if "MECHANICAL" in bottleneck:
        why_text = f"Kompresja naczyniowa przy momencie >{torque_20 or 0:.0f} Nm ogranicza perfuzję mięśniową mimo dostępnego O₂ systemowego."
        what_text = "Szybszy spadek SmO₂, wcześniejsze zmęczenie nóg, utrata reaktywności na ataki."
        how_text = "Zwiększ kadencję do 95-105 rpm. Trenuj wysoko-kadencyjnie. Sprawdź ustawienie siodła."
    elif "THERMAL" in bottleneck:
        why_text = f"Core temp {max_core_temp:.1f}°C + drift {hr_drift_pct:.0f}% → redystrybucja krwi do skóry ogranicza dostawę do mięśni."
        what_text = "Postępujący spadek mocy po 45-60 min, wysokie HR przy niskiej mocy, ryzyko DNF."
        how_text = "Heat acclimation 10-14 dni. Pre-cooling przed startem. Nawodnienie 750ml/h + Na+."
    elif "CENTRAL" in bottleneck:
        why_text = f"Układ krążenia przy {vo2max or 0:.0f} ml/kg/min dyktuje limit – mięśnie mają rezerwę."
        what_text = "Limit tętna osiągany przed zmęczeniem mięśni. Płaski profil SmO₂ przy wysokim HR."
        how_text = "Interwały VO₂max (5×5 min @ 106-120% FTP). Z2 dla podniesienia SV. Hill repeats."
    elif "PERIPHERAL" in bottleneck:
        why_text = "Ekstrakcja O₂ w mięśniu jest limitem – niska kapilaryzacja lub wysoka glikoliza."
        what_text = "SmO₂ spada szybko przy submaksymalnych wysiłkach. Szybka lokalna kwasica."
        how_text = "Sweet spot + threshold work. Siła na rowerze. Trening low-cadence."
    else:
        why_text = "Brak jednoznacznego limitera – wydolność zbalansowana między systemami."
        what_text = "Równomierne obciążenie wszystkich układów. Brak dominującego ograniczenia."
        how_text = "Kontynuuj polaryzowany trening. Monitoruj wszystkie KPI równolegle."
    
    # Build 4 decision blocks - wrap text in Paragraphs for proper wrapping
    body_style = ParagraphStyle('matrix_body', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=8)
    
    matrix_rows = [
        [Paragraph("<b>GŁÓWNE OGRANICZENIE</b>", body_style), Paragraph(f"<b>{bottleneck}</b>", body_style)],
        [Paragraph("<b>DLACZEGO OGRANICZA WYDAJNOŚĆ</b>", body_style), Paragraph(why_text, body_style)],
        [Paragraph("<b>CO POWODUJE W WYŚCIGU</b>", body_style), Paragraph(what_text, body_style)],
        [Paragraph("<b>JAK TO NAPRAWIĆ</b>", body_style), Paragraph(how_text, body_style)],
    ]
    
    matrix_table = Table(matrix_rows, colWidths=[45 * mm, 130 * mm])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor("#2C3E50")),
        ('BACKGROUND', (1, 0), (1, 0), HexColor(bottleneck_color)),
        ('BACKGROUND', (1, 1), (1, -1), HexColor("#34495E")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#1a1a2e")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(matrix_table)
    elements.append(Spacer(1, 2 * mm))

    # Limiter reconciliation note — multiple methods may disagree
    limiter_methods = []
    if limiter_type in ("central", "local"):
        smo2_label = "CENTRALNY" if limiter_type == "central" else "OBWODOWY"
        limiter_methods.append(f"SmO₂ → {smo2_label}")
    if occlusion_level in ("high", "moderate"):
        limiter_methods.append(f"Biomechanika → OKLUZJA ({occlusion_level.upper()})")
    if max_core_temp > 38.0 or peak_hsi > 6:
        limiter_methods.append("Termoregulacja → OBCIĄŻENIE CIEPLNE")
    if abs(hr_coupling) > 0.7:
        coup_label = "dostarczanie O₂ centralne" if hr_coupling < -0.7 else "ekstrakcja lokalna"
        limiter_methods.append(f"HR-SmO₂ coupling → {coup_label}")

    if len(limiter_methods) > 1:
        reconciliation_note = (
            "<font size='7' color='#95A5A6'><i>Uwaga: różne metody detekcji mogą wskazywać "
            "różnych limiterów. Powyższy werdykt integruje: " +
            " | ".join(limiter_methods) +
            ". Priorytet ustalony wg hierarchii: okluzja mechaniczna > termoregulacja > "
            "profil centralny/obwodowy.</i></font>"
        )
        elements.append(Paragraph(reconciliation_note, styles["body"]))
    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # B. GREEN BOX - STRENGTHS
    # ==========================================================================
    
    elements.append(Paragraph("<b>CO JEST MOCNE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    # Prepare strength items
    strengths = []
    
    if vo2max:
        vo2_interp = "Wysoka wydolność aerobowa" if vo2max > 55 else ("Dobra wydolność" if vo2max > 45 else "Do poprawy")
        strengths.append(f"• <b>VO₂max (canonical):</b> {vo2max:.1f} ml/kg/min → {vo2_interp}")
    
    if abs(hr_coupling) > 0.5:
        coup_interp = "Silna korelacja HR-SmO₂ – układ spójny" if abs(hr_coupling) > 0.7 else "Umiarkowana korelacja"
        strengths.append(f"• <b>HR–SmO₂ coupling (r):</b> {hr_coupling:.2f} → {coup_interp}")
    
    if halftime and halftime < 60:
        ht_interp = "Szybka reoksygenacja – dobra kapilaryzacja" if halftime < 30 else "Akceptowalna reoksygenacja"
        strengths.append(f"• <b>Reoxy half-time:</b> {halftime:.0f} s → {ht_interp}")
    
    if ef > 1.8:
        ef_interp = "Wysoka efektywność sercowa"
        strengths.append(f"• <b>Efficiency Factor:</b> {ef:.2f} W/bpm → {ef_interp}")
    
    if not strengths:
        strengths.append("• Brak wyróżniających się mocnych stron w danych")
    
    # Strength conclusion
    strength_conclusion = "Wniosek: układ krążenia jest gotowy na dalszą intensyfikację treningową."
    
    strength_text = "<br/>".join(strengths) + f"<br/><br/><i>{strength_conclusion}</i>"
    
    green_style = ParagraphStyle('green_box', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    green_box = Table(
        [[Paragraph(strength_text, green_style)]],
        colWidths=[170 * mm]
    )
    green_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#27AE60")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(green_box)
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # C. RED/AMBER BOX - LIMITERS (wrapped in KeepTogether to prevent page break)
    # ==========================================================================
    
    # Build limiter elements first, then wrap in KeepTogether
    limiter_elements = []
    
    limiter_elements.append(Paragraph("<b>CO OGRANICZA WYNIK</b>", styles["subheading"]))
    limiter_elements.append(Spacer(1, 2 * mm))
    
    # --- OCCLUSION LIMITER ---
    occlusion_items = []
    occlusion_items.append("<b>1. OKLUZJA MECHANICZNA</b>")
    occlusion_items.append(f"• Moment przy −10% SmO₂: {torque_10:.0f} Nm" if torque_10 else "• Moment przy −10% SmO₂: ---")
    occlusion_items.append(f"• Moment przy −20% SmO₂: {torque_20:.0f} Nm" if torque_20 else "• Moment przy −20% SmO₂: ---")
    occlusion_items.append(f"• Occlusion Index: {occlusion_index:.3f}" if occlusion_index else "• Occlusion Index: ---")
    occlusion_items.append("<i>Moc dostępna centralnie, ale ograniczona przez kompresję naczyń w mięśniu.</i>")
    
    occlusion_color = "#E74C3C" if occlusion_level == "high" else ("#F39C12" if occlusion_level == "moderate" else "#7F8C8D")
    
    red_style = ParagraphStyle('red_box', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    occlusion_box = Table(
        [[Paragraph("<br/>".join(occlusion_items), red_style)]],
        colWidths=[82 * mm]
    )
    occlusion_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(occlusion_color)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    # --- THERMOREGULATION LIMITER ---
    thermo_items = []
    thermo_items.append("<b>2. TERMOREGULACJA</b>")
    thermo_items.append(f"• Max temp. głęboka: {max_core_temp:.1f} °C" if max_core_temp > 0 else "• Max temp. głęboka: ---")
    thermo_items.append(f"• Peak HSI: {peak_hsi:.1f}" if peak_hsi else "• Peak HSI: ---")
    thermo_items.append(f"• Cardiac Drift (ΔEF): {ef_delta_pct:+.1f}%" if ef_delta_pct is not None else "• Cardiac Drift: ---")
    thermo_items.append("<i>Wzrost temperatury zwiększa koszt utrzymania mocy i przyspiesza dryf serca.</i>")
    
    thermo_color = "#E74C3C" if max_core_temp > 38.5 or peak_hsi > 8 else ("#F39C12" if max_core_temp > 38.0 or peak_hsi > 6 else "#7F8C8D")
    
    thermo_box = Table(
        [[Paragraph("<br/>".join(thermo_items), red_style)]],
        colWidths=[82 * mm]
    )
    thermo_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(thermo_color)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    # Two columns
    limiter_row = Table([[occlusion_box, thermo_box]], colWidths=[85 * mm, 85 * mm])
    limiter_row.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    limiter_elements.append(limiter_row)
    
    # Wrap in KeepTogether to prevent page break
    elements.append(KeepTogether(limiter_elements))
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # D. 3 STARTUP RISKS
    # ==========================================================================
    
    elements.append(Paragraph("<b>3 NAJWAŻNIEJSZE RYZYKA STARTOWE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    risks = []
    
    # Risk 1: Power drop
    if abs(ef_delta_pct) > 15 or max_core_temp > 38.0:
        risks.append("Ryzyko spadku mocy w drugiej połowie wysiłku z powodu narastającego kosztu termoregulacyjnego")
    else:
        risks.append("Umiarkowane ryzyko spadku mocy przy wydłużonych wysiłkach – monitoruj EF")
    
    # Risk 2: SmO2 collapse
    if occlusion_level == "high" or abs(smo2_slope) > 6:
        risks.append("Ryzyko załamania SmO₂ przy niskiej kadencji / wysokim momencie obrotowym")
    else:
        risks.append("Profil okluzyjny w normie – zachowaj ostrożność przy ekstremalnych momentach")
    
    # Risk 3: HR drift
    if abs(hr_drift_pct) > 8:
        risks.append("Ryzyko nieproporcjonalnego wzrostu HR przy stałej mocy (cardiac drift >8%)")
    else:
        risks.append("Stabilność HR w zakresie normy – kontynuuj obecną strategię nawodnienia")
    
    risk_style = ParagraphStyle('risk_box', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    risk_text = "<br/>".join([f"<b>{i+1}.</b> {r}" for i, r in enumerate(risks)])
    
    risk_box = Table(
        [[Paragraph(risk_text, risk_style)]],
        colWidths=[170 * mm]
    )
    risk_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#2C3E50")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(risk_box)
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # E. 3 TRAINING PRIORITIES
    # ==========================================================================
    
    elements.append(Paragraph("<b>3 PRIORYTETY TRENINGOWE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    priorities = []
    
    # Priority 1: Based on limiter
    if limiter_type == "central":
        priorities.append({
            "name": "VO₂max (centralnie)",
            "example": "4–6 × 4 min @ 106–120% FTP, Cel: zwiększyć rzut serca bez wzrostu kosztu EF"
        })
    else:
        priorities.append({
            "name": "Baza aerobowa (objętość)",
            "example": "3–4h @ Z2 (60–75% FTP), Cel: rozbudowa kapilaryzacji i objętości wyrzutowej"
        })
    
    # Priority 2: Occlusion reduction
    if occlusion_level in ["high", "moderate"]:
        priorities.append({
            "name": "Redukcja okluzji (kadencja, SmO₂)",
            "example": "Treningi @ 95–105 rpm, unikaj momentów >50 Nm, monitoruj SmO₂ w czasie rzeczywistym"
        })
    else:
        priorities.append({
            "name": "Siła wytrzymałościowa",
            "example": "4×8 min @ 50–60 rpm pod LT1, Cel: poprawa rekrutacji włókien wolnokurczliwych"
        })
    
    # Priority 3: Heat adaptation
    if max_core_temp > 38.0 or peak_hsi > 6:
        priorities.append({
            "name": "Adaptacja cieplna",
            "example": "10–14 dni treningu w cieple (sauna post-workout), nawodnienie 500–750 ml/h + elektrolity"
        })
    else:
        priorities.append({
            "name": "Strategia nawodnienia",
            "example": "500 ml/h minimum, CHO 60–80g/h podczas wysiłków >90 min"
        })
    
    priority_style = ParagraphStyle('priority', parent=styles["body"], fontSize=9)
    
    for i, p in enumerate(priorities):
        prio_text = f"<b>PRIORYTET {i+1} — {p['name']}</b><br/>• {p['example']}"
        prio_box = Table(
            [[Paragraph(prio_text, priority_style)]],
            colWidths=[170 * mm]
        )
        prio_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor("#ECF0F1")),
            ('BOX', (0, 0), (-1, -1), 1, HexColor("#3498DB")),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(prio_box)
        elements.append(Spacer(1, 2 * mm))
    
    # ==========================================================================
    # F. TAGLINE
    # ==========================================================================
    
    elements.append(Spacer(1, 2 * mm))
    tagline_style = ParagraphStyle('tagline', parent=styles["center"], fontSize=10, textColor=HexColor("#7F8C8D"))
    elements.append(Paragraph(
        "<i>Ten raport pokazuje nie 'ile możesz', ale 'dlaczego tracisz' – i jak to naprawić.</i>",
        tagline_style
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # G. TECHNICAL FOOTER
    # ==========================================================================
    
    footer_style = ParagraphStyle('footer', parent=styles["body"], fontSize=7, textColor=HexColor("#95A5A6"))
    footer_text = (
        f"<b>Typ testu:</b> Ramp Test | "
        f"<b>Metodologia:</b> Ventilatory & BreathRate + SmO₂ + Core Temp | "
        f"<b>Źródło VO₂max:</b> {vo2max_source} | "
        f"<b>System:</b> Tri Dashboard v{RAMP_METHOD_VERSION}"
    )
    elements.append(Paragraph(footer_text, footer_style))
    
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
    is_conditional: bool = False,
    vt1_onset_watts: Optional[int] = None,
    rcp_onset_watts: Optional[int] = None,
    training_zones: Optional[Dict[str, Any]] = None
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
    method_version = metadata.get("method_version", RAMP_METHOD_VERSION)
    
    elements.append(Paragraph("1. PODSUMOWANIE WYKONAWCZE", styles["title"]))
    elements.append(Paragraph("<font size='14'>1.1 RAPORT POTESTOWY</font>", styles["center"]))
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
    
    # Calculate Upper Aerobic range as Z3 Tempo: midpoint(VT1_onset, RCP_onset) → RCP_onset
    if vt1_onset_watts and rcp_onset_watts:
        z3_lower = int((vt1_onset_watts + rcp_onset_watts) / 2)
        upper_aerobic_range = f"{z3_lower}–{int(rcp_onset_watts)} W"
    else:
        upper_aerobic_range = "brak danych"
    
    # Use Paragraph cells for automatic text wrapping
    cell_style = ParagraphStyle('cell', parent=styles["body"], fontSize=9, leading=11,
                                alignment=0, fontName=FONT_FAMILY)
    cell_style_bold = ParagraphStyle('cell_bold', parent=cell_style, fontName=FONT_FAMILY_BOLD)
    cell_style_center = ParagraphStyle('cell_center', parent=cell_style, alignment=1)
    header_style = ParagraphStyle('header_white', parent=cell_style_bold,
                                  textColor=HexColor("#FFFFFF"), alignment=1)

    data = [
        [Paragraph("Parametr", header_style),
         Paragraph("Wartość", header_style),
         Paragraph("Interpretacja", header_style)],
        [Paragraph("VT1 (Próg wentylacyjny 1)", cell_style_bold),
         Paragraph(f"~{vt1_watts} W (+/- 15 W)", cell_style_center),
         Paragraph("Strefa równowagi tlenowej", cell_style)],
        [Paragraph("VT2 (Próg wentylacyjny 2)", cell_style_bold),
         Paragraph(f"~{vt2_watts} W (+/- 12 W)", cell_style_center),
         Paragraph("Strefa progowa", cell_style)],
        [Paragraph("Zakres Upper Aerobic*", cell_style_bold),
         Paragraph(upper_aerobic_range, cell_style_center),
         Paragraph("Pośrednia VT1→VT2 (inny model niż strefy poniżej)", cell_style)],
        [Paragraph("Critical Power (CP)", cell_style_bold),
         Paragraph(f"{cp_watts} W", cell_style_center),
         Paragraph("Moc krytyczna", cell_style)],
        [Paragraph("W' (Rezerwa)", cell_style_bold),
         Paragraph(f"{w_prime_kj} kJ", cell_style_center),
         Paragraph("Rezerwa anaerobowa", cell_style)],
    ]

    table = Table(data, colWidths=[55 * mm, 40 * mm, 75 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === ZONES TABLE (6-zone VT2-based model, matching UI) ===
    elements.append(Paragraph("Strefy Treningowe", styles["heading"]))
    elements.append(Spacer(1, 3 * mm))

    if training_zones and training_zones.get("power"):
        pz = training_zones["power"]
        hz = training_zones["hr"]
        desc = training_zones["description"]

        zone_keys = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
        zones_data = [["Strefa", "Moc [W]", "Tętno [bpm]", "Fizjologia"]]

        for zk in zone_keys:
            p_low, p_high = pz.get(zk, (0, 0))
            hr_range = hz.get(zk, (None, None))
            hr_str = f"{hr_range[0]} – {hr_range[1]}" if hr_range[0] is not None else "—"
            zones_data.append([zk, f"{p_low} – {p_high} W", hr_str, desc.get(zk, "")])

        zones_table = Table(zones_data, colWidths=[25 * mm, 40 * mm, 35 * mm, 45 * mm])
    else:
        zones_data = [
            ["Strefa", "Moc [W]", "Tętno [bpm]", "Fizjologia"],
            ["Z1", "-", "-", "Regeneracja"],
            ["Z2", "-", "-", "Baza tlenowa"],
            ["Z3", "-", "-", "Tempo / Sweet Spot"],
            ["Z4", "-", "-", "Próg FTP"],
            ["Z5", "-", "-", "VO2max"],
            ["Z6", "-", "-", "Beztlenowa"],
        ]
        zones_table = Table(zones_data, colWidths=[25 * mm, 40 * mm, 35 * mm, 45 * mm])

    zones_table.setStyle(get_table_style())
    elements.append(zones_table)
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph(
        "Strefy obliczone na podstawie VT2 (100% = moc krytyczna). "
        "Przed zastosowaniem skonsultuj je z trenerem, który może dostosować je do Twoich celów.",
        styles["small"]
    ))
    
    return elements


def build_page_test_profile(
    metadata: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 1.2: Test Profile (chart and protocol description).
    
    Contains:
    - Ramp profile chart
    - Test protocol description
    """
    elements = []
    
    elements.append(Paragraph("<font size='14'>1.2 PRZEBIEG TESTU</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))
    
    # === RAMP PROFILE CHART ===
    if figure_paths and "ramp_profile" in figure_paths:
        elements.extend(_build_chart(figure_paths["ramp_profile"], "", styles, max_height_mm=130))
        elements.append(Spacer(1, 4 * mm))
    
    # === TEST PROTOCOL DESCRIPTION ===
    # Extract protocol info from metadata (populated from UI manual inputs)
    test_start_power = metadata.get("test_start_power", "---")
    test_end_power = metadata.get("test_end_power", metadata.get("pmax_watts", "---"))
    test_duration = metadata.get("test_duration", "---")
    step_increment = metadata.get("step_increment", "20")

    elements.append(Paragraph(
        "<font size='8' color='#7F8C8D'>"
        "Test wykonywany do odmowy, każdy interwał trwał 3 minuty. "
        f"Zwiększenie obciążenia w każdym interwale +{step_increment}W. "
        f"Początek testu rozpoczął się od wartości {test_start_power} W, "
        f"koniec testu nastąpił na wartości {test_end_power} W. "
        f"Test trwał łącznie {test_duration}."
        "</font>",
        styles["center"]
    ))
    
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
    
    elements.append(Paragraph("2. PROGI METABOLICZNE", styles["title"]))
    elements.append(Paragraph("<font size='14'>2.1 SZCZEGÓŁY VT1 / VT2</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))
    
    # === EXPLANATION ===
    elements.append(Paragraph(
        "Progi zostały wykryte na podstawie zmian w wentylacji (oddychaniu) podczas testu. "
        "<b>Uwaga:</b> VT1/VT2 to progi <i>wentylacyjne</i>, powiązane z progami mleczanowymi "
        "(LT1/LT2), ale <b>nie identyczne</b> z nimi. VT1 typowo wypada ~10-15 W poniżej LT1 "
        "u wytrenowanych kolarzy (Pallares et al. 2016). Wartości podane jako zakresy "
        "odzwierciedlają niepewność metody detekcji z VE (Kim et al. 2021, ACSM/ESSA 2024).",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>VT1 (Próg wentylacyjny 1):</b> Moment, gdy organizm zaczyna intensywniej pracować. "
        "Możesz jechać komfortowo przez wiele godzin. Powiązany z LT1, ale nie identyczny.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "<b>VT2 (Próg wentylacyjny 2 / RCP):</b> Punkt kompensacji oddechowej, powyżej którego wysiłek staje się bardzo ciężki. "
        "Oddychasz ciężko, nie możesz swobodnie mówić. Odpowiada w przybliżeniu LT2/MLSS.",
        styles["body"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === THRESHOLDS TABLE ===
    elements.append(Paragraph("Tabela Progów", styles["heading"]))
    
    vt1_watts = thresholds.get("vt1_watts", "brak danych")
    vt1_range = thresholds.get("vt1_range_watts", "brak danych")
    vt1_hr = thresholds.get("vt1_hr", "brak danych")
    vt1_ve = thresholds.get("vt1_ve", "brak danych")
    vt1_br = thresholds.get("vt1_br", "brak danych")
    
    vt2_watts = thresholds.get("vt2_watts", "brak danych")
    vt2_range = thresholds.get("vt2_range_watts", "brak danych")
    vt2_hr = thresholds.get("vt2_hr", "brak danych")
    vt2_ve = thresholds.get("vt2_ve", "brak danych")
    vt2_br = thresholds.get("vt2_br", "brak danych")
    
    def format_thresh(mid, rng):
        if mid == "brak danych": return mid
        if rng == "brak danych": return f"~{mid}"
        return f"{rng} (środek: {mid})"
        
    def fmt(val):
        if val == "brak danych": return val
        try:
            return f"{float(val):.0f}"
        except (ValueError, TypeError):
            return str(val)

    data = [
        ["Próg", "Moc [W]", "HR [bpm]", "VE [L/min]", "BR [br/min]"],
        ["VT1 (Próg wentylacyjny 1)", format_thresh(vt1_watts, vt1_range), fmt(vt1_hr), fmt(vt1_ve), fmt(vt1_br)],
        ["VT2 (Próg went. 2 / RCP)", format_thresh(vt2_watts, vt2_range), fmt(vt2_hr), fmt(vt2_ve), fmt(vt2_br)],
    ]
    
    table = Table(data, colWidths=[42 * mm, 42 * mm, 28 * mm, 28 * mm, 28 * mm])
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

    
    elements = []
    smo2_advanced = smo2_data.get("advanced_metrics", {})
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='14'>3.3 OKSYGENACJA MIĘŚNIOWA (SmO₂)</font>",
        styles['center']
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
    
    slope_color = "#E74C3C" if slope < -10 else ("#F39C12" if slope < -5 else "#2ECC71")
    slope_interp = "Szybka (>10%/100W)" if slope < -10 else ("Umiarkowana (5-10%)" if slope < -5 else "Łagodna (<5%)")
    card1 = build_metric_card("DESATURACJA", f"{slope:.1f}", "%/100W", slope_interp, slope_color)

    if halftime:
        ht_color = "#2ECC71" if halftime < 20 else ("#F39C12" if halftime <= 45 else "#E74C3C")
        ht_interp = "Szybki (<20s)" if halftime < 20 else ("Typowy po rampie (20-45s)" if halftime <= 45 else f"Wolny (>45s)")
        card2 = build_metric_card("REOKSYGENACJA", f"{halftime:.0f}", "sekund (po rampie)", ht_interp, ht_color)
    else:
        card2 = build_metric_card("REOKSYGENACJA", "---", "sekund", "Brak danych", "#7F8C8D")

    coup_color = "#3498DB" if abs(coupling) > 0.6 else ("#F39C12" if abs(coupling) > 0.3 else "#2ECC71")
    coup_interp = "Silna (centralna)" if abs(coupling) > 0.6 else ("Umiarkowana" if abs(coupling) > 0.3 else "Słaba (lokalna)")
    card3 = build_metric_card("KORELACJA HR", f"{coupling:.2f}", "r-Pearson", coup_interp, coup_color)
    
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
    mechanism_names = {"local": "OBWODOWY", "central": "CENTRALNY", "metabolic": "MIESZANY", "unknown": "NIEOKREŚLONY"}
    mechanism_icons = {"local": "💪", "central": "❤️", "metabolic": "🔥", "unknown": "❓"}
    
    mech_color = HexColor(mechanism_colors.get(limiter_type, "#7F8C8D"))
    mech_name = mechanism_names.get(limiter_type, "UNDEFINED")
    mech_icon = mechanism_icons.get(limiter_type, "❓")
    
    elements.append(Paragraph("<b>DOMINUJĄCY MECHANIZM DOSTAWY TLENU</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{mech_icon} {mech_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(limiter_conf)}</font>", styles["center"]),
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
            line_with_confidence = get_confidence_prefix(limiter_conf) + line + get_confidence_suffix(limiter_conf)
            elements.append(Paragraph(line_with_confidence, styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 3. SmO2 THRESHOLDS (compact cards)
    # ==========================================================================
    
    lt1 = smo2_manual.get("lt1_watts", "---")
    lt2 = smo2_manual.get("lt2_watts", "---")
    lt1_hr = smo2_manual.get("lt1_hr", "---")
    lt2_hr = smo2_manual.get("lt2_hr", "---")
    lt1_smo2_pct = smo2_manual.get("lt1_smo2", "---")
    lt2_smo2_pct = smo2_manual.get("lt2_smo2", "---")

    def fmt(val):
        if val in ("brak danych", None, "---"): return "---"
        try: return f"{float(val):.0f}"
        except (ValueError, TypeError): return str(val)

    def fmt_pct(val):
        if val in ("brak danych", None, "---", 0, "0", "0.0"): return ""
        try: return f"SmO₂: {float(val):.1f}%"
        except (ValueError, TypeError): return ""

    elements.append(Paragraph("<b>PROGI OKSYGENACJI MIĘŚNIOWEJ</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    lt1_smo2_line = fmt_pct(lt1_smo2_pct)
    lt2_smo2_line = fmt_pct(lt2_smo2_pct)

    lt1_card = [Paragraph("<font size='9' color='#7F8C8D'>SmO₂-T1 (powiązany z LT1)</font>", styles["center"]),
                Paragraph(f"<font size='14'><b>{fmt(lt1)} W</b></font>", styles["center"]),
                Paragraph(f"<font size='9'>@ {fmt(lt1_hr)} bpm</font>", styles["center"])]
    if lt1_smo2_line:
        lt1_card.append(Paragraph(f"<font size='9' color='#1ABC9C'>{lt1_smo2_line}</font>", styles["center"]))

    lt2_card = [Paragraph("<font size='9' color='#7F8C8D'>SmO₂-T2 (powiązany z LT2)</font>", styles["center"]),
                Paragraph(f"<font size='14'><b>{fmt(lt2)} W</b></font>", styles["center"]),
                Paragraph(f"<font size='9'>@ {fmt(lt2_hr)} bpm</font>", styles["center"])]
    if lt2_smo2_line:
        lt2_card.append(Paragraph(f"<font size='9' color='#E74C3C'>{lt2_smo2_line}</font>", styles["center"]))
    
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
        elements.append(Paragraph("<b>DECYZJE TRENINGOWE NA PODSTAWIE KINETYKI O₂</b>", styles["subheading"]))
        elements.append(Spacer(1, 3 * mm))
        
        expected = {"local": ["Wzrost bazowego SmO₂ o 2-4%", "Szybsza reoksygenacja", "Zmniejszenie slope"],
                    "central": ["Wyższe SmO₂ przy tym samym HR", "Lepsza korelacja", "Stabilniejsza saturacja"],
                    "metabolic": ["Późniejszy drop point", "Mniejszy slope", "Lepsza tolerancja kwasu"]}
        exp_list = expected.get(limiter_type, ["Poprawa ogólna", "Stabilniejsza saturacja", "Lepszy klirens"])
        
        for i, rec in enumerate(recommendations[:5]):
            exp_resp = exp_list[i] if i < len(exp_list) else "Poprawa wydolności"
            card_content = [Paragraph(f"<font size='10'><b>{i+1}. {rec}</b></font>", styles["body"]),
                           Paragraph(f"<font size='8' color='#27AE60'>Spodziewany efekt: {exp_resp}</font>", styles["body"])]
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
    elements.append(Paragraph(f"<font size='8' color='#7F8C8D'>Jakość danych: </font><font size='8' color='{quality_color}'><b>{quality_label}</b></font>", styles["body"]))
    
    # ==========================================================================
    # 5. REFERENCE BENCHMARK TABLE (MINI-BENCHMARK)
    # ==========================================================================
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("<b>WZORZEC PORÓWNAWCZY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    # Interpret metrics for benchmark (Vasquez Bonilla 2023: -5 to -15%/100W trained cyclists)
    slope_interp_full = "Szybka desaturacja (ref: -5 do -15%/100W wytren.)" if slope < -10 else ("Umiarkowana (typowa dla wytrenowanych)" if slope < -5 else "Łagodna — dobra zdolność oksydacyjna")
    if halftime:
        # Arnold et al. 2024: HRT vastus lat trained = 8-17s (submaks), post-ramp slower
        ht_interp_full = "Szybki (post-ramp <20s = wysoka kapil.)" if halftime < 20 else ("Typowy po teście maks. (20-45s)" if halftime <= 45 else "Wolny — sugeruje niską gęstość kapil.")
    else:
        ht_interp_full = "Brak danych"
    coup_interp_full = "Silna dominacja serca (centralny)" if abs(coupling) > 0.6 else ("Zrównoważona" if abs(coupling) > 0.3 else "Dominacja obwodowa (lokalna)")
    
    bench_data = [
        ["Metryka", "Twoja wartość", "Interpretacja kliniczna"],
        ["Desaturacja SmO₂", f"{slope:.1f} %/100W", slope_interp_full],
        ["Czas reoksygenacji", f"{halftime:.0f} s" if halftime else "---", ht_interp_full],
        ["Korelacja HR-SmO₂", f"{coupling:.2f}", coup_interp_full],
    ]
    
    bench_table = Table(bench_data, colWidths=[40 * mm, 40 * mm, 85 * mm])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 9 * mm),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 1), (-1, 1), HexColor("#333333")),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 2), (-1, 2), HexColor("#333333")),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 3), (-1, 3), HexColor("#333333")),
    ]))
    elements.append(bench_table)
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 6. CONCLUSIVE STATEMENT (Links SmO2 with Biomechanics)
    # ==========================================================================
    
    # Generate conclusive statement based on limiter type
    if limiter_type == "central":
        conclusion = (
            "<b>WNIOSEK:</b> Poprawa VO2max da realny wzrost mocy tylko jeśli utrzymasz "
            "niską okluzję mechaniczną. Priorytet: treningi Z2/Z3 + interwały <95% HR max."
        )
        conclusion_color = "#E74C3C"
    elif limiter_type == "local":
        conclusion = (
            "<b>WNIOSEK:</b> Perfuzja mięśniowa jest limitująca - poprawa siły lub kadencji "
            "może zredukować okluzję i zwolnić desaturację. Priorytet: Strength Endurance."
        )
        conclusion_color = "#3498DB"
    else:
        conclusion = (
            "<b>WNIOSEK:</b> Balans między dostawą a zużyciem O₂ jest dobry. "
            "Kontynuuj zróżnicowany trening, monitorując SmO₂ w sesjach tempo."
        )
        conclusion_color = "#27AE60"
    
    conclusion_style = ParagraphStyle('conclusion', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    conclusion_box = Table(
        [[Paragraph(conclusion, conclusion_style)]],
        colWidths=[165 * mm]
    )
    conclusion_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(conclusion_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(conclusion_box)
    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 7. CROSS-VALIDATION VT vs SmO2 + THb CONTEXT
    # ==========================================================================

    # Cross-validation note
    vt1_cross = smo2_manual.get("lt1_watts", "---")
    vt2_cross = smo2_manual.get("lt2_watts", "---")

    cross_note_parts = []
    cross_note_parts.append(
        "<b>WALIDACJA KRZYŻOWA VT vs SmO₂:</b> "
        "Progi wentylacyjne (VT) i progi oksygenacji mięśniowej (SmO₂-T) "
        "to <i>różne sygnały fizjologiczne</i> (Feldmann et al. 2022, Perrey & Ferrari 2024). "
    )

    try:
        vt1_f = float(smo2_manual.get("lt1_watts", 0) or 0)
        vt2_f = float(smo2_manual.get("lt2_watts", 0) or 0)
        vt1_ref = float(smo2_data.get("thresholds", {}).get("vt1_watts", 0) or 0) if isinstance(smo2_data.get("thresholds"), dict) else 0
        vt2_ref = float(smo2_data.get("thresholds", {}).get("vt2_watts", 0) or 0) if isinstance(smo2_data.get("thresholds"), dict) else 0

        if vt1_f > 0 and vt1_ref > 0:
            div1 = abs(vt1_f - vt1_ref) / vt1_ref * 100
            cross_note_parts.append(
                f"Rozbieżność T1 vs VT1: {div1:.0f}% "
                f"({'dobra zgodność' if div1 < 10 else 'umiarkowana — interpretować ostrożnie' if div1 < 15 else 'KRYTYCZNA — wymagana weryfikacja'}). "
            )
    except (ValueError, TypeError):
        pass

    cross_note_parts.append(
        "Zgodność ICC SmO₂-T1 vs VT1 = 0.53 (umiarkowana), T2 vs VT2 = 0.80 (dobra) wg Perrey & Ferrari 2024."
    )

    from reportlab.platypus import KeepTogether as KT
    cross_thb_block = [
        Paragraph("".join(cross_note_parts), styles["small"]),
        Spacer(1, 2 * mm),
        Paragraph(
            "<b>Nota THb:</b> THb (Total Hemoglobin) odzwierciedla objętość krwi w mięśniu "
            "(Alvares et al. 2020). Rosnący THb = wazodylatacja; spadający = okluzja naczyniowa. "
            "THb pomaga odróżnić zwiększoną ekstrakcję O₂ od zmniejszonego przepływu.",
            styles["small"]
        ),
    ]
    elements.append(KT(cross_thb_block))

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
    
    elements.append(Paragraph("<font size='14'>2.5 KRZYWA MOCY (PDC)</font>", styles["center"]))
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
    
    # Additional theory - FAKT / INTERPRETACJA / AKCJA structure
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        "<font color='#3498DB'><b>● FAKT:</b></font> Każdy skok powyżej CP kosztuje konkretną ilość dżuli z W'. "
        "Przy W'=15kJ i mocy 50W powyżej CP, wystarczy na ~5 min powyżej progu.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<font color='#9B59B6'><b>● INTERPRETACJA:</b></font> Regeneracja W' zachodzi TYLKO poniżej CP. "
        "Im głębiej poniżej CP, tym szybsza regeneracja (ok. 1-2% W'/s przy głębokim Z2).",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<font color='#27AE60'><b>● AKCJA:</b></font> W ataku kalkuluj koszt: krótki intensywny atak (30s @ +100W) kosztuje ~3kJ. "
        "Czy masz rezerwę? Decyduj na podstawie danych, nie intuicji.",
        styles["body"]
    ))

    # === W' RECONSTITUTION INFO ===
    try:
        w_prime_val = float(w_prime_kj) if w_prime_kj not in ("brak danych", None, "") else 0
        cp_val = float(cp_watts) if cp_watts not in ("brak danych", None, "") else 0
    except (ValueError, TypeError):
        w_prime_val = 0
        cp_val = 0

    if w_prime_val > 0 and cp_val > 0:
        elements.append(Spacer(1, 6 * mm))
        elements.append(Paragraph("<b>REKONSTYTUCJA W' (Caen et al. 2021)</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))
        # Biexponential recovery model parameters (Caen 2021)
        tau_fast = 300  # ~5min fast component
        tau_slow = 900  # ~15min slow component

        recon_data = [
            ["Czas poniżej CP", "% W' odzyskane", "W' dostępne", "Praktyczny kontekst"],
            ["30 sekund", "~10%", f"~{w_prime_val * 0.10:.1f} kJ", "Mikro-recovery w peletonie"],
            ["2 minuty", "~30%", f"~{w_prime_val * 0.30:.1f} kJ", "Przerwa między atakami"],
            ["5 minut", "~55%", f"~{w_prime_val * 0.55:.1f} kJ", "Recovery po podjeździe"],
            ["10 minut", "~75%", f"~{w_prime_val * 0.75:.1f} kJ", "Pełna regeneracja po ataku"],
            ["20 minut", "~90%", f"~{w_prime_val * 0.90:.1f} kJ", "Prawie pełne recovery"],
        ]

        recon_table = Table(recon_data, colWidths=[35 * mm, 30 * mm, 30 * mm, 70 * mm])
        recon_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2C3E50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), FONT_FAMILY),
            ('FONTNAME', (0, 0), (-1, 0), FONT_FAMILY_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
            ('ROWHEIGHT', (0, 0), (-1, -1), 9 * mm),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#f8f9fa")),
            ('BACKGROUND', (2, 1), (2, -1), HexColor("#E8F6EF")),
        ]))
        elements.append(recon_table)
        elements.append(Spacer(1, 2 * mm))
        elements.append(Paragraph(
            "<font size='7' color='#95A5A6'><i>Wartości orientacyjne — rzeczywista rekonstytucja zależy od "
            "intensywności recovery (im niżej pod CP, tym szybciej), stanu zmęczenia i treningu. "
            "Model biexponential (Caen et al. 2021, MSSE).</i></font>",
            styles["body"]
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
    - VT1 explanation with values + CHO/metabolic info + example workouts
    - VT2 explanation with values + CHO/metabolic info + example workouts
    - Tempo zone explanation + substrate usage
    - CP practical usage + pacing examples
    """
    elements = []
    
    elements.append(Paragraph("<font size='14'>2.2 CO OZNACZAJĄ TE WYNIKI?</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))
    
    vt1_watts_raw = thresholds.get("vt1_watts", "brak danych")
    vt2_watts_raw = thresholds.get("vt2_watts", "brak danych")
    cp_watts_raw = cp_model.get("cp_watts", "brak danych")
    
    # Convert to numeric values for calculations
    try:
        vt1_num = float(vt1_watts_raw) if vt1_watts_raw not in [None, "brak danych", "---"] else None
    except (ValueError, TypeError):
        vt1_num = None
        
    try:
        vt2_num = float(vt2_watts_raw) if vt2_watts_raw not in [None, "brak danych", "---"] else None
    except (ValueError, TypeError):
        vt2_num = None
        
    try:
        cp_num = float(cp_watts_raw) if cp_watts_raw not in [None, "brak danych", "---"] else None
    except (ValueError, TypeError):
        cp_num = None
    
    # Format display values
    vt1_watts = f"{vt1_num:.0f}" if vt1_num else "brak danych"
    vt2_watts = f"{vt2_num:.0f}" if vt2_num else "brak danych"
    cp_watts = f"{cp_num:.0f}" if cp_num else "brak danych"
    
    # === VT1 ===
    elements.append(Paragraph("Próg wentylacyjny 1 (VT1)", styles["heading"]))
    elements.append(Paragraph(
        f"Twój pierwszy próg wentylacyjny wynosi około <b>{vt1_watts} W (+/- 15 W)</b>. "
        "To moc, przy której możesz jechać komfortowo przez wiele godzin. "
        "Oddychasz spokojnie, możesz swobodnie rozmawiać. "
        "Treningi poniżej VT1 budują bazę tlenową i służą regeneracji. "
        "<i>Uwaga: VT1 jest powiązany z pierwszym progiem mleczanowym (LT1), ale typowo "
        "wypada ~10-15 W niżej niż LT1 wyznaczony z krwi (Pallares et al. 2016).</i>",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Metabolizm:</b> Poniżej VT1 spalasz głównie tłuszcze (~60-70% energii z lipidów). "
        "Zużycie węglowodanów (CHO) wynosi ok. <b>30-50g/h</b>. "
        "To strefa idealna do długich treningów bez uzupełniania CHO.",
        styles["small"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"<b>Przykładowe jednostki:</b> "
        f"• Regeneracja: 60-90 min @ {int(vt1_num*0.65) if vt1_num else '?'}-{int(vt1_num*0.75) if vt1_num else '?'} W | "
        f"• Baza aerobowa: 2-4h @ {int(vt1_num*0.8) if vt1_num else '?'}-{vt1_watts} W",
        styles["small"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === VT2 ===
    elements.append(Paragraph("Próg wentylacyjny 2 / RCP (VT2)", styles["heading"]))
    elements.append(Paragraph(
        f"Twój drugi próg wentylacyjny (punkt kompensacji oddechowej) wynosi około <b>{vt2_watts} W (+/- 12 W)</b>. "
        "Powyżej tej mocy wysiłek staje się bardzo wymagający. "
        "Oddychasz ciężko, nie możesz swobodnie mówić. "
        "Treningi powyżej VT2 rozwijają VO₂max, ale wymagają pełnej regeneracji. "
        "<i>VT2/RCP odpowiada w przybliżeniu drugiemu progowi mleczanowemu (LT2/MLSS).</i>",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Metabolizm:</b> Powyżej VT2 dominuje glikoliza beztlenowa. "
        "Zużycie CHO rośnie do <b>90-120g/h</b>. Akumulacja mleczanu prowadzi do szybkiego zmęczenia. "
        "Wysiłek powyżej VT2 można utrzymać max 20-60 min.",
        styles["small"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"<b>Przykładowe jednostki:</b> "
        f"• VO₂max: 5×5 min @ {int(vt2_num*1.05) if vt2_num else '?'}-{int(vt2_num*1.15) if vt2_num else '?'} W (4 min odpoczynku) | "
        f"• Tolerancja mleczanu: 3×8 min @ {vt2_watts} W (5 min odpoczynku)",
        styles["small"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === TEMPO ZONE ===
    elements.append(Paragraph("Strefa Tempo/Sweet Spot", styles["heading"]))
    elements.append(Paragraph(
        f"Strefa między <b>~{vt1_watts}</b> a <b>~{vt2_watts} W</b> to Twoja strefa 'tempo'. "
        "Intensywne treningi wytrzymałościowe na granicy progu wentylacyjnego. "
        "W tej strefie możesz spędzać znaczną część czasu treningowego bez nadmiernego zmęczenia.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Metabolizm:</b> W strefie tempo następuje przejście z lipidów na CHO. "
        "Spalanie tłuszczów spada do ~30-40%, rośnie zużycie CHO (<b>60-80g/h</b>). "
        "To strefa kluczowa dla rozwoju FatMax i przesunięcia krzywej spalania.",
        styles["small"]
    ))
    elements.append(Spacer(1, 2 * mm))
    
    tempo_mid = ""
    if vt1_num and vt2_num:
        tempo_mid = f"{int((vt1_num + vt2_num) / 2)}"
    
    elements.append(Paragraph(
        f"<b>Przykładowe jednostki:</b> "
        f"• Sweet Spot: 2×20 min @ {tempo_mid if tempo_mid else '?'}-{int(vt2_num*0.94) if vt2_num else '?'} W | "
        f"• Tempo długie: 1×45-60 min @ {vt1_watts}-{tempo_mid if tempo_mid else '?'} W",
        styles["small"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === CP ===
    elements.append(Paragraph("Critical Power", styles["heading"]))
    elements.append(Paragraph(
        f"CP ({cp_watts} W) to matematyczne przybliżenie Twojej mocy progowej. "
        "Możesz używać tej wartości do planowania interwałów i wyznaczania stref treningowych. "
        "CP jest przydatne do pacing'u podczas zawodów i długich treningów.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Praktyka:</b> CP reprezentuje moc możliwą do utrzymania przez ~30-60 min. "
        "Treningi @ CP rozwijają próg mleczanowy i wydolność tlenową. "
        "Używaj CP jako górnej granicy dla długich, rytmicznych interwałów.",
        styles["small"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        f"<b>Przykładowe jednostki:</b> "
        f"• Under/Over: 3×12 min (2 min @ {int(cp_num*0.92) if cp_num else '?'} W / 1 min @ {int(cp_num*1.08) if cp_num else '?'} W) | "
        f"• Threshold: 2×20 min @ {int(cp_num*0.95) if cp_num else '?'}-{cp_watts} W",
        styles["small"]
    ))
    
    return elements


# ============================================================================
# PAGE: CARDIOVASCULAR COST DIAGNOSTIC (PREMIUM)
# ============================================================================

def build_page_cardiovascular(cardio_data: Dict[str, Any], styles: Dict) -> List:
    """Build Cardiovascular Cost Diagnostic page - PREMIUM."""

    
    elements = []
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='14'>3.2 UKŁAD SERCOWO-NACZYNIOWY</font>",
        styles['center']
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
    card1 = build_card("MOC PULSOWA", f"{pp:.2f}", "W/bpm", pp_interp, pp_color)
    
    # Efficiency Factor
    ef_color = "#2ECC71" if ef > 1.8 else ("#F39C12" if ef > 1.4 else "#E74C3C")
    ef_interp = "Wysoki" if ef > 1.8 else ("Średni" if ef > 1.4 else "Niski")
    card2 = build_card("WSP. EFEKTYWNOŚCI", f"{ef:.2f}", "W/bpm", ef_interp, ef_color)
    
    # HR Drift
    drift_color = "#2ECC71" if drift < 3 else ("#F39C12" if drift < 6 else "#E74C3C")
    drift_interp = "Stabilny" if drift < 3 else ("Drift" if drift < 6 else "Wysoki Drift")
    card3 = build_card("DRYF HR", f"{drift:.1f}", "%", drift_interp, drift_color)
    
    # HR Recovery or CCI
    if recovery and recovery > 0:
        rec_color = "#2ECC71" if recovery > 25 else ("#F39C12" if recovery > 15 else "#E74C3C")
        rec_interp = "Szybki" if recovery > 25 else ("Średni" if recovery > 15 else "Wolny")
        card4 = build_card("REGENERACJA HR", f"{recovery:.0f}", "bpm/min", rec_interp, rec_color)
    else:
        cci_color = "#2ECC71" if cci < 0.15 else ("#F39C12" if cci < 0.25 else "#E74C3C")
        cci_interp = "Efektywny" if cci < 0.15 else ("Średni" if cci < 0.25 else "Wysoki koszt")
        card4 = build_card("CCI (avg)", f"{cci:.3f}", "bpm/W", cci_interp, cci_color)
    
    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'TOP')]))
    elements.append(cards_row)
    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # 2. CCI METRIC PANEL
    # ==========================================================================

    elements.append(Paragraph("<b>INDEKS KOSZTU SERCOWO-NACZYNIOWEGO (CCI)</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    cci_text = f"<b>CCI = {cci:.4f}</b> bpm/W – koszt tętna na jednostkę mocy."
    if cci_bp:
        cci_text += f" <b>Breakpoint</b> przy {cci_bp:.0f}W – punkt załamania efektywności."
    elements.append(Paragraph(cci_text, styles["body"]))
    elements.append(Spacer(1, 3 * mm))
    
    # ==========================================================================
    # 3. EFFICIENCY VERDICT PANEL
    # ==========================================================================
    
    elements.append(Paragraph("<b>WERDYKT EFEKTYWNOŚCI SERCOWO-NACZYNIOWEJ</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    status_colors = {"efficient": "#2ECC71", "compensating": "#F39C12", "decompensating": "#E74C3C", "unknown": "#7F8C8D"}
    status_names = {"efficient": "EFEKTYWNY", "compensating": "KOMPENSUJĄCY", "decompensating": "DEKOMPENSUJĄCY", "unknown": "NIEOKREŚLONY"}
    status_icons = {"efficient": "✓", "compensating": "⚠", "decompensating": "✗", "unknown": "?"}
    
    st_color = HexColor(status_colors.get(status, "#7F8C8D"))
    st_name = status_names.get(status, "NIEOKREŚLONY")
    st_icon = status_icons.get(status, "?")
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{st_icon} {st_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(confidence)}</font>", styles["center"]),
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
    elements.append(Spacer(1, 2 * mm))

    # Interpretation
    if interpretation:
        for line in interpretation.split('\n')[:2]:
            line_with_confidence = get_confidence_prefix(confidence) + line + get_confidence_suffix(confidence)
            elements.append(Paragraph(line_with_confidence, styles["body"]))
    elements.append(Spacer(1, 3 * mm))

    # ==========================================================================
    # 4. DECISION CARDS
    # ==========================================================================

    if recommendations:
        elements.append(Paragraph("<b>DECYZJE TRENINGOWE I ŚRODOWISKOWE</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))

        type_colors = {"TRENINGOWA": "#3498DB", "ŚRODOWISKOWA": "#9B59B6", "REGENERACJA": "#1ABC9C", "WYDAJNOŚĆ": "#2ECC71", "DIAGNOSTYCZNA": "#E74C3C"}

        for rec in recommendations[:4]:
            rec_type = rec.get("type", "TRENINGOWA")
            action = rec.get("action", "---")
            expected = rec.get("expected", "---")
            risk = rec.get("risk", "low")
            
            type_color = type_colors.get(rec_type, "#7F8C8D")
            risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
            risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
            
            card_content = [
                Paragraph(f"<font size='9' color='{type_color}'><b>[{rec_type}]</b></font> {action}", styles["body"]),
                Paragraph(f"<font size='8' color='#27AE60'>Spodziewany efekt: {expected}</font> | <font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
            ]
            card_table = Table([[card_content]], colWidths=[170 * mm])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), COLORS["background"]),
                ('BOX', (0, 0), (-1, -1), 0.5, COLORS["border"]),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(card_table)
            elements.append(Spacer(1, 1.5 * mm))

    # ==========================================================================
    # 5. HR RECOVERY KINETICS
    # ==========================================================================
    if recovery and recovery > 0:
        elements.append(Spacer(1, 3 * mm))
        elements.append(Paragraph("<b>KINETYKA REGENERACJI HR</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))

        # Classify recovery quality (Daanen et al. 2012; Buchheit 2014)
        if recovery > 30:
            rec_class = "DOSKONAŁA"
            rec_color = "#2ECC71"
            rec_note = ("Szybki powrót HR po wysiłku wskazuje na dobrą reaktywność "
                       "parasympatyczną i wysoki poziom wytrenowania aerobowego.")
        elif recovery > 20:
            rec_class = "DOBRA"
            rec_color = "#27AE60"
            rec_note = ("Regeneracja HR w normie dla wytrenowanego zawodnika. "
                       "Układ autonomiczny dobrze reaguje na zaprzestanie wysiłku.")
        elif recovery > 12:
            rec_class = "ŚREDNIA"
            rec_color = "#F39C12"
            rec_note = ("Umiarkowana regeneracja HR może wskazywać na niedobór treningu "
                       "aerobowego lub skumulowane zmęczenie. Monitoruj HRV w kolejnych dniach.")
        else:
            rec_class = "WOLNA"
            rec_color = "#E74C3C"
            rec_note = ("Wolna regeneracja HR (<12 bpm/min) może sygnalizować overreaching, "
                       "odwodnienie lub niedostateczną bazę aerobową. Rozważ dodatkowy odpoczynek.")

        rec_box_style = ParagraphStyle('rec_box', parent=styles["body"],
                                        textColor=HexColor("#FFFFFF"), fontSize=9)
        rec_content = (
            f"<b>HR Recovery 1 min: {recovery:.0f} bpm/min → {rec_class}</b><br/>"
            f"{rec_note}<br/>"
            f"<font size='7'><i>Ref: Buchheit 2014 — HRR1 >25 bpm = bardzo dobra fitness aerobowa; "
            f"<12 bpm = ryzyko overreaching/niedostatecznej bazy Z2.</i></font>"
        )
        rec_box = Table(
            [[Paragraph(rec_content, rec_box_style)]],
            colWidths=[170 * mm]
        )
        rec_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(rec_color)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(rec_box)

    return elements


# ============================================================================
# PAGE: BREATHING & METABOLIC CONTROL DIAGNOSTIC (PREMIUM)
# ============================================================================

def build_page_ventilation(vent_data: Dict[str, Any], styles: Dict) -> List:
    """Build Breathing & Metabolic Control Diagnostic page - PREMIUM."""

    
    elements = []
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "3. DIAGNOSTYKA UKŁADÓW",
        styles['title']
    ))
    elements.append(Paragraph(
        "<font size='14'>3.1 KONTROLA ODDYCHANIA</font>",
        styles['center']
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
    elements.append(Spacer(1, 2 * mm))

    # Ramp test context note for peak ventilatory values
    elements.append(Paragraph(
        "<font size='8' color='#7F8C8D'><i>Uwaga: wartości szczytowe (VE max, RR max) dotyczą "
        "końcowej fazy rampy — przy wyczerpaniu. Nie należy ich porównywać z normami "
        "dla wysiłku stacjonarnego. Kluczowa jest dynamika wzrostu (VE slope) i stosunek "
        "VE/RR (głębokość oddechu), które lepiej odzwierciedlają ekonomię wentylacyjną.</i></font>",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # 2. BREATHING PATTERN
    # ==========================================================================
    
    pattern_colors = {"efficient": "#2ECC71", "shallow": "#E74C3C", "hyperventilation": "#F39C12", "mixed": "#7F8C8D", "unknown": "#7F8C8D"}
    pattern_names = {"efficient": "EFEKTYWNY ODDECH", "shallow": "PŁYTKI/PANIKA", "hyperventilation": "HIPERWENTYLACJA", "mixed": "WZÓR MIESZANY", "unknown": "NIEOKREŚLONY"}
    
    elements.append(Paragraph("<b>WYKRYWANIE WZORCA ODDECHOWEGO</b>", styles["subheading"]))
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
    
    elements.append(Paragraph("<b>WERDYKT KONTROLI WENTYLACJI</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    status_colors = {"controlled": "#2ECC71", "compensatory": "#F39C12", "unstable": "#E74C3C", "unknown": "#7F8C8D"}
    status_names = {"controlled": "KONTROLOWANY", "compensatory": "KOMPENSACYJNY", "unstable": "NIESTABILNY", "unknown": "NIEOKREŚLONY"}
    status_icons = {"controlled": "✓", "compensatory": "⚠", "unstable": "✗", "unknown": "?"}
    
    st_color = HexColor(status_colors.get(status, "#7F8C8D"))
    st_name = status_names.get(status, "NIEOKREŚLONY")
    st_icon = status_icons.get(status, "?")
    
    verdict_content = [
        Paragraph(f"<font color='white'><b>{st_icon} {st_name}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(confidence)}</font>", styles["center"]),
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
            line_with_confidence = get_confidence_prefix(confidence) + line + get_confidence_suffix(confidence)
            elements.append(Paragraph(line_with_confidence, styles["body"]))
    
    if ve_bp:
        elements.append(Paragraph(f"<b>VE Breakpoint:</b> {ve_bp:.0f}W – punkt załamania kontroli wentylacyjnej", styles["body"]))

    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # 3b. TIDAL VOLUME DECOMPOSITION (VE = TV × RR)
    # ==========================================================================
    elements.append(Paragraph("<b>DEKOMPOZYCJA WENTYLACJI (VE = TV × RR)</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    # TV = VE/RR at different intensities
    tv_avg = ve_avg / rr_avg if rr_avg > 0 else 0
    tv_max = ve_max / rr_max if rr_max > 0 else 0

    if tv_avg > 0:
        # Determine breathing strategy
        if tv_max < tv_avg * 0.85:
            strategy = "PŁYTKI-SZYBKI"
            strategy_color = "#E74C3C"
            strategy_note = ("Przy wysokiej intensywności VE rośnie głównie przez RR (częstotliwość), "
                           "nie TV (głębokość). Wskazuje na limit mechaniczny klatki piersiowej lub "
                           "nieefektywny wzorzec oddechowy.")
        elif tv_max > tv_avg * 1.15:
            strategy = "GŁĘBOKI-WOLNY"
            strategy_color = "#27AE60"
            strategy_note = ("Przy wysokiej intensywności TV rośnie proporcjonalnie do RR — "
                           "efektywny wzorzec oddechowy z dobrą mechaniką klatki piersiowej.")
        else:
            strategy = "ZBALANSOWANY"
            strategy_color = "#3498DB"
            strategy_note = ("TV i RR rosną proporcjonalnie. Typowy wzorzec dla umiarkowanie "
                           "wytrenowanych zawodników.")

        tv_data = [
            ["Parametr", "Średnia", "Szczyt", "Interpretacja"],
            ["Objętość oddechowa (TV)", f"{tv_avg:.2f} L", f"{tv_max:.2f} L",
             f"{'↑' if tv_max > tv_avg else '↓'} {abs(tv_max - tv_avg) / tv_avg * 100:.0f}% zmiana"],
            ["Częstość (RR)", f"{rr_avg:.0f} /min", f"{rr_max:.0f} /min",
             f"↑ {(rr_max - rr_avg) / rr_avg * 100:.0f}% wzrost"],
            ["Strategia oddechowa", "", "", strategy],
        ]

        tv_table = Table(tv_data, colWidths=[45 * mm, 30 * mm, 30 * mm, 60 * mm])
        tv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2C3E50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), FONT_FAMILY),
            ('FONTNAME', (0, 0), (-1, 0), FONT_FAMILY_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
            ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#f8f9fa")),
            ('BACKGROUND', (3, 3), (3, 3), HexColor(strategy_color)),
            ('TEXTCOLOR', (3, 3), (3, 3), HexColor("#FFFFFF")),
            ('FONTNAME', (3, 3), (3, 3), FONT_FAMILY_BOLD),
        ]))
        elements.append(tv_table)
        elements.append(Spacer(1, 2 * mm))
        elements.append(Paragraph(f"<font size='8'><i>{strategy_note}</i></font>", styles["body"]))

    elements.append(Spacer(1, 4 * mm))

    # ==========================================================================
    # 3c. VENTILATORY RESERVE
    # ==========================================================================
    # MVV (Maximal Voluntary Ventilation) estimation: MVV ≈ FEV1 × 40
    # Without FEV1, use population estimate: MVV ≈ 200 L/min (trained male)
    # Breathing reserve = (1 - VEmax/MVV) × 100
    mvv_estimated = 200.0  # Conservative estimate for trained male cyclist
    breathing_reserve = (1 - ve_max / mvv_estimated) * 100 if ve_max > 0 else 100
    br_color = "#27AE60" if breathing_reserve > 30 else ("#F39C12" if breathing_reserve > 15 else "#E74C3C")
    br_label = "Wystarczająca" if breathing_reserve > 30 else ("Ograniczona" if breathing_reserve > 15 else "Wyczerpana")

    elements.append(Paragraph(
        f"<b>REZERWA WENTYLACYJNA:</b> <font color='{br_color}'><b>{breathing_reserve:.0f}%</b> ({br_label})</font>"
        f" — VE max {ve_max:.0f} L/min vs szacowane MVV ~{mvv_estimated:.0f} L/min. "
        f"{'Płuca NIE są czynnikiem limitującym.' if breathing_reserve > 30 else 'Płuca MOGĄ być czynnikiem limitującym — rozważ spirometrię.'}"
        f" <font size='7' color='#95A5A6'>(MVV szacowane; do precyzyjnej oceny wymagana spirometria)</font>",
        styles["body"]
    ))

    elements.append(Spacer(1, 6 * mm))

    # ==========================================================================
    # 4. DECISION CARDS
    # ==========================================================================
    
    if recommendations:
        elements.append(Paragraph("<b>DECYZJE TRENINGOWE WENTYLACJI</b>", styles["subheading"]))
        elements.append(Spacer(1, 3 * mm))
        
        type_colors = {"TRENINGOWA": "#3498DB", "TECHNICZNA": "#9B59B6", "WYDAJNOŚĆ": "#2ECC71", "INTENSYWNOŚĆ": "#1ABC9C", "PILNA": "#E74C3C", "DIAGNOSTYKA": "#F39C12", "MEDYCZNA": "#E74C3C"}
        
        for rec in recommendations[:5]:
            rec_type = rec.get("type", "TRENINGOWA")
            action = rec.get("action", "---")
            expected = rec.get("expected", "---")
            risk = rec.get("risk", "low")
            
            type_color = type_colors.get(rec_type, "#7F8C8D")
            risk_color = "#2ECC71" if risk == "low" else ("#F39C12" if risk == "medium" else "#E74C3C")
            risk_label = "NISKIE" if risk == "low" else ("ŚREDNIE" if risk == "medium" else "WYSOKIE")
            
            card_content = [
                Paragraph(f"<font size='9' color='{type_color}'><b>[{rec_type}]</b></font> {action}", styles["body"]),
                Paragraph(f"<font size='8' color='#27AE60'>Spodziewany efekt: {expected}</font> | <font size='8' color='{risk_color}'>Ryzyko: {risk_label}</font>", styles["body"]),
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

    
    elements = []
    
    profile = metabolic_data.get("profile", {})
    block = metabolic_data.get("training_block", {})
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    elements.append(Paragraph(
        "<font size='14'>2.4 SILNIK METABOLICZNY</font>",
        styles['center']
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
    
    elements.append(Paragraph("<b>PROFIL METABOLICZNY</b>", styles["subheading"]))
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
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
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
    card2 = build_metric_card("VLaMax", f"{vlamax:.2f}", "mmol/L/s", vla_color, "model metaboliczny")
    card3 = build_metric_card("CP / FTP", f"{cp:.0f}", "W", "#3498DB", "obliczone")
    
    # Ratio card - show n/a if insufficient data
    if ratio and ratio > 0 and vo2max > 0:
        ratio_color = "#2ECC71" if ratio > 130 else ("#F39C12" if ratio > 90 else "#E74C3C")
        ratio_val = f"{ratio:.0f}"
    else:
        ratio_color = "#7F8C8D"
        ratio_val = "n/a"
    card4 = build_metric_card("VO₂/VLa RATIO", ratio_val, "(ratio)", ratio_color, "analiza wieloczynnikowa")
    
    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),  # Align cards to top for consistent height
    ]))
    elements.append(cards_row)
    elements.append(Spacer(1, 3 * mm))
    
    # Phenotype badge
    phenotype_colors = {"diesel": "#27AE60", "allrounder": "#3498DB", "puncher": "#F39C12", "sprinter": "#E74C3C", "climber": "#8E44AD"}
    phenotype_names = {"diesel": "DIESEL (Aerobic Dominant)", "allrounder": "ALLROUNDER", "puncher": "PUNCHER", "sprinter": "SPRINTER (Glycolytic)", "climber": "CLIMBER (Low VLaMax)"}
    
    phenotype_badge = Paragraph(
        f"<font color='white'><b>FENOTYP: {phenotype_names.get(phenotype, phenotype.upper())}</b></font>",
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
    
    elements.append(Paragraph("<b>DIAGNOZA OGRANICZEŃ</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    limiter_colors = {"aerobic": "#E74C3C", "glycolytic": "#F39C12", "mixed": "#3498DB", "unknown": "#7F8C8D"}
    limiter_names = {"aerobic": "WYDOLNOŚĆ TLENOWA", "glycolytic": "DOMINACJA GLIKOLITYCZNA", "mixed": "MIESZANY / ZBALANSOWANY", "unknown": "NIEOKREŚLONY"}
    
    limiter_content = [
        Paragraph(f"<font color='white'><b>{limiter_names.get(limiter, 'NIEOKREŚLONY')}</b></font>", styles["center"]),
        Paragraph(f"<font size='10' color='white'>jakość sygnału: {_signal_quality_label(limiter_conf)}</font>", styles["center"]),
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
    
    elements.append(Paragraph("<b>GŁÓWNY CEL ADAPTACYJNY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))
    
    target_colors = {"increase_vo2max": "#E74C3C", "lower_vlamax": "#27AE60", "maintain_balance": "#3498DB"}
    target_names = {"increase_vo2max": "↑ ZWIĘKSZ VO₂max", "lower_vlamax": "↓ OBNIŻ VLaMax", "maintain_balance": "↔ UTRZYMAJ BALANS"}
    
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
        # Skip first line which duplicates target name in badge above
        lines = strategy_interp.split('\n')
        for line in lines[1:4]:  # Start from line 2, take up to 3 lines
            elements.append(Paragraph(line, styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # 4. TRAINING BLOCK
    # ==========================================================================
    
    sessions = block.get("sessions", [])
    block_name = block.get("name", "Training Block")
    weeks = block.get("duration_weeks", 6)
    
    elements.append(Paragraph(f"<b>BLOK TRENINGOWY: {block_name}</b>", styles["subheading"]))
    elements.append(Paragraph(f"<font size='9' color='#7F8C8D'>{weeks} tygodni | {block.get('primary_focus', '')}</font>", styles["body"]))
    elements.append(Spacer(1, 3 * mm))
    
    for i, session in enumerate(sessions[:5]):
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
            Paragraph(f"<font size='8' color='#27AE60'>Spodziewany efekt: SmO₂ {exp_smo2}, HR {exp_hr}</font>", styles["body"]),
            Paragraph(f"<font size='8' color='#E74C3C'>⚠ Sygnał ostrz.: {failure}</font>", styles["body"]),
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
            elements.append(Spacer(1, 4 * mm))  # Extra spacing before regress section
            elements.append(Paragraph("<font color='#E74C3C'><b>✗ Sygnały regresu / overreaching:</b></font>", styles["body"]))
            for kpi in kpi_regress[:3]:
                elements.append(Paragraph(f"<font size='9'>• {kpi}</font>", styles["body"]))
    
    return elements


# ============================================================================
# PAGE: LIMITER RADAR (20 MIN FTP ANALYSIS)
# ============================================================================

def build_page_limiter_radar(
    limiter_data: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build page for 20min FTP Limiter Radar analysis.
    
    Contains:
    - Diagnosis table (Serce, Płuca, Mięśnie, Moc percentages)
    - Limiting factor identification
    - Training suggestions
    """

    
    elements = []
    
    # Title
    elements.append(Paragraph("4. LIMITERY I OBCIĄŻENIE CIEPLNE", styles["title"]))
    elements.append(Paragraph("<font size='14'>4.1 RADAR OBCIĄŻENIA SYSTEMÓW</font>", styles["center"]))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Analiza limiterów fizjologicznych dla 20 min (FTP)</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # Extract data
    window = limiter_data.get("window", "20 min (FTP)")
    peak_power = limiter_data.get("peak_power", 0)
    pct_hr = limiter_data.get("pct_hr", 0)
    pct_ve = limiter_data.get("pct_ve", 0)
    pct_smo2 = limiter_data.get("pct_smo2_util", 0)
    pct_power = limiter_data.get("pct_power", 0)
    limiting_factor = limiter_data.get("limiting_factor", "unknown")
    interpretation = limiter_data.get("interpretation", {})
    
    # ==========================================================================
    # RADAR CHART (if available)
    # ==========================================================================
    
    if figure_paths and "limiter_radar" in figure_paths:
        elements.extend(_build_chart(figure_paths["limiter_radar"], f"Profil Obciążenia: {window} ({peak_power:.0f} W)", styles))
        elements.append(Spacer(1, 6 * mm))
    
    # ==========================================================================
    # DIAGNOSIS TABLE
    # ==========================================================================
    
    elements.append(Paragraph(f"<b>DIAGNOZA: {window}</b>", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    
    # Determine status for each system
    def get_status(system):
        if limiting_factor == system:
            return f"<font color='#E74C3C'>●</font> Limiter", "#E74C3C"
        return f"<font color='#27AE60'>●</font> OK", "#27AE60"
    
    hr_status_str, hr_color = get_status("Serce")
    ve_status_str, ve_color = get_status("Płuca")
    smo2_status_str, smo2_color = get_status("Mięśnie")
    
    hr_status_para = Paragraph(hr_status_str, styles["center"])
    ve_status_para = Paragraph(ve_status_str, styles["center"])
    smo2_status_para = Paragraph(smo2_status_str, styles["center"])
    
    # Build diagnosis table
    header = ["System", "Wartość", "Interpretacja"]
    rows = [
        header,
        ["Serce", f"{pct_hr:.1f}% HRmax", hr_status_para],
        ["Płuca", f"{pct_ve:.1f}% VEmax", ve_status_para],
        ["Mięśnie", f"{pct_smo2:.1f}% Desat", smo2_status_para],
        ["Moc", f"{pct_power:.0f}% CP", "—"],
    ]
    
    table = Table(rows, colWidths=[50 * mm, 50 * mm, 70 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLORS["primary"]),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, 0), FONT_FAMILY_BOLD),
        ('FONTNAME', (0, 1), (-1, -1), FONT_FAMILY),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS["border"]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        # Color the Limiter row
        ('BACKGROUND', (0, 1), (0, 1), HexColor(hr_color) if limiting_factor == "Serce" else HexColor("#F8F9FA")),
        ('BACKGROUND', (0, 2), (0, 2), HexColor(ve_color) if limiting_factor == "Płuca" else HexColor("#F8F9FA")),
        ('BACKGROUND', (0, 3), (0, 3), HexColor(smo2_color) if limiting_factor == "Mięśnie" else HexColor("#F8F9FA")),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 4 * mm))
    
    # Main limiter
    elements.append(Paragraph(f"<b>Główny Limiter: {limiting_factor}</b>", styles["subheading"]))
    elements.append(Spacer(1, 4 * mm))
    
    # ==========================================================================
    # INTERPRETATION BOX
    # ==========================================================================
    
    title = interpretation.get("title", f"Ograniczenie: {limiting_factor}")
    description = interpretation.get("description", "")
    suggestions = interpretation.get("suggestions", [])
    
    # Color based on limiter type
    limiter_colors = {
        "Serce": "#E74C3C",
        "Płuca": "#3498DB", 
        "Mięśnie": "#F39C12"
    }
    box_color = limiter_colors.get(limiting_factor, "#7F8C8D")
    
    interp_content = f"<b>{title}</b><br/><br/>{description}<br/><br/><b>Sugestie:</b>"
    for s in suggestions:
        interp_content += f"<br/>• {s}"
    
    interp_style = ParagraphStyle('interp', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    interp_box = Table(
        [[Paragraph(interp_content, interp_style)]],
        colWidths=[170 * mm]
    )
    interp_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(box_color)),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(interp_box)
    
    return elements


# ============================================================================
# PAGE 5: STREFY TRENINGOWE
# ============================================================================

def build_page_zones(
    thresholds: Dict[str, Any],
    styles: Dict,
    training_zones: Optional[Dict[str, Any]] = None
) -> List:
    """Build Page 5: Training Zones (6-zone VT2-based model).

    Contains:
    - Zones table with power, HR, and physiology
    - Zone descriptions
    """
    elements = []

    elements.append(Paragraph("<font size='14'>1.2 STREFY TRENINGOWE</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))

    if training_zones and training_zones.get("power"):
        pz = training_zones["power"]
        hz = training_zones["hr"]
        desc = training_zones["description"]

        zone_keys = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
        data = [["Strefa", "Moc [W]", "Tętno [bpm]", "Fizjologia"]]

        for zk in zone_keys:
            p_low, p_high = pz.get(zk, (0, 0))
            hr_range = hz.get(zk, (None, None))
            hr_str = f"{hr_range[0]} – {hr_range[1]}" if hr_range[0] is not None else "—"
            data.append([zk, f"{p_low} – {p_high} W", hr_str, desc.get(zk, "")])
    else:
        data = [
            ["Strefa", "Moc [W]", "Tętno [bpm]", "Fizjologia"],
            ["Z1", "-", "-", "Regeneracja"],
            ["Z2", "-", "-", "Baza tlenowa"],
            ["Z3", "-", "-", "Tempo / Sweet Spot"],
            ["Z4", "-", "-", "Próg FTP"],
            ["Z5", "-", "-", "VO2max"],
            ["Z6", "-", "-", "Beztlenowa"],
        ]

    table = Table(data, colWidths=[25 * mm, 40 * mm, 35 * mm, 45 * mm])
    table.setStyle(get_table_style())
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === USAGE NOTE ===
    elements.append(Paragraph(
        "Strefy obliczone na podstawie VT2 (100% = moc krytyczna). "
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
    
    elements.append(Paragraph("<font size='14'>5.5 OGRANICZENIA INTERPRETACJI</font>", styles["center"]))
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

def _build_chart(chart_path: str, title: str, styles: Dict, max_height_mm: int = 90) -> List:
    """Build a chart section with image.

    Args:
        chart_path: Path to chart PNG file
        title: Section title
        styles: Paragraph styles dictionary
        max_height_mm: Maximum image height in mm (default 90)

    Returns:
        List of flowables
    """
    from reportlab.platypus import KeepTogether

    chart_elements = []

    # Title for the chart
    chart_elements.append(Paragraph(title, styles["subheading"]))

    # 1. Check if file exists
    if not chart_path or not os.path.exists(chart_path):
        logger.warning(f"PDF Layout: Chart file missing for '{title}' at path: {chart_path}")
        chart_elements.append(Paragraph("Wykres niedostępny", styles["small"]))
        return [KeepTogether(chart_elements)]

    # 2. Embed image
    try:
        available_width = PAGE_WIDTH - 2 * MARGIN
        img = Image(chart_path)

        # Scale to fit width
        aspect = img.imageHeight / img.imageWidth
        img_width = min(available_width, 150 * mm)
        img_height = img_width * aspect

        # Limit height
        if img_height > max_height_mm * mm:
            img_height = max_height_mm * mm
            img_width = img_height / aspect

        img.drawWidth = img_width
        img.drawHeight = img_height

        chart_elements.append(img)
    except Exception as e:
        logger.error(f"PDF Layout: Error embedding chart '{title}' from {chart_path}: {e}")

    # Wrap in KeepTogether to prevent title/chart separation across pages
    return [KeepTogether(chart_elements)]


def _build_education_block(title: str, content: str, styles: Dict) -> List:
    """Helper to build a consistent education block with 'Dlaczego to ma znaczenie?'."""
    elements = []
    
    # Label
    label = Paragraph("", styles["small"])  # Removed educational label
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
    """Build Page: Model Metaboliczny (2.3)."""

    from reportlab.platypus import TableStyle
    elements = []
    
    elements.append(Paragraph("<font size='14'>2.3 MODEL METABOLICZNY</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))
    
    # Intro
    elements.append(Paragraph(
        "Twoja wydajność zależy od interakcji trzech systemów energetycznych. "
        "Zrozumienie ich pozwala na precyzyjne dopasowanie treningu.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # VO2max
    elements.append(Paragraph("VO₂max (System Tlenowy)", styles["heading"]))
    elements.append(Paragraph(
        "Maksymalna ilość tlenu, jaką Twój organizm może przyswoić. "
        "Jest to Twój „silnik diesla” – odpowiada za moc na długim dystansie. "
        "Wysokie VO₂max jest kluczowe dla każdego kolarza wytrzymałościowego.",
        styles["body"]
    ))
    
    # VLaMax
    elements.append(Paragraph("VLaMax (System Glikolityczny)", styles["heading"]))
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
    
    elements.append(Paragraph("Typy Zawodników i Strategie", styles["heading"]))
    data = [
        ["Typ", "VO2max", "VLaMax", "Charakterystyka"],
        ["Sprinter", "Średni", "Wysoki", "Dynamika, punch, sprinty"],
        ["Climber", "Wysoki", "Niski", "Długie wspinaczki, tempo"],
        ["Time Trialist", "Wysoki", "Niski", "Równe tempo, aerodynamika"],
        ["Puncheur", "Wysoki", "Średni", "Ataki, krótkie górki"]
    ]
    t = Table(data, colWidths=[30*mm, 30*mm, 30*mm, 80*mm])
    # Table should use DejaVuSans for Polish chars if needed, though ASCII used above
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 8 * mm),
    ]))
    elements.append(t)
    
    return elements


# ============================================================================
# NEW PAGE: PROTOKÓŁ I HIERARCHIA SYGNAŁÓW (Page 20 simulation)
# ============================================================================

def build_page_protocol(
    styles: Dict
) -> List:
    """Build Page for Signal Hierarchy and Protocol information."""
    elements = []
    
    elements.append(Paragraph("<font size='14'>5.4 PROTOKÓŁ TESTU</font>", styles["center"]))
    elements.append(Spacer(1, 8 * mm))
    
    # Extensive Theory Section
    elements.append(Paragraph("<b>FUNDAMENT METODOLOGII</b>", styles["heading"]))
    elements.append(Spacer(1, 4 * mm))
    
    theory_text = (
        "Współczesna analiza fizjologiczna opiera się na integracji wielu sygnałów, "
        "jednak nie wszystkie mają taką samą wagę diagnostyczną. Kluczem do sukcesu "
        "jest zrozumienie 'opóźnienia' i 'reaktywności' każdego z nich.<br/><br/>"
        "<b>1. WENTYLACJA (VE):</b> Nasz główny sygnał (Golden Standard). Dlaczego? Ponieważ "
        "wentylacja reaguje niemal natychmiast na zmiany pH krwi i poziomu CO2. Jest to "
        "bezpośrednie odbicie metabolizmu całego organizmu.<br/><br/>"
        "<b>2. OKSYDACJA MIĘŚNIOWA (SmO2):</b> Sygnał o najwyższej reaktywności lokalnej. "
        "Pozwala zobaczyć co dzieje się bezpośrednio w pracującym mięśniu (dostawa vs zapotrzebowanie). "
        "SmO2 reaguje najszybciej na zmiany obciążenia, ale jest sygnałem punktowym.<br/><br/>"
        "<b>3. TĘTNO (HR):</b> Sygnał najbardziej opóźniony (Heart Rate Lag). HR jest sterowane "
        "częścią autonomiczną układu nerwowego i potrzebuje czasu, aby 'dogonić' zapotrzebowanie "
        "tlenowe. HR jest doskonałym wskaźnikiem kosztu ustrojowego, ale słabym narzędziem "
        "do precyzyjnej detekcji progów w krótkich interwałach."
    )
    elements.append(Paragraph(theory_text, styles["body"]))
    elements.append(Spacer(1, 10 * mm))
    
    elements.append(Paragraph("<b>ZNACZENIE PROTOKOŁU</b>", styles["heading"]))
    elements.append(Spacer(1, 4 * mm))
    
    protocol_text = (
        "Długość kroku (rampy) jest krytyczna. Standardowy protokół 1-minutowy często "
        "prowadzi do przeszacowania mocy progowej, ponieważ sygnały (szczególnie HR i VE) "
        "nie zdążą osiągnąć stanu stabilnego (Steady State).<br/><br/>"
        "W naszej analizie stosujemy matematyczną korektę opóźnień lub zalecamy protokoły "
        "o długości 2-3 minut na stopień, co pozwala na pełną stabilizację kinetyki gazowej "
        "i parametrów krążeniowych. Dzięki temu wyznaczone progi VT1/VT2 oraz SmO2-LT "
        "są powtarzalne i mają realne przełożenie na trening w terenie."
    )
    elements.append(Paragraph(protocol_text, styles["body"]))
    
    return elements


# ============================================================================
# PAGE 6: TERMOREGULACJA
# ============================================================================

def build_page_thermal(
    thermo_data: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Page 6: Thermal Analysis - Enhanced with metrics and recommendations."""

    # Removed redundant local import

    from reportlab.platypus import TableStyle
    
    elements = []
    
    elements.append(Paragraph("<font size='14'>4.3 TERMOREGULACJA</font>", styles["center"]))
    elements.append(Paragraph("<font size='10' color='#7F8C8D'>Dynamika temperatury, tolerancja cieplna, rekomendacje</font>", styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    elements.append(Paragraph(
        "Ciepło jest cichym zabójcą wydajności. Wzrost temperatury głębokiej (Core Temp) "
        "powoduje przekierowanie krwi do skóry (chłodzenie), co zabiera tlen pracującym mięśniom.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # Chart 1: Core Temp vs HSI
    if figure_paths and "thermal_hsi" in figure_paths:
        elements.extend(_build_chart(figure_paths["thermal_hsi"], "Temp. Głęboka vs Indeks Zmęczenia (HSI)", styles))
        elements.append(Spacer(1, 6 * mm))
    
    # === KEY NUMBERS TABLE (Thermoregulation) ===
    # thermo_data is now passed directly
    metrics = thermo_data.get("metrics", {})
    classification = thermo_data.get("classification", {})
    
    # Check if we have actual data (not empty/default)
    has_thermo_data = bool(metrics) and metrics.get("max_core_temp", 0) > 35
    
    # Get values with proper fallbacks
    if has_thermo_data:
        max_temp = metrics.get("max_core_temp", 0)
        delta_10min = metrics.get("delta_per_10min", 0)
        time_38_0 = metrics.get("time_to_38_0_min")
        time_38_5 = metrics.get("time_to_38_5_min")
        peak_hsi = metrics.get("peak_hsi", 0)
        
        max_temp_str = f"{max_temp:.1f} C"
        delta_str = f"{delta_10min:.2f} C"
        peak_hsi_str = f"{peak_hsi:.1f}"
    else:
        max_temp_str = "---"
        delta_str = "---"
        time_38_0 = None
        time_38_5 = None
        peak_hsi_str = "---"
    
    # Classification
    tolerance = classification.get("heat_tolerance", "unknown") if has_thermo_data else "unknown"
    tolerance_color = classification.get("color", "#808080") if has_thermo_data else "#808080"
    tolerance_label = {"good": "DOBRA", "moderate": "ŚREDNIA", "poor": "SŁABA"}.get(tolerance, "BRAK DANYCH")
    
    elements.append(Paragraph("KLUCZOWE LICZBY", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    
    key_data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Max temp. głęboka", max_temp_str, "Szczytowa temperatura głęboka"],
        ["Delta temp. / 10 min", delta_str, f"Tolerancja: {tolerance_label}"],
        ["Czas do 38.0°C", f"{time_38_0:.0f} min" if time_38_0 else "---", "Próg ostrzegawczy"],
        ["Czas do 38.5°C", f"{time_38_5:.0f} min" if time_38_5 else "---", "Próg krytyczny"],
        ["Peak HSI", peak_hsi_str, "Indeks obciążenia cieplnego"],
    ]
    
    table = Table(key_data, colWidths=[45 * mm, 35 * mm, 85 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
        # Delta row gets classification color
        ('BACKGROUND', (0, 2), (-1, 2), HexColor(tolerance_color)),
        ('TEXTCOLOR', (0, 2), (-1, 2), HexColor("#FFFFFF")),
        # Other rows light
        ('BACKGROUND', (0, 1), (-1, 1), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 1), (-1, 1), HexColor("#333333")),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 3), (-1, 3), HexColor("#333333")),
        ('BACKGROUND', (0, 4), (-1, 4), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 4), (-1, 4), HexColor("#333333")),
        ('BACKGROUND', (0, 5), (-1, 5), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 5), (-1, 5), HexColor("#333333")),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 6 * mm))
    
    # === CLASSIFICATION VERDICT ===
    if tolerance == "poor":
        verdict_text = (
            "<b>SŁABA TOLERANCJA CIEPLNA</b><br/>"
            "Tempo narastania temperatury przekracza próg bezpieczny. "
            "Redystrybucja krwi do skóry konkuruje z dostawą O₂ do mięśni. "
            "Ryzyko przegrzania jest wysokie."
        )
        verdict_color = "#E74C3C"
    elif tolerance == "moderate":
        verdict_text = (
            "<b>ŚREDNIA TOLERANCJA</b><br/>"
            "Układ chłodzenia radzi sobie, ale istnieje margines do poprawy. "
            "Adaptacja cieplna nie jest pełna."
        )
        verdict_color = "#F39C12"
    else:
        verdict_text = (
            "<b>DOBRA TOLERANCJA</b><br/>"
            "Tempo narastania temperatury mieści się w normie. "
            "Układ termoregulacji skutecznie balansuje między chłodzeniem a perfuzją."
        )
        verdict_color = "#27AE60"
    
    white_style = ParagraphStyle('thermo_verdict', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    verdict_box = Table(
        [[Paragraph(verdict_text, white_style)]],
        colWidths=[165 * mm]
    )
    verdict_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(verdict_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(verdict_box)
    elements.append(Spacer(1, 6 * mm))

    # === HR/EF CONNECTION ===
    elements.append(Paragraph("<b>POŁĄCZENIE Z DRYFEM HR I EF</b>", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Mechanizm fizjologiczny:</b> Wysoka temperatura wymusza redystrybucję krwi do skóry "
        "w celu chłodzenia. Serce musi pompować większą objętość krwi, by utrzymać zarówno chłodzenie, "
        "jak i dostawę O₂ do mięśni. Efekt: wzrost HR przy stałej mocy (cardiac drift), "
        "spadek Efficiency Factor (EF). To jest kardynalny syndrom przegrzania.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Konsekwencje dla wydolności:</b> Każdy 1°C wzrostu temperatury rdzenia powoduje "
        "wzrost HR o 8-10 bpm oraz spadek VO₂max o 1.5-2%. Przy temperaturze >39°C następuje "
        "ochronne ograniczenie rekrutacji jednostek motorycznych przez OUN.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(
        "<b>Adaptacja cieplna:</b> Po 10-14 dniach treningu w cieple (60-90min @ Z2, temp >28°C) "
        "obserwuje się: wcześniejsze pocenie, niższy HR przy tej samej mocy, zmniejszony drift, "
        "oraz poprawę tolerancji cieplnej o 15-20%.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))
    
    # === TRAINING RECOMMENDATIONS ===
    elements.append(Paragraph("<b>REKOMENDACJE</b>", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    
    if tolerance == "poor":
        recommendations = [
            "PRIORYTET: Trening w cieple (heat acclimation) - 10-14 dni, 60-90min @ Z2",
            "Pre-cooling: kamizelka lodowa przed startem",
            "Nawodnienie: 500-800ml/h + elektrolity",
            "Unikaj zawodów >28°C do czasu adaptacji",
        ]
    elif tolerance == "moderate":
        recommendations = [
            "Rozważ 5-7 dni treningu w cieple przed ważnymi zawodami",
            "Chłodzenie zewnętrzne: woda na głowę co 15-20 min",
            "Kontroluj wagę przed/po treningu (max -2%)",
        ]
    else:
        recommendations = [
            "Adaptacja wystarczająca - możesz startować w cieple",
            "Utrzymuj nawodnienie 500-750ml/h + elektrolity (Na+ 500-700mg/L)",
            "Kontynuuj okresowy trening w cieple (1x/tyg)",
        ]
    
    white_rec_style = ParagraphStyle('rec_white', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    rec_data = [[Paragraph(f"* {rec}", white_rec_style)] for rec in recommendations]
    rec_table = Table(rec_data, colWidths=[165 * mm])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#16213e")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(rec_table)
    
    # ========================================================================
    # CARDIAC DRIFT ANALYSIS - PRO LAYOUT
    # ========================================================================
    # Use KeepTogether to prevent orphan heading
    drift_header = [
        Spacer(1, 6 * mm),
        Paragraph("Analiza Dryfu Efektywności (Cardiac Drift)", styles["title"]),
        Paragraph("<font size='10' color='#7F8C8D'>Dynamika EF, klasyfikacja dryfu, implikacje treningowe</font>", styles["body"]),
        Spacer(1, 4 * mm),
        Paragraph("KLUCZOWE SYGNAŁY", styles["heading"]),
    ]
    elements.append(KeepTogether(drift_header))
    
    # Get drift data from thermo_data (will be populated by persistence.py)
    drift_data = thermo_data.get("cardiac_drift", {})
    drift_metrics = drift_data.get("metrics", {})
    drift_signals = drift_data.get("key_signals", {})
    drift_class = drift_data.get("classification", {})
    drift_interp = drift_data.get("interpretation", {})
    
    # Check if we have drift data
    has_drift_data = bool(drift_metrics) and drift_metrics.get("ef_start", 0) > 0
    
    if has_drift_data:
        ef_start = drift_metrics.get("ef_start", 0)
        ef_end = drift_metrics.get("ef_end", 0)
        delta_pct = drift_metrics.get("delta_ef_pct", 0)
        ef_slope = drift_metrics.get("ef_vs_temp_slope")
        hsi_peak = drift_signals.get("hsi_peak", 0)
        smo2_drift = drift_signals.get("smo2_drift_pct", 0)
        
        drift_level = drift_class.get("drift_level", "unknown")
        drift_type = drift_class.get("drift_type", "unknown")
        drift_color = drift_class.get("color", "#808080")
        
        # Status helpers
        def get_delta_status(d):
            if abs(d) < 5: return ("STABILNY", "#27AE60")
            elif abs(d) < 10: return ("UMIARKOWANY", "#F39C12")
            else: return ("WYSOKI", "#E74C3C")
        
        def get_hsi_status(h):
            if h < 5: return ("NISKI", "#27AE60")
            elif h < 8: return ("OSTRZEŻENIE", "#F39C12")
            else: return ("KRYTYCZNY", "#E74C3C")
        
        def get_smo2_status(s):
            if abs(s) < 5: return ("STABILNY", "#27AE60")
            else: return ("DRYF OBWODOWY", "#F39C12")
        
        delta_status, delta_color = get_delta_status(delta_pct)
        hsi_status, hsi_color = get_hsi_status(hsi_peak)
        smo2_status, smo2_color = get_smo2_status(smo2_drift)
    else:
        ef_start = 0
        ef_end = 0
        delta_pct = 0
        ef_slope = None
        hsi_peak = 0
        smo2_drift = 0
        drift_level = "unknown"
        drift_type = "unknown"
        drift_color = "#808080"
        delta_status, delta_color = ("BRAK", "#808080")
        hsi_status, hsi_color = ("BRAK", "#808080")
        smo2_status, smo2_color = ("BRAK", "#808080")
    
    # === KEY SIGNALS BOX ===
    elements.append(Spacer(1, 2 * mm))

    key_signals_data = [
        ["Sygnał", "Wartość", "Status"],
        ["EF Start", f"{ef_start:.2f} W/bpm" if ef_start > 0 else "---", "BAZOWY"],
        ["EF End", f"{ef_end:.2f} W/bpm" if ef_end > 0 else "---", f"{delta_pct:+.1f}%" if has_drift_data else "---"],
        ["dEF / dC", f"{ef_slope:.3f} W/bpm/C" if ef_slope else "---", drift_type.upper() if has_drift_data else "---"],
        ["HSI Szczyt", f"{hsi_peak:.1f}" if hsi_peak > 0 else "---", hsi_status],
        ["SmO2 Dryf", f"{smo2_drift:+.1f}%" if has_drift_data else "---", smo2_status],
    ]
    
    signals_table = Table(key_signals_data, colWidths=[45 * mm, 50 * mm, 70 * mm])
    signals_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 9 * mm),
        # Row backgrounds
        ('BACKGROUND', (0, 1), (1, 1), HexColor("#e8f5e9")),
        ('TEXTCOLOR', (0, 1), (1, 1), HexColor("#333333")),
        ('BACKGROUND', (2, 1), (2, 1), HexColor("#3498DB")),
        ('TEXTCOLOR', (2, 1), (2, 1), HexColor("#FFFFFF")),
        # Delta row - color by status
        ('BACKGROUND', (0, 2), (1, 2), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 2), (1, 2), HexColor("#333333")),
        ('BACKGROUND', (2, 2), (2, 2), HexColor(delta_color)),
        ('TEXTCOLOR', (2, 2), (2, 2), HexColor("#FFFFFF")),
        # EF slope row
        ('BACKGROUND', (0, 3), (1, 3), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 3), (1, 3), HexColor("#333333")),
        ('BACKGROUND', (2, 3), (2, 3), HexColor(drift_color)),
        ('TEXTCOLOR', (2, 3), (2, 3), HexColor("#FFFFFF")),
        # HSI row
        ('BACKGROUND', (0, 4), (1, 4), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 4), (1, 4), HexColor("#333333")),
        ('BACKGROUND', (2, 4), (2, 4), HexColor(hsi_color)),
        ('TEXTCOLOR', (2, 4), (2, 4), HexColor("#FFFFFF")),
        # SmO2 row
        ('BACKGROUND', (0, 5), (1, 5), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 5), (1, 5), HexColor("#333333")),
        ('BACKGROUND', (2, 5), (2, 5), HexColor(smo2_color)),
        ('TEXTCOLOR', (2, 5), (2, 5), HexColor("#FFFFFF")),
    ]))
    elements.append(signals_table)
    elements.append(Spacer(1, 6 * mm))
    
    # === CLASSIFICATION VERDICT BOX ===
    # INTEGRATION: Cardiac drift interpretation now considers thermal verdict
    # NOTE: This is the MECHANISM diagnosis box. The RACE SIMULATION box follows separately.
    if has_drift_data:
        base_mechanism = drift_interp.get("mechanism", "")
        delta_pct_abs = abs(delta_pct)

        # Integrate thermal verdict with cardiac drift interpretation
        if tolerance == "poor" and delta_pct_abs > 5:
            verdict_text = (
                "<b>DIAGNOZA MECHANIZMU: ZMĘCZENIE CENTRALNE</b><br/>"
                f"Słaba tolerancja cieplna + dryf EF {delta_pct:+.1f}% → "
                "Redystrybucja krwi do skóry ogranicza perfuzję mięśniową. "
                "Priorytet: adaptacja cieplna przed intensyfikacją treningu."
            )
            drift_color = "#E74C3C"
        elif tolerance == "moderate" and delta_pct_abs > 7:
            verdict_text = (
                "<b>DIAGNOZA MECHANIZMU: STRES TERMICZNY</b><br/>"
                f"Umiarkowana tolerancja cieplna + dryf EF {delta_pct:+.1f}% → "
                "Obciążenie termiczne przyspiesza dryf sercowy. "
                "Zalecana adaptacja cieplna 5-10 dni przed zawodami."
            )
            drift_color = "#F39C12"
        else:
            verdict_text = (
                "<b>DIAGNOZA MECHANIZMU:</b> " +
                (f"{base_mechanism}" if base_mechanism else "Stabilność EF w akceptowalnym zakresie — normalny dryf sercowy.")
            )
    else:
        verdict_text = "<b>DIAGNOZA MECHANIZMU:</b> Brak danych dryfu — analiza wymaga danych EF (power/HR)."
        drift_color = "#808080"

    verdict_style = ParagraphStyle('verdict_white', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    verdict_box = Table(
        [[Paragraph(verdict_text, verdict_style)]],
        colWidths=[165 * mm]
    )
    verdict_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(drift_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(verdict_box)
    elements.append(Spacer(1, 6 * mm))

    # === RACE CONSEQUENCE SIMULATION ===
    # Get core temp from thermo_data
    thermo_metrics = thermo_data.get("metrics", {})
    core_temp_peak = thermo_metrics.get("max_core_temp", 0)
    
    # Calculate expected power loss after 60 min
    # Threshold: dEF/dC exists AND core_temp > 38
    ef_slope_val = ef_slope if ef_slope else 0
    
    if abs(ef_slope_val) > 0.01 and core_temp_peak > 38.0:
        # Assume conservative: 1% EF loss ≈ 0.7% power loss
        # If we have delta_pct (change over test), extrapolate to 60 min
        # Use absolute delta_pct as base (already computed)
        power_loss_pct = abs(delta_pct) * 0.7
        
        # Clamp to reasonable range (2-15%)
        power_loss_pct = max(2.0, min(15.0, power_loss_pct))
        
        simulation_text = (
            f"<b>SYMULACJA KONSEKWENCJI WYŚCIGOWYCH:</b> Przy obecnym koszcie termicznym (temp. rd. {core_temp_peak:.1f}°C, "
            f"dryf EF {delta_pct:+.1f}%), oczekiwany spadek efektywnej mocy wynosi ~{power_loss_pct:.0f}% "
            "po 60 min w gorących warunkach."
        )
        sim_color = "#E74C3C" if power_loss_pct > 8 else "#F39C12"
    else:
        simulation_text = (
            "<b>SYMULACJA KONSEKWENCJI WYŚCIGOWYCH:</b> Obciążenie termiczne w akceptowalnych granicach. "
            "Brak przewidywanego znaczącego spadku mocy dla 60 min wysokości w bieżących warunkach."
        )
        sim_color = "#27AE60"
    
    sim_style = ParagraphStyle('sim_text', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
    sim_box = Table(
        [[Paragraph(simulation_text, sim_style)]],
        colWidths=[165 * mm]
    )
    sim_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(sim_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(sim_box)
    elements.append(Spacer(1, 6 * mm))
    
    # Chart: EF vs Time/Temp
    if figure_paths and "thermal_efficiency" in figure_paths:
        elements.extend(_build_chart(figure_paths["thermal_efficiency"], "Efektywność vs Czas/Temperatura", styles))
        elements.append(Spacer(1, 4 * mm))
    
    
    # === TRAINING IMPLICATIONS BOX ===
    # UWAGA: Usunięto PageBreak aby zapobiec pustej stronie
    elements.append(Spacer(1, 6 * mm))
    elements.append(Paragraph("IMPLIKACJE TRENINGOWE", styles["heading"]))
    elements.append(Spacer(1, 2 * mm))
    
    # Add extended introduction
    elements.append(Paragraph(
        "<b>Zastosowanie w praktyce:</b> Poniższe zalecenia treningowe wynikają bezpośrednio "
        "z analizy dryfu sercowego i termicznego. Dostosuj intensywność i strategię "
        "nawodnienia do warunków środowiskowych.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))
    
    if has_drift_data:
        implications = drift_interp.get("training_implications", [])[:5]
    else:
        implications = [
            "Brak danych do wygenerowania rekomendacji",
            "Upewnij się, że plik źródłowy zawiera kolumny Power i HR",
        ]
    
    impl_style = ParagraphStyle('impl_white', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=8)
    impl_data = [[Paragraph(f"→ {impl}", impl_style)] for impl in implications]
    impl_table = Table(impl_data, colWidths=[165 * mm])
    impl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#16213e")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(impl_table)
    
    return elements


# ============================================================================
# PAGE 7: PROFIL METABOLICZNY (LIMITERY)
# ============================================================================

# ============================================================================
# NEW PAGE: BIOMECHANIKA (Premium INSCYD/WKO Quality)
# ============================================================================

def build_page_biomech(
    figure_paths: Dict[str, str],
    styles: Dict,
    biomech_data: Optional[Dict[str, Any]] = None
) -> List:
    """Build Biomechanics & Occlusion Physiology page - INSCYD/WKO Quality."""

    from reportlab.lib.styles import ParagraphStyle
    
    elements = []
    
    elements.append(Paragraph("<font size='14'>3.4 BIOMECHANIKA</font>", styles["center"]))
    elements.append(Paragraph("<font size='10' color='#7F8C8D'>Fizjologia okluzji i relacja siła–tlen</font>", styles["body"]))
    elements.append(Spacer(1, 6 * mm))
    
    # Chart 1: Torque vs Cadence
    if figure_paths and "biomech_summary" in figure_paths:
        elements.extend(_build_chart(figure_paths["biomech_summary"], "Moment Obrotowy vs Kadencja", styles))
        elements.append(Spacer(1, 4 * mm))
        
    # Chart 2: Torque vs SmO2 (Occlusion)
    if figure_paths and "biomech_torque_smo2" in figure_paths:
        elements.extend(_build_chart(figure_paths["biomech_torque_smo2"], "Fizjologia Okluzji (Siła vs Tlen)", styles))
        elements.append(Spacer(1, 6 * mm))
    
    # === KEY NUMBERS BOX ===
    if biomech_data and "metrics" in biomech_data:
        metrics = biomech_data["metrics"]
        classification = biomech_data.get("classification", {})
        
        elements.append(Spacer(1, 6 * mm))
        elements.append(Paragraph("KLUCZOWE LICZBY", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))
        
        # Determine colors based on classification
        level = classification.get("level", "unknown")
        level_color = classification.get("color", "#808080")
        level_label = {"low": "NISKA", "moderate": "UMIARKOWANA", "high": "WYSOKA"}.get(level, "---")
        
        # Build metrics table
        occlusion_idx = metrics.get("occlusion_index", 0)
        slope = metrics.get("regression_slope", 0)
        r2 = metrics.get("regression_r2", 0)
        smo2_base = metrics.get("smo2_baseline", 0)
        torque_base = metrics.get("torque_at_baseline", 0)
        torque_10 = metrics.get("torque_at_minus_10")
        torque_20 = metrics.get("torque_at_minus_20")
        
        key_data = [
            ["Metryka", "Wartość", "Interpretacja"],
            ["INDEKS OKLUZJI", f"{occlusion_idx:.3f}", f"Okluzja: {level_label}"],
            ["Nachylenie SmO2/Torque", f"{slope:.4f} %/Nm", f"R² = {r2:.2f}"],
            ["SmO2 bazowe", f"{smo2_base:.1f} %", f"@ {torque_base:.0f} Nm"],
            ["Torque @ SmO2 -10%", f"{torque_10:.0f} Nm" if torque_10 else "---", "Próg umiarkowanej okluzji"],
            ["Torque @ SmO2 -20%", f"{torque_20:.0f} Nm" if torque_20 else "---", "Próg istotnej okluzji"],
        ]
        
        table = Table(key_data, colWidths=[50 * mm, 40 * mm, 75 * mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('BACKGROUND', (0, 1), (-1, 1), HexColor(level_color)),  # OCCLUSION INDEX row
            ('TEXTCOLOR', (0, 1), (-1, 1), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),  # ALL cells
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
            ('ROWHEIGHT', (0, 0), (-1, -1), 12 * mm),
            # Light backgrounds for data rows (except OCCLUSION INDEX row)
            ('BACKGROUND', (0, 2), (-1, 2), HexColor("#f5f5f5")),
            ('TEXTCOLOR', (0, 2), (-1, 2), HexColor("#333333")),
            ('BACKGROUND', (0, 3), (-1, 3), HexColor("#e8e8e8")),
            ('TEXTCOLOR', (0, 3), (-1, 3), HexColor("#333333")),
            ('BACKGROUND', (0, 4), (-1, 4), HexColor("#f5f5f5")),
            ('TEXTCOLOR', (0, 4), (-1, 4), HexColor("#333333")),
            ('BACKGROUND', (0, 5), (-1, 5), HexColor("#e8e8e8")),
            ('TEXTCOLOR', (0, 5), (-1, 5), HexColor("#333333")),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 6 * mm))
        
        # === SAFE TORQUE WINDOW ===
        elements.append(Paragraph("<b>BEZPIECZNE OKNO MOMENTU OBROTOWEGO</b>", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))
        
        # Define zones
        safe_max = torque_10 if torque_10 else torque_base
        warning_min = torque_10 if torque_10 else 0
        warning_max = torque_20 if torque_20 else 0
        critical_min = torque_20 if torque_20 else 0
        
        # Build zone table
        zone_rows = [
            ["Strefa", "Zakres Momentu", "Status"],
            ["BEZPIECZNA", f"< {safe_max:.0f} Nm" if safe_max else "---", "Pełna dostawa tlenowa utrzymana"],
            ["OSTRZEŻENIE", f"{warning_min:.0f}–{warning_max:.0f} Nm" if warning_min and warning_max else "---", "Częściowa okluzja, SmO2 spada"],
            ["KRYTYCZNA", f"> {critical_min:.0f} Nm" if critical_min else "---", "Silna okluzja, szybka desaturacja"],
        ]
        
        zone_table = Table(zone_rows, colWidths=[35 * mm, 50 * mm, 80 * mm])
        zone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2C3E50")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), FONT_FAMILY),
            ('FONTNAME', (0, 0), (-1, 0), FONT_FAMILY_BOLD),
            ('FONTNAME', (0, 1), (0, -1), FONT_FAMILY_BOLD),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
            ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
            # Zone colors
            ('BACKGROUND', (0, 1), (0, 1), HexColor("#27AE60")),  # SAFE
            ('TEXTCOLOR', (0, 1), (0, 1), HexColor("#FFFFFF")),
            ('BACKGROUND', (0, 2), (0, 2), HexColor("#F39C12")),  # WARNING
            ('TEXTCOLOR', (0, 2), (0, 2), HexColor("#FFFFFF")),
            ('BACKGROUND', (0, 3), (0, 3), HexColor("#E74C3C")),  # CRITICAL
            ('TEXTCOLOR', (0, 3), (0, 3), HexColor("#FFFFFF")),
            # Data cells
            ('BACKGROUND', (1, 1), (-1, -1), HexColor("#f8f9fa")),
            ('TEXTCOLOR', (1, 1), (-1, -1), HexColor("#333333")),
        ]))
        elements.append(zone_table)
        elements.append(Spacer(1, 3 * mm))
        
        # Collapse verdict
        collapse_point = torque_20 if torque_20 else (torque_10 if torque_10 else 0)
        if collapse_point:
            collapse_text = f"Efektywna dostawa tlenowa załamuje się powyżej {collapse_point:.0f} Nm."
        else:
            collapse_text = "Progi momentu obrotowego niedostępne dla oceny okluzji."
        elements.append(Paragraph(collapse_text, styles["body"]))
        elements.append(Spacer(1, 6 * mm))
        
        # === VERDICT BOX ===
        elements.append(Paragraph("WERDYKT", styles["heading"]))
        elements.append(Spacer(1, 2 * mm))
        
        # Verdict based on classification
        if level == "high":
            verdict_text = (
                "<b>WYSOKA OKLUZJA MECHANICZNA</b><br/>"
                "Przy wysokich momentach obrotowych naczynia mięśniowe są mechanicznie "
                "kompresowane, ograniczając perfuzję mimo dostępnego VO₂ systemowego. "
                "Styl siłowy (niska kadencja) prowadzi do przedwczesnej hipoksji lokalnej."
            )
            verdict_color = "#E74C3C"
        elif level == "moderate":
            verdict_text = (
                "<b>UMIARKOWANA OKLUZJA</b><br/>"
                "Spadek SmO₂ jest proporcjonalny do wzrostu momentu obrotowego. "
                "Mięśnie wykazują pewną tolerancję na siły, ale istnieje wyraźna granica "
                "powyżej której desaturacja przyspiesza."
            )
            verdict_color = "#F39C12"
        else:
            verdict_text = (
                "<b>NISKA OKLUZJA</b><br/>"
                "Kapilaryzacja mięśniowa jest wystarczająca, aby utrzymać perfuzję "
                "nawet przy wysokich momentach obrotowych. Możliwość efektywnej pracy siłowej."
            )
            verdict_color = "#27AE60"
        
        verdict_box = Table(
            [[Paragraph(verdict_text, ParagraphStyle('verdict', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9))]],
            colWidths=[165 * mm]
        )
        verdict_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(verdict_color)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BOX', (0, 0), (-1, -1), 1, HexColor(verdict_color)),
        ]))
        elements.append(verdict_box)
        elements.append(Spacer(1, 6 * mm))
        
        # === MECHANISM DESCRIPTION ===
        interpretation = biomech_data.get("interpretation", {})
        mechanism = interpretation.get("mechanism", "")
        riding_style = interpretation.get("riding_style", "")
        
        if mechanism:
            elements.append(Paragraph("<b>MECHANIZM FIZJOLOGICZNY</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(mechanism, styles["body"]))
            elements.append(Spacer(1, 4 * mm))
        
        if riding_style:
            elements.append(Paragraph("<b>WPŁYW NA STYL JAZDY</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))
            elements.append(Paragraph(riding_style, styles["body"]))
            elements.append(Spacer(1, 2 * mm))
            # Cadence context for occlusion interpretation (Hammer et al. 2021)
            avg_cadence = metrics.get("avg_cadence", 0)
            if avg_cadence and avg_cadence > 0:
                if avg_cadence < 75:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Niska kadencja zwiększa "
                        "moment obrotowy na obrót i wydłuża fazę skurczu izometrycznego, "
                        "nasilając mechaniczną okluzję naczyń (Hammer et al. 2021). "
                        "Rozważ podniesienie kadencji do 85-95 RPM w strefach Z3-Z4, "
                        "aby zmniejszyć szczytowe ciśnienie śródmięśniowe."
                    )
                elif avg_cadence > 100:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Wysoka kadencja "
                        "minimalizuje okluzję mechaniczną, ale zwiększa koszt metaboliczny "
                        "koordynacji nerwowo-mięśniowej. Optymalny zakres: 85-95 RPM."
                    )
                else:
                    cad_note = (
                        f"<b>Kontekst kadencji ({avg_cadence:.0f} RPM):</b> Kadencja w optymalnym "
                        "zakresie, równoważącym okluzję mechaniczną z kosztem metabolicznym."
                    )
                elements.append(Paragraph(cad_note, styles["body"]))
            elements.append(Spacer(1, 4 * mm))
        
        # === TRAINING RECOMMENDATIONS ===
        recommendations = interpretation.get("recommendations", [])
        if recommendations:
            elements.append(Paragraph("<b>ZALECENIA TRENINGOWE</b>", styles["heading"]))
            elements.append(Spacer(1, 2 * mm))
            
            # Use white text style for visibility on dark background
            white_style = ParagraphStyle('white_body', parent=styles["body"], textColor=HexColor("#FFFFFF"), fontSize=9)
            rec_data = [[Paragraph(f"• {rec}", white_style)] for rec in recommendations]
            rec_table = Table(rec_data, colWidths=[165 * mm])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), HexColor("#16213e")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(rec_table)
    else:
        # Fallback for legacy data without biomech analysis
        elements.append(Paragraph(
            "<b>Interpretacja:</b> Spadek saturacji (SmO₂) przy wysokich momentach obrotowych "
            "może świadczyć o okluzji mechanicznej lub niskiej efektywności układu krążenia "
            "w warunkach wysokiego napięcia mięśniowego.",
            styles["body"]
        ))
    
    return elements


# ============================================================================
# NEW PAGE: DRYF FIZJOLOGICZNY
# ============================================================================

def build_page_drift(
    kpi: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Build Physiological Drift page (heatmaps only, no KPI table)."""
    elements = []
    
    elements.append(Paragraph("<font size='14'>4.2 DRYF FIZJOLOGICZNY</font>", styles["center"]))
    elements.append(Spacer(1, 6 * mm))
    
    # Heatmaps
    if figure_paths and "drift_heatmap_hr" in figure_paths:
        elements.extend(_build_chart(figure_paths["drift_heatmap_hr"], "Mapa Dryfu (HR vs Power)", styles))
        elements.append(Spacer(1, 4 * mm))
        
    if figure_paths and "drift_heatmap_smo2" in figure_paths:
        elements.extend(_build_chart(figure_paths["drift_heatmap_smo2"], "Mapa Oksydacji (SmO2 vs Power)", styles))
        elements.append(Spacer(1, 4 * mm))
    
    # Drift heatmap interpretation guide
    elements.append(Spacer(1, 6 * mm))
    elements.extend(_build_education_block(
        "Jak czytać mapy dryfu?",
        "Mapy cieplne pokazują zmianę HR i SmO₂ w funkcji mocy i czasu. "
        "Ciepłe kolory (czerwony/żółty) oznaczają narastający koszt utrzymania danej mocy — "
        "im szybciej pojawiają się przy niższych watach, tym gorsza tolerancja obciążenia.",
        styles
    ))

    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        "<b>Mapa HR vs Power:</b> Pionowe przejście kolorów (zielony→żółty→czerwony) "
        "przy stałej mocy oznacza dryf sercowy — serce pracuje coraz ciężej bez wzrostu "
        "mocy mechanicznej. Próg >5% decoupling (Pa:Hr) sygnalizuje konieczność korekty pacingu.",
        styles["body"]
    ))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        "<b>Mapa SmO₂ vs Power:</b> Spadek SmO₂ przy rosnącej mocy jest fizjologiczny. "
        "Kluczowe jest tempo spadku: >15%/100W sugeruje ograniczoną zdolność ekstrakcji O₂ "
        "lub mechaniczną okluzję naczyń (Vasquez Bonilla et al. 2023). Stabilizacja SmO₂ "
        "przy wysokich watach może wskazywać na osiągnięcie plateau ekstrakcji.",
        styles["body"]
    ))
    
    return elements


# ============================================================================
# NEW PAGE: KLUCZOWE WSKAŹNIKI WYDAJNOŚCI (KPI) - PREMIUM DASHBOARD
# ============================================================================

def build_page_kpi_dashboard(
    kpi: Dict[str, Any],
    styles: Dict
) -> List:
    """Build dedicated KPI Dashboard page - premium quality."""

    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import TableStyle
    
    elements = []
    
    elements.append(Paragraph("5. PODSUMOWANIE", styles["title"]))
    elements.append(Paragraph("<font size='14'>5.1 WSKAŹNIKI KPI</font>", styles["center"]))
    elements.append(Paragraph("<font size='10' color='#7F8C8D'>Dashboard stabilności układu krążenia i kosztu energetycznego</font>", styles["body"]))
    elements.append(Spacer(1, 8 * mm))
    
    # === HELPER FUNCTIONS ===
    def get_ef_status(val):
        """Efficiency Factor status."""
        if val is None: return ("n/a", "BRAK", "#808080")
        try:
            v = float(val)
            if v < 1.8: return (f"{v:.2f}", "SŁABO", "#E74C3C")
            elif v < 2.2: return (f"{v:.2f}", "OK", "#27AE60")
            else: return (f"{v:.2f}", "BARDZO DOBRZE", "#2ECC71")
        except (ValueError, TypeError): return ("n/a", "BRAK", "#808080")

    def get_pahr_status(val):
        """Pa:Hr Decoupling status."""
        if val is None: return ("n/a", "BRAK", "#808080")
        try:
            v = float(val)
            if v < 5: return (f"{v:.1f}%", "STABILNY", "#27AE60")
            elif v < 8: return (f"{v:.1f}%", "OSTRZEŻENIE", "#F39C12")
            else: return (f"{v:.1f}%", "RYZYKO", "#E74C3C")
        except (ValueError, TypeError): return ("n/a", "BRAK", "#808080")

    def get_smo2_drift_status(val):
        """SmO2 Drift status — uses absolute value for classification."""
        if val is None: return ("n/a", "BRAK", "#808080")
        try:
            v = float(val)
            v_abs = abs(v)
            if v_abs < 5: return (f"{v:.1f}%", "STABILNY", "#27AE60")
            elif v_abs < 15: return (f"{v:.1f}%", "UMIARKOWANY DRYF", "#F39C12")
            else: return (f"{v:.1f}%", "WYSOKI DRYF OBWODOWY", "#E74C3C")
        except (ValueError, TypeError): return ("n/a", "BRAK", "#808080")
    
    # === GET VALUES ===
    ef_val, ef_status, ef_color = get_ef_status(kpi.get("ef"))
    pahr_val, pahr_status, pahr_color = get_pahr_status(kpi.get("pa_hr"))
    smo2_val, smo2_status, smo2_color = get_smo2_drift_status(kpi.get("smo2_drift"))
    
    # VO2max from canonical source
    vo2max_raw = kpi.get("vo2max_est")
    vo2max_source = kpi.get("vo2max_source", "")
    if vo2max_raw and vo2max_raw != "brak danych":
        try:
            vo2max_val = f"{float(vo2max_raw):.1f} ml/kg"
            vo2max_status = "CANONICAL"
            vo2max_color = "#3498DB"
        except (ValueError, TypeError):
            vo2max_val = "n/a"
            vo2max_status = "BRAK"
            vo2max_color = "#808080"
    else:
        vo2max_val = "n/a"
        vo2max_status = "BRAK"
        vo2max_color = "#808080"
    
    # === BUILD KPI TABLE - Polish chars with proper encoding, Paragraph wrapping ===
    kpi_cell = ParagraphStyle('kpi_cell', parent=styles["body"], fontSize=9, leading=11,
                              fontName=FONT_FAMILY, alignment=0)
    kpi_cell_c = ParagraphStyle('kpi_cell_c', parent=kpi_cell, alignment=1)
    kpi_header = ParagraphStyle('kpi_header', parent=kpi_cell, fontName=FONT_FAMILY_BOLD,
                                textColor=HexColor("#FFFFFF"), alignment=1)
    kpi_status = ParagraphStyle('kpi_status', parent=kpi_cell, fontName=FONT_FAMILY_BOLD,
                                textColor=HexColor("#FFFFFF"), alignment=1, fontSize=8, leading=10)

    rows = [
        [Paragraph("Metryka", kpi_header), Paragraph("Wartość", kpi_header),
         Paragraph("Zakres Ref.", kpi_header), Paragraph("Status", kpi_header)],
        [Paragraph("Efficiency Factor (EF)", kpi_cell), Paragraph(ef_val, kpi_cell_c),
         Paragraph("<1.8 słabo | 1.8-2.2 ok | >2.2 b.dobrze", kpi_cell),
         Paragraph(ef_status, kpi_status)],
        [Paragraph("Pa:Hr Decoupling", kpi_cell), Paragraph(pahr_val, kpi_cell_c),
         Paragraph("<5% stab. | 5-8% ostrz. | >8% ryzyko", kpi_cell),
         Paragraph(pahr_status, kpi_status)],
        [Paragraph("SmO₂ Drift", kpi_cell), Paragraph(smo2_val, kpi_cell_c),
         Paragraph("<5% stab. | 5-15% umiark. | >15% wysoki", kpi_cell),
         Paragraph(smo2_status, kpi_status)],
        [Paragraph("VO₂max", kpi_cell), Paragraph(vo2max_val, kpi_cell_c),
         Paragraph(f"Źródło: {vo2max_source}", kpi_cell),
         Paragraph(vo2max_status, kpi_status)],
    ]

    table = Table(rows, colWidths=[38 * mm, 25 * mm, 60 * mm, 42 * mm])
    
    # Dynamic styling based on status colors
    table_style = [
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # ALL CELLS must use DejaVuSans for Polish characters
        ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 14 * mm),
        # Data row backgrounds - lighter for readability
        ('BACKGROUND', (0, 1), (2, 1), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 1), (2, 1), HexColor("#333333")),
        ('BACKGROUND', (0, 2), (2, 2), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 2), (2, 2), HexColor("#333333")),
        ('BACKGROUND', (0, 3), (2, 3), HexColor("#f5f5f5")),
        ('TEXTCOLOR', (0, 3), (2, 3), HexColor("#333333")),
        ('BACKGROUND', (0, 4), (2, 4), HexColor("#e8e8e8")),
        ('TEXTCOLOR', (0, 4), (2, 4), HexColor("#333333")),
        # Status column colors per row - bright colors with white text
        ('BACKGROUND', (3, 1), (3, 1), HexColor(ef_color)),
        ('TEXTCOLOR', (3, 1), (3, 1), HexColor("#FFFFFF")),
        ('BACKGROUND', (3, 2), (3, 2), HexColor(pahr_color)),
        ('TEXTCOLOR', (3, 2), (3, 2), HexColor("#FFFFFF")),
        ('BACKGROUND', (3, 3), (3, 3), HexColor(smo2_color)),
        ('TEXTCOLOR', (3, 3), (3, 3), HexColor("#FFFFFF")),
        ('BACKGROUND', (3, 4), (3, 4), HexColor(vo2max_color)),
        ('TEXTCOLOR', (3, 4), (3, 4), HexColor("#FFFFFF")),
        ('FONTNAME', (3, 1), (3, -1), 'DejaVuSans-Bold'),
    ]
    
    table.setStyle(TableStyle(table_style))
    elements.append(table)
    elements.append(Spacer(1, 8 * mm))
    
    # === LEGEND ===
    legend_style = ParagraphStyle('legend', parent=styles["body"], fontSize=8, textColor=HexColor("#95A5A6"))
    elements.append(Paragraph(
        "<b>Legenda statusów:</b> "
        "<font color='#27AE60'>■ OK/STABILNY</font> | "
        "<font color='#F39C12'>■ OSTRZEŻENIE</font> | "
        "<font color='#E74C3C'>■ RYZYKO/SŁABO</font> | "
        "<font color='#808080'>■ BRAK DANYCH</font>",
        legend_style
    ))
    elements.append(Spacer(1, 6 * mm))
    
    # === INTERPRETIVE FOOTER ===
    footer_text = (
        "<b>Interpretacja:</b> KPI odzwierciedlają stabilność układu krążenia i koszt energetyczny "
        "utrzymania mocy. Efficiency Factor (EF) pokazuje ile watów generujesz na każde uderzenie "
        "serca - wyższy EF oznacza lepszą sprawność aerobową. Pa:Hr Decoupling >5% świadczy o "
        "niepełnej adaptacji termicznej lub chronicznym zmęczeniu. SmO2 Drift wskazuje na lokalne "
        "wyczerpanie mięśni niezależne od układu sercowo-naczyniowego."
    )
    elements.append(Paragraph(footer_text, styles["body"]))
    elements.append(Spacer(1, 10 * mm))
    
    # ==========================================================================
    # KPI – EXPECTED TREND (6–8 weeks)
    # ==========================================================================
    
    elements.append(Paragraph("<b>OCZEKIWANY TREND (6–8 tygodni)</b>", styles["subheading"]))
    elements.append(Spacer(1, 3 * mm))
    
    # Calculate target values heuristically
    def calc_target(current_str, improvement_pct, is_reduction=False):
        try:
            # Extract numeric value
            val = float(current_str.replace('%', '').replace(' ml/kg', '').replace('n/a', '0'))
            if val == 0:
                return "—"
            if is_reduction:
                target = val * (1 - improvement_pct / 100)
            else:
                target = val * (1 + improvement_pct / 100)
            return f"{target:.2f}" if target < 10 else f"{target:.1f}"
        except (ValueError, TypeError, ZeroDivisionError):
            return "—"
    
    # Get current values (parsed earlier)
    ef_current = ef_val
    pahr_current = pahr_val
    smo2_current = smo2_val
    vo2_current = vo2max_val
    
    # Calculate targets
    ef_target = calc_target(ef_val, 6.5)  # +5-8% → 6.5% avg
    pahr_target = calc_target(pahr_val.replace('%', ''), 7.5, is_reduction=True)  # improve by 5-10 pp → 7.5% reduction
    smo2_target = calc_target(smo2_val.replace('%', ''), 35, is_reduction=True)  # reduce by 30-40% → 35%
    vo2_target = calc_target(vo2max_val.replace(' ml/kg', ''), 4)  # +3-5% → 4% avg
    
    # Build trend table
    trend_header = ["Metryka", "Obecny", "Cel (6–8 tyg.)", "Droga do celu"]
    trend_rows = [
        trend_header,
        ["Wsp. Efektywności (EF)", ef_current, ef_target, "+5–8% przez objętość Z2"],
        ["Pa:Hr (Rozjechanie)", pahr_current, f"{pahr_target}%", "Adaptacja cieplna + pacing"],
        ["% Dryf SmO2", smo2_current, f"{smo2_target}%", "−30–40% przez siłę + kadencję"],
        ["VO2max", vo2_current, f"{vo2_target} ml/kg", "+3–5% przez interwały VO2max"],
    ]
    
    trend_table = Table(trend_rows, colWidths=[45 * mm, 30 * mm, 40 * mm, 55 * mm])
    trend_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#2C3E50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, 0), FONT_FAMILY_BOLD),
        ('FONTNAME', (0, 1), (-1, -1), FONT_FAMILY),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#f8f9fa")),
        ('BACKGROUND', (2, 1), (2, -1), HexColor("#E8F6EF")),  # Target column highlight
    ]))
    elements.append(trend_table)
    elements.append(Spacer(1, 3 * mm))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('disclaimer', parent=styles["body"], fontSize=8, textColor=HexColor("#7F8C8D"), fontName=FONT_FAMILY_ITALIC)
    elements.append(Paragraph("Cele zakładają ustrukturyzowany trening i adaptację cieplną.", disclaimer_style))
    
    return elements


# Legacy compatibility wrapper
def build_page_drift_kpi(
    kpi: Dict[str, Any],
    figure_paths: Dict[str, str],
    styles: Dict
) -> List:
    """Legacy wrapper - combines drift page content only, KPI is on separate page now."""
    return build_page_drift(kpi, figure_paths, styles)


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

# ============================================================================
# PAGE: HRV / DFA ALPHA-1 ANALYSIS
# ============================================================================


def build_page_hrv(hrv_data: Dict[str, Any], styles: Dict) -> List:
    """Build HRV / DFA Alpha-1 analysis page.

    Displays DFA Alpha-1 zone classification, RMSSD/SDNN metrics,
    and autonomic fitness interpretation with literature references.
    """

    elements = []

    elements.append(Paragraph(
        "<font size='14'>3.5 ZMIENNOŚĆ RYTMU SERCA (HRV / DFA Alpha-1)</font>",
        styles["center"]
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>Analiza fraktalna RR — klasyfikacja stref, fitness autonomiczny</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))

    # Intro
    elements.append(Paragraph(
        "DFA Alpha-1 to wskaźnik fraktalnej korelacji odstępów R-R serca. "
        "Wartość Alpha-1 ≈ 0.75 odpowiada progowi aerobowemu (HRVT1), "
        "a Alpha-1 ≈ 0.50 progowi anaerobowemu (HRVT2). "
        "Metoda jest niezależna od progów wentylacyjnych i SmO₂ — pozwala na krzyżową walidację.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))

    # Extract data
    summary = hrv_data.get("summary", {})
    quality = hrv_data.get("quality", {})
    zones = hrv_data.get("zone_classification", {})

    mean_alpha1 = summary.get("mean_alpha1")
    mean_rmssd = summary.get("mean_rmssd")
    mean_sdnn = summary.get("mean_sdnn")
    windows_count = summary.get("windows_analyzed", 0)
    quality_grade = quality.get("grade", "n/a")
    is_uncertain = quality.get("is_uncertain", True)
    quality_reasons = quality.get("reasons", [])

    # === METRIC CARDS ===
    elements.append(Paragraph("<b>KLUCZOWE METRYKI HRV</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    def _hrv_card(title, value, unit, color, subtitle=""):
        card_content = [
            Paragraph(f"<font size='8' color='#7F8C8D'>{title}</font>", styles["center"]),
            Paragraph(f"<font size='14' color='{color}'><b>{value}</b></font>", styles["center"]),
            Paragraph(f"<font size='9'>{unit}</font>", styles["center"]),
        ]
        if subtitle:
            card_content.append(Paragraph(
                f"<font size='7' color='#95A5A6'>{subtitle}</font>", styles["center"]
            ))
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

    # Alpha-1 classification
    if mean_alpha1 is not None:
        if mean_alpha1 > 0.75:
            a1_color = "#27AE60"
            a1_zone = "Aerobowa (Z1-Z2)"
        elif mean_alpha1 > 0.5:
            a1_color = "#F39C12"
            a1_zone = "Przejściowa (Z3)"
        else:
            a1_color = "#E74C3C"
            a1_zone = "Anaerobowa (Z4-Z5)"
        a1_val = f"{mean_alpha1:.2f}"
    else:
        a1_color = "#808080"
        a1_zone = "brak danych"
        a1_val = "n/a"

    # RMSSD classification
    if mean_rmssd is not None:
        if mean_rmssd > 50:
            rmssd_color = "#27AE60"
            rmssd_interp = "Wysoka"
        elif mean_rmssd > 20:
            rmssd_color = "#F39C12"
            rmssd_interp = "Umiarkowana"
        else:
            rmssd_color = "#E74C3C"
            rmssd_interp = "Niska"
        rmssd_val = f"{mean_rmssd:.0f}"
    else:
        rmssd_color = "#808080"
        rmssd_interp = "brak"
        rmssd_val = "n/a"

    # SDNN
    sdnn_val = f"{mean_sdnn:.0f}" if mean_sdnn is not None else "n/a"
    sdnn_color = "#3498DB"

    # Quality grade color
    grade_colors = {"A": "#27AE60", "B": "#2ECC71", "C": "#F39C12", "D": "#E74C3C", "F": "#C0392B"}
    grade_color = grade_colors.get(quality_grade, "#808080")

    card1 = _hrv_card("DFA Alpha-1", a1_val, "fraktalny", a1_color, a1_zone)
    card2 = _hrv_card("RMSSD", rmssd_val, "ms", rmssd_color, rmssd_interp)
    card3 = _hrv_card("SDNN", sdnn_val, "ms", sdnn_color, "zmienność ogólna")
    card4 = _hrv_card("JAKOŚĆ", quality_grade, f"({windows_count} okien)", grade_color,
                       "niepewny" if is_uncertain else "wiarygodny")

    cards_row = Table([[card1, card2, card3, card4]], colWidths=[44 * mm] * 4)
    cards_row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(cards_row)
    elements.append(Spacer(1, 6 * mm))

    # === ZONE CLASSIFICATION TABLE ===
    elements.append(Paragraph("<b>KLASYFIKACJA STREF DFA</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    zone_table_data = [
        ["Strefa", "Alpha-1", "Opis fizjologiczny", "Zastosowanie treningowe"],
        ["Z1-Z2 Aerobowa", "> 0.75", "Korelacje fraktalne zachowane, dominacja parasympatyczna",
         "Treningi bazowe, recovery, długie Z2"],
        ["Z3 Przejściowa", "0.50 - 0.75", "Utrata korelacji, równowaga sympatyczna/parasympatyczna",
         "Tempo, Sweet Spot — strefa 'szarego pola'"],
        ["Z4-Z5 Anaerobowa", "< 0.50", "Korelacje antypersystentne, dominacja sympatyczna",
         "VO₂max interwały, powtórzenia ponadprogowe"],
    ]

    zt = Table(zone_table_data, colWidths=[30 * mm, 22 * mm, 55 * mm, 55 * mm])
    zt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 12 * mm),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor("#d5f5e3")),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor("#fdebd0")),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor("#fadbd8")),
    ]))
    elements.append(zt)
    elements.append(Spacer(1, 6 * mm))

    # === QUALITY WARNINGS ===
    if quality_reasons:
        elements.append(Paragraph("<b>UWAGI DOTYCZĄCE JAKOŚCI</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))
        for reason in quality_reasons[:4]:
            elements.append(Paragraph(
                f"<font size='8' color='#E74C3C'>• {reason}</font>", styles["body"]
            ))
        elements.append(Spacer(1, 4 * mm))

    # === INTERPRETATION ===
    if mean_alpha1 is not None:
        if mean_alpha1 > 0.75:
            interp_text = (
                "<b>DOMINACJA AEROBOWA (Alpha-1 > 0.75)</b><br/>"
                "Przeciętna wartość Alpha-1 wskazuje na dominację metabolizmu tlenowego. "
                "Układ autonomiczny jest dobrze zbalansowany. Trening bazowy (Z1-Z2) jest optymalny — "
                "moment na budowanie objętości i gęstości mitochondriów."
            )
            interp_color = "#27AE60"
        elif mean_alpha1 > 0.5:
            interp_text = (
                "<b>STREFA PRZEJŚCIOWA (Alpha-1 0.50-0.75)</b><br/>"
                "Przeciętna wartość Alpha-1 odpowiada strefie progu aerobowego (HRVT1). "
                "Jest to 'szare pole' — system glikolizowy jest aktywny, ale nie dominuje. "
                "Odpowiada to pracy Tempo/Sweet Spot."
            )
            interp_color = "#F39C12"
        else:
            interp_text = (
                "<b>DOMINACJA ANAEROBOWA (Alpha-1 < 0.50)</b><br/>"
                "Przeciętna wartość Alpha-1 wskazuje na intensywność powyżej HRVT2. "
                "Układ sympatyczny dominuje. Ten zakres odpowiada pracy VO₂max / Z4-Z5."
            )
            interp_color = "#E74C3C"

        white_style = ParagraphStyle(
            'hrv_interp', parent=styles["body"],
            textColor=HexColor("#FFFFFF"), fontSize=9
        )
        interp_box = Table(
            [[Paragraph(interp_text, white_style)]],
            colWidths=[165 * mm]
        )
        interp_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(interp_color)),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(interp_box)
        elements.append(Spacer(1, 6 * mm))

    # === REFERENCES ===
    elements.append(Paragraph(
        "<font size='7' color='#95A5A6'><i>"
        "Ref: Mateo-March et al. 2024 — HRVT1 ICC=0.87, HRVT2 ICC=0.97. "
        "Iannetta et al. 2024 — DFA-a1 reliability ICC=0.76-0.86. "
        "Rogers et al. 2023 — validated in female cyclists. "
        "Cassirame et al. 2025 — SNR and motion artifact concerns."
        "</i></font>",
        styles["body"]
    ))

    return elements


# ============================================================================
# PAGE: SKIN TEMPERATURE GRADIENT (CORE − SKIN)
# ============================================================================


def build_page_skin_temp(thermo_data: Dict[str, Any], time_series: Dict[str, Any],
                         styles: Dict) -> List:
    """Build Skin Temperature Gradient analysis section.

    Displays core-skin temperature gradient as indicator of
    thermoregulatory efficiency (Périard et al. 2021).
    """

    elements = []

    elements.append(Paragraph(
        "<font size='14'>4.4 GRADIENT TEMPERATURY (RDZEŃ − SKÓRA)</font>",
        styles["center"]
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>"
        "Efektywność chłodzenia — Périard et al. 2021"
        "</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph(
        "Gradient core-skin (ΔT = T<sub>core</sub> − T<sub>skin</sub>) jest wskaźnikiem "
        "efektywności termoregulacji. Wysoki gradient (>3°C) oznacza skuteczne chłodzenie — "
        "ciepło jest efektywnie odprowadzane z rdzenia do skóry. Spadek gradientu poniżej 2°C "
        "sygnalizuje, że skóra nie nadąża z chłodzeniem lub warunki otoczenia ograniczają konwekcję.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))

    # Extract time series data
    core_temps = time_series.get("core_temp", [])
    skin_temps = time_series.get("skin_temp", [])

    if not core_temps or not skin_temps or len(core_temps) < 10:
        elements.append(Paragraph(
            "<font color='#E74C3C'><b>Brak wystarczających danych temperatury skóry i/lub rdzenia.</b></font>",
            styles["body"]
        ))
        return elements

    import numpy as np
    core_arr = np.array(core_temps, dtype=float)
    skin_arr = np.array(skin_temps, dtype=float)

    # Filter valid values
    valid_mask = (core_arr > 35) & (core_arr < 42) & (skin_arr > 25) & (skin_arr < 40)
    if np.sum(valid_mask) < 10:
        elements.append(Paragraph(
            "<font color='#E74C3C'><b>Zbyt mało prawidłowych pomiarów temperatury.</b></font>",
            styles["body"]
        ))
        return elements

    core_valid = core_arr[valid_mask]
    skin_valid = skin_arr[valid_mask]
    gradient = core_valid - skin_valid

    # Metrics
    grad_start = float(np.mean(gradient[:30])) if len(gradient) > 30 else float(gradient[0])
    grad_end = float(np.mean(gradient[-30:])) if len(gradient) > 30 else float(gradient[-1])
    grad_min = float(np.min(gradient))
    grad_max = float(np.max(gradient))
    grad_mean = float(np.mean(gradient))
    grad_delta = grad_end - grad_start

    # Classification
    if grad_mean > 3.0:
        class_label = "EFEKTYWNE CHŁODZENIE"
        class_color = "#27AE60"
        class_desc = (
            "Gradient >3°C — termoregulacja skuteczna. Ciepło jest efektywnie "
            "odprowadzane z rdzenia do skóry i dalej do otoczenia."
        )
    elif grad_mean > 2.0:
        class_label = "UMIARKOWANE CHŁODZENIE"
        class_color = "#F39C12"
        class_desc = (
            "Gradient 2-3°C — chłodzenie obecne, ale nie optymalne. "
            "W warunkach ciepłych (>28°C) może dojść do kumulacji ciepła."
        )
    else:
        class_label = "OGRANICZONE CHŁODZENIE"
        class_color = "#E74C3C"
        class_desc = (
            "Gradient <2°C — chłodzenie niewystarczające. Skóra jest blisko temperatury "
            "rdzenia, co ogranicza odprowadzanie ciepła. Ryzyko przegrzania."
        )

    # === KEY METRICS TABLE ===
    elements.append(Paragraph("<b>KLUCZOWE METRYKI</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    metrics_data = [
        ["Metryka", "Wartość", "Interpretacja"],
        ["Gradient start", f"{grad_start:.1f} °C", "Początkowy gradient (pierwsze 30s)"],
        ["Gradient koniec", f"{grad_end:.1f} °C", "Końcowy gradient (ostatnie 30s)"],
        ["Gradient średni", f"{grad_mean:.1f} °C", class_label],
        ["Gradient min", f"{grad_min:.1f} °C", "Najwęższy punkt chłodzenia"],
        ["Zmiana gradientu", f"{grad_delta:+.1f} °C", "Spadek = pogorszenie chłodzenia"],
    ]

    mt = Table(metrics_data, colWidths=[40 * mm, 30 * mm, 95 * mm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#555555")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor(class_color)),
        ('TEXTCOLOR', (0, 3), (-1, 3), HexColor("#FFFFFF")),
    ]))
    elements.append(mt)
    elements.append(Spacer(1, 6 * mm))

    # === CLASSIFICATION VERDICT ===
    white_style = ParagraphStyle(
        'skin_verdict', parent=styles["body"],
        textColor=HexColor("#FFFFFF"), fontSize=9
    )
    verdict_box = Table(
        [[Paragraph(f"<b>{class_label}</b><br/>{class_desc}", white_style)]],
        colWidths=[165 * mm]
    )
    verdict_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor(class_color)),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(verdict_box)
    elements.append(Spacer(1, 6 * mm))

    # === TRAINING IMPLICATIONS ===
    elements.append(Paragraph("<b>IMPLIKACJE TRENINGOWE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    if grad_mean > 3.0:
        recs = [
            "Termoregulacja nie jest limiterem — kontynuuj intensywny trening w normalnych warunkach",
            "Przy starcie w gorącu (>30°C): pre-cooling kamizelka + zimny napój jako dodatkowe zabezpieczenie",
            "Monitoruj gradient w treningach >2h — nawet dobra termoregulacja może się wyczerpać",
        ]
    elif grad_mean > 2.0:
        recs = [
            "Heat acclimation: 7-10 dni, 60min @ Z2 w ubraniu izolacyjnym lub temp. >28°C",
            "Chłodzenie aktywne: woda na kark/głowę co 10-15min w treningach >90min",
            "Preferuj poranne sesje (6-9) latem — niższa temperatura otoczenia = lepszy gradient",
            "Nawodnienie: 500-750ml/h + 500mg Na+/h — odwodnienie pogarsza gradient",
        ]
    else:
        recs = [
            "PILNE: Heat acclimation 10-14 dni, 60-90min @ Z2 w kontrolowanym cieple",
            "Pre-cooling obowiązkowe: kamizelka lodowa 20min + 500ml zimnego napoju przed startem",
            "Unikaj intensywnych sesji (>Z3) w temp. >25°C do czasu poprawy gradientu",
            "Diagnostyka: sprawdź nawodnienie (waga przed/po), sprawdź odzież treningową (wentylacja)",
            "Sauna post-trening: 15-20min × 3-4×/tydz. — wspomaganie adaptacji cieplnej",
        ]

    white_rec_style = ParagraphStyle(
        'rec_skin', parent=styles["body"],
        textColor=HexColor("#FFFFFF"), fontSize=9
    )
    rec_data = [[Paragraph(f"• {rec}", white_rec_style)] for rec in recs]
    rec_table = Table(rec_data, colWidths=[165 * mm])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#16213e")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 6 * mm))

    # === REFERENCE ===
    elements.append(Paragraph(
        "<font size='7' color='#95A5A6'><i>"
        "Ref: Périard et al. 2021 — Core-skin gradient as thermoregulatory efficiency marker. "
        "Racinais et al. 2019 — Heat acclimation improves gradient by 0.5-1.0°C. "
        "Tyler et al. 2015 — Pre-cooling effects on core-skin differential."
        "</i></font>",
        styles["body"]
    ))

    return elements


# ============================================================================
# PAGE: ALTITUDE / ENVIRONMENTAL ADJUSTMENT
# ============================================================================


def build_page_altitude(canonical_data: Dict[str, Any], styles: Dict) -> List:
    """Build altitude/environmental adjustment page.

    Shows VO2max adjustment for altitude using Wehrlin & Hallen 2006 model
    and environmental context for performance interpretation.
    """

    elements = []

    elements.append(Paragraph(
        "<font size='14'>2.6 KOREKTA ŚRODOWISKOWA (WYSOKOŚĆ)</font>",
        styles["center"]
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>"
        "Wehrlin &amp; Hallen 2006 — wpływ wysokości na VO₂max"
        "</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph(
        "VO₂max spada liniowo o ~6.3% na każde 1000m powyżej poziomu morza (Wehrlin &amp; Hallen 2006). "
        "Efekt jest widoczny już od ~500m n.p.m. i wynika z obniżonego ciśnienia parcjalnego O₂. "
        "Przeliczenie na sea-level equivalent pozwala porównać wyniki testów wykonanych na różnych wysokościach.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))

    # Get altitude data
    altitude_adj = canonical_data.get("altitude_adjustment", {})
    vo2max_data = canonical_data.get("vo2max", {})

    altitude_m = altitude_adj.get("altitude_m", 0)
    is_adjusted = altitude_adj.get("is_adjusted", False)
    reduction_pct = altitude_adj.get("reduction_pct", 0)
    vo2max_adjusted = altitude_adj.get("vo2max_adjusted", 0)
    vo2max_sea = vo2max_data.get("value", 0)

    # === ALTITUDE MODEL TABLE ===
    elements.append(Paragraph("<b>MODEL REDUKCJI VO₂max</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    altitude_levels = [
        ["Wysokość (m)", "Redukcja VO₂max", "Szacowany VO₂max", "Kontekst"],
        ["0 (p.m.)", "0%", f"{vo2max_sea:.1f} ml/kg/min" if vo2max_sea > 0 else "---", "Poziom morza — wartość referencyjna"],
        ["500", "3.2%", f"{vo2max_sea * 0.968:.1f}" if vo2max_sea > 0 else "---", "Minimalna hipoksja"],
        ["1000", "6.3%", f"{vo2max_sea * 0.937:.1f}" if vo2max_sea > 0 else "---", "Kraków, Monachium"],
        ["1500", "9.5%", f"{vo2max_sea * 0.905:.1f}" if vo2max_sea > 0 else "---", "Livigno, Font Romeu"],
        ["2000", "12.6%", f"{vo2max_sea * 0.874:.1f}" if vo2max_sea > 0 else "---", "Sierra Nevada, Flagstaff"],
        ["2500", "15.8%", f"{vo2max_sea * 0.842:.1f}" if vo2max_sea > 0 else "---", "La Paz, Bogota"],
    ]

    at = Table(altitude_levels, colWidths=[28 * mm, 28 * mm, 38 * mm, 70 * mm])
    at.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 9 * mm),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor("#d5f5e3")),
    ]))
    elements.append(at)
    elements.append(Spacer(1, 6 * mm))

    # === CURRENT TEST CONTEXT ===
    if is_adjusted and altitude_m > 0:
        elements.append(Paragraph("<b>KOREKTA DLA BIEŻĄCEGO TESTU</b>", styles["subheading"]))
        elements.append(Spacer(1, 2 * mm))

        adj_data = [
            ["Parametr", "Wartość"],
            ["Wysokość testu", f"{altitude_m:.0f} m n.p.m."],
            ["Redukcja VO₂max", f"-{reduction_pct:.1f}%"],
            ["VO₂max na wysokości", f"{vo2max_adjusted:.1f} ml/kg/min"],
            ["VO₂max sea-level equiv.", f"{vo2max_sea:.1f} ml/kg/min"],
        ]

        adj_t = Table(adj_data, colWidths=[50 * mm, 50 * mm])
        adj_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#8E44AD")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
        ]))
        elements.append(adj_t)
        elements.append(Spacer(1, 4 * mm))
    else:
        elements.append(Paragraph(
            "<font color='#27AE60'><b>Test wykonany na niskiej wysokości (&lt;500m n.p.m.) — "
            "korekta nie jest wymagana.</b></font>",
            styles["body"]
        ))
        elements.append(Spacer(1, 4 * mm))

    # === PRACTICAL IMPLICATIONS ===
    elements.append(Paragraph("<b>IMPLIKACJE PRAKTYCZNE</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    elements.append(Paragraph(
        "<b>Dla zawodnika:</b> Jeśli trenujesz lub startujesz na wysokości, Twoja moc na progu "
        "i VO₂max będą niższe niż na poziomie morza. Strefy treningowe należy obniżyć proporcjonalnie. "
        "Pełna aklimatyzacja wysokościowa wymaga 14-21 dni (live high, train low model).",
        styles["body"]
    ))
    elements.append(Spacer(1, 2 * mm))

    elements.append(Paragraph(
        "<b>Altitude Training Camp:</b> Optymalny zakres to 2000-2500m n.p.m. do mieszkania "
        "i <1200m do treningu intensywnego (LHTL). Efekt: wzrost masy hemoglobiny o 1-3% w 3 tygodnie "
        "(Garvican-Lewis et al. 2016), przekładający się na +1-2% VO₂max po powrocie na niziny.",
        styles["body"]
    ))
    elements.append(Spacer(1, 6 * mm))

    # === REFERENCE ===
    elements.append(Paragraph(
        "<font size='7' color='#95A5A6'><i>"
        "Ref: Wehrlin &amp; Hallen 2006 — 6.3%/1000m linear model. "
        "Pühringer et al. 2022 — 5.0-11.6%/1000m above 1500m. "
        "Townsend et al. 2017 — curvilinear meta-regression. "
        "Garvican-Lewis et al. 2016 — hemoglobin mass response to altitude."
        "</i></font>",
        styles["body"]
    ))

    return elements


# ============================================================================
# PAGE: MICROCYCLE PERIODIZATION
# ============================================================================


def build_page_microcycle(metabolic_data: Dict[str, Any], styles: Dict) -> List:
    """Build weekly microcycle periodization page.

    Generates a 7-day training schedule based on the training block
    from metabolic engine, distributing sessions across the week.
    """

    elements = []

    elements.append(Paragraph(
        "<font size='14'>2.5 PERIODYZACJA MIKROCYKLOWA</font>",
        styles["center"]
    ))
    elements.append(Paragraph(
        "<font size='10' color='#7F8C8D'>"
        "Tygodniowy rozkład treningowy — automatyczna periodyzacja"
        "</font>",
        styles["center"]
    ))
    elements.append(Spacer(1, 6 * mm))

    block = metabolic_data.get("training_block", {})
    sessions = block.get("sessions", [])
    block_name = block.get("name", "Training Block")
    primary_focus = block.get("primary_focus", "")

    if not sessions:
        elements.append(Paragraph(
            "<font color='#E74C3C'><b>Brak danych bloku treningowego.</b></font>",
            styles["body"]
        ))
        return elements

    elements.append(Paragraph(
        f"Na podstawie bloku <b>{block_name}</b> ({primary_focus}), "
        "poniżej proponowany rozkład tygodniowy. Kolejność sesji uwzględnia "
        "zasadę superkompensacji: sesje intensywne rozdzielone 48h, "
        "sesje bazowe (Z2) jako aktywna regeneracja.",
        styles["body"]
    ))
    elements.append(Spacer(1, 4 * mm))

    # Build weekly schedule from sessions
    # Strategy: distribute sessions across 7 days with recovery logic
    day_names_pl = [
        "Poniedziałek", "Wtorek", "Środa", "Czwartek",
        "Piątek", "Sobota", "Niedziela"
    ]

    # Classify sessions by intensity
    intense_sessions = []
    moderate_sessions = []
    easy_sessions = []

    for s in sessions:
        name_lower = s.get("name", "").lower()
        if any(k in name_lower for k in ["vo2", "hill", "sprint", "over-under", "frc", "threshold", "race"]):
            intense_sessions.append(s)
        elif any(k in name_lower for k in ["tempo", "sweet spot", "strength", "cadence", "double"]):
            moderate_sessions.append(s)
        else:
            easy_sessions.append(s)

    # Build weekly plan
    weekly_plan = [None] * 7  # Mon=0 ... Sun=6

    # Place intense sessions: Tue(1), Thu(3), Sat(5) — separated by 48h
    intense_slots = [1, 3, 5]
    for i, slot in enumerate(intense_slots):
        if i < len(intense_sessions):
            weekly_plan[slot] = intense_sessions[i]

    # Place moderate sessions: Wed(2), Fri(4)
    moderate_slots = [2, 4]
    for i, slot in enumerate(moderate_slots):
        if i < len(moderate_sessions):
            weekly_plan[slot] = moderate_sessions[i]

    # Place easy sessions on remaining days
    easy_slots = [s for s in range(7) if weekly_plan[s] is None]
    for i, slot in enumerate(easy_slots):
        if i < len(easy_sessions):
            weekly_plan[slot] = easy_sessions[i]

    # === WEEKLY TABLE ===
    elements.append(Paragraph("<b>PRZYKŁADOWY MIKROCYKL TYGODNIOWY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    week_data = [["Dzień", "Sesja", "Moc / Czas", "Intensywność"]]

    for day_idx, session in enumerate(weekly_plan):
        day_name = day_names_pl[day_idx]
        if session is None:
            if day_idx == 0:  # Monday = rest
                week_data.append([day_name, "ODPOCZYNEK / REGENERACJA", "---", "REST"])
            elif day_idx == 6:  # Sunday
                week_data.append([day_name, "Z2 Długie lub ODPOCZYNEK", "2-4h @ 55-70% FTP", "Z1-Z2"])
            else:
                week_data.append([day_name, "Z2 Bazowe", "60-90min @ 55-70% FTP", "Z2"])
        else:
            name = session.get("name", "---")
            power = session.get("power_range", "---")
            duration = session.get("duration", "---")
            # Determine intensity label
            name_l = name.lower()
            if any(k in name_l for k in ["vo2", "hill", "sprint", "over-under", "frc", "threshold", "race"]):
                intensity = "Z4-Z5"
            elif any(k in name_l for k in ["tempo", "sweet", "strength", "cadence"]):
                intensity = "Z3-Z4"
            else:
                intensity = "Z1-Z2"
            week_data.append([day_name, name, f"{power}\n{duration}", intensity])

    wt = Table(week_data, colWidths=[28 * mm, 48 * mm, 52 * mm, 28 * mm])
    wt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1A5276")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 12 * mm),
    ]))

    # Color-code intensity column
    for row_idx in range(1, len(week_data)):
        intensity = week_data[row_idx][3]
        if intensity == "REST":
            color = "#d5f5e3"
        elif intensity in ("Z1-Z2", "Z2"):
            color = "#d5f5e3"
        elif intensity in ("Z3-Z4",):
            color = "#fdebd0"
        elif intensity in ("Z4-Z5",):
            color = "#fadbd8"
        else:
            color = "#f5f5f5"
        wt.setStyle(TableStyle([
            ('BACKGROUND', (3, row_idx), (3, row_idx), HexColor(color)),
        ]))
        # Monday rest = full row green
        if intensity == "REST":
            wt.setStyle(TableStyle([
                ('BACKGROUND', (0, row_idx), (-1, row_idx), HexColor("#d5f5e3")),
            ]))

    elements.append(wt)
    elements.append(Spacer(1, 6 * mm))

    # === PERIODIZATION RULES ===
    elements.append(Paragraph("<b>ZASADY PERIODYZACJI</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    rules = [
        "Sesje intensywne (Z4-Z5) rozdzielone minimum 48h — pełna resynteza glikogenu i adaptacja nerwowo-mięśniowa",
        "Poniedziałek = odpoczynek po weekendowej długiej sesji (superkompensacja)",
        "Bloki Tempo/Sweet Spot w środku tygodnia — kompromis między bodźcem a recovery",
        "Tygodniowy rozkład objętości: ~80% Z1-Z2, ~15% Z3-Z4, ~5% Z5 (model polaryzowany)",
        "Co 3-4 tygodnie: tydzień rozładowczy (TSS -40%) — zapobieganie overreaching",
    ]

    for rule in rules:
        elements.append(Paragraph(f"<font size='8'>• {rule}</font>", styles["body"]))

    elements.append(Spacer(1, 6 * mm))

    # === TSS ESTIMATION ===
    elements.append(Paragraph("<b>SZACUNKOWY TSS TYGODNIOWY</b>", styles["subheading"]))
    elements.append(Spacer(1, 2 * mm))

    # Estimate weekly TSS from sessions
    total_estimated_tss = 0
    for session in weekly_plan:
        if session is None:
            total_estimated_tss += 30  # Rest / easy day
        else:
            name_l = session.get("name", "").lower()
            if any(k in name_l for k in ["vo2", "hill", "sprint", "frc", "race"]):
                total_estimated_tss += 100
            elif any(k in name_l for k in ["tempo", "sweet", "threshold", "over-under"]):
                total_estimated_tss += 85
            elif any(k in name_l for k in ["strength", "cadence", "double"]):
                total_estimated_tss += 70
            elif any(k in name_l for k in ["long", "base", "z2", "fasted", "sub-lt1"]):
                total_estimated_tss += 120
            else:
                total_estimated_tss += 60

    tss_data = [
        ["Metryka", "Wartość", "Kontekst"],
        ["TSS/tydzień (szacunkowy)", f"~{total_estimated_tss}", "Na podstawie rozkładu sesji"],
        ["TSS/tydzień rozładowczy", f"~{int(total_estimated_tss * 0.6)}", "Co 3-4 tygodnie, -40%"],
        ["CTL target (6 tyg.)", f"~{int(total_estimated_tss * 0.8)}", "Chronic Training Load"],
    ]

    tss_t = Table(tss_data, colWidths=[45 * mm, 35 * mm, 80 * mm])
    tss_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1F77B4")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
        ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ('ROWHEIGHT', (0, 0), (-1, -1), 10 * mm),
    ]))
    elements.append(tss_t)

    return elements


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
