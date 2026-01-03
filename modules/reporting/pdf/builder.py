"""
PDF Builder Module.

Orchestrates the construction of complete Ramp Test PDF reports.
Assembles all 6 pages as per methodology/ramp_test/10_pdf_layout.md.

Pages:
1. Okładka / Podsumowanie
2. Szczegóły Progów VT1/VT2
3. Power-Duration Curve / CP
4. Interpretacja Wyników
5. Strefy Treningowe
6. Ograniczenia Interpretacji

No physiological calculations - only document assembly.
"""
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer, Paragraph
from reportlab.lib.units import mm

from .styles import PDFConfig, create_styles, PAGE_SIZE, MARGIN, COLORS, FONT_FAMILY
from .layout import (
    build_page_cover,
    build_page_thresholds,
    build_page_pdc,
    build_page_interpretation,
    build_page_zones,
    build_page_limitations,
)


def build_ramp_pdf(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    config: Optional[PDFConfig] = None
) -> bytes:
    """Build complete Ramp Test PDF report (6 pages).
    
    Assembles all sections into a multi-page PDF document
    following the specification in 10_pdf_layout.md.
    
    Args:
        report_data: Canonical JSON report dictionary
        figure_paths: Dict mapping figure name to file path
        output_path: Optional file path to save PDF
        config: PDF configuration
        
    Returns:
        PDF bytes
    """
    config = config or PDFConfig()
    figure_paths = figure_paths or {}
    
    # Setup document with custom page callback for footer
    buffer = BytesIO()
    
    # Extract data sections
    metadata = report_data.get("metadata", {})
    thresholds = report_data.get("thresholds", {})
    cp_model = report_data.get("cp_model", {})
    confidence = report_data.get("confidence", {})
    
    # Store metadata for footer
    session_id = metadata.get("session_id", "")[:8]
    method_version = metadata.get("method_version", "1.0.0")
    
    def add_page_footer(canvas, doc):
        """Add footer to each page."""
        canvas.saveState()
        
        # Footer text
        page_num = doc.page
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        footer_text = f"Strona {page_num} | ID: {session_id} | v{method_version} | {timestamp} | Tri_Dashboard"
        
        canvas.setFont(FONT_FAMILY, 8)
        canvas.setFillColor(COLORS["text_light"])
        canvas.drawCentredString(PAGE_SIZE[0] / 2, 10 * mm, footer_text)
        
        canvas.restoreState()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=config.page_size,
        leftMargin=config.margin,
        rightMargin=config.margin,
        topMargin=config.margin,
        bottomMargin=20 * mm  # Extra space for footer
    )
    
    # Create styles
    styles = create_styles()
    
    # Build story (list of flowables)
    story = []
    
    # === PAGE 1: Okładka / Podsumowanie ===
    story.extend(build_page_cover(
        metadata=metadata,
        thresholds=thresholds,
        cp_model=cp_model,
        confidence=confidence,
        figure_paths=figure_paths,
        styles=styles,
        is_conditional=config.is_conditional
    ))
    story.append(PageBreak())
    
    # === PAGE 2: Szczegóły Progów VT1/VT2 ===
    story.extend(build_page_thresholds(
        thresholds=thresholds,
        figure_paths=figure_paths,
        styles=styles
    ))
    story.append(PageBreak())
    
    # === PAGE 3: Power-Duration Curve / CP ===
    story.extend(build_page_pdc(
        cp_model=cp_model,
        metadata=metadata,
        figure_paths=figure_paths,
        styles=styles
    ))
    story.append(PageBreak())
    
    # === PAGE 4: Interpretacja Wyników ===
    story.extend(build_page_interpretation(
        thresholds=thresholds,
        cp_model=cp_model,
        styles=styles
    ))
    story.append(PageBreak())
    
    # === PAGE 5: Strefy Treningowe ===
    story.extend(build_page_zones(
        thresholds=thresholds,
        styles=styles
    ))
    story.append(PageBreak())
    
    # === PAGE 6: Ograniczenia Interpretacji ===
    story.extend(build_page_limitations(
        styles=styles,
        is_conditional=config.is_conditional
    ))
    
    # Build PDF with footer on each page
    doc.build(story, onFirstPage=add_page_footer, onLaterPages=add_page_footer)
    
    # Get bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        print(f"PDF saved to: {output_path}")
    
    return pdf_bytes


# Alias for backward compatibility
generate_ramp_pdf = build_ramp_pdf
