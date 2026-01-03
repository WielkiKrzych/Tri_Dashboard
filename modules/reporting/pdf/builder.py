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
import logging
from typing import Dict, Any, Optional, List

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


# Setup logger
logger = logging.getLogger("Tri_Dashboard.PDFBuilder")


def map_ramp_json_to_pdf_data(report_json: Dict[str, Any]) -> Dict[str, Any]:
    """Map canonical JSON report to internal PDF data structure.
    
    This is the ONLY function that reads the raw JSON structure.
    Ensures all required fields for layouts are present with safe fallbacks.
    
    Args:
        report_json: Raw canonical JSON report
        
    Returns:
        Dict mapped for PDF generation
    """
    def get_num(section: str, key: str, path: List[str], fallback: str = "brak danych"):
        """Deep get with formatting and logging."""
        curr = report_json.get(section, {})
        for p in path[:-1]:
            curr = curr.get(p, {})
        
        val = curr.get(path[-1]) if curr else None
        
        if val is None or val == "" or val == "-":
            logger.warning(f"PDF Mapping: Missing expected field {section}.{'.'.join(path)}")
            return fallback
            
        if isinstance(val, (int, float)):
            if val == int(val):
                return str(int(val))
            return f"{val:.0f}"
        
        # Handle lists (ranges)
        if isinstance(val, list) and len(val) == 2:
            try:
                return f"{float(val[0]):.0f}–{float(val[1]):.0f}"
            except (ValueError, TypeError):
                pass
                
        return str(val)

    # 1. Metadata extraction
    meta = report_json.get("metadata", {})
    
    # Try to get pmax from various places
    validity = report_json.get("test_validity", {})
    metrics = validity.get("metrics", {})
    pmax_list = metrics.get("power_range_watts")
    pmax_val = "brak danych"
    if isinstance(pmax_list, list) and len(pmax_list) == 2:
        pmax_val = f"{pmax_list[1]:.0f}"
    elif "pmax_watts" in meta:
        pmax_val = str(round(meta["pmax_watts"]))
    
    if pmax_val == "brak danych":
         # logger.warning("PDF Mapping: Pmax not found in metadata or test_validity")
         pass

    mapped_meta = {
        "test_date": meta.get("test_date", "brak danych"),
        "session_id": meta.get("session_id", "nieznany"),
        "method_version": meta.get("method_version", "1.0.0"),
        "protocol": meta.get("protocol", "Ramp Test"),
        "notes": meta.get("notes", "-"),
        "pmax_watts": pmax_val,
        "athlete_weight_kg": meta.get("athlete_weight_kg", 0)
    }

    # 2. Thresholds (midpoints and ranges)
    thresholds = report_json.get("thresholds", {})
    
    # Debug helper for VE
    vt1_data = thresholds.get("vt1", {})
    vt2_data = thresholds.get("vt2", {})
    
    mapped_thresholds = {
        "vt1_watts": get_num("thresholds", "vt1", ["vt1", "midpoint_watts"]),
        "vt1_hr": get_num("thresholds", "vt1", ["vt1", "midpoint_hr"]),
        "vt1_ve": get_num("thresholds", "vt1", ["vt1", "midpoint_ve"]),
        "vt1_range_watts": get_num("thresholds", "vt1", ["vt1", "range_watts"]),
        
        "vt2_watts": get_num("thresholds", "vt2", ["vt2", "midpoint_watts"]),
        "vt2_hr": get_num("thresholds", "vt2", ["vt2", "midpoint_hr"]),
        "vt2_ve": get_num("thresholds", "vt2", ["vt2", "midpoint_ve"]),
        "vt2_range_watts": get_num("thresholds", "vt2", ["vt2", "range_watts"]),
        
        "vt1_raw_midpoint": report_json.get("thresholds", {}).get("vt1", {}).get("midpoint_watts"), # for calcs
        "vt2_raw_midpoint": report_json.get("thresholds", {}).get("vt2", {}).get("midpoint_watts"),
    }
    
    # Range formatting is now handled inside get_num

    # 3. SmO2 Context
    smo2 = report_json.get("smo2_context", {})
    mapped_smo2 = {
        "drop_point_watts": "brak danych",
        "interpretation": smo2.get("interpretation", "nie przeanalizowano")
    }
    if smo2 and "drop_point" in smo2 and smo2["drop_point"]:
        mapped_smo2["drop_point_watts"] = get_num("smo2_context", "drop_point", ["drop_point", "midpoint_watts"])

    # 4. CP Model mapping (if available)
    cp = report_json.get("cp_model", {})
    mapped_cp = {
        "cp_watts": get_num("cp_model", "cp_watts", ["cp_watts"]),
        "w_prime_kj": "brak danych"
    }
    w_prime = cp.get("w_prime_joules")
    if w_prime is not None and isinstance(w_prime, (int, float)):
        mapped_cp["w_prime_kj"] = f"{w_prime / 1000:.0f}"

    # 5. Interpretation & Confidence
    interp = report_json.get("interpretation", {})
    mapped_confidence = {
        "overall_confidence": interp.get("overall_confidence", 0.0),
        "confidence_level": interp.get("confidence_level", "low"),
        "warnings": interp.get("warnings", []),
        "notes": interp.get("notes", [])
    }

    return {
        "metadata": mapped_meta,
        "thresholds": mapped_thresholds,
        "smo2": mapped_smo2,
        "cp_model": mapped_cp,
        "confidence": mapped_confidence
    }


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
    
    # Map data with robust fallbacks
    pdf_data = map_ramp_json_to_pdf_data(report_data)
    
    metadata = pdf_data["metadata"]
    thresholds = pdf_data["thresholds"]
    cp_model = pdf_data["cp_model"]
    confidence = pdf_data["confidence"]
    
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
        smo2=pdf_data["smo2"],
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
