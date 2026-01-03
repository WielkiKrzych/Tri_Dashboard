"""
PDF Builder Module.

Orchestrates the construction of complete Ramp Test PDF reports.
Combines styles and layout sections into final document.
No physiological calculations - only document assembly.
"""
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.platypus import SimpleDocTemplate, PageBreak, Spacer
from reportlab.lib.units import mm

from .styles import PDFConfig, create_styles, PAGE_SIZE, MARGIN
from .layout import (
    build_header,
    build_confidence_badge,
    build_conditional_warning,
    build_results_summary,
    build_chart_section,
    build_thresholds_section,
    build_limitations_section,
)


def build_ramp_pdf(
    report_data: Dict[str, Any],
    figure_paths: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    config: Optional[PDFConfig] = None
) -> bytes:
    """Build complete Ramp Test PDF report.
    
    Assembles all sections into a multi-page PDF document.
    
    Args:
        report_data: Canonical JSON report dictionary
        figure_paths: Dict mapping figure name to file path
        output_path: Optional file path to save PDF
        config: PDF configuration
        
    Returns:
        PDF bytes if output_path is None
    """
    config = config or PDFConfig()
    
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
    
    # Create styles
    styles = create_styles()
    
    # Extract data sections
    metadata = report_data.get("metadata", {})
    thresholds = report_data.get("thresholds", {})
    cp_model = report_data.get("cp_model", {})
    confidence = report_data.get("confidence", {})
    
    # Build story (list of flowables)
    story = []
    
    # === PAGE 1: Header + Summary ===
    story.extend(build_header(metadata, styles))
    story.append(Spacer(1, 10 * mm))
    
    story.extend(build_confidence_badge(confidence, styles))
    story.append(Spacer(1, 8 * mm))
    
    # Conditional warning (if applicable)
    if config.is_conditional:
        story.extend(build_conditional_warning(styles))
        story.append(Spacer(1, 8 * mm))
    
    story.extend(build_results_summary(thresholds, cp_model, metadata, styles))
    story.append(Spacer(1, 8 * mm))
    
    # Ramp profile chart
    if figure_paths and "ramp_profile" in figure_paths:
        story.extend(build_chart_section(
            figure_paths["ramp_profile"],
            "Przebieg Testu Ramp",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 2: Thresholds + SmO2 ===
    story.extend(build_thresholds_section(thresholds, styles))
    story.append(Spacer(1, 8 * mm))
    
    # SmO2 chart
    if figure_paths and "smo2_power" in figure_paths:
        story.extend(build_chart_section(
            figure_paths["smo2_power"],
            "SmO₂ vs Moc",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 3: PDC / CP ===
    if figure_paths and "pdc" in figure_paths:
        story.extend(build_chart_section(
            figure_paths["pdc"],
            "Power-Duration Curve (PDC)",
            styles
        ))
    
    story.append(PageBreak())
    
    # === PAGE 4: Limitations ===
    story.extend(build_limitations_section(styles))
    
    # Build PDF
    doc.build(story)
    
    # Get bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes


# Alias for backward compatibility
generate_ramp_pdf = build_ramp_pdf
