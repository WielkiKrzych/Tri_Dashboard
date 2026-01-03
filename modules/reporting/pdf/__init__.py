"""
PDF Generator Module for Ramp Test Reports.

Generates publication-ready PDF reports from canonical JSON data.
Uses ReportLab library. No Streamlit dependency.

Module Structure:
- styles.py: Typography, colors, reusable styles
- layout.py: Page sections and content builders
- builder.py: Document orchestration and assembly

Usage:
    from modules.reporting.pdf import build_ramp_pdf, PDFConfig
    
    pdf_bytes = build_ramp_pdf(report_data, figure_paths, output_path)
"""
from .styles import PDFConfig, create_styles, COLORS, PAGE_SIZE, MARGIN
from .builder import build_ramp_pdf, generate_ramp_pdf


__all__ = [
    # Main API
    "build_ramp_pdf",
    "generate_ramp_pdf",  # Alias for backward compatibility
    # Configuration
    "PDFConfig",
    # Styles (for advanced usage)
    "create_styles",
    "COLORS",
    "PAGE_SIZE",
    "MARGIN",
]
