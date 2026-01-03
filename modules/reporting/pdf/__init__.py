"""
PDF Report Generation Module.

Contains PDF generator for Ramp Test reports.
No Streamlit dependency - pure reportlab.
"""
from .generator import generate_ramp_pdf, PDFConfig

__all__ = [
    "generate_ramp_pdf",
    "PDFConfig",
]
