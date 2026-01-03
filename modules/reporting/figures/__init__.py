"""
Static Figure Generator for Ramp Test PDF Reports.

Generates PNG/SVG charts based on JSON report data.
No Streamlit dependency - pure matplotlib.
"""
from .ramp_figures import (
    generate_ramp_profile_chart,
    generate_smo2_power_chart,
    generate_pdc_chart,
    generate_all_ramp_figures,
    FigureConfig,
)

__all__ = [
    "generate_ramp_profile_chart",
    "generate_smo2_power_chart", 
    "generate_pdc_chart",
    "generate_all_ramp_figures",
    "FigureConfig",
]
