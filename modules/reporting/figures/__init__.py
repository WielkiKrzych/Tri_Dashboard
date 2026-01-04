"""
Static Figure Generator for Ramp Test PDF Reports.
Orchestrates figure generation for various chart types.
"""
from pathlib import Path
from typing import Dict, Any, Optional

from .common import DPI, get_color, save_figure
from .ramp_profile import generate_ramp_profile_chart
from .smo2_vs_power import generate_smo2_power_chart
from .cp_curve import generate_cp_curve_chart, generate_pdc_chart
from .ve_profile import generate_ve_profile_chart
from .thermal import generate_thermal_chart, generate_efficiency_chart
from .limiters import generate_radar_chart
from .vent_full import generate_full_vent_chart
from .drift import generate_power_hr_scatter, generate_power_smo2_scatter


def generate_all_ramp_figures(
    report_data: Dict[str, Any],
    output_dir: str,
    config: Optional[Any] = None,
    source_df: Optional["pd.DataFrame"] = None
) -> Dict[str, str]:
    """Generate all ramp test figures and save to directory.
    
    Args:
        report_data: Canonical JSON report dictionary
        output_dir: Directory to save figures
        config: Optional configuration dictionary or object
        source_df: Optional source DataFrame with raw data
        
    Returns:
        Dict mapping figure name to file path
    """
    config = config or {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_id = report_data.get("metadata", {}).get("session_id", "unknown")[:8]
    
    # Handle config as dict if passed, or use attributes
    if isinstance(config, dict):
        ext = config.get('format', 'png')
    else:
        ext = getattr(config, 'format', 'png')
    
    paths = {}
    
    # 1. Ramp profile
    ramp_path = output_path / f"ramp_profile_{session_id}.{ext}"
    generate_ramp_profile_chart(report_data, config, str(ramp_path), source_df=source_df)
    paths["ramp_profile"] = str(ramp_path)
    
    # 2. SmO2 vs Power
    smo2_path = output_path / f"smo2_power_{session_id}.{ext}"
    generate_smo2_power_chart(report_data, config, str(smo2_path), source_df=source_df)
    paths["smo2_power"] = str(smo2_path)
    
    # 3. PDC / CP Curve
    pdc_path = output_path / f"pdc_{session_id}.{ext}"
    generate_pdc_chart(report_data, config, str(pdc_path))
    paths["pdc_curve"] = str(pdc_path)
    
    # 4. VE Profile
    ve_path = output_path / f"ve_profile_{session_id}.{ext}"
    generate_ve_profile_chart(report_data, config, str(ve_path), source_df=source_df)
    paths["ve_profile"] = str(ve_path)
    
    # 5. Thermal
    thermal_hsi_path = output_path / f"thermal_hsi_{session_id}.{ext}"
    generate_thermal_chart(report_data, config, str(thermal_hsi_path), source_df=source_df)
    paths["thermal_hsi"] = str(thermal_hsi_path)
    
    thermal_eff_path = output_path / f"thermal_eff_{session_id}.{ext}"
    generate_efficiency_chart(report_data, config, str(thermal_eff_path), source_df=source_df)
    paths["thermal_efficiency"] = str(thermal_eff_path)
    
    # 6. Limiters (Radar)
    radar_path = output_path / f"limiters_radar_{session_id}.{ext}"
    generate_radar_chart(report_data, config, str(radar_path), source_df=source_df)
    paths["limiters_radar"] = str(radar_path)
    
    # 7. Vent Full
    vent_full_path = output_path / f"vent_full_{session_id}.{ext}"
    generate_full_vent_chart(report_data, config, str(vent_full_path), source_df=source_df)
    paths["vent_full"] = str(vent_full_path)
    
    # 8. Drift Scatters
    drift_hr_path = output_path / f"drift_hr_{session_id}.{ext}"
    generate_power_hr_scatter(report_data, config, str(drift_hr_path), source_df=source_df)
    paths["drift_hr"] = str(drift_hr_path)
    
    drift_smo2_path = output_path / f"drift_smo2_{session_id}.{ext}"
    generate_power_smo2_scatter(report_data, config, str(drift_smo2_path), source_df=source_df)
    paths["drift_smo2"] = str(drift_smo2_path)
    
    return paths


__all__ = [
    "generate_all_ramp_figures",
    "generate_ramp_profile_chart",
    "generate_smo2_power_chart",
    "generate_pdc_chart",
    "generate_cp_curve_chart",
    "DPI",
]
