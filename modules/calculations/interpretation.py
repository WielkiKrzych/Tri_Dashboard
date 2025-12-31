"""
Interpretation & Prescription Engine.

Translates physiological metrics into actionable training advice.
Strictly gates advice based on data quality (Reliability Check).
"""
from typing import Dict, List, Optional, Any

def generate_training_advice(
    metrics: Dict[str, Any],
    quality_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate diagnostic and prescriptive advice based on metrics.
    
    Args:
        metrics: Dictionary containing:
            - vt1_watts, vt2_watts (Thresholds)
            - smo2_tau (Recovery Kinetics)
            - smo2_min, smo2_max (Range)
            - lag_hr, lag_smo2 (Response times)
            - vo2max_est (Aerobic capacity)
        quality_report: Output from quality.check_data_quality / protocol_check
        
    Returns:
        Dict with:
        - 'diagnostics': List of observations ("Aerobic Base Weak")
        - 'prescriptions': List of actions ("Do Zone 2")
        - 'warnings': List of reliability warnings
        - 'is_valid': Boolean
    """
    
    # 1. Reliability Gate
    if not quality_report.get('is_valid', True):
        return {
            "diagnostics": [],
            "prescriptions": [],
            "warnings": ["Data Unreliable: " + "; ".join(quality_report.get('issues', []))],
            "is_valid": False
        }
        
    diagnostics = []
    prescriptions = []
    warnings = []
    
    vt1 = metrics.get('vt1_watts')
    vt2 = metrics.get('vt2_watts')
    tau = metrics.get('smo2_tau')
    
    # 2. Aerobic Profile Analysis (VT1/VT2)
    if vt1 and vt2 and vt2 > 0:
        ratio = vt1 / vt2
        if ratio < 0.65:
            diagnostics.append("Aerobic Deficiency: VT1 is low relative to VT2 (<65%).")
            prescriptions.append("Focus on Base Building: High volume of Zone 2 (LSD) to improve fat oxidation.")
        elif ratio > 0.85:
            diagnostics.append("High Aerobic Base: VT1 is close to VT2 (>85%). 'Diesel Engine' profile.")
            prescriptions.append("Polarized Training: Introduce high-intensity VO2max intervals to raise the ceiling.")
        else:
            diagnostics.append("Balanced Aerobic Profile (VT1 is 65-85% of VT2).")
            prescriptions.append("Maintenance: Mix of Tempo and Threshold work.")
            
    # 3. Recovery Kinetics Analysis (SmO2 Tau)
    if tau:
        if tau > 45:
            diagnostics.append(f"Slow Recovery Kinetics (Tau={tau:.1f}s). Phosphocreatine resynthesis is sluggish.")
            prescriptions.append("Interval Training: short-rest intervals (e.g. 30/15s) to stress recovery systems.")
        elif tau < 25:
            diagnostics.append(f"Fast Recovery Kinetics (Tau={tau:.1f}s). Excellent oxidative capacity.")
            prescriptions.append("Repeated Sprint Ability: Focus on capacity/durability as recovery is already a strength.")
            
    # 4. Limiter Analysis (SmO2 Trends)
    # Using 'context' if passed, or heuristics
    # E.g. if Max SmO2 is low (<50%)?
    smo2_max = metrics.get('smo2_max')
    if smo2_max and smo2_max < 60:
        # This might be sensor placement or true limit
        warnings.append("Low absolute SmO2: Check sensor placement or potential delivery limitation.")

    # 5. Lag Analysis
    lag_hr = metrics.get('lag_hr')
    if lag_hr and lag_hr > 60:
        diagnostics.append(f"Slow HR Response (Lag {lag_hr}s). HR intersects power late.")
        prescriptions.append("Pacing Strategy: Do not pace short intervals by HR; use Power or RPE.")

    if not diagnostics:
        diagnostics.append("Profile Normal. No specific limiting factors identified.")
    
    return {
        "diagnostics": diagnostics,
        "prescriptions": prescriptions,
        "warnings": warnings,
        "is_valid": True
    }

def get_feedback_style(severity: str) -> str:
    """Return styling color/icon for severity."""
    if severity == 'high':
        return "🔴"
    elif severity == 'medium':
        return "🟠"
    return "🟢"
