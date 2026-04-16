"""
Breathing & Metabolic Control Diagnostics Module.

Calculates advanced ventilation metrics for breathing pattern analysis:
- VE (L/min)
- RR (breaths/min)
- VE/RR Ratio (tidal volume proxy)
- VE Slope vs Power

Classifies breathing patterns:
- Controlled (efficient)
- Compensatory (shallow/panic)
- Unstable (hyperventilation)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger("Tri_Dashboard.VentilationAdvanced")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class VentilationMetrics:
    """Container for ventilation metrics."""

    # Core metrics
    ve_avg: float = 0.0  # Average VE (L/min)
    ve_max: float = 0.0  # Max VE
    rr_avg: float = 0.0  # Average respiratory rate
    rr_max: float = 0.0  # Max RR
    ve_rr_ratio: float = 0.0  # VE/RR = tidal volume proxy
    ve_slope: float = 0.0  # VE change per 100W

    # Breakpoints
    ve_breakpoint_watts: Optional[float] = None  # VE inflection (VT1/VT2)
    rr_breakpoint_watts: Optional[float] = None  # RR inflection

    # Pattern classification
    breathing_pattern: str = "unknown"  # controlled, shallow, hyperventilation
    control_status: str = "unknown"  # controlled, compensatory, unstable
    control_confidence: float = 0.0

    # Interpretation
    interpretation: str = ""
    recommendations: List[Dict[str, str]] = field(default_factory=list)

    # Profile for charting
    ve_profile: List[Dict[str, float]] = field(default_factory=list)
    rr_profile: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================


def calculate_ve_metrics(
    df: pd.DataFrame, ve_col: str = "ve", power_col: str = "watts"
) -> Tuple[float, float, float]:
    """
    Calculate VE metrics.

    Returns:
        (ve_avg, ve_max, ve_slope_per_100w)
    """
    # Try multiple column names for VE
    ve_cols = ["ve", "tymeventilation", "ventilation", "ve_lmin"]
    ve_data = None
    for col in ve_cols:
        if col in df.columns:
            ve_data = df[col]
            break

    if ve_data is None or power_col not in df.columns:
        return 0.0, 0.0, 0.0

    mask = df[power_col] > 100
    if mask.sum() < 20:
        return 0.0, 0.0, 0.0

    filtered_ve = ve_data[mask]
    filtered_power = df.loc[mask, power_col]

    ve_avg = float(filtered_ve.mean())
    ve_max = float(filtered_ve.max())

    # VE slope
    if filtered_power.nunique() > 1:
        slope, _, r, _, _ = stats.linregress(filtered_power, filtered_ve)
        ve_slope = slope * 100  # per 100W
    else:
        ve_slope = 0.0

    return ve_avg, ve_max, ve_slope


def calculate_rr_metrics(
    df: pd.DataFrame, rr_col: str = "rr", power_col: str = "watts"
) -> Tuple[float, float]:
    """
    Calculate respiratory rate metrics.

    Returns:
        (rr_avg, rr_max)
    """
    # Try multiple column names - including Tyme/Garmin variations
    rr_cols = [
        "rr",
        "resprate",
        "respiratory_rate",
        "breaths",
        "respiration_rate",
        "breathing_rate",
        "bf",  # Common aliases
        "tymerespirationrate",
        "respirationrate",  # Tyme wear
        "tymebreathrate",
        "breathrate",  # Tyme breath rate (CONFIRMED)
        "enhancedresprate",
        "enhanced_resp_rate",  # Garmin
    ]
    rr_data = None
    for col in rr_cols:
        matching = [c for c in df.columns if c.lower().replace("_", "") == col.replace("_", "")]
        if matching:
            rr_data = df[matching[0]]
            break

    if rr_data is None:
        return 0.0, 0.0

    mask = df[power_col] > 100 if power_col in df.columns else pd.Series(True, index=df.index)
    if mask.sum() < 10:
        return 0.0, 0.0

    filtered_rr = rr_data[mask]

    return float(filtered_rr.mean()), float(filtered_rr.max())


def calculate_ve_rr_ratio(ve_avg: float, rr_avg: float) -> float:
    """
    Calculate VE/RR ratio (proxy for tidal volume).

    VE = TV × RR, so TV ≈ VE/RR
    """
    if rr_avg > 0:
        return ve_avg / rr_avg
    return 0.0


def find_ve_breakpoint(
    df: pd.DataFrame, ve_col: str = "ve", power_col: str = "watts", window: int = 30
) -> Optional[float]:
    """
    Find VE inflection point (ventilatory threshold).

    Uses slope change detection.
    """
    # Try multiple column names
    ve_cols = ["ve", "tymeventilation", "ventilation"]
    ve_data = None
    for col in ve_cols:
        if col in df.columns:
            ve_data = df[col]
            break

    if ve_data is None or power_col not in df.columns:
        return None

    mask = df[power_col] > 100
    if mask.sum() < 50:
        return None

    filtered = df[mask].copy()
    power = filtered[power_col].values
    ve = ve_data[mask].values

    # Smooth
    ve_smooth = pd.Series(ve).rolling(window=window, min_periods=1).mean().values

    # Calculate local slope in windows
    n = len(ve_smooth)
    slopes = []
    for i in range(0, n - 20, 10):
        segment_power = power[i : i + 20]
        segment_ve = ve_smooth[i : i + 20]
        if len(segment_power) > 5 and np.unique(segment_power).size > 1:
            try:
                s, _, _, _, _ = stats.linregress(segment_power, segment_ve)
                slopes.append((power[i + 10], s))
            except Exception as e:
                logger.debug(f"Linregress failed in segment: {e}")
                pass

    if len(slopes) < 3:
        return None

    # Find where slope increases significantly
    slopes_arr = np.array([s[1] for s in slopes])
    baseline = np.mean(slopes_arr[:3])

    for power_val, slope in slopes:
        if slope > baseline * 1.5:  # 50% increase
            return float(power_val)

    return None


def classify_breathing_pattern(
    rr_avg: float, rr_max: float, ve_rr_ratio: float, ve_slope: float
) -> Tuple[str, str]:
    """
    Classify breathing pattern based on metrics.

    Returns:
        (pattern, description)
    """
    # High RR with low TV = shallow/panic
    if rr_max > 50 and ve_rr_ratio < 1.5:
        return "shallow", "Shallow/Panic Breathing – płytkie oddechy, niska objętość oddechowa"

    # Very high VE slope = hyperventilation
    if ve_slope > 0.5:  # > 0.5 L/min per watt
        return "hyperventilation", "Hyperventilation – nadmierna kompensacja wentylacyjna"

    # Normal pattern
    if rr_avg < 35 and ve_rr_ratio > 2.0:
        return "efficient", "Efficient Breathing – zrównoważony wzorzec oddechowy"

    # Default
    return "mixed", "Mixed Pattern – profil oddechowy do optymalizacji"


# --- Ported from Analiza Kolarska ---


def _resolve_ve_column(df: pd.DataFrame, ve_col: str) -> Optional[pd.Series]:
    """Resolve VE column from multiple possible names."""
    if ve_col in df.columns:
        return df[ve_col]
    for col in ("ve", "tymeventilation", "ventilation"):
        if col in df.columns:
            return df[col]
    return None


def _compute_segment_slopes(
    power: np.ndarray,
    ve_smooth: np.ndarray,
    segment_len: int = 20,
    step: int = 10,
) -> List[Tuple[float, float]]:
    """Compute VE slope in rolling segments for breakpoint detection."""
    n = len(ve_smooth)
    slopes: List[Tuple[float, float]] = []
    for i in range(0, n - segment_len, step):
        seg_power = power[i : i + segment_len]
        seg_ve = ve_smooth[i : i + segment_len]
        if len(seg_power) > 5 and np.unique(seg_power).size > 1:
            try:
                s, _, _, _, _ = stats.linregress(seg_power, seg_ve)
                slopes.append((power[i + segment_len // 2], s))
            except Exception as e:
                logger.debug(f"Linregress failed in segment: {e}")
    return slopes


def _find_slope_breakpoint(slopes: List[Tuple[float, float]]) -> Optional[float]:
    """Find the first power where VE slope exceeds 1.5× baseline."""
    if len(slopes) < 3:
        return None
    slopes_arr = np.array([s[1] for s in slopes])
    baseline = np.mean(slopes_arr[:3])
    for power_val, slope in slopes:
        if slope > baseline * 1.5:
            return float(power_val)
    return None


def classify_ventilatory_control(metrics: VentilationMetrics) -> Tuple[str, float, str]:
    """
    Classify overall ventilatory control status.

    Returns:
        (status, confidence, interpretation)
    """
    scores = {"controlled": 0.0, "compensatory": 0.0, "unstable": 0.0}

    ve_slope = metrics.ve_slope
    rr_avg = metrics.rr_avg
    rr_max = metrics.rr_max
    ve_rr = metrics.ve_rr_ratio
    pattern = metrics.breathing_pattern

    # --- VE SLOPE ANALYSIS ---
    if ve_slope < 0.25:
        scores["controlled"] += 2.0
    elif ve_slope < 0.4:
        scores["compensatory"] += 1.5
    else:
        scores["unstable"] += 2.0

    # --- RR ANALYSIS ---
    if rr_max < 45:
        scores["controlled"] += 1.5
    elif rr_max < 55:
        scores["compensatory"] += 1.0
    else:
        scores["unstable"] += 2.0

    # --- TIDAL VOLUME ANALYSIS ---
    if ve_rr > 2.5:
        scores["controlled"] += 2.0
    elif ve_rr > 1.5:
        scores["compensatory"] += 1.0
    else:
        scores["unstable"] += 1.5

    # --- PATTERN CONTRIBUTION ---
    if pattern == "efficient":
        scores["controlled"] += 1.0
    elif pattern == "shallow":
        scores["unstable"] += 1.5
    elif pattern == "hyperventilation":
        scores["compensatory"] += 1.5

    # Determine winner
    total = sum(scores.values()) or 1.0
    max_score = max(scores.values())
    status = max(scores, key=scores.get)
    confidence = max_score / total

    # Generate interpretation
    interpretation = _generate_vent_interpretation(status, metrics)

    return status, confidence, interpretation


def _generate_vent_interpretation(status: str, metrics: VentilationMetrics) -> str:
    """Generate physiology-oriented interpretation."""

    ve_slope = metrics.ve_slope
    rr_max = metrics.rr_max
    ve_rr = metrics.ve_rr_ratio
    bp = metrics.ve_breakpoint_watts

    if status == "controlled":
        base = "Efektywna kontrola wentylacji"
        detail = (
            f"Niski VE slope ({ve_slope:.2f} L/min/100W) i umiarkowane RR max ({rr_max:.0f}/min) "
            "wskazują na dobrą tolerancję CO₂ i ekonomię oddechową. "
        )
        if bp:
            detail += f"Punkt załamania VE przy {bp:.0f}W potwierdza wysoką sprawność metaboliczną."

    elif status == "compensatory":
        base = "Oddech kompensuje stres metaboliczny"
        detail = (
            f"Umiarkowany VE slope ({ve_slope:.2f}) i RR max {rr_max:.0f}/min sugerują, "
            "że układ oddechowy działa blisko granicy efektywności. "
        )
        if ve_rr < 2.0:
            detail += "Niska objętość oddechowa wymaga uwagi – może ograniczać saturację. "

    else:  # unstable
        base = "Dekompensacja wentylacyjna"
        detail = (
            f"Wysoki VE slope ({ve_slope:.2f}) i/lub RR max ({rr_max:.0f}) wskazują na "
            "niekontrolowany wzorzec oddechowy. Priorytet: trening tolerancji CO₂."
        )

    return f"{base}\n{detail}"


def generate_vent_recommendations(status: str, metrics: VentilationMetrics) -> List[Dict[str, str]]:
    """Generate training recommendations for ventilatory improvement."""

    if status == "controlled":
        return [
            {
                "type": "PERFORMANCE",
                "action": "Trening polaryzowany 80/20 — utrzymanie bieżącego planu z kontrolą VE/RR",
                "expected": "Stabilny VE/RR ratio, utrzymanie ekonomii oddechowej",
                "risk": "low",
            },
            {
                "type": "INTENSYWNOŚĆ",
                "action": "VO₂max interwały: 5×4min @ 106-120% FTP, świadomy oddech 2:3 (wdech:wydech)",
                "expected": "Poprawa VE peak o 5-10%, wzrost tolerancji hipoksji",
                "risk": "low",
            },
            {
                "type": "TRENINGOWA",
                "action": "Tempo z kontrolą oddechu: 2×20min @ 88-93% FTP, rytm oddechowy zsynchronizowany z kadencją",
                "expected": "Utrzymanie niskiego RR przy wysokiej mocy, VE/RR >2.5",
                "risk": "low",
            },
            {
                "type": "TECHNICZNA",
                "action": "Nasal breathing Z2: 30min oddychanie wyłącznie nosem @ 55-65% FTP — tolerancja CO₂",
                "expected": "Wzrost BOLT score o 3-5s, głębszy oddech na co dzień",
                "risk": "low",
            },
            {
                "type": "PERFORMANCE",
                "action": "Symulacja wyścigu: 60-90min z 3× blokami 10min @ 95% FTP — walidacja kontroli wentylacji",
                "expected": "Stabilny VE slope mimo kumulacji zmęczenia",
                "risk": "low",
            },
        ]

    if status == "compensatory":
        return [
            {
                "type": "TRENINGOWA",
                "action": "Tempo z kontrolą oddechu: 3×12min @ 85-90% FTP, wdech 3s / wydech 5s",
                "expected": "Wzrost VE/RR o 0.3-0.5, głębszy oddech",
                "risk": "low",
            },
            {
                "type": "TECHNICZNA",
                "action": "Box breathing: 4×(4s wdech / 4s pauza / 4s wydech / 4s pauza) × 5 cykli, 2×/dziennie",
                "expected": "Spadek spoczynkowego RR o 2-4/min, lepsza tolerancja CO₂",
                "risk": "low",
            },
            {
                "type": "TRENINGOWA",
                "action": "Z2 z oddechem przeponowym: 2h @ 60-70% FTP, focus na brzuchu nie klatce",
                "expected": "Poprawa TV (objętości oddechowej) o 10-15%",
                "risk": "low",
            },
            {
                "type": "INTENSYWNOŚĆ",
                "action": "Over-under z kontrolą RR: 3×(4min@95% + 2min@80% FTP), RR <45/min nawet w bloku wyżej",
                "expected": "Lepsza kontrola wentylacji w zmiennym obciążeniu",
                "risk": "low",
            },
            {
                "type": "DIAGNOSTYKA",
                "action": "Spirometria kontrolna: FEV1, FVC, MVV — wykluczenie ograniczeń mechanicznych",
                "expected": "Identyfikacja ewentualnych restrykcji/obstrukcji",
                "risk": "low",
            },
        ]

    # unstable
    return [
        {
            "type": "PILNA",
            "action": "Redukcja intensywności o 15-20%: max Z3, sesje do 75min — stabilizacja oddechu",
            "expected": "Spadek RR max poniżej 50/min w ciągu 2-3 tygodni",
            "risk": "medium",
        },
        {
            "type": "TECHNICZNA",
            "action": "Oddychanie przeponowe pod wysiłkiem: 3×10min @ Z2, ręce na brzuchu — nauka wzorca",
            "expected": "Wzrost TV (objętości oddechowej), spadek RR o 5-10/min",
            "risk": "low",
        },
        {
            "type": "TRENINGOWA",
            "action": "Z2 krótkie: 5×60min @ <65% FTP w tygodniu — baza aerobowa bez stresu wentylacyjnego",
            "expected": "Poprawa ekonomii oddechowej w 4-6 tygodni",
            "risk": "low",
        },
        {
            "type": "TECHNICZNA",
            "action": "Wydłużony wydech: trening 2×/dziennie 5min — wydech 2× dłuższy niż wdech (np. 3s/6s)",
            "expected": "Aktywacja układu przywspółczulnego, redukcja hiperwen­tylacji",
            "risk": "low",
        },
        {
            "type": "MEDYCZNA",
            "action": "Konsultacja pulmonologiczna: spirometria + test prowokacyjny metacholiną — wykluczenie EIB/astmy",
            "expected": "Diagnostyka i ewentualne leczenie ograniczeń wentylacyjnych",
            "risk": "high",
        },
    ]


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================


def analyze_ventilation(
    df: pd.DataFrame, ve_col: str = "ve", rr_col: str = "rr", power_col: str = "watts"
) -> VentilationMetrics:
    """
    Perform complete ventilation analysis.
    """
    metrics = VentilationMetrics()

    # Calculate core metrics
    metrics.ve_avg, metrics.ve_max, metrics.ve_slope = calculate_ve_metrics(df, ve_col, power_col)
    metrics.rr_avg, metrics.rr_max = calculate_rr_metrics(df, rr_col, power_col)
    metrics.ve_rr_ratio = calculate_ve_rr_ratio(metrics.ve_avg, metrics.rr_avg)

    # Find breakpoints
    metrics.ve_breakpoint_watts = find_ve_breakpoint(df, ve_col, power_col)

    # Classify pattern
    pattern, pattern_desc = classify_breathing_pattern(
        metrics.rr_avg, metrics.rr_max, metrics.ve_rr_ratio, metrics.ve_slope
    )
    metrics.breathing_pattern = pattern

    # Classify control
    status, confidence, interpretation = classify_ventilatory_control(metrics)
    metrics.control_status = status
    metrics.control_confidence = confidence
    metrics.interpretation = interpretation

    # Generate recommendations
    metrics.recommendations = generate_vent_recommendations(status, metrics)

    # Build profiles for charting
    ve_cols = ["ve", "tymeventilation", "ventilation"]
    for col in ve_cols:
        if col in df.columns:
            mask = (
                df[power_col] > 50 if power_col in df.columns else pd.Series(True, index=df.index)
            )
            for i in range(0, mask.sum(), 20):
                idx = df.index[mask][i] if i < mask.sum() else df.index[mask][-1]
                metrics.ve_profile.append(
                    {
                        "power": float(df.loc[idx, power_col]) if power_col in df.columns else i,
                        "ve": float(df.loc[idx, col]),
                    }
                )
            break

    return metrics


def format_vent_metrics_for_report(metrics: VentilationMetrics) -> Dict[str, Any]:
    """Format metrics for inclusion in JSON/PDF reports."""
    return {
        "ve_avg": round(metrics.ve_avg, 1),
        "ve_max": round(metrics.ve_max, 1),
        "rr_avg": round(metrics.rr_avg, 1),
        "rr_max": round(metrics.rr_max, 1),
        "ve_rr_ratio": round(metrics.ve_rr_ratio, 2),
        "ve_slope": round(metrics.ve_slope, 3),
        "ve_breakpoint_watts": round(metrics.ve_breakpoint_watts, 0)
        if metrics.ve_breakpoint_watts
        else None,
        "breathing_pattern": metrics.breathing_pattern,
        "control_status": metrics.control_status,
        "control_confidence": round(metrics.control_confidence, 2),
        "interpretation": metrics.interpretation,
        "recommendations": metrics.recommendations,
        "ve_profile": metrics.ve_profile[:15],
    }


__all__ = [
    "VentilationMetrics",
    "analyze_ventilation",
    "calculate_ve_metrics",
    "calculate_rr_metrics",
    "classify_breathing_pattern",
    "classify_ventilatory_control",
    "format_vent_metrics_for_report",
]
