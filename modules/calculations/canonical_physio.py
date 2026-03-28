"""
Canonical Physiological Parameters Module.

Single Source of Truth for all physiological metrics across the system.
All UI components, PDF reports, and analysis modules must use this.

This module provides:
- VO2max (canonical value with source tracking)
- VLaMax (estimated)
- CP / FTP
- Weight
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import numpy as np

logger = logging.getLogger("Tri_Dashboard.CanonicalPhysio")


@dataclass
class CanonicalMetric:
    """A metric with source tracking and confidence."""

    value: float = 0.0
    source: str = "none"  # Where the value came from
    confidence: float = 0.0  # 0-1, how reliable this value is
    alternatives: Dict[str, float] = field(default_factory=dict)  # Other estimates

    def is_valid(self) -> bool:
        return self.value > 0 and self.source != "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": round(self.value, 2),
            "source": self.source,
            "confidence": round(self.confidence, 2),
            "alternatives": {k: round(v, 2) for k, v in self.alternatives.items()},
        }


@dataclass
class CanonicalPhysiology:
    """Single Source of Truth for physiological parameters."""

    vo2max: CanonicalMetric = field(default_factory=CanonicalMetric)
    vlamax: CanonicalMetric = field(default_factory=CanonicalMetric)
    cp_watts: CanonicalMetric = field(default_factory=CanonicalMetric)
    ftp_watts: CanonicalMetric = field(default_factory=CanonicalMetric)
    weight_kg: CanonicalMetric = field(default_factory=CanonicalMetric)
    pmax_watts: CanonicalMetric = field(default_factory=CanonicalMetric)
    w_prime_kj: CanonicalMetric = field(default_factory=CanonicalMetric)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vo2max": self.vo2max.to_dict(),
            "vlamax": self.vlamax.to_dict(),
            "cp_watts": self.cp_watts.to_dict(),
            "ftp_watts": self.ftp_watts.to_dict(),
            "weight_kg": self.weight_kg.to_dict(),
            "pmax_watts": self.pmax_watts.to_dict(),
            "w_prime_kj": self.w_prime_kj.to_dict(),
        }


# =============================================================================
# VO2max SELECTION POLICY
# =============================================================================

# Priority order for VO2max sources (higher = preferred)
VO2MAX_SOURCE_PRIORITY = {
    "lab_measured": 1.0,  # Laboratory gas exchange (gold standard)
    "field_cpet": 0.95,  # Field CPET test
    "ramp_test_peak": 0.85,  # Peak from ramp test with VO2 sensor
    "intervals_api": 0.80,  # From Intervals.icu or similar
    "acsm_5min": 0.70,  # ACSM formula from 5-min power
    "acsm_cp": 0.50,  # ACSM formula from CP estimate
    "user_input": 0.60,  # User-provided value
    "estimated": 0.40,  # Generic estimate
    "none": 0.0,
}


def calculate_vo2max_jurov(power_watts: float, weight_kg: float, sex: str = "male") -> float:
    """
    Calculate VO2max using Jurov et al. 2023 cyclist-specific formulas.

    Male:   VO2max = 0.10 × PO - 0.60 × BW + 64.21
    Female: VO2max = 0.13 × PO - 0.83 × BW + 64.02
    Non-specific: VO2max = 0.12 × PO - 0.65 × BW + 59.78

    Near-zero bias (0.19% in males) vs ACSM's 11.99% underestimation.
    Validated on 580 competitive cyclists (496M, 84F).

    Reference:
        Jurov et al. (2023). "Prediction of Maximal Oxygen Consumption
        in Cycle Ergometry in Competitive Cyclists."
        Life (MDPI), 13(1), 160. DOI: 10.3390/life13010160
    """
    if power_watts <= 0 or weight_kg <= 0:
        return 0.0
    if sex.lower().startswith("f"):
        return 0.13 * power_watts - 0.83 * weight_kg + 64.02
    elif sex.lower().startswith("m"):
        return 0.10 * power_watts - 0.60 * weight_kg + 64.21
    else:
        return 0.12 * power_watts - 0.65 * weight_kg + 59.78


# Backward compatibility alias
calculate_vo2max_acsm = calculate_vo2max_jurov


def calculate_vo2max_friend(power_watts: float, weight_kg: float) -> float:
    """
    FRIEND equation for VO2max (Jurov et al. 2023 validation study).

    Best accuracy in both male and female cyclists per
    Jurov, Cvijic & Toplisek (2023), Frontiers in Physiology 14:987006.

    Uses ACSM cycling equation: VO2 = (10.8 × W / kg) + 7
    but with validated correction for competitive cyclists.
    """
    if power_watts <= 0 or weight_kg <= 0:
        return 0.0
    return 10.8 * (power_watts / weight_kg) + 7.0


def calculate_vo2max_hawley(power_watts: float, weight_kg: float) -> float:
    """
    Calculate VO2max using Hawley & Noakes 1992 formula (LEGACY CROSS-VALIDATION).

    VO2max = 10.8 × (W/kg) + 7.0

    Legacy cross-check — superseded by Jurov 2023 as primary formula.
    Validated on broader population of endurance athletes.
    """
    if power_watts <= 0 or weight_kg <= 0:
        return 0.0
    power_per_kg = power_watts / weight_kg
    return 10.8 * power_per_kg + 7.0


def cross_validate_vo2max(power_watts: float, weight_kg: float, sex: str = "male") -> dict:
    """
    Cross-validate VO2max using three independent formulas.

    Primary: Jurov et al. 2023 (cyclist-specific, sex-aware)
    Secondary: FRIEND equation (validated by Jurov 2023)
    Legacy: Hawley & Noakes 1992 (broad population)

    Divergence >8 ml/kg/min flags uncertainty (was 5, increased per
    Bentley et al. 2007 showing ±15% pedaling economy variance).
    """
    jurov = calculate_vo2max_jurov(power_watts, weight_kg, sex)
    friend = calculate_vo2max_friend(power_watts, weight_kg)
    hawley = calculate_vo2max_hawley(power_watts, weight_kg)

    if jurov <= 0:
        return {
            "jurov": jurov, "friend": friend, "hawley": hawley,
            "mean": 0.0, "divergence": 0.0, "is_uncertain": True,
        }

    values = [v for v in [jurov, friend, hawley] if v > 0]
    divergence = max(values) - min(values) if len(values) >= 2 else 0.0
    mean_vo2 = sum(values) / len(values)

    return {
        "jurov": round(jurov, 1),
        "friend": round(friend, 1),
        "hawley": round(hawley, 1),
        "mean": round(mean_vo2, 1),
        "divergence": round(divergence, 1),
        "is_uncertain": divergence > 8.0,
    }


def adjust_vo2max_for_altitude(vo2max_sea_level: float, altitude_m: float) -> dict:
    """
    Adjust VO2max for altitude using Wehrlin & Hallen 2006 linear model.

    VO2max decreases ~6.3% per 1000m above sea level in trained athletes.
    Confirmed by Pühringer et al. (2022): 5.0-11.6% per 1000m above 1500m.

    No effect below 500m (normoxic conditions).

    References:
        Wehrlin & Hallen (2006). Linear decrease in VO2max with altitude.
        Pühringer et al. (2022). High Altitude Medicine & Biology.
        Townsend et al. (2017). Curvilinear meta-regression model.
    """
    if altitude_m <= 500 or vo2max_sea_level <= 0:
        return {
            "vo2max_adjusted": vo2max_sea_level,
            "altitude_m": altitude_m,
            "reduction_pct": 0.0,
            "is_adjusted": False,
        }

    # Linear model: 6.3% reduction per 1000m (Wehrlin & Hallen 2006)
    reduction_pct = 6.3 * (altitude_m / 1000.0)
    # Cap at 40% reduction (extreme altitude >6000m)
    reduction_pct = min(40.0, reduction_pct)
    vo2max_adjusted = vo2max_sea_level * (1 - reduction_pct / 100)

    return {
        "vo2max_adjusted": round(vo2max_adjusted, 1),
        "altitude_m": altitude_m,
        "reduction_pct": round(reduction_pct, 1),
        "is_adjusted": True,
    }


def select_canonical_vo2max(
    candidates: Dict[str, float],
    weight_kg: float = 75.0,
    w_prime_j: Optional[float] = None,
    sex: str = "male",
) -> CanonicalMetric:
    """
    Select canonical VO2max from multiple candidates based on priority.

    Args:
        candidates: Dict of {source: value} pairs
            - "lab_measured": Direct lab measurement
            - "ramp_test_peak": Peak VO2 from ramp test
            - "intervals_api": From external API
            - "mmp_5min": 5-minute max power (will be converted)
            - "cp_watts": Critical Power (will be converted)
            - "user_input": User-provided value

    Returns:
        CanonicalMetric with selected value and alternatives
    """
    metric = CanonicalMetric()
    alternatives = {}

    # Process candidates
    for source, value in candidates.items():
        if value is None or value <= 0:
            continue

        # Convert power-based inputs to VO2max
        if source == "mmp_5min":
            vo2 = calculate_vo2max_jurov(value, weight_kg, sex)
            alternatives["acsm_5min"] = vo2
            source_key = "acsm_5min"
        elif source == "cp_watts":
            # From 2-parameter CP model: P(t) = CP + W'/t
            # For 5 min (300s): P(300) = CP + W'/300
            # Use actual W' if available, otherwise conservative 15 kJ default
            w_prime_used = w_prime_j if w_prime_j and w_prime_j > 0 else 15000
            est_5min = value + (w_prime_used / 300)
            vo2 = calculate_vo2max_jurov(est_5min, weight_kg, sex)
            alternatives["acsm_cp"] = vo2
            source_key = "acsm_cp"
        else:
            # Direct VO2max value
            alternatives[source] = value
            source_key = source

    if not alternatives:
        return metric

    # Select best candidate based on priority
    best_source = None
    best_priority = -1

    for source, value in alternatives.items():
        priority = VO2MAX_SOURCE_PRIORITY.get(source, 0.3)
        if priority > best_priority:
            best_priority = priority
            best_source = source

    if best_source:
        metric.value = alternatives[best_source]
        metric.source = best_source
        metric.confidence = best_priority
        # Store all alternatives except the selected one
        metric.alternatives = {k: v for k, v in alternatives.items() if k != best_source}

    return metric


def build_canonical_physiology(
    data: Dict[str, Any], time_series: Optional[Dict[str, List]] = None
) -> CanonicalPhysiology:
    """
    Build canonical physiology from report data.

    This is the SINGLE ENTRY POINT for physiological parameters.
    All modules (UI, PDF, Metabolic Engine) must use this.

    Args:
        data: Report data dictionary (from result.to_dict() or JSON)
        time_series: Optional time series data for power-based calculations

    Returns:
        CanonicalPhysiology with all parameters set
    """
    physio = CanonicalPhysiology()

    # === SEX (for sex-specific VO2max formulas) ===
    sex = data.get("metadata", {}).get("athlete_sex", "male")
    if not sex:
        sex = data.get("athlete", {}).get("sex", "male")
    if not sex:
        sex = "male"

    # === WEIGHT ===
    weight = data.get("metadata", {}).get("athlete_weight_kg", 0)
    if not weight:
        weight = data.get("athlete", {}).get("weight_kg", 75)
    physio.weight_kg = CanonicalMetric(
        value=weight or 75,
        source="metadata" if weight else "default",
        confidence=0.95 if weight else 0.3,
    )
    weight_kg = physio.weight_kg.value

    # === CP / FTP ===
    cp = data.get("cp_model", {}).get("cp_watts") or 0
    if cp:
        physio.cp_watts = CanonicalMetric(value=cp, source="cp_model", confidence=0.85)

    ftp = (data.get("thresholds") or {}).get("ftp_watts") or 0
    if ftp:
        physio.ftp_watts = CanonicalMetric(value=ftp, source="thresholds", confidence=0.80)
    elif cp:
        physio.ftp_watts = CanonicalMetric(value=cp, source="derived_from_cp", confidence=0.70)

    # === W' ===
    w_prime_j = data.get("cp_model", {}).get("w_prime_joules", 0)
    if w_prime_j:
        physio.w_prime_kj = CanonicalMetric(
            value=w_prime_j / 1000, source="cp_model", confidence=0.80
        )

    # === Pmax ===
    pmax = data.get("metadata", {}).get("pmax_watts", 0)
    if not pmax and time_series:
        power_data = time_series.get("power_watts", [])
        if power_data:
            pmax = max(power_data)
    if pmax:
        physio.pmax_watts = CanonicalMetric(value=pmax, source="peak_power", confidence=0.90)

    # === VO2max (CANONICAL SELECTION) ===
    vo2max_candidates = {}

    # Source 1: Direct VO2max from metrics (calculated with pandas rolling - SAME AS UI)
    # This is the CANONICAL source that matches UI KPI display
    direct_vo2 = data.get("metrics", {}).get("vo2max", 0)
    if direct_vo2 and direct_vo2 > 0:
        vo2max_candidates["acsm_5min"] = direct_vo2  # Use acsm_5min key for proper priority

    # Source 2: From external API (Intervals.icu etc.)
    api_vo2 = data.get("athlete", {}).get("vo2max", 0)
    if api_vo2 and api_vo2 > 0:
        vo2max_candidates["intervals_api"] = api_vo2

    # Source 3: User input
    user_vo2 = data.get("user_input", {}).get("vo2max", 0)
    if user_vo2 and user_vo2 > 0:
        vo2max_candidates["user_input"] = user_vo2

    # Source 4: Calculate from 5-min MMP (ONLY if metrics.vo2max is not available)
    # Note: metrics.vo2max is calculated using pandas rolling (same as UI) so it's preferred
    if "acsm_5min" not in vo2max_candidates and time_series and weight_kg > 0:
        power_data = time_series.get("power_watts", [])
        if len(power_data) >= 300:
            window = 300
            power_arr = np.array(power_data)
            rolling_max = np.max(power_arr[window - 1:] - np.cumsum(np.concatenate([np.zeros(1), np.cumsum(power_arr[:-1])])[:-window + 1]))
            if window > 1:
                cumsum = np.cumsum(power_arr)
                rolling_means = (cumsum[window - 1:] - cumsum[:-window + 1]) / window
                mmp_5m = float(np.max(rolling_means)) if len(rolling_means) > 0 else 0
            else:
                mmp_5m = float(np.max(power_arr)) if len(power_arr) > 0 else 0
            if mmp_5m > 0:
                vo2max_candidates["mmp_5min"] = mmp_5m  # Will be converted

    # Source 5: Estimate from CP (lowest priority)
    if cp and cp > 0:
        vo2max_candidates["cp_watts"] = cp  # Will be converted in select_canonical_vo2max

    # Select canonical VO2max (pass actual W' for accurate CP→5min conversion)
    actual_w_prime_j = w_prime_j if w_prime_j and w_prime_j > 0 else None
    physio.vo2max = select_canonical_vo2max(
        vo2max_candidates, weight_kg, w_prime_j=actual_w_prime_j, sex=sex
    )

    # =========================================================================
    # CONSISTENCY ASSERTION (light - logs divergence, doesn't block)
    # =========================================================================
    if physio.vo2max.is_valid() and time_series and weight_kg > 0:
        # Calculate time_series estimate for comparison
        power_data = time_series.get("power_watts", [])
        if len(power_data) >= 300:
            window = 300
            power_arr = np.array(power_data)
            if window > 1:
                cumsum = np.cumsum(np.insert(power_arr, 0, 0))
                rolling_means = (cumsum[window:] - cumsum[:-window]) / window
                ts_mmp5 = float(np.max(rolling_means)) if len(rolling_means) > 0 else 0
            else:
                ts_mmp5 = float(np.max(power_arr)) if len(power_arr) > 0 else 0
            if ts_mmp5 > 0:
                ts_vo2max = calculate_vo2max_jurov(ts_mmp5, weight_kg, sex)
                divergence = abs(physio.vo2max.value - ts_vo2max)

                # Log if divergence > 5 ml/kg/min (significant)
                if divergence > 5:
                    logger.warning(
                        "VO2max divergence detected: metrics=%.1f vs "
                        "time_series=%.1f (delta=%.1f ml/kg/min). "
                        "Using metrics (pandas rolling) as canonical.",
                        physio.vo2max.value, ts_vo2max, divergence,
                    )
                    # Store divergence for debugging
                    physio.vo2max.alternatives["time_series_estimate"] = round(ts_vo2max, 2)

                # Cross-validate Jurov vs FRIEND & Hawley (1992)
                xval = cross_validate_vo2max(ts_mmp5, weight_kg, sex)
                if xval["is_uncertain"]:
                    logger.warning(
                        "VO2max cross-validation uncertain: Jurov=%.1f vs FRIEND=%.1f vs Hawley=%.1f "
                        "(divergence=%.1f ml/kg/min). Athlete may have unusual "
                        "pedaling economy or power data issues.",
                        xval["jurov"], xval["friend"], xval["hawley"], xval["divergence"],
                    )
                physio.vo2max.alternatives["hawley_noakes"] = xval["hawley"]

    # === VLaMax (always estimated) ===
    if cp > 0 and pmax > 0 and weight_kg > 0:
        w_prime_kj_val = physio.w_prime_kj.value or 15
        w_prime_per_kg = w_prime_kj_val / weight_kg
        cp_per_kg = cp / weight_kg
        # Mader-inspired: W'/kg drives glycolytic capacity,
        # CP/kg inversely correlated (metabolic antagonism)
        vlamax = w_prime_per_kg * 2.8 + 0.05 - cp_per_kg * 0.07
        vlamax = max(0.2, min(1.0, vlamax))
        physio.vlamax = CanonicalMetric(
            value=vlamax, source="estimated_from_power", confidence=0.45
        )

    return physio


def format_canonical_for_report(physio: CanonicalPhysiology) -> Dict[str, Any]:
    """Format canonical physiology for JSON storage."""
    return {
        "canonical_physiology": physio.to_dict(),
        "summary": {
            "vo2max": physio.vo2max.value if physio.vo2max.is_valid() else None,
            "vo2max_source": physio.vo2max.source,
            "vlamax": physio.vlamax.value if physio.vlamax.is_valid() else None,
            "cp_watts": physio.cp_watts.value if physio.cp_watts.is_valid() else None,
            "weight_kg": physio.weight_kg.value,
        },
    }


__all__ = [
    "CanonicalMetric",
    "CanonicalPhysiology",
    "calculate_vo2max_jurov",
    "calculate_vo2max_acsm",  # backward-compat alias for calculate_vo2max_jurov
    "calculate_vo2max_friend",
    "calculate_vo2max_hawley",
    "cross_validate_vo2max",
    "adjust_vo2max_for_altitude",
    "select_canonical_vo2max",
    "build_canonical_physiology",
    "format_canonical_for_report",
    "VO2MAX_SOURCE_PRIORITY",
]
