"""
Cardiac Drift Analysis Module.

Analyzes Efficiency Factor (EF) decline relative to core temperature,
time, and peripheral muscle fatigue. Differentiates between cardiac,
thermal, and metabolic drift origins.

Key metrics:
- EF Start/End and ΔEF%
- EF slope vs Core Temperature [W/bpm / °C]
- Temperature at 10% EF drop
- Drift classification: minimal (<5%), moderate (5-10%), high (>10%)
- Drift type: cardiac, thermal, metabolic
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import logging

logger = logging.getLogger("Tri_Dashboard.CardiacDrift")


@dataclass
class CardiacDriftProfile:
    """Cardiac drift analysis results."""
    # Core EF metrics
    ef_start: float = 0.0              # EF at test start [W/bpm]
    ef_end: float = 0.0                # EF at test end [W/bpm]
    delta_ef_abs: float = 0.0          # Absolute change [W/bpm]
    delta_ef_pct: float = 0.0          # Percentage change [%]
    
    # EF vs Temperature relationship
    ef_vs_temp_slope: float = 0.0      # EF change per °C [W/bpm/°C]
    temp_at_10pct_drop: Optional[float] = None  # Core temp where EF drops 10%
    
    # Key signals for correlation
    hsi_peak: float = 0.0              # Peak Heat Strain Index
    smo2_drift_pct: float = 0.0        # SmO2 drift percentage
    hr_drift_pct: float = 0.0          # HR drift percentage
    
    # Classification
    drift_classification: str = "unknown"  # minimal, moderate, high
    drift_type: str = "unknown"            # cardiac, thermal, metabolic, mixed
    classification_color: str = "#808080"
    
    # Interpretation
    mechanism_description: str = ""
    key_signals_summary: str = ""
    training_implications: List[str] = field(default_factory=list)
    
    # Quality
    data_points: int = 0
    confidence: float = 0.0


def calculate_efficiency_factor(power: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """
    Calculate Efficiency Factor: EF = Power / HR [W/bpm].
    
    Higher EF = more watts per heartbeat = better aerobic efficiency.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ef = np.where(hr > 50, power / hr, np.nan)
    return ef


def analyze_cardiac_drift(
    power: np.ndarray,
    hr: np.ndarray,
    time_seconds: np.ndarray,
    core_temp: Optional[np.ndarray] = None,
    smo2: Optional[np.ndarray] = None,
    hsi: Optional[np.ndarray] = None
) -> CardiacDriftProfile:
    """
    from scipy import stats

    Analyze cardiac drift from power, HR, and optional thermal/muscle data.
    
    Args:
        power: Power values [W]
        hr: Heart rate values [bpm]
        time_seconds: Time array [seconds]
        core_temp: Optional core temperature [°C]
        smo2: Optional SmO2 values [%]
        hsi: Optional Heat Strain Index
        
    Returns:
        CardiacDriftProfile with comprehensive drift analysis
    """
    profile = CardiacDriftProfile()
    
    # Filter valid data
    mask = (~np.isnan(power)) & (~np.isnan(hr)) & (hr > 50) & (power > 50)
    if np.sum(mask) < 60:
        logger.warning("Insufficient data for drift analysis")
        profile.mechanism_description = "Niewystarczające dane do analizy dryfu."
        return profile
    
    power_valid = power[mask]
    hr_valid = hr[mask]
    time_valid = time_seconds[mask]
    
    profile.data_points = len(power_valid)
    
    # === CALCULATE EF ===
    ef = calculate_efficiency_factor(power_valid, hr_valid)
    ef_valid = ef[~np.isnan(ef)]
    
    if len(ef_valid) < 30:
        profile.mechanism_description = "Za mało danych EF do analizy."
        return profile
    
    # === ADAPTIVE WINDOW for EF comparison ===
    # Short sessions (<10 min): use 25% of data
    # Long sessions (>30 min / steady-state): use 25% of data (not fixed 60s!)
    # Fixed 60s window is misleading for long steady-state efforts because
    # it compares warmup (low HR) vs end, inflating drift artificially.
    n_points = len(ef_valid)
    start_window = max(30, n_points // 4)
    end_window = max(30, n_points // 4)
    
    profile.ef_start = float(np.nanmean(ef_valid[:start_window]))
    profile.ef_end = float(np.nanmean(ef_valid[-end_window:]))
    profile.delta_ef_abs = profile.ef_end - profile.ef_start
    
    if profile.ef_start > 0:
        profile.delta_ef_pct = ((profile.ef_end - profile.ef_start) / profile.ef_start) * 100
    
    # === EF vs TEMPERATURE SLOPE ===
    if core_temp is not None:
        temp_valid = core_temp[mask]
        temp_mask = (~np.isnan(temp_valid)) & (temp_valid > 35) & (temp_valid < 42)
        
        if np.sum(temp_mask) > 30:
            ef_for_temp = ef[temp_mask]
            temp_for_ef = temp_valid[temp_mask]
            
            # Only use valid EF values
            valid_mask = ~np.isnan(ef_for_temp)
            if np.sum(valid_mask) > 20 and np.unique(temp_for_ef[valid_mask]).size > 1:
                try:
                    slope, intercept, r, p, se = stats.linregress(
                        temp_for_ef[valid_mask], 
                        ef_for_temp[valid_mask]
                    )
                    profile.ef_vs_temp_slope = float(slope)
                    
                    # Calculate temp at 10% EF drop
                    if slope < 0 and profile.ef_start > 0:
                        target_ef = profile.ef_start * 0.9
                        profile.temp_at_10pct_drop = float((target_ef - intercept) / slope)
                except Exception as e:
                    logger.debug(f"EF vs temp regression failed: {e}")
    
    # === KEY SIGNALS ===
    if hsi is not None:
        hsi_valid = hsi[~np.isnan(hsi)]
        if len(hsi_valid) > 0:
            profile.hsi_peak = float(np.nanmax(hsi_valid))
    
    if smo2 is not None:
        smo2_valid = smo2[mask]
        smo2_clean = smo2_valid[~np.isnan(smo2_valid)]
        if len(smo2_clean) > 30:
            smo2_start = np.nanmean(smo2_clean[:start_window])
            smo2_end = np.nanmean(smo2_clean[-end_window:])
            if smo2_start > 0:
                profile.smo2_drift_pct = ((smo2_end - smo2_start) / smo2_start) * 100
    
    # HR drift
    hr_start = np.nanmean(hr_valid[:start_window])
    hr_end = np.nanmean(hr_valid[-end_window:])
    if hr_start > 0:
        profile.hr_drift_pct = ((hr_end - hr_start) / hr_start) * 100
    
    # === CLASSIFICATION ===
    abs_delta = abs(profile.delta_ef_pct)
    if abs_delta < 5:
        profile.drift_classification = "minimal"
        profile.classification_color = "#27AE60"  # Green
    elif abs_delta < 10:
        profile.drift_classification = "moderate"
        profile.classification_color = "#F39C12"  # Orange
    else:
        profile.drift_classification = "high"
        profile.classification_color = "#E74C3C"  # Red
    
    # === DRIFT TYPE ===
    profile.drift_type = _classify_drift_type(profile, core_temp)
    
    # === INTERPRETATION ===
    profile.mechanism_description = _generate_drift_mechanism(profile)
    profile.key_signals_summary = _generate_key_signals_summary(profile)
    profile.training_implications = _generate_training_implications(profile)
    
    # Confidence
    profile.confidence = min(1.0, profile.data_points / 500)
    
    return profile


def _classify_drift_type(profile: CardiacDriftProfile, core_temp: Optional[np.ndarray]) -> str:
    """
    Differentiate between drift types based on signal correlations.
    
    - Cardiac: HR rises, temp stable, SmO2 stable → CV fatigue
    - Thermal: temp rises, HR follows → thermoregulation stress
    - Metabolic: SmO2 drops significantly → peripheral muscle fatigue
    """
    has_temp_data = core_temp is not None and np.sum(~np.isnan(core_temp)) > 30
    
    # Thresholds for signal significance
    significant_hr_drift = abs(profile.hr_drift_pct) > 8
    significant_smo2_drop = profile.smo2_drift_pct < -8
    significant_temp_slope = abs(profile.ef_vs_temp_slope) > 0.1
    
    if significant_smo2_drop and not significant_hr_drift:
        return "metabolic"
    elif has_temp_data and significant_temp_slope:
        return "thermal"
    elif significant_hr_drift and not significant_smo2_drop:
        return "cardiac"
    else:
        return "mixed"


def _generate_drift_mechanism(profile: CardiacDriftProfile) -> str:
    """Generate physiological mechanism description."""
    drift_pct = abs(profile.delta_ef_pct)
    
    if profile.drift_classification == "minimal":
        base = f"MINIMALNY DRYF ({drift_pct:.1f}%)"
        detail = (
            "Efficiency Factor pozostaje stabilny przez cały test. "
            "Układ krążenia efektywnie dostarcza tlen bez przeciążenia. "
            "To wskazuje na dobrą ekonomię wysiłkową i adaptację treningową."
        )
    elif profile.drift_classification == "moderate":
        base = f"UMIARKOWANY DRYF ({drift_pct:.1f}%)"
        if profile.drift_type == "thermal":
            detail = (
                "Spadek EF jest skorelowany ze wzrostem temperatury głębokiej. "
                "Serce pompuje dodatkową krew do skóry (chłodzenie), co zmniejsza "
                "ilość tlenu dostarczanego do mięśni. Rozważ strategie chłodzenia."
            )
        elif profile.drift_type == "metabolic":
            detail = (
                "SmO2 spada znacząco, wskazując na lokalne zmęczenie mięśniowe. "
                "Perfuzja obwodowa jest ograniczona niezależnie od HR. "
                "Priorytet: praca nad wytrzymałością mięśniową."
            )
        else:
            detail = (
                "Dryf o złożonej etiologii (mieszany). Obserwujemy wpływ "
                "zarówno zmęczenia centralnego jak i obwodowego."
            )
    else:  # high
        base = f"WYSOKI DRYF ({drift_pct:.1f}%)"
        if profile.drift_type == "thermal":
            detail = (
                "KRYTYCZNY: Temperatura głęboka powoduje masywną redystrybucję krwi. "
                "EF spada o {:.2f} W/bpm na każdy °C wzrostu temperatury. "
                "Ryzyko przegrzania i dekompensacji krążeniowej jest wysokie. "
                "PRIORYTET: adaptacja cieplna i strategie chłodzenia."
            ).format(abs(profile.ef_vs_temp_slope))
        elif profile.drift_type == "metabolic":
            detail = (
                "Znaczny dryf obwodowy. SmO2 spada o {:.1f}%, wskazując na "
                "krytyczne ograniczenie perfuzji mięśniowej. "
                "Układ krążenia nie nadąża z dostawą tlenu do pracujących mięśni."
            ).format(abs(profile.smo2_drift_pct))
        else:
            detail = (
                "Dryf kardiologiczny wskazuje na przeciążenie układu krążenia. "
                "HR rośnie o {:.1f}% przy stałej mocy. "
                "Wymagana konsultacja z fizjologiem/kardiologiem."
            ).format(profile.hr_drift_pct)
    
    return f"{base}\n\n{detail}"


def _generate_key_signals_summary(profile: CardiacDriftProfile) -> str:
    """Generate summary of key signals for the drift analysis."""
    signals = []
    
    # EF trend
    if profile.delta_ef_pct < -5:
        signals.append(f"EF ↓ {abs(profile.delta_ef_pct):.1f}%")
    elif profile.delta_ef_pct > 5:
        signals.append(f"EF ↑ {profile.delta_ef_pct:.1f}%")
    else:
        signals.append("EF stabilny")
    
    # HR drift
    if profile.hr_drift_pct > 5:
        signals.append(f"HR ↑ {profile.hr_drift_pct:.1f}%")
    
    # SmO2 drift
    if profile.smo2_drift_pct < -5:
        signals.append(f"SmO2 ↓ {abs(profile.smo2_drift_pct):.1f}%")
    
    # HSI
    if profile.hsi_peak > 7:
        signals.append(f"HSI peak: {profile.hsi_peak:.1f}")
    
    return " | ".join(signals)


def _generate_training_implications(profile: CardiacDriftProfile) -> List[str]:
    """Generate specific training recommendations based on drift analysis.

    Always returns ~5 concrete training units with intensity, duration,
    cadence and recovery context.
    """
    if profile.drift_classification == "minimal":
        return [
            "INTERWAŁY VO₂max: 5×4min @ 106-120% FTP, przerwa 3min, kadencja 90-100rpm",
            "TEMPO DŁUGIE: 2×20min @ 88-93% FTP, przerwa 5min, EF powinno być stabilne",
            "SWEET SPOT: 3×15min @ 88-94% FTP, przerwa 3min — budowa wytrzymałości progowej",
            "Z2 DŁUGIE: 3-4h @ 55-75% FTP, kadencja swobodna — utrzymanie bazy aerobowej",
            "SPRINT POWTARZALNY: 8×30s all-out / 4.5min recovery — praca nad W' i rekrutacją",
        ]

    if profile.drift_classification == "moderate":
        if profile.drift_type == "thermal":
            return [
                "HEAT ADAPT Z2: 60-90min @ 60-70% FTP w cieple (>25°C) lub w ubraniu izolacyjnym",
                "TEMPO KRÓTKIE: 4×10min @ 85-90% FTP, przerwa 5min — skrócone vs standardu",
                "KADENCJA CHŁODZĄCA: 3×12min @ Z3, kadencja 95-105rpm — mniejszy koszt termiczny",
                "Z2 Z CHŁODZENIEM: 2h @ 60-70% FTP + woda na głowę co 15min — adaptacja cieplna",
                "INTERWAŁY WIECZORNE: 5×5min @ 95-100% FTP, temp. <20°C — obejście stresu termicznego",
            ]
        if profile.drift_type == "metabolic":
            return [
                "TEMPO PROGOWE: 2×20min @ 88-93% FTP, przerwa 5min — budowa progu metabolicznego",
                "STRENGTH ENDURANCE: 4×8min @ Z3, 50-60rpm — poprawa siły mięśniowej",
                "OVER-UNDER: 3×(4min@95% + 2min@85% FTP) × 3 — tolerancja zmian pH",
                "Z2 DŁUGIE Z KADENCJĄ: 3h @ 65% FTP, co 30min blok 5min @ 100rpm — perfuzja",
                "FARTLEK Z2/Z3: 90min z 6× wstawkami 3min @ Z3 — stymulacja bez przeciążenia",
            ]
        return [
            "TEMPO UMIARKOWANE: 3×12min @ 85-90% FTP, przerwa 4min — redukcja obciążenia vs standardu",
            "Z2 ROZBUDOWANE: 2.5-3h @ 60-70% FTP — budowa bazy bez dryfu",
            "INTERWAŁY SUBMAKS: 5×5min @ 90-95% FTP, przerwa 3min — kontrolowany bodziec",
            "KADENCJA ROTACYJNA: 3×10min @ Z3 (80→90→100rpm co 3min) — poprawa pedalowania",
            "RECOVERY AKTYWNE: 60-90min @ <55% FTP + stretching — regeneracja układu krążenia",
        ]

    # high drift
    if profile.drift_type == "thermal":
        return [
            "PILNE — HEAT ACCLIMATION: 10-14 dni, 60min @ Z2 w warunkach >28°C lub sauna post-trening",
            "INTERWAŁY KRÓTKIE: 6×3min @ 95-105% FTP, przerwa 4min — bodźce bez kumulacji ciepła",
            "PRE-COOLING SESSION: 15min zimny napój (500ml) + kamizelka, potem 4×5min @ Z3",
            "Z2 NOCNE/PORANNE: 2h @ 60-65% FTP, temp. <18°C — obejście stresu termicznego",
            "SIŁOWE W KLIMAT.: 4×6min @ 50-60rpm, Z3 w klimatyzowanym pomieszczeniu — siła bez ciepła",
        ]
    if profile.drift_type == "metabolic":
        return [
            "REDUKCJA TSS: 2 tygodnie @ 70% planowanej objętości — rozładowanie metaboliczne",
            "Z2 REGENERACYJNE: 3×90min @ 55-65% FTP — odbudowa mitochondriów bez obciążenia",
            "TEMPO OSTROŻNE: 2×12min @ 82-88% FTP, przerwa 6min — łagodna stymulacja progu",
            "SUPLEMENTACJA + DIAGNOSTYKA: badanie żelaza/ferrytyny/B12, korekcja niedoborów",
            "MICRO-INTERWAŁY: 10×1min @ 100% FTP / 2min recovery — bodziec przy niskim koszcie",
        ]
    return [
        "KONSULTACJA LEKARSKA: EKG wysiłkowe + echo serca przed kontynuacją intensywnego treningu",
        "Z2 WYŁĄCZNIE: max 90min @ <65% FTP, HR <75% HRmax — bezpieczna baza",
        "KONTROLOWANY TEST: 20min @ Z3 z monitoringiem EF — ocena postępu co 2 tygodnie",
        "ODDYCHANIE + CORE: 30min ćwiczenia oddechowe + stabilizacja tułowia — wsparcie układu",
        "PŁYWANIE/ROWER STACJONARNY: 45-60min @ Z2 — niskoudarowy trening w kontrolowanych warunkach",
    ]


def format_drift_for_report(profile: CardiacDriftProfile) -> Dict[str, Any]:
    """Format cardiac drift analysis for JSON/PDF reports."""
    return {
        "metrics": {
            "ef_start": round(profile.ef_start, 3),
            "ef_end": round(profile.ef_end, 3),
            "delta_ef_abs": round(profile.delta_ef_abs, 3),
            "delta_ef_pct": round(profile.delta_ef_pct, 1),
            "ef_vs_temp_slope": round(profile.ef_vs_temp_slope, 4) if profile.ef_vs_temp_slope else None,
            "temp_at_10pct_drop": round(profile.temp_at_10pct_drop, 1) if profile.temp_at_10pct_drop else None,
        },
        "key_signals": {
            "hsi_peak": round(profile.hsi_peak, 1),
            "smo2_drift_pct": round(profile.smo2_drift_pct, 1),
            "hr_drift_pct": round(profile.hr_drift_pct, 1),
        },
        "classification": {
            "drift_level": profile.drift_classification,
            "drift_type": profile.drift_type,
            "color": profile.classification_color,
            "thresholds": {
                "minimal": "<5%",
                "moderate": "5-10%",
                "high": ">10%"
            }
        },
        "interpretation": {
            "mechanism": profile.mechanism_description,
            "key_signals_summary": profile.key_signals_summary,
            "training_implications": profile.training_implications
        },
        "quality": {
            "data_points": profile.data_points,
            "confidence": round(profile.confidence, 2)
        }
    }


__all__ = [
    "CardiacDriftProfile",
    "analyze_cardiac_drift",
    "calculate_efficiency_factor",
    "format_drift_for_report",
]
