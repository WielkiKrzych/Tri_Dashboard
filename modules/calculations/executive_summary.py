"""
Executive Summary Calculations - Premium Edition.

Generates commercial-grade, decision-oriented summary data for Ramp Test reports.
Designed to match INSCYD/WKO5/WHOOP quality standards.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("Tri_Dashboard.ExecutiveSummary")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalStatus:
    """Status of a single signal."""
    name: str
    status: str  # "ok", "warning", "conflict"
    icon: str
    note: str = ""


@dataclass 
class ConfidenceBreakdown:
    """Detailed confidence components."""
    ve_stability: int  # 0-100
    hr_lag: int        # 0-100
    smo2_noise: int    # 0-100
    protocol_quality: int  # 0-100
    limiting_factor: str


@dataclass
class TrainingCard:
    """Single training recommendation card."""
    strategy_name: str
    power_range: str
    volume: str
    adaptation_goal: str
    expected_response: str
    risk_level: str  # "low", "medium", "high"


# =============================================================================
# LIMITER CLASSIFICATION (Enhanced)
# =============================================================================

LIMITER_TYPES = {
    "central": {
        "name": "CENTRALNY",
        "subtitle": "Układ Krążenia",
        "icon": "❤️",
        "system_icon": "heart",
        "color": "#E74C3C",
        "gradient": "linear-gradient(135deg, #E74C3C 0%, #C0392B 100%)",
        "verdict": "Serce i płuca są głównym ograniczeniem wydajności.",
        "interpretation": [
            "Próg tlenowy (VT1) jest niski względem VT2 – słaba baza aerobowa.",
            "HR szybko osiąga plateau, ograniczając możliwość dalszego wzrostu intensywności.",
            "Priorytet: zwiększenie objętości tlenowej i trening interwałowy VO₂max."
        ]
    },
    "peripheral": {
        "name": "OBWODOWY", 
        "subtitle": "Układ Mięśniowy",
        "icon": "💪",
        "system_icon": "muscle",
        "color": "#3498DB",
        "gradient": "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
        "verdict": "Mięśnie desaturyzują wcześniej niż wskazują progi systemowe.",
        "interpretation": [
            "SmO₂ spada przed osiągnięciem VT2 – lokalna kapilaryzacja jest limitująca.",
            "Układ krążenia dostarcza tlen, ale mięśnie nie wykorzystują go efektywnie.",
            "Priorytet: trening siłowy, sweet spot, praca pod progiem."
        ]
    },
    "metabolic": {
        "name": "METABOLICZNY",
        "subtitle": "Klirens Mleczanu",
        "icon": "🔥",
        "system_icon": "flame",
        "color": "#F39C12",
        "gradient": "linear-gradient(135deg, #F39C12 0%, #E67E22 100%)",
        "verdict": "Wysoka produkcja mleczanu (VLaMax) ogranicza moc progową.",
        "interpretation": [
            "Duża różnica między CP a VT2 wskazuje na wysoki VLaMax.",
            "Organizm szybko produkuje mleczan, co ogranicza wydolność tempo.",
            "Priorytet: długie jazdy Z2, obniżenie VLaMax, trening tlenowy."
        ]
    },
    "thermal": {
        "name": "TERMOREGULACYJNY",
        "subtitle": "Odprowadzanie Ciepła",
        "icon": "🌡️",
        "system_icon": "thermometer",
        "color": "#9B59B6",
        "gradient": "linear-gradient(135deg, #9B59B6 0%, #8E44AD 100%)",
        "verdict": "Dryf tętna wskazuje na problemy z termoregulacją.",
        "interpretation": [
            "Cardiac Drift przekracza normę – serce musi kompensować wzrost temperatury.",
            "Efektywność mechaniczna spada wraz z wzrostem temperatury głębokiej.",
            "Priorytet: adaptacja do ciepła, nawodnienie, chłodzenie przed wysiłkiem."
        ]
    },
    "balanced": {
        "name": "ZBALANSOWANY",
        "subtitle": "Profil Optymalny",
        "icon": "⚖️",
        "system_icon": "balance",
        "color": "#2ECC71",
        "gradient": "linear-gradient(135deg, #2ECC71 0%, #27AE60 100%)",
        "verdict": "Brak dominującego limitera – profil zrównoważony.",
        "interpretation": [
            "Wszystkie systemy fizjologiczne pracują w harmonii.",
            "Możliwość dalszego rozwoju w każdym kierunku.",
            "Priorytet: trening spolaryzowany, utrzymanie równowagi."
        ]
    }
}


def identify_main_limiter(
    thresholds: Dict[str, Any],
    smo2_manual: Dict[str, Any],
    cp_model: Dict[str, Any],
    kpi: Dict[str, Any]
) -> Dict[str, Any]:
    """Identify the main performance limiter with enhanced details."""
    scores = {"central": 0, "peripheral": 0, "metabolic": 0, "thermal": 0}
    
    def safe_float(val, default=0.0):
        if val is None or val == "brak danych":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    vt1 = safe_float(thresholds.get("vt1_raw_midpoint"))
    vt2 = safe_float(thresholds.get("vt2_raw_midpoint"))
    smo2_lt2 = safe_float(smo2_manual.get("lt2_watts"))
    cp = safe_float(cp_model.get("cp_watts"))
    pa_hr = safe_float(kpi.get("pa_hr"))
    
    # Scoring logic
    if vt1 > 0 and vt2 > 0:
        ratio = vt1 / vt2
        if ratio < 0.65:
            scores["central"] += 3
        elif ratio < 0.72:
            scores["central"] += 1
    
    if smo2_lt2 > 0 and vt2 > 0:
        diff = vt2 - smo2_lt2
        if diff > 25:
            scores["peripheral"] += 4
        elif diff > 15:
            scores["peripheral"] += 2
        elif diff > 8:
            scores["peripheral"] += 1
    
    if cp > 0 and vt2 > 0:
        gap = abs(cp - vt2)
        if gap > 35:
            scores["metabolic"] += 3
        elif gap > 20:
            scores["metabolic"] += 1
    
    if pa_hr > 6:
        scores["thermal"] += 4
    elif pa_hr > 4:
        scores["thermal"] += 2
    elif pa_hr > 3:
        scores["thermal"] += 1
    
    # Determine winner
    max_score = max(scores.values())
    if max_score < 2:
        limiter_type = "balanced"
        severity = "low"
    else:
        limiter_type = max(scores, key=scores.get)
        severity = "critical" if max_score >= 4 else ("high" if max_score >= 3 else "medium")
    
    limiter_info = LIMITER_TYPES[limiter_type].copy()
    limiter_info["limiter_type"] = limiter_type
    limiter_info["severity"] = severity
    limiter_info["scores"] = scores
    
    return limiter_info


# =============================================================================
# SIGNAL AGREEMENT MATRIX
# =============================================================================

def build_signal_matrix(
    thresholds: Dict[str, Any],
    smo2_manual: Dict[str, Any],
    kpi: Dict[str, Any]
) -> Dict[str, Any]:
    """Build signal agreement matrix."""
    
    def safe_float(val, default=0.0):
        if val is None or val == "brak danych":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    vt2 = safe_float(thresholds.get("vt2_raw_midpoint"))
    smo2_lt2 = safe_float(smo2_manual.get("lt2_watts"))
    pa_hr = safe_float(kpi.get("pa_hr"))
    
    signals = []
    conflict_score = 0.0
    
    # VE Signal
    ve_status = "ok"
    ve_note = "Progi VT1/VT2 wykryte poprawnie"
    if vt2 == 0:
        ve_status = "warning"
        ve_note = "VT2 nie wykryty"
        conflict_score += 0.3
    signals.append(SignalStatus("VE", ve_status, "🫁", ve_note))
    
    # HR Signal
    hr_status = "ok"
    hr_note = "HR koreluje z VE"
    if pa_hr > 5:
        hr_status = "conflict"
        hr_note = f"Cardiac Drift {pa_hr:.1f}% – niestabilny HR"
        conflict_score += 0.3
    elif pa_hr > 3:
        hr_status = "warning"
        hr_note = f"Lekki drift {pa_hr:.1f}%"
        conflict_score += 0.1
    signals.append(SignalStatus("HR", hr_status, "❤️", hr_note))
    
    # SmO2 Signal
    smo2_status = "ok"
    smo2_note = "SmO₂ potwierdza progi systemowe"
    if smo2_lt2 > 0 and vt2 > 0:
        diff = vt2 - smo2_lt2
        if diff > 20:
            smo2_status = "conflict"
            smo2_note = f"SmO₂ LT2 {int(smo2_lt2)}W < VT2 {int(vt2)}W"
            conflict_score += 0.4
        elif diff > 10:
            smo2_status = "warning"
            smo2_note = f"Lekka rozbieżność ({int(diff)}W)"
            conflict_score += 0.15
    elif smo2_lt2 == 0:
        smo2_status = "warning"
        smo2_note = "Brak danych SmO₂"
        conflict_score += 0.1
    signals.append(SignalStatus("SmO₂", smo2_status, "💪", smo2_note))
    
    # Normalize conflict score to 0-1
    conflict_index = min(1.0, conflict_score)
    agreement_index = 1.0 - conflict_index
    
    return {
        "signals": [{"name": s.name, "status": s.status, "icon": s.icon, "note": s.note} for s in signals],
        "conflict_index": round(conflict_index, 2),
        "agreement_index": round(agreement_index, 2),
        "agreement_label": "Wysoka" if agreement_index >= 0.8 else ("Średnia" if agreement_index >= 0.5 else "Niska")
    }


# =============================================================================
# CONFIDENCE PANEL (Enhanced)
# =============================================================================

def calculate_confidence_panel(
    confidence: Dict[str, Any],
    thresholds: Dict[str, Any],
    kpi: Dict[str, Any],
    signal_matrix: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate detailed confidence breakdown."""
    
    # VE Stability (based on VT detection)
    ve_stability = 85
    if thresholds.get("vt2_raw_midpoint") is None:
        ve_stability -= 30
    if thresholds.get("vt1_raw_midpoint") is None:
        ve_stability -= 20
    
    # HR Lag (based on cardiac drift)
    hr_lag = 90
    pa_hr = kpi.get("pa_hr")
    if pa_hr and pa_hr != "brak danych":
        try:
            drift = float(pa_hr)
            if drift > 5:
                hr_lag -= 30
            elif drift > 3:
                hr_lag -= 15
        except (ValueError, TypeError):
            pass
    
    # SmO2 Noise (based on conflict with VT)
    smo2_noise = 85
    conflict_idx = signal_matrix.get("conflict_index", 0)
    smo2_noise -= int(conflict_idx * 40)
    
    # Protocol Quality (base confidence)
    protocol_quality = int(confidence.get("overall_confidence", 0.7) * 100)
    
    # Clamp all values
    ve_stability = max(0, min(100, ve_stability))
    hr_lag = max(0, min(100, hr_lag))
    smo2_noise = max(0, min(100, smo2_noise))
    protocol_quality = max(0, min(100, protocol_quality))
    
    # Overall score (weighted average)
    overall = int(0.3 * ve_stability + 0.25 * hr_lag + 0.25 * smo2_noise + 0.2 * protocol_quality)
    
    # Determine limiting factor
    components = {
        "Stabilność VE": ve_stability,
        "Lag HR": hr_lag,
        "Szum SmO₂": smo2_noise,
        "Protokół": protocol_quality
    }
    limiting_factor = min(components, key=components.get)
    
    return {
        "overall_score": overall,
        "breakdown": {
            "ve_stability": ve_stability,
            "hr_lag": hr_lag,
            "smo2_noise": smo2_noise,
            "protocol_quality": protocol_quality
        },
        "limiting_factor": limiting_factor,
        "label": "Wysoka" if overall >= 75 else ("Średnia" if overall >= 50 else "Niska"),
        "color": "#2ECC71" if overall >= 75 else ("#F39C12" if overall >= 50 else "#E74C3C")
    }


# =============================================================================
# TRAINING DECISION CARDS (Enhanced)
# =============================================================================

def generate_training_cards(
    limiter: Dict[str, Any],
    thresholds: Dict[str, Any],
    cp_model: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate 3 premium training decision cards."""
    
    def safe_int(val, default=0):
        if val is None or val == "brak danych":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default
    
    vt1 = safe_int(thresholds.get("vt1_raw_midpoint"))
    vt2 = safe_int(thresholds.get("vt2_raw_midpoint"))
    cp = safe_int(cp_model.get("cp_watts"))
    
    limiter_type = limiter.get("limiter_type", "balanced")
    
    cards = []
    
    if limiter_type == "central":
        cards = [
            {
                "strategy_name": "BUDOWA BAZY AEROBOWEJ",
                "power_range": f"{int(vt1*0.70)}–{int(vt1*0.85)}W" if vt1 else "Z2",
                "volume": "3–4h / sesja, 12–16h / tydzień",
                "adaptation_goal": "Zwiększenie wydolności tlenowej i kapilaryzacji",
                "expected_response": "Wzrost VT1 o 5–10W w ciągu 4–6 tygodni",
                "risk_level": "low"
            },
            {
                "strategy_name": "INTERWAŁY VO₂max",
                "power_range": f"{int(vt2*1.05)}–{int(vt2*1.15)}W" if vt2 else "106–120% FTP",
                "volume": "4–6 × 4min, 2× / tydzień",
                "adaptation_goal": "Podniesienie pułapu tlenowego",
                "expected_response": "Wzrost VO₂max o 3–5% w 8 tygodni",
                "risk_level": "medium"
            },
            {
                "strategy_name": "TEMPO / THRESHOLD",
                "power_range": f"{int(vt1*0.95)}–{int(vt2*0.95)}W" if vt1 else "Sweet Spot",
                "volume": "2 × 20min, 1–2× / tydzień",
                "adaptation_goal": "Podniesienie VT1 bliżej VT2",
                "expected_response": "Poprawa stosunku VT1/VT2 o 3–5%",
                "risk_level": "low"
            }
        ]
    elif limiter_type == "peripheral":
        cards = [
            {
                "strategy_name": "SWEET SPOT + SIŁA",
                "power_range": f"{int(vt1*1.0)}–{int(vt2*0.92)}W" if vt1 else "88–94% FTP",
                "volume": "2 × 20min + 3 × 10min niska kadencja",
                "adaptation_goal": "Poprawa kapilaryzacji i siły mięśniowej",
                "expected_response": "Zbliżenie SmO₂ LT2 do VT2 o 10–15W",
                "risk_level": "medium"
            },
            {
                "strategy_name": "TRENING SIŁOWY NA ROWERZE",
                "power_range": f"{int(vt1*0.85)}–{int(vt1*0.95)}W @ 50–60rpm" if vt1 else "Z3 @ niska kadencja",
                "volume": "4 × 8min, 1× / tydzień",
                "adaptation_goal": "Rozwój włókien wolnokurczliwych",
                "expected_response": "Poprawa momentu obrotowego",
                "risk_level": "low"
            },
            {
                "strategy_name": "OBJĘTOŚĆ AEROBOWA",
                "power_range": f"{int(vt1*0.70)}–{int(vt1*0.82)}W" if vt1 else "Z2",
                "volume": "2–3h / sesja, 10–14h / tydzień",
                "adaptation_goal": "Rozbudowa sieci naczyń włosowatych",
                "expected_response": "Wzrost SmO₂ bazowego o 2–4%",
                "risk_level": "low"
            }
        ]
    elif limiter_type == "metabolic":
        cards = [
            {
                "strategy_name": "OBNIŻENIE VLaMax",
                "power_range": f"{int(vt1*0.65)}–{int(vt1*0.78)}W" if vt1 else "Z2 niskie",
                "volume": "4–5h / sesja, 14–20h / tydzień",
                "adaptation_goal": "Redukcja maksymalnej produkcji mleczanu",
                "expected_response": "Spadek VLaMax, wzrost FatMax",
                "risk_level": "low"
            },
            {
                "strategy_name": "TEMPO DŁUGIE",
                "power_range": f"{int(vt1*0.92)}–{int(vt2*0.88)}W" if vt1 else "85–92% FTP",
                "volume": "60–90min ciągłe, 1× / tydzień",
                "adaptation_goal": "Efektywność metaboliczna na progu",
                "expected_response": "Poprawa klirensu mleczanu",
                "risk_level": "medium"
            },
            {
                "strategy_name": "TRENINGI NA CZCZO",
                "power_range": f"{int(vt1*0.60)}–{int(vt1*0.72)}W" if vt1 else "Z2 bardzo niskie",
                "volume": "1.5–2.5h, 1–2× / tydzień",
                "adaptation_goal": "Optymalizacja spalania tłuszczy",
                "expected_response": "Wzrost FatMax o 10–15W",
                "risk_level": "medium"
            }
        ]
    elif limiter_type == "thermal":
        cards = [
            {
                "strategy_name": "ADAPTACJA DO CIEPŁA",
                "power_range": f"{int(vt1*0.75)}–{int(vt1*0.88)}W w cieple" if vt1 else "Z2 w cieple",
                "volume": "1–1.5h, 10–14 dni protokół",
                "adaptation_goal": "Poprawa termoregulacji i pocenia się",
                "expected_response": "Spadek HR o 10–15 bpm w cieple",
                "risk_level": "medium"
            },
            {
                "strategy_name": "NAWODNIENIE + ELEKTROLITY",
                "power_range": "Wszystkie strefy",
                "volume": "500–750ml/h + Na 500–1000mg/h",
                "adaptation_goal": "Utrzymanie objętości osocza",
                "expected_response": "Redukcja Cardiac Drift o 2–3%",
                "risk_level": "low"
            },
            {
                "strategy_name": "PRE-COOLING",
                "power_range": f"{int(vt2*0.95)}–{int(vt2*1.05)}W" if vt2 else "Threshold",
                "volume": "Lód + zimna woda przed startem",
                "adaptation_goal": "Większy margines termiczny",
                "expected_response": "Dłuższy czas do przegrzania",
                "risk_level": "low"
            }
        ]
    else:  # balanced
        cards = [
            {
                "strategy_name": "TRENING SPOLARYZOWANY",
                "power_range": f"80% @ {int(vt1*0.75)}W / 20% @ {int(vt2*1.10)}W" if vt1 else "80/20",
                "volume": "10–14h / tydzień, 2 sesje intensywne",
                "adaptation_goal": "Utrzymanie formy + rozwój VO₂max",
                "expected_response": "Stabilizacja lub wzrost progów",
                "risk_level": "low"
            },
            {
                "strategy_name": "SWEET SPOT MAINTENANCE",
                "power_range": f"{int(vt2*0.88)}–{int(vt2*0.94)}W" if vt2 else "88–94% FTP",
                "volume": "2 × 20min, 1× / tydzień",
                "adaptation_goal": "Podtrzymanie mocy progowej",
                "expected_response": "Utrzymanie CP/FTP",
                "risk_level": "low"
            },
            {
                "strategy_name": "RACE SIMULATION",
                "power_range": "Specyficzne dla dyscypliny",
                "volume": "1× wyścig lub symulacja / tydzień",
                "adaptation_goal": "Dopracowanie taktyki i pacingu",
                "expected_response": "Lepsze zarządzanie wysiłkiem",
                "risk_level": "medium"
            }
        ]
    
    return cards


# =============================================================================
# MAIN ENTRY POINT (Enhanced)
# =============================================================================

def generate_executive_summary(
    thresholds: Dict[str, Any],
    smo2_manual: Dict[str, Any],
    cp_model: Dict[str, Any],
    kpi: Dict[str, Any],
    confidence: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate complete premium executive summary data."""
    
    limiter = identify_main_limiter(thresholds, smo2_manual, cp_model, kpi)
    signal_matrix = build_signal_matrix(thresholds, smo2_manual, kpi)
    confidence_panel = calculate_confidence_panel(confidence, thresholds, kpi, signal_matrix)
    training_cards = generate_training_cards(limiter, thresholds, cp_model)
    
    return {
        "limiter": limiter,
        "signal_matrix": signal_matrix,
        "confidence_panel": confidence_panel,
        "training_cards": training_cards,
        # Legacy compatibility
        "conflicts": " | ".join([s["note"] for s in signal_matrix["signals"] if s["status"] != "ok"]) or "Brak konfliktów",
        "confidence_score": confidence_panel["overall_score"],
        "recommendations": [
            {"zone": c["power_range"], "duration": c["volume"], "goal": c["adaptation_goal"]}
            for c in training_cards
        ]
    }


__all__ = [
    "generate_executive_summary",
    "identify_main_limiter",
    "build_signal_matrix",
    "calculate_confidence_panel",
    "generate_training_cards",
    "LIMITER_TYPES",
]
