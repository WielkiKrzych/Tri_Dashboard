"""
SmO2 Limiter Classifier Module

Classification logic for SmO2 limiters.
"""

from typing import Dict, Tuple
from .types import SmO2AdvancedMetrics
from .constants import LIMITER_THRESHOLDS, RECOMMENDATIONS


class SmO2LimiterClassifier:
    """Classifier for SmO2 limiters."""
    
    @classmethod
    def classify(cls, metrics: SmO2AdvancedMetrics) -> Tuple[str, float, str]:
        """
        Classify the primary limiter based on SmO2 metrics.
        
        Returns:
            Tuple of (limiter_type, confidence, interpretation)
        """
        scores = {"local": 0.0, "central": 0.0, "metabolic": 0.0}
        
        slope = metrics.slope_per_100w
        halftime = metrics.halftime_reoxy_sec
        coupling = metrics.hr_coupling_r
        
        # Slope analysis
        if slope < LIMITER_THRESHOLDS["slope_severe"]:
            scores["local"] += 3.0
            scores["metabolic"] += 1.0
        elif slope < LIMITER_THRESHOLDS["slope_moderate"]:
            scores["local"] += 1.5
        else:
            scores["central"] += 1.0
        
        # Halftime analysis
        if halftime is not None:
            if halftime > LIMITER_THRESHOLDS["halftime_slow"]:
                scores["local"] += 2.0
            elif halftime < LIMITER_THRESHOLDS["halftime_fast"]:
                scores["central"] += 1.5
            else:
                scores["metabolic"] += 1.0
        
        # Coupling analysis
        if coupling < LIMITER_THRESHOLDS["coupling_strong"]:
            scores["central"] += 2.5
        elif coupling > LIMITER_THRESHOLDS["coupling_weak"]:
            scores["local"] += 2.0
        else:
            scores["metabolic"] += 1.5
        
        total = sum(scores.values()) or 1.0
        max_score = max(scores.values())
        limiter_type = max(scores, key=scores.get)
        confidence = max_score / total
        
        interpretation = cls._generate_interpretation(limiter_type, metrics, scores)
        
        return limiter_type, confidence, interpretation
    
    @classmethod
    def _generate_interpretation(
        cls,
        limiter_type: str,
        metrics: SmO2AdvancedMetrics,
        scores: Dict[str, float]
    ) -> str:
        """Generate coach-oriented interpretation text."""
        slope = metrics.slope_per_100w
        halftime = metrics.halftime_reoxy_sec
        coupling = metrics.hr_coupling_r
        
        if limiter_type == "local":
            return cls._generate_local_interpretation(slope, halftime, coupling)
        elif limiter_type == "central":
            return cls._generate_central_interpretation(slope, halftime, coupling)
        else:
            return cls._generate_metabolic_interpretation(slope, halftime, coupling)
    
    @staticmethod
    def _generate_local_interpretation(slope, halftime, coupling):
        base = "LIMIT OBWODOWY (KAPILARYZACJA)"
        detail = f"SmO₂ spada o {abs(slope):.1f}%/100W – mięsień szybko wyczerpuje tlen lokalnie. "
        
        if halftime and halftime > 60:
            detail += f"Powolna reoksygenacja ({halftime:.0f}s) potwierdza słabą kapilaryzację. "
        if coupling > -0.5:
            detail += "Niska korelacja z HR wskazuje na niezależność od układu centralnego. "
        
        return f"{base}\n{detail}"
    
    @staticmethod
    def _generate_central_interpretation(slope, halftime, coupling):
        base = "LIMIT CENTRALNY (RZUT SERCA)"
        detail = f"Silna korelacja SmO₂-HR (r={coupling:.2f}) wskazuje, że serce dyktuje dostawę tlenu. "
        
        if abs(slope) < 4:
            detail += f"Umiarkowany spadek SmO₂ ({abs(slope):.1f}%/100W) potwierdza wystarczającą kapilaryzację. "
        if halftime and halftime < 45:
            detail += f"Szybka reoksygenacja ({halftime:.0f}s) – mięśnie sprawnie odbierają tlen. "
        
        return f"{base}\n{detail}"
    
    @staticmethod
    def _generate_metabolic_interpretation(slope, halftime, coupling):
        base = "LIMIT METABOLICZNY (GLIKOLIZA)"
        detail = (
            f"Profil mieszany: spadek SmO₂ {abs(slope):.1f}%/100W przy umiarkowanej korelacji z HR. "
            "Sugeruje wysoką produkcję mleczanu (VLaMax) jako główny czynnik."
        )
        if halftime and 45 < halftime < 90:
            detail += f" Umiarkowana reoksygenacja ({halftime:.0f}s) potwierdza stres metaboliczny. "
        
        return f"{base}\n{detail}"
    
    @staticmethod
    def get_recommendations(limiter_type: str) -> Tuple[str, ...]:
        """Get training recommendations for a given limiter type."""
        return RECOMMENDATIONS.get(limiter_type, RECOMMENDATIONS["unknown"])
