"""
SRP: Moduł odpowiedzialny za obliczenia W' Balance (Skarbiec Beztlenowy).
"""
from typing import Union
import numpy as np
import pandas as pd
import io
from numba import jit

from ..utils import _serialize_df_to_parquet_bytes


@jit(nopython=True, fastmath=True)
def calculate_w_prime_fast(watts, time, cp, w_prime_cap):
    """W' Balance calculation using Skiba integral model with dynamic tau.

    Implements the Skiba et al. (2012, 2015) W' balance model with
    intensity-dependent recovery time constant (dynamic tau).

    During depletion (P > CP): W' decreases linearly with (P - CP) × dt.
    During recovery (P < CP): W' reconstitutes exponentially with
        tau = 546 × e^(-0.01 × DCP) + 316
    where DCP = CP - P (recovery intensity below CP).

    Reference:
        Skiba PF et al. (2012). "Modeling the expenditure and reconstitution
        of work capacity above critical power." Med Sci Sports Exerc.
        Skiba PF et al. (2015). "Validation of a novel intermittent W' model."

    Args:
        watts: Power array [W]
        time: Time array [s]
        cp: Critical Power [W]
        w_prime_cap: W' capacity [J]

    Returns:
        Array of W' Balance values over time [J]
    """
    n = len(watts)
    w_bal = np.empty(n, dtype=np.float64)
    curr_w = w_prime_cap

    prev_time = time[0]

    for i in range(n):
        if i == 0:
            dt = 1.0
        else:
            dt = time[i] - prev_time
            if dt <= 0:
                dt = 1.0
            prev_time = time[i]

        if watts[i] > cp:
            # Depletion: linear drain
            delta = (cp - watts[i]) * dt
            curr_w += delta
        elif watts[i] < cp:
            # Recovery: exponential reconstitution with dynamic tau (Skiba 2015)
            # tau = 546 × e^(-0.01 × DCP) + 316
            dcp = cp - watts[i]
            tau = 546.0 * np.exp(-0.01 * dcp) + 316.0

            # Exponential recovery toward w_prime_cap
            # W'(t+dt) = W'_cap - (W'_cap - W'(t)) × e^(-dt/tau)
            curr_w = w_prime_cap - (w_prime_cap - curr_w) * np.exp(-dt / tau)
        # else: watts[i] == cp → no change (steady state at CP)

        # Boundary conditions
        if curr_w > w_prime_cap:
            curr_w = w_prime_cap
        elif curr_w < 0.0:
            curr_w = 0.0

        w_bal[i] = curr_w

    return w_bal


@jit(nopython=True, fastmath=True)
def calculate_w_prime_biexp(watts, time, cp, w_prime_cap, sport: int = 0):
    """W' Balance with bi-exponential reconstitution (Caen et al. 2021).

    Two-phase recovery: fast (PCr resynthesis) + slow (metabolic recovery).
    Superior fit vs mono-exponential Skiba 2015 model.

    Sport-specific tau values (Welburn et al. 2025):
        Cycling:  tau_fast=120s, tau_slow=600s (validated)
        Running:  tau_fast=150s, tau_slow=750s (estimated, Fukuda 2019)
        Swimming: tau_fast=90s,  tau_slow=500s (estimated, Raimundo 2022)

    Note: Welburn et al. (2025) found poor predictive capability of
    generalized tau — individual calibration recommended.

    References:
        Caen et al. (2021). Bi-exponential W' reconstitution. EJAP.
        Welburn et al. (2025). W' reconstitution modelling. EJAP.
        Raimundo et al. (2022). Swimming D' reconstitution differs from cycling.

    Args:
        watts: Power array [W]
        time: Time array [s]
        cp: Critical Power [W]
        w_prime_cap: W' capacity [J]
        sport: 0=cycling, 1=running, 2=swimming
    """
    n = len(watts)
    w_bal = np.empty(n, dtype=np.float64)
    curr_w = w_prime_cap

    # Sport-specific time constants
    if sport == 1:    # Running
        tau_fast = 150.0
        tau_slow = 750.0
        fast_fraction = 0.45
    elif sport == 2:  # Swimming
        tau_fast = 90.0
        tau_slow = 500.0
        fast_fraction = 0.55
    else:             # Cycling (default)
        tau_fast = 120.0
        tau_slow = 600.0
        fast_fraction = 0.50

    slow_fraction = 1.0 - fast_fraction
    prev_time = time[0]

    for i in range(n):
        if i == 0:
            dt = 1.0
        else:
            dt = time[i] - prev_time
            if dt <= 0:
                dt = 1.0
            prev_time = time[i]

        if watts[i] > cp:
            delta = (cp - watts[i]) * dt
            curr_w += delta
        elif watts[i] < cp:
            dcp = cp - watts[i]
            # Bi-exponential: intensity modulates both tau values
            tau_f = tau_fast * np.exp(-0.008 * dcp) + tau_fast * 0.3
            tau_s = tau_slow * np.exp(-0.005 * dcp) + tau_slow * 0.3

            recovery_fast = fast_fraction * (1.0 - np.exp(-dt / tau_f))
            recovery_slow = slow_fraction * (1.0 - np.exp(-dt / tau_s))

            depleted = w_prime_cap - curr_w
            curr_w += depleted * (recovery_fast + recovery_slow)

        if curr_w > w_prime_cap:
            curr_w = w_prime_cap
        elif curr_w < 0.0:
            curr_w = 0.0

        w_bal[i] = curr_w

    return w_bal


def _calculate_w_prime_balance_cached(df_bytes: bytes, cp: float, w_prime: float):
    """Cached version of W' Balance calculation."""
    try:
        bio = io.BytesIO(df_bytes)
        try:
            df_pd = pd.read_parquet(bio)
        except Exception:
            bio.seek(0)
            df_pd = pd.read_csv(bio)

        if 'watts' not in df_pd.columns:
            df_pd['w_prime_balance'] = np.nan
            return df_pd

        watts_arr = df_pd['watts'].to_numpy(dtype=np.float64)
        
        if 'time' in df_pd.columns:
            time_arr = df_pd['time'].to_numpy(dtype=np.float64)
        else:
            time_arr = np.arange(len(watts_arr), dtype=np.float64)

        w_bal = calculate_w_prime_fast(watts_arr, time_arr, float(cp), float(w_prime))

        df_pd['w_prime_balance'] = w_bal
        return df_pd

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"W' calculation failed: {e}")
        # Try to return DataFrame with zero W' balance
        try:
            bio = io.BytesIO(df_bytes)
            try:
                df_pd = pd.read_parquet(bio)
            except (ImportError, ValueError):
                bio.seek(0)
                df_pd = pd.read_csv(bio)
            df_pd['w_prime_balance'] = 0.0
            return df_pd
        except (pd.errors.ParserError, ValueError, KeyError) as recovery_error:
            logging.getLogger(__name__).error(f"W' recovery failed: {recovery_error}")
            return pd.DataFrame({'w_prime_balance': []})


def calculate_w_prime_balance(_df_pl_active, cp: float, w_prime: float, model: str = "biexp", sport: int = 0) -> pd.DataFrame:
    """Calculate W' Balance for the entire workout.

    Args:
        _df_pl_active: DataFrame with workout data
        cp: Critical Power [W]
        w_prime: W' capacity [J]
        model: Reconstitution model — "biexp" (Caen 2021, default) or "skiba" (Skiba 2015)
        sport: Sport type — 0=cycling (default), 1=running, 2=swimming

    Returns:
        DataFrame with added 'w_prime_balance' column
    """
    if isinstance(_df_pl_active, dict):
        df_pd = pd.DataFrame(_df_pl_active)
    elif hasattr(_df_pl_active, 'to_pandas'):
        df_pd = _df_pl_active.to_pandas()
    else:
        df_pd = _df_pl_active.copy()

    if 'time' not in df_pd.columns:
        df_pd['time'] = np.arange(len(df_pd), dtype=float)

    if 'watts' not in df_pd.columns:
        df_pd['w_prime_balance'] = np.nan
        return df_pd

    watts_arr = df_pd['watts'].to_numpy(dtype=np.float64)
    time_arr = df_pd['time'].to_numpy(dtype=np.float64)

    if model == "biexp":
        w_bal = calculate_w_prime_biexp(watts_arr, time_arr, float(cp), float(w_prime), sport)
    else:
        w_bal = calculate_w_prime_fast(watts_arr, time_arr, float(cp), float(w_prime))

    df_pd['w_prime_balance'] = w_bal
    return df_pd


# ============================================================
# NEW: Recovery Score - TrainerRoad Readiness Inspired
# ============================================================

def calculate_recovery_score(
    w_bal_end: float,
    w_prime_capacity: float,
    time_since_effort_sec: int = 0,
    tau_seconds: float = 400.0,
    time_bonus_max: float = 30.0,
    return_rich: bool = False,
    smo2_baseline_drop_pct: float = 0.0,
    cardiac_drift_pct: float = 0.0,
) -> Union[float, 'RecoveryScoreResult']:
    """Calculate Recovery Score based on W' balance state.
    
    Estimates readiness for next high-intensity effort based on
    current W' balance and time since last effort.
    
    Recovery Score 0-100:
    - 90-100: Fully recovered, ready for any intensity
    - 70-90: Well recovered, can do threshold work
    - 50-70: Partially recovered, endurance zone preferred
    - 30-50: Fatigued, recovery ride only
    - <30: Exhausted, rest needed
    
    Args:
        w_bal_end: Current W' balance (J)
        w_prime_capacity: Full W' capacity (J)
        time_since_effort_sec: Time since last high-intensity effort
        tau_seconds: Time constant for W' reconstitution (default: 400s)
        time_bonus_max: Maximum time bonus points (default: 30)
        return_rich: If True, return RecoveryScoreResult; if False, return float
        smo2_baseline_drop_pct: SmO2 drop from baseline [%]. Indicates impaired O2 extraction.
        cardiac_drift_pct: Cardiac drift [%]. Indicates dehydration/fatigue.
        
    Returns:
        RecoveryScoreResult object (or float if return_rich=False)
    """
    from models import RecoveryScoreResult
    
    if w_prime_capacity <= 0:
        if return_rich:
            return RecoveryScoreResult(
                score=0.0, w_pct=0.0, time_bonus=0.0,
                tau_seconds=tau_seconds, time_bonus_max=time_bonus_max,
                recommendation=("❌ Brak danych", "Brak danych W'")
            )
        return 0.0
    
    # Base W' component (40% weight)
    w_component = (w_bal_end / w_prime_capacity) * 100

    # Time bonus
    time_bonus = 0.0
    if time_since_effort_sec > 0:
        recovery_factor = 1 - np.exp(-time_since_effort_sec / tau_seconds)
        time_bonus = recovery_factor * time_bonus_max

    w_readiness = min(100, w_component + time_bonus)

    # SmO2 component (30% weight) — baseline drop indicates impaired extraction
    smo2_readiness = max(0, 100 - smo2_baseline_drop_pct * 10)

    # Cardiac component (30% weight) — drift indicates dehydration/fatigue
    cardiac_readiness = max(0, 100 - cardiac_drift_pct * 8)

    # Composite score (Guimaraes Couto et al. 2025 pacing concept)
    if smo2_baseline_drop_pct > 0 or cardiac_drift_pct > 0:
        score = 0.4 * w_readiness + 0.3 * smo2_readiness + 0.3 * cardiac_readiness
    else:
        # Fallback to W'-only when no other data
        score = w_readiness

    score = round(max(0, min(100, score)), 0)

    # Keep w_pct for return_rich path
    w_pct = w_component
    
    if return_rich:
        recommendation = get_recovery_recommendation(score)
        return RecoveryScoreResult(
            score=score,
            w_pct=w_pct,
            time_bonus=time_bonus,
            tau_seconds=tau_seconds,
            time_bonus_max=time_bonus_max,
            recommendation=recommendation
        )
    return score


def get_recovery_recommendation(score: float) -> tuple:
    """Get training recommendation based on Recovery Score.
    
    Args:
        score: Recovery Score (0-100)
        
    Returns:
        Tuple of (zone_recommendation, description)
    """
    if score >= 90:
        return ("🟢 Pełna gotowość", "Możesz wykonać dowolny trening, włącznie z VO2max i sprintami.")
    elif score >= 70:
        return ("🟢 Dobra gotowość", "Trening progowy lub Sweet Spot OK. Unikaj maksymalnych wysiłków.")
    elif score >= 50:
        return ("🟡 Częściowe odzyskanie", "Zalecana strefa Z2/Z3. Skup się na objętości, nie intensywności.")
    elif score >= 30:
        return ("🟠 Zmęczenie", "Tylko łatwa jazda regeneracyjna (Z1). Odpoczywaj.")
    else:
        return ("🔴 Wyczerpanie", "Dzień wolny lub bardzo łatwa aktywność. Priorytet: regeneracja.")


def estimate_w_prime_reconstitution(
    depleted_pct: float,
    recovery_time_sec: int,
    tau: float = 400
) -> float:
    """Estimate W' reconstitution after recovery period.
    
    Uses exponential recovery model: W'(t) = W'_depleted * (1 - e^(-t/tau))
    
    Args:
        depleted_pct: How much W' was depleted (0-100%)
        recovery_time_sec: Recovery time in seconds
        tau: Time constant for W' reconstitution (default 400s)
        
    Returns:
        Estimated W' as percentage of capacity after recovery
    """
    remaining_pct = 100 - depleted_pct
    
    # How much of the depletion is recovered
    recovery_factor = 1 - np.exp(-recovery_time_sec / tau)
    recovered = depleted_pct * recovery_factor
    
    return round(remaining_pct + recovered, 1)

