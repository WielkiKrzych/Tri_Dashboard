"""
Caching layer for expensive computations.

Provides memoization for CPU-intensive operations with TTL support.
Uses diskcache as a simple alternative to Redis (no external dependencies).
"""

import hashlib
import json
import pickle
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to use diskcache (file-based, no external dependencies)
try:
    from diskcache import Cache

    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    Cache = None

from modules.config import Config

T = TypeVar("T")

# Global cache instance
_cache: Optional[Cache] = None


def get_cache() -> Optional[Cache]:
    """Get or create global cache instance."""
    global _cache
    if not _CACHE_AVAILABLE:
        return None

    if _cache is None:
        cache_dir = (
            Path(Config.DB_PATH).parent / "cache" if hasattr(Config, "DB_PATH") else Path("./cache")
        )
        cache_dir.mkdir(exist_ok=True)
        _cache = Cache(str(cache_dir))

    return _cache


def cache_result(ttl: int = 3600, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        key_func: Optional function to generate cache key from arguments

    Usage:
        @cache_result(ttl=3600)
        def expensive_calculation(df: pd.DataFrame, param: int) -> dict:
            return heavy_computation(df, param)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = get_cache()
            if cache is None:
                return func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            try:
                result = cache.get(cache_key)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug("Cache read error for key %s: %s", cache_key, e)

            # Compute and cache
            result = func(*args, **kwargs)
            try:
                cache.set(cache_key, result, expire=ttl)
            except Exception as e:
                logger.debug("Cache write error for key %s: %s", cache_key, e)

            return result

        # Add cache invalidation method
        wrapper.invalidate_cache = lambda *args, **kwargs: _invalidate_cache(
            func.__name__, args, kwargs, key_func
        )

        return wrapper

    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a deterministic cache key from function arguments."""
    # Convert arguments to hashable form
    key_parts = [func_name]

    for arg in args:
        key_parts.append(_hash_arg(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={_hash_arg(v)}")

    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _hash_arg(arg: Any) -> str:
    """Convert argument to hashable string representation."""
    if isinstance(arg, pd.DataFrame):
        # Hash based on column names and shape (not full data for performance)
        cols = ",".join(sorted(arg.columns))
        return f"DF:{cols}:{len(arg)}"
    elif isinstance(arg, pd.Series):
        return f"SER:{arg.name}:{len(arg)}"
    elif isinstance(arg, np.ndarray):
        return f"ARR:{arg.shape}:{arg.dtype}"
    elif isinstance(arg, (list, tuple)):
        return f"LIST:{len(arg)}"
    elif isinstance(arg, dict):
        return f"DICT:{len(arg)}"
    else:
        return str(arg)


def _invalidate_cache(func_name: str, args: tuple, kwargs: dict, key_func: Optional[Callable]):
    """Invalidate cache entry for specific arguments."""
    cache = get_cache()
    if cache is None:
        return

    if key_func:
        cache_key = key_func(*args, **kwargs)
    else:
        cache_key = _generate_cache_key(func_name, args, kwargs)

    try:
        cache.delete(cache_key)
    except Exception as e:
        logger.debug("Cache invalidation error for key %s: %s", cache_key, e)


def clear_cache():
    """Clear all cached results."""
    cache = get_cache()
    if cache:
        cache.clear()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache = get_cache()
    if cache is None:
        return {"enabled": False}

    try:
        return {"enabled": True, "size": len(cache), "volume": cache.volume()}
    except Exception:
        return {"enabled": True, "error": "Could not get stats"}


def make_cache_key(*args) -> str:
    """Generate an in-memory cache key from arguments. DataFrames are hashed by content."""
    parts = []
    for a in args:
        if isinstance(a, pd.DataFrame):
            parts.append(str(pd.util.hash_pandas_object(a).sum()))
        else:
            parts.append(repr(a))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# Pre-configured cache decorators for common use cases

cache_1h = cache_result(ttl=3600)  # 1 hour
cache_24h = cache_result(ttl=86400)  # 24 hours
cache_7d = cache_result(ttl=604800)  # 7 days


# Cached versions of expensive operations


@cache_result(ttl=3600)
def cached_analyze_step_test(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of step test analysis."""
    from modules.calculations.thresholds import analyze_step_test

    return analyze_step_test(df, **kwargs)


@cache_result(ttl=3600)
def cached_detect_smo2_thresholds(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of SmO2 threshold detection."""
    from modules.calculations.smo2_advanced import detect_smo2_thresholds_moxy

    return detect_smo2_thresholds_moxy(df, **kwargs)


@cache_result(ttl=86400)  # 24 hours - CP doesn't change often
def cached_calculate_cp_wprime(df: pd.DataFrame, **kwargs) -> tuple:
    """Cached version of CP/W' calculation."""
    from modules.calculations.power import calculate_cp_wprime

    return calculate_cp_wprime(df, **kwargs)


@cache_result(ttl=3600)
def cached_generate_summary_pdf(*args, **kwargs) -> bytes:
    """Cached version of PDF generation."""
    from modules.reporting.pdf.summary_pdf import generate_summary_pdf

    return generate_summary_pdf(*args, **kwargs)


# =============================================================================
# POWER CALCULATIONS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_calculate_power_duration_curve(df: pd.DataFrame, durations: list = None) -> dict:
    """Cached version of power duration curve calculation."""
    from modules.calculations.power import calculate_power_duration_curve

    return calculate_power_duration_curve(df, durations)


@cache_result(ttl=3600)
def cached_calculate_normalized_power(df: pd.DataFrame) -> float:
    """Cached version of normalized power calculation."""
    from modules.calculations.power import calculate_normalized_power

    return calculate_normalized_power(df)


@cache_result(ttl=3600)
def cached_calculate_fatigue_resistance_index(mmp_5min: float, mmp_20min: float) -> float:
    """Cached version of fatigue resistance index calculation."""
    from modules.calculations.power import calculate_fatigue_resistance_index

    return calculate_fatigue_resistance_index(mmp_5min, mmp_20min)


@cache_result(ttl=3600)
def cached_estimate_tte(target_power: float, cp: float, w_prime: float) -> float:
    """Cached version of TTE estimation."""
    from modules.calculations.power import estimate_tte

    return estimate_tte(target_power, cp, w_prime)


@cache_result(ttl=3600)
def cached_calculate_power_zones_time(df: pd.DataFrame, cp_input: int) -> dict:
    """Cached version of power zones time calculation."""
    from modules.calculations.power import calculate_power_zones_time

    return calculate_power_zones_time(df, cp_input)


# =============================================================================
# VENTILATORY THRESHOLDS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_detect_vt_from_steps(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of VT detection from step test."""
    from modules.calculations.vt_step import detect_vt_from_steps

    return detect_vt_from_steps(df, **kwargs)


@cache_result(ttl=3600)
def cached_detect_ve_only_thresholds(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of VE-only threshold detection."""
    from modules.calculations.vt_cpet_ve_only import detect_ve_only_thresholds

    return detect_ve_only_thresholds(df, **kwargs)


@cache_result(ttl=3600)
def cached_calculate_ve_metrics(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of VE metrics calculation."""
    from modules.calculations.vent_advanced import calculate_ve_metrics

    return calculate_ve_metrics(df, **kwargs)


# =============================================================================
# SmO2 ANALYSIS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_analyze_smo2_advanced(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of advanced SmO2 analysis."""
    from modules.calculations.smo2_analysis import analyze_smo2_advanced

    return analyze_smo2_advanced(df, **kwargs)


@cache_result(ttl=3600)
def cached_detect_smo2_breakpoints(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of SmO2 breakpoint detection."""
    from modules.calculations.kinetics import detect_smo2_breakpoints

    return detect_smo2_breakpoints(df, **kwargs)


@cache_result(ttl=3600)
def cached_fit_smo2_kinetics(time: np.ndarray, smo2: np.ndarray, **kwargs) -> dict:
    """Cached version of SmO2 kinetics fitting."""
    from modules.calculations.kinetics import fit_smo2_kinetics

    return fit_smo2_kinetics(time, smo2, **kwargs)


# =============================================================================
# CARDIAC DRIFT & EFFICIENCY - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_analyze_cardiac_drift(df: pd.DataFrame, cp: float, **kwargs) -> Any:
    """Cached version of cardiac drift analysis."""
    from modules.calculations.cardiac_drift import analyze_cardiac_drift

    return analyze_cardiac_drift(df, cp, **kwargs)


@cache_result(ttl=3600)
def cached_calculate_efficiency_factor(power: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """Cached version of efficiency factor calculation."""
    from modules.calculations.cardiac_drift import calculate_efficiency_factor

    return calculate_efficiency_factor(power, hr)


# =============================================================================
# THERMAL ANALYSIS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_calculate_heat_strain_index(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Cached version of heat strain index calculation."""
    from modules.calculations.thermal import calculate_heat_strain_index

    return calculate_heat_strain_index(df, **kwargs)


@cache_result(ttl=3600)
def cached_analyze_thermoregulation(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of thermoregulation analysis."""
    from modules.calculations.thermoregulation import analyze_thermoregulation

    return analyze_thermoregulation(df, **kwargs)


# =============================================================================
# HRV ANALYSIS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_calculate_dynamic_dfa_v2(rr_values: np.ndarray, **kwargs) -> float:
    """Cached version of DFA-alpha1 calculation."""
    from modules.calculations.hrv import calculate_dynamic_dfa_v2

    return calculate_dynamic_dfa_v2(rr_values, **kwargs)


# =============================================================================
# METABOLIC ENGINE - Cached for 24 hours (doesn't change often)
# =============================================================================


@cache_result(ttl=86400)
def cached_analyze_metabolic_engine(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of metabolic engine analysis (24h cache - changes rarely)."""
    from modules.calculations.metabolic_engine import analyze_metabolic_engine

    return analyze_metabolic_engine(df, **kwargs)


@cache_result(ttl=86400)
def cached_estimate_vlamax(df: pd.DataFrame, **kwargs) -> Optional[float]:
    """Cached version of VLamax estimation (24h cache)."""
    from modules.calculations.metabolic_engine import estimate_vlamax

    return estimate_vlamax(df, **kwargs)


# =============================================================================
# STAMINA & DURABILITY - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_calculate_stamina_score(
    vo2max: float, fri: float, w_prime: float, cp: float, weight: float
) -> float:
    """Cached version of stamina score calculation."""
    from modules.calculations.stamina import calculate_stamina_score

    return calculate_stamina_score(vo2max, fri, w_prime, cp, weight)


@cache_result(ttl=3600)
def cached_calculate_durability_index(df: pd.DataFrame, min_duration_min: int = 20) -> tuple:
    """Cached version of durability index calculation."""
    from modules.calculations.stamina import calculate_durability_index

    return calculate_durability_index(df, min_duration_min)


# =============================================================================
# NUTRITION - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_estimate_carbs_burned(df: pd.DataFrame, vt1_watts: float, vt2_watts: float) -> float:
    """Cached version of carbohydrate estimation."""
    from modules.calculations.nutrition import estimate_carbs_burned

    return estimate_carbs_burned(df, vt1_watts, vt2_watts)


# =============================================================================
# BIOMECHANICS - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_analyze_biomech_occlusion(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of biomechanics/occlusion analysis."""
    from modules.calculations.biomech_occlusion import analyze_biomech_occlusion

    return analyze_biomech_occlusion(df, **kwargs)


# =============================================================================
# CARDIOVASCULAR - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_analyze_cardiovascular(df: pd.DataFrame, **kwargs) -> Any:
    """Cached version of cardiovascular analysis."""
    from modules.calculations.cardio_advanced import analyze_cardiovascular

    return analyze_cardiovascular(df, **kwargs)


# =============================================================================
# W' BALANCE - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_calculate_w_prime_balance(df: pd.DataFrame, cp: float, w_prime: float) -> pd.DataFrame:
    """Cached version of W' balance calculation."""
    from modules.calculations.w_prime import calculate_w_prime_balance

    return calculate_w_prime_balance(df, cp, w_prime)


@cache_result(ttl=3600)
def cached_calculate_recovery_score(
    w_bal_end: float, w_prime: float, time_since_effort_sec: float
) -> float:
    """Cached version of recovery score calculation."""
    from modules.calculations.w_prime import calculate_recovery_score

    return calculate_recovery_score(w_bal_end, w_prime, time_since_effort_sec)


# =============================================================================
# EXECUTIVE SUMMARY - Cached for 1 hour
# =============================================================================


@cache_result(ttl=3600)
def cached_generate_executive_summary(df: pd.DataFrame, result: Any, **kwargs) -> str:
    """Cached version of executive summary generation."""
    from modules.calculations.executive_summary import generate_executive_summary

    return generate_executive_summary(df, result, **kwargs)


# =============================================================================
# STREAMLIT CACHE_RESOURCE - Singletons (cached at app level)
# =============================================================================
# These functions create singleton instances that persist across reruns
# Using @st.cache_resource for global resources that don't change


def get_session_store() -> "SessionStore":
    """
    Get cached SessionStore singleton.

    Uses st.cache_resource to persist the database connection
    across Streamlit reruns. This avoids re-initializing SQLite
    on every interaction.

    Returns:
        SessionStore: Cached database instance
    """
    import streamlit as st
    from modules.db import SessionStore

    @st.cache_resource
    def _get_store() -> "SessionStore":
        return SessionStore()

    return _get_store()


def get_theme_manager() -> "ThemeManager":
    """
    Get cached ThemeManager singleton.

    Uses st.cache_resource to persist the theme manager
    across reruns, avoiding re-loading CSS on each interaction.

    Returns:
        ThemeManager: Cached theme instance
    """
    import streamlit as st
    from modules.frontend.theme import ThemeManager

    @st.cache_resource
    def _get_theme() -> "ThemeManager":
        return ThemeManager()

    return _get_theme()


def get_training_notes() -> "TrainingNotes":
    """
    Get cached TrainingNotes singleton.

    Uses st.cache_resource to persist training notes
    across reruns.

    Returns:
        TrainingNotes: Cached notes instance
    """
    import streamlit as st
    from modules.notes import TrainingNotes

    @st.cache_resource
    def _get_notes() -> "TrainingNotes":
        return TrainingNotes()

    return _get_notes()


# =============================================================================
# DATAFRAME HASHING UTILITIES
# =============================================================================


def hash_dataframe_for_cache(df: pd.DataFrame, sample_size: int = 100) -> str:
    """
    Generate a stable hash for DataFrame caching.

    Uses a sample of rows (not full content) for performance.
    The hash is deterministic across reruns with same data.

    Args:
        df: DataFrame to hash
        sample_size: Number of rows to sample for hashing

    Returns:
        str: Hex digest of hash
    """
    # Use column names + shape + sample for fast hashing
    cols = ",".join(sorted(df.columns))
    n_rows = len(df)

    # Sample for large DataFrames
    if n_rows > sample_size:
        sample = df.head(sample_size)
    else:
        sample = df

    # Create hash from metadata + sample
    content = f"{cols}:{n_rows}:{sample.to_numpy().tobytes()}"
    return hashlib.md5(content.encode()).hexdigest()


def hash_params_for_cache(**kwargs) -> str:
    """
    Generate a stable hash for parameters caching.

    Args:
        **kwargs: Parameters to hash

    Returns:
        str: Hex digest of hash
    """
    # Sort keys for deterministic output
    items = sorted(kwargs.items())
    content = str(items)
    return hashlib.md5(content.encode()).hexdigest()
