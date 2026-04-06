"""
Caching layer for expensive computations.

Provides memoization for CPU-intensive operations with TTL support.
Uses diskcache as a simple alternative to Redis (no external dependencies).
"""

import hashlib
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
        # Hash DataFrame content to avoid collisions
        # Use pandas hash for correctness, combined with shape for uniqueness
        content_hash = str(pd.util.hash_pandas_object(arg).sum())
        return f"DF:{content_hash}:{len(arg)}:{len(arg.columns)}"
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


def make_cache_key(*args) -> str:
    """Generate an in-memory cache key from arguments. DataFrames are hashed by content."""
    parts = []
    for a in args:
        if isinstance(a, pd.DataFrame):
            parts.append(str(pd.util.hash_pandas_object(a).sum()))
        else:
            parts.append(repr(a))
    return hashlib.md5("|".join(parts).encode()).hexdigest()


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


# =============================================================================
# STREAMLIT CACHE_DATA - Rolling window computations
# =============================================================================

import streamlit as st  # noqa: E402


def _hash_df(df: pd.DataFrame) -> str:
    """Hash a DataFrame for cache key generation."""
    return str(pd.util.hash_pandas_object(df).sum())


@st.cache_data(hash_funcs={pd.DataFrame: _hash_df}, show_spinner=False)
def cached_rolling_mean(
    series: pd.Series,
    window: int,
    center: bool = True,
    min_periods: int = 1,
) -> pd.Series:
    """
    Cached rolling mean computation.

    Avoids recomputing rolling windows on every Streamlit rerun.

    Args:
        series: Input pandas Series
        window: Rolling window size
        center: Whether to center the window
        min_periods: Minimum number of observations in window

    Returns:
        Series with rolling mean values
    """
    return series.rolling(window=window, center=center, min_periods=min_periods).mean()


@st.cache_data(hash_funcs={pd.DataFrame: _hash_df}, show_spinner=False)
def cached_rolling_median(
    series: pd.Series,
    window: int,
    center: bool = True,
    min_periods: int = 1,
) -> pd.Series:
    """
    Cached rolling median computation.

    Args:
        series: Input pandas Series
        window: Rolling window size
        center: Whether to center the window
        min_periods: Minimum number of observations in window

    Returns:
        Series with rolling median values
    """
    return series.rolling(window=window, center=center, min_periods=min_periods).median()
