"""
Polars DataFrame adapter — backward-compatible re-export shim.

Canonical implementation: modules.calculations.polars_adapter
This module re-exports everything for backward compatibility.
"""

from modules.calculations.polars_adapter import *  # noqa: F401,F403
from modules.calculations.polars_adapter import PolarsAdapter  # explicit for test imports

__all__ = [
    "PolarsAdapter",
    "is_polars_available",
    "is_polars_df",
    "is_pandas_df",
    "to_polars",
    "to_pandas",
    "ensure_polars",
    "fast_rolling_mean",
    "fast_groupby_agg",
    "fast_filter",
    "fast_read_csv",
    "fast_normalized_power",
    "fast_power_duration_curve",
]
