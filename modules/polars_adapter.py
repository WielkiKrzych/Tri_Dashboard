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


# --- Ported from Analiza Kolarska: additional Polars utilities ---

def read_csv_fast(file_path, **kwargs):
    """Fast CSV reading using Polars when available. 10-50x faster for large files."""
    try:
        import polars as pl
        pl_df = pl.read_csv(file_path, **kwargs)
        return pl_df.to_pandas()
    except Exception:
        import pandas as pd
        return pd.read_csv(file_path, **kwargs)


def fast_moving_average(arr, window):
    """Fast moving average using convolution. O(n) complexity."""
    import numpy as np
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def benchmark_operation(func_pandas, func_polars, data_size=100000):
    """Benchmark Pandas vs Polars operation. Returns dict with timing results."""
    import time
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({
        "group": np.random.choice(["A", "B", "C"], data_size),
        "value": np.random.randn(data_size),
    })
    # Pandas timing
    t0 = time.perf_counter()
    func_pandas(df)
    pandas_time = time.perf_counter() - t0
    # Polars timing (if available)
    polars_time = None
    try:
        import polars as pl
        pl_df = pl.from_pandas(df)
        t0 = time.perf_counter()
        func_polars(pl_df)
        polars_time = time.perf_counter() - t0
    except Exception:
        pass
    result = {"pandas_time": round(pandas_time, 4), "data_size": data_size}
    if polars_time is not None:
        result["polars_time"] = round(polars_time, 4)
        result["speedup"] = round(pandas_time / polars_time, 2)
    return result
