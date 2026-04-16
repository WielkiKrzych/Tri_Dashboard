"""
Polars Adapter Module.

Provides seamless interoperability between Pandas and Polars.
Allows gradual migration without breaking existing code.
"""

import logging
from typing import Union, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

import pandas as pd


# Type alias for DataFrame compatibility
DataFrame = Union[pd.DataFrame, "pl.DataFrame"] if POLARS_AVAILABLE else pd.DataFrame


def is_polars_df(df: Any) -> bool:
    """Check if object is a Polars DataFrame."""
    if not POLARS_AVAILABLE:
        return False
    return isinstance(df, pl.DataFrame)


def is_pandas_df(df: Any) -> bool:
    """Check if object is a Pandas DataFrame."""
    return isinstance(df, pd.DataFrame)


def to_polars(df: Any) -> "pl.DataFrame":
    """Convert any DataFrame-like object to Polars.

    Args:
        df: Pandas DataFrame, Polars DataFrame, dict, or similar

    Returns:
        Polars DataFrame

    Raises:
        ImportError: If Polars is not installed
        TypeError: If conversion not possible
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is not installed. Run: pip install polars")

    if is_polars_df(df):
        return df

    if is_pandas_df(df):
        return pl.from_pandas(df)

    if isinstance(df, dict):
        return pl.DataFrame(df)

    if hasattr(df, "to_pandas"):
        # Some libraries have to_pandas method
        return pl.from_pandas(df.to_pandas())

    raise TypeError(f"Cannot convert {type(df)} to Polars DataFrame")


def to_pandas(df: Any) -> pd.DataFrame:
    """Convert any DataFrame-like object to Pandas.

    Args:
        df: Polars DataFrame, Pandas DataFrame, dict, or similar

    Returns:
        Pandas DataFrame
    """
    if is_pandas_df(df):
        return df

    if POLARS_AVAILABLE and is_polars_df(df):
        return df.to_pandas()

    if isinstance(df, dict):
        return pd.DataFrame(df)

    if hasattr(df, "to_pandas"):
        return df.to_pandas()

    raise TypeError(f"Cannot convert {type(df)} to Pandas DataFrame")


# ============================================================
# PolarsAdapter class (backward-compatible OOP interface)
# ============================================================


class PolarsAdapter:
    """
    Adapter class that provides Polars-accelerated operations
    with Pandas-compatible interface.

    Preserved for backward compatibility with existing tests.
    Uses module-level helpers (to_polars, to_pandas) internally.
    """

    def __init__(self, df: Union[pd.DataFrame, "pl.DataFrame"]):
        """Initialize with either Pandas or Polars DataFrame."""
        if POLARS_AVAILABLE and is_polars_df(df):
            self._df = df
            self._is_polars = True
        else:
            self._df = df if isinstance(df, pd.DataFrame) else to_pandas(df)
            self._is_polars = False

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "PolarsAdapter":
        """Convert Pandas DataFrame to Polars."""
        if not POLARS_AVAILABLE:
            return cls(df)

        try:
            pl_df = to_polars(df)
            return cls(pl_df)
        except Exception:
            return cls(df)

    def to_pandas(self) -> pd.DataFrame:
        """Convert back to Pandas DataFrame."""
        if self._is_polars:
            return self._df.to_pandas()
        return self._df

    def groupby_agg(self, group_col: str, agg_dict: dict[str, str]) -> pd.DataFrame:
        """
        Fast groupby aggregation.

        Args:
            group_col: Column to group by
            agg_dict: {column: aggregation_function}

        Returns:
            Aggregated DataFrame
        """
        if not self._is_polars:
            return self._df.groupby(group_col).agg(agg_dict).reset_index()

        agg_exprs = []
        for col, func in agg_dict.items():
            if func == "mean":
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif func == "sum":
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif func == "max":
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif func == "min":
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif func == "count":
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif func == "std":
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            else:
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_{func}"))

        result = self._df.group_by(group_col).agg(agg_exprs)
        return result.to_pandas()

    def rolling_mean(self, col: str, window: int) -> np.ndarray:
        """Fast rolling mean calculation."""
        if not self._is_polars:
            return self._df[col].rolling(window=window, min_periods=1).mean().values

        result = self._df.select(pl.col(col).rolling_mean(window_size=window, min_periods=1))
        return result.to_pandas()[col].values

    def filter(self, condition: str) -> "PolarsAdapter":
        """Filter rows based on a simple 'column op value' condition.

        Accepts conditions like "watts > 100", "heartrate >= 60".
        Only simple numeric comparisons are supported for safety.
        """
        import re

        _SAFE_CONDITION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*([\d.]+)$")
        match = _SAFE_CONDITION.match(condition.strip())
        if not match:
            raise ValueError(f"Unsafe or unsupported filter condition: {condition!r}")

        col_name, op, val_str = match.groups()
        val = float(val_str)

        if not self._is_polars:
            ops = {
                "==": "__eq__",
                "!=": "__ne__",
                ">": "__gt__",
                ">=": "__ge__",
                "<": "__lt__",
                "<=": "__le__",
            }
            mask = getattr(self._df[col_name], ops[op])(val)
            return PolarsAdapter(self._df[mask])

        expr_map = {
            "==": pl.col(col_name) == val,
            "!=": pl.col(col_name) != val,
            ">": pl.col(col_name) > val,
            ">=": pl.col(col_name) >= val,
            "<": pl.col(col_name) < val,
            "<=": pl.col(col_name) <= val,
        }
        result = self._df.filter(expr_map[op])
        return PolarsAdapter(result)

    def sort(self, by: str, descending: bool = False) -> "PolarsAdapter":
        """Sort by column."""
        if not self._is_polars:
            return PolarsAdapter(self._df.sort_values(by, ascending=not descending))

        result = self._df.sort(by, descending=descending)
        return PolarsAdapter(result)

    @property
    def shape(self) -> tuple:
        """Get DataFrame shape."""
        return self._df.shape

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        if self._is_polars:
            return self._df.columns
        return list(self._df.columns)


# ============================================================
# High-Performance Operations (Polars-first, Pandas fallback)
# ============================================================


def fast_rolling_mean(df: DataFrame, column: str, window: int, min_periods: int = 1) -> np.ndarray:
    """Fast rolling mean using Polars if available.

    Falls back to Pandas if Polars not installed.

    Returns numpy array for maximum compatibility.
    """
    if POLARS_AVAILABLE:
        try:
            pl_df = to_polars(df)
            result = (
                pl_df.select(
                    pl.col(column).rolling_mean(window_size=window, min_periods=min_periods)
                )
                .to_numpy()
                .flatten()
            )
            return result
        except Exception:
            pass

    # Pandas fallback
    pd_df = to_pandas(df)
    return pd_df[column].rolling(window=window, min_periods=min_periods).mean().values


# --- Ported from Analiza Kolarska: standalone function aliases ---

def is_polars_available():
    """Check if Polars is installed."""
    try:
        import polars
        return True
    except ImportError:
        return False

def ensure_polars(df):
    """Convert any DataFrame to Polars."""
    return PolarsAdapter.from_pandas(df) if hasattr(df, 'columns') and not hasattr(df, 'to_polars') else df

def ensure_pandas(df):
    """Convert any DataFrame to Pandas."""
    return PolarsAdapter.to_pandas(df) if hasattr(df, 'to_pandas') else df

def fast_filter(df, column, value):
    """Fast filter using Polars if available."""
    return PolarsAdapter.filter(df, column, value)

def fast_groupby_agg(df, group_col, agg_col, agg_func="mean"):
    """Fast groupby aggregation."""
    return PolarsAdapter.groupby_agg(df, group_col, agg_col, agg_func)

def fast_normalized_power(df, power_col="watts"):
    """Fast NP calculation using Polars if available."""
    import numpy as np
    import pandas as pd
    if power_col not in df.columns:
        return 0.0
    watts = df[power_col].dropna().values
    if len(watts) < 30:
        return float(np.mean(watts))
    rolling = np.mean(watts ** 4) ** 0.25
    return float(rolling)

def fast_power_duration_curve(df, power_col="watts", time_col="time"):
    """Fast PDC calculation."""
    import numpy as np
    import pandas as pd
    if power_col not in df.columns:
        return [], []
    watts = df[power_col].dropna().values
    durations = []
    powers = []
    for dur in [5, 10, 15, 30, 60, 120, 180, 300, 600, 1200, 1800, 3600]:
        if len(watts) < dur:
            break
        best = max(np.convolve(watts, np.ones(dur)/dur, mode='valid'))
        durations.append(dur)
        powers.append(float(best))
    return durations, powers

def fast_read_csv(filepath):
    """Fast CSV reading - uses Polars if available."""
    try:
        import polars as pl
        return pl.read_csv(filepath).to_pandas()
    except ImportError:
        import pandas as pd
        return pd.read_csv(filepath)
