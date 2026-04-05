"""
Canonical column aliases and normalization per DOMAIN_MODEL.md §2.

Single source of truth for column name normalization and alias resolution.
All UI and pipeline modules should use these helpers instead of inline
df.columns.str.lower().str.strip() and manual alias loops.
"""

from __future__ import annotations

import pandas as pd

HR_ALIASES: tuple[str, ...] = (
    "heart_rate",
    "heart rate",
    "bpm",
    "tętno",
    "heartrate",
    "heart_rate_bpm",
)
POWER_ALIASES: tuple[str, ...] = ("power",)
VE_ALIASES: tuple[str, ...] = ()
BREATH_RATE_ALIASES: tuple[str, ...] = ("br", "rr", "breath_rate")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names. Mutates in-place, returns df for chaining."""
    df.columns = pd.Index([str(column).lower().strip() for column in df.columns])
    return df


def resolve_hr_column(df: pd.DataFrame) -> str | None:
    """Rename first matching HR alias to canonical 'hr'. Returns 'hr' or None."""
    if "hr" in df.columns:
        return "hr"
    for alias in HR_ALIASES:
        if alias in df.columns:
            df.rename(columns={alias: "hr"}, inplace=True)
            return "hr"
    return None


def resolve_power_column(df: pd.DataFrame) -> str | None:
    """Rename first matching power alias to canonical 'watts'. Returns 'watts' or None."""
    if "watts" in df.columns:
        return "watts"
    for alias in POWER_ALIASES:
        if alias in df.columns:
            df.rename(columns={alias: "watts"}, inplace=True)
            return "watts"
    return None


def resolve_breath_rate_column(df: pd.DataFrame) -> str | None:
    """Rename first matching breath-rate alias to canonical 'tymebreathrate'. Returns it or None."""
    if "tymebreathrate" in df.columns:
        return "tymebreathrate"
    for alias in BREATH_RATE_ALIASES:
        if alias in df.columns:
            df.rename(columns={alias: "tymebreathrate"}, inplace=True)
            return "tymebreathrate"
    return None


def resolve_all_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize + resolve all known aliases in one pass. Mutates in-place."""
    normalize_columns(df)
    resolve_hr_column(df)
    resolve_power_column(df)
    resolve_breath_rate_column(df)
    return df
