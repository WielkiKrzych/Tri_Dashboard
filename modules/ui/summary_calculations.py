"""
Summary Calculations — pure math helpers, no Streamlit dependency.
"""

import pandas as pd
from typing import Any, Tuple

from modules.calculations.column_aliases import resolve_breath_rate_column, resolve_hr_column
from modules.ui.utils import hash_dataframe as _hash_dataframe


def _calculate_np(watts_series: pd.Series) -> float:
    """Obliczenie Normalized Power."""
    if len(watts_series) < 30:
        return float(watts_series.mean())
    rolling_avg = watts_series.rolling(30, min_periods=1).mean()
    fourth_power = rolling_avg**4
    return float(fourth_power.mean() ** 0.25)


def _estimate_cp_wprime(df_plot: pd.DataFrame) -> Tuple[float, float]:
    """Estymacja CP i W' z danych MMP."""
    from scipy import stats

    if "watts" not in df_plot.columns or len(df_plot) < 1200:
        return 0, 0

    durations = [180, 300, 600, 900, 1200]
    valid_durations = [d for d in durations if d < len(df_plot)]

    if len(valid_durations) < 3:
        return 0, 0

    work_values = []
    for d in valid_durations:
        p_raw = df_plot["watts"].rolling(window=d).mean().max()
        if isinstance(p_raw, pd.Series) or pd.isna(p_raw):
            return 0, 0
        p = float(p_raw)
        work_values.append(p * d)

    try:
        regression: Any = stats.linregress(valid_durations, work_values)
        return float(regression.slope), float(regression.intercept)
    except Exception:
        return 0, 0


def _get_vent_metrics_for_power(
    df_plot: pd.DataFrame, power_watts: float
) -> Tuple[float, float, float]:
    """
    Pobiera metryki wentylacyjne (HR, VE, BR) dla zadanej mocy z df_plot.
    Znajduje najbliższy punkt w danych do zadanej mocy i zwraca wartości.
    """
    if not power_watts or power_watts <= 0:
        return 0, 0, 0

    if "watts" not in df_plot.columns:
        return 0, 0, 0

    power_col = "watts_smooth_5s" if "watts_smooth_5s" in df_plot.columns else "watts"
    hr_col = resolve_hr_column(df_plot)
    br_col = resolve_breath_rate_column(df_plot)

    idx = (df_plot[power_col] - power_watts).abs().idxmin()

    start_idx = max(0, idx - 5)
    end_idx = min(len(df_plot), idx + 5)
    window_data = df_plot.iloc[start_idx:end_idx]

    hr_val = window_data[hr_col].mean() if hr_col else 0

    ve_val = window_data["tymeventilation"].mean() if "tymeventilation" in df_plot.columns else 0

    br_val = window_data[br_col].mean() if br_col else 0

    return hr_val, ve_val, br_val
