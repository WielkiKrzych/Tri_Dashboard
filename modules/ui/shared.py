"""Shared UI components — eliminates repeated Streamlit patterns."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.plots import CHART_CONFIG


def chart(
    fig: go.Figure,
    *,
    key: str | None = None,
    use_selection: bool = False,
    config: dict | None = None,
) -> go.Figure | None:
    """Render a Plotly chart with project defaults.

    Replaces: st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    """
    kwargs: dict = {"width": "stretch", "config": config or CHART_CONFIG}
    if key:
        kwargs["key"] = key
    if use_selection:
        return st.plotly_chart(fig, on_select="rerun", **kwargs)
    st.plotly_chart(fig, **kwargs)
    return None


def metric(
    label: str,
    value: float | int | str | None,
    *,
    delta: str | float | int | None = None,
    delta_color: str | None = None,
    help: str | None = None,
    column: st.columns | None = None,
    prefix: str = "",
    suffix: str = "",
    fmt: str | None = None,
    none_placeholder: str = "—",
) -> None:
    """Display a metric with formatting and null-safety.

    Replaces: if val: st.metric("Label", f"{val:.0f} W") else: st.metric("Label", "—")
    """
    if value is None:
        display = none_placeholder
    elif isinstance(value, str):
        display = value
    elif fmt:
        display = f"{prefix}{value:{fmt}}{suffix}"
    else:
        display = f"{prefix}{value}{suffix}"

    target = column if column is not None else st
    target.metric(label, display, delta=delta, delta_color=delta_color, help=help)


def require_data(df: pd.DataFrame | None, *, column: str | None = None) -> bool:
    """Show standard error/info banners and return False if data missing.

    Replaces:
        if target_df is None or target_df.empty:
            st.error("Brak danych.")
            return
    """
    if df is None or df.empty:
        st.error("Brak danych. Najpierw wgraj plik w sidebar.")
        return False
    if column and column not in df.columns:
        st.info(f"ℹ️ Brak danych {column} w tym pliku.")
        return False
    return True


def dataframe(
    df: pd.DataFrame,
    *,
    height: int | None = None,
    hide_index: bool = True,
) -> None:
    """Render a DataFrame with project-standard defaults."""
    st.dataframe(df, width="stretch", hide_index=hide_index, height=height)


def alert(kind: str, message: str, *, icon: str | None = None) -> None:
    """Display a status alert. kind: error, warning, success, info."""
    icons = {"error": "❌", "warning": "⚠️", "success": "✅", "info": "ℹ️"}
    prefix = icon or icons.get(kind, "")
    full = f"{prefix} {message}" if prefix else message
    fn = {"error": st.error, "warning": st.warning, "success": st.success, "info": st.info}.get(
        kind, st.info
    )
    fn(full)
