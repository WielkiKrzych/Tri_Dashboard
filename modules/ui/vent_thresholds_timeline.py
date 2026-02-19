"""
Vent Thresholds — interactive Plotly timeline with zone rectangles and threshold markers.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_threshold_timeline(cpet_result: dict, target_df: pd.DataFrame) -> None:
    """
    Render the interactive Plotly timeline showing VE, Power, HR over time
    with coloured zone backgrounds and 4 threshold vertical markers.
    """
    st.subheader("📈 Wizualizacja Progów")

    fig = go.Figure()

    # VE (Primary)
    fig.add_trace(
        go.Scatter(
            x=target_df["time"],
            y=target_df["ve_smooth"],
            customdata=target_df["time_str"],
            mode="lines",
            name="VE (L/min)",
            line=dict(color="#ffa15a", width=2),
            hovertemplate="<b>Czas:</b> %{customdata}<br><b>VE:</b> %{y:.1f} L/min<extra></extra>",
        )
    )

    # Power (Secondary axis)
    if "watts_smooth_5s" in target_df.columns:
        fig.add_trace(
            go.Scatter(
                x=target_df["time"],
                y=target_df["watts_smooth_5s"],
                customdata=target_df["time_str"],
                mode="lines",
                name="Power",
                line=dict(color="#1f77b4", width=1),
                yaxis="y2",
                opacity=0.3,
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>Moc:</b> %{y:.0f} W<extra></extra>",
            )
        )

    # HR (Secondary axis, dotted)
    if "hr" in target_df.columns:
        fig.add_trace(
            go.Scatter(
                x=target_df["time"],
                y=target_df["hr"],
                customdata=target_df["time_str"],
                mode="lines",
                name="Heart Rate",
                line=dict(color="#d62728", width=1, dash="dot"),
                yaxis="y2",
                opacity=0.5,
                hovertemplate="<b>Czas:</b> %{customdata}<br><b>HR:</b> %{y:.0f} bpm<extra></extra>",
            )
        )

    # Resolve threshold times and HR values
    def get_time_at_power(power_val):
        if power_val and "watts" in target_df.columns:
            mask = target_df["watts"] >= power_val
            if mask.any():
                return target_df.loc[mask, "time"].iloc[0]
        return None

    def get_hr_at_power(power_val, hr_key):
        hr = cpet_result.get(hr_key)
        if hr is None and power_val and "hr" in target_df.columns:
            mask = (target_df["watts"] >= power_val - 10) & (target_df["watts"] <= power_val + 10)
            if mask.any():
                hr_val = target_df.loc[mask, "hr"].mean()
                if pd.notna(hr_val) and hr_val > 0:
                    return int(hr_val)
        return hr

    vt1_onset_w = cpet_result.get("vt1_onset_watts") or cpet_result.get("vt1_watts")
    vt1_steady_w = cpet_result.get("vt1_steady_watts")
    rcp_onset_w = cpet_result.get("rcp_onset_watts") or cpet_result.get("vt2_watts")
    rcp_steady_w = cpet_result.get("rcp_steady_watts")

    vt1_onset_t = get_time_at_power(vt1_onset_w)
    vt1_steady_t = get_time_at_power(vt1_steady_w)
    rcp_onset_t = get_time_at_power(rcp_onset_w)
    rcp_steady_t = get_time_at_power(rcp_steady_w)

    zone_colors = {
        "z1": "rgba(100, 200, 100, 0.10)",
        "z2": "rgba(255, 161, 90, 0.15)",
        "z3": "rgba(0, 204, 150, 0.15)",
        "z4": "rgba(239, 85, 59, 0.15)",
    }

    t_min = target_df["time"].min()
    t_max = target_df["time"].max()

    if vt1_onset_t is not None:
        fig.add_vrect(x0=t_min, x1=vt1_onset_t, fillcolor=zone_colors["z1"], layer="below", line_width=0)

    if vt1_onset_t and vt1_steady_t:
        fig.add_vrect(x0=vt1_onset_t, x1=vt1_steady_t, fillcolor=zone_colors["z2"], layer="below", line_width=0)

    if vt1_steady_t and rcp_onset_t:
        fig.add_vrect(x0=vt1_steady_t, x1=rcp_onset_t, fillcolor=zone_colors["z3"], layer="below", line_width=0)

    if rcp_onset_t:
        fig.add_vrect(x0=rcp_onset_t, x1=t_max, fillcolor=zone_colors["z4"], layer="below", line_width=0)

    is_vt1_steady_interpolated = cpet_result.get("vt1_steady_is_interpolated", False)

    markers = [
        ("vt1_onset", vt1_onset_w, vt1_onset_t, "#ffa15a", "VT1_onset", "vt1_hr", 1.0, "top"),
        ("vt1_steady", vt1_steady_w, vt1_steady_t, "#00cc96", "VT1_steady", "vt1_steady_hr", 0.85, "top"),
        ("rcp_onset", rcp_onset_w, rcp_onset_t, "#ef553b", "RCP_onset", "vt2_hr", 1.0, "bottom"),
        ("rcp_steady", rcp_steady_w, rcp_steady_t, "#ab63fa", "RCP_steady", "rcp_steady_hr", 0.85, "bottom"),
    ]

    for name, power, time, color, label, hr_key, y_pos, anchor in markers:
        if time is None or power is None:
            continue

        hr = get_hr_at_power(power, hr_key)
        hr_str = f"{int(hr)}" if hr else "--"

        if name == "vt1_steady" and is_vt1_steady_interpolated:
            fig.add_vline(x=time, line=dict(color="#666", width=2, dash="dot"), layer="above")
            fig.add_annotation(
                x=time,
                y=y_pos,
                yref="paper",
                text=f"<b>{label}</b><br>(interp.)<br>{int(power)}W",
                showarrow=False,
                font=dict(color="white", size=8),
                bgcolor="rgba(100, 100, 100, 0.7)",
                bordercolor="#666",
                borderwidth=1,
                borderpad=3,
                align="center",
                xanchor="center",
                yanchor=anchor,
            )
        else:
            fig.add_vline(x=time, line=dict(color=color, width=2, dash="dash"), layer="above")
            yshift = -35 if anchor == "bottom" else 0
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_annotation(
                x=time,
                y=y_pos,
                yref="paper",
                text=f"<b>{label}</b><br>{int(power)}W @ {hr_str}",
                showarrow=False,
                font=dict(color="white", size=9),
                bgcolor=f"rgba({r}, {g}, {b}, 0.85)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                align="center",
                xanchor="center",
                yanchor=anchor,
                yshift=yshift,
            )

    fig.update_layout(
        title="Dynamika Wentylacji z Progami VT1/VT2",
        xaxis_title="Czas",
        yaxis=dict(title=dict(text="Wentylacja (L/min)", font=dict(color="#ffa15a"))),
        yaxis2=dict(
            title=dict(text="Moc (W)", font=dict(color="#1f77b4")),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
