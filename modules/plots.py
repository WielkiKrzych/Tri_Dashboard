"""
Shared Plotly chart helpers.

Applies consistent dark-theme styling and common layout settings
to all charts rendered across the dashboard.
"""
import plotly.graph_objects as go

# ── Chart height constants ──────────────────────────────────────────────────
CHART_HEIGHT_MAIN = 550      # primary timeline charts (Power, HR, SmO2, etc.)
CHART_HEIGHT_SUB = 380       # secondary charts (scatter, trend, model)
CHART_HEIGHT_SMALL = 280     # gauges and small summary charts
CHART_HEIGHT_HEATMAP = 420   # biomech heatmaps
CHART_HEIGHT_RADAR = 460     # radar / polar charts

# ── Global Plotly config (passed to st.plotly_chart(config=...)) ────────────
CHART_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["sendDataToCloud", "select2d", "lasso2d"],
    "displayModeBar": "hover",
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "tri_dashboard_chart",
        "height": 600,
        "width": 1400,
        "scale": 2,
    },
}


def apply_chart_style(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=title,
            font=dict(family="Rajdhani", size=24, color="#f0f6fc")
        ) if title else None,
        font=dict(family="Inter", size=12, color="#c9d1d9"),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    return fig


def add_stats_to_legend(fig, stats_list):
    for stat in stats_list:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(color='rgba(0,0,0,0)'),
            name=stat, hoverinfo='none'
        ))
