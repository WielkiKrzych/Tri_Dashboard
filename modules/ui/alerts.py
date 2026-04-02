"""Alerts tab UI — renders alert report with summary, gauge, and cards."""

import streamlit as st

from modules.calculations.alert_engine import Alert, AlertReport, OvertrainingRiskIndex


def render_alerts_tab(report: AlertReport) -> None:
    """Render the alerts & red flags tab."""
    st.subheader("\U0001f514 Alerty & Red Flags")

    if report.has_critical:
        st.error(
            f"\U0001f6a8 {report.critical_count} krytycznych alertów | "
            f"\u26a0\ufe0f {report.warning_count} ostrze\u017ce\u0144"
        )
    elif report.warning_count > 0:
        st.warning(f"\u26a0\ufe0f {report.warning_count} ostrze\u017ce\u0144")
    else:
        st.success("\u2705 Brak krytycznych alert\u00f3w")

    if report.overtraining_index:
        _render_overtraining_gauge(report.overtraining_index)

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    sorted_alerts = sorted(report.alerts, key=lambda a: severity_order.get(a.severity, 3))
    for alert in sorted_alerts:
        _render_alert_card(alert)

    for flag in report.data_quality_flags:
        st.info(f"\u2139\ufe0f {flag}")


def _render_overtraining_gauge(idx: OvertrainingRiskIndex) -> None:
    """Render overtraining risk as a styled metric + progress bar."""
    level_pl = {
        "low": "Niskie",
        "moderate": "Umiarkowane",
        "high": "Wysokie",
        "critical": "Krytyczne",
    }

    st.markdown("### \U0001f3cb\ufe0f Ryzyko Przetrenowania")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Wynik", f"{idx.score:.0f}/100")
    with col2:
        st.progress(min(idx.score / 100, 1.0))
        st.caption(f"Poziom: **{level_pl.get(idx.risk_level, idx.risk_level)}**")

    if idx.components:
        with st.expander("Sk\u0142adowe indeksu"):
            for name, value in idx.components.items():
                st.write(f"\u2022 **{name}**: {value:.0f}%")
    if idx.interpretation:
        st.info(idx.interpretation)


def _render_alert_card(alert: Alert) -> None:
    """Render a single alert as an expandable card."""
    severity_method = {
        "critical": st.error,
        "warning": st.warning,
        "info": st.info,
    }
    method = severity_method.get(alert.severity, st.info)

    with st.expander(f"{alert.icon} {alert.title}", expanded=(alert.severity == "critical")):
        st.write(alert.message)
        if alert.value is not None and alert.threshold is not None:
            st.write(
                f"**Warto\u015b\u0107:** {alert.value:.1f} | **Pr\u00f3g:** {alert.threshold:.1f}"
            )
        if alert.recommendation:
            st.write(f"\U0001f4a1 **Zalecenie:** {alert.recommendation}")
        if alert.supporting_metrics:
            st.write("**Dodatkowe metryki:**")
            for k, v in alert.supporting_metrics.items():
                st.write(f"  \u2022 {k}: {v:.1f}")
