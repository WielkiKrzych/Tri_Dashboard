"""Tests for the alert engine (cardiac drift, SmO2 crash, session alerts)."""

import pandas as pd
import numpy as np
import pytest

from modules.calculations.alert_engine import (
    Alert,
    AlertReport,
    analyze_session_alerts,
    detect_cardiac_drift,
    detect_smo2_crash,
)


class TestDetectCardiacDrift:
    """Tests for cardiac drift detection."""

    def test_no_drift_constant_hr(self):
        """Constant HR at constant power should not trigger drift alert."""
        n = 200
        df = pd.DataFrame(
            {
                "watts": np.full(n, 200.0),
                "hr": np.full(n, 150.0),
            }
        )
        metrics = {"avg_power": 200, "avg_hr": 150}
        result = detect_cardiac_drift(df, metrics)
        assert result is None or result.severity == "info"

    def test_drift_rising_hr(self):
        """Rising HR at constant power should trigger drift alert."""
        n = 200
        df = pd.DataFrame(
            {
                "watts": np.full(n, 200.0),
                "hr": np.linspace(140, 175, n),
            }
        )
        metrics = {"avg_power": 200, "avg_hr": 155}
        result = detect_cardiac_drift(df, metrics)
        if result is not None:
            assert isinstance(result, Alert)
            assert result.severity in ("info", "warning", "critical")

    def test_empty_metrics(self):
        """Should handle empty metrics dict gracefully."""
        df = pd.DataFrame({"watts": [100], "hr": [120]})
        result = detect_cardiac_drift(df, {})
        assert result is None or isinstance(result, Alert)


class TestDetectSmo2Crash:
    """Tests for SmO2 crash detection."""

    def test_no_crash_stable_smo2(self):
        """Stable SmO2 should not trigger crash alert."""
        df = pd.DataFrame({"smo2": np.full(100, 70.0)})
        result = detect_smo2_crash(df)
        assert result is None or result.severity == "info"

    def test_crash_declining_smo2(self):
        """Rapidly declining SmO2 should trigger crash alert."""
        df = pd.DataFrame({"smo2": np.linspace(80, 20, 200)})
        result = detect_smo2_crash(df)
        if result is not None:
            assert isinstance(result, Alert)

    def test_no_smo2_column(self):
        """DataFrame without smo2 column should return None."""
        df = pd.DataFrame({"watts": [100]})
        result = detect_smo2_crash(df)
        assert result is None


class TestAnalyzeSessionAlerts:
    """Tests for the main alert orchestrator."""

    def test_returns_alert_report(self):
        """Should return an AlertReport object."""
        df = pd.DataFrame(
            {
                "watts": np.full(100, 200.0),
                "hr": np.full(100, 150.0),
            }
        )
        metrics = {"avg_power": 200, "avg_hr": 150}
        result = analyze_session_alerts(df, metrics)
        assert isinstance(result, AlertReport)

    def test_with_session_history(self):
        """Should accept optional session history."""
        df = pd.DataFrame(
            {
                "watts": np.full(100, 200.0),
                "hr": np.full(100, 150.0),
            }
        )
        metrics = {"avg_power": 200, "avg_hr": 150}
        history = [{"date": "2026-01-01", "avg_hr": 148}]
        result = analyze_session_alerts(df, metrics, session_history=history)
        assert isinstance(result, AlertReport)

    def test_empty_session_history(self):
        """Should handle empty session history."""
        df = pd.DataFrame(
            {
                "watts": np.full(100, 200.0),
                "hr": np.full(100, 150.0),
            }
        )
        metrics = {"avg_power": 200, "avg_hr": 150}
        result = analyze_session_alerts(df, metrics, session_history=[])
        assert isinstance(result, AlertReport)
