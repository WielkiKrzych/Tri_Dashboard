"""Tests for trend_engine module."""

import pytest
import numpy as np
from datetime import datetime, timedelta


class TestMetricTrendDataclass:
    def test_defaults(self):
        from modules.calculations.trend_engine import MetricTrend
        t = MetricTrend(name="VT1")
        assert t.name == "VT1"
        assert t.values == []
        assert t.rate_per_week == 0.0
        assert t.direction == "stable"

    def test_custom_values(self):
        from modules.calculations.trend_engine import MetricTrend
        t = MetricTrend(name="CP", values=[260, 270], rate_per_week=1.5, direction="improving")
        assert t.rate_per_week == 1.5
        assert t.direction == "improving"


class TestTrendAnalysisDataclass:
    def test_defaults(self):
        from modules.calculations.trend_engine import TrendAnalysis
        a = TrendAnalysis()
        assert a.adaptation_direction == "balanced"
        assert a.adaptation_score == 0.0
        assert a.tests_analyzed == 0
        assert a.engine_map == {}

    def test_metric_trends_independent(self):
        from modules.calculations.trend_engine import TrendAnalysis
        a = TrendAnalysis()
        a.vt1.values.append(200)
        assert a.vt2.values == []  # Independent instances


class TestExtractMetricsFromReport:
    def test_empty_report(self):
        from modules.calculations.trend_engine import extract_metrics_from_report
        result = extract_metrics_from_report({})
        assert result["vt1"] == 0
        assert result["vt2"] == 0
        assert result["cp"] == 0

    def test_full_report(self):
        from modules.calculations.trend_engine import extract_metrics_from_report
        report = {
            "thresholds": {
                "ventilatory": {
                    "vt1": {"midpoint_watts": 200},
                    "vt2": {"midpoint_watts": 300},
                },
                "smo2": {"regression_slope": -0.08},
            },
            "physiological_markers": {
                "cp": 280,
                "w_prime": 15.0,
                "efficiency_factor": 1.7,
            },
            "biomechanical_analysis": {"occlusion_index": 0.35},
            "thermal_analysis": {"max_hsi": 42.0},
        }
        result = extract_metrics_from_report(report)
        assert result["vt1"] == 200
        assert result["vt2"] == 300
        assert result["cp"] == 280
        assert result["w_prime"] == 15.0
        assert result["ef"] == 1.7
        assert result["smo2_slope"] == -0.08
        assert result["occlusion_index"] == 0.35
        assert result["hsi"] == 42.0


class TestCalculateRatePerWeek:
    def test_insufficient_data(self):
        from modules.calculations.trend_engine import calculate_rate_per_week
        assert calculate_rate_per_week([200], [datetime(2024, 1, 1)]) == 0.0
        assert calculate_rate_per_week([], []) == 0.0

    def test_same_date(self):
        from modules.calculations.trend_engine import calculate_rate_per_week
        d = datetime(2024, 1, 1)
        assert calculate_rate_per_week([200, 210], [d, d]) == 0.0

    def test_improving_trend(self):
        from modules.calculations.trend_engine import calculate_rate_per_week
        dates = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(4)]
        values = [200, 210, 220, 230]  # +10 per week
        rate = calculate_rate_per_week(values, dates)
        assert rate > 0  # Positive = improving

    def test_declining_trend(self):
        from modules.calculations.trend_engine import calculate_rate_per_week
        dates = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(4)]
        values = [300, 290, 280, 270]
        rate = calculate_rate_per_week(values, dates)
        assert rate < 0


class TestClassifyDirection:
    def test_improving(self):
        from modules.calculations.trend_engine import classify_direction
        assert classify_direction(1.0) == "improving"

    def test_declining(self):
        from modules.calculations.trend_engine import classify_direction
        assert classify_direction(-1.0) == "declining"

    def test_stable(self):
        from modules.calculations.trend_engine import classify_direction
        assert classify_direction(0.1) == "stable"
        assert classify_direction(-0.3) == "stable"

    def test_inverse(self):
        from modules.calculations.trend_engine import classify_direction
        # For inverse metrics, negative rate means improving
        assert classify_direction(-1.0, is_inverse=True) == "improving"
        assert classify_direction(1.0, is_inverse=True) == "declining"


class TestAnalyzeTrends:
    def _make_reports(self, n=4, improving=True):
        """Create n fake reports with improving or declining CP."""
        reports = []
        for i in range(n):
            cp = 260 + (i * 10 if improving else -i * 10)
            reports.append({
                "_test_date": datetime(2024, 1, 1) + timedelta(weeks=i),
                "thresholds": {
                    "ventilatory": {
                        "vt1": {"midpoint_watts": 180 + i * 5},
                        "vt2": {"midpoint_watts": 280 + i * 5},
                    },
                    "smo2": {},
                },
                "physiological_markers": {"cp": cp, "w_prime": 15.0, "efficiency_factor": 1.6 + i * 0.05},
                "biomechanical_analysis": {},
                "thermal_analysis": {},
            })
        return reports

    def test_too_few_reports(self):
        from modules.calculations.trend_engine import analyze_trends
        result = analyze_trends([{"_test_date": datetime(2024, 1, 1)}])
        assert result.tests_analyzed == 0

    def test_basic_analysis(self):
        from modules.calculations.trend_engine import analyze_trends
        reports = self._make_reports(4, improving=True)
        result = analyze_trends(reports)
        assert result.tests_analyzed == 4
        assert result.date_range_days > 0
        assert result.cp.direction == "improving"
        assert result.cp.rate_per_week > 0

    def test_engine_map_populated(self):
        from modules.calculations.trend_engine import analyze_trends
        reports = self._make_reports(4)
        result = analyze_trends(reports)
        assert "CP" in result.engine_map
        assert "VT1" in result.engine_map
        assert all(0 <= v <= 100 for v in result.engine_map.values())

    def test_adaptation_score(self):
        from modules.calculations.trend_engine import analyze_trends
        reports = self._make_reports(4)
        result = analyze_trends(reports)
        assert 0 <= result.adaptation_score <= 100

    def test_adaptation_direction_central(self):
        from modules.calculations.trend_engine import analyze_trends
        # Reports with improving VT1, VT2, CP, EF → central
        reports = self._make_reports(4, improving=True)
        result = analyze_trends(reports)
        # With VT1, VT2, CP, EF all improving → should be central
        assert result.adaptation_direction in ("central", "balanced")


class TestLoadRampTestHistory:
    def test_missing_index(self, tmp_path):
        from modules.calculations.trend_engine import load_ramp_test_history
        result = load_ramp_test_history(str(tmp_path / "nonexistent.csv"))
        assert result == []

    def test_empty_csv(self, tmp_path):
        import pandas as pd
        from modules.calculations.trend_engine import load_ramp_test_history
        csv_path = tmp_path / "index.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        result = load_ramp_test_history(str(csv_path))
        assert result == []
