"""Tests for canonical_values.py and conflicts.py modules."""

import pytest
import pandas as pd
import numpy as np


# =========================================================================
# canonical_values
# =========================================================================

class TestCanonicalValues:
    def test_metric_source_enum(self):
        from modules.canonical_values import MetricSource
        assert MetricSource.MANUAL.value == "manual"
        assert MetricSource.AUTO.value == "auto"
        assert MetricSource.MISSING.value == "missing"

    def test_resolved_metric_to_dict(self):
        from modules.canonical_values import ResolvedMetric, MetricSource
        m = ResolvedMetric(value=280, source=MetricSource.MANUAL, confidence=1.0, name="cp")
        d = m.to_dict()
        assert d["value"] == 280
        assert d["source"] == "manual"
        assert d["confidence"] == 1.0

    def test_resolved_metric_is_valid(self):
        from modules.canonical_values import ResolvedMetric, MetricSource
        m = ResolvedMetric(value=280, source=MetricSource.MANUAL, confidence=1.0)
        assert m.is_valid()
        m2 = ResolvedMetric(value=None, source=MetricSource.MISSING, confidence=0.0)
        assert not m2.is_valid()

    def test_resolve_metric_manual_priority(self):
        from modules.canonical_values import resolve_metric, MetricSource
        result = resolve_metric("cp", manual=300, auto=280)
        assert result.value == 300
        assert result.source == MetricSource.MANUAL
        assert result.confidence == 1.0

    def test_resolve_metric_auto_fallback(self):
        from modules.canonical_values import resolve_metric, MetricSource
        result = resolve_metric("cp", manual=None, auto=280)
        assert result.value == 280
        assert result.source == MetricSource.AUTO

    def test_resolve_metric_missing(self):
        from modules.canonical_values import resolve_metric, MetricSource
        result = resolve_metric("cp", manual=None, auto=None)
        assert result.value is None
        assert result.source == MetricSource.MISSING
        assert result.confidence == 0.0

    def test_resolve_metric_zero_treated_as_missing(self):
        from modules.canonical_values import resolve_metric, MetricSource
        result = resolve_metric("cp", manual=0, auto=280)
        assert result.value == 280
        assert result.source == MetricSource.AUTO

    def test_resolve_all_thresholds(self):
        from modules.canonical_values import resolve_all_thresholds
        manual = {"manual_vt1_watts": 200}
        auto = {"vt1": 190, "vt2": 280, "cp": 260}
        result = resolve_all_thresholds(manual, auto)
        assert "vt1" in result
        assert result["vt1"].value == 200  # Manual override
        assert result["vt2"].value == 280  # Auto fallback

    def test_build_data_policy(self):
        from modules.canonical_values import build_data_policy
        from modules.canonical_values import resolve_all_thresholds
        manual = {}
        auto = {"vt1": 200, "vt2": 300}
        resolved = resolve_all_thresholds(manual, auto)
        policy = build_data_policy(resolved)
        assert isinstance(policy, dict)

    def test_log_resolution(self):
        from modules.canonical_values import log_resolution, resolve_all_thresholds
        manual = {"manual_vt1_watts": 200}
        auto = {"vt1": 190}
        resolved = resolve_all_thresholds(manual, auto)
        # Should not raise
        log_resolution(resolved)


# =========================================================================
# conflicts
# =========================================================================

class TestConflicts:
    def test_conflict_descriptions_complete(self):
        from modules.calculations.conflicts import CONFLICT_DESCRIPTIONS
        from models.results import ConflictType
        # Every ConflictType should have a description entry
        for ct in ConflictType:
            assert ct in CONFLICT_DESCRIPTIONS, f"Missing description for {ct}"

    def test_detect_conflicts_none_inputs(self):
        from modules.calculations.conflicts import detect_conflicts
        report = detect_conflicts(vt_result=None, smo2_result=None)
        assert report is not None
        assert len(report.conflicts) == 0

    def test_detect_conflicts_with_df(self):
        from modules.calculations.conflicts import detect_conflicts
        n = 600
        np.random.seed(42)
        df = pd.DataFrame({
            "watts": np.linspace(100, 350, n),
            "hr": 100 + np.linspace(0, 80, n),
            "time": np.arange(n, dtype=float),
        })
        report = detect_conflicts(vt_result=None, smo2_result=None, df=df)
        assert report is not None
        # May or may not detect cardiac drift etc.
        assert isinstance(report.conflicts, list)
