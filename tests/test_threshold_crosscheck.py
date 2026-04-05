"""
Tests for modules.domain.threshold_crosscheck.

Locks in the agreement-band contract from docs/DOMAIN_MODEL.md §6.
"""

from __future__ import annotations

import pytest

from modules.domain.threshold_crosscheck import (
    AgreementLevel,
    CrosscheckVerdict,
    crosscheck_threshold,
)


class TestAgreementBands:
    """DOMAIN_MODEL.md §6 — band boundaries at 10 / 20 / 30 W spread."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (250.0, 255.0, AgreementLevel.STRONG),   # spread 5
            (250.0, 260.0, AgreementLevel.STRONG),   # spread 10 (boundary)
            (250.0, 261.0, AgreementLevel.MODERATE), # spread 11
            (250.0, 270.0, AgreementLevel.MODERATE), # spread 20 (boundary)
            (250.0, 271.0, AgreementLevel.WEAK),     # spread 21
            (250.0, 280.0, AgreementLevel.WEAK),     # spread 30 (boundary)
            (250.0, 281.0, AgreementLevel.CONFLICT), # spread 31
            (250.0, 320.0, AgreementLevel.CONFLICT), # spread 70
        ],
    )
    def test_two_source_classification(self, a, b, expected):
        verdict = crosscheck_threshold("VT2", {"src_a": a, "src_b": b})
        assert verdict.level is expected, (
            f"|{a} - {b}| = {abs(a-b)} W → expected {expected}, got {verdict.level}"
        )


class TestInsufficientSources:
    def test_zero_sources(self):
        verdict = crosscheck_threshold("VT2", {})
        assert verdict.level is AgreementLevel.INSUFFICIENT
        assert verdict.spread_watts is None
        assert verdict.consensus_watts is None

    def test_one_source(self):
        verdict = crosscheck_threshold("VT2", {"ventilation": 280.0})
        assert verdict.level is AgreementLevel.INSUFFICIENT
        assert verdict.consensus_watts == 280.0

    def test_none_values_filtered(self):
        verdict = crosscheck_threshold(
            "VT2", {"ventilation": 280.0, "smo2": None, "hr": None}
        )
        assert verdict.level is AgreementLevel.INSUFFICIENT

    def test_nan_values_filtered(self):
        verdict = crosscheck_threshold(
            "VT2", {"ventilation": 280.0, "smo2": float("nan")}
        )
        assert verdict.level is AgreementLevel.INSUFFICIENT


class TestConsensusAndSpread:
    def test_median_of_three_sources(self):
        verdict = crosscheck_threshold(
            "VT2", {"a": 270.0, "b": 280.0, "c": 290.0}
        )
        assert verdict.consensus_watts == 280.0
        assert verdict.spread_watts == 20.0

    def test_median_of_even_sources(self):
        verdict = crosscheck_threshold(
            "VT2", {"a": 260.0, "b": 270.0, "c": 280.0, "d": 290.0}
        )
        assert verdict.consensus_watts == 275.0
        assert verdict.spread_watts == 30.0

    def test_spread_uses_max_minus_min_not_stddev(self):
        verdict = crosscheck_threshold(
            "VT2", {"a": 250.0, "b": 255.0, "c": 260.0, "outlier": 290.0}
        )
        assert verdict.spread_watts == 40.0
        assert verdict.level is AgreementLevel.CONFLICT


class TestPairwiseDeltas:
    def test_largest_delta_first(self):
        verdict = crosscheck_threshold(
            "VT2", {"a": 250.0, "b": 255.0, "c": 290.0}
        )
        # Pairs: (a,b)=5, (a,c)=40, (b,c)=35 → sorted desc: 40, 35, 5
        assert len(verdict.deltas) == 3
        assert verdict.deltas[0][2] == 40.0
        assert verdict.deltas[-1][2] == 5.0

    def test_delta_count_equals_n_choose_2(self):
        verdict = crosscheck_threshold(
            "VT2", {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
        )
        assert len(verdict.deltas) == 6  # C(4, 2)


class TestVerdictProperties:
    def test_is_conflict_true_only_for_conflict(self):
        conflict = crosscheck_threshold("VT2", {"a": 200.0, "b": 300.0})
        strong = crosscheck_threshold("VT2", {"a": 250.0, "b": 253.0})
        assert conflict.is_conflict is True
        assert strong.is_conflict is False

    def test_is_trustworthy_for_strong_and_moderate(self):
        strong = crosscheck_threshold("VT2", {"a": 250.0, "b": 253.0})
        moderate = crosscheck_threshold("VT2", {"a": 250.0, "b": 265.0})
        weak = crosscheck_threshold("VT2", {"a": 250.0, "b": 275.0})
        assert strong.is_trustworthy is True
        assert moderate.is_trustworthy is True
        assert weak.is_trustworthy is False


class TestMessagePL:
    def test_strong_agreement_has_checkmark(self):
        verdict = crosscheck_threshold("VT2", {"vent": 250.0, "smo2": 253.0})
        assert "✅" in verdict.message_pl
        assert "spójne" in verdict.message_pl

    def test_conflict_lists_pairwise_deltas(self):
        verdict = crosscheck_threshold(
            "VT2", {"vent": 250.0, "smo2": 230.0, "hr": 300.0}
        )
        assert "KONFLIKT" in verdict.message_pl
        assert "vent" in verdict.message_pl
        assert "hr" in verdict.message_pl

    def test_conflict_message_warns_against_averaging(self):
        verdict = crosscheck_threshold("VT2", {"a": 200.0, "b": 300.0})
        assert "Nie uśredniaj" in verdict.message_pl

    def test_insufficient_message(self):
        verdict = crosscheck_threshold("VT2", {"vent": 280.0})
        assert "za mało źródeł" in verdict.message_pl

    def test_metric_name_appears_in_message(self):
        verdict = crosscheck_threshold("VT1", {"a": 180.0, "b": 185.0})
        assert "VT1" in verdict.message_pl


class TestVerdictIsImmutable:
    def test_verdict_is_frozen_dataclass(self):
        verdict = crosscheck_threshold("VT2", {"a": 250.0, "b": 253.0})
        with pytest.raises((AttributeError, Exception)):
            verdict.level = AgreementLevel.CONFLICT  # type: ignore[misc]
