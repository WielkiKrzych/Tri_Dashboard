"""
Tests for modules.domain.data_quality.

Locks in the DataQuality band contract from docs/DOMAIN_MODEL.md §4.
"""

from __future__ import annotations

import math

import pytest

from modules.domain.data_quality import (
    DataQuality,
    format_band_badge_html,
    quality_band,
)


class TestQualityBandBoundaries:
    """DOMAIN_MODEL.md §4 band thresholds — lock in the boundary values."""

    @pytest.mark.parametrize(
        "confidence, expected",
        [
            (1.00, DataQuality.HIGH),
            (0.95, DataQuality.HIGH),
            (0.80, DataQuality.HIGH),
            (0.7999, DataQuality.CONDITIONAL),
            (0.65, DataQuality.CONDITIONAL),
            (0.60, DataQuality.CONDITIONAL),
            (0.5999, DataQuality.LOW),
            (0.45, DataQuality.LOW),
            (0.40, DataQuality.LOW),
            (0.3999, DataQuality.INVALID),
            (0.10, DataQuality.INVALID),
            (0.00, DataQuality.INVALID),
        ],
    )
    def test_bands_at_documented_thresholds(self, confidence, expected):
        assert quality_band(confidence) is expected


class TestQualityBandEdgeCases:
    def test_none_is_invalid(self):
        assert quality_band(None) is DataQuality.INVALID

    def test_nan_is_invalid(self):
        assert quality_band(float("nan")) is DataQuality.INVALID

    def test_above_one_clipped_to_high(self):
        assert quality_band(1.5) is DataQuality.HIGH

    def test_negative_clipped_to_invalid(self):
        assert quality_band(-0.3) is DataQuality.INVALID

    def test_non_numeric_is_invalid(self):
        assert quality_band("high") is DataQuality.INVALID  # type: ignore[arg-type]

    def test_int_input_accepted(self):
        assert quality_band(1) is DataQuality.HIGH
        assert quality_band(0) is DataQuality.INVALID


class TestDataQualityMetadata:
    def test_all_bands_have_presentation_metadata(self):
        for band in DataQuality:
            assert band.label_pl
            assert band.label_en
            assert band.color.startswith("#")
            assert band.short_caveat_pl
            assert isinstance(band.is_trustworthy, bool)

    def test_high_and_conditional_are_trustworthy(self):
        assert DataQuality.HIGH.is_trustworthy is True
        assert DataQuality.CONDITIONAL.is_trustworthy is True

    def test_low_and_invalid_are_not_trustworthy(self):
        assert DataQuality.LOW.is_trustworthy is False
        assert DataQuality.INVALID.is_trustworthy is False

    def test_colors_are_distinct(self):
        colors = {band.color for band in DataQuality}
        assert len(colors) == len(DataQuality), "Each band should have a unique color"


class TestFormatBandBadgeHtml:
    def test_badge_contains_band_label(self):
        html_out = format_band_badge_html(DataQuality.HIGH)
        assert DataQuality.HIGH.label_pl in html_out

    def test_badge_contains_band_color(self):
        html_out = format_band_badge_html(DataQuality.CONDITIONAL)
        assert DataQuality.CONDITIONAL.color in html_out

    def test_badge_prefix_is_html_escaped(self):
        html_out = format_band_badge_html(DataQuality.HIGH, prefix="<script>")
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out

    def test_badge_wraps_in_span(self):
        html_out = format_band_badge_html(DataQuality.LOW)
        assert html_out.startswith("<span")
        assert html_out.endswith("</span>")
