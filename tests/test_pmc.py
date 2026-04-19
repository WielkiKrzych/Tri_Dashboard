"""Tests for PMC (Performance Management Chart) calculations."""

import numpy as np
import pytest

from modules.calculations.pmc import (
    PMCDataPoint,
    get_form_interpretation,
    predict_future_pmc,
)


class TestFormInterpretation:
    """Tests for get_form_interpretation."""

    def test_positive_tsb_fresh(self):
        assert "Świeży" in get_form_interpretation(30)

    def test_positive_tsb_ready(self):
        result = get_form_interpretation(15)
        assert "Gotowy" in result

    def test_near_zero_optimal(self):
        result = get_form_interpretation(-5)
        assert "Optymalne" in result

    def test_negative_tsb_tired(self):
        result = get_form_interpretation(-20)
        assert "Zmęczony" in result

    def test_very_negative_overtrained(self):
        result = get_form_interpretation(-35)
        assert "Przepracowany" in result

    def test_boundary_26(self):
        assert "Świeży" in get_form_interpretation(26)

    def test_boundary_6(self):
        result = get_form_interpretation(6)
        assert "Gotowy" in result

    def test_boundary_minus_9(self):
        result = get_form_interpretation(-9)
        assert "Optymalne" in result

    def test_boundary_minus_29(self):
        result = get_form_interpretation(-29)
        assert "Zmęczony" in result


class TestPredictFuturePMC:
    """Tests for predict_future_pmc."""

    def test_predict_known_values(self):
        """CTL/ATL should converge correctly with constant TSS input."""
        ctl = 50.0
        atl = 50.0
        # Constant TSS = 50 → CTL and ATL should stay near 50
        planned = [50.0] * 14
        result = predict_future_pmc(ctl, atl, planned)

        assert len(result) == 14
        assert all(isinstance(p, PMCDataPoint) for p in result)

        # With constant TSS equal to current CTL/ATL, TSB should stay near 0
        tsb_values = [p.tsb for p in result]
        assert all(abs(t) < 5 for t in tsb_values)

    def test_predict_high_tss_raises_atl(self):
        """High planned TSS should increase ATL faster than CTL."""
        result = predict_future_pmc(50.0, 50.0, [150.0] * 7)

        # ATL should rise quickly (7-day EWMA)
        assert result[-1].atl > 60
        # CTL should rise slowly (42-day EWMA)
        assert result[-1].ctl < result[-1].atl

    def test_predict_rest_lowers_both(self):
        """Zero TSS should lower both CTL and ATL."""
        result = predict_future_pmc(60.0, 40.0, [0.0] * 14)

        # ATL should drop quickly
        assert result[-1].atl < 20
        # CTL should drop slowly
        assert result[-1].ctl > result[-1].atl

    def test_predict_empty_tss(self):
        """Empty planned TSS should return empty list."""
        result = predict_future_pmc(50.0, 50.0, [])
        assert result == []

    def test_predict_custom_dates(self):
        """Should accept custom date strings."""
        dates = [f"2026-01-{d:02d}" for d in range(1, 8)]
        result = predict_future_pmc(50.0, 50.0, [50.0] * 7, dates)

        assert len(result) == 7
        assert result[0].date == "2026-01-01"
        assert result[-1].date == "2026-01-07"

    def test_form_status_populated(self):
        """Each prediction should have form_status."""
        result = predict_future_pmc(50.0, 50.0, [50.0] * 5)
        assert all(p.form_status for p in result)


class TestPMCDataPoint:
    """Tests for PMCDataPoint dataclass."""

    def test_creation(self):
        p = PMCDataPoint(
            date="2026-01-01",
            tss=50.0,
            atl=40.0,
            ctl=50.0,
            tsb=10.0,
            form_status="🟡 Gotowy",
        )
        assert p.date == "2026-01-01"
        assert p.tsb == 10.0
