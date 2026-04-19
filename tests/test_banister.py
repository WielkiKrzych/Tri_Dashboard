"""Tests for Banister impulse-response model."""

import numpy as np
import pytest

from modules.calculations.banister import (
    BanisterModel,
    BanisterPrediction,
    TSSRecommendation,
    default_banister_model,
    predict_performance,
    optimize_peaking,
)
from datetime import datetime, timedelta


class TestDefaultBanisterModel:
    def test_default_parameters(self):
        model = default_banister_model()
        assert model.k1 == 1.0
        assert model.k2 == 2.0
        assert model.tau1 == 42.0
        assert model.tau2 == 7.0
        assert model.p0 == 0.0


class TestPredictPerformance:
    def test_training_increases_fitness(self):
        model = default_banister_model()
        tss = [80.0] * 28
        result = predict_performance(model, tss, days_ahead=0)
        assert len(result) == 28
        assert result[-1].predicted_performance > result[0].predicted_performance

    def test_rest_changes_performance(self):
        model = default_banister_model()
        tss = [80.0] * 28
        trained = predict_performance(model, tss, days_ahead=0)
        trained_perf = trained[-1].predicted_performance

        rest_result = predict_performance(model, tss, days_ahead=14)
        rest_perf = rest_result[-1].predicted_performance
        assert rest_perf != trained_perf

    def test_empty_tss_returns_empty(self):
        model = default_banister_model()
        assert predict_performance(model, []) == []

    def test_predictions_have_dates(self):
        model = default_banister_model()
        result = predict_performance(model, [50.0] * 7, days_ahead=7)
        assert len(result) == 14
        for p in result:
            assert p.date

    def test_ctl_atl_tsb_populated(self):
        model = default_banister_model()
        result = predict_performance(model, [50.0] * 14, days_ahead=7)
        for p in result:
            assert isinstance(p.ctl, float)
            assert isinstance(p.atl, float)
            assert isinstance(p.tsb, float)

    def test_performance_formula(self):
        model = BanisterModel(k1=1.0, k2=0.5, tau1=42.0, tau2=7.0, p0=100.0)
        result = predict_performance(model, [100.0], days_ahead=0)
        assert len(result) == 1
        assert result[0].predicted_performance > 100.0


class TestOptimizePeaking:
    def test_taper_plan_created(self):
        target = datetime.now() + timedelta(days=14)
        recs = optimize_peaking(60.0, 40.0, target, days_out=14)
        assert len(recs) == 14
        assert all(isinstance(r, TSSRecommendation) for r in recs)

    def test_tss_reduces_toward_race(self):
        target = datetime.now() + timedelta(days=14)
        recs = optimize_peaking(60.0, 40.0, target, days_out=14)
        first_week_avg = np.mean([r.recommended_tss for r in recs[:7]])
        second_week_avg = np.mean([r.recommended_tss for r in recs[7:]])
        assert second_week_avg <= first_week_avg

    def test_recommendations_have_notes(self):
        target = datetime.now() + timedelta(days=7)
        recs = optimize_peaking(60.0, 40.0, target, days_out=7)
        for r in recs:
            assert r.notes
            assert r.date
