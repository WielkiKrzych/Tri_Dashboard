"""Tests for remaining 0% coverage modules: genetics, reports, chart_exporters, environment, async_utils."""

import pytest
import numpy as np
import pandas as pd
from io import BytesIO


# =========================================================================
# genetics
# =========================================================================

class TestGenetics:
    def test_genetic_profile_default(self):
        from modules.genetics import GeneticProfile
        profile = GeneticProfile()
        assert profile.endurance_score == 50
        assert profile.power_score == 50

    def test_genetic_profile_endurance(self):
        from modules.genetics import GeneticProfile
        profile = GeneticProfile(actn3="XX", ace="II", ppargc1a="AA")
        assert profile.endurance_score > 70
        assert profile.athlete_type == "🏃 Wytrzymałościowiec"

    def test_genetic_profile_power(self):
        from modules.genetics import GeneticProfile
        profile = GeneticProfile(actn3="RR", ace="DD")
        assert profile.power_score > 70
        assert profile.athlete_type == "💪 Sprinter/Siłowiec"

    def test_genetic_profile_balanced(self):
        from modules.genetics import GeneticProfile
        profile = GeneticProfile(actn3="RX", ace="ID")
        assert "Wszechstronny" in profile.athlete_type

    def test_gene_variant_enum(self):
        from modules.genetics import GeneVariant
        assert GeneVariant.ACTN3_RR.value == "RR"
        assert GeneVariant.ACE_II.value == "II"

    def test_genetic_analyzer_parse_23andme(self):
        from modules.genetics import GeneticAnalyzer
        analyzer = GeneticAnalyzer()
        raw = "# header\nrs1815739\t1\t100\tCC\nrs1799752\t1\t200\tDD\nrs8192678\t1\t300\tGA\n"
        profile = analyzer.parse_23andme(raw)
        assert profile.actn3 == "RR"
        assert profile.ace == "DD"
        assert profile.ppargc1a == "GA"

    def test_genetic_analyzer_empty_data(self):
        from modules.genetics import GeneticAnalyzer
        analyzer = GeneticAnalyzer()
        profile = analyzer.parse_23andme("# header only\n")
        assert profile.actn3 is None

    def test_genetic_analyzer_get_recommendations(self):
        from modules.genetics import GeneticAnalyzer, GeneticProfile
        analyzer = GeneticAnalyzer()
        profile = GeneticProfile(actn3="XX", ace="II")
        recs = analyzer.get_recommendations(profile)
        assert isinstance(recs, list)
        assert len(recs) > 0


# =========================================================================
# reports (DOCX generation)
# =========================================================================

class TestReports:
    def test_calculate_fallback_metrics(self):
        from modules.reports import _calculate_fallback_metrics
        df = pd.DataFrame({
            "watts": np.random.randint(100, 300, 100),
            "core_temperature": 37.0 + np.random.normal(0, 0.5, 100),
        })
        metrics = {"np": 0, "work_kj": 0, "max_core": 0}
        result = _calculate_fallback_metrics(metrics, df)
        assert result["np"] > 0
        assert result["work_kj"] > 0
        assert result["max_core"] > 37

    def test_calculate_fallback_metrics_already_set(self):
        from modules.reports import _calculate_fallback_metrics
        df = pd.DataFrame({"watts": [200, 250, 300]})
        metrics = {"np": 230, "work_kj": 50}
        result = _calculate_fallback_metrics(metrics, df)
        assert result["np"] == 230  # Not overwritten

    def test_generate_docx_report_importable(self):
        from modules.reports import generate_docx_report
        assert callable(generate_docx_report)

    def test_export_all_charts_importable(self):
        from modules.reports import export_all_charts_as_png
        assert callable(export_all_charts_as_png)


# =========================================================================
# chart_exporters
# =========================================================================

class TestChartExporters:
    def test_chart_context_creation(self):
        from modules.chart_exporters import ChartContext
        df = pd.DataFrame({
            "watts": [200, 250, 300],
            "time": [0, 1, 2],
            "heartrate": [120, 130, 140],
        })
        ctx = ChartContext(
            df_plot=df, df_plot_resampled=df,
            rider_weight=75, cp_input=260,
            vt1_watts=200, vt2_watts=280,
            metrics={"avg_watts": 250}
        )
        assert ctx.rider_weight == 75
        assert ctx.layout_args is not None

    def test_chart_registry_exists(self):
        from modules.chart_exporters import CHART_REGISTRY
        assert isinstance(CHART_REGISTRY, list)
        assert len(CHART_REGISTRY) > 0

    def test_chart_exporters_have_required_methods(self):
        from modules.chart_exporters import CHART_REGISTRY
        for exporter in CHART_REGISTRY:
            assert hasattr(exporter, 'filename')
            assert hasattr(exporter, 'can_export')
            assert hasattr(exporter, 'create_figure')

    def test_chart_exporters_can_export_check(self):
        from modules.chart_exporters import CHART_REGISTRY, ChartContext
        n = 200
        df = pd.DataFrame({
            "watts": np.linspace(100, 350, n),
            "watts_smooth": np.linspace(100, 350, n),
            "heartrate_smooth": 100 + np.linspace(0, 60, n),
            "time": np.arange(n, dtype=float),
        })
        ctx = ChartContext(
            df_plot=df, df_plot_resampled=df,
            rider_weight=75, cp_input=260,
            vt1_watts=200, vt2_watts=280,
            metrics={"avg_watts": 200}
        )
        can_export_count = sum(1 for exp in CHART_REGISTRY if exp.can_export(ctx))
        assert can_export_count > 0

    def test_power_chart_exporter_creates_figure(self):
        from modules.chart_exporters import CHART_REGISTRY, ChartContext
        import plotly.graph_objects as go
        n = 200
        time_sec = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "watts": np.linspace(100, 350, n),
            "watts_smooth": np.linspace(100, 350, n),
            "heartrate_smooth": 100 + np.linspace(0, 60, n),
            "time": time_sec,
            "time_min": time_sec / 60,
            "cadence": np.full(n, 90.0),
        })
        ctx = ChartContext(
            df_plot=df, df_plot_resampled=df,
            rider_weight=75, cp_input=260,
            vt1_watts=200, vt2_watts=280,
            metrics={"avg_watts": 225, "max_watts": 350, "np": 240}
        )
        for exporter in CHART_REGISTRY:
            if exporter.can_export(ctx):
                fig = exporter.create_figure(ctx)
                assert isinstance(fig, go.Figure)
                break


# =========================================================================
# environment
# =========================================================================

class TestEnvironment:
    def test_weather_data_creation(self):
        from modules.environment import WeatherData
        wd = WeatherData(
            temperature=25.0, humidity=60, wind_speed=10.0,
            feels_like=28.0, description="Sunny", location="Warsaw", timestamp="2024-01-01"
        )
        assert wd.temperature == 25.0

    def test_environment_service_importable(self):
        from modules.environment import EnvironmentService
        svc = EnvironmentService()
        assert svc is not None

    def test_calculate_tss_correction(self):
        from modules.environment import EnvironmentService, WeatherData
        svc = EnvironmentService()
        wd = WeatherData(
            temperature=35.0, humidity=80, wind_speed=5.0,
            feels_like=40.0, description="Hot", location="Warsaw", timestamp="2024-01-01"
        )
        result = svc.calculate_tss_correction(wd)
        # Returns (correction_factor, description)
        assert isinstance(result, tuple)
        assert result[0] >= 0  # Correction factor

    def test_get_mock_conditions(self):
        from modules.environment import EnvironmentService
        from datetime import datetime
        svc = EnvironmentService()
        conditions = svc.get_conditions(datetime.now(), 52.23, 21.01)
        assert conditions is not None
        assert conditions.temperature > -50  # Sane value


# =========================================================================
# async_utils
# =========================================================================

class TestAsyncUtils:
    def test_import(self):
        from modules.async_utils import run_in_thread, shutdown_executor
        assert callable(run_in_thread)
        assert callable(shutdown_executor)

    def test_run_in_thread_decorator(self):
        from modules.async_utils import run_in_thread

        @run_in_thread
        def add(a, b):
            return a + b

        # Decorated function returns a future
        assert callable(add)
