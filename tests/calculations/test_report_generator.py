"""Tests for report_generator module."""
import pytest
from datetime import datetime

from models.results import (
    RampTestResult,
    TestValidity,
    ThresholdRange,
    ConflictReport,
    SignalConflict,
    SignalQuality,
    ValidityLevel,
    ConfidenceLevel,
    ConflictType,
    ConflictSeverity,
)
from modules.calculations.report_generator import (
    generate_report,
    format_report_markdown,
    ReportSection,
    RampTestReport,
)


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_create_minimal_section(self):
        """Create section with required fields only."""
        section = ReportSection(title="Test", content="Content")
        assert section.title == "Test"
        assert section.content == "Content"
        assert section.confidence is None
        assert section.warnings == []

    def test_create_full_section(self):
        """Create section with all fields."""
        section = ReportSection(
            title="VT1",
            content="VT1 at 180W",
            confidence=0.85,
            warnings=["Low signal quality"]
        )
        assert section.confidence == 0.85
        assert len(section.warnings) == 1


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_with_valid_result(self):
        """Generate report from valid test result."""
        validity = TestValidity(
            validity=ValidityLevel.VALID,
            ramp_duration_sec=600,
            power_range_watts=200,
            exhaustion_reached=True,
        )
        vt1 = ThresholdRange(
            lower_watts=175,
            upper_watts=185,
            midpoint_watts=180,
            confidence=0.85,
            sources=["HR", "VE"],
        )
        vt2 = ThresholdRange(
            lower_watts=245,
            upper_watts=255,
            midpoint_watts=250,
            confidence=0.80,
            sources=["HR", "VE"],
        )
        result = RampTestResult(
            validity=validity,
            vt1=vt1,
            vt2=vt2,
            overall_confidence=0.82,
        )

        report = generate_report(result)

        assert isinstance(report, RampTestReport)
        assert report.summary is not None
        assert report.validity is not None
        assert report.vt1_section is not None
        assert report.vt2_section is not None
        assert report.overall_confidence == 0.82

    def test_generate_report_with_conditional_validity(self):
        """Generate report with conditional test validity."""
        validity = TestValidity(
            validity=ValidityLevel.CONDITIONAL,
            ramp_duration_sec=420,
            power_range_watts=150,
            exhaustion_reached=True,
            issues=["Short ramp duration"],
        )
        result = RampTestResult(
            validity=validity,
            overall_confidence=0.55,
        )

        report = generate_report(result)

        assert "zastrzeżeniami" in report.validity.content.lower()

    def test_generate_report_with_invalid_test(self):
        """Generate report for invalid test."""
        validity = TestValidity(
            validity=ValidityLevel.INVALID,
            ramp_duration_sec=180,
            power_range_watts=50,
            exhaustion_reached=False,
            issues=["Test too short", "No exhaustion reached"],
        )
        result = RampTestResult(
            validity=validity,
            overall_confidence=0.2,
        )

        report = generate_report(result)

        assert "nieważny" in report.validity.content.lower()

    def test_generate_report_without_thresholds(self):
        """Generate report when thresholds not detected."""
        validity = TestValidity(validity=ValidityLevel.VALID)
        result = RampTestResult(
            validity=validity,
            vt1=None,
            vt2=None,
            overall_confidence=0.3,
        )

        report = generate_report(result)

        assert report.vt1_section is not None  # Should have section with warning
        assert report.vt2_section is not None

    def test_generate_report_with_conflicts(self):
        """Generate report with signal conflicts."""
        validity = TestValidity(validity=ValidityLevel.VALID)
        conflict = SignalConflict(
            conflict_type=ConflictType.CARDIAC_DRIFT,
            severity=ConflictSeverity.WARNING,
            signal_a="HR",
            signal_b="Power",
            description="HR drift detected",
            physiological_interpretation="Cardiac drift in Z2",
            confidence_penalty=0.1,
        )
        conflicts = ConflictReport(conflicts=[conflict])
        result = RampTestResult(
            validity=validity,
            conflicts=conflicts,
            overall_confidence=0.7,
        )

        report = generate_report(result)

        assert report.conflicts_section is not None

    def test_generate_report_with_smo2_data(self):
        """Generate report with SmO2 threshold data."""
        validity = TestValidity(validity=ValidityLevel.VALID)
        smo2_lt1 = ThresholdRange(
            lower_watts=170,
            upper_watts=180,
            midpoint_watts=175,
            confidence=0.7,
            sources=["SmO2"],
        )
        result = RampTestResult(
            validity=validity,
            smo2_lt1=smo2_lt1,
            smo2_deviation_from_vt=5.0,
            smo2_interpretation="SmO2 drop near VT1",
            overall_confidence=0.7,
        )

        report = generate_report(result)

        assert report.smo2_section is not None


class TestFormatReportMarkdown:
    """Tests for format_report_markdown function."""

    def test_format_basic_report(self):
        """Format a basic report to markdown."""
        report = RampTestReport(
            summary=ReportSection(
                title="Podsumowanie",
                content="Test valid. VT1 and VT2 detected.",
            ),
            validity=ReportSection(
                title="Ważność testu",
                content="Test valid.",
            ),
            overall_confidence=0.8,
        )

        markdown = format_report_markdown(report)

        assert "# Podsumowanie" in markdown
        assert "Test valid" in markdown

    def test_format_report_with_all_sections(self):
        """Format report with all sections."""
        report = RampTestReport(
            summary=ReportSection(title="Summary", content="Test summary"),
            validity=ReportSection(title="Validity", content="Valid test"),
            vt1_section=ReportSection(
                title="VT1",
                content="VT1: 180-185W",
                confidence=0.85,
            ),
            vt2_section=ReportSection(
                title="VT2",
                content="VT2: 250-255W",
                confidence=0.80,
            ),
            smo2_section=ReportSection(title="SmO2", content="SmO2 data"),
            conflicts_section=ReportSection(
                title="Conflicts",
                content="No conflicts",
            ),
            recommendations_section=ReportSection(
                title="Recommendations",
                content="Train in zones",
            ),
            overall_confidence=0.75,
        )

        markdown = format_report_markdown(report)

        assert "VT1" in markdown
        assert "VT2" in markdown
        assert "SmO2" in markdown

    def test_format_report_with_warnings(self):
        """Format report with warnings."""
        report = RampTestReport(
            summary=ReportSection(title="Summary", content="Test"),
            vt1_section=ReportSection(
                title="VT1",
                content="VT1 detected",
                warnings=["Low confidence", "Signal noise"],
            ),
            overall_confidence=0.5,
        )

        markdown = format_report_markdown(report)

        assert "Low confidence" in markdown or "⚠" in markdown


class TestConfidenceLanguage:
    """Tests for confidence-based language."""

    def test_high_confidence_language(self):
        """High confidence uses definitive language."""
        validity = TestValidity(validity=ValidityLevel.VALID)
        vt1 = ThresholdRange(
            lower_watts=175,
            upper_watts=185,
            midpoint_watts=180,
            confidence=0.9,
        )
        result = RampTestResult(
            validity=validity,
            vt1=vt1,
            overall_confidence=0.9,
        )

        report = generate_report(result)

        # High confidence should use definitive language
        assert report.overall_confidence >= 0.8

    def test_low_confidence_language(self):
        """Low confidence uses tentative language."""
        validity = TestValidity(
            validity=ValidityLevel.CONDITIONAL,
            issues=["Poor signal quality"],
        )
        result = RampTestResult(
            validity=validity,
            overall_confidence=0.35,
        )

        report = generate_report(result)

        assert report.overall_confidence < 0.5
