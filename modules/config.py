"""Application configuration with domain-grouped frozen dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# ── CSS Path Validation (module-level, runs once) ─────────────────────

_raw_css = os.getenv("CSS_FILE", "style.css")
_css_path = Path(BASE_DIR) / _raw_css if not Path(_raw_css).is_absolute() else Path(_raw_css)
if not _css_path.resolve().is_relative_to(BASE_DIR) or _css_path.suffix.lower() != ".css":
    _raw_css = "style.css"

# ── Domain Dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class AppSettings:
    title: str = os.getenv("APP_TITLE", "Tri Dashboard")
    icon: str = os.getenv("APP_ICON", "\u26a1")
    layout: str = os.getenv("APP_LAYOUT", "wide")
    css_file: str = _raw_css


@dataclass(frozen=True)
class AnalysisParams:
    rolling_window_5min: int = int(os.getenv("ROLLING_WINDOW_5MIN", "300"))
    rolling_window_30s: int = int(os.getenv("ROLLING_WINDOW_30S", "30"))
    rolling_window_60s: int = int(os.getenv("ROLLING_WINDOW_60S", "60"))
    smooth_window: int = int(os.getenv("SMOOTH_WINDOW", "30"))
    smooth_window_short: int = int(os.getenv("SMOOTH_WINDOW_SHORT", "5"))
    resample_threshold: int = int(os.getenv("RESAMPLE_THRESHOLD", "10000"))
    resample_step: int = int(os.getenv("RESAMPLE_STEP", "5"))
    min_watts_active: int = int(os.getenv("MIN_WATTS_ACTIVE", "10"))
    min_hr_active: int = int(os.getenv("MIN_HR_ACTIVE", "40"))
    min_records_for_rolling: int = int(os.getenv("MIN_RECORDS_FOR_ROLLING", "30"))
    min_df_length: int = int(os.getenv("MIN_DF_LENGTH", "10"))


@dataclass(frozen=True)
class ValidationParams:
    max_watts: int = int(os.getenv("VALIDATION_MAX_WATTS", "3000"))
    max_hr: int = int(os.getenv("VALIDATION_MAX_HR", "250"))
    max_cadence: int = int(os.getenv("VALIDATION_MAX_CADENCE", "250"))
    required_cols: tuple[str, ...] = ("time",)
    data_cols: tuple[str, ...] = (
        "watts",
        "heartrate",
        "cadence",
        "smo2",
        "power",
    )


@dataclass(frozen=True)
class DbSettings:
    name: str = os.getenv("DB_NAME", "training_history.db")
    data_dir: Path = BASE_DIR / "data"

    @property
    def path(self) -> Path:
        return self.data_dir / self.name


@dataclass(frozen=True)
class MlSettings:
    model_file: str = os.getenv("MODEL_FILE", "cycling_brain_weights.npz")
    history_file: str = os.getenv("HISTORY_FILE", "brain_evolution_history.json")
    epochs: int = int(os.getenv("ML_EPOCHS", "200"))
    learning_rate: float = float(os.getenv("ML_LEARNING_RATE", "0.02"))


@dataclass(frozen=True)
class UIColors:
    power: str = os.getenv("COLOR_POWER", "#00cc96")
    hr: str = os.getenv("COLOR_HR", "#ef553b")
    smo2: str = os.getenv("COLOR_SMO2", "#ab63fa")
    ve: str = os.getenv("COLOR_VE", "#ffa15a")
    rr: str = os.getenv("COLOR_RR", "#19d3f3")
    thb: str = os.getenv("COLOR_THB", "#e377c2")
    torque: str = os.getenv("COLOR_TORQUE", "#e377c2")


@dataclass(frozen=True)
class ThresholdParams:
    vt1_slope_threshold: float = float(os.getenv("VT1_SLOPE_THRESHOLD", "0.05"))
    vt2_slope_threshold: float = float(os.getenv("VT2_SLOPE_THRESHOLD", "0.10"))
    vt1_slope_spike_skip: float = float(os.getenv("VT1_SLOPE_SPIKE_SKIP", "0.10"))
    slope_confidence_max: float = float(os.getenv("SLOPE_CONFIDENCE_MAX", "0.4"))
    stability_confidence_max: float = float(os.getenv("STABILITY_CONFIDENCE_MAX", "0.4"))
    base_confidence: float = float(os.getenv("BASE_CONFIDENCE", "0.2"))
    max_confidence: float = float(os.getenv("MAX_CONFIDENCE", "0.95"))
    lower_step_weight: float = float(os.getenv("LOWER_STEP_WEIGHT", "0.3"))
    upper_step_weight: float = float(os.getenv("UPPER_STEP_WEIGHT", "0.7"))
    ramp_min_step_duration: int = int(os.getenv("RAMP_MIN_STEP_DURATION", "120"))
    ramp_power_increment_min: int = int(os.getenv("RAMP_POWER_INCREMENT_MIN", "15"))
    ramp_power_increment_max: int = int(os.getenv("RAMP_POWER_INCREMENT_MAX", "40"))


# ── Config Facade ──────────────────────────────────────────────────────


class Config:
    """Application configuration facade.

    Structured access: Config.db.path, Config.colors.smo2
    Legacy access (still works): Config.DB_PATH, Config.COLOR_SMO2
    """

    # Sub-configs (new structured access)
    app = AppSettings()
    analysis = AnalysisParams()
    validation = ValidationParams()
    db = DbSettings()
    ml = MlSettings()
    colors = UIColors()
    thresholds = ThresholdParams()

    # ── Module-level ──
    BASE_DIR = BASE_DIR

    # ── Backward-compat passthrough: App ──
    APP_TITLE = app.title
    APP_ICON = app.icon
    APP_LAYOUT = app.layout
    CSS_FILE = app.css_file

    # ── Backward-compat passthrough: Analysis ──
    ROLLING_WINDOW_5MIN = analysis.rolling_window_5min
    ROLLING_WINDOW_30S = analysis.rolling_window_30s
    ROLLING_WINDOW_60S = analysis.rolling_window_60s
    SMOOTH_WINDOW = analysis.smooth_window
    SMOOTH_WINDOW_SHORT = analysis.smooth_window_short
    RESAMPLE_THRESHOLD = analysis.resample_threshold
    RESAMPLE_STEP = analysis.resample_step
    MIN_WATTS_ACTIVE = analysis.min_watts_active
    MIN_HR_ACTIVE = analysis.min_hr_active
    MIN_RECORDS_FOR_ROLLING = analysis.min_records_for_rolling
    MIN_DF_LENGTH = analysis.min_df_length

    # ── Backward-compat passthrough: Validation ──
    VALIDATION_MAX_WATTS = validation.max_watts
    VALIDATION_MAX_HR = validation.max_hr
    VALIDATION_MAX_CADENCE = validation.max_cadence
    VALIDATION_REQUIRED_COLS = list(validation.required_cols)
    VALIDATION_DATA_COLS = list(validation.data_cols)

    # ── Backward-compat passthrough: ML ──
    MODEL_FILE = ml.model_file
    HISTORY_FILE = ml.history_file
    ML_EPOCHS = ml.epochs
    ML_LEARNING_RATE = ml.learning_rate

    # ── Backward-compat passthrough: Database ──
    DATA_DIR = db.data_dir
    DB_NAME = db.name
    DB_PATH = db.path

    # ── Backward-compat passthrough: Colors ──
    COLOR_POWER = colors.power
    COLOR_HR = colors.hr
    COLOR_SMO2 = colors.smo2
    COLOR_VE = colors.ve
    COLOR_RR = colors.rr
    COLOR_THB = colors.thb
    COLOR_TORQUE = colors.torque

    # ── Backward-compat passthrough: Thresholds ──
    VT1_SLOPE_THRESHOLD = thresholds.vt1_slope_threshold
    VT2_SLOPE_THRESHOLD = thresholds.vt2_slope_threshold
    VT1_SLOPE_SPIKE_SKIP = thresholds.vt1_slope_spike_skip
    SLOPE_CONFIDENCE_MAX = thresholds.slope_confidence_max
    STABILITY_CONFIDENCE_MAX = thresholds.stability_confidence_max
    BASE_CONFIDENCE = thresholds.base_confidence
    MAX_CONFIDENCE = thresholds.max_confidence
    LOWER_STEP_WEIGHT = thresholds.lower_step_weight
    UPPER_STEP_WEIGHT = thresholds.upper_step_weight
    RAMP_MIN_STEP_DURATION = thresholds.ramp_min_step_duration
    RAMP_POWER_INCREMENT_MIN = thresholds.ramp_power_increment_min
    RAMP_POWER_INCREMENT_MAX = thresholds.ramp_power_increment_max
