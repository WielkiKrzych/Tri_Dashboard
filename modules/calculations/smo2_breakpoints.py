"""
SmO2 Breakpoint Detection using Segmented Regression.

Based on:
1. Feldmann et al. (2022) - Muscle Oxygen Saturation Breakpoints Reflect Ventilatory Thresholds
2. Moxy Monitor methodology - piecewise linear regression
3. Bhambhani (2004) - Four phases of muscle oxygenation

Algorithm:
- Fit piecewise linear function with 2 breakpoints (3-segment) OR 1 breakpoint (2-segment)
- Minimize sum of squared residuals
- Breakpoints correspond to LT1 and LT2

UPDATED 2026-02-28:
- Added double-linear (2-segment) regression method (recommended by Contreras-Briceño 2023)
- Added Exp-Dmax method for T2 detection (ICC = 0.79-0.91)
- 2-segment is now the default as it's more robust (ICC = 0.80 for second threshold)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class SmO2Breakpoints:
    """Container for SmO2 breakpoint detection results."""

    bp1_power: Optional[float] = None  # LT1
    bp1_smo2: Optional[float] = None
    bp2_power: Optional[float] = None  # LT2
    bp2_smo2: Optional[float] = None
    slope1: Optional[float] = None  # Baseline slope
    slope2: Optional[float] = None  # Moderate drop slope
    slope3: Optional[float] = None  # Steep drop slope (3-segment only)
    r_squared: Optional[float] = None
    method: str = "3-segment"
    is_valid: bool = False
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class ExpDmaxResult:
    """Container for Exp-Dmax detection results."""

    breakpoint_power: Optional[float] = None
    breakpoint_smo2: Optional[float] = None
    max_distance: Optional[float] = None
    r_squared: Optional[float] = None
    is_valid: bool = False
    notes: List[str] = field(default_factory=list)


# =============================================================================
# DOUBLE-LINEAR (2-SEGMENT) REGRESSION - Recommended method
# =============================================================================


def detect_smo2_breakpoints_double_linear(
    df: pd.DataFrame,
    smo2_col: str = "smo2",
    power_col: str = "watts",
    bp_range: Tuple[float, float] = (250, 380),
    force_bp: Optional[float] = None,
) -> SmO2Breakpoints:
    """
    Detect SmO2 breakpoint using double-linear regression (2-segment).

    This is the RECOMMENDED method based on Contreras-Briceño et al. (2023):
    - ICC = 0.80 for second threshold (more reliable than 3-segment)
    - Used by 46% of studies and WKO5 standard

    Fits 2 segments:
    - Segment 1: Baseline/Plateau (before LT1)
    - Segment 2: Progressive desaturation (after LT1)

    Args:
        df: DataFrame with SmO2 and power data
        smo2_col: Column name for SmO2 values
        power_col: Column name for power values
        bp_range: Search range for breakpoint
        force_bp: Force a specific breakpoint value

    Returns:
        SmO2Breakpoints object with detected breakpoint (bp1 = LT1)
    """
    result = SmO2Breakpoints(method="2-segment")

    if smo2_col not in df.columns or power_col not in df.columns:
        result.notes.append("Missing required columns")
        return result

    # Filter to ramp phase (exclude recovery)
    max_power_idx = df[power_col].idxmax()
    df_ramp = df.iloc[:max_power_idx].copy()

    if len(df_ramp) < 100:
        result.notes.append("Insufficient data points")
        return result

    # Smooth SmO2 data
    window = min(51, len(df_ramp) // 10 * 2 + 1)
    if window < 5:
        window = 5
    df_ramp["smo2_smooth"] = savgol_filter(df_ramp[smo2_col], window_length=window, polyorder=3)

    x = df_ramp[power_col].values
    y = df_ramp["smo2_smooth"].values

    # Grid search for optimal breakpoint
    best_rss = np.inf
    best_bp = None
    best_slopes = None

    if force_bp is not None:
        try:
            rss, slopes = _fit_piecewise_2segment(x, y, force_bp)
            best_rss = rss
            best_bp = force_bp
            best_slopes = slopes
            result.notes.append(f"Using forced breakpoint: {force_bp}W")
        except Exception as e:
            result.notes.append(f"Forced breakpoint failed: {e}")

    if best_bp is None:
        best_bp, best_slopes, best_rss = _search_breakpoint_2segment(x, y, bp_range)

    if best_bp is None:
        result.notes.append("Could not find valid breakpoint")
        return result

    # Calculate R-squared
    y_pred = _predict_piecewise_2segment(x, y, best_bp, best_slopes)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Get SmO2 value at breakpoint
    bp_idx = np.argmin(np.abs(x - best_bp))

    result.bp1_power = float(best_bp)
    result.bp1_smo2 = float(y[bp_idx])
    result.slope1, result.slope2 = best_slopes
    result.r_squared = float(r_squared)
    result.is_valid = True

    # Confidence based on R² and slope change
    slope_change = abs(best_slopes[1] - best_slopes[0])
    result.confidence = min(1.0, r_squared * 0.5 + min(slope_change * 10, 0.5))

    result.notes.append(f"BP1 (LT1): {best_bp:.0f}W @ {result.bp1_smo2:.1f}% SmO2")
    result.notes.append(f"Slopes: {best_slopes[0]:.4f} → {best_slopes[1]:.4f}")
    result.notes.append(f"R² = {r_squared:.3f}")
    result.notes.append(f"Method: 2-segment (double-linear, ICC=0.80)")

    return result


def _fit_piecewise_2segment(
    x: np.ndarray, y: np.ndarray, bp: float
) -> Tuple[float, Tuple[float, float]]:
    """Fit 2-segment piecewise linear function and return RSS."""
    mask1 = x < bp
    mask2 = x >= bp

    slopes = []

    # Segment 1
    if np.sum(mask1) > 1:
        slope, intercept, _, _, _ = stats.linregress(x[mask1], y[mask1])
    else:
        slope, intercept = 0, np.mean(y[mask1]) if np.any(mask1) else y[0]
    slopes.append(slope)

    # Segment 2
    if np.sum(mask2) > 1:
        slope, intercept, _, _, _ = stats.linregress(x[mask2], y[mask2])
    else:
        slope, intercept = slopes[0], np.mean(y[mask2]) if np.any(mask2) else y[-1]
    slopes.append(slope)

    # Calculate RSS
    y_pred = _predict_piecewise_2segment(x, y, bp, tuple(slopes))
    rss = np.sum((y - y_pred) ** 2)

    return rss, tuple(slopes)


def _predict_piecewise_2segment(
    x: np.ndarray, y: np.ndarray, bp: float, slopes: Tuple[float, float]
) -> np.ndarray:
    """Predict y values using 2-segment piecewise linear function."""
    mask1 = x < bp
    mask2 = x >= bp

    y_pred = np.zeros_like(x)

    # Segment 1
    if np.sum(mask1) > 0:
        intercept1 = np.mean(y[mask1]) - slopes[0] * np.mean(x[mask1])
        y_pred[mask1] = slopes[0] * x[mask1] + intercept1

    # Segment 2 - ensure continuity at bp
    if np.sum(mask2) > 0:
        if np.sum(mask1) > 0:
            y_bp = slopes[0] * bp + (np.mean(y[mask1]) - slopes[0] * np.mean(x[mask1]))
        else:
            y_bp = y[0]
        intercept2 = y_bp - slopes[1] * bp
        y_pred[mask2] = slopes[1] * x[mask2] + intercept2

    return y_pred


def _search_breakpoint_2segment(
    x: np.ndarray, y: np.ndarray, bp_range: Tuple[float, float]
) -> Tuple[Optional[float], Optional[Tuple[float, float]], float]:
    """
    Two-phase breakpoint search for 2-segment model.

    Phase 1: Coarse grid search (step=10)
    Phase 2: Fine grid search (step=1) around best coarse result
    """
    best_rss = np.inf
    best_bp = None
    best_slopes = None

    # Phase 1: Coarse search
    coarse_step = 10
    bp_coarse = np.arange(bp_range[0], bp_range[1], coarse_step)

    for bp in bp_coarse:
        try:
            rss, slopes = _fit_piecewise_2segment(x, y, bp)
            if rss < best_rss:
                best_rss = rss
                best_bp = bp
                best_slopes = slopes
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue

    if best_bp is None:
        return None, None, np.inf

    # Phase 2: Fine search around best coarse
    fine_step = 1
    fine_range = 10

    bp_fine = np.arange(
        max(bp_range[0], best_bp - fine_range),
        min(bp_range[1], best_bp + fine_range + fine_step),
        fine_step,
    )

    for bp in bp_fine:
        try:
            rss, slopes = _fit_piecewise_2segment(x, y, bp)
            if rss < best_rss:
                best_rss = rss
                best_bp = bp
                best_slopes = slopes
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue

    return best_bp, best_slopes, best_rss


# =============================================================================
# EXP-DMAX METHOD - Best for T2 detection (ICC = 0.79-0.91)
# =============================================================================


def detect_exp_dmax(
    df: pd.DataFrame,
    smo2_col: str = "smo2",
    power_col: str = "watts",
    min_power: float = 100.0,
    baseline: Optional[float] = None,
) -> ExpDmaxResult:
    """
    Detect SmO2 threshold using Exp-Dmax method.

    This is the BEST method for T2 detection based on 2023-2024 research:
    - ICC = 0.79-0.91 (highest reliability for second threshold)
    - Superior to classic Dmax (ICC = 0.35)

    Algorithm:
    1. Fit exponential curve to SmO2 vs Power
    2. Draw line from first to last point
    3. Find point of maximum distance from curve to line

    Args:
        df: DataFrame with SmO2 and power data
        smo2_col: Column name for SmO2 values
        power_col: Column name for power values
        min_power: Minimum power to include in analysis

    Returns:
        ExpDmaxResult with detected threshold
    """
    result = ExpDmaxResult()

    if smo2_col not in df.columns or power_col not in df.columns:
        result.notes.append("Missing required columns")
        return result

    # Filter to ramp phase and minimum power
    max_power_idx = df[power_col].idxmax()
    df_ramp = df.iloc[: max_power_idx + 1].copy()
    df_ramp = df_ramp[df_ramp[power_col] >= min_power].copy()

    if len(df_ramp) < 30:
        result.notes.append("Insufficient data points (need >= 30)")
        return result

    # Smooth data
    window = min(21, len(df_ramp) // 5 * 2 + 1)
    if window < 5:
        window = 5

    x = df_ramp[power_col].values
    y_raw = df_ramp[smo2_col].values

    # Signal inversion per Sendra-Pérez 2024: invert ΔSmO2 so it
    # behaves incrementally (like lactate) for proper Exp-Dmax fit
    if baseline is not None:
        y_raw = baseline - y_raw

    y = savgol_filter(y_raw, window_length=window, polyorder=2)

    # Fit exponential curve: y = a * exp(b * x) + c
    try:
        # Initial parameter estimates
        y_range = y.max() - y.min()
        a_init = -y_range  # Negative because SmO2 decreases
        b_init = -0.01
        c_init = y.max()

        popt, _ = curve_fit(
            _exp_func,
            x,
            y,
            p0=[a_init, b_init, c_init],
            bounds=([-np.inf, -1, -np.inf], [0, 0, np.inf]),
            maxfev=5000,
        )
        a, b, c = popt

        # Calculate exponential curve values
        y_exp = _exp_func(x, a, b, c)

        # Line from first to last point
        x_first, y_first = x[0], y_exp[0]
        x_last, y_last = x[-1], y_exp[-1]

        # Find point of maximum perpendicular distance
        max_dist = 0
        max_dist_idx = 0

        for i in range(len(x)):
            # Distance from point to line
            dist = _point_to_line_distance(x[i], y_exp[i], x_first, y_first, x_last, y_last)
            if dist > max_dist:
                max_dist = dist
                max_dist_idx = i

        # Calculate R-squared for the exponential fit
        ss_res = np.sum((y - y_exp) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        result.breakpoint_power = float(x[max_dist_idx])
        result.breakpoint_smo2 = float(y[max_dist_idx])
        result.max_distance = float(max_dist)
        result.r_squared = float(r_squared)
        result.is_valid = True

        result.notes.append(
            f"Exp-Dmax threshold: {result.breakpoint_power:.0f}W @ {result.breakpoint_smo2:.1f}% SmO2"
        )
        result.notes.append(f"Max distance: {max_dist:.3f}")
        result.notes.append(f"Exp fit R² = {r_squared:.3f}")
        result.notes.append("Method: Exp-Dmax (ICC = 0.79-0.91 for T2)")

    except (RuntimeError, ValueError) as e:
        result.notes.append(f"Exp-Dmax fitting failed: {e}")
        return result

    return result


def detect_t2_exp_dmax(
    power: List[float],
    smo2: List[float],
    min_last_segment: int = 3,
    baseline: Optional[float] = None,
) -> Optional[Dict]:
    """
    Detect T2 using Exp-Dmax method.

    Exp-Dmax algorithm (modified Dmax):
    1. Fit exponential curve: y = a * exp(b * x) + c
    2. Draw line from first point to last point
    3. Find point with maximum distance to line

    Args:
        power: List of power values (watts)
        smo2: List of SmO2 values (percent)
        min_last_segment: Minimum points after breakpoint for validity

    Returns:
        Dict with:
        - t2_power: float (watts)
        - t2_smo2: float (percent)
        - distance: float
        - slope_at_t2: float (percent per watt)
        - confidence: float (0.79-0.91 per ICC from literature)
        - method: str = "exp_dmax"
        Or None if detection fails
    """
    if len(power) < 30 or len(smo2) < 30:
        return None

    if len(power) != len(smo2):
        return None

    x = np.array(power)
    y = np.array(smo2)

    # Signal inversion per Sendra-Pérez 2024
    if baseline is not None:
        y = baseline - y

    # Fit exponential curve: y = a * exp(b * x) + c
    try:
        # Initial parameter estimates
        y_range = y.max() - y.min()
        a_init = -y_range  # Negative because SmO2 decreases
        b_init = -0.01
        c_init = y.max()

        popt, _ = curve_fit(
            _exp_func,
            x,
            y,
            p0=[a_init, b_init, c_init],
            bounds=([-np.inf, -1, -np.inf], [0, 0, np.inf]),
            maxfev=5000,
        )
        a, b, c = popt

        # Calculate exponential curve values
        y_exp = _exp_func(x, a, b, c)

        # Line from first to last point
        x_first, y_first = x[0], y_exp[0]
        x_last, y_last = x[-1], y_exp[-1]

        # Find point of maximum perpendicular distance
        max_dist = 0
        max_dist_idx = 0

        for i in range(len(x)):
            # Distance from point to line
            dist = _point_to_line_distance(x[i], y_exp[i], x_first, y_first, x_last, y_last)
            if dist > max_dist:
                max_dist = dist
                max_dist_idx = i

        # Validate: enough points after breakpoint
        points_after = len(x) - max_dist_idx - 1
        if points_after < min_last_segment:
            return None

        # Calculate slope at T2
        if max_dist_idx < len(x) - 1:
            slope_at_t2 = (y[-1] - y[max_dist_idx]) / (x[-1] - x[max_dist_idx])
        else:
            slope_at_t2 = 0.0

        return {
            "t2_power": float(x[max_dist_idx]),
            "t2_smo2": float(y[max_dist_idx]),
            "distance": float(max_dist),
            "slope_at_t2": float(slope_at_t2),
            "confidence": 0.85,  # Default confidence based on ICC range 0.79-0.91
            "method": "exp_dmax",
        }

    except (RuntimeError, ValueError):
        return None


def _exp_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential function: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def _point_to_line_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate perpendicular distance from point (px, py) to line through (x1,y1)-(x2,y2)."""
    # Line length
    line_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_len == 0:
        return 0

    # Perpendicular distance formula
    dist = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / line_len
    return dist


# =============================================================================
# 3-SEGMENT REGRESSION - Original method (for simultaneous LT1+LT2 detection)
# =============================================================================


def detect_smo2_breakpoints_segmented(
    df: pd.DataFrame,
    smo2_col: str = "smo2",
    power_col: str = "watts",
    bp1_range: Tuple[float, float] = (250, 330),
    bp2_range: Tuple[float, float] = (340, 420),
    min_separation: float = 40,
    force_bp1: Optional[float] = None,
    force_bp2: Optional[float] = None,
) -> SmO2Breakpoints:
    """
    Detect SmO2 breakpoints using 3-segment regression (piecewise linear).

    NOTE: 2-segment (double-linear) is now the RECOMMENDED method.
    Use 3-segment only when you need simultaneous LT1+LT2 detection.

    This method fits a 3-segment piecewise linear function to SmO2 vs Power data:
    - Segment 1: Baseline/Plateau (before LT1)
    - Segment 2: Moderate desaturation (LT1 to LT2)
    - Segment 3: Steep desaturation (after LT2)

    Args:
        df: DataFrame with SmO2 and power data
        smo2_col: Column name for SmO2 values
        power_col: Column name for power values
        bp1_range: Search range for first breakpoint (LT1)
        bp2_range: Search range for second breakpoint (LT2)
        min_separation: Minimum power separation between breakpoints
        force_bp1: Force a specific value for BP1
        force_bp2: Force a specific value for BP2

    Returns:
        SmO2Breakpoints object with detected breakpoints
    """
    result = SmO2Breakpoints(method="3-segment")

    # Validate input
    if smo2_col not in df.columns or power_col not in df.columns:
        result.notes.append("Missing required columns")
        return result

    # Filter to ramp phase (exclude recovery)
    max_power_idx = df[power_col].idxmax()
    df_ramp = df.iloc[:max_power_idx].copy()

    if len(df_ramp) < 100:
        result.notes.append("Insufficient data points")
        return result

    # Smooth SmO2 data
    window = min(51, len(df_ramp) // 10 * 2 + 1)
    if window < 5:
        window = 5
    df_ramp["smo2_smooth"] = savgol_filter(df_ramp[smo2_col], window_length=window, polyorder=3)

    x = df_ramp[power_col].values
    y = df_ramp["smo2_smooth"].values

    # Grid search for optimal breakpoints
    best_rss = np.inf
    best_bp1, best_bp2 = None, None
    best_slopes = None

    # Use forced breakpoints if provided, otherwise search
    if force_bp1 is not None and force_bp2 is not None:
        # Validate forced breakpoints
        if force_bp2 > force_bp1 + min_separation:
            try:
                rss, slopes = _fit_piecewise_3segment(x, y, force_bp1, force_bp2)
                best_rss = rss
                best_bp1, best_bp2 = force_bp1, force_bp2
                best_slopes = slopes
                result.notes.append(f"Using forced breakpoints: {force_bp1}W, {force_bp2}W")
            except Exception as e:
                result.notes.append(f"Forced breakpoints failed: {e}")
        else:
            result.notes.append("Forced breakpoints too close, using auto-detection")

    # Auto-detection if no forced breakpoints or forced failed
    if best_bp1 is None:
        # Use two-phase search: coarse grid + fine optimization
        best_bp1, best_bp2, best_slopes, best_rss = _two_phase_breakpoint_search(
            x, y, bp1_range, bp2_range, min_separation
        )

    if best_bp1 is None or best_bp2 is None:
        result.notes.append("Could not find valid breakpoints")
        return result

    # Calculate R-squared
    y_pred = _predict_piecewise_3segment(x, y, best_bp1, best_bp2, best_slopes)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Get SmO2 values at breakpoints
    bp1_idx = np.argmin(np.abs(x - best_bp1))
    bp2_idx = np.argmin(np.abs(x - best_bp2))

    result.bp1_power = float(best_bp1)
    result.bp1_smo2 = float(y[bp1_idx])
    result.bp2_power = float(best_bp2)
    result.bp2_smo2 = float(y[bp2_idx])
    result.slope1, result.slope2, result.slope3 = best_slopes
    result.r_squared = float(r_squared)
    result.is_valid = True

    # Confidence based on R²
    result.confidence = min(1.0, r_squared * 0.7 + 0.1)

    # Add interpretation
    result.notes.append(f"BP1 (LT1): {best_bp1:.0f}W @ {result.bp1_smo2:.1f}% SmO2")
    result.notes.append(f"BP2 (LT2): {best_bp2:.0f}W @ {result.bp2_smo2:.1f}% SmO2")
    result.notes.append(f"Slopes: {best_slopes[0]:.4f}, {best_slopes[1]:.4f}, {best_slopes[2]:.4f}")
    result.notes.append(f"R² = {r_squared:.3f}")
    result.notes.append("Method: 3-segment (for simultaneous LT1+LT2)")

    # Validate against expected ranges
    if not (280 <= best_bp1 <= 310):
        result.notes.append(f"⚠️ BP1 outside expected range (280-310W)")
    if not (360 <= best_bp2 <= 400):
        result.notes.append(f"⚠️ BP2 outside expected range (360-400W)")

    return result


def _fit_piecewise_3segment(
    x: np.ndarray, y: np.ndarray, bp1: float, bp2: float
) -> Tuple[float, Tuple[float, float, float]]:
    """Fit 3-segment piecewise linear function and return RSS."""
    mask1 = x < bp1
    mask2 = (x >= bp1) & (x < bp2)
    mask3 = x >= bp2

    slopes = []

    # Segment 1
    if np.sum(mask1) > 1:
        slope, intercept, _, _, _ = stats.linregress(x[mask1], y[mask1])
    else:
        slope, intercept = 0, np.mean(y[mask1]) if np.any(mask1) else y[0]
    slopes.append(slope)
    y1_pred = slope * x[mask1] + intercept

    # Segment 2
    if np.sum(mask2) > 1:
        slope, intercept, _, _, _ = stats.linregress(x[mask2], y[mask2])
    else:
        slope, intercept = slopes[0], np.mean(y[mask2]) if np.any(mask2) else y[len(y) // 2]
    slopes.append(slope)
    y2_pred = slope * x[mask2] + intercept

    # Segment 3
    if np.sum(mask3) > 1:
        slope, intercept, _, _, _ = stats.linregress(x[mask3], y[mask3])
    else:
        slope, intercept = slopes[1], np.mean(y[mask3]) if np.any(mask3) else y[-1]
    slopes.append(slope)
    y3_pred = slope * x[mask3] + intercept

    # Calculate RSS
    rss = (
        np.sum((y[mask1] - y1_pred) ** 2)
        + np.sum((y[mask2] - y2_pred) ** 2)
        + np.sum((y[mask3] - y3_pred) ** 2)
    )

    return rss, tuple(slopes)


def _predict_piecewise_3segment(
    x: np.ndarray, y: np.ndarray, bp1: float, bp2: float, slopes: Tuple[float, float, float]
) -> np.ndarray:
    """Predict y values using piecewise linear function."""
    mask1 = x < bp1
    mask2 = (x >= bp1) & (x < bp2)
    mask3 = x >= bp2

    y_pred = np.zeros_like(x)

    # Fit intercepts to ensure continuity
    # Segment 1
    if np.sum(mask1) > 0:
        intercept1 = np.mean(y[mask1]) - slopes[0] * np.mean(x[mask1])
        y_pred[mask1] = slopes[0] * x[mask1] + intercept1

    # Segment 2 - ensure continuity at bp1
    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
        y_bp1 = y_pred[mask1][-1] if len(y_pred[mask1]) > 0 else np.mean(y[mask1])
        intercept2 = y_bp1 - slopes[1] * bp1
        y_pred[mask2] = slopes[1] * x[mask2] + intercept2

    # Segment 3 - ensure continuity at bp2
    if np.sum(mask2) > 0 and np.sum(mask3) > 0:
        y_bp2 = y_pred[mask2][-1] if len(y_pred[mask2]) > 0 else np.mean(y[mask2])
        intercept3 = y_bp2 - slopes[2] * bp2
        y_pred[mask3] = slopes[2] * x[mask3] + intercept3

    return y_pred


def _two_phase_breakpoint_search(
    x: np.ndarray,
    y: np.ndarray,
    bp1_range: Tuple[float, float],
    bp2_range: Tuple[float, float],
    min_separation: float,
) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float, float]], float]:
    """
    Two-phase breakpoint search: coarse grid + fine optimization.

    Phase 1: Coarse grid search (step=20) to find approximate region.
    Phase 2: Fine grid search (step=2) around best coarse result.

    Complexity: O((n/20)^2 + 15^2) = O(n^2/400) vs original O(n^2)
    For typical ranges: ~25 iterations vs ~256 iterations (10x speedup)
    """
    best_rss = np.inf
    best_bp1, best_bp2 = None, None
    best_slopes = None

    # Phase 1: Coarse search with step=20
    coarse_step = 20
    bp1_coarse = np.arange(bp1_range[0], bp1_range[1], coarse_step)
    bp2_coarse = np.arange(bp2_range[0], bp2_range[1], coarse_step)

    for bp1 in bp1_coarse:
        for bp2 in bp2_coarse:
            if bp2 <= bp1 + min_separation:
                continue
            try:
                rss, slopes = _fit_piecewise_3segment(x, y, bp1, bp2)
                if rss < best_rss:
                    best_rss = rss
                    best_bp1, best_bp2 = bp1, bp2
                    best_slopes = slopes
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                continue

    if best_bp1 is None:
        return None, None, None, np.inf

    # Phase 2: Fine search around best coarse (±15W, step=2)
    fine_step = 2
    fine_range = 15

    bp1_fine = np.arange(
        max(bp1_range[0], best_bp1 - fine_range),
        min(bp1_range[1], best_bp1 + fine_range + fine_step),
        fine_step,
    )
    bp2_fine = np.arange(
        max(bp2_range[0], best_bp2 - fine_range),
        min(bp2_range[1], best_bp2 + fine_range + fine_step),
        fine_step,
    )

    for bp1 in bp1_fine:
        for bp2 in bp2_fine:
            if bp2 <= bp1 + min_separation:
                continue
            try:
                rss, slopes = _fit_piecewise_3segment(x, y, bp1, bp2)
                if rss < best_rss:
                    best_rss = rss
                    best_bp1, best_bp2 = bp1, bp2
                    best_slopes = slopes
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                continue

    return best_bp1, best_bp2, best_slopes, best_rss


# =============================================================================
# UNIFIED DETECTION FUNCTION
# =============================================================================


def detect_smo2_breakpoints(
    df: pd.DataFrame,
    smo2_col: str = "smo2",
    power_col: str = "watts",
    method: str = "2-segment",
    baseline: Optional[float] = None,
    **kwargs,
) -> SmO2Breakpoints:
    """
    Unified SmO2 breakpoint detection function.

    Args:
        df: DataFrame with SmO2 and power data
        smo2_col: Column name for SmO2 values
        power_col: Column name for power values
        method: Detection method - one of:
            - "2-segment": Double-linear regression (RECOMMENDED, ICC=0.80)
            - "3-segment": Triple-linear for simultaneous LT1+LT2
            - "exp-dmax": Exp-Dmax method (best for T2, ICC=0.79-0.91)
        **kwargs: Additional arguments passed to specific method

    Returns:
        SmO2Breakpoints object with detected breakpoints
    """
    if method == "2-segment":
        return detect_smo2_breakpoints_double_linear(df, smo2_col, power_col, **kwargs)
    elif method == "3-segment":
        return detect_smo2_breakpoints_segmented(df, smo2_col, power_col, **kwargs)
    elif method == "exp-dmax":
        result = detect_exp_dmax(df, smo2_col, power_col, baseline=baseline, **kwargs)
        # Convert ExpDmaxResult to SmO2Breakpoints
        return SmO2Breakpoints(
            bp1_power=result.breakpoint_power,
            bp1_smo2=result.breakpoint_smo2,
            r_squared=result.r_squared,
            method="exp-dmax",
            is_valid=result.is_valid,
            notes=result.notes,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use '2-segment', '3-segment', or 'exp-dmax'")
