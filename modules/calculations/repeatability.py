"""
Repeatability Analysis Module.

Calculates intra-individual metabolic stability and reproducibility metrics.
Key Metrics:
- CV (Coefficient of Variation): Noise vs Signal.
- SEM (Standard Error of Measurement): Typical error range.
- Reproducibility Classification: Excellent (<3%), Good (<6%), Moderate (<10%), Unstable (>10%).
"""

import numpy as np
from typing import List, Dict, Union, Any

from modules.calculations.common import calculate_cv


def classify_reproducibility(cv: float) -> str:
    """Classify the stability of a metric based on CV."""
    if cv < 3.0:
        return "Excellent"
    elif cv < 6.0:
        return "Good"
    elif cv < 10.0:
        return "Moderate"
    else:
        return "Unstable"


def calculate_repeatability_metrics(
    sessions_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Calculate repeatability stats for a list of session results.

    Args:
        sessions_metrics: List of dicts, e.g. [{'vt1': 200}, {'vt1': 210}]

    Returns:
        Dict of stats per metric:
        {
            'vt1': {'mean': 205, 'std': 7.07, 'cv': 3.4, 'class': 'Good'},
            ...
        }
    """
    if not sessions_metrics:
        return {}

    # Aggregate values by key
    aggregated = {}
    for session in sessions_metrics:
        for k, v in session.items():
            if v is None:
                continue
            if k not in aggregated:
                aggregated[k] = []
            aggregated[k].append(float(v))

    results = {}
    for metric, values in aggregated.items():
        if len(values) < 2:
            results[metric] = {
                "mean": values[0] if values else 0,
                "std": 0,
                "cv": 0,
                "sem": 0,
                "class": "N/A (1 sample)",
            }
            continue

        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = calculate_cv(values)
        sem = std_val / np.sqrt(len(values))

        results[metric] = {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "cv": round(cv, 2),
            "sem": round(sem, 2),
            "class": classify_reproducibility(cv),
        }

    return results


def compare_session_to_baseline(
    current_metrics: Dict[str, float], baseline_stats: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, any]]:
    """
    Compare a single session against a baseline (repeatability stats).

    Args:
        current_metrics: Dict of current values {'vt1': 215}
        baseline_stats: Output from calculate_repeatability_metrics

    Returns:
        Dict with comparison details per metric.
    """
    comparison = {}

    for metric, current_val in current_metrics.items():
        if metric not in baseline_stats:
            continue

        stats = baseline_stats[metric]
        baseline_mean = stats.get("mean", 0)
        baseline_cv = stats.get("cv", 0)

        if baseline_mean == 0:
            pct_diff = 0
        else:
            pct_diff = ((current_val - baseline_mean) / baseline_mean) * 100.0

        # Interpretation
        # If change is > 1.5-2x CV, it's likely significant
        threshold_cv = max(baseline_cv, 1.0)  # Min threshold 1%

        is_significant = abs(pct_diff) > (threshold_cv * 1.5)

        status = "Stable"
        if is_significant:
            status = (
                "Significant Change" if abs(pct_diff) > (threshold_cv * 2.0) else "Possible Change"
            )

        comparison[metric] = {
            "current": current_val,
            "baseline": baseline_mean,
            "pct_diff": round(pct_diff, 1),
            "is_significant": is_significant,
            "status": status,
            "baseline_cv": baseline_cv,
        }

    return comparison


def calculate_icc(values: List[float]) -> Dict[str, Any]:
    """
    Calculate simplified single-subject ICC with bootstrap 95% CI.

    Args:
        values: List of >=3 repeated measurements for a single metric.

    Returns:
        Dict with icc, ci_lower, ci_upper, interpretation.

    Raises:
        ValueError: If fewer than 3 measurements are provided.
    """
    if len(values) < 3:
        raise ValueError("ICC requires at least 3 measurements")

    values_arr = np.array(values, dtype=float)
    mean_val = float(np.mean(values_arr))
    std_val = float(np.std(values_arr, ddof=1))
    cv = calculate_cv(values_arr, percentage=False)
    icc = max(0.0, 1.0 - cv**2)

    # Bootstrap 95% CI (1000 resamples)
    rng = np.random.default_rng(seed=42)
    bootstrap_iccs: List[float] = []
    for _ in range(1000):
        sample = rng.choice(values_arr, size=len(values_arr), replace=True)
        s_mean = float(np.mean(sample))
        s_std = float(np.std(sample, ddof=1))
        s_cv = s_std / s_mean if s_mean != 0 else 0.0
        bootstrap_iccs.append(max(0.0, 1.0 - s_cv**2))

    ci_lower = float(np.percentile(bootstrap_iccs, 2.5))
    ci_upper = float(np.percentile(bootstrap_iccs, 97.5))

    # Interpretation
    if icc > 0.90:
        interpretation = "Excellent"
    elif icc > 0.75:
        interpretation = "Good"
    elif icc > 0.50:
        interpretation = "Moderate"
    else:
        interpretation = "Poor"

    return {
        "icc": round(icc, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "interpretation": interpretation,
    }


def bland_altman_data(values: List[float]) -> Dict[str, Any]:
    """
    Compute Bland-Altman statistics for consecutive measurement pairs.

    Args:
        values: List of >=3 repeated measurements.

    Returns:
        Dict with means, diffs, mean_diff, upper_loa, lower_loa, sd_diff.

    Raises:
        ValueError: If fewer than 3 measurements are provided.
    """
    if len(values) < 3:
        raise ValueError("Bland-Altman requires at least 3 measurements")

    means: List[float] = []
    diffs: List[float] = []
    for i in range(len(values) - 1):
        a, b = float(values[i]), float(values[i + 1])
        means.append((a + b) / 2.0)
        diffs.append(a - b)

    diffs_arr = np.array(diffs)
    mean_diff = float(np.mean(diffs_arr))
    sd_diff = float(np.std(diffs_arr, ddof=1))
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    return {
        "means": means,
        "diffs": diffs,
        "mean_diff": round(mean_diff, 3),
        "upper_loa": round(upper_loa, 3),
        "lower_loa": round(lower_loa, 3),
        "sd_diff": round(sd_diff, 3),
    }


# --- Ported from Analiza Kolarska ---

def calculate_sem(values):
    """Calculate Standard Error of Measurement.

    SEM = SD / sqrt(n)
    Represents the typical error range for repeated measurements.

    Args:
        values: List of float measurements

    Returns:
        float: SEM value, or 0.0 if insufficient data
    """
    import numpy as np
    if len(values) < 2:
        return 0.0
    std_val = np.std(values, ddof=1)
    return float(std_val / np.sqrt(len(values)))


# --- Ported from Analiza Kolarska: standalone CV ---
from modules.calculations.common import calculate_cv as _cv_from_common
def calculate_cv(values):
    """Calculate Coefficient of Variation (%)."""
    return _cv_from_common(values, percentage=True)
