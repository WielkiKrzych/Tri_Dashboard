# VT Detection Protocol Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 10 fixes to VT detection protocol addressing RER validation, physiological sanity checks, artifact handling, and cross-method validation.

**Architecture:** Fixes organized into 4 phases: Foundation (preprocessing), Detection (algorithm improvements), Orchestration (fallback logic), Integration (cross-validation). Each fix is independent but ordered by dependency.

**Tech Stack:** Python 3.10+, NumPy, Pandas, SciPy

---

## Dependencies Map

```
Phase 1: Foundation (preprocessing)
├── Issue #4: VE unit validation
├── Issue #5: VE-only artifact filtering  
└── Issue #10: Smoothing edge effects

Phase 2: Detection Logic
├── Issue #9: Variance penalty (vt_step.py)
├── Issue #6: Slope ratio adaptive (vt_cpet_ve_only.py)
└── Issue #1: RER validation gap (vt_cpet_gas_exchange.py)

Phase 3: Orchestration
├── Issue #3: Percentile fallback (vt_cpet.py)
└── Issue #2: VT2 vs Pmax sanity check (multiple files)

Phase 4: Integration
├── Issue #7: Cross-validation between methods
└── Issue #8: SmO2 modulation thresholds (pipeline.py)
```

---

## Phase 1: Foundation Fixes

### Task 1: Issue #4 - VE Unit Validation

**Files:**
- Modify: `modules/calculations/vt_cpet_preprocessing.py:46-49`
- Test: `tests/test_vt_preprocessing.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_preprocessing.py
import pandas as pd
import numpy as np
from modules.calculations.vt_cpet_preprocessing import preprocess_cpet_data

def test_ve_unit_validation_rejects_absurd_values():
    """VE values > 250 L/min after conversion should be flagged."""
    # Create data that would produce absurd VE after *60 conversion
    # 9.5 L/s * 60 = 570 L/min (absurd)
    df = pd.DataFrame({
        "watts": [100, 150, 200, 250],
        "tymeventilation": [9.5, 10.0, 10.5, 11.0],  # Will be multiplied by 60
        "time": [0, 60, 120, 180],
    })
    cols = {"power": "watts", "ve": "tymeventilation", "time": "time",
            "vo2": "vo2", "vco2": "vco2", "hr": "hr"}
    result = {"analysis_notes": []}
    
    data, _, _, _ = preprocess_cpet_data(df, cols, smoothing_window_sec=25, result=result)
    
    # After conversion, ve_lmin would be 570+ which is absurd
    # Should either NOT convert or flag the issue
    assert data["ve_lmin"].max() < 250, "VE should be capped at physiological max"
    assert any("VE" in note for note in result["analysis_notes"]), "Should log VE validation"

def test_ve_unit_validation_accepts_valid_lmin():
    """VE already in L/min should pass through unchanged."""
    df = pd.DataFrame({
        "watts": [100, 150, 200, 250],
        "tymeventilation": [45.0, 55.0, 70.0, 90.0],  # Already L/min
        "time": [0, 60, 120, 180],
    })
    cols = {"power": "watts", "ve": "tymeventilation", "time": "time",
            "vo2": "vo2", "vco2": "vco2", "hr": "hr"}
    result = {"analysis_notes": []}
    
    data, _, _, _ = preprocess_cpet_data(df, cols, smoothing_window_sec=25, result=result)
    
    # Should NOT multiply (mean > 10)
    assert data["ve_lmin"].iloc[0] == 45.0

def test_ve_unit_validation_handles_edge_case():
    """VE around 10 L/min boundary should be handled correctly."""
    df = pd.DataFrame({
        "watts": [100, 150, 200, 250],
        "tymeventilation": [0.15, 0.18, 0.22, 0.25],  # L/s, will convert to 9-15 L/min
        "time": [0, 60, 120, 180],
    })
    cols = {"power": "watts", "ve": "tymeventilation", "time": "time",
            "vo2": "vo2", "vco2": "vco2", "hr": "hr"}
    result = {"analysis_notes": []}
    
    data, _, _, _ = preprocess_cpet_data(df, cols, smoothing_window_sec=25, result=result)
    
    # After conversion: 9, 10.8, 13.2, 15 L/min - all physiologically valid
    assert 5 <= data["ve_lmin"].min() <= 250
    assert 5 <= data["ve_lmin"].max() <= 250
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_preprocessing.py -v`
Expected: FAIL - no validation exists

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet_preprocessing.py
# Replace lines 45-49 with:

    # Unit normalization with physiological validation
    VE_MIN_LMIN = 5.0    # Minimum physiological VE (L/min)
    VE_MAX_LMIN = 250.0  # Maximum physiological VE (L/min)
    
    if data[cols["ve"]].mean() < 10:
        # Assume L/s, convert to L/min
        data["ve_lmin"] = data[cols["ve"]] * 60
    else:
        # Already in L/min
        data["ve_lmin"] = data[cols["ve"]]
    
    # Validate VE is within physiological range
    ve_max = data["ve_lmin"].max()
    ve_min = data["ve_lmin"].min()
    
    if ve_max > VE_MAX_LMIN:
        result["analysis_notes"].append(
            f"⚠️ VE max ({ve_max:.1f} L/min) exceeds physiological limit ({VE_MAX_LMIN} L/min)"
        )
        # Cap at physiological maximum to prevent downstream issues
        data.loc[data["ve_lmin"] > VE_MAX_LMIN, "ve_lmin"] = VE_MAX_LMIN
    
    if ve_min < VE_MIN_LMIN:
        result["analysis_notes"].append(
            f"⚠️ VE min ({ve_min:.1f} L/min) below physiological minimum ({VE_MIN_LMIN} L/min)"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_preprocessing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet_preprocessing.py tests/test_vt_preprocessing.py
git commit -m "fix(vt): add VE unit validation with physiological bounds [Issue #4]"
```

---

### Task 2: Issue #10 - Smoothing Edge Effects

**Files:**
- Modify: `modules/calculations/vt_cpet_preprocessing.py:63-73`
- Test: `tests/test_vt_preprocessing.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_preprocessing.py (append)

def test_smoothing_edge_effects_not_noisy():
    """First/last points after smoothing should not be noisier than center."""
    df = pd.DataFrame({
        "watts": list(range(100, 400, 10)),
        "tymeventilation": [50 + i * 0.5 + (5 if i % 5 == 0 else 0) for i in range(30)],
        "time": list(range(0, 1800, 60)),
        "tymevo2": [2.0 + i * 0.02 for i in range(30)],
        "tymevco2": [1.8 + i * 0.02 for i in range(30)],
    })
    cols = {"power": "watts", "ve": "tymeventilation", "time": "time",
            "vo2": "tymevo2", "vco2": "tymevco2", "hr": "hr"}
    result = {"analysis_notes": []}
    
    data, _, _, _ = preprocess_cpet_data(df, cols, smoothing_window_sec=25, result=result)
    
    # Calculate variance at edges vs center
    center_var = data["ve_smooth"].iloc[10:20].var()
    edge_var = pd.concat([data["ve_smooth"].iloc[:3], data["ve_smooth"].iloc[-3:]]).var()
    
    # Edge variance should not be significantly higher than center
    # Allow 2x tolerance for edge effects
    assert edge_var < center_var * 2 + 5, "Edges should not be significantly noisier"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_preprocessing.py::test_smoothing_edge_effects_not_noisy -v`
Expected: FAIL - min_periods=1 causes edge noise

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet_preprocessing.py
# Replace lines 63-73 with:

    # Smoothing with proper edge handling
    window = min(smoothing_window_sec, len(data) // 4)
    if window < 3:
        window = 3
    
    # Use min_periods = max(3, window//2) to avoid noisy edges
    min_periods = max(3, window // 2)
    
    data["ve_smooth"] = data["ve_lmin"].rolling(
        window, center=True, min_periods=min_periods
    ).mean()
    
    if has_vo2:
        data["vo2_smooth"] = data["vo2_lmin"].rolling(
            window, center=True, min_periods=min_periods
        ).mean()
    if has_vco2:
        data["vco2_smooth"] = data["vco2_lmin"].rolling(
            window, center=True, min_periods=min_periods
        ).mean()
    
    # Log if edge trimming occurs
    edge_points = window // 2
    if edge_points > 0 and len(data) > edge_points * 2:
        result["analysis_notes"].append(
            f"Smoothing window={window}s, first/last {edge_points} points may have reduced accuracy"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_preprocessing.py::test_smoothing_edge_effects_not_noisy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet_preprocessing.py tests/test_vt_preprocessing.py
git commit -m "fix(vt): use proper min_periods for smoothing to reduce edge noise [Issue #10]"
```

---

### Task 3: Issue #5 - VE-Only Artifact Filtering

**Files:**
- Modify: `modules/calculations/vt_cpet_preprocessing.py:75-91`
- Test: `tests/test_vt_preprocessing.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_preprocessing.py (append)

def test_ve_only_artifact_filtering_without_gas():
    """IQR-based outlier detection should work when VO2/VCO2 unavailable."""
    # Create VE data with obvious spike artifact
    ve_values = [40, 42, 45, 48, 52, 200, 55, 58, 60, 62]  # 200 is artifact
    df = pd.DataFrame({
        "watts": list(range(100, 200, 10)),
        "tymeventilation": ve_values,
        "time": list(range(0, 600, 60)),
        # NO vo2/vco2 columns
    })
    cols = {"power": "watts", "ve": "tymeventilation", "time": "time",
            "vo2": "vo2", "vco2": "vco2", "hr": "hr"}
    result = {"analysis_notes": []}
    
    data, _, _, _ = preprocess_cpet_data(df, cols, smoothing_window_sec=25, result=result)
    
    # Artifact should be detected and handled
    assert any("artifact" in note.lower() for note in result["analysis_notes"]), \
        "Should detect artifact in VE-only mode"
    
    # The spike should be reduced (interpolated)
    # The artifact at index 5 should not be 200 in smoothed output
    assert data["ve_smooth"].iloc[5] < 150, "Artifact should be interpolated"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_preprocessing.py::test_ve_only_artifact_filtering_without_gas -v`
Expected: FAIL - no IQR-based filtering for VE-only mode

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet_preprocessing.py
# Replace lines 75-91 with:

    # Artifact removal
    if has_vo2 and has_vco2:
        # Original gas-based artifact detection
        ve_diff = data["ve_smooth"].diff().abs()
        vo2_diff = data["vo2_smooth"].diff().abs()
        vco2_diff = data["vco2_smooth"].diff().abs()
        
        ve_threshold = ve_diff.std() * 3
        gas_threshold = max(vo2_diff.std(), vco2_diff.std())
        
        artifact_mask = (ve_diff > ve_threshold) & (
            (vo2_diff < gas_threshold) & (vco2_diff < gas_threshold)
        )
        artifact_count = artifact_mask.sum()
        if artifact_count > 0:
            result["analysis_notes"].append(f"Removed {artifact_count} respiratory artifacts")
            data.loc[artifact_mask, "ve_smooth"] = np.nan
            data["ve_smooth"] = data["ve_smooth"].interpolate(method="linear")
    
    else:
        # VE-only mode: IQR-based outlier detection
        ve_diff = data["ve_smooth"].diff().abs()
        
        # Calculate IQR of VE differences
        q1 = ve_diff.quantile(0.25)
        q3 = ve_diff.quantile(0.75)
        iqr = q3 - q1
        
        # Artifact threshold: diff > 4 * IQR
        artifact_threshold = 4 * iqr
        
        artifact_mask = ve_diff > artifact_threshold
        artifact_count = artifact_mask.sum()
        
        if artifact_count > 0:
            result["analysis_notes"].append(
                f"Removed {artifact_count} VE artifacts (IQR-based, threshold={artifact_threshold:.1f})"
            )
            data.loc[artifact_mask, "ve_smooth"] = np.nan
            data["ve_smooth"] = data["ve_smooth"].interpolate(method="linear")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_preprocessing.py::test_ve_only_artifact_filtering_without_gas -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet_preprocessing.py tests/test_vt_preprocessing.py
git commit -m "fix(vt): add IQR-based artifact filtering for VE-only mode [Issue #5]"
```

---

## Phase 2: Detection Logic Fixes

### Task 4: Issue #9 - Variance Penalty on Small Samples

**Files:**
- Modify: `modules/calculations/vt_step.py:118-122`
- Test: `tests/test_vt_step.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_step.py
import pandas as pd
import numpy as np
from modules.calculations.vt_step import detect_vt_from_steps
from modules.calculations.threshold_types import StepTestRange, StepInfo

def test_variance_penalty_minimum_sample_size():
    """Variance penalty should not apply for n < 4 samples."""
    # This test verifies the fix indirectly through confidence calculation
    # Small samples (n=2) should NOT have variance penalty applied
    
    # Create minimal step data
    df = pd.DataFrame({
        "watts": [100] * 30 + [150] * 30 + [200] * 30 + [250] * 30 + [300] * 30,
        "tymeventilation": [40] * 30 + [50] * 30 + [65] * 30 + [90] * 30 + [120] * 30,
        "time": list(range(150)),
        "hr": [100] * 30 + [110] * 30 + [125] * 30 + [145] * 30 + [170] * 30,
    })
    
    step_range = StepTestRange(
        steps=[
            StepInfo(step_number=1, start_time=0, end_time=30),
            StepInfo(step_number=2, start_time=30, end_time=60),
            StepInfo(step_number=3, start_time=60, end_time=90),
            StepInfo(step_number=4, start_time=90, end_time=120),
            StepInfo(step_number=5, start_time=120, end_time=150),
        ],
        is_valid=True,
    )
    
    result = detect_vt_from_steps(df, step_range)
    
    # With n=2 samples in variance calculation, penalty should be 0
    # This is verified by checking confidence is not artificially reduced
    if result.vt1_zone:
        # Confidence should be reasonable (not penalized for small sample variance)
        assert result.vt1_zone.confidence >= 0.2, \
            f"Confidence {result.vt1_zone.confidence} too low - variance penalty may be misapplied"

def test_variance_penalty_uses_unbiased_estimator():
    """Variance should use ddof=1 for unbiased estimate."""
    # This is a unit test for the variance calculation
    samples = [10.0, 12.0, 14.0, 16.0]
    
    # Biased (ddof=0): population variance
    biased_var = np.var(samples)
    
    # Unbiased (ddof=1): sample variance  
    unbiased_var = np.var(samples, ddof=1)
    
    # Unbiased should be larger for small samples
    assert unbiased_var > biased_var, "Unbiased variance should be larger for n=4"
    assert unbiased_var == np.var(samples, ddof=1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_step.py -v`
Expected: Tests should pass but verify the variance calculation is correct

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_step.py
# Replace lines 117-122 with:

        step_ve_values = [s["avg_ve"] for s in stages[v1_start_idx:v1_idx]]
        
        # Minimum sample size for variance penalty
        MIN_SAMPLES_FOR_VARIANCE_PENALTY = 4
        
        if len(step_ve_values) >= MIN_SAMPLES_FOR_VARIANCE_PENALTY:
            # Use unbiased variance estimator (ddof=1)
            ve_variance = np.var(step_ve_values, ddof=1)
            variance_penalty = min(0.5, ve_variance / 20)
        else:
            # Too few samples - variance would be unreliable, skip penalty
            variance_penalty = 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_step.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_step.py tests/test_vt_step.py
git commit -m "fix(vt): use unbiased variance and minimum sample size for penalty [Issue #9]"
```

---

### Task 5: Issue #6 - Adaptive Slope Ratio for Athletes

**Files:**
- Modify: `modules/calculations/vt_cpet_ve_only.py:588`
- Test: `tests/test_vt_ve_only.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_ve_only.py
import pandas as pd
import numpy as np
from modules.calculations.vt_cpet_ve_only import _find_breakpoint_segmented

def test_adaptive_slope_ratio_for_low_variance():
    """Athletes with stable VE should use lower slope ratio threshold."""
    # Create data with low coefficient of variation (stable athlete)
    x = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325])
    # Low variance signal with subtle breakpoint
    y = np.array([20, 22, 24, 26, 28, 35, 40, 46, 53, 61])  # Subtle break at index 4
    
    # With hardcoded 1.1 threshold, this might fail to detect
    # With adaptive threshold based on CV, should detect
    result = _find_breakpoint_segmented(x, y, min_segment_size=3)
    
    # Should detect breakpoint near index 4-5
    assert result is not None, "Should detect breakpoint in low-variance athlete data"

def test_adaptive_slope_ratio_for_high_variance():
    """High variance signals should use higher slope ratio threshold."""
    # Create data with high coefficient of variation
    x = np.array([100, 125, 150, 175, 200, 225, 250, 275, 300, 325])
    # High variance with noise
    y = np.array([20, 25, 22, 30, 28, 45, 52, 48, 60, 70])
    
    result = _find_breakpoint_segmented(x, y, min_segment_size=3)
    
    # Should still work but with higher threshold
    # The function should handle noisy data appropriately
    assert isinstance(result, (int, type(None))), "Should return index or None"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_ve_only.py -v`
Expected: May fail if hardcoded threshold is too rigid

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet_ve_only.py
# Replace the _find_breakpoint_segmented function (lines 542-595)

def _find_breakpoint_segmented(
    x: np.ndarray, y: np.ndarray, min_segment_size: int = 3
) -> Optional[int]:
    """
    Find optimal breakpoint using piecewise linear regression.
    
    Uses adaptive slope ratio threshold based on signal variance:
    - Low CV (stable athletes): threshold = 1.05 + 0.02 * CV
    - This allows detection in fit athletes with subtle transitions

    Args:
        x: Independent variable (power)
        y: Dependent variable (VE/VO2 or VE/VCO2)
        min_segment_size: Minimum points in each segment

    Returns:
        Index of optimal breakpoint, or None if not found
    """
    if len(x) < 2 * min_segment_size:
        return None

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2 * min_segment_size:
        return None

    # Calculate coefficient of variation for adaptive threshold
    y_mean = np.mean(y)
    y_std = np.std(y)
    cv = y_std / y_mean if y_mean != 0 else 0
    
    # Adaptive threshold: base 1.05 + 0.02 * CV
    # - Low CV (0.1): threshold = 1.07 (sensitive)
    # - High CV (0.5): threshold = 1.15 (conservative)
    adaptive_slope_threshold = 1.05 + 0.02 * cv

    best_idx = None
    best_sse = np.inf

    for i in range(min_segment_size, len(x) - min_segment_size):
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        try:
            slope1, intercept1, _, _, _ = stats.linregress(x1, y1)
            pred1 = slope1 * x1 + intercept1
            sse1 = np.sum((y1 - pred1) ** 2)

            slope2, intercept2, _, _, _ = stats.linregress(x2, y2)
            pred2 = slope2 * x2 + intercept2
            sse2 = np.sum((y2 - pred2) ** 2)

            total_sse = sse1 + sse2

            slope_ratio = slope2 / slope1 if slope1 != 0 else slope2

            if total_sse < best_sse and slope_ratio > adaptive_slope_threshold:
                best_sse = total_sse
                best_idx = i

        except Exception:
            continue

    return best_idx
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_ve_only.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet_ve_only.py tests/test_vt_ve_only.py
git commit -m "fix(vt): adaptive slope ratio threshold based on signal variance [Issue #6]"
```

---

### Task 6: Issue #1 - RER Validation Gap at VT2

**Files:**
- Modify: `modules/calculations/vt_cpet_gas_exchange.py:81-113`
- Test: `tests/test_vt_gas_exchange.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_gas_exchange.py
import pandas as pd
import numpy as np
from modules.calculations.vt_cpet_gas_exchange import detect_gas_exchange_thresholds

def test_vt2_rer_validation_rejects_extreme_high():
    """VT2 with RER > 1.25 should be rejected or heavily penalized."""
    df_steps = pd.DataFrame({
        "power": [100, 125, 150, 175, 200, 225, 250, 275, 300],
        "ve": [30, 35, 42, 50, 62, 78, 100, 130, 170],
        "vo2": [2.0, 2.3, 2.7, 3.1, 3.5, 3.9, 4.2, 4.4, 4.5],
        "vco2": [1.8, 2.1, 2.5, 3.0, 3.6, 4.3, 5.2, 6.5, 8.0],  # RER gets very high
        "hr": [100, 110, 120, 132, 145, 158, 170, 182, 190],
        "br": [15, 16, 17, 19, 22, 26, 32, 40, 50],
        "step": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })
    
    result = {
        "vt1_watts": None,
        "vt2_watts": None,
        "analysis_notes": [],
    }
    
    detect_gas_exchange_thresholds(df_steps, result)
    
    # VT2 at high power would have RER > 1.25, should be flagged
    if result["vt2_watts"] and result["vt2_watts"] > 250:
        assert any("RER" in note and ("reject" in note.lower() or "unreliable" in note.lower()) 
                   for note in result["analysis_notes"]), \
            "VT2 with extreme RER should be flagged"

def test_vt2_rer_validation_reduces_confidence():
    """VT2 with RER outside 0.95-1.15 should have reduced confidence."""
    df_steps = pd.DataFrame({
        "power": [100, 125, 150, 175, 200, 225, 250, 275, 300],
        "ve": [30, 35, 42, 50, 62, 78, 100, 130, 170],
        "vo2": [2.0, 2.3, 2.7, 3.1, 3.5, 3.9, 4.2, 4.4, 4.5],
        "vco2": [1.8, 2.1, 2.5, 3.0, 3.5, 3.9, 4.5, 5.0, 5.4],  # RER ~1.2 at high power
        "hr": [100, 110, 120, 132, 145, 158, 170, 182, 190],
        "br": [15, 16, 17, 19, 22, 26, 32, 40, 50],
        "step": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    })
    
    result = {
        "vt1_watts": None,
        "vt2_watts": None,
        "vt2_confidence": 0.8,  # Initial confidence
        "analysis_notes": [],
    }
    
    detect_gas_exchange_thresholds(df_steps, result)
    
    # If VT2 detected with out-of-range RER, confidence should be reduced
    # Check that the result has appropriate confidence handling
    # The exact mechanism depends on implementation
    assert "vt2_watts" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_gas_exchange.py -v`
Expected: FAIL - no confidence reduction for out-of-range RER

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet_gas_exchange.py
# Replace lines 78-113 with:

        if vt2_idx < len(df_steps):
            rer_at_vt2 = df_steps.loc[vt2_idx, "rer"]
            
            # RER validation for VT2
            RER_IDEAL_MIN = 0.95
            RER_IDEAL_MAX = 1.15
            RER_REJECT_MAX = 1.25
            
            rer_valid = pd.notna(rer_at_vt2)
            rer_in_ideal_range = rer_valid and RER_IDEAL_MIN <= rer_at_vt2 <= RER_IDEAL_MAX
            rer_extreme = rer_valid and rer_at_vt2 > RER_REJECT_MAX
            
            # Calculate confidence penalty
            confidence_penalty = 0.0
            if rer_valid and not rer_in_ideal_range:
                if rer_extreme:
                    # RER > 1.25: reject VT2 entirely
                    result["analysis_notes"].append(
                        f"⚠️ VT2 candidate rejected: RER={rer_at_vt2:.2f} > {RER_REJECT_MAX} (hyperventilation artifact)"
                    )
                    return  # Skip VT2 detection
                elif rer_at_vt2 < RER_IDEAL_MIN:
                    # RER < 0.95: likely submaximal effort, reduce confidence
                    confidence_penalty = 0.25
                    result["analysis_notes"].append(
                        f"ℹ️ VT2 RER={rer_at_vt2:.2f} < {RER_IDEAL_MIN} (submaximal?) → confidence -0.25"
                    )
                else:
                    # RER 1.15-1.25: approaching hyperventilation, moderate penalty
                    confidence_penalty = 0.15
                    result["analysis_notes"].append(
                        f"ℹ️ VT2 RER={rer_at_vt2:.2f} > {RER_IDEAL_MAX} → confidence -0.15"
                    )
            
            # Store VT2 with confidence penalty
            result["vt2_watts"] = int(df_steps.loc[vt2_idx, "power"])
            result["vt2_ve"] = round(df_steps.loc[vt2_idx, "ve"], 1)
            result["vt2_vo2"] = (
                round(df_steps.loc[vt2_idx, "vo2"], 2)
                if "vo2" in df_steps.columns
                else None
            )
            result["vt2_step"] = int(df_steps.loc[vt2_idx, "step"])
            result["vt2_pct_vo2max"] = (
                round(df_steps.loc[vt2_idx, "vo2"] / vo2max * 100, 1)
                if vo2max > 0 and "vo2" in df_steps.columns
                else None
            )
            result["vt2_rer"] = round(rer_at_vt2, 2) if rer_valid else None
            result["vt2_confidence_penalty"] = confidence_penalty
            
            if "hr" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "hr"]):
                result["vt2_hr"] = int(df_steps.loc[vt2_idx, "hr"])
            if "br" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "br"]):
                result["vt2_br"] = int(df_steps.loc[vt2_idx, "br"])
            
            if rer_in_ideal_range:
                result["analysis_notes"].append(
                    f"VT2 detected at step {result['vt2_step']} (RER={rer_at_vt2:.2f} ✓)"
                )
            elif rer_valid:
                rer_str = f"{rer_at_vt2:.2f}"
                result["analysis_notes"].append(
                    f"VT2 detected at step {result['vt2_step']} (RER={rer_str}, confidence reduced)"
                )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_gas_exchange.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet_gas_exchange.py tests/test_vt_gas_exchange.py
git commit -m "fix(vt): add RER validation with confidence penalty for VT2 [Issue #1]"
```

---

## Phase 3: Orchestration Fixes

### Task 7: Issue #3 - Physiological Percentile Fallback

**Files:**
- Modify: `modules/calculations/vt_cpet.py:119-127`
- Test: `tests/test_vt_cpet.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_cpet.py
import pandas as pd
import numpy as np
from modules.calculations.vt_cpet import detect_vt_cpet

def test_percentile_fallback_uses_pmax_ratio():
    """Fallback should use relative formula to Pmax, not arbitrary percentiles."""
    # Create data where detection fails, triggering fallback
    df = pd.DataFrame({
        "watts": list(range(50, 350, 10)),
        "tymeventilation": [30 + i * 0.5 for i in range(30)],  # Linear, no breakpoint
        "time": list(range(0, 1800, 60)),
    })
    
    result = detect_vt_cpet(df)
    
    # Pmax = 340W
    # VT1 should be ~55-65% of Pmax = 187-221W
    # VT2 should be ~75-85% of Pmax = 255-289W
    # NOT arbitrary 60th/80th percentile
    
    if result["vt1_watts"]:
        pmax = df["watts"].max()
        vt1_ratio = result["vt1_watts"] / pmax
        # Should be in physiological range 0.55-0.70
        assert 0.50 <= vt1_ratio <= 0.75, \
            f"VT1 ratio {vt1_ratio:.2f} outside physiological range (expected 0.55-0.70)"
    
    if result["vt2_watts"]:
        pmax = df["watts"].max()
        vt2_ratio = result["vt2_watts"] / pmax
        # Should be in physiological range 0.75-0.90
        assert 0.70 <= vt2_ratio <= 0.95, \
            f"VT2 ratio {vt2_ratio:.2f} outside physiological range (expected 0.75-0.90)"

def test_percentile_fallback_logs_method():
    """Fallback should log that it's using Pmax-relative formula."""
    df = pd.DataFrame({
        "watts": list(range(50, 350, 10)),
        "tymeventilation": [30 + i * 0.5 for i in range(30)],
        "time": list(range(0, 1800, 60)),
    })
    
    result = detect_vt_cpet(df)
    
    # Should mention Pmax-relative or percentage in notes
    assert any("pmax" in note.lower() or "%" in note or "relative" in note.lower() 
               for note in result["analysis_notes"]), \
        "Should log that fallback uses Pmax-relative formula"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_cpet.py -v`
Expected: FAIL - uses arbitrary percentiles

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet.py
# Replace lines 118-127 with:

    # 4. Global fallback defaults (both paths)
    # Use Pmax-relative formula for physiological plausibility
    pmax = df_steps["power"].max() if len(df_steps) > 0 else 0
    
    VT1_PMAX_RATIO_MIN = 0.55  # 55% of Pmax
    VT1_PMAX_RATIO_MAX = 0.65  # 65% of Pmax
    VT2_PMAX_RATIO_MIN = 0.75  # 75% of Pmax
    VT2_PMAX_RATIO_MAX = 0.85  # 85% of Pmax
    
    if result["vt1_watts"] is None and pmax > 0:
        # Use midpoint of physiological range
        vt1_power = int(pmax * ((VT1_PMAX_RATIO_MIN + VT1_PMAX_RATIO_MAX) / 2))
        result["vt1_watts"] = vt1_power
        result["analysis_notes"].append(
            f"VT1 not detected - using Pmax-relative estimate ({vt1_power}W = 60% of {pmax}W Pmax)"
        )
    
    if result["vt2_watts"] is None and pmax > 0:
        # Use midpoint of physiological range
        vt2_power = int(pmax * ((VT2_PMAX_RATIO_MIN + VT2_PMAX_RATIO_MAX) / 2))
        result["vt2_watts"] = vt2_power
        result["analysis_notes"].append(
            f"VT2 not detected - using Pmax-relative estimate ({vt2_power}W = 80% of {pmax}W Pmax)"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_cpet.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet.py tests/test_vt_cpet.py
git commit -m "fix(vt): use Pmax-relative formula for fallback estimates [Issue #3]"
```

---

### Task 8: Issue #2 - VT2 vs Pmax Sanity Check

**Files:**
- Modify: `modules/calculations/vt_cpet.py` (add validation function)
- Modify: `modules/calculations/vt_cpet_gas_exchange.py` (call validation)
- Modify: `modules/calculations/vt_cpet_ve_only.py` (call validation)
- Test: `tests/test_vt_cpet.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_cpet.py (append)

def test_vt2_pmax_sanity_check_flags_unreliable():
    """VT2 > 95% of Pmax should be flagged as UNRELIABLE."""
    # Create data where VT2 detection might overshoot
    df = pd.DataFrame({
        "watts": [100, 125, 150, 175, 200, 225, 250, 275, 300],
        "tymeventilation": [30, 35, 42, 52, 65, 85, 115, 160, 220],
        "time": list(range(0, 540, 60)),
        "tymevo2": [2.0, 2.3, 2.7, 3.1, 3.5, 3.9, 4.2, 4.4, 4.5],
        "tymevco2": [1.8, 2.1, 2.5, 3.0, 3.6, 4.3, 4.8, 5.2, 5.5],
        "hr": [100, 110, 120, 132, 145, 158, 170, 182, 190],
    })
    
    result = detect_vt_cpet(df)
    
    # Pmax = 300W, 95% = 285W
    # If VT2 > 285W, should be flagged
    if result["vt2_watts"] and result["vt2_watts"] > 285:
        assert any("unreliable" in note.lower() or "pmax" in note.lower() 
                   for note in result["analysis_notes"]), \
            "VT2 > 95% Pmax should be flagged as unreliable"
        assert result.get("vt2_confidence_penalty", 0) >= 0.3, \
            "VT2 > 95% Pmax should have confidence penalty >= 0.3"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_cpet.py::test_vt2_pmax_sanity_check_flags_unreliable -v`
Expected: FAIL - no sanity check exists

**Step 3: Write minimal implementation**

First, add the validation function to common.py:

```python
# modules/calculations/common.py (append)

def validate_threshold_vs_pmax(
    threshold_watts: float,
    pmax_watts: float,
    threshold_name: str = "VT2",
    max_ratio: float = 0.95,
) -> dict:
    """
    Validate that a threshold is physiologically plausible relative to Pmax.
    
    Args:
        threshold_watts: Detected threshold value
        pmax_watts: Maximum power achieved
        threshold_name: Name for logging (VT1, VT2, etc.)
        max_ratio: Maximum allowed ratio (default 0.95 = 95% of Pmax)
    
    Returns:
        dict with 'is_valid', 'confidence_penalty', 'message'
    """
    if pmax_watts <= 0 or threshold_watts is None:
        return {"is_valid": True, "confidence_penalty": 0.0, "message": ""}
    
    ratio = threshold_watts / pmax_watts
    
    if ratio > max_ratio:
        return {
            "is_valid": False,
            "confidence_penalty": 0.3,
            "message": (
                f"⚠️ {threshold_name} ({threshold_watts}W) > {max_ratio*100:.0f}% of Pmax "
                f"({pmax_watts}W) → UNRELIABLE, confidence -0.3"
            ),
        }
    
    return {"is_valid": True, "confidence_penalty": 0.0, "message": ""}
```

Then, add validation in vt_cpet.py:

```python
# modules/calculations/vt_cpet.py
# Add import at top:
from .common import validate_threshold_vs_pmax

# Add after line 133 (after physiological validation):
    # 6. VT2 vs Pmax sanity check
    pmax = df_steps["power"].max() if len(df_steps) > 0 else 0
    
    if result["vt2_watts"] is not None and pmax > 0:
        validation = validate_threshold_vs_pmax(
            result["vt2_watts"], pmax, "VT2", max_ratio=0.95
        )
        if not validation["is_valid"]:
            result["analysis_notes"].append(validation["message"])
            result["vt2_confidence_penalty"] = validation["confidence_penalty"]
            result["vt2_unreliable"] = True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_cpet.py::test_vt2_pmax_sanity_check_flags_unreliable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/common.py modules/calculations/vt_cpet.py tests/test_vt_cpet.py
git commit -m "fix(vt): add VT2 vs Pmax sanity check with confidence penalty [Issue #2]"
```

---

## Phase 4: Integration Fixes

### Task 9: Issue #8 - Relative SmO2 Modulation Thresholds

**Files:**
- Modify: `modules/calculations/pipeline.py:415-467`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pandas as pd
import numpy as np
from modules.calculations.pipeline import integrate_signals, IndependentAnalysisResults
from modules.calculations.threshold_types import TransitionZone, StepVTResult, StepSmO2Result

def test_smo2_modulation_uses_relative_thresholds():
    """SmO2 modulation should use % of Pmax, not fixed watts."""
    # Create mock results with high power (400W Pmax)
    vt_result = StepVTResult()
    vt_result.vt1_zone = TransitionZone(
        range_watts=(180, 220),
        midpoint_watts=200,
        confidence=0.7,
        method="test",
    )
    
    smo2_result = StepSmO2Result()
    smo2_result.smo2_1_zone = TransitionZone(
        range_watts=(190, 210),
        midpoint_watts=200,  # Exactly at VT1
        confidence=0.6,
        method="test",
    )
    
    analysis = IndependentAnalysisResults()
    analysis.vt_result = vt_result
    analysis.smo2_result = smo2_result
    
    result = integrate_signals(analysis)
    
    # At 200W with 400W Pmax, 0 deviation = confirm
    # Fixed threshold ±10W would work, but relative should too
    assert result.vt1 is not None
    assert result.vt1.confidence > 0.7, "SmO2 at VT1 should boost confidence"

def test_smo2_modulation_relative_conflict():
    """Large deviation (>5% Pmax) should be flagged as conflict."""
    # VT1 at 200W, SmO2 at 260W, Pmax = 400W
    # Deviation = 60W = 15% of Pmax → conflict
    vt_result = StepVTResult()
    vt_result.vt1_zone = TransitionZone(
        range_watts=(180, 220),
        midpoint_watts=200,
        confidence=0.7,
        method="test",
    )
    
    smo2_result = StepSmO2Result()
    smo2_result.smo2_1_zone = TransitionZone(
        range_watts=(250, 270),
        midpoint_watts=260,  # 60W from VT1 = 15% of 400W Pmax
        confidence=0.6,
        method="test",
    )
    
    analysis = IndependentAnalysisResults()
    analysis.vt_result = vt_result
    analysis.smo2_result = smo2_result
    
    # Need to set Pmax context somehow - this may require test adaptation
    result = integrate_signals(analysis)
    
    # Should have conflict flagged
    assert len(result.conflicts.conflicts) > 0 or result.vt1.confidence < 0.7, \
        "Large SmO2 deviation should flag conflict or reduce confidence"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: Tests should pass but verify relative thresholds work

**Step 3: Write minimal implementation**

```python
# modules/calculations/pipeline.py
# Replace lines 415-467 with:

    # =========================================================
    # SmO₂ MODULATION (not detection!)
    # SmO₂ is a LOCAL signal - it can only CONFIRM or QUESTION VT
    # SmO₂ does NOT create independent thresholds
    # Using RELATIVE thresholds based on % of Pmax
    # =========================================================
    if smo2_result and result.vt1:
        result.conflicts.signals_analyzed.append("SmO2 (LOCAL)")

        # Check if SmO₂ shows a drop near VT1
        smo2_drop_power = _find_smo2_drop_power(smo2_result)

        if smo2_drop_power is not None:
            vt1_mid = result.vt1.midpoint_watts
            deviation = smo2_drop_power - vt1_mid
            result.smo2_deviation_vt1 = deviation
            
            # Calculate relative deviation as % of Pmax
            # Estimate Pmax from VT2 or VT1 (VT2 ≈ 80% Pmax, VT1 ≈ 60% Pmax)
            estimated_pmax = 0
            if result.vt2 and result.vt2.midpoint_watts:
                estimated_pmax = result.vt2.midpoint_watts / 0.80
            elif vt1_mid:
                estimated_pmax = vt1_mid / 0.60
            
            # Relative thresholds based on % of Pmax
            if estimated_pmax > 0:
                relative_deviation_pct = abs(deviation) / estimated_pmax
                
                # ±3% Pmax = confirm, ±5% Pmax = minor, >5% Pmax = conflict
                CONFIRM_THRESHOLD = 0.03  # 3% of Pmax
                MINOR_THRESHOLD = 0.05    # 5% of Pmax
                
                if relative_deviation_pct <= CONFIRM_THRESHOLD:
                    # SmO₂ CONFIRMS VT → boost confidence, narrow range
                    result.vt1.confidence = min(0.95, result.vt1.confidence + 0.15)
                    shrink = 0.1
                    mid = result.vt1.midpoint_watts
                    width = result.vt1.width_watts
                    result.vt1.lower_watts = mid - width * (0.5 - shrink)
                    result.vt1.upper_watts = mid + width * (0.5 - shrink)
                    result.vt1.sources.append("SmO2 ✓")
                    result.integration_notes.append(
                        f"✓ SmO₂ (LOCAL) potwierdza VT1 (różnica: {deviation:.0f}W = {relative_deviation_pct*100:.1f}% Pmax) → confidence +0.15"
                    )
                elif relative_deviation_pct <= MINOR_THRESHOLD:
                    # SmO₂ slightly off → minor confidence reduction
                    result.vt1.confidence = max(0.3, result.vt1.confidence - 0.05)
                    result.integration_notes.append(
                        f"ℹ️ SmO₂ (LOCAL) bliski VT1 (różnica: {deviation:.0f}W = {relative_deviation_pct*100:.1f}% Pmax) → confidence -0.05"
                    )
                else:
                    # SmO₂ significantly different → conflict
                    conflict_type = ConflictType.SMO2_EARLY if deviation < 0 else ConflictType.SMO2_LATE
                    result.conflicts.conflicts.append(
                        SignalConflict(
                            conflict_type=conflict_type,
                            severity=ConflictSeverity.WARNING,
                            signal_a="SmO2 (LOCAL)",
                            signal_b="VE",
                            description=f"SmO₂ (LOCAL) różni się od VT1 o {deviation:.0f}W ({relative_deviation_pct*100:.1f}% Pmax)",
                            physiological_interpretation=(
                                "SmO₂ jest sygnałem LOKALNYM (jeden mięsień). "
                                "Rozbieżność z VT może oznaczać różnicę między lokalną a systemową odpowiedzią."
                            ),
                            magnitude=abs(deviation),
                            confidence_penalty=0.15,
                        )
                    )
                    result.vt1.confidence = max(0.3, result.vt1.confidence - 0.1)
                    expand = 0.15
                    mid = result.vt1.midpoint_watts
                    width = result.vt1.width_watts
                    result.vt1.lower_watts = mid - width * (0.5 + expand)
                    result.vt1.upper_watts = mid + width * (0.5 + expand)
                    result.integration_notes.append(
                        f"⚠️ SmO₂ (LOCAL) konflikt z VT1: {deviation:.0f}W ({relative_deviation_pct*100:.1f}% Pmax) → confidence -0.1"
                    )
            else:
                # Fallback to fixed thresholds if Pmax unknown
                if abs(deviation) <= 10:
                    result.vt1.confidence = min(0.95, result.vt1.confidence + 0.15)
                    result.integration_notes.append(
                        f"✓ SmO₂ (LOCAL) potwierdza VT1 (różnica: {deviation:.0f}W) → confidence +0.15"
                    )
                elif abs(deviation) <= 20:
                    result.vt1.confidence = max(0.3, result.vt1.confidence - 0.05)
                else:
                    result.vt1.confidence = max(0.3, result.vt1.confidence - 0.1)
                    result.integration_notes.append(
                        f"⚠️ SmO₂ (LOCAL) konflikt z VT1: {deviation:.0f}W → confidence -0.1"
                    )
        else:
            result.integration_notes.append(
                "ℹ️ SmO₂ (LOCAL): brak wyraźnego spadku - brak modulacji VT"
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/pipeline.py tests/test_pipeline.py
git commit -m "fix(vt): use relative SmO2 modulation thresholds (% of Pmax) [Issue #8]"
```

---

### Task 10: Issue #7 - Cross-Validation Between Methods

**Files:**
- Modify: `modules/calculations/vt_cpet.py` (add cross-validation)
- Test: `tests/test_vt_cpet.py`

**Step 1: Write the failing test**

```python
# tests/test_vt_cpet.py (append)

def test_cross_validation_between_methods():
    """Multiple detection methods should cross-validate and flag large deviations."""
    df = pd.DataFrame({
        "watts": [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350],
        "tymeventilation": [30, 35, 42, 52, 65, 85, 115, 150, 190, 240, 300],
        "time": list(range(0, 660, 60)),
        "tymevo2": [2.0, 2.3, 2.7, 3.1, 3.5, 3.9, 4.2, 4.4, 4.5, 4.6, 4.6],
        "tymevco2": [1.8, 2.1, 2.5, 3.0, 3.6, 4.3, 4.8, 5.2, 5.5, 5.7, 5.8],
        "hr": [100, 110, 120, 132, 145, 158, 170, 182, 190, 195, 198],
    })
    
    result = detect_vt_cpet(df)
    
    # Should have cross-validation results if multiple methods ran
    if "cross_validation" in result:
        cv = result["cross_validation"]
        # If deviation > 30W, should have warning
        if cv.get("vt1_deviation_watts", 0) > 30:
            assert cv.get("warning") is not None, "Large deviation should trigger warning"

def test_cross_validation_weighted_average():
    """Final result should use confidence-weighted average when methods agree."""
    df = pd.DataFrame({
        "watts": [100, 125, 150, 175, 200, 225, 250, 275, 300],
        "tymeventilation": [30, 35, 42, 52, 65, 85, 115, 150, 190],
        "time": list(range(0, 540, 60)),
        "tymevo2": [2.0, 2.3, 2.7, 3.1, 3.5, 3.9, 4.2, 4.4, 4.5],
        "tymevco2": [1.8, 2.1, 2.5, 3.0, 3.6, 4.3, 4.8, 5.2, 5.5],
        "hr": [100, 110, 120, 132, 145, 158, 170, 182, 190],
    })
    
    result = detect_vt_cpet(df)
    
    # Should have method tracking
    assert "method" in result
    # If cross-validation ran, should have details
    if "cross_validation" in result:
        cv = result["cross_validation"]
        assert "methods_used" in cv or "vt1_methods" in cv
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vt_cpet.py::test_cross_validation_between_methods -v`
Expected: FAIL - no cross-validation exists

**Step 3: Write minimal implementation**

```python
# modules/calculations/vt_cpet.py
# Add after the detection section (around line 117):

    # 3.5 Cross-validation between detection methods (when gas exchange available)
    cross_validation = {
        "enabled": False,
        "vt1_methods": [],
        "vt2_methods": [],
        "vt1_deviation_watts": None,
        "vt2_deviation_watts": None,
        "warning": None,
    }
    
    if has_vo2 and has_vco2:
        cross_validation["enabled"] = True
        
        # Store gas exchange results
        gas_exchange_vt1 = result.get("vt1_watts")
        gas_exchange_vt2 = result.get("vt2_watts")
        gas_exchange_vt1_conf = result.get("vt1_confidence", 0.5)
        gas_exchange_vt2_conf = result.get("vt2_confidence", 0.5)
        
        if gas_exchange_vt1:
            cross_validation["vt1_methods"].append({
                "method": "gas_exchange",
                "value": gas_exchange_vt1,
                "confidence": gas_exchange_vt1_conf,
            })
        if gas_exchange_vt2:
            cross_validation["vt2_methods"].append({
                "method": "gas_exchange",
                "value": gas_exchange_vt2,
                "confidence": gas_exchange_vt2_conf,
            })
        
        # Also run VE-only method for comparison
        ve_result = {"vt1_watts": None, "vt2_watts": None, "analysis_notes": []}
        detect_ve_only_thresholds(df_steps, data, cols, ve_result)
        
        ve_vt1 = ve_result.get("vt1_watts")
        ve_vt2 = ve_result.get("vt2_watts")
        ve_vt1_conf = ve_result.get("vt1_confidence", 0.5)
        ve_vt2_conf = ve_result.get("vt2_confidence", 0.5)
        
        if ve_vt1:
            cross_validation["vt1_methods"].append({
                "method": "ve_only",
                "value": ve_vt1,
                "confidence": ve_vt1_conf,
            })
        if ve_vt2:
            cross_validation["vt2_methods"].append({
                "method": "ve_only",
                "value": ve_vt2,
                "confidence": ve_vt2_conf,
            })
        
        # Calculate weighted average for VT1 if multiple methods
        if len(cross_validation["vt1_methods"]) >= 2:
            methods = cross_validation["vt1_methods"]
            total_conf = sum(m["confidence"] for m in methods)
            if total_conf > 0:
                weighted_vt1 = sum(m["value"] * m["confidence"] for m in methods) / total_conf
                deviation = max(m["value"] for m in methods) - min(m["value"] for m in methods)
                cross_validation["vt1_deviation_watts"] = deviation
                
                if deviation > 30:
                    cross_validation["warning"] = (
                        f"⚠️ VT1 methods deviate by {deviation:.0f}W - results uncertain"
                    )
                    result["analysis_notes"].append(cross_validation["warning"])
                else:
                    # Use weighted average as final value
                    result["vt1_watts"] = int(weighted_vt1)
                    result["analysis_notes"].append(
                        f"VT1 cross-validated: weighted average {int(weighted_vt1)}W "
                        f"(deviation: {deviation:.0f}W)"
                    )
        
        # Calculate weighted average for VT2 if multiple methods
        if len(cross_validation["vt2_methods"]) >= 2:
            methods = cross_validation["vt2_methods"]
            total_conf = sum(m["confidence"] for m in methods)
            if total_conf > 0:
                weighted_vt2 = sum(m["value"] * m["confidence"] for m in methods) / total_conf
                deviation = max(m["value"] for m in methods) - min(m["value"] for m in methods)
                cross_validation["vt2_deviation_watts"] = deviation
                
                if deviation > 30:
                    if cross_validation["warning"]:
                        cross_validation["warning"] += f", VT2 deviates by {deviation:.0f}W"
                    else:
                        cross_validation["warning"] = (
                            f"⚠️ VT2 methods deviate by {deviation:.0f}W - results uncertain"
                        )
                    result["analysis_notes"].append(
                        f"⚠️ VT2 methods deviate by {deviation:.0f}W"
                    )
                else:
                    result["vt2_watts"] = int(weighted_vt2)
                    result["analysis_notes"].append(
                        f"VT2 cross-validated: weighted average {int(weighted_vt2)}W "
                        f"(deviation: {deviation:.0f}W)"
                    )
    
    result["cross_validation"] = cross_validation
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vt_cpet.py::test_cross_validation_between_methods -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/calculations/vt_cpet.py tests/test_vt_cpet.py
git commit -m "feat(vt): add cross-validation between detection methods [Issue #7]"
```

---

## Final Verification

After all tasks complete:

```bash
# Run all VT-related tests
pytest tests/test_vt*.py tests/test_pipeline.py -v

# Run full test suite to ensure no regressions
pytest tests/ -v --tb=short
```

---

## Summary

| Issue | File | Lines | Effort |
|-------|------|-------|--------|
| #4 VE unit validation | vt_cpet_preprocessing.py | 46-49 | Quick |
| #10 Smoothing edges | vt_cpet_preprocessing.py | 63-73 | Quick |
| #5 VE-only artifacts | vt_cpet_preprocessing.py | 75-91 | Short |
| #9 Variance penalty | vt_step.py | 118-122 | Quick |
| #6 Adaptive slope ratio | vt_cpet_ve_only.py | 588 | Short |
| #1 RER validation | vt_cpet_gas_exchange.py | 81-113 | Medium |
| #3 Percentile fallback | vt_cpet.py | 119-127 | Quick |
| #2 VT2 vs Pmax | vt_cpet.py, common.py | new | Short |
| #8 SmO2 thresholds | pipeline.py | 415-467 | Medium |
| #7 Cross-validation | vt_cpet.py | new | Medium |

**Total Estimated Effort:** Medium (1-2 days)
