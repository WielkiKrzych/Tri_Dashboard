# Domain Model — Physiological Thresholds

> **Purpose:** Canonical reference for threshold detection in Tri_Dashboard. Defines
> the contracts between signal processing, detection, and UI layers so the same
> metric is computed and displayed consistently everywhere.
>
> **Audience:** Developers extending threshold logic or UI tabs. For the physiology
> and signal-processing algorithms themselves, see
> [`physiological_methodology.md`](./physiological_methodology.md).

---

## 1. Canonical entry points

Each physiological domain has **one** recommended entry point. Do not call
lower-level helpers directly from UI or orchestration code.

| Domain | Entry point | File | Returns |
|---|---|---|---|
| Ventilatory thresholds (CPET) | `detect_vt_cpet()` | `modules/calculations/vt_cpet.py` | `dict` (see §3.1) |
| Ventilatory thresholds (step test) | `detect_vt_from_steps()` | `modules/calculations/vt_step.py` | `StepVTResult` |
| VT sliding-window zones | `detect_vt_transition_zone()` | `modules/calculations/vt_sliding.py` | `(TransitionZone, TransitionZone)` |
| SmO₂ thresholds (ramp) | `detect_smo2_thresholds_moxy()` | `modules/calculations/smo2_thresholds.py` | `SmO2ThresholdResult` |
| Ramp classification | `classify_ramp_test()` | `modules/domain/session_type.py` | `RampClassificationResult` |
| Session-type classification | `classify_session_type()` | `modules/domain/session_type.py` | `SessionType` |
| Manual-override resolution | `resolve_metric()`, `resolve_all_thresholds()` | `modules/canonical_values.py` | value or dict |
| Pipeline orchestration (ramp) | `run_ramp_test_pipeline()` | `modules/calculations/pipeline.py` | integrated dict |

**UI tabs MUST consume these entry points** and must not re-implement detection
locally. If a tab needs a derived metric, add it to the result object or build a
thin read-only wrapper in `modules/domain/`.

---

## 2. Column aliases and units

All calculation modules lowercase + strip column names on entry. Primary names
are what you should use when writing new code; aliases are accepted for backward
compatibility with uploaded CSVs.

| Signal | Primary column | Accepted aliases | Unit |
|---|---|---|---|
| Power | `watts` | `power` | W |
| Ventilation | `tymeventilation` | — | L/min (L/s auto-detected and converted) |
| VO₂ | `tymevo2` | — | L/min |
| VCO₂ | `tymevco2` | — | L/min |
| SmO₂ | `smo2` | — | % (0–100) |
| Heart rate | `hr` | `heart_rate`, `heart rate`, `bpm`, `tętno`, `heartrate`, `heart_rate_bpm` | bpm |
| Breath rate | `tymebreathrate` | `br`, `rr`, `breath_rate` | breaths/min |
| Time | `time` | — | seconds |
| ATT (adipose thickness) | `att_mm` | — | mm |

**Validation ranges** (reject or flag as low-quality outside these):

| Metric | Range | Source |
|---|---|---|
| Power | 0 – 2 500 W | `common.py` |
| VE | 5 – 250 L/min | `vt_cpet_preprocessing.py` |
| RER (at VT2) | 0.95 – 1.15 ideal; reject > 1.25 | `vt_cpet_gas_exchange.py` |
| HR | user-provided `hr_max` bounds | UI settings |
| SmO₂ | 40 – 90 % typical (individual baseline varies) | — |

**Unit conversion rule:** VE is the only signal with automatic unit detection.
If `VE.mean() < 10`, assume L/s and multiply by 60. All other signals must
arrive in their primary unit.

---

## 3. Result objects

### 3.1 `detect_vt_cpet()` — dict keys

```python
{
    "vt1_watts": int | None,
    "vt2_watts": int | None,
    "vt1_hr": int | None, "vt2_hr": int | None,
    "vt1_ve": float | None, "vt2_ve": float | None,         # L/min
    "vt1_br": int | None, "vt2_br": int | None,             # breaths/min
    "vt1_vo2": float | None, "vt2_vo2": float | None,
    "vt1_pct_vo2max": float | None, "vt2_pct_vo2max": float | None,
    "vt1_confidence": float | None, "vt2_confidence": float | None,   # 0.0–1.0
    "vt1_range_low": float | None, "vt1_range_high": float | None,
    "vt2_range_low": float | None, "vt2_range_high": float | None,
    "df_steps": pd.DataFrame,
    "method": "cpet_segmented_regression",
    "has_gas_exchange": bool,
    "analysis_notes": list[str],
    "validation": dict,
}
```

### 3.2 `StepVTResult` (see `modules/calculations/threshold_types.py`)

Range-based output with `TransitionZone` for VT1/VT2 plus legacy point values
(`vt1_watts`, `vt1_hr`, …) for compatibility.

### 3.3 `SmO2ThresholdResult` (see `modules/calculations/threshold_types.py`)

Fields:
- **T1** (LT1 analog): `t1_watts`, `t1_hr`, `t1_smo2`, `t1_gradient`, `t1_trend`, `t1_sd`, `t1_step`
- **T2_onset** (RCP analog): `t2_onset_watts`, `t2_onset_hr`, `t2_onset_smo2`, `t2_onset_gradient`, `t2_onset_curvature`, `t2_onset_sd`, `t2_onset_step`
- **T2_steady** (MLSS analog): present in the dataclass but **MUST remain None for ramp tests**. Only valid for constant-load step protocols.
- **Correlations**: `vt1_correlation_watts`, `rcp_onset_correlation_watts`, `rcp_steady_correlation_watts`, `physiological_agreement` ∈ {`"strong"`, `"moderate"`, `"weak"`, `"not_checked"`}
- **Validation**: `att_mm`, `att_warning`, `att_unreliable`, `analysis_notes`, `method`, `step_data`, `zones`

### 3.4 `TransitionZone` (see `modules/calculations/threshold_types.py`)

```python
TransitionZone(
    range_watts: tuple[float, float],
    range_hr: tuple[float, float] | None,
    confidence: float,          # 0.0–1.0
    stability_score: float,     # 0.0–1.0
    method: str,
    detection_sources: list[str],
)
```
Methods: `is_high_confidence(threshold=0.7)`, `is_stable(threshold=0.7)`.

### 3.5 `RampClassificationResult` (see `modules/domain/session_type.py`)

```python
RampClassificationResult(
    is_ramp: bool,
    confidence: float,              # met_criteria / 4
    reason: str,
    criteria_met: list[str],
    criteria_failed: list[str],
    suggested_type: SessionType,    # RAMP_TEST | RAMP_TEST_CONDITIONAL | TRAINING
)
```
Thresholds: ≥ 3/4 criteria → RAMP_TEST (conf ≥ 0.75); 2/4 → RAMP_TEST_CONDITIONAL
(conf 0.50); < 2 → TRAINING.

---

## 4. Confidence model

Confidence is a float in `[0.0, 1.0]` attached to every detected threshold.
The canonical producer functions live in `modules/calculations/threshold_types.py`:

- `calculate_detection_confidence(...)` — agreement between detection sources
- `calculate_temporal_stability(...)` — stability across time windows / sensitivity
- `create_transition_zone(...)` — composes the two above into a `TransitionZone`

### Reference bands (display-layer contract)

| Confidence | Band | Meaning for coach/physiologist |
|---|---|---|
| ≥ 0.80 | **HIGH** | Treat the value as a reliable measurement. |
| 0.60 – 0.79 | **CONDITIONAL** | Use with caveats (short protocol, signal noise, one source). |
| 0.40 – 0.59 | **LOW** | Estimate only; flag to the user. |
| < 0.40 | **INVALID** | Do not use; show reason, not a number. |

UI tabs MUST render these bands consistently (badge / color / caveat text).
Helper: `modules/domain/data_quality.py::quality_band(confidence)` (to be added
in Phase 3 of the refactor).

### Factors that lower confidence

| Factor | Penalty | Source |
|---|---|---|
| RER > 1.0 at VT1 | −0.20 | `vt_cpet_gas_exchange.py` |
| RER < 0.95 at VT2 (submaximal) | −0.25 | `vt_cpet_gas_exchange.py` |
| RER 1.15 – 1.25 at VT2 | −0.15 | `vt_cpet_gas_exchange.py` |
| RER > 1.25 at VT2 | reject VT2 | `vt_cpet_gas_exchange.py` |
| SmO₂ CV > 6 % per step | reject step | `smo2_thresholds.py` |
| ATT > 10 mm | warning | `smo2_thresholds.py` |
| ATT > 15 mm | unreliable | `smo2_thresholds.py` |
| VT2 > 95 % of Pmax | sanity warning | `common.py` |
| Dropout rate > 20 % | −0.5 signal score | `quality.py` |
| Single detection source | no bonus | `threshold_types.py` |
| Additional source agreement | +0.05 per source | `threshold_types.py` |

---

## 5. Manual overrides

User-entered thresholds always win. Resolution happens via
`modules/canonical_values.py::resolve_metric(name)`:

1. Manual override in `st.session_state` (set by Manual Thresholds tab).
2. Auto-detected value from pipeline.
3. `None` if neither.

`resolve_all_thresholds()` returns the resolved dict for Summary consumption.
**Any UI tab that reads a threshold from `st.session_state` directly is a bug**
— it should use `canonical_values` to respect manual overrides.

---

## 6. Cross-source interpretation

VT (ventilation) and SmO₂ are independent signals. Their thresholds should
roughly coincide, but SmO₂ is a **local** signal (one muscle group) and is
supporting evidence only — it must not override VT on its own.

The SmO₂ result exposes `physiological_agreement` against VT correlates. Phase 4
of the refactor adds a unified cross-check surface in
`modules/domain/threshold_crosscheck.py` that compares all available sources
(VT_ventilation, T_SmO2, HR-based) and returns an `Agreement | Conflict`
verdict with delta in watts, consumed by the Summary tab.

**Rules of thumb** for interpreting deltas between sources:

| |Δ watts| | Verdict |
|---|---|
| ≤ 10 W | Strong agreement |
| 10 – 20 W | Moderate agreement |
| 20 – 30 W | Weak — investigate data quality |
| > 30 W | Conflict — surface to user, do not average silently |

---

## 7. UI tab contract

| Tab | File | Must consume | Must NOT |
|---|---|---|---|
| Vent Thresholds | `modules/ui/vent_thresholds.py` | `detect_vt_cpet()` or `detect_vt_from_steps()` | Recompute VT locally |
| SmO₂ Thresholds | `modules/ui/smo2_thresholds.py` | `detect_smo2_thresholds_moxy()` | Redefine T1/T2 |
| SmO₂ Manual | `modules/ui/smo2_manual_thresholds.py` | `canonical_values.resolve_metric()` | Write directly to session_state without going through the resolver |
| Manual Thresholds | `modules/ui/manual_thresholds.py` | Writes overrides consumed by `canonical_values` | Bypass the resolver |
| Summary | `modules/ui/summary.py`, `summary_thresholds.py` | `resolve_all_thresholds()` + cross-check | Call detection functions itself |
| Threshold Analysis | `modules/ui/threshold_analysis_ui.py` | VT + SmO₂ results from session_state | Re-run detection |

---

## 8. Glossary

| Term | Definition in this app |
|---|---|
| **VT1 / LT1** | First ventilatory threshold. Aerobic → moderate domain transition. |
| **VT2 / LT2 / RCP** | Second ventilatory threshold / Respiratory Compensation Point. Moderate → heavy-severe transition. |
| **T1** | SmO₂ analog of LT1 (desaturation slope change). |
| **T2_onset** | SmO₂ analog of RCP in a **ramp** protocol. Only valid in ramps. |
| **T2_steady** | SmO₂ analog of MLSS in a **constant-load step** protocol. **Must be None in ramps.** |
| **CP** | Critical Power (hyperbolic P-t model asymptote). Distinct from VT2. |
| **Pmax** | Peak power reached during the test. |
| **ATT** | Adipose Tissue Thickness at the NIRS sensor site. Affects SmO₂ signal validity. |
| **CPET** | Cardio-Pulmonary Exercise Test (gas-exchange data available). |
| **Ramp test** | Monotonic stepped protocol ending at exhaustion. Detected by `classify_ramp_test()`. |
| **Step test** | Discrete constant-load steps (longer than ramp steps). |
