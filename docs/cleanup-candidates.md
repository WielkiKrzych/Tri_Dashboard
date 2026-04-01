# Cleanup Candidates

Tracks dead code, legacy scripts, and duplication discovered during codebase audits.

---

## Completed

| Item | File(s) | Action | Date |
|:-----|:--------|:-------|:-----|
| Sidebar UI module | `modules/ui/sidebar.py` | **Deleted** — verified zero imports across the codebase | 2026-04-01 |

---

## Pending — Low Risk (documentation / labels only)

### 1. Legacy manual scripts (labelled)

| File | Status | Risk | Notes |
|:-----|:-------|:-----|:------|
| `train_history.py` | **Labelled** `[LEGACY / MANUAL SCRIPT]` in docstring | None | Standalone CLI; not imported by `app.py`. Safe to keep. |
| `init_db.py` | **Labelled** `[LEGACY / MANUAL SCRIPT]` in docstring | None | Standalone CLI; not imported by `app.py`. Safe to keep. |

**Next step**: No further action needed. Labels make intent explicit.

---

## Pending — Under Investigation

### 2. Duplicate `polars_adapter.py` modules

Two separate Polars adapter modules exist with overlapping but different APIs:

| File | Style | Consumers | Key exports |
|:-----|:------|:----------|:------------|
| `modules/polars_adapter.py` | Class-based (`PolarsAdapter`) + standalone helpers | `tests/test_metabolic_and_misc.py` (3 test classes) | `PolarsAdapter`, `read_csv_fast`, `fast_groupby_agg`, `fast_rolling_mean`, `fast_moving_average`, `benchmark_operation` |
| `modules/calculations/polars_adapter.py` | Functional + domain-specific ops | `services/session_analysis.py`, `modules/calculations/__init__.py` (re-exports) | `to_polars`, `to_pandas`, `fast_rolling_mean`, `fast_groupby_agg`, `fast_normalized_power`, `fast_power_duration_curve` |

**Risk**: Medium — both are actively imported.

**Analysis**:
- The top-level `modules/polars_adapter.py` is **only used in tests**. The production code exclusively uses `modules/calculations/polars_adapter.py`.
- The top-level module has a `benchmark_operation` utility not present in the calculations version.
- Function signatures differ (e.g., `fast_groupby_agg` takes different args: `agg_dict` vs `(agg_col, agg_func)`).

**Next step**: Consider migrating the 3 test classes in `test_metabolic_and_misc.py` to use `modules/calculations/polars_adapter.py` or inline the `PolarsAdapter` class into the test file. Defer deletion until tests are migrated.

### 3. `modules/ui/header.py` — partially deprecated

| Function | Status | Consumers |
|:---------|:-------|:----------|
| `render_sticky_header()` | Active but **not called from `app.py`** (app uses `modules.frontend.components.UIComponents.render_sticky_header`) | Unknown (possibly dead) |
| `render_metric_cards()` | Active but **not called from `app.py`** | Unknown (possibly dead) |
| `show_breadcrumb()` | Active but **not called from `app.py`** (app uses `UIComponents.show_breadcrumb`) | Unknown (possibly dead) |
| `extract_header_data()` | **DEPRECATED** (labelled in source, line 120-121) | `tests/test_ui_components.py` |

**Risk**: Medium — `app.py` uses `modules.frontend.components.UIComponents` for header rendering, not `modules/ui/header.py`. However, the non-deprecated functions in `modules/ui/header.py` may be used by other UI files not yet audited.

**Analysis**:
- `extract_header_data` is already marked deprecated and only used in one test.
- The other 3 functions appear to be superseded by `modules.frontend.components.UIComponents`.
- `modules/ui/sidebar.py` was already deleted from this same directory, suggesting the `modules/ui/` directory is being phased out in favor of `modules/frontend/`.

**Next step**: Audit all files in `modules/ui/` for remaining imports of `modules.ui.header` functions. If zero production imports found, the entire file can be deprecated with a single comment and test updated.

### 4. Backward-compatibility wrappers in `modules/calculations/`

Several files contain legacy compatibility shims:

| File | Pattern | Description |
|:-----|:--------|:-------------|
| `modules/calculations/threshold_types.py` | Legacy point fields | `vt1_watts`, `vt2_watts` kept for backward compatibility |
| `modules/calculations/interpretation.py` | `generate_training_advice_legacy()` | Legacy function wrapper |
| `modules/calculations/vt_cpet.py` | `detect_vt_vslope_savgol()` | Deprecated wrapper for backward compatibility |
| `modules/calculations/smo2/__init__.py` | Re-export shims | Backwards compatibility imports |
| `modules/calculations/executive_summary.py` | Legacy compatibility section | Line 739 |
| `modules/calculations/smo2_thresholds.py` | Legacy compatibility section | Line 131 |
| `modules/calculations/canonical_physio.py` | Alias `calculate_vo2max_acsm` | Backward-compat alias for `calculate_vo2max_jurov` |

**Risk**: Low-Medium — these are deliberately kept for API stability.

**Analysis**: These wrappers are intentional backward-compatibility shims. Removing them would break any external consumers or saved notebooks that use the old API names. No action recommended unless a major version bump is planned.

**Next step**: If a breaking-change release is planned, add deprecation warnings (e.g., `warnings.warn`) to these wrappers rather than removing them.

---

## Summary

| Phase | Action Taken | Status |
|:------|:-------------|:-------|
| Phase 1 | Labelled `train_history.py` and `init_db.py` as legacy/manual scripts | Done |
| Phase 2 | Created this document (`docs/cleanup-candidates.md`) | Done |
| Phase 3 | Investigated duplicate `polars_adapter.py` — documented findings, no deletion (both actively imported) | Documented |
| Phase 4 | Investigated `modules/ui/header.py` and compatibility wrappers — documented findings, no deletion (insufficient evidence for safe removal) | Documented |
