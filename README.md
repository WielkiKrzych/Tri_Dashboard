# 🚴 Tri_Dashboard

<div align="center">

![Tri_Dashboard Logo](./assets/Logo.jpg)

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-21A421?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-40+-18A808?style=for-the-badge&label=tests)](tests/)

**Advanced Physiological Analysis Platform for Triathletes**

[Features](#-features) • [Install](#-installation) • [Stack](#-tech-stack) • [Changelog](#-changelog) • [License](#-license)

</div>

---

## 🔬 Overview

Tri_Dashboard to platforma analityczna dla **trenerów**, **naukowców sportu** i **zaawansowanych atletów**. Oferuje:

- 🔬 Wielosensorowa fuzja danych (SmO2, HRV, wentylacja)
- 🤖 AI Coach z diagnozą limitów wydolnościowych
- 📊 Automatyczna detekcja progów (VT1/VT2, LT1/LT2)
- 📈 Generowanie raportów PDF/DOCX z confidence scoring
- ⚡ Wysoka wydajność (Polars, Numba, caching)

---

## ⭐ Features

| Moduł | Funkcjonalność |
|:-------|:---------------|
| **Power** | PDC, CP/W', NP, IF, TSS, phenotype klasyfikacja |
| **Physiology** | SmO2 kinetics (ΔSmO2, Exp-Dmax, 4-knot regression), HRV (DFA α1), termoregulacja, biomechanika |
| **Thresholds** | 4-point CPET: VT1_onset, VT1_steady, RCP_onset, RCP_steady |
| **AI Coach** | Multi-sensor fusion, limiter diagnosis, rekomendacje |
| **Reports** | PDF ~36-stronicowy, DOCX, PNG export, SQLite baza danych, CLI generator |

---

## 📂 Project Structure

```
Tri_Dashboard/
├── app.py                  # Main Streamlit app
├── modules/
│   ├── calculations/       # Core algorithms (VT, SmO2, power, HRV...)
│   │   ├── vt_cpet.py              # CPET orchestration + cross-validation
│   │   ├── vt_cpet_preprocessing.py # VE validation, smoothing, artifacts
│   │   ├── vt_cpet_gas_exchange.py  # RER validation + confidence penalty
│   │   ├── vt_cpet_ve_only.py       # Adaptive slope ratio detection
│   │   ├── vt_step.py               # Unbiased variance penalty
│   │   ├── smo2_thresholds.py       # Ramp test T1/T2 + ATT validation
│   │   ├── smo2_breakpoints.py      # 2-segment + Exp-Dmax methods
│   │   ├── smo2_analysis.py         # Feldmann 4-phase + advanced metrics
│   │   ├── metabolic_engine.py      # VO2max CI, VLaMax estimation
│   │   ├── pipeline.py              # Relative SmO2 thresholds + HR consistency
│   │   └── common.py                # VT2 vs Pmax validation
│   ├── ui/                # Streamlit tabs & components
│   ├── reporting/         # PDF/DOCX generators
│   │   ├── pdf/layout.py           # KPI dashboard, limiter classification
│   │   ├── pdf/builder.py          # CP vs VT2 validation
│   │   └── figures/smo2_vs_power.py # Artifact filtering
│   ├── frontend/          # Theme, state, layout
│   ├── db/                # SQLite session store
│   ├── ai/                # AI Coach & interval detection
│   └── cache_utils.py     # Caching layer (40+ cached functions)
├── services/              # Data pipeline orchestrator
├── tests/                 # 40+ test files
└── assets/                # Logo, backgrounds
```

---

## ⚡ Performance

| Operation | Speedup |
|:----------|:--------|
| Rolling Mean | 10-50x (Numba) |
| GroupBy | 10-100x (Polars) |
| DataFrame Ops | 10x (vectorized) |
| Column Mapping | 5-10x (O(1) lookup) |
| Caching | TTL-based memoization |

---

## 📋 Changelog


### 2026-04-02 — Evidence-Based SmO2 Threshold Detection Overhaul

**Scientific basis:** Perrey, Ferrari et al. (2024) Sports Medicine systematic review (191 studies), Sendra-Pérez et al. (2023, 2024), Feldmann et al. (2022), Racinais et al. (2014).

**CRITICAL — Threshold Detection Accuracy (3):**
| Change | Reference | Description |
|:-------|:----------|:------------|
| ΔSmO2 baseline correction | Sendra-Pérez 2024 | Normalizes SmO2 relative to warm-up baseline, eliminates inter-individual differences |
| Signal inversion for Exp-Dmax | Sendra-Pérez 2024 | Inverted ΔSmO2 behaves like lactate (increasing) — correct Exp-Dmax application |
| First 60s exclusion | Feldmann 2022 | Abrupt test starts cause NIRS artifacts; excluded from breakpoint analysis |

**HIGH — Analysis Quality (3):**
| Change | Reference | Description |
|:-------|:----------|:------------|
| SmO2min → VO2peak estimation | Feldmann 2022 (R²=0.85) | Non-invasive fitness proxy: VO2max ≈ 88.2 − 0.62 × SmO2min |
| BP2 inflection type detection | Feldmann 2022 | Classifies T2 as positive (plateau) or negative (further desaturation) |
| 4-knot segmented regression | Feldmann 2022 | Cross-validation of T1/T2 via 3-segment piecewise regression |

**MEDIUM — New Capabilities (2):**
| Change | Reference | Description |
|:-------|:----------|:------------|
| Butterworth low-pass filter | Sendra-Pérez 2024 | 3rd smoothing option (median / Savgol / Butterworth 0.2 Hz) |
| Multi-muscle MOT2 consistency | Sendra-Pérez 2024 | Utility for comparing MOT2 across multiple muscle sensors (ICC≈0.64) |

### 2026-04-02 — Architecture Improvements & Codebase Cleanup

**Architecture (5 improvements):**
| Change | Files | Description |
|:-------|:------|:------------|
| Config grouped dataclasses | `config.py` | 7 frozen dataclasses replace flat namespace, 47 passthrough attributes preserved |
| BaseStore unified DB | `db/base.py`, 3 stores | New BaseStore ABC, ~100 lines boilerplate removed |
| State key registry | `frontend/state.py` | 34 keys in `_defaults` dict, 11 guard clauses removed |
| UI shared components | `ui/shared.py` | chart(), metric(), require_data(), dataframe(), alert() — 8+2 replacements |
| Typed SessionPipeline | `services/session_pipeline.py` | Typed dataclasses for analysis pipeline (purely additive) |

**Codebase Cleanup:**
- 14 orphaned files deleted (removed tabs, dead plugins, legacy scripts)
- 11 unused imports removed from 10 files
- 19 dead functions removed from 9 files
- `style-light.css` deleted (dark-mode only)
- Tests updated for removed references

**UI Changes:**
- Dark mode forced (theme toggle removed)
- 6 tabs removed: Longitudinal Trends, Niezawodność, Benchmarking, Porównanie Sesji, Multi-Sport
- Athlete profile selector removed from sidebar
- App title changed to "Tri Dashboard"

### 2026-04-01 — UX Fixes, Tech Debt Cleanup & Dead Code Removal

**UX/Runtime (2 fixes):**
- SQLite schema migration — legacy DBs auto-migrated on startup (idempotent, no data loss)
- Streamlit deprecation warnings removed — 84 occurrences of `use_container_width=True` replaced with `width="stretch"`

**Dead Code Removal:**
- `modules/ui/sidebar.py` removed (zero imports)
- ~880 lines of dead code removed across 21 files (unused functions, orphaned imports, dead plugins)
- `modules/ui/registry.py` and `modules/ui/vent_thresholds_report.py` deleted

### 2026-03-29 — PDF Report Fixes & Data Analysis v3.3

**PDF Layout (6 fixes):**
- Thresholds table column width 42→52mm (text overflow fix)
- VO2max altitude table: fixed nested data path (`canonical_physiology.vo2max`) — was showing "---"
- Torque outlier filtering: cadence ≥60rpm guard removes post-exhaustion artifacts (65→149Nm → 12→35Nm)
- `PageBreak` before "KLUCZOWE LICZBY" heading (page 21 orphan fix)
- `PageBreak` before "POŁĄCZENIE Z DRYFEM HR I EF" heading (page 27 orphan fix)
- Copyright © Krzysztof Kubicz footer on every page + end section

**App Fixes (4 fixes):**
- Manual Thresholds tab: KeyError on V-Slope plot resolved (dynamic column mapping for VE/slope)
- CP model now fitted from MMP curve when CPET data unavailable (3-point power-duration fallback)
- VT spike skip logic: limited to first 2 stages only — improves VT2 detection accuracy
- Scroll zoom on all Plotly charts disabled globally (`scrollZoom: false`)

### 2026-03-28 — Chart & Visualization Overhaul v1.0

- Main timeline chart replaced with 2-row subplots (Power+HR / SmO2+VE) — 4 overlapping axes eliminated
- Training zone bands Z1–Z6 as background `hrect` on power panel
- CP/VT1/VT2 `hline` annotations on power panel
- Range slider on x-axis for session navigation
- FRI gauge axis `[0.6, 1.0]` → `[0, 1.0]` + critical zone `<0.60`
- SmO2 color unified to `Config.COLOR_SMO2` across all charts
- W' Balance hover: value in J + % of W' reserve
- Radar: elite reference ring + % tick labels on radial axis
- `CHART_CONFIG` (logo off, hover modebar, scroll zoom, retina export) applied to all 20+ chart calls
- `CHART_HEIGHT_*` constants replace hardcoded px values in 7 files
- `_get_smooth()` helper eliminates double rolling-mean on pre-smoothed columns

### 2026-03-26 — PDF Report v3.2: New Sections & Microcycle Periodization

4 new PDF report pages, expanded training recommendations, CLI PDF generator.

| Category | Change | Reference |
|:---------|:-------|:----------|
| **NEW** | HRV/DFA Alpha-1 page: zone classification, RMSSD/SDNN | Mateo-March 2024, Iannetta 2024 |
| **NEW** | Skin Temperature Gradient page: core-skin ΔT | Périard 2021, Racinais 2019 |
| **NEW** | Altitude Adjustment page: VO₂max sea-level equivalent | Wehrlin & Hallen 2006 |
| **NEW** | Microcycle Periodization page: weekly schedule with TSS | Auto-generated |
| **NEW** | CLI PDF generator: headless PDF generation | — |
| **IMPROVED** | 5 diverse training recommendations per classification | 6 calc modules |

### 2026-03-25 — PDF Report Quality Audit v3.1

Comprehensive audit from exercise physiology professor perspective.
**8 critical fixes, 6 new analysis sections, 16+ cited publications.**

| Category | Change | Reference |
|:---------|:-------|:----------|
| **FIX** | Deduplicated cardiac drift blocks | — |
| **FIX** | Reconciled contradictory limiter diagnoses | Multi-method hierarchy |
| **FIX** | SmO₂ drift -26% mislabeled as "STABILNY" | Threshold recalibration |
| **FIX** | Cardiac Drift value not propagating to verdict page | Key mapping fix |
| **FIX** | Confidence score visual flagging (<50% = warning) | — |
| **FIX** | Unified hydration recommendations (500-750ml/h) | — |
| **FIX** | Unified version strings (RAMP_METHOD_VERSION) | version.py SSoT |
| **FIX** | Clarified Upper Aerobic zone as separate model | — |
| **NEW** | Tidal Volume decomposition (VE = TV × RR) | — |
| **NEW** | Ventilatory Reserve (Breathing Reserve %) | MVV estimation |
| **NEW** | W' Reconstitution table (30s→20min recovery) | Caen et al. 2021 |
| **NEW** | HR Recovery Kinetics classification | Buchheit 2014 |
| **NEW** | Cadence context in occlusion analysis | Hammer et al. 2021 |
| **NEW** | 5 diverse training sessions per metabolic block | — |

### 2026-03-21 — Evidence-Based Physiology Overhaul v3.0

Complete audit based on **42 peer-reviewed publications (2021-2026)**.

**CRITICAL — New Models (5):**
| Change | Reference | Description |
|:-------|:----------|:------------|
| VO2max: Jurov 2023 sex-specific | Life (MDPI) 13(1):160 | M: `0.10×PO - 0.60×BW + 64.21`, F: `0.13×PO - 0.83×BW + 64.02` |
| VLaMax: Wackerhage 2025 disclaimer | Sports Med 55:1853-1866 | No validated glycolytic equivalent to VO2max |
| W' Balance: Caen 2021 bi-exponential | EJAP | Two-phase tau (fast PCr + slow metabolic) |
| VT confidence intervals | Gronwald 2024 meta-analysis | VT1/VT2 as ranges (±5-20W) |
| VT vs LT terminology | Cerezuela-Espejo 2023 | VT1 ≠ LT1 — different mechanisms |

**HIGH — Updated Models (12):** DFA α1 window 600s, PSI → aPSI adaptive, RER validation, VO2max zone floor 110% CP, cardiac drift analysis, SmO2 T1 cross-validation, altitude VO2max correction, strict data validation, cross-signal validation, temporal bounds, DB athlete_id, composite recovery score

**MEDIUM — New Features (4):** Pacing analysis, SmO2 context interpretation, contextual limiter recommendations, phenotype disclaimer

### 2026-03-17 — Exercise Physiology Audit

9 physiological fixes + 4 code quality fixes. See commit `6e1f4d8` for details.

### 2026-03-16 — Security & Code Quality Audit

**5 critical, 10 high, 7 medium issues.**

**CRITICAL (5):** Missing logger crash fix, dead duplicate code removal, wrong `np.interp` argument order, orphaned code in cache_utils, duplicate subprocess.run

**HIGH (10):** XSS fixes, information disclosure prevention, DataFrame mutation fixes (6 files), dead code removal, redundant imports, silent exception logging

**MEDIUM (7):** User note XSS, Plotly hover injection, filename sanitization, CSS path traversal, relative .git path, magic number extraction, requirements lock warning

### 2026-03-01 — PDF Polish & VO2max/VLaMax Fixes

- PNG export available immediately after CSV upload
- Professional PDF report polish for commercial quality
- VO2max and VLaMax calculations corrected per exercise physiology

### 2026-02-28 - Ramp Report Review Fixes

**CRITICAL (4):**
- [C-01] SmO2 Drift status - `abs()` for negative values + 3-level thresholds
- [C-02] Exp-Dmax method for T2 detection - ICC=0.79-0.91
- [C-03] CP vs VT2 validation - warning when CP > VT2 by >2%
- [C-04] HR consistency fix - linear interpolation

**HIGH (3):**
- [H-01] VO2max confidence interval - ±ml/kg/min + absolute value (L/min)
- [H-02] VLaMax marked as estimated - ±15-20% error range
- [H-03] Thermal + cardiac drift integration

**MEDIUM (4):**
- [M-01] SmO2 chart artifact filtering - >10% deviation removal
- [M-02] Z4 Threshold range widened - 106-120% of VT1
- [M-03] Reoxy half-time reference ranges - <15s/15-30s/>30s
- [M-04] BR column in VT thresholds table

**NARRATIVE (2):**
- [N-02] Tempo zone description fix
- [N-03] Milder language for low confidence

### 2026-02-28 - SmO2 Threshold Analysis v2.1

**HIGH (3):**
- Double-linear (2-segment) regression
- Exp-Dmax method for T2
- MOT1 reliability warning - 3 consecutive steps

**MEDIUM (4):**
- Adaptive curvature thresholds (percentile-based)
- ATT (adipose tissue) validation
- Feldmann 4-phase model
- Savitzky-Golay filter option

**LOW (2):**
- Confidence rescaled 0.6→0.8
- Adaptive filter window

### 2026-02-27 - VT Detection Protocol v2.0

**CRITICAL (3):**
- RER validation with confidence penalty
- VT2 vs Pmax sanity check
- Pmax-relative fallback formula

**WARNING (5):**
- VE unit validation (5-250 L/min)
- IQR-based artifact filtering
- Adaptive slope ratio threshold
- Cross-validation between methods
- Relative SmO2 modulation thresholds

**MINOR (2):**
- Unbiased variance penalty
- Edge effects fix in smoothing

---

## 📄 License

MIT License - zobacz [LICENSE](LICENSE) file.

---

## ⚠️ Disclaimer

> **To oprogramowanie służy wyłącznie celom edukacyjnym i treningowym.** Nie jest urządzeniem medycznym i nie może być używane do diagnozowania lub leczenia schorzeń. Zawsze konsultuj się z wykwalifikowanymi specjalistami.

---

<div align="center">

⭐ Star na GitHubie | 🍴 Fork | 👁️ Watch

</div>
