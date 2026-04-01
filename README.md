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
| **Physiology** | SmO2 kinetics, HRV (DFA α1), termoregulacja, biomechanika |
| **Thresholds** | 4-point CPET: VT1_onset, VT1_steady, RCP_onset, RCP_steady |
| **AI Coach** | Multi-sensor fusion, limiter diagnosis, rekomendacje |
| **Reports** | PDF ~36-stronicowy, DOCX, PNG export, SQLite baza danych, CLI generator |

### 🧹 Codebase Cleanup (2026-04-01)

- `modules/ui/sidebar.py` removed (zero imports, dead code)
- `train_history.py`, `init_db.py` labelled as legacy/manual scripts (not runtime app modules)
- Duplicate `polars_adapter.py` modules and `modules/ui/header.py` investigated — findings documented in [`docs/cleanup-candidates.md`](docs/cleanup-candidates.md)

### 🆕 PDF Report Fixes & Data Analysis v3.3 (2026-03-29)

Poprawki jakości raportu PDF oraz dokładności analizy danych.

| Kategoria | Zmiana | Pliki |
|:----------|:-------|:------|
| **FIX** | Tabela progów (str. 6): pierwsza kolumna poszerzona z 42→52mm — brak overflow tekstu | `pdf/layout.py` |
| **FIX** | Tabela wysokościowa (str. 14): VO₂max "---" zastąpione poprawną wartością (naprawa zagnieżdżonej ścieżki danych) | `pdf/layout.py` |
| **FIX** | Torque Nm (str. 22): filtr kadencji ≥60 rpm wyklucza artefakty post-exhaustion (65→149 Nm → realistyczne 12→35 Nm) | `pdf/layout.py`, `report_generator.py`, `figures/biomech.py` |
| **FIX** | `PageBreak` przed nagłówkiem "KLUCZOWE LICZBY" — koniec osierocenia nagłówka na dole strony 21 | `pdf/layout.py` |
| **FIX** | `PageBreak` przed "POŁĄCZENIE Z DRYFEM HR I EF" — koniec osierocenia nagłówka na dole strony 27 | `pdf/layout.py` |
| **FIX** | Stopka copyright © widoczna na każdej stronie + w sekcji końcowej raportu | `pdf/builder.py`, `pdf/layout.py` |
| **FIX** | KeyError w zakładce Manual Thresholds: dynamiczne mapowanie kolumn VE/slope dla V-Slope wykresu | `ui/manual_thresholds.py` |
| **FIX** | CP model fittowany z krzywej MMP gdy brak danych CPET (fallback 3-point power-duration fit) | `calculations/pipeline.py` |
| **FIX** | VT spike skip: ograniczony do pierwszych 2 etapów zamiast wszystkich — lepsza detekcja VT2 | `calculations/vt_step.py` |
| **FIX** | Scroll zoom na wykresach Plotly wyłączony globalnie (`scrollZoom: false` w `CHART_CONFIG`) | `plots.py` |

### 🆕 Chart & Visualization Overhaul v1.0 (2026-03-28)

Complete audit and improvement of all Plotly charts in the Streamlit dashboard.

| Category | Change | Files |
|:---------|:-------|:------|
| **BREAKING** | Main timeline split into 2 stacked subplots: Power+HR (top) / SmO2+VE (bottom) — eliminates 4 overlapping right-hand axes | `summary_charts.py` |
| **NEW** | Training zone bands Z1–Z6 rendered as `add_hrect()` background on power panel | `summary_charts.py` |
| **NEW** | CP / VT1 / VT2 threshold `hline` annotations on power panel | `summary_charts.py` |
| **NEW** | Range slider (`rangeslider`) on x-axis for session navigation | `summary_charts.py` |
| **NEW** | Radar chart: elite reference ring + `tickvals`/`ticksuffix` on radial axis | `limiters.py` |
| **FIX** | SmO2 color unified — `Config.COLOR_SMO2` used everywhere (was hardcoded `#2ca02c` in 3 places) | `summary_charts.py` |
| **FIX** | FRI gauge: axis range extended from `[0.6, 1.0]` to `[0, 1.0]` + critical zone `<0.60` added | `power.py` |
| **FIX** | W' Balance hovertemplate now shows `%{y:.0f} J (%{customdata:.0f}% zasobu)` | `power.py` |
| **FIX** | Legend position changed to `y=1.03, yanchor="bottom"` — no more clipping by Streamlit container | `summary_charts.py` |
| **FIX** | Eliminated double rolling-mean via `_get_smooth()` helper (was applying `.rolling(5)` on pre-smoothed data) | `summary_charts.py` |
| **GLOBAL** | `CHART_CONFIG` (no Plotly logo, hover modebar, scroll zoom, retina PNG export) applied to all `st.plotly_chart()` calls | `plots.py` + 7 UI files |
| **GLOBAL** | `CHART_HEIGHT_MAIN/SUB/SMALL/HEATMAP/RADAR` constants replace scattered hardcoded px values | `plots.py` |

### 🆕 PDF Report v3.2 — New Sections & Microcycle Periodization (2026-03-26)

4 new PDF report pages, expanded training recommendations, CLI PDF generator.

| Category | Change | Reference |
|:---------|:-------|:----------|
| **NEW** | HRV/DFA Alpha-1 page: zone classification, RMSSD/SDNN, autonomic fitness | Mateo-March 2024, Iannetta 2024 |
| **NEW** | Skin Temperature Gradient page: core-skin ΔT for thermoregulatory efficiency | Périard 2021, Racinais 2019 |
| **NEW** | Altitude Adjustment page: VO₂max sea-level equivalent table | Wehrlin & Hallen 2006 |
| **NEW** | Microcycle Periodization page: weekly schedule from training block with TSS | Auto-generated |
| **NEW** | CLI PDF generator: `scripts/generate_pdf_from_csv.py` for headless PDF generation | — |
| **IMPROVED** | 5 diverse training recommendations per classification in all 6 calc modules | biomech, cardiac_drift, cardio, smo2, thermo, vent |

### 🆕 PDF Report Quality Audit v3.1 (2026-03-25)

Comprehensive PDF report audit from exercise physiology professor & triathlon coach perspective.
**8 critical fixes, 6 new analysis sections, 16+ cited publications.**

| Category | Change | Reference |
|:---------|:-------|:----------|
| **FIX** | Deduplicated cardiac drift blocks (verdict vs simulation) | — |
| **FIX** | Reconciled contradictory limiter diagnoses across pages | Multi-method hierarchy |
| **FIX** | Fixed SmO₂ drift -26% mislabeled as "STABILNY" | Threshold recalibration |
| **FIX** | Fixed Cardiac Drift value not propagating to verdict page | Key mapping fix |
| **FIX** | Added confidence score visual flagging (<50% = warning) | — |
| **FIX** | Unified hydration recommendations (500-750ml/h) | — |
| **FIX** | Unified version strings (RAMP_METHOD_VERSION) | version.py SSoT |
| **FIX** | Clarified Upper Aerobic zone as separate model from Z3/Z4 | — |
| **NEW** | Tidal Volume decomposition (VE = TV × RR) with breathing strategy | — |
| **NEW** | Ventilatory Reserve (Breathing Reserve %) | MVV estimation |
| **NEW** | W' Reconstitution table (30s→20min recovery) | Caen et al. 2021 |
| **NEW** | HR Recovery Kinetics classification | Buchheit 2014 |
| **NEW** | Cadence context in occlusion analysis | Hammer et al. 2021 |
| **NEW** | 5 diverse training sessions per metabolic block (was 3) | — |

### 🆕 Evidence-Based Physiology Overhaul v3.0 (2026-03-21)

Complete audit based on **42 peer-reviewed publications (2021-2026)**.
All formulas, models, and algorithms updated to latest available evidence.

**CRITICAL — New Models (5):**
| Fix | Reference | Opis |
|:----|:----------|:-----|
| VO2max: Jurov 2023 sex-specific | Life (MDPI) 13(1):160 | Replaces Sitko 2021. M: `0.10×PO - 0.60×BW + 64.21`, F: `0.13×PO - 0.83×BW + 64.02`. Bias 0.19% vs ACSM 12% |
| VLaMax: Wackerhage 2025 disclaimer | Sports Med 55:1853-1866 | No validated glycolytic equivalent to VO2max. PCr correction by phenotype |
| W' Balance: Caen 2021 bi-exponential | EJAP (Welburn 2025 validation) | Two-phase tau (fast PCr + slow metabolic). Sport-specific: cycling/running/swimming |
| VT confidence intervals | Gronwald 2024 meta-analysis | VT1/VT2 displayed as ranges (±5-20W) based on detection confidence |
| VT vs LT terminology | Cerezuela-Espejo 2023 | VT1 ≠ LT1 — different mechanisms, clear disclaimers added |

**HIGH — Updated Models (12):**
| Fix | Reference | Opis |
|:----|:----------|:-----|
| DFA Alpha-1 window 600s | Iannetta 2024, ICC 0.76-0.86 | Was 300s. Extreme values logged instead of clipped |
| PSI → aPSI adaptive | Buller 2023, Physiol Meas | Acclimatization correction for 10+ days heat exposure |
| RER validation for VT1 | Cerezuela-Espejo 2023 | RER > 1.0 at VT1 = hyperventilation artifact flag |
| VO2max zone floor 110% CP | Garcia-Tabar 2024 | Was 105%. True VO2max work = 110-120% CP |
| Cardiac drift analysis | Sperlich 2025, Papini 2024 | HR:Power decoupling (EF 1st vs 2nd half) |
| SmO2 T1 cross-validation | Yogev 2023, Sendra-Perez 2023 | Flag if SmO2 T1 diverges from VT1 by >15% |
| Altitude VO2max correction | Wehrlin & Hallen 2006, Pühringer 2022 | 6.3%/1000m linear reduction above 500m |
| Strict data validation | — | No silent type coercion, EU locale (comma decimals), 10% NaN rejection |
| Cross-signal validation | — | Power/HR envelope checks (450W @ 90bpm = sensor fault) |
| Temporal bounds validation | — | Sample rate, gap detection (>60s), merged file detection |
| DB schema: athlete_id | — | Multi-athlete support, session_type, test_validity fields |
| Composite recovery score | Guimaraes Couto 2025 | 40% W' + 30% SmO2 + 30% cardiac drift (was W'-only) |

**MEDIUM — New Features (4):**
| Fix | Reference | Opis |
|:----|:----------|:-----|
| Pacing analysis | Guimaraes Couto 2025/2026, Konings 2025 | Positive/negative/even split detection with coaching causes |
| SmO2 context interpretation | Perrey 2024, 191 studies SR | Different SmO2 meaning for sprint vs threshold vs Z2 |
| Contextual limiter recs | Garcia-Tabar 2024 | Z2 volume-aware, EIB screening, mechanical occlusion check |
| Phenotype disclaimer | INSCYD n~2000, Garcia-Tabar 2024 | Marked as provisional, 40× metabolic response heterogeneity noted |

### 🆕 Exercise Physiology & Code Quality Review (2026-03-17)

Previous audit with 9 physiological + 4 code quality fixes.
See commit `6e1f4d8` for details.

### 🆕 Security & Code Quality Audit (2026-03-16)

**CRITICAL Fixes (5):**
| Fix | Opis |
|:----|:-----|
| Missing `logger` in session_orchestrator | `NameError` crash on any parallel calculation failure |
| Dead duplicate code in `_calculate_parallel` | Unreachable `try/except` blocks from bad merge |
| Wrong `np.interp` argument order in pipeline | Produced garbage LT1/LT2 HR values |
| Orphaned code in `cache_utils.py` | Triple-duplicated imports + dead `_hash_arg` body |
| Duplicate `subprocess.run` in `report_io` | Git tracking check ran twice, exceptions silenced |

**HIGH Fixes (10):**
| Fix | Opis |
|:----|:-----|
| XSS: duplicate `show_breadcrumb` | Unsafe version shadowed safe `html.escape()` version |
| Information disclosure via `st.error()` | Raw exceptions exposed filesystem paths in UI |
| DataFrame mutation in `vent.py` | Missing `.copy()` caused cross-tab side effects |
| `vt_cpet_ve_only` immutability | Now returns new dict/DataFrame instead of mutating |
| `session_analysis` immutability | Removed `inplace` param, `calculate_extended_metrics` returns new dict |
| `ramp_archive.py` mutation | Added `.copy()` before DataFrame modifications |
| `training_load.py` mutation | Added `.copy()` to prevent caller side effects |
| Dead code in `app.py` | Removed unreachable hash fallback after `return` |
| Redundant imports in `pdf/layout.py` | Removed 11 function-level `HexColor` re-imports |
| Silent exception swallowing | Added logging to `report_io`, `vent_thresholds_display` |

**MEDIUM Fixes (7):**
| Fix | Opis |
|:----|:-----|
| User note XSS | `html.escape()` for note text in `smo2.py`, `vent.py` |
| Plotly hover injection | `html.escape()` for filenames in `ai_coach.py` |
| Filename sanitization | `train_history.py` now sanitizes batch filenames |
| CSS path traversal | `config.py` validates CSS path stays in project dir |
| Relative `.git` path | `report_io.py` uses absolute path via `Path(__file__)` |
| Magic number extraction | `OCCLUSION_TORQUE_CRITICAL_NM = 70` constant |
| Requirements lock warning | Added comment about missing pinned versions |

### 🆕 Ramp Report Review (2026-02-28)

**CRITICAL Fixes:**
| Fix | Opis |
|:----|:-----|
| SmO2 Drift `abs()` | 3-level thresholds (STABILNY/OSTRZEŻENIE/RYZYKO) |
| Exp-Dmax for T2 | ICC=0.79-0.91 reliability |
| CP vs VT2 validation | Warning when CP > VT2 by >2% |
| HR consistency | Linear interpolation for VT/SmO2 alignment |

**HIGH Fixes:**
| Fix | Opis |
|:----|:-----|
| VO2max CI | ±ml/kg/min + absolute value (L/min) |
| VLaMax disclaimer | ±25-30% error range (estimated, updated 2026-03-17) |
| Thermal integration | Combined cardiac drift interpretation |

### 🆕 SmO2 Threshold Analysis v2.1 (2026-02-28)

| Kategoria | Ulepszenie | Opis |
|:----------|:-----------|:-----|
| **Algorytm** | Double-linear regression | 2-segment piecewise regression |
| **Algorytm** | Exp-Dmax for T2 | ICC=0.79-0.91 dla T2 detection |
| **Walidacja** | MOT1 reliability | 3 consecutive steps, ICC=0.53 warning |
| **Preprocessing** | Savitzky-Golay filter | Opcjonalny S-G filter |
| **Preprocessing** | Adaptive window | Smoothing oparte na sampling rate |
| **Preprocessing** | Adaptive curvature | Percentyle zamiast hardcoded |
| **Walidacja** | ATT validation | >10mm warn, >15mm UNRELIABLE |
| **Model** | Feldmann 4-phase | Phase 1→2 transition detection |
| **Scoring** | Confidence rescaled | Cap 0.6→0.8 z bonusami |

### 🆕 VT Detection Protocol v2.0 (2026-02-27)

| Kategoria | Ulepszenie | Opis |
|:----------|:-----------|:-----|
| **Walidacja** | RER validation | Kary confidence, odrzucenie gdy >1.25 |
| **Walidacja** | VT2 vs Pmax check | UNRELIABLE gdy VT2 > 95% Pmax |
| **Walidacja** | VE bounds check | Fizjologiczne limity 5-250 L/min |
| **Algorytm** | Pmax-relative fallback | VT1 ≈ 60% Pmax, VT2 ≈ 80% Pmax |
| **Algorytm** | Adaptive slope ratio | Próg dla sportowców z płynnym przejściem |
| **Preprocessing** | IQR artifact filtering | Detekcja artefaktów w VE-only mode |
| **Statystyka** | Unbiased variance | `ddof=1` + minimum n≥4 |

### Supported Data Formats
- **Import**: FIT, TCX, CSV (Garmin, TrainingPeaks, Intervals.icu)
- **NIRS**: TrainRed, Moxy (auto-detekcja)

---

## 🛠️ Tech Stack

| Warstwa | Technologia |
|:--------|:------------|
| UI | Streamlit |
| Data | Polars + Pandas |
| Analysis | SciPy, NumPy, Statsmodels |
| HRV | NeuroKit2 |
| Acceleration | Numba JIT, MLX (Apple Silicon) |
| Reports | ReportLab, python-docx |
| Viz | Matplotlib, Plotly |
| Storage | SQLite |

---

## 📥 Installation

```bash
# Clone
git clone https://github.com/WielkiKrzych/Tri_Dashboard.git
cd Tri_Dashboard

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

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
