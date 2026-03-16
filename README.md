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
| **Reports** | PDF 7-stronicowy, DOCX, PNG export, SQLite baza danych |

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
| VLaMax disclaimer | ±15-20% error range (estimated) |
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
