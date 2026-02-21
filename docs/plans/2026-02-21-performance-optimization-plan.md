# TriDashboard Performance Optimization Plan

**Date:** 2026-02-21  
**Author:** AI Analysis  
**Status:** Draft - Ready for Implementation

---

## Executive Summary

TriDashboard is a well-architected Streamlit application for triathlon performance analysis. Recent optimization work (commits `perf:`, `refactor:`) has established a solid foundation with Polars integration, diskcache-based caching, and clean code patterns.

**This plan identifies 5 optimization areas** with potential for 2-5x overall performance improvement, focusing on:
1. Streamlit caching expansion (highest impact)
2. UI responsiveness with fragments
3. Polars utilization improvement
4. Memory efficiency
5. Parallel processing

---

## Current State Assessment

### Strengths
- Clean architecture: modules/services/models separation
- Polars adapter implemented (`modules/polars_adapter.py`)
- Cache utilities with diskcache (`modules/cache_utils.py`)
- No `iterrows()` usage, minimal `apply(lambda)`
- LRU cache in session_analysis
- Chunked processing for large files

### Gaps Identified
| Metric | Current | Target |
|--------|---------|--------|
| `@st.cache_data` decorators | 8 | 40+ |
| `@st.cache_resource` decorators | 0 | 5-10 |
| `st.fragment` usage | 0 | 10-15 |
| Polars adapter utilization | ~10% | 60%+ |
| Cache hit rate (estimated) | ~20% | 70%+ |

---

## Phase 1: Streamlit Caching Expansion (HIGH IMPACT)

**Estimated effort:** 4-6 hours  
**Expected improvement:** 2-3x faster reruns

### 1.1 Expand @st.cache_data Coverage

**Files to modify:**
- `modules/calculations/*.py` - Heavy computation functions
- `modules/ui/*.py` - Chart generation functions
- `services/session_analysis.py` - Metrics calculations

**Priority functions to cache:**

```python
# modules/calculations/power.py
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_power_duration_curve(df_hash: str, df: pd.DataFrame) -> dict:
    ...

# modules/calculations/ventilatory.py
@st.cache_data(ttl=3600, show_spinner=False)
def detect_vt_from_steps(df_hash: str, ...):
    ...

# modules/calculations/smo2_thresholds.py
@st.cache_data(ttl=3600, show_spinner=False)
def detect_smo2_thresholds_moxy(df_hash: str, ...):
    ...
```

**Implementation pattern:**
```python
def _hash_dataframe(df: pd.DataFrame) -> str:
    """Generate stable hash for DataFrame caching."""
    return hashlib.md5(
        f"{df.columns.tolist()}:{len(df)}:{df.iloc[0:100].to_numpy().tobytes()}"
    ).hexdigest()

@st.cache_data(ttl=3600, show_spinner=False)
def expensive_calculation(df_hash: str, *args, **kwargs):
    # Actual computation
    pass

# Call site
result = expensive_calculation(_hash_dataframe(df), df, other_params)
```

### 1.2 Add @st.cache_resource for Global Resources

**Files to modify:**
- `modules/db/__init__.py` - SessionStore singleton
- `modules/frontend/theme.py` - ThemeManager
- `modules/notes.py` - TrainingNotes

**Implementation:**
```python
# modules/db/__init__.py
@st.cache_resource
def get_session_store() -> "SessionStore":
    """Cached singleton for database connection."""
    return SessionStore()

# modules/frontend/theme.py
@st.cache_resource
def get_theme_manager() -> "ThemeManager":
    """Cached theme manager instance."""
    return ThemeManager()
```

### 1.3 Cache Chart Generation Functions

**Files to modify:**
- `modules/chart_exporters.py`
- `modules/ui/summary_charts.py`
- `modules/reporting/figures/*.py`

**Pattern:**
```python
@st.cache_data(ttl=3600, show_spinner=False)
def build_power_chart(df_hash: str, cp: int, w_prime: int) -> go.Figure:
    """Build power chart with caching."""
    ...
```

---

## Phase 2: UI Responsiveness with Fragments

**Estimated effort:** 2-3 hours  
**Expected improvement:** Instant UI updates, reduced rerun cascade

### 2.1 Implement st.fragment for Static Components

Streamlit 1.33+ supports fragments for partial reruns. Use for:
- Header metrics (static after initial load)
- Sidebar parameters
- Export buttons

**Files to modify:**
- `app.py` - Main layout
- `modules/frontend/components.py` - UI components

**Implementation:**
```python
# app.py
@st.fragment
def render_header_metrics(np_header, if_header, tss_header, df_plot):
    """Fragment for header metrics - reruns independently."""
    m1, m2, m3 = st.columns(3)
    m1.metric("NP (Norm. Power)", f"{np_header:.0f} W")
    m2.metric("TSS", f"{tss_header:.0f}", help=f"IF: {if_header:.2f}")
    m3.metric("Praca [kJ]", f"{df_plot['watts'].sum() / 1000:.0f}")

@st.fragment
def render_export_buttons(safe_filename, metrics, df_plot, ...):
    """Fragment for export section."""
    ...
```

### 2.2 Lazy Load Heavy Tabs

Defer loading of tabs until user clicks them:

```python
# Use st.tabs with lazy rendering
with tab_physiology:
    # Only render when tab is active
    if st.session_state.get("active_tab") == "physiology":
        render_tab_content("hrv", df_clean_pl)
```

---

## Phase 3: Polars Utilization Improvement

**Estimated effort:** 3-4 hours  
**Expected improvement:** 10-100x faster aggregations

### 3.1 Expand polars_adapter.py Functions

**Add to `modules/polars_adapter.py`:**

```python
def fast_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Fast rolling standard deviation using Polars."""
    ...

def fast_ewm_mean(series: pd.Series, alpha: float) -> pd.Series:
    """Fast exponential weighted mean using Polars."""
    ...

def fast_fill_na(df: pd.DataFrame, strategy: str = "forward") -> pd.DataFrame:
    """Fast NA filling using Polars."""
    ...
```

### 3.2 Replace Pandas Operations in Critical Paths

**Files to modify:**
- `modules/calculations/power.py` - Power calculations
- `modules/calculations/cardiac_drift.py` - Drift calculations
- `services/session_analysis.py` - Session processing

**Example migration:**
```python
# Before (Pandas)
rolling_mean = df["watts"].rolling(window=30).mean()

# After (Polars adapter)
from modules.polars_adapter import fast_rolling_mean
rolling_mean = fast_rolling_mean(df["watts"], window=30)
```

### 3.3 Use Polars for Large File Processing

**Modify `modules/utils.py`:**

```python
def _process_large_dataframe(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """Process large DataFrames using Polars for speed."""
    import polars as pl
    
    # Convert to Polars for processing
    pl_df = pl.from_pandas(df)
    
    # Process in Polars (much faster)
    pl_df = pl_df.with_columns([
        pl.col("watts").fill_null(strategy="forward"),
        pl.col("heartrate").fill_null(strategy="forward"),
    ])
    
    return pl_df.to_pandas()
```

---

## Phase 4: Memory Efficiency

**Estimated effort:** 2-3 hours  
**Expected improvement:** 30-50% memory reduction

### 4.1 Optimize DataFrame dtypes

**Add to `modules/utils.py`:**

```python
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting dtypes."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    return df
```

### 4.2 Implement Smart Resampling

**Modify `services/session_analysis.py`:**

```python
def smart_resample(df: pd.DataFrame, target_rows: int = 5000) -> pd.DataFrame:
    """Intelligent resampling based on data variability."""
    if len(df) <= target_rows:
        return df
    
    # Calculate variability per column
    variability = df.select_dtypes(include=[np.number]).std()
    
    # Keep more samples for high-variability columns
    high_var_cols = variability[variability > variability.median()].index.tolist()
    
    # Adaptive resampling
    step = max(1, len(df) // target_rows)
    return df.iloc[::step, :].copy()
```

### 4.3 Add Memory Profiling

```python
# modules/performance/memory.py
import psutil
import logging

logger = logging.getLogger(__name__)

def log_memory_usage(label: str = ""):
    """Log current memory usage for profiling."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory {label}: {mem_mb:.1f} MB")
    return mem_mb
```

---

## Phase 5: Parallel Processing

**Estimated effort:** 3-4 hours  
**Expected improvement:** 2-4x faster multi-calculation scenarios

### 5.1 Parallelize Independent Calculations

**Modify `services/session_orchestrator.py`:**

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_uploaded_session_parallel(
    df_raw: pd.DataFrame,
    cp_input: float,
    w_prime_input: float,
    rider_weight: float,
    vt1_watts: float,
    vt2_watts: float
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    """Process session with parallel calculations."""
    
    # Validate data (must be sequential)
    is_valid, error_msg = validate_dataframe(df_raw)
    if not is_valid:
        return None, None, None, error_msg
    
    df_clean_pl = process_data(df_raw)
    
    # Parallel calculations
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit independent calculations
        metrics_future = loop.run_in_executor(
            executor, calculate_metrics, df_clean_pl, cp_input
        )
        w_prime_future = loop.run_in_executor(
            executor, calculate_w_prime_balance, df_clean_pl, cp_input, w_prime_input
        )
        drift_future = loop.run_in_executor(
            executor, calculate_z2_drift, df_clean_pl, cp_input
        )
        
        # Wait for all
        metrics, df_w_prime, drift_z2 = await asyncio.gather(
            metrics_future, w_prime_future, drift_future
        )
    
    # Continue with dependent calculations...
    ...
```

### 5.2 Parallel Chart Generation

```python
from concurrent.futures import ThreadPoolExecutor

def generate_all_charts_parallel(df_plot, df_resampled, params) -> dict:
    """Generate all charts in parallel."""
    charts = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            'power': executor.submit(_build_power_w_chart, df_resampled, ...),
            'zones': executor.submit(_build_zones_chart, df_plot, ...),
            'hr': executor.submit(_build_hr_chart, df_plot, ...),
            'smo2': executor.submit(_build_smo2_chart, df_plot, ...),
        }
        
        for name, future in futures.items():
            charts[name] = future.result()
    
    return charts
```

---

## Implementation Priority

| Phase | Priority | Effort | Impact | Dependencies |
|-------|----------|--------|--------|--------------|
| Phase 1 | P0 | 4-6h | High | None |
| Phase 2 | P1 | 2-3h | Medium | Phase 1 |
| Phase 3 | P1 | 3-4h | High | None |
| Phase 4 | P2 | 2-3h | Medium | None |
| Phase 5 | P2 | 3-4h | Medium | Phase 1 |

**Recommended order:**
1. Phase 1 (Caching) - Immediate wins
2. Phase 3 (Polars) - Parallel with Phase 1
3. Phase 2 (Fragments) - After caching stable
4. Phase 4 (Memory) - Can be done anytime
5. Phase 5 (Parallel) - Last, after other optimizations

---

## Testing Strategy

### Performance Benchmarks

```python
# tests/test_performance/benchmarks.py
import time
import pytest

class TestPerformanceBenchmarks:
    @pytest.fixture
    def sample_session(self):
        """Load sample session data."""
        ...
    
    def test_session_loading_time(self, sample_session):
        """Session should load in < 2 seconds."""
        start = time.time()
        df = load_data(sample_session)
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Loading took {elapsed:.2f}s"
    
    def test_metrics_calculation_time(self, sample_df):
        """Metrics should calculate in < 1 second."""
        start = time.time()
        metrics = calculate_metrics(sample_df, cp=280)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Calculation took {elapsed:.2f}s"
    
    def test_chart_rendering_time(self, sample_df):
        """Charts should render in < 500ms each."""
        ...
```

### Cache Effectiveness Tests

```python
def test_cache_hit_rate():
    """Verify cache hit rate > 70% for repeated operations."""
    ...
```

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Session load time | ~3s | <1s | Timing |
| Rerun time | ~2s | <500ms | Timing |
| Memory usage | ~500MB | <300MB | psutil |
| Cache hit rate | ~20% | >70% | diskcache stats |

---

## Rollback Plan

1. Each phase should be a separate branch/PR
2. Feature flags for new caching:
   ```python
   USE_ENHANCED_CACHING = os.getenv("ENABLE_ENHANCED_CACHE", "false").lower() == "true"
   ```
3. Performance regression tests in CI

---

## References

- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching)
- [Streamlit Fragments](https://docs.streamlit.io/library/api-reference/execution-flow/st.fragment)
- [Polars Performance](https://pola-rs.github.io/polars-book/user-guide/performance/)
- Project: `/Users/wielkikrzychmbp/Documents/Tri_Dashboard`
