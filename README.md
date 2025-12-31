# Tri_Dashboard: Advanced Physiological Analysis Platform

## Purpose
Tri_Dashboard is a specialized analytics platform designed for sports scientists and elite coaches. It goes beyond standard power duration curves to provide deep insights into:
- **W' Balance & Reconstitution**: Modeling anaerobic work capacity depletion and recovery.
- **Muscle Oxygenation (SmO2)**: Analysis of NIRS data (Moxy/TrainRed) to identify physiological breakpoints.
- **VO2 Kinetics**: Estimation of O2 deficit and aerobic contribution.
- **Fatigue Resistance**: Quantification of durability (FRI) and cardiac drift.

## Architecture Overview
The application follows a modular **Service-Oriented Architecture (SOA)** within a monolithic codebase, separating scientific logic from presentation.

```mermaid
graph TD
    UI[Frontend (Streamlit)] --> Layout[Layout/Theme Manager]
    Layout --> App[App Orchestrator]
    App --> Services[Services Layer]
    
    subgraph Core Logic
        Services --> Validation[Data Validation]
        Services --> Analysis[Session Metrics]
        Services --> Orch[Orchestrator]
    end
    
    subgraph Scientific Modules
        Analysis --> Calc[modules.calculations]
        Calc --> WPrime[W' Balance]
        Calc --> Power[Power/PDC]
        Calc --> Reform[Physiology Models]
    end
    
    subgraph Data Persistence
        Orch --> SQLite[(Session Store)]
        Orch --> Config[Config/Env]
    end
```

## Data Flow
1.  **Ingestion**: Files (CSV/FIT/TCX converted) are loaded via `modules.utils.load_data`.
2.  **Validation**: `services.data_validation` enforces schema integrity, types, and range limits.
3.  **Processing**: 
    -   Data is normalized to a standard schema (`watts`, `heartrate`, `smo2`, `time`).
    -   Resampling occurs if high-frequency noise is detected.
4.  **Analysis**:
    -   **Metrics**: TSS, NP, IF calculated via Coggan's formulas.
    -   **Models**: W' bal (Integral), CP, and VO2max estimates are computed.
5.  **Persistence**: valid sessions are indexed in `data/training_history.db`.

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-org/tri_dashboard.git
    cd tri_dashboard
    ```
2.  Install dependencies:
    ```bash
    pip install -e .[dev]
    ```
3.  Initialize the database:
    ```bash
    python init_db.py
    ```

### Configuration
Create a `.env` file to override defaults (optional):
```ini
APP_TITLE="Lab Analytics"
CP_DEFAULT=300
W_PRIME_DEFAULT=20000
DB_NAME="lab_data.db"
```

## Example Usage

### Running the Dashboard
```bash
streamlit run app.py
```
Access the interface at `http://localhost:8501`.

### Batch Import
To process a folder of historical CSV files:
1.  Place files in `treningi_csv/`.
2.  Open **Analytics > Import** tab.
3.  Click **Import All** to run the batch processor.

### Manual Analysis
The platform enables comparative analysis of:
- **Intervals**: Auto-detection of work/recovery intervals.
- **Desaturation**: SmO2 slopes during high-intensity efforts.
- **Decoupling**: Aerobic efficiency (Pa:HR) drift over long durations.
