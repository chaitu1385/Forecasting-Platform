# Architecture

Visual architecture diagrams for the Forecasting Product. Each section starts with a **big-picture overview** (grasp in 5 minutes) followed by a **deep dive** with class-level detail.

> All diagrams use [Mermaid](https://mermaid.js.org/) — GitHub renders them natively.

---

## 1. System Overview

How the major subsystems connect. Config drives everything; data flows left-to-right from ingestion through pipelines to consumers.

```mermaid
graph TD
    subgraph Config["Configuration"]
        YAML["YAML Files<br/><small>base / platform / fabric / LOB</small>"]
        Schema["PlatformConfig<br/><small>dataclass schema</small>"]
        YAML --> Schema
    end

    subgraph DataLayer["Data Layer"]
        FC["FileClassifier"]
        FM["MultiFileMerger"]
        DV["DataValidator"]
        DC["DemandCleanser"]
        RS["RegressorScreen"]
        ER["External Regressors<br/><small>holidays, promotions</small>"]
    end

    subgraph SeriesPrep["Series Preparation"]
        SB["SeriesBuilder<br/><small>orchestrates all data prep</small>"]
    end

    subgraph Models["Model Registry"]
        REG["ForecasterRegistry"]
        NAI["Naive"]
        STAT["Statistical<br/><small>ARIMA, ETS, Theta, MSTL</small>"]
        ML["ML<br/><small>LightGBM, XGBoost</small>"]
        NEU["Neural<br/><small>N-BEATS, NHITS, TFT</small>"]
        FND["Foundation<br/><small>Chronos, TimeGPT</small>"]
        INT["Intermittent<br/><small>Croston, TSB</small>"]
        ENS["Ensemble<br/><small>Weighted</small>"]
        REG --- NAI & STAT & ML & NEU & FND & INT & ENS
    end

    subgraph Pipelines["Pipelines"]
        BP["Backtest Pipeline<br/><small>walk-forward CV</small>"]
        FP["Forecast Pipeline<br/><small>champion inference</small>"]
        BR["BatchRunner<br/><small>parallel execution</small>"]
        SCH["Scheduler<br/><small>recurring runs</small>"]
    end

    subgraph Hierarchy["Hierarchy"]
        HT["HierarchyTree"]
        AGG["Aggregator"]
        REC["Reconciler<br/><small>OLS / WLS / MinT</small>"]
    end

    subgraph Storage["Parquet Storage"]
        MS["MetricStore<br/><small>backtest metrics</small>"]
        FS["Forecast Files"]
        MF["Manifests<br/><small>provenance JSON</small>"]
        AL["Audit Logs"]
    end

    subgraph Consumers["Consumers"]
        API["FastAPI REST<br/><small>auth-protected</small>"]
        NX["Next.js Frontend<br/><small>10 pages, REST client</small>"]
        AI["AI Features<br/><small>NL Query, Triage,<br/>Commentary, Config Tuner</small>"]
    end

    subgraph Observability["Observability"]
        LOG["StructuredLogger"]
        MET["MetricsEmitter"]
        ALR["AlertDispatcher"]
        CST["CostEstimator"]
    end

    Schema --> SB & BP & FP
    FC & FM --> SB
    DV & DC & RS & ER --> SB
    SB --> BP & FP
    REG --> BP & FP
    BP --> MS
    FP --> FS & MF
    BP --> REC
    FP --> REC
    HT & AGG --> REC
    MS & FS & MF --> API
    API --> AI
    BP & FP --> LOG & MET & ALR & CST
    LOG & MET --> AL
```

### Deep Dive: Configuration Hierarchy

Every pipeline parameter is defined via nested dataclasses in `src/config/schema.py`.

```mermaid
graph TD
    PC["PlatformConfig"]
    PC --> FC2["ForecastConfig<br/><small>horizon, frequency,<br/>models, quantiles</small>"]
    PC --> BC["BacktestConfig<br/><small>n_folds, val_periods,<br/>champion strategy</small>"]
    PC --> DQ["DataQualityConfig"]
    PC --> ERC["ExternalRegressorConfig"]
    PC --> HC["HierarchyConfig<br/><small>levels, method</small>"]
    PC --> RECC["ReconciliationConfig<br/><small>method, shrinkage</small>"]
    PC --> OC["ObservabilityConfig"]
    PC --> PAR["ParallelismConfig<br/><small>n_workers, batch_size,<br/>backend</small>"]
    PC --> AIC["AIConfig<br/><small>model, temperature,<br/>max_tokens</small>"]
    PC --> CAL["CalibrationConfig<br/><small>conformal alpha,<br/>method</small>"]
    PC --> CON["ConstraintConfig<br/><small>capacity, budget,<br/>min/max bounds</small>"]

    DQ --> VC["ValidationConfig<br/><small>enabled, checks,<br/>fail_on_error</small>"]
    DQ --> CC["CleansingConfig<br/><small>outlier method,<br/>stockout imputation</small>"]

    ERC --> RSC["RegressorScreenConfig<br/><small>variance threshold,<br/>correlation, MI</small>"]

    OC --> AC["AlertConfig<br/><small>drift threshold,<br/>webhook URL</small>"]
```

---

## 2. Data Flow

How raw data transforms into model-ready series, then splits into backtest and forecast paths.

```mermaid
graph LR
    RAW["Raw CSV / Parquet"]

    subgraph Preparation["SeriesBuilder.build()"]
        direction LR
        V["Validate<br/><small>schema, duplicates,<br/>frequency checks</small>"]
        SID["Build Series ID<br/><small>concat grain columns<br/>with | separator</small>"]
        TR["SKU Transitions<br/><small>stitch discontinued<br/>→ new SKUs</small>"]
        GF["Gap Fill<br/><small>complete date range,<br/>zero-fill missing</small>"]
        SBD["Structural Breaks<br/><small>detect & truncate<br/>to post-break</small>"]
        CL["Demand Cleansing<br/><small>outlier cap/replace,<br/>stockout imputation</small>"]
        DS["Drop Short Series<br/><small>min_series_length,<br/>zero-only filter</small>"]
        JEF["Join External<br/>Features<br/><small>promotions, holidays</small>"]
        SCR["Regressor Screen<br/><small>variance, correlation,<br/>MI filtering</small>"]

        V --> SID --> TR --> GF --> SBD --> CL --> DS --> JEF --> SCR
    end

    RAW --> V
    SCR --> MR["Model-Ready Series<br/><small>[series_id, week, quantity, ...features]</small>"]

    MR --> BT["Backtest Path"]
    MR --> FC3["Forecast Path"]

    subgraph BacktestPath["Backtest"]
        direction LR
        WF["Walk-Forward CV<br/><small>n_folds × n_models</small>"]
        MET2["Compute Metrics<br/><small>WMAPE, bias, MAE,<br/>MASE, RMSE</small>"]
        CS["Champion Selection<br/><small>rank by WMAPE,<br/>pick best model(s)</small>"]
        WF --> MET2 --> CS
    end

    subgraph ForecastPath["Forecast"]
        direction LR
        FIT["Fit Champion<br/><small>on full history</small>"]
        PRED["Predict Horizon<br/><small>point forecasts</small>"]
        QNT["Quantiles<br/><small>P10, P50, P90</small>"]
        CONF["Conformal<br/>Correction<br/><small>calibrate intervals</small>"]
        HREC["Reconcile<br/><small>hierarchy coherence</small>"]
        OUT["Output<br/><small>Parquet + Manifest</small>"]
        FIT --> PRED --> QNT --> CONF --> HREC --> OUT
    end

    BT --> WF
    FC3 --> FIT
    CS -.->|champion model| FIT
```

### Deep Dive: SeriesBuilder Internals

Each step is toggled by config flags. The `build()` method is the single entry point.

```mermaid
graph TD
    BUILD["build(actuals, external_features, product_master)"]

    BUILD --> CHK1{{"data_quality.validation<br/>.enabled?"}}
    CHK1 -->|Yes| VAL["DataValidator.validate()<br/><small>→ ValidationReport</small>"]
    CHK1 -->|No| SID2["Build series_id"]
    VAL --> SID2

    SID2 --> CHK2{{"sku_mapping<br/>.enabled?"}}
    CHK2 -->|Yes| TRANS["TransitionEngine<br/>.compute_plans()"]
    CHK2 -->|No| FILL["Fill gaps"]
    TRANS --> FILL

    FILL --> CHK3{{"structural_break<br/>.enabled?"}}
    CHK3 -->|Yes| BREAK["StructuralBreakDetector<br/>.detect()"]
    CHK3 -->|No| CLEAN["Demand cleansing"]
    BREAK --> CLEAN

    CLEAN --> CHK4{{"data_quality.cleansing<br/>.enabled?"}}
    CHK4 -->|Yes| CLNS["DemandCleanser.cleanse()<br/><small>→ CleansingResult</small>"]
    CHK4 -->|No| DROP["Drop short series"]
    CLNS --> DROP

    DROP --> CHK5{{"external_features<br/>provided?"}}
    CHK5 -->|Yes| JOIN["Left-join on<br/>[series_id, time_col]"]
    CHK5 -->|No| RET["Return model-ready DataFrame"]
    JOIN --> CHK6{{"regressor_screen<br/>.enabled?"}}
    CHK6 -->|Yes| SCRN["screen_regressors()<br/><small>drop low-quality features</small>"]
    CHK6 -->|No| RET
    SCRN --> RET
```

---

## 3. Backtest Pipeline

Walk-forward cross-validation evaluates every configured model across multiple time folds, then selects the champion.

```mermaid
sequenceDiagram
    participant S as run_backtest.py
    participant BP as BacktestPipeline
    participant SB as SeriesBuilder
    participant R as Registry
    participant BE as BacktestEngine
    participant PP as ProcessPool
    participant F as Forecaster
    participant MS as MetricStore
    participant CS as ChampionSelector

    S->>BP: run(actuals, product_master)
    activate BP

    BP->>SB: build(actuals)
    activate SB
    SB-->>BP: model-ready series
    deactivate SB

    BP->>R: build_from_config(model_list)
    R-->>BP: [Forecaster, Forecaster, ...]

    BP->>BE: run(series, forecasters)
    activate BE

    Note over BE: Split into n_folds<br/>train/validation windows

    loop Each fold x model
        BE->>PP: submit(_run_model_in_process)
        activate PP
        PP->>F: fit(train_fold)
        F-->>PP: fitted model
        PP->>F: predict(horizon)
        F-->>PP: forecasts
        PP->>PP: compute_all_metrics(actual, forecast)
        PP-->>BE: IPC bytes (metrics)
        deactivate PP
    end

    BE->>MS: write(all_fold_metrics)
    BE-->>BP: backtest_results
    deactivate BE

    BP->>CS: select(backtest_results)
    CS-->>BP: champions DataFrame

    BP->>MS: leaderboard(lob)
    MS-->>BP: ranked model table

    BP-->>S: {backtest_results, champions,<br/>leaderboard, calibration_report}
    deactivate BP
```

### Deep Dive: Walk-Forward Cross-Validation

The training window expands (or slides) and the validation window moves forward in time. Each fold evaluates all models on the same held-out period.

```mermaid
gantt
    title Walk-Forward CV (expanding window, 3 folds)
    dateFormat  YYYY-MM
    axisFormat  %Y-%m
    section Fold 1
        Training      :a1, 2023-01, 12M
        Validation    :crit, a2, 2024-01, 3M
    section Fold 2
        Training      :a3, 2023-01, 15M
        Validation    :crit, a4, 2024-04, 3M
    section Fold 3
        Training      :a5, 2023-01, 18M
        Validation    :crit, a6, 2024-07, 3M
```

### Deep Dive: Model Registry

All forecasters inherit from `BaseForecaster` and register via the `@registry.register()` decorator.

```mermaid
classDiagram
    class BaseForecaster {
        <<abstract>>
        +name: str
        +fit(df, target_col, time_col, id_col)
        +predict(horizon, id_col, time_col) DataFrame
        +predict_quantiles(horizon, quantiles) DataFrame
        +validate_and_prepare(df) DataFrame
    }

    class SeasonalNaiveForecaster {
        name = "naive_seasonal"
    }

    class AutoARIMAForecaster {
        name = "auto_arima"
    }
    class AutoETSForecaster {
        name = "auto_ets"
    }
    class AutoThetaForecaster {
        name = "auto_theta"
    }
    class MSTLForecaster {
        name = "mstl"
    }

    class LGBMDirectForecaster {
        name = "lgbm_direct"
    }
    class XGBoostDirectForecaster {
        name = "xgboost_direct"
    }

    class NBEATSForecaster {
        name = "nbeats"
    }
    class NHITSForecaster {
        name = "nhits"
    }
    class TFTForecaster {
        name = "tft"
    }

    class ChronosForecaster {
        name = "chronos"
    }
    class TimeGPTForecaster {
        name = "timegpt"
    }

    class CrostonForecaster {
        name = "croston"
    }
    class CrostonSBAForecaster {
        name = "croston_sba"
    }
    class TSBForecaster {
        name = "tsb"
    }

    class WeightedEnsembleForecaster {
        name = "weighted_ensemble"
    }

    class HierarchicalForecaster {
        name = "hierarchical"
    }

    class ConstrainedDemandEstimator {
        name = "constrained"
    }

    BaseForecaster <|-- SeasonalNaiveForecaster
    BaseForecaster <|-- AutoARIMAForecaster
    BaseForecaster <|-- AutoETSForecaster
    BaseForecaster <|-- AutoThetaForecaster
    BaseForecaster <|-- MSTLForecaster
    BaseForecaster <|-- LGBMDirectForecaster
    BaseForecaster <|-- XGBoostDirectForecaster
    BaseForecaster <|-- NBEATSForecaster
    BaseForecaster <|-- NHITSForecaster
    BaseForecaster <|-- TFTForecaster
    BaseForecaster <|-- ChronosForecaster
    BaseForecaster <|-- TimeGPTForecaster
    BaseForecaster <|-- CrostonForecaster
    BaseForecaster <|-- CrostonSBAForecaster
    BaseForecaster <|-- TSBForecaster
    BaseForecaster <|-- WeightedEnsembleForecaster
    BaseForecaster <|-- HierarchicalForecaster
    BaseForecaster <|-- ConstrainedDemandEstimator
```

---

## 4. Forecast Pipeline

Production inference: fit the champion model on all available data and generate forecasts with prediction intervals.

```mermaid
sequenceDiagram
    participant S as run_forecast.py
    participant FP as ForecastPipeline
    participant SB as SeriesBuilder
    participant R as Registry
    participant F as Forecaster
    participant RC as Reconciler
    participant M as Manifest

    S->>FP: run(actuals, champion_model_id)
    activate FP

    FP->>SB: build(actuals, external_features)
    SB-->>FP: model-ready series

    FP->>R: build(champion_model_id)
    R-->>FP: Forecaster instance

    FP->>F: fit(series, target_col, time_col, id_col)
    F-->>FP: fitted

    FP->>F: predict(horizon)
    F-->>FP: point forecasts

    opt Quantile forecasts enabled
        FP->>F: predict_quantiles(horizon, [0.1, 0.5, 0.9])
        F-->>FP: quantile forecasts
    end

    opt Conformal correction
        FP->>FP: apply_conformal_correction(forecast, residuals)
    end

    opt Hierarchy enabled
        FP->>RC: reconcile(forecasts, hierarchy_tree)
        RC-->>FP: coherent forecasts
    end

    FP->>FP: write forecast Parquet

    FP->>M: build_manifest(run_id, config, stats)
    M-->>FP: manifest JSON
    FP->>M: write_manifest(path)

    FP-->>S: forecast DataFrame
    deactivate FP
```

### Deep Dive: Hierarchical Reconciliation

Ensures forecasts are coherent — child node forecasts sum to parent nodes at every level of the hierarchy.

```mermaid
graph TD
    LEAF["Leaf-Level Base Forecasts<br/><small>e.g., per store × SKU</small>"]

    LEAF --> TREE["HierarchyTree<br/><small>build summing matrix S</small>"]
    TREE --> AGG2["Aggregator<br/><small>bottom-up aggregation<br/>to all hierarchy levels</small>"]
    AGG2 --> BASE["Base Forecasts at All Levels<br/><small>total, region, store, SKU</small>"]

    BASE --> METHOD{{"Reconciliation<br/>Method"}}

    METHOD -->|bottom_up| BU["Keep leaf forecasts<br/>aggregate upward"]
    METHOD -->|top_down| TD2["Disaggregate top-level<br/>by historical proportions"]
    METHOD -->|middle_out| MO["Slice at mid-level<br/>reconcile up & down"]
    METHOD -->|ols| OLS2["G = inv(S'S) * S'<br/>P_reconciled = S * G * P_base"]
    METHOD -->|wls| WLS2["G = inv(S'WS) * S'W<br/>W = diag(1/variance)"]
    METHOD -->|mint| MINT["G = inv(S'W_hS) * S'W_h<br/>W_h = shrinkage covariance"]

    BU & TD2 & MO & OLS2 & WLS2 & MINT --> COH["Coherent Forecasts<br/><small>sum(children) = parent<br/>at every node</small>"]
```

---

## 5. Data Onboarding

The multi-file upload workflow: classify files, merge them, analyze the result, and generate a recommended config.

```mermaid
graph TD
    UP["User uploads<br/>1+ CSV/Parquet files"]

    UP --> CLASS["FileClassifier.classify_files()"]

    subgraph TwoPass["Two-Pass Classification"]
        direction TB
        P1["Pass 1: Isolation Scoring<br/><small>Score each file independently<br/>for time_series signals</small>"]
        P2["Pass 2: Cross-File Resolution<br/><small>Pick best primary, then re-evaluate<br/>remaining as dimension / regressor</small>"]
        P1 --> P2
    end

    CLASS --> TwoPass
    TwoPass --> ROLES["Classification Result<br/><small>primary: sales.csv<br/>dimension: stores.csv<br/>regressor: promotions.csv</small>"]

    ROLES --> CONFIRM1{"User confirms<br/>or overrides roles"}

    CONFIRM1 --> MERGE["MultiFileMerger"]

    subgraph MergeSteps["Merge Process"]
        direction TB
        DJK["Detect Join Keys<br/><small>dimension: shared ID cols<br/>regressor: time_col ± id_col</small>"]
        PREV["Generate Preview<br/><small>sample rows, match rate,<br/>null fills, conflicts</small>"]
        EXEC["Execute Left-Joins<br/><small>dimension on IDs<br/>regressor on time ± IDs<br/>fill nulls with 0</small>"]
        DJK --> PREV --> EXEC
    end

    MERGE --> MergeSteps

    EXEC --> CONFIRM2{"User confirms<br/>merge preview"}

    CONFIRM2 --> ANALYZE["DataAnalyzer.analyze()"]

    subgraph Analysis["Analysis"]
        direction TB
        SCHEMA["Schema Detection<br/><small>time col, target col,<br/>ID cols, grain</small>"]
        FORE["Forecastability<br/><small>CV, entropy,<br/>intermittent ratio</small>"]
        HIER["Hierarchy Inference<br/><small>cardinality-based<br/>level detection</small>"]
        SCHEMA --> FORE --> HIER
    end

    ANALYZE --> Analysis
    Analysis --> CFG["Recommended PlatformConfig<br/><small>YAML with models, horizon,<br/>frequency, hierarchy</small>"]
    CFG --> ACCEPT{"User accepts config"}
    ACCEPT --> READY["Ready for Pipeline<br/><small>run_backtest.py or<br/>run_forecast.py</small>"]
```

### Deep Dive: Classification Heuristics

The isolation scoring pass uses weighted signals to determine if a file is a time series.

```mermaid
graph LR
    FILE["Input File"]

    FILE --> S1["Date Column?<br/><small>Date/Datetime dtype<br/>or name in patterns</small><br/><strong>+0.30</strong>"]
    FILE --> S2["Target Column?<br/><small>name matches: quantity,<br/>sales, demand, revenue...</small><br/><strong>+0.20</strong>"]
    FILE --> S3["ID Repetition?<br/><small>columns with repeated<br/>values (group keys)</small><br/><strong>+0.20</strong>"]
    FILE --> S4["Row Count?<br/><small>high row count suggests<br/>panel/time series data</small><br/><strong>+0.15</strong>"]
    FILE --> S5["Time Column Name?<br/><small>matches: week, date,<br/>ds, timestamp, period</small><br/><strong>+0.15</strong>"]

    S1 & S2 & S3 & S4 & S5 --> SCORE["Time-Series Score<br/><small>0.0 – 1.0</small>"]
    SCORE --> DEC{{"Score > 0.5?"}}
    DEC -->|Yes| TS["Candidate: time_series"]
    DEC -->|No| OTHER["Candidate: dimension<br/>or regressor or unknown"]
```

---

## 6. API & Frontend

How external consumers — REST clients, frontend users, and AI features — access the platform.

```mermaid
graph TD
    subgraph Clients["External Clients"]
        REST["REST API Clients"]
        BROWSER["Browser<br/><small>Next.js Frontend</small>"]
    end

    subgraph Auth["Authentication"]
        SETTINGS["AuthSettings<br/><small>invalid production config fails startup</small>"]
        JWT["Strict JWT Verification"]
        ME["/auth/me<br/><small>verified identity and expiry</small>"]
        RBAC["RBAC<br/><small>5 roles, 11 permissions</small>"]
    end

    OPERATOR["Trusted local operator<br/><small>signing key, role assignment</small>"]
    ISSUE["issue_token.py<br/><small>new owner-only file, short expiry</small>"]
    PUBLIC["Public /auth/status and /health<br/><small>no signing configuration</small>"]

    subgraph APILayer["FastAPI Endpoints"]
        F2["/forecast/{lob}"]
        F3["/forecast/{lob}/{series_id}"]
        L["/metrics/leaderboard/{lob}"]
        D["/metrics/drift/{lob}"]
        AN["/analyze"]
        AIE1["/ai/explain"]
        AIE2["/ai/triage"]
        AIE3["/ai/recommend-config"]
        AIE4["/ai/commentary"]
    end

    subgraph DataStores["Data Stores"]
        MS2["MetricStore<br/><small>Parquet</small>"]
        FS2["Forecast Files<br/><small>Parquet</small>"]
        MF2["Manifests<br/><small>JSON</small>"]
        CLAUDE2["Claude API<br/><small>AI features</small>"]
    end

    REST --> JWT --> RBAC --> APILayer
    BROWSER --> JWT
    SETTINGS -.-> JWT
    JWT --> ME
    OPERATOR --> ISSUE --> BROWSER
    BROWSER --> PUBLIC

    F2 & F3 --> FS2
    L & D --> MS2
    AIE1 & AIE2 & AIE3 & AIE4 --> CLAUDE2
```

### Deep Dive: API Endpoint Map

| Method | Path | Auth | Data Source | Description |
|--------|------|------|-------------|-------------|
| `GET` | `/health` | Public | — | Bounded readiness status |
| `GET` | `/auth/status` | Public | AuthSettings mode only | Fail-closed mode discovery |
| `GET` | `/auth/me` | Verified JWT | Signed identity + server RBAC | Identity, role, permissions, expiry |
| `GET` | `/forecast/{lob}` | Verified JWT | Forecast Parquet | Latest forecast for LOB |
| `GET` | `/forecast/{lob}/{series_id}` | Verified JWT | Forecast Parquet | Single series forecast |
| `GET` | `/metrics/leaderboard/{lob}` | Verified JWT | MetricStore | Model leaderboard |
| `GET` | `/metrics/drift/{lob}` | Verified JWT | MetricStore | Drift alerts |
| `POST` | `/analyze` | Verified JWT | Upload CSV | Auto-detect schema, recommend config |
| `POST` | `/ai/explain` | Verified JWT | Claude API | NL query about forecasts |
| `POST` | `/ai/triage` | Verified JWT | Claude API | Triage drift alerts by impact |
| `POST` | `/ai/recommend-config` | Verified JWT | Claude API | Config tuning recommendations |
| `POST` | `/ai/commentary` | Verified JWT | Claude API | Executive forecast commentary |

This table describes production authentication. Individual endpoints additionally
enforce their server-side permission dependencies. Explicit demo replaces identity
verification with simulated roles and must never serve production data.

### Deep Dive: Next.js Frontend

The Next.js frontend communicates with the FastAPI backend over REST. It provides the 10-page workflow UI (login + 9 workflow pages) and runs as a standalone Node.js application.

```mermaid
graph TB
    subgraph NextJS["Next.js Frontend (port 3000)"]
        direction TB
        NL["Login"] --> NP["9 Workflow Pages"]
        NP --> NH["React Query Hooks"]
        NH --> NC["API Client<br/><small>typed fetch + JWT</small>"]
    end

    subgraph API["FastAPI Backend (port 8000)"]
        EP["41 REST Endpoints"]
    end

    NC -->|HTTP/JSON| EP

    subgraph LiveFeatures["Core Features"]
        LF1["File upload / analysis"]
        LF2["Model leaderboard"]
        LF3["Drift alerts + audit log"]
        LF4["AI: explain, triage, config, commentary"]
    end

    subgraph ImplementedFeatures["Now Live"]
        IF1["Multi-file classification → /pipeline/analyze-multi-file"]
        IF2["Pipeline execution → /pipeline/backtest, /pipeline/forecast"]
        IF3["Hierarchy / SKU ops → /hierarchy/*, /sku-mapping/*"]
        IF4["SHAP / BI export → /analytics/shap, /governance/export"]
    end
```

---

## 7. Observability & Audit

Cross-cutting concerns that run alongside every pipeline execution.

```mermaid
graph TD
    subgraph Pipeline["Pipeline Execution"]
        CTX["PipelineContext<br/><small>run_id, lob, started_at,<br/>parent_run_id, tags</small>"]
    end

    CTX --> LOG2["StructuredLogger<br/><small>JSON logs with<br/>correlation IDs</small>"]
    CTX --> MET3["MetricsEmitter<br/><small>timing, counters,<br/>gauges</small>"]
    CTX --> CST2["CostEstimator<br/><small>compute time, model<br/>inference cost</small>"]

    LOG2 --> STDOUT["stdout / file<br/><small>structured JSON</small>"]
    MET3 --> STATSD["StatsD / Log<br/><small>backend</small>"]
    CST2 --> MANIFEST2["Pipeline Manifest<br/><small>total_seconds, cost</small>"]

    subgraph DriftDetection["Drift Detection"]
        MS3["MetricStore<br/><small>historical metrics</small>"]
        DD["ForecastDriftDetector<br/><small>compare recent vs.<br/>historical WMAPE/bias</small>"]
        MS3 --> DD
    end

    DD --> ALR2["AlertDispatcher"]
    ALR2 --> WH["Webhooks<br/><small>Slack, Teams,<br/>PagerDuty</small>"]

    subgraph Audit["Audit Trail"]
        AUD["AuditLogger<br/><small>append-only</small>"]
        AUD --> PQ["Parquet Files<br/><small>partitioned by date</small>"]
    end

    LOG2 --> AUD
    MANIFEST2 --> JSON2["JSON Sidecar<br/><small>alongside each<br/>forecast Parquet</small>"]
```

---

## 8. Data Schemas

The four core data structures that flow through the platform.

### Series DataFrame (input to models)

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | `Utf8` | Composite key, e.g. `"US\|East\|SKU123"` |
| `week` | `Date` | Period start date (Monday for weekly) |
| `quantity` | `Float64` | Demand / sales value |
| `<feature_1>` | `Float64` | External regressor (optional) |
| `<feature_N>` | `Float64` | Additional features (optional) |

### Forecast Output

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | `Utf8` | Matches input series |
| `week` | `Date` | Future period date |
| `forecast` | `Float64` | Point forecast |
| `forecast_p10` | `Float64` | 10th percentile (optional) |
| `forecast_p50` | `Float64` | Median forecast (optional) |
| `forecast_p90` | `Float64` | 90th percentile (optional) |

### Metric Store Record

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | `Utf8` | Unique run identifier |
| `run_type` | `Utf8` | `"backtest"` or `"live"` |
| `run_date` | `Date` | Execution date |
| `lob` | `Utf8` | Line of business |
| `model_id` | `Utf8` | Model name from registry |
| `fold` | `Int32` | CV fold index |
| `series_id` | `Utf8` | Series identifier |
| `target_week` | `Date` | Forecast target date |
| `forecast_step` | `Int32` | Steps ahead (1..horizon) |
| `actual` | `Float64` | Ground truth value |
| `forecast` | `Float64` | Predicted value |
| `wmape` | `Float64` | Weighted MAPE |
| `normalized_bias` | `Float64` | Bias / mean(actuals) |
| `mae` | `Float64` | Mean Absolute Error |
| `rmse` | `Float64` | Root Mean Squared Error |
| `mase` | `Float64` | Mean Absolute Scaled Error |

### Pipeline Manifest (JSON)

```json
{
  "run_id": "abc123def456",
  "timestamp": "2024-06-01T14:30:00",
  "lob": "retail",
  "input_data_hash": "sha256:...",
  "input_row_count": 52000,
  "input_series_count": 1000,
  "date_range_start": "2022-01-01",
  "date_range_end": "2024-05-31",
  "cleansing_applied": true,
  "outliers_clipped": 145,
  "stockout_periods_imputed": 23,
  "validation_passed": true,
  "regressors_dropped": ["low_variance_feature"],
  "config_hash": "sha256:...",
  "champion_model_id": "lgbm_direct",
  "backtest_wmape": 0.0842,
  "forecast_horizon": 39,
  "forecast_row_count": 39000,
  "forecast_file": "forecast_retail_2024-06-01.parquet"
}
```
