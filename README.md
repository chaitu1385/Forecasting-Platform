# Forecasting Product

A production-grade, modular multi-frequency sales forecasting product (daily, weekly, monthly, quarterly). Covers the full lifecycle from raw data ingestion to hierarchically reconciled, explained, and governed forecasts — with a REST API, Microsoft Fabric/Delta Lake deployment layer, Spark distributed execution, and S&OP exception management.

**See also:** [QUICKSTART.md](QUICKSTART.md) — get running in 2 minutes | [ARCHITECTURE.md](ARCHITECTURE.md) — visual diagrams of system architecture and data flow | [CONCEPTS.md](CONCEPTS.md) — why each component exists | [EDGE_CASES.md](EDGE_CASES.md) — failure modes and how the platform handles them

---

## User Guides

| Guide | Description |
|-------|-------------|
| [Data Format](docs/DATA_FORMAT.md) | Input/output schemas, column rules, sample data |
| [Deployment](docs/DEPLOYMENT.md) | Docker, env vars, production checklist, scaling |
| [User Guide](docs/USER_GUIDE.md) | Model selection, backtest interpretation, AI features |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common errors, FAQ, debugging tips |
| [Frontend](docs/FRONTEND.md) | Frontend architecture, components, API client, auth flow |
| [API Reference](docs/API_REFERENCE.md) | REST API endpoints, authentication, request/response schemas |
| [Python API](docs/PYTHON_API.md) | Python library usage — pipelines, metrics, validation |
| [Production Deployment](docs/PRODUCTION_DEPLOYMENT.md) | Kubernetes, orchestration (Airflow/Dagster), monitoring |
| [AI Features](docs/AI_FEATURES.md) | Claude-powered NLQ, triage, config tuning, commentary |
| [Adding Models](docs/ADDING_MODELS.md) | How to add custom forecasting models |
| [Foundation Models](docs/FOUNDATION_MODELS.md) | Zero-shot models: Amazon Chronos, Nixtla TimeGPT |

---

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│  REST API  (FastAPI, JWT-protected)   src/api/              │
│  GET /auth/status · GET /auth/me                             │
│  41 endpoints: /health /forecast /metrics /series /hierarchy│
│  /sku-mapping /overrides /pipeline /governance /ai/*       │
│  GET /forecast/{lob}/{series_id}  /metrics/drift/{lob}     │
│  GET /audit                                                 │
│  POST /ai/explain  /ai/triage  /ai/recommend-config        │
│  POST /ai/commentary                                        │
├─────────────────────────────────────────────────────────────┤
│  AI Layer                              src/ai/              │
│  NaturalLanguageQueryEngine · AnomalyTriageEngine           │
│  ConfigTunerEngine · CommentaryEngine                       │
├─────────────────────────────────────────────────────────────┤
│  Auth & Audit                         src/auth/ src/audit/  │
│  RBAC (5 roles, 11 permissions) · JWT tokens                │
│  AuditLogger (append-only, Parquet, date-partitioned)       │
├─────────────────────────────────────────────────────────────┤
│  Analytics Layer                      src/analytics/        │
│  DataAnalyzer · ForecastabilityAnalyzer · CausalAnalyzer    │
│  LLMAnalyzer (Claude-powered interpretation)                │
│  ForecastAnalytics (notebook API) · BIExporter              │
│  ForecastComparator · ExceptionEngine                       │
│  ForecastExplainer (STL + SHAP + narrative)                 │
│  DriftDetector · ModelCard · ModelCardRegistry              │
│  ForecastLineage · FVAAnalyzer                              │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Layer                       src/pipeline/         │
│  BacktestPipeline · ForecastPipeline · PipelineManifest     │
│  BatchInferenceRunner · PipelineScheduler                   │
├─────────────────────────────────────────────────────────────┤
│  Backtest Engine                      src/backtesting/      │
│  BacktestEngine · WalkForwardCV · ChampionSelector          │
├─────────────────────────────────────────────────────────────┤
│  Observability                        src/observability/    │
│  PipelineContext · StructuredLogger · MetricsEmitter        │
│  AlertDispatcher · CostEstimator                            │
├─────────────────────────────────────────────────────────────┤
│  Override Store                       src/overrides/        │
│  OverrideStore (DuckDB) · ParquetOverrideStore (fallback)   │
├─────────────────────────────────────────────────────────────┤
│  Hierarchy Layer                      src/hierarchy/        │
│  HierarchyTree · HierarchyNode · HierarchyAggregator       │
│  Reconciler (bottom_up · top_down · middle_out)             │
│            (ols · wls · mint)                               │
├─────────────────────────────────────────────────────────────┤
│  Model Library                        src/forecasting/      │
│  SeasonalNaive · AutoARIMA · AutoETS · AutoTheta · MSTL    │
│  LGBMDirect · XGBoostDirect                                 │
│  Chronos · TimeGPT (zero-shot foundation)                   │
│  N-BEATS · NHITS · TFT (neural)                             │
│  Croston · CrostonSBA · TSB (intermittent)                  │
│  WeightedEnsemble · HierarchicalForecaster                  │
│  ConstrainedDemandEstimator · ForecasterRegistry            │
├─────────────────────────────────────────────────────────────┤
│  Metrics                              src/metrics/          │
│  MetricStore · ForecastDriftDetector · FVA engine           │
├─────────────────────────────────────────────────────────────┤
│  SKU Mapping                          src/sku_mapping/      │
│  AttributeMatching · NamingConvention · CurveFitting        │
│  TemporalComovement · CandidateFusion                       │
│  BayesianProportionEstimator                                │
├─────────────────────────────────────────────────────────────┤
│  Series Management                    src/series/           │
│  SeriesBuilder · SparseDetector · TransitionEngine          │
│  StructuralBreakDetector (CUSUM / PELT)                     │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                           src/data/             │
│  DataLoader · DataPreprocessor · FeatureEngineer            │
│  ExternalRegressorLoader · HolidayCalendar                  │
│  DataValidator · DemandCleanser · RegressorScreen           │
│  DataQualityAnalyzer (profiling & gap/zero reporting)        │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                       src/fabric/ src/spark/│
│  DeploymentOrchestrator · FabricLakehouse · DeltaWriter     │
│  SparkForecastPipeline · SparkSeriesBuilder                 │
│  SparkFeatureEngineer · SparkDataLoader                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
forecasting-product/
├── src/
│   ├── analytics/          # Data profiling, forecastability, causal analysis, LLM interpretation, BI export, explainability, governance, FVA
│   ├── api/                # FastAPI REST serving layer (auth-protected)
│   ├── audit/              # Append-only Parquet audit log (immutable, date-partitioned)
│   ├── auth/               # Fail-closed settings, RBAC/JWT, private operator token output
│   ├── backtesting/        # Walk-forward backtest engine, champion selection
│   ├── config/             # YAML config schema + loader (incl. external regressor config)
│   ├── data/               # Data loading, preprocessing, validation, demand cleansing, regressor screening, external regressors, quality profiling
│   ├── evaluation/         # Metrics (WMAPE, RMSPE, bias, MAE) + evaluator
│   ├── fabric/             # Microsoft Fabric / Delta Lake deployment
│   ├── forecasting/        # Model implementations + registry (statistical, ML, neural, foundation, intermittent)
│   ├── hierarchy/          # Hierarchy tree, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore, drift detection, metric definitions, FVA computation
│   ├── models/             # LightGBM + XGBoost wrappers (legacy)
│   ├── observability/      # Structured logging, metrics, alerts, cost tracking
│   ├── overrides/          # Planner manual override store (DuckDB + Parquet fallback)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines, provenance manifest, batch runner, scheduler
│   ├── series/             # Series builder, sparse detector, lifecycle transitions, structural break detection
│   ├── sku_mapping/        # New/discontinued SKU mapping (4 methods + Bayesian fusion)
│   ├── spark/              # PySpark distributed execution layer
│   └── utils/              # Logger, config utilities
├── frontend/               # Next.js 16 / Node 24 (TypeScript, Tailwind, Recharts)
│   ├── src/app/            # App Router pages (login + 9 workflow pages)
│   ├── src/components/     # Reusable components (charts, AI panels, layout, shared)
│   │   ├── forecast/       # Decomposition, comparison, constrained forecast panels
│   │   ├── governance/     # Model cards, lineage, BI export panels
│   │   ├── pipeline/       # Multi-file analysis, pipeline execution panels
│   │   └── sku/            # SKU mapping, override management panels
│   ├── src/hooks/          # React Query hooks + useAsyncOperation utility
│   └── src/lib/            # API client, auth, types, constants
├── tests/                  # 2047 unit + integration cases
├── configs/                # YAML configuration files
├── scripts/                # Entry points (run_backtest, run_forecast, serve, spark_*)
├── notebooks/              # Jupyter notebooks for exploration
├── data/rossmann/          # Rossmann dataset for demos
├── requirements.txt
└── setup.py
```

---

## Modules

### `src/forecasting/` — Model Library

All models implement a common interface (`BaseForecaster`):

```python
def fit(self, data: pl.DataFrame, config: ForecastConfig) -> "BaseForecaster"
def predict(self, horizon: int, ...) -> pl.DataFrame               # P50 point forecast
def predict_quantiles(self, horizon: int, quantiles: list[float]) -> pl.DataFrame
```

**Statistical & Naive:**

| Class | Description |
|-------|-------------|
| `SeasonalNaiveForecaster` | Last-year same-week seasonal naive baseline; `season_length=52` |
| `AutoARIMAForecaster` | Auto-ARIMA via statsforecast; `season_length=52` |
| `AutoETSForecaster` | ETS (Error/Trend/Season) via statsforecast; `season_length=52` |
| `AutoThetaForecaster` | AutoTheta via statsforecast; automatic trend/seasonality decomposition |
| `MSTLForecaster` | Multiple Seasonal-Trend decomposition via statsforecast; multi-period seasonality |

**ML (gradient boosting):**

| Class | Description |
|-------|-------------|
| `LGBMDirectForecaster` | LightGBM direct multi-step with lag/rolling features; supports external regressors |
| `XGBoostDirectForecaster` | XGBoost direct multi-step with lag/rolling features; supports external regressors |

**Neural (deep learning):**

| Class | Description |
|-------|-------------|
| `NBEATSForecaster` | N-BEATS neural forecaster via neuralforecast; basis expansion architecture |
| `NHITSForecaster` | N-HiTS neural forecaster via neuralforecast; hierarchical interpolation |
| `TFTForecaster` | Temporal Fusion Transformer via neuralforecast; attention-based with interpretable components |

**Foundation models (zero-shot):**

| Class | Description |
|-------|-------------|
| `ChronosForecaster` | Zero-shot foundation model (Amazon Chronos); no fine-tuning needed |
| `TimeGPTForecaster` | Zero-shot foundation model (Nixtla TimeGPT) via REST API |

**Intermittent demand:**

| Class | Description |
|-------|-------------|
| `CrostonForecaster` | Croston's method for intermittent demand |
| `CrostonSBAForecaster` | Croston-SBA (bias-corrected variant) |
| `TSBForecaster` | Teunter-Syntetos-Babai method for lumpy demand |

**Ensemble, hierarchical & constrained:**

| Class | Description |
|-------|-------------|
| `WeightedEnsembleForecaster` | Weighted mixture of any base models; bootstrapped P10/P50/P90 |
| `HierarchicalForecaster` | Forecaster that produces coherent hierarchical forecasts with built-in reconciliation |
| `ConstrainedDemandEstimator` | Wraps any base forecaster; enforces non-negativity, capacity limits, aggregate budgets; preserves quantile monotonicity |
| `ForecasterRegistry` | Register, retrieve, and instantiate models by name |

### `src/data/` — Data Layer

#### External Regressors

| Function | Description |
|----------|-------------|
| `load_external_features(path)` | Load external feature data from Parquet or CSV |
| `generate_holiday_calendar(country, start, end)` | Generate period-level holiday flags using the `holidays` library (optional dep) |
| `validate_regressors(features, actuals, columns)` | Validate grain alignment, null checks, future coverage for forecast horizon |

ML models (`LGBMDirectForecaster`, `XGBoostDirectForecaster`) automatically detect and use external feature columns during `fit()` and `predict()`. Statistical and naive models silently ignore them.

#### DataValidator (schema enforcement)

| Class/Dataclass | Description |
|-----------------|-------------|
| `DataValidator` | Validates input DataFrames: schema checks, duplicate detection, frequency validation, value range enforcement, completeness checks |
| `ValidationReport` | Result with `passed` flag, `issues` list, counts for duplicates/negatives/frequency violations |
| `ValidationIssue` | Single issue with `level` (error/warning), `check` name, `message`, optional `series_id` |

Runs as the first step in `SeriesBuilder.build()` when `validation.enabled = True`. Raises `ValueError` on errors in strict mode.

#### DemandCleanser (outlier & stockout correction)

| Class/Dataclass | Description |
|-----------------|-------------|
| `DemandCleanser` | Detects outliers (IQR/z-score), identifies stockout periods, applies corrections per-series |
| `CleansingResult` | Contains cleaned DataFrame + `CleansingReport` summary |
| `CleansingReport` | Counts: outliers, stockout periods/weeks, rows modified, per-series breakdown |

Runs after gap-filling in `SeriesBuilder.build()` when `cleansing.enabled = True`. All statistics computed per-series (not globally). Supports period exclusion (e.g., COVID) with interpolate/drop/flag actions.

#### RegressorScreen (pre-training feature quality)

| Class/Dataclass | Description |
|-----------------|-------------|
| `screen_regressors()` | Screens external features before model training: zero/near-zero variance, high pairwise correlation, optional mutual information with target |
| `RegressorScreenReport` | Result with `screened_columns`, `dropped_columns`, `low_variance_columns`, `high_correlation_pairs`, `low_mi_columns`, per-column stats |

Runs after feature join in `SeriesBuilder.build()` when `external_regressors.screen.enabled = True`. Optionally auto-drops flagged columns (`auto_drop=True`).

#### FileClassifier (multi-file upload role detection)

| Class/Dataclass | Description |
|-----------------|-------------|
| `FileClassifier` | Classifies uploaded DataFrames into roles: `time_series`, `dimension`, `regressor`, `unknown`. Two-pass: isolation scoring then cross-file resolution |
| `FileProfile` | Per-file profile with `role`, `confidence`, `time_column`, `id_columns`, `numeric_columns`, `reasoning` |
| `ClassificationResult` | All files bucketed into `primary_file`, `dimension_files`, `regressor_files`, `unknown_files` with warnings |

#### MultiFileMerger (join key detection and merge)

| Class/Dataclass | Description |
|-----------------|-------------|
| `MultiFileMerger` | Detects join keys between classified files, generates merge preview with stats, executes left-join merge |
| `JoinSpec` | Join specification: keys, overlap percentage, warnings |
| `MergePreview` | Pre-merge stats: sample rows, matched/unmatched counts, column conflicts, null-fill columns |
| `MergeResult` | Final merged DataFrame with preview metadata and join specs |

Used by the Data Onboarding page to accept N CSV files, classify roles, preview the merge, and feed a single combined DataFrame to `DataAnalyzer`.
#### DataQualityAnalyzer (data profiling)

| Class | Description |
|-------|-------------|
| `DataQualityAnalyzer` | Profiles input data: missing percentage, zero-value series count, short series count, gap detection; complements `DataValidator` with aggregate-level quality metrics |

### `src/auth/` — RBAC & Authentication

| Class/Function | Description |
|----------------|-------------|
| `Role` (enum) | `ADMIN`, `DATA_SCIENTIST`, `PLANNER`, `MANAGER`, `VIEWER` |
| `Permission` (enum) | 11 permissions: `VIEW_FORECASTS`, `VIEW_METRICS`, `VIEW_AUDIT_LOG`, `CREATE_OVERRIDE`, `DELETE_OVERRIDE`, `APPROVE_OVERRIDE`, `RUN_BACKTEST`, `RUN_PIPELINE`, `PROMOTE_MODEL`, `MODIFY_CONFIG`, `MANAGE_USERS` |
| `ROLE_PERMISSIONS` | Complete role -> permission mapping |
| `AuthSettings` | Shared strict production/demo settings; missing/weak signing configuration fails startup |
| `User` | Dataclass with `user_id`, `email`, `role`, `is_active`; `has_permission()` method |
| `get_current_user()` | FastAPI dependency — extracts/validates JWT from `Authorization: Bearer` header |
| `require_permission(perm)` | FastAPI dependency factory for fine-grained permission checks |
| `require_role(*roles)` | FastAPI dependency factory for role-based checks |
| `create_token(user_id, email, role)` | Create signed JWT (HS256, 24h default expiry) |
| `decode_token(token, secret)` | Decode and validate JWT; returns `None` on failure |
| `write_private_token(path, token)` | Exclusive new-file credential output: POSIX 0600 or verified protected Windows DACL |

Production is the default across the factory, launcher and image. A trusted
operator uses `scripts/issue_token.py` to assign a role and issue a private-file
token (one-hour default, 1-24 hours). The browser verifies `/auth/me` before
sign-in and keeps credentials only in per-tab memory. There is no HTTP issuer,
password login, SSO or user registry. Explicit demo requires
`AUTH_ENABLED=false`, `API_DEV_MODE=true`, and no signing key; role selection is
only a simulation. See [provisioning, expiry and rotation](docs/DEPLOYMENT.md#authentication-setup).

**Permission matrix:**

| Action | ADMIN | DATA_SCIENTIST | PLANNER | MANAGER | VIEWER |
|--------|-------|----------------|---------|---------|--------|
| View forecasts/metrics | Y | Y | Y | Y | Y |
| Create overrides | Y | Y | Y | N | N |
| Approve overrides | Y | N | N | Y | N |
| Run backtest/pipeline | Y | Y | N | N | N |
| Promote champion model | Y | Y | N | N | N |
| View audit log | Y | Y | N | Y | N |
| Manage users | Y | N | N | N | N |

### `src/audit/` — Audit Trail

| Class | Description |
|-------|-------------|
| `AuditEvent` | Immutable dataclass: `action`, `resource_type`, `resource_id`, `user_id`, `user_role`, `status`, `old_value`, `new_value`, `ip_address`, `request_id`, auto-generated `timestamp` (UTC) and `audit_id` |
| `AuditLogger` | Append-only Parquet-backed logger; date-partitioned (`audit_log/date=YYYY-MM-DD/`) |

```python
logger = AuditLogger("data/audit_log/")
logger.log(event)                       # single event
logger.log_batch(events)                # batch write
logger.query(user_id=..., action=..., start_date=..., limit=100)  # filtered reads
logger.count_by_action()                # aggregation by action + status
```

No UPDATE or DELETE operations — append-only by design for SOX compliance.

### `src/series/` — Series Management

| Class | Description |
|-------|-------------|
| `SeriesBuilder` | Builds frequency-aware panel DataFrames from raw transactional data; fills gaps; integrates validation and cleansing |
| `SparseDetector` | Classifies series as smooth / intermittent / erratic / lumpy using CV-squared and ADI |
| `TransitionEngine` | Handles new-product launches: stitches history, applies linear/S-curve/step ramps |
| `StructuralBreakDetector` | Identifies permanent level shifts via CUSUM (zero-dependency) or PELT (`ruptures`); truncates history to post-break regime |

**Sparse classification thresholds:**

| Class | CV-squared | ADI | Recommended model |
|-------|------------|-----|-------------------|
| Smooth | <= 0.49 | <= 1.32 | Statistical / ML |
| Intermittent | <= 0.49 | > 1.32 | Croston / CrostonSBA |
| Erratic | > 0.49 | <= 1.32 | TSB |
| Lumpy | > 0.49 | > 1.32 | TSB / Ensemble |

**Transition scenarios:**

| Scenario | Condition | Action |
|----------|-----------|--------|
| A — Already launched | `launch_date <= forecast_origin` | Stitch old SKU history onto new SKU |
| B — In horizon | `0 < gap <= transition_window` | Ramp-down old, ramp-up new (linear/scurve/step) |
| C — Beyond horizon | `gap > transition_window` | Forecast old only; new SKU flagged "pending" |

### `src/hierarchy/` — Hierarchical Reconciliation

| Class | Description |
|-------|-------------|
| `HierarchyTree` | Build and traverse a multi-level node hierarchy; exposes `summing_matrix()` |
| `HierarchyNode` | Single node with `descendants()`, `ancestors()`, `leaf_descendants()`, `is_leaf()` |
| `HierarchyAggregator` | `aggregate_to()`, `disaggregate_to()`, `compute_historical_proportions()` |
| `Reconciler` | Six reconciliation methods; non-negativity enforced |

**Reconciliation methods:**

| Method | W matrix | Notes |
|--------|----------|-------|
| `bottom_up` | — | Leaf forecasts are authoritative; upper levels are sums |
| `top_down` | — | Top level disaggregated by historical proportions |
| `middle_out` | — | Mid-level is authoritative; disaggregated down, aggregated up |
| `ols` | Identity | Equal uncertainty at all levels |
| `wls` | `diag(n_leaf_descendants)` or per-series residual variance | Structural or data-driven weights |
| `mint` | Ledoit-Wolf diagonal shrinkage covariance | Falls back to WLS-structural when `T < n` |

**Linear reconciliation formula (OLS/WLS/MinT):**
```
G       = (S'W-1 S)-1 S'W-1       # projection matrix
P_leaf  = G * P_all                # reconciled leaf forecasts
P_all   = S * P_leaf               # all-level coherent forecasts
```
Tikhonov regularisation (lambda = 1e-6) applied for numerical stability.

### `src/backtesting/` — Backtest Engine

| Class | Description |
|-------|-------------|
| `BacktestEngine` | Validates complete series/date/lead coverage and excludes incomplete models; pipeline persistence is deferred until validation and atomic bundle commit |
| `WalkForwardCV` | Expanding-window fold generator; configurable `n_folds`, `horizon_weeks`, `step_weeks` |
| `CVFold` | Dataclass for one fold: train indices, test indices, fold number |
| `ChampionSelector` | Recomputes volume WMAPE (or an explicit alternative) over comparable raw rows and horizon buckets; `compute_ensemble_weights()` |

### `src/metrics/` — Metric Store & Drift

| Class/Function | Description |
|----------------|-------------|
| `MetricStore` | Unique atomic appends, committed/legacy raw reads, and immutable stored leaderboard retrieval; explicit current scoring from records |
| `ForecastDriftDetector` | Detects accuracy drift, bias drift, and volume anomalies per series |
| `DriftAlert` | `series_id`, `metric`, `severity` (warning/critical), `current_value`, `baseline_value` |
| `DriftConfig` | `baseline_weeks`, `recent_weeks` configuration |
| `wmape()` / `normalized_bias()` / `mape()` / `mae()` / `rmse()` | Standalone metric functions |
| `compute_all_metrics()` | Compute all metrics in one pass |

### `src/analytics/` — Data Profiling, Explainability, Governance & BI

#### `DataAnalyzer` (automated data profiling)

| Class | Description |
|-------|-------------|
| `DataAnalyzer` | Orchestrates automated data profiling: schema detection, hierarchy detection, forecastability assessment, data quality profiling, and `PlatformConfig` recommendation |
| `ForecastabilityAnalyzer` | Per-series statistical signals: CV, approximate entropy (ApEn), spectral entropy, SNR, trend/seasonal strength; produces a forecastability score per series |
| `CausalAnalyzer` | Econometric analysis: price elasticity estimation (detrended/deseasonalized), cannibalization detection (cross-correlation), promotional lift (before/after comparison) |
| `LLMAnalyzer` | Sends analysis reports to Claude for plain-language interpretation: key findings, hypotheses, model selection rationale, risk identification; gracefully degrades when `ANTHROPIC_API_KEY` is unset |

#### `ForecastAnalytics` (notebook API)
Notebook-ready queries over the MetricStore:
- `model_leaderboard(lob, run_id=...)` — retrieve stored rank, values and competition scope without rescoring; use `latest=True` within one LOB
- `model_comparison_by_fold(lob)` — per-fold performance breakdown
- `accuracy_over_time(model, channel)` — trend of accuracy metrics
- `accuracy_by_grain(lob)` — performance by hierarchy level
- `bias_distribution(lob)` — normalized bias distribution
- `transition_impact(lob)` — before/after metrics for SKU transitions
- `backtest_vs_live(lob)` — backtest vs live accuracy comparison

#### `BIExporter`
Writes Parquet in Hive-partitioned layout for direct Power BI consumption:
- `export_forecast_vs_actual(forecasts, actuals, lob)` -> `bi_exports/forecast_vs_actual/lob=.../`
- `export_leaderboard(lob)` -> `bi_exports/model_leaderboard/lob=.../`
- `export_bias_report(lob)` -> `bi_exports/bias_report/lob=.../`
- `export_fva(fva_detail, fva_summary, lob)` -> `bi_exports/fva_detail/lob=.../` and `fva_summary/lob=.../`

#### `ForecastComparator`
Aligns model forecast with external sources and prior cycle:
- `compare(model_forecast, external_forecasts, prior_model_forecast)` — adds `{name}_gap`, `{name}_gap_pct`, `uncertainty_ratio`, `cycle_change`, `cycle_change_pct`
- `summary(comparison)` — one row per series_id

#### `ExceptionEngine`
Business-rule exception flags for S&OP review queues:

| Flag | Default threshold |
|------|------------------|
| `exc_large_cycle_change` | \|cycle_change_pct\| > 20% |
| `exc_high_uncertainty` | uncertainty_ratio > 0.50 |
| `exc_field_disagree` | \|gap_pct\| > 25% vs any external source |
| `exc_overforecast` | gap_pct > 30% |
| `exc_underforecast` | gap_pct < -30% |
| `exc_no_prior` | prior_model_forecast is null |

All thresholds configurable at construction. `exception_summary()` groups by series with flagged-week counts, sorted by `total_exception_weeks` descending.

#### `ForecastExplainer`
- **`decompose(history, forecast)`** — STL-style: trend (centered MA, window=`season_length`) + seasonal (mean de-trended per position) + residual; trend extrapolated linearly over `trend_window` periods into forecast horizon
- **`explain_ml(model, features_df, top_k=5)`** — SHAP attribution for LightGBM/XGBoost; lazy import, graceful empty-DataFrame fallback if `shap` not installed
- **`narrative(decomposition, comparison)`** — templated natural-language string per series (YoY direction, primary driver, gap vs external, uncertainty label)

#### Model Governance (`governance.py`)

| Class | Description |
|-------|-------------|
| `DriftDetector` | Compares live WMAPE to backtest WMAPE; returns `ok` / `warning` / `alert` / `insufficient_data` |
| `ModelCard` | Structured metadata dataclass: training window, series count, backtest metrics, features, config hash |
| `ModelCardRegistry` | In-memory + Parquet-backed registry; `register()`, `get()`, `all_cards()` |
| `ForecastLineage` | Append-only audit log of which model produced each forecast run; `record()`, `history()`, `latest()` |

#### FVA Analysis (`metrics/fva.py` + `analytics/fva_analyzer.py`)

Measures how much accuracy each forecast layer contributes — from naive baseline through statistical, ML, and planner overrides.

**Layers:**

| Layer | Models | Role |
|-------|--------|------|
| L1: Naive | `seasonal_naive` | Baseline (always computed) |
| L2: Statistical | `auto_arima`, `auto_ets`, `croston`, `croston_sba`, `tsb` | Best statistical per series |
| L3: ML | `lgbm_direct`, `xgboost_direct` | Best ML per series |
| L4: Override | Planner-adjusted | If override exists, else L4 = L3 |

**Key functions (`src/metrics/fva.py`):**

| Function | Description |
|----------|-------------|
| `classify_fva(value)` | `ADDS_VALUE` (>2pp), `NEUTRAL`, or `DESTROYS_VALUE` (<-2pp) |
| `compute_layer_metrics(actual, forecast)` | WMAPE, bias, MAE for one layer |
| `compute_fva_between_layers(actual, parent, child)` | Incremental FVA with classification |
| `compute_fva_cascade(actual, forecasts)` | Full cascade metrics across all layers |
| `compute_total_fva(actual, forecasts)` | Total WMAPE reduction baseline -> final |

**FVA Analyzer (`src/analytics/fva_analyzer.py`):**

| Method | Description |
|--------|-------------|
| `compute_fva_detail(backtest_results)` | Per-series, per-fold FVA from backtest results |
| `summarize(fva_detail, group_by)` | Aggregate by layer: mean WMAPE, FVA, % adds/neutral/destroys |
| `layer_leaderboard(fva_detail)` | Rank layers by contribution; recommends Keep/Review/Remove |

### `src/observability/` — Pipeline Observability

| Class | Description |
|-------|-------------|
| `PipelineContext` | Correlation ID threading for pipeline runs; `run_id`, `lob`, `parent_run_id`; `child()` for sub-pipelines |
| `StructuredLogger` | JSON-formatted logging enriched with `run_id`/`lob` from context; wraps stdlib `logging` |
| `MetricsEmitter` | Emit timing, counters, gauges; backends: `"log"` (JSON), `"statsd"` (UDP); `timer()` context manager |
| `AlertDispatcher` | Routes `DriftAlert` objects to log and/or webhook channels; severity filtering |
| `CostEstimator` | Track per-model compute time; estimate cloud costs at configurable $/second rate |
| `setup_logging()` | Configure root logger for `"text"` or `"json"` format |

### `src/pipeline/` — Distributed Execution & Scheduling

| Class | Description |
|-------|-------------|
| `BatchInferenceRunner` | Partition series → parallel fit/predict → merge; isolated spawn workers preserve configured models, and failed batches raise rather than return partial forecasts |
| `PipelineScheduler` | Recurring pipeline execution with retry (exponential backoff) and dead-letter JSON for failed runs |

### `src/overrides/` — Planner Overrides

| Class | Description |
|-------|-------------|
| `OverrideStore` | DuckDB-backed store for planner **transition overrides** (old_sku -> new_sku proportion, scenario, ramp shape); `add_override()`, `get_overrides()`, `delete_override()` |
| `ParquetOverrideStore` | Parquet-backed fallback for environments where DuckDB is unavailable (e.g. Microsoft Fabric) |
| `get_override_store()` | Factory function — auto-selects DuckDB or Parquet backend based on available imports |

### `src/sku_mapping/` — New / Discontinued SKU Mapping

Maps new SKUs to analogues and splits multi-mapped forecasts proportionally.

**Mapping methods:**

| Class | Similarity | Input |
|-------|-----------|-------|
| `AttributeMatchingMethod` | Cosine similarity | Product attribute vectors |
| `NamingConventionMethod` | Token overlap (Jaccard) + base-name parsing | SKU names/descriptions |
| `CurveFittingMethod` | S-curve / step-ramp shape fit (R-squared) | Sales trajectory |
| `TemporalCovementMethod` | Pearson correlation of growth rates | Historical weekly sales |

**Fusion:**

| Class | Description |
|-------|-------------|
| `CandidateFusion` | Combines method scores with configurable weights; confidence classification |
| `BayesianProportionEstimator` | Dirichlet posterior for 1-to-many splits; `concentration=0.5` |

### `src/api/` — REST API

Built with FastAPI. Auto-generated Swagger docs at `/docs`. All business endpoints
require verified JWTs by default; only an explicit unauthenticated demo opts out.

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/health` | GET | Bounded readiness status, version and boolean checks | No |
| `/auth/status` | GET | Public mode and authentication-required flag only | No |
| `/auth/me` | GET | Verified identity, role, permissions and expiry; never a token | Yes |
| `/forecast/{lob}` | GET | Latest forecasts for a LOB (optional `series_id`, `horizon` query params) | Yes |
| `/forecast/{lob}/{series_id}` | GET | Latest forecast for a single series | Yes |
| `/metrics/leaderboard/{lob}` | GET | Stored exact/latest ranking with criteria, competition scope and scoring provenance | Yes |
| `/metrics/drift/{lob}` | GET | Drift alerts (`baseline_weeks`, `recent_weeks` params) | Yes |
| `/audit` | GET | Query audit log (`action`, `resource_type`, `limit` params) | Yes (`VIEW_AUDIT_LOG`) |
| `/series/{lob}` | GET | List series with SBC classification (ADI, CV², demand class) | Yes |
| `/series/breaks` | POST | Structural break detection (CUSUM/PELT) | Yes |
| `/series/cleansing-audit` | POST | Demand cleansing before/after audit | Yes |
| `/series/regressor-screen` | POST | Regressor screening (variance, correlation, MI) | Yes |
| `/hierarchy/build` | POST | Build hierarchy tree, return structure stats | Yes |
| `/hierarchy/aggregate` | POST | Aggregate data to target hierarchy level | Yes |
| `/hierarchy/reconcile` | POST | Run hierarchical reconciliation (OLS/WLS/MinT) | Yes (`RUN_PIPELINE`) |
| `/sku-mapping/phase1` | POST | Phase 1 SKU mapping (attribute + naming) | Yes (`RUN_PIPELINE`) |
| `/sku-mapping/phase2` | POST | Phase 2 SKU mapping (+ curve fitting) | Yes (`RUN_PIPELINE`) |
| `/overrides` | GET/POST | List / create planner overrides | Yes |
| `/overrides/{id}` | PUT/DELETE | Update / delete override | Yes |
| `/pipeline/backtest` | POST | Run backtest pipeline | Yes (`RUN_BACKTEST`) |
| `/pipeline/forecast` | POST | Run forecast pipeline | Yes (`RUN_PIPELINE`) |
| `/pipeline/manifests` | GET | List recent pipeline run manifests | Yes |
| `/pipeline/costs` | GET | Cost tracking from manifests | Yes |
| `/pipeline/analyze-multi-file` | POST | Multi-file classification and merge | Yes (`RUN_PIPELINE`) |
| `/metrics/{lob}/fva` | GET | FVA cascade analysis | Yes |
| `/metrics/{lob}/calibration` | GET | Prediction interval calibration | Yes |
| `/metrics/{lob}/shap` | POST | SHAP feature attribution | Yes |
| `/forecast/decompose` | POST | STL seasonal decomposition | Yes |
| `/forecast/compare` | POST | Cross-forecast comparison | Yes |
| `/forecast/constrain` | POST | Apply capacity/budget constraints | Yes (`RUN_PIPELINE`) |
| `/governance/model-cards` | GET | List model cards | Yes |
| `/governance/lineage` | GET | Forecast lineage history | Yes |
| `/governance/export/{type}` | POST | BI export (forecast-actual, leaderboard, bias) | Yes |

### `src/fabric/` — Microsoft Fabric / Delta Lake

| Class | Description |
|-------|-------------|
| `DeploymentOrchestrator` | End-to-end pipeline: preflight checks -> champion selection -> forecast -> write -> audit log |
| `FabricLakehouse` | `read_table()`, `write_table()`, `table_exists()`, `vacuum()`, `optimize()`, `history()` |
| `DeltaWriter` | `upsert()`, `overwrite_partition()`, `append()`, `write_forecasts()` |
| `FabricConfig` | Workspace/lakehouse/capacity config; `from_env()`, `from_dict()`, `abfss_base` property |

### `src/spark/` — Distributed Execution

PySpark wrappers for large-scale runs on Fabric/Databricks:

| Class/Function | Description |
|----------------|-------------|
| `SparkForecastPipeline` | `run_forecast()`, `run_backtest()`, `select_champion()` |
| `SparkSeriesBuilder` | Distributed panel construction from raw Spark DataFrames |
| `SparkFeatureEngineer` | `create_temporal_features()`, `create_lag_features()`, `create_rolling_features()` |
| `SparkDataLoader` | Reads actuals, product master, Delta tables, Rossmann data |
| `get_or_create_spark()` | Fabric-aware session factory (detects `notebookutils`) |
| `polars_to_spark()` / `spark_to_polars()` | Schema-preserving conversion utilities |
| `repartition_by_series()` | Optimise parallelism by series count |
| `abfss_uri()` | Build `abfss://` URIs for Fabric/ADLS |

### `src/pipeline/` — End-to-End Pipelines

| Class | Description |
|-------|-------------|
| `BacktestPipeline` | Current-run evaluation/selection/validation, then one immutable input/config/metric bundle |
| `ForecastPipeline` | Prepared series, selected model/constraints, validation, then one immutable forecast bundle |
| `RunStore` | Exclusive UUID directories, no-replace manifest commit marker, checksums, exact/latest reads and explicit uncertain outcomes |
| `RunReader` | Shared committed resolver and read-only legacy forecast/history/sidecar compatibility |
| `PipelineManifest` | Versioned provenance: raw/prepared inputs, safe canonical settings, models/source, validation and artifact checksums |
| `build_manifest()` | Collects provenance from SeriesBuilder reports into a single manifest |
| `write_manifest()` | Atomic no-overwrite legacy sidecar helper; not the versioned run publisher |
| `read_manifest()` | Loads manifest from JSON file |

Use an explicit `output.runs_path` / server `API_RUNS_DIR` to share one local
authority across API, CLI and notebook readers. New successful runs never
overwrite a date/second-named predecessor. Exact run ID selection survives
service restart; default latest follows attempt-start UTC with ID tie-break.
The Forecast Viewer lists prior runs, shows their exact values and stored
configuration, and labels legacy provenance unknown. Read the
[storage and recovery contract](docs/DEPLOYMENT.md#immutable-local-runs-and-recovery):
local hard-link visibility is not a distributed/OneLake or power-loss ACID claim.

### `src/config/` — Configuration Schema & Loader

YAML-driven config system with dataclass schema validation. `FREQUENCY_PROFILES` is the single source of truth for all frequency-dependent behaviour.

| Class/Function | Description |
|----------------|-------------|
| `PlatformConfig` | Root dataclass for entire pipeline configuration |
| `ForecastConfig` | Forecast settings (horizon, models, quantiles, external regressors) |
| `BacktestConfig` | Walk-forward validation settings (folds, step size) |
| `DataQualityConfig` | Validation + cleansing settings |
| `parse_config_yaml(content)` | Parse uploaded YAML with strict known-field types and existing period aliases; invalid documents fail rather than falling back |
| `load_config(path)` | Load and parse YAML config into `PlatformConfig` |
| `load_config_with_overrides(path, overrides)` | Load config with LOB-level overrides merged |
| `get_frequency_profile(freq)` | Get temporal properties for `"D"` / `"W"` / `"M"` / `"Q"` |
| `freq_timedelta(freq, periods)` | Fixed D/W durations only; M/Q require an anchored calendar |
| `utils.calendar.PeriodCalendar` | Exact D/W/M/Q grids, preserving weekday and calendar start/end labels |

See [Numerical reliability and score migration](docs/NUMERICAL_RELIABILITY.md)
for scoring definitions, explicit macro weighting, missing-prediction failures,
gap semantics, fold-local preprocessing restrictions, and historical rankings.

Pipeline upload precedence is explicit model/horizon override, then valid YAML,
then platform defaults. The UI carries the current backtest winner into forecasting
and displays the API's typed, allowlisted `execution` settings. Invalid uploads
return HTTP 400 before computation/publication; invalid query horizons return 422.
See [the selected-plan workflow](docs/USER_GUIDE.md#running-the-selected-forecast-plan)
and [execution response contract](docs/DATA_FORMAT.md#output-pipeline-execution-response).

### `src/evaluation/` — Metric Computation

Forecast accuracy metrics used by backtest engine and analytics.

| Function | Description |
|----------|-------------|
| `rmse(y_true, y_pred)` | Root Mean Squared Error |
| `rmspe(y_true, y_pred)` | Root Mean Squared Percentage Error |
| `mae(y_true, y_pred)` | Mean Absolute Error |
| `mape(y_true, y_pred)` | Mean Absolute Percentage Error |
| `ModelEvaluator` | Framework for evaluating forecasts against actuals |

### `src/errors/` — User-Friendly Error Translation

Converts cryptic Python/Polars exceptions into plain-English messages with actionable suggestions.

| Function | Description |
|----------|-------------|
| `friendly_error(exc, context, run_id)` | Translate exception to user-friendly dict with message and suggestion |
| `safe_execute(func, *args, fallback, context)` | Execute callable with automatic error translation and fallback |
| `suggest_column(target, available)` | Suggest closest matching column names for typo fixes |

### `src/health/` — Pre-Pipeline Health Checks

Validates system state, module imports, dependencies, config integrity, and directory availability before pipeline execution.

| Function | Description |
|----------|-------------|
| `run_health_check(config_path, data_dir)` | Run all health checks and return combined report |
| `check_dependencies()` | Verify required/optional Python packages are installed |
| `check_module_imports()` | Validate all core/optional modules can be imported |
| `check_config(config_path)` | Validate pipeline config file exists and parses |
| `check_data_directory(data_dir)` | Verify data directory structure |

### `src/lineage/` — Data Provenance Tracking

Records data flow through pipeline steps with parent linkage for end-to-end traceability.

| Class/Function | Description |
|----------------|-------------|
| `LineageTracker` | Track data lineage through pipeline steps |
| `track(step, agent, inputs, outputs)` | Record a pipeline step via singleton tracker |
| `LineageTracker.get_lineage_for_output(path)` | Walk parent chain back to root inputs |

### `src/models/` — Legacy Model Wrappers

Thin wrappers around XGBoost and LightGBM providing a unified interface. Superseded by `src/forecasting/ml.py` for new development.

| Class | Description |
|-------|-------------|
| `BaseForecaster` | Abstract base class for model wrappers |
| `XGBoostForecaster` | XGBoost forecast model wrapper |
| `LightGBMForecaster` | LightGBM forecast model wrapper |

### `src/profiler/` — Deep Data Profiling

Distribution analysis, temporal pattern detection, correlation scanning, completeness checks, and rolling anomaly scoring.

| Function | Description |
|----------|-------------|
| `run_deep_profile(df, date_col, metric_cols, freq)` | Run all profiling checks in one call |
| `profile_distributions(df, cols)` | Profile shape, skewness, kurtosis per column |
| `profile_temporal_patterns(df, date_col, cols, freq)` | Detect gaps, day-of-week/monthly patterns, trend |
| `profile_correlations(df, cols, threshold)` | Find strong pairwise correlations |
| `profile_anomalies(df, date_col, cols, window)` | Scan for anomalies using rolling statistics |

### `src/stats/` — Statistical Testing Utilities

Hypothesis testing, power analysis, and forecast accuracy comparison with structured results and human-readable interpretations.

| Function | Description |
|----------|-------------|
| `two_sample_mean_test(a, b, alpha)` | Welch's t-test comparing means |
| `mann_whitney_test(a, b, alpha)` | Non-parametric U test for skewed data |
| `bootstrap_ci(series, stat_func, n, confidence)` | Non-parametric bootstrap confidence interval |
| `forecast_accuracy_test(errors_a, errors_b)` | Paired Diebold-Mariano test for model comparison |
| `rank_dimensions(df, metric_col, dims)` | Rank categorical dimensions by explanatory power |

### `src/tieout/` — Data Integrity Verification

Dual-path verification comparing Polars direct reads vs DuckDB SQL reads. Checks structural metrics and aggregations with tolerance-based gating.

| Function | Description |
|----------|-------------|
| `run_full_tieout(source_path, db_con, label)` | Run complete tie-out comparing dual read paths |
| `compare_profiles(source, db)` | Compare two profiles with tier-based gating |
| `overall_status(results)` | Determine PASS / WARN / FAIL gate decision |

### `src/validation/` — Four-Layer Validation Framework

Structural, logical, business-rule, and Simpson's paradox checks producing a confidence grade (A–F).

| Function | Description |
|----------|-------------|
| `run_full_validation(df, config)` | Run all 4 layers + confidence scoring |
| `run_structural_checks(df, config)` | Validate schema, keys, completeness, date range |
| `run_logical_checks(df, config)` | Check aggregation consistency, trends, monotonicity |
| `validate_business_rules(df, config)` | Check ranges, metric relationships, temporal spikes |
| `check_simpsons_paradox(df, segment_col, value_col)` | Detect Simpson's paradox across segments |
| `score_confidence(structural, logical, business, paradox)` | Compute A–F confidence grade |

### `src/visualization/` — Chart Builders

SWD-style (Storytelling with Data) chart templates using matplotlib/Plotly for publication-quality visualizations.

| Function | Description |
|----------|-------------|
| `forecast_plot(ax, actuals, forecast, ci_lower, ci_upper)` | Time series with forecast and confidence intervals |
| `control_chart_plot(ax, values, dates, limits)` | Control chart with control limits |
| `leaderboard_chart(ax, models, metrics)` | Rank models by performance |
| `fva_cascade_chart(ax, components, values)` | Waterfall chart for FVA analysis |
| `drift_timeline(ax, dates, drift_scores)` | Timeline showing model drift |

### `src/presentation/` — Marp Slide Deck Builder

Generates presentation-ready markdown decks from analysis results for S&OP meetings and backtest reviews.

| Function | Description |
|----------|-------------|
| `DeckBuilder` | Build slide content from structured data |
| `title_slide(title, subtitle, date, author)` | Generate title slide |
| `kpi_slide(metrics, headline)` | Generate KPI dashboard slide |
| `chart_slide(chart_path, headline, findings)` | Generate chart presentation slide |
| `export_pdf(deck_path)` / `export_html(deck_path)` | Export to PDF or HTML |

### `src/utils/` — General Utilities

Logging setup and configuration I/O helpers used across the codebase.

| Function | Description |
|----------|-------------|
| `get_logger(name)` | Get configured logger instance |
| `load_config(path)` | Load configuration from file |

---

## Quick Start

```python
from src.forecasting import LGBMDirectForecaster, WeightedEnsembleForecaster, ForecasterRegistry
from src.series import SparseDetector, TransitionEngine
from src.hierarchy import HierarchyTree, HierarchyAggregator, Reconciler
from src.backtesting import BacktestEngine, WalkForwardCV, ChampionSelector
from src.metrics import MetricStore, ForecastDriftDetector
from src.metrics.fva import compute_fva_cascade, classify_fva
from src.analytics import (
    ForecastAnalytics, ForecastComparator, ExceptionEngine,
    ForecastExplainer, DriftDetector, ModelCard, ModelCardRegistry, ForecastLineage,
    BIExporter,
)
from src.analytics.fva_analyzer import FVAAnalyzer
from src.analytics.analyzer import DataAnalyzer
from src.analytics.forecastability import ForecastabilityAnalyzer
from src.analytics.causal import CausalAnalyzer
from src.analytics.llm_analyzer import LLMAnalyzer
from src.data.regressors import load_external_features, validate_regressors
from src.data.validator import DataValidator
from src.data.cleanser import DemandCleanser
from src.data.regressor_screen import screen_regressors
from src.forecasting.constrained import ConstrainedDemandEstimator
from src.auth.models import User, Role, Permission
from src.auth.token import create_token, decode_token
from src.audit.logger import AuditLogger
from src.overrides import OverrideStore
from src.pipeline import BacktestPipeline, ForecastPipeline
from src.pipeline.manifest import PipelineManifest, read_manifest

# 1. Load and validate external features (promotions, holidays, price)
ext_features = load_external_features("data/external_features.parquet")
warnings = validate_regressors(ext_features, panel_df, config.forecast.external_regressors.feature_columns)

# 2. Classify series and route to appropriate model
detector = SparseDetector()
classification = detector.classify(panel_df)
smooth, intermittent = detector.split(panel_df)

# 3. Backtest all candidate models (with external features)
engine = BacktestEngine(n_folds=4, horizon_weeks=13)
results = engine.run(models=[LGBMDirectForecaster(), SeasonalNaiveForecaster()], data=smooth)

# 4. Select champion
champion = ChampionSelector(config).select(results)

# 5. FVA Analysis — measure value added by each forecast layer
fva_analyzer = FVAAnalyzer()
fva_detail = fva_analyzer.compute_fva_detail(results)
fva_summary = fva_analyzer.summarize(fva_detail)
fva_leaderboard = fva_analyzer.layer_leaderboard(fva_detail)

# 6. Reconcile across hierarchy
tree = HierarchyTree(config.get_hierarchy("product"), hierarchy_df)
reconciler = Reconciler(tree)
reconciled = reconciler.reconcile(forecasts_df, method="mint", residuals=residuals_df)

# 7. Apply planner overrides
store = OverrideStore()
overrides = store.get_overrides(lob="retail", week=current_week)

# 8. Compare vs external forecasts and flag exceptions
comparison = ForecastComparator().compare(
    reconciled,
    external_forecasts={"field": field_df, "financial": finance_df},
    prior_model_forecast=last_cycle_df,
)
flagged = ExceptionEngine().flag(comparison)
actionable = flagged.filter(pl.col("has_exception"))

# 9. Explain
explainer = ForecastExplainer(season_length=52, trend_window=12)
decomp = explainer.decompose(history_df, reconciled)
narratives = explainer.narrative(decomp, comparison)

# 10. Governance
registry = ModelCardRegistry()
registry.register(ModelCard.from_backtest("lgbm_direct", "retail", results))
ForecastLineage().record(lob="retail", model_id="lgbm_direct", n_series=500, horizon_weeks=13)

# 11. Export to Power BI (including FVA tables)
exporter = BIExporter()
exporter.export_forecast_vs_actual(reconciled, actuals_df, lob="retail")
exporter.export_fva(fva_detail, fva_summary, lob="retail")

# 12. Auth — trusted operator utility (signing key injected from protected configuration)
import os
token = create_token(user_id="analyst_1", email="a@co.com", role="data_scientist", secret_key=os.environ["JWT_SECRET_KEY"])
# Do not print or persist the token insecurely; prefer scripts/issue_token.py for operator use.
user = User(user_id="analyst_1", email="a@co.com", role=Role.DATA_SCIENTIST)
assert user.has_permission(Permission.RUN_BACKTEST)

# 13. Audit — log an action
audit = AuditLogger("data/audit_log/")
from src.audit.schemas import AuditEvent
audit.log(AuditEvent(action="promote_model", resource_type="model_card",
                     resource_id="lgbm_direct", user_id="analyst_1",
                     user_role="data_scientist", user_email="a@co.com", status="SUCCESS"))
```

---

## Configuration

```yaml
lob: retail

forecast:
  frequency: W               # D | W | M | Q (drives season_length, lags, etc.)
  horizon_weeks: 39          # periods; use horizon_periods for clarity
  forecasters: [lgbm_direct, auto_ets, seasonal_naive]
  quantiles: [0.1, 0.5, 0.9]
  sparse_detection: true
  intermittent_forecasters: [croston_sba, tsb]
  external_regressors:
    enabled: true
    feature_columns:
      - promotion_flag
      - holiday_flag
      - price_index
    future_features_path: data/future_features.parquet
    screen:                    # RegressorScreen
      enabled: false
      variance_threshold: 1.0e-6
      correlation_threshold: 0.95
      mi_enabled: false
      mi_threshold: 0.01
      auto_drop: true
  constraints:                # ConstrainedDemandEstimator
    enabled: false
    min_demand: 0.0
    max_capacity: null
    aggregate_max: null

backtest:
  n_folds: 3
  val_weeks: 13
  gap_weeks: 0
  champion_granularity: lob  # lob | product_group | series
  selection_strategy: champion   # champion | weighted_ensemble

hierarchies:
  - name: product
    levels: [total, category, sku]
    reconciliation:
      method: mint    # bottom_up | top_down | middle_out | ols | wls | mint

transition:
  transition_window_weeks: 13
  ramp_shape: linear   # linear | scurve | step
  enable_overrides: true

data_quality:
  validation:              # DataValidator
    enabled: false
    check_duplicates: true
    check_frequency: true
    check_non_negative: true
    max_missing_pct: 100.0
    strict: false
  cleansing:               # DemandCleanser
    enabled: false
    outlier_method: iqr    # iqr | zscore
    iqr_multiplier: 1.5
    outlier_action: clip   # clip | interpolate | flag_only
    stockout_detection: true
    min_zero_run: 2

parallelism:
  backend: local           # local | spark | ray
  n_workers: -1            # -1 = all CPU cores
  n_jobs_statsforecast: -1
  num_threads_mlforecast: -1
  batch_size: 0            # 0 = all series at once; >0 = chunked
  gpu: false

observability:
  log_format: text         # text | json
  log_level: INFO
  metrics_backend: log     # log | statsd
  statsd_host: localhost
  statsd_port: 8125
  metrics_prefix: forecast_platform
  cost_per_second: 0.0     # cloud cost rate for estimation
  alerts:
    channels: [log]        # log | webhook
    webhook_url: ""
    min_severity: warning  # warning | critical
```

---

## Testing

```bash
python -m pip install --require-hashes --only-binary=:all: -r forecasting-product/requirements/py312/dev.txt
cd forecasting-product
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python -m pytest -v --timeout=180 --timeout-method=thread \
    -o faulthandler_timeout=60
# 2047 cases across 77 test modules; no test module exclusions
# Use the matching py311/py313 profile with those interpreters; no module exclusions.
```

On Windows, set the same process environment variables with PowerShell
(`$env:POLARS_MAX_THREADS = "2"`, etc.) before starting Python. CI runs the complete
suite with the existing 30% coverage gate. Its native-process regressions have a
90-second deadline; only the expensive M5 backtest consumers have a 20-minute
allowance inside the 30-minute backend job limit. The checked-in sample has 105
series and 76,755 rows. One real three-model/two-fold backtest is shared by those
consumers, with separate artifact copies and application/response state; the
forecast still runs through the real API. No model search settings, fixture rows,
folds or numerical assertions are reduced.

The full M5 empirical gates also require the matching approved Linux x86-64
reference selected through `M5_BASELINE_PATH`, with native install/qualification
receipts from the same test environment. CI supplies these per profile; see the
[reference protocol](forecasting-product/tests/integration/baselines/README.md).
The initial references are not historical learned-model no-regression evidence
or a generic/Windows M5 baseline.

The stable **CI gate** requires successful lint, focused validation/intent,
process safety, the full backend matrix, frontend, Docker cold/restart smoke,
and automation safety jobs. Missing, skipped, cancelled or failed constituents
cannot produce a successful gate. Branch protection must require this context;
workflow code alone does not configure repository protection.

Runtime support and dependency reproducibility are described in the
[deployment profiles](docs/DEPLOYMENT.md#supported-runtimes-and-locked-profiles).
CI ships/tests supported Python releases rather than EOL 3.10, verifies generated
profile freshness, and restores exact hashes. The frontend additionally runs
noninteractive lint and real production-bundle Plotly browser regressions, not
only mocked chart component tests.

| Test file | Tests | Covers |
|-----------|------:|--------|
| `test_platform.py` | 85 | Config, hierarchy, reconciliation, metrics, transitions, registry, backtest, REST API, deployment, drift |
| `test_dependency_profiles.py` | 30 | Profile locks/hashes, CPU-only boundaries, audited closure/marker coverage, release policy, stale/tampered exports and native capture/reference wiring |
| `test_m5_baseline.py` | 67 | Original empirical thresholds, complete observations, profile compatibility and audit-bound initial reference integrity |
| `test_forecast_intent.py` | 113 | Real ASGI champion/horizon/config handoff, aliases, safe execution settings, invalid-input no-publication guarantees |
| `test_forecast_explainability.py` | 59 | Comparator, ExceptionEngine, Explainer (STL+SHAP+narrative), ModelCard, Registry, DriftDetector, Lineage |
| `test_intermittent_demand.py` | 55 | SparseDetector, Croston, CrostonSBA, TSB, backtest routing |
| `test_mint_reconciliation.py` | 46 | S-matrix math, OLS/WLS/MinT correctness, coherence, edge cases |
| `test_foundation_models.py` | 41 | Chronos, TimeGPT fit/predict/quantiles, error handling, zero-shot property |
| `test_nixtla_models.py` | 29 | AutoTheta, MSTL extended statsforecast models |
| `test_data_analyzer.py` | 29 | Data analysis module |
| `test_forecastability.py` | 28 | Forecastability signals and scoring |
| `test_frequency_profiles.py` | 25 | Multi-frequency support: FREQUENCY_PROFILES, helpers, config properties, model frequency wiring |
| `test_causal_analyzer.py` | 27 | Causal/econometric analysis |
| `test_probabilistic_ensemble.py` | 24 | Ensemble quantiles, weight computation, config |
| `test_demand_cleansing.py` | 24 | Outlier detection, stockout imputation, period exclusion, CleansingReport |
| `test_data_validator.py` | 24 | Schema checks, duplicates, frequency, value range, completeness, builder integration |
| `test_calibration.py` | 20 | Interval calibration and coverage |
| `test_llm_analyzer.py` | 20 | LLM-based analysis integration |
| `test_quality_report.py` | 19 | Data quality reporting |
| `test_hierarchical_forecaster.py` | 17 | Hierarchical forecaster with reconciliation |
| `test_constrained_demand.py` | 16 | Non-negativity, capacity limits, aggregate budgets, quantile monotonicity |
| `test_rbac.py` | 14 | Role permissions, User model, AuditEvent, AuditLogger log/query/filters |
| `test_override_store.py` | 19 | Override CRUD, approval workflow, old/current DuckDB Arrow result APIs and conversion errors |
| `test_break_detection.py` | 14 | Structural break detection |
| `test_mase.py` | 13 | MASE metric computation |
| `test_statistical_forecasters.py` | 12 | AutoARIMA, AutoETS fit/predict/quantiles |
| `test_fva.py` | 12 | FVA classification, layer metrics, cascade computation, FVAAnalyzer |
| `test_data_preprocessor.py` | 12 | Data preprocessing pipeline |
| `test_ml_forecasters.py` | 19 | LightGBM/XGBoost, native thread budgets, fresh quantile models after refitting |
| `test_hierarchy_aggregator.py` | 11 | Aggregation, disaggregation, proportions |
| `test_data_integrity.py` | 9 | Data integrity checks |
| `test_multi_horizon.py` | 8 | Multi-horizon forecast evaluation |
| `test_regressor_screen.py` | 16 | Variance, correlation, MI screening; auto-drop; SeriesBuilder integration |
| `test_file_classifier.py` | 26 | File role classification: time_series, dimension, regressor, unknown; confidence scoring; cross-file resolution |
| `test_file_merger.py` | 20 | Join key detection, merge preview, full merge, column conflict resolution, null filling |
| `test_pipeline_manifest.py` | 15 | Manifest build, write, read roundtrip; hash determinism; ForecastPipeline integration |
| `test_external_regressors.py` | 6 | Regressor validation, SeriesBuilder with/without features |
| `test_data_loader.py` | 6 | Data loading from files |
| `test_sku_mapping.py` | 81 | SKU transition engine, predecessor matching (attribute, naming, curve fitting, temporal), Bayesian fusion |
| `test_observability.py` | 41 | PipelineContext, StructuredLogger, MetricsEmitter, AlertDispatcher, CostEstimator, PipelineScheduler |
| `test_batch_runner.py` | 8 | BatchInferenceRunner, ParallelismConfig |
| `test_process_execution.py` | 5 | Bounded real ASGI/process execution, payload parity, fitted-model quantile isolation and serialization failures |
| `test_docker_seed.py` | 11 | Identical cold/restart paths, complete seed success and child-process cleanup |
| `test_fabric_portability.py` | 19 | requirements-fabric.txt, ParquetOverrideStore, get_override_store factory, FabricNotebookAdapter |
| `test_evaluator.py` | 5 | Evaluation orchestration |

Repository automation also has 57 dependency-free Node tests under
`.github/scripts/`: cleanup eligibility, atomic expected-SHA deletion against
disposable local repositories, and every CI-gate failure state. They never delete
a real GitHub branch.

---

## Next.js Frontend

A production-grade UI built with Next.js 15 (App Router), TypeScript, and Tailwind CSS. Provides a 10-page workflow (login + 9 workflow pages) and communicates with the FastAPI backend over REST.

| Feature | Details |
|---------|---------|
| **Framework** | Next.js 15, React 19, TypeScript 5.7 |
| **Styling** | Tailwind CSS, Radix UI primitives |
| **Charts** | Recharts (bar, line, pie, area), Plotly (fan chart, sunburst, SBC scatter) |
| **Data fetching** | TanStack React Query with typed API client |
| **Auth** | Server-verified operator JWTs in per-tab memory; explicit demo-only role simulation |
| **Pages** | Login + 9 workflow pages |
| **Dark mode** | Toggle with localStorage persistence |

**Live features** (connected to existing API): file upload/analysis, model leaderboard, drift alerts, audit log, AI explain/triage/config-tuner/commentary.

All frontend features are backed by live API endpoints — no placeholder "Coming Soon" components remain.

**Run locally:**
```bash
cd forecasting-product/frontend
npm ci
npm run dev
# → Open http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `.env.local` to connect to the FastAPI backend.

**Explicit local Docker demo** (loopback API on port 8000, frontend on port 3000;
use a fresh shell without production signing credentials):
```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build
# → Open http://localhost:3000
```

---

## Dependencies

**Supported runtime:** CPython 3.12.16 by default; separately locked/tested
3.11.18 and 3.13.18 on Windows/glibc Linux x86-64. The numerical stack uses
Polars internally and pandas/PyArrow at model-library boundaries. Exact versions
and SHA-256 hashes are in `forecasting-product/requirements/py*/`, generated from
`pyproject.toml`; the ranges below are not promises that every combination works.

**Core CPU/scientific profile:**
```
polars >= 2.0, < 3          # Honest whole-application floor, not the old 0.20-only smoke
numpy / scipy / pandas / pyarrow
statsforecast / mlforecast / hierarchicalforecast
lightgbm / xgboost-cpu / scikit-learn
rapidfuzz / pyyaml
```

The API profile adds FastAPI, patched Starlette, Uvicorn, multipart parsing,
DuckDB, PyJWT and the optional-use Anthropic SDK. Dev adds tests, lint, packaging,
audit and profile-generation tooling. CPU XGBoost does not provide CUDA training.

**Frontend** (Node.js 24.22.1 / npm 11.19.1, installed using `npm ci`):
```
next 16.4.2                # Supported Active LTS, existing Webpack build retained
react 19                   # UI library
tailwindcss 3              # Utility-first CSS
recharts 2 / plotly.js 4    # Chart components
@tanstack/react-query       # Server state management
```

**Optional:**
```
holidays / matplotlib / seaborn / plotly   # Included in dev/Fabric candidate
pyspark / delta-spark                     # Separate local Spark profile
azure-identity                           # External Azure/Fabric integration
shap / neuralforecast                    # Separate, not qualified by the CPU locks
```

NextAuth and bcrypt were unused and are no longer installed by the declared
profiles; operator-issued JWT authentication is unchanged. Managed Fabric is
externally supplied: the local candidate profile does not certify or replace a
workspace's Python/Spark/Delta/Azure runtime. See the
[profile boundaries](docs/DEPLOYMENT.md#supported-runtimes-and-locked-profiles)
before installing optional features.
