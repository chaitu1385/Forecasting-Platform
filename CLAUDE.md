# CLAUDE.md

## Project Overview

Multi-frequency (daily/weekly/monthly/quarterly) sales forecasting product for retail S&OP. Python 3.8+, built on FastAPI (REST API), PySpark (distributed execution), and Polars (data processing). Combines statistical, ML, neural, and foundation model forecasting with hierarchical reconciliation.

Main code lives in `forecasting-product/`.

## Common Commands

```bash
# Install dependencies (full)
pip install -r forecasting-product/requirements.txt

# Install dependencies (Fabric-compatible subset — no DuckDB, PySpark, neuralforecast)
pip install -r forecasting-product/requirements-fabric.txt

# Run all tests
python -m pytest forecasting-product/tests/ -v

# Run a specific test file
python -m pytest forecasting-product/tests/test_platform.py -v

# Start the REST API server
python forecasting-product/scripts/serve.py --demo --port 8000 --data-dir data/

# Start Next.js frontend
cd forecasting-product/frontend && npm install && npm run dev

# Docker quick-start (API + Next.js)
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build

# Run forecast pipeline
python forecasting-product/scripts/run_forecast.py --config forecasting-product/configs/platform_config.yaml --lob retail

# Run backtest pipeline
python forecasting-product/scripts/run_backtest.py --config forecasting-product/configs/platform_config.yaml --lob retail

# Build package
python forecasting-product/setup.py sdist bdist_wheel
```

## Architecture

```
forecasting-product/
├── src/                    # Source modules (125 files across 36 packages)
│   ├── ai/                 # AI-native features (Claude-powered)
│   │   ├── base.py         # AIFeatureBase — shared client wrapper
│   │   ├── nl_query.py     # NaturalLanguageQueryEngine — POST /ai/explain
│   │   ├── anomaly_triage.py # AnomalyTriageEngine — POST /ai/triage
│   │   ├── config_tuner.py # ConfigTunerEngine — POST /ai/recommend-config
│   │   └── commentary.py   # CommentaryEngine — POST /ai/commentary
│   ├── api/                # FastAPI REST endpoints (auth-protected)
│   │   ├── app.py          # create_app() factory, core endpoints, router registration
│   │   ├── schemas.py      # Pydantic request/response models
│   │   ├── deps.py         # Shared dependencies
│   │   └── routers/        # Domain-specific API routers
│   │       ├── series.py       # GET /series/{lob}, POST /series/breaks, cleansing-audit, regressor-screen
│   │       ├── hierarchy.py    # POST /hierarchy/build, aggregate, reconcile
│   │       ├── sku_mapping.py  # POST /sku-mapping/phase1, phase2
│   │       ├── overrides.py    # CRUD /overrides
│   │       ├── pipeline.py     # POST /pipeline/backtest, forecast; GET manifests, costs; POST analyze-multi-file
│   │       ├── analytics.py    # GET /metrics/{lob}/fva, calibration; POST shap, decompose, compare, constrain
│   │       └── governance.py   # GET /governance/model-cards, lineage; POST export
│   ├── audit/              # Append-only Parquet audit logging
│   ├── auth/               # AuthSettings, strict JWT/RBAC, owner-only operator token output
│   ├── backtesting/        # Walk-forward validation, champion selection
│   ├── config/             # YAML schema + loader (dataclass-driven)
│   ├── data/               # Data loading, preprocessing, validation, demand cleansing, regressor screening, external regressors
│   │   ├── validator.py    # DataValidator — schema enforcement, duplicate/frequency/range checks
│   │   ├── cleanser.py     # DemandCleanser — outlier detection, stockout imputation, period exclusion
│   │   ├── quality_report.py # DataQualityAnalyzer — profiling, gap/zero/short series reporting
│   │   ├── regressor_screen.py # RegressorScreen — variance, correlation, MI screening
│   │   ├── regressors.py   # External regressor loader, holiday calendar, validation
│   │   ├── file_classifier.py # FileClassifier — auto-detect file roles (time_series, dimension, regressor)
│   │   └── file_merger.py  # MultiFileMerger — join key detection, multi-file merge with preview
│   ├── evaluation/         # Metric computations (WMAPE, RMSPE, bias, MAE)
│   ├── forecasting/        # Model implementations + registry
│   │   ├── naive.py        # SeasonalNaiveForecaster
│   │   ├── statistical.py  # AutoARIMA, AutoETS, AutoTheta, MSTL
│   │   ├── ml.py           # LGBMDirect, XGBoostDirect
│   │   ├── neural.py       # N-BEATS, NHITS, TFT (via neuralforecast)
│   │   ├── foundation.py   # Chronos, TimeGPT (zero-shot)
│   │   ├── intermittent.py # Croston, CrostonSBA, TSB
│   │   ├── ensemble.py     # WeightedEnsembleForecaster
│   │   ├── hierarchical.py # HierarchicalForecaster
│   │   └── constrained.py  # ConstrainedDemandEstimator (capacity/budget constraints)
│   ├── hierarchy/          # Tree structure, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore (Parquet), drift detection, FVA
│   ├── observability/      # Structured logging, metrics, alerts, cost tracking
│   │   ├── context.py      # PipelineContext — correlation ID threading
│   │   ├── logging.py      # StructuredLogger — JSON logging with context
│   │   ├── metrics.py      # MetricsEmitter — timing, counters, gauges (log/statsd)
│   │   ├── alerts.py       # AlertDispatcher — drift alerts → webhooks
│   │   └── cost.py         # CostEstimator — compute cost tracking
│   ├── overrides/          # Planner manual override store (DuckDB + Parquet fallback)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines, provenance manifest
│   │   ├── manifest.py     # PipelineManifest — provenance sidecar (JSON) for each forecast run
│   │   ├── batch_runner.py # BatchInferenceRunner — partitioned parallel forecasting
│   │   └── scheduler.py    # PipelineScheduler — recurring runs with retry + dead-letter
│   ├── series/             # Series builder, sparse detector, SKU transitions, structural break detection
│   │   ├── break_detector.py # StructuralBreakDetector — CUSUM/PELT changepoint detection
│   ├── sku_mapping/        # New/discontinued SKU mapping
│   ├── spark/              # PySpark distributed execution
│   ├── fabric/             # Microsoft Fabric / Delta Lake deployment
│   └── analytics/          # BI export, comparators, explainability, governance, FVA
├── frontend/               # Next.js 15 frontend (TypeScript, Tailwind, Recharts)
│   ├── src/app/            # App Router pages (login + 8 workflow pages)
│   ├── src/components/     # Reusable components (charts, AI panels, layout, shared)
│   │   ├── forecast/       # Decomposition, comparison, constrained forecast panels
│   │   ├── governance/     # Model cards, lineage, BI export panels
│   │   ├── pipeline/       # Multi-file analysis, pipeline execution panels
│   │   └── sku/            # SKU mapping, override management panels
│   ├── src/hooks/          # React Query hooks + useAsyncOperation utility
│   └── src/lib/            # API client, auth, types, constants
├── tests/                  # 1689 tests (pytest)
├── configs/                # YAML configuration files
├── scripts/                # Entry points (run_backtest, run_forecast, serve, spark_*)
└── notebooks/              # Jupyter notebooks for exploration
```

## Code Style

- **Classes**: PascalCase (`LGBMDirectForecaster`, `HierarchyTree`)
- **Functions/methods**: snake_case (`compute_wmape`, `fit_model`)
- **Private members**: leading underscore (`_make_weekly_actuals`)
- **Type hints** on all public APIs
- **Docstrings** in Google/NumPy style with Parameters, Returns, Examples
- **Polars DataFrames** throughout production code (not Pandas)
- **Dataclasses** for configuration (`src/config/schema.py`)
- **ABCs** for extensible interfaces (`BaseForecaster`)
- PEP 8 conventions (no explicit linter configured)

## Configuration

YAML-driven config system with dataclass schema validation:
- `configs/base_config.yaml` — platform defaults
- `configs/platform_config.yaml` — standard deployment
- `configs/fabric_config.yaml` — Microsoft Fabric settings
- `configs/lob/` — line-of-business overrides (inherit from base)
- Schema defined in `src/config/schema.py`

Key config dataclasses: `ForecastConfig`, `BacktestConfig`, `DataQualityConfig` (contains `ValidationConfig`, `CleansingConfig`), `ConstraintConfig`, `ExternalRegressorConfig` (contains `RegressorScreenConfig`), `ParallelismConfig`, `ObservabilityConfig` (contains `AlertConfig`), `AIConfig`

`src/config/loader.py::parse_config_yaml` validates known field types without
coercing strings/booleans into numeric settings. Pipeline uploads reject invalid
config with HTTP 400 before pipeline construction. Execution precedence is explicit
request override > valid YAML > platform defaults; the required
`PipelineRunResponse.execution` contract in `src/api/schemas.py` is an explicit
allowlist, not `asdict(PlatformConfig)`. Keep server paths and credentials out of it.
The UI forwards only the current backtest champion/horizon and invalidates dependent
results when data, LOB, or configuration changes.

### Multi-frequency support

The platform supports daily (`"D"`), weekly (`"W"`), monthly (`"M"`), and quarterly (`"Q"`) data frequencies. The `FREQUENCY_PROFILES` dict in `src/config/schema.py` is the single source of truth, mapping each frequency to: `season_length`, `default_lags`, `min_series_length`, `default_val_periods`, `default_horizon`, `statsforecast_freq`, and `timedelta_kwargs`.

Set `frequency` in the YAML config:
```yaml
forecast:
  frequency: "M"          # "D" | "W" | "M" | "Q"
  horizon_periods: 12     # alias for horizon_weeks (backward-compat)
backtest:
  val_periods: 3          # alias for val_weeks (backward-compat)
```

Helper functions: `get_frequency_profile(freq)` returns the profile dict; `freq_timedelta(freq, periods)` returns a `timedelta` for date arithmetic. All models, backtesting, validation, and data processing use these instead of hardcoded weekly values.

## Authentication contract

The factory, `serve.py` and container use `AUTH_ENABLED=true`, `API_DEV_MODE=false`
by default and require a dedicated random `JWT_SECRET_KEY`. Invalid booleans,
contradictory modes and missing/weak keys fail startup. Explicit demo requires
auth false/dev true/no key; `--demo` does not discard contradictory environment
settings. Test clients must declare `auth_enabled=False, dev_mode=True, jwt_secret=""`.
Never replace RBAC globally with an admin stub.

There is no HTTP token issuer. `scripts/issue_token.py` is a trusted local
operator tool, writing one new private file (POSIX 0600; atomic verified Windows
DACL), never a credential argument or stdout token. It is an MVP JWT trust model,
not SSO/password auth or a user directory. Signed inactive claims are denied;
per-user revocation is not live. Use short expiry or coordinated key rotation.

`/auth/status` is nonsensitive and public; `/auth/me` verifies identity and role.
Frontend credentials stay in tab memory only. Logout, expiry, 401 and replacement
login clear protected caches; stale requests cannot affect a newer session.
Production never trusts a role picker or restores legacy localStorage auth.
Only explicit demo renders role simulation; production seeding is disabled.

## Testing

- Framework: pytest
- Test files mirror source structure with `test_` prefix
- Helper fixtures use `_make_*` factory functions (e.g., `_make_weekly_actuals`)
- 1689 cases across 64 test modules; run the full suite without module exclusions
- Key test modules: `test_platform.py` (85 tests), `test_sku_mapping.py` (81), `test_ai_*.py` (73), `test_forecast_explainability.py` (59), `test_intermittent_demand.py` (55), `test_observability.py` (41), `test_file_classifier.py` (26), `test_file_merger.py` (20)
- Focused forecast-intent coverage: `test_forecast_intent.py` (107 real ASGI cases),
  plus frontend `e2e/pipeline-execution.test.tsx` and `e2e/data-onboarding-state.test.tsx`
  (21 handoff/state cases). These require only core/API and existing frontend test dependencies.
- Process safety: `test_process_execution.py` has five real subprocess cases,
  including the ASGI backtest path, configured model/frequency parity, and
  serialization errors. Both production pools use a local `spawn` context, never
  a global multiprocessing start-method override. Worker model objects must
  remain importable/pickleable; failed forecast batches must not return partial success.
  ML refits invalidate training-dependent quantile caches; a previously fitted
  model must produce the same fold-local intervals as a fresh model.
- CI limits native pools to Polars 2 and OpenMP/OpenBLAS/MKL/Numba 1 per process.
  ML forecasters pass `num_threads` to native learners as `n_jobs` unless the
  caller explicitly overrides the learner budget. Pool size is capped by useful tasks.
- The M5 tests preserve all 105 checked-in series, 76,755 rows, three models and
  two folds. They share only the expensive real backtest result; each consuming
  class gets private files, client and response state. Only these classes receive
  a 1200-second test timeout; process regressions retain their 90-second deadline.
- `test_docker_seed.py` covers cold/restart path consistency and seed/process
  failure handling. `.github/scripts/*.test.cjs` contains 61 automation tests.
  Keep the `CI gate` required-job list aligned with its `needs` and evaluator;
  no failed/cancelled/skipped constituent may pass it.
- Authentication: 60 JWT/RBAC/private-output cases, 18 real startup cases and
  11 seeder cases run in a required Linux/Windows job. Frontend coverage includes
  20 memory/session cases, eight explicit-demo browser cases and an isolated
  production browser/API flow with private synthetic credentials and capture
  disabled. Never upload that fixture's credentials or enable tracing.

## Key Dependencies

Core: polars, statsforecast, mlforecast, lightgbm, xgboost, scikit-learn, fastapi, pyyaml, duckdb, pyjwt
Frontend: next, react, typescript, tailwindcss, recharts, @tanstack/react-query, next-auth, plotly
Optional: neuralforecast, pyspark, shap, bcrypt, holidays, delta-spark, azure-identity

## Documentation Convention

When adding a new module or capability, update these files:

1. **`README.md`** (root) — Add module docs (class table, description). Update test count and dependency list if changed.
2. **`CLAUDE.md`** (root) — Update architecture tree. Update test count. Add new config dataclasses if applicable.
3. **`CONCEPTS.md`** (root) — Add a concept entry if the feature introduces a new "why" that non-domain-experts need to understand (3-4 sentences: what, why, when).
4. **`EDGE_CASES.md`** (root) — Add an entry if the feature handles a new failure mode (what happens, how we handle it, what to watch for).
5. **`ARCHITECTURE.md`** (root) — Visual Mermaid diagrams showing system overview, data flow, pipeline sequences, and component interactions. Update if adding new subsystems or changing data flow.
6. **`QUICKSTART.md`** (root) — Get-running-in-2-minutes guide. Update if setup steps, Docker config, or prerequisites change.
7. **`docs/`** (directory) — User-facing guides: `DATA_FORMAT.md` (input/output schemas), `DEPLOYMENT.md` (deployment & config), `USER_GUIDE.md` (end-to-end usage), `TROUBLESHOOTING.md` (common issues & FAQ), `FRONTEND.md` (frontend architecture & component reference). Update the relevant guide when adding user-visible features or changing data schemas.

These are the only documentation files in the repo. All other docs (plans, specs, analyses) are transient working documents — delete them once the work is merged.
