# CLAUDE.md

## Project Overview

Multi-frequency (daily/weekly/monthly/quarterly) sales forecasting product for retail S&OP. Supported CPython 3.11-3.13 (production 3.12.16), built on FastAPI (REST API), PySpark (distributed execution), and Polars (data processing). Combines statistical, ML, neural, and foundation model forecasting with hierarchical reconciliation.

Main code lives in `forecasting-product/`.

## Common Commands

```bash
# Install the hash-locked Python 3.12 development profile
pip install -r forecasting-product/requirements.txt

# Python 3.11/3.13 use the matching requirements/py311 or py313 directory.
# requirements-fabric.txt is a local Python 3.13 candidate, not a managed-runtime certification.

# Run all tests
python -m pytest forecasting-product/tests/ -v

# Run a specific test file
python -m pytest forecasting-product/tests/test_platform.py -v

# Start the REST API server
python forecasting-product/scripts/serve.py --demo --port 8000 --data-dir data/

# Start Next.js frontend
cd forecasting-product/frontend && npm ci && npm run dev

# Docker quick-start (API + Next.js)
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build

# Run forecast pipeline
python forecasting-product/scripts/run_forecast.py --config forecasting-product/configs/platform_config.yaml --lob retail

# Run backtest pipeline
python forecasting-product/scripts/run_backtest.py --config forecasting-product/configs/platform_config.yaml --lob retail

# Build package
python -m build --no-isolation forecasting-product
```

## Architecture

```
forecasting-product/
├── src/                    # Application source modules
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
│   │       ├── pipeline.py     # POST backtest/forecast; GET exact runs/config/input, manifests, costs; multi-file analysis
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
│   │   └── constrained.py  # apply_point_constraints + ConstrainedDemandEstimator (shared bounds/point budgets)
│   ├── hierarchy/          # Tree structure, aggregation, reconciliation (OLS/WLS/MinT)
│   ├── metrics/            # MetricStore (Parquet), drift detection, FVA
│   ├── observability/      # Structured logging, metrics, alerts, cost tracking
│   │   ├── context.py      # PipelineContext — correlation ID threading
│   │   ├── logging.py      # StructuredLogger — JSON logging with context
│   │   ├── metrics.py      # MetricsEmitter — timing, counters, gauges (log/statsd)
│   │   ├── alerts.py       # AlertDispatcher — drift alerts → webhooks
│   │   └── cost.py         # CostEstimator — compute cost tracking
│   ├── overrides/          # Planner manual override store (DuckDB + Parquet fallback)
│   ├── pipeline/           # End-to-end backtest + forecast pipelines, immutable local bundles
│   │   ├── run_store.py    # RunStore — UUID authority, atomic commit marker, integrity/exact/latest reads
│   │   ├── readers.py      # RunReader — shared forecast/history discovery and explicit legacy adapters
│   │   ├── manifest.py     # PipelineManifest — safe versioned provenance, canonical config/model metadata
│   │   ├── batch_runner.py # BatchInferenceRunner — partitioned parallel forecasting
│   │   └── scheduler.py    # PipelineScheduler — recurring runs with retry + dead-letter
│   ├── series/             # Series builder, sparse detector, SKU transitions, structural break detection
│   │   ├── break_detector.py # StructuralBreakDetector — CUSUM/PELT changepoint detection
│   ├── sku_mapping/        # New/discontinued SKU mapping
│   ├── spark/              # PySpark distributed execution
│   ├── fabric/             # Microsoft Fabric / Delta Lake deployment
│   └── analytics/          # BI export, comparators, explainability, governance, FVA
├── frontend/               # Next.js 16 / Node 24 (TypeScript, Tailwind, Recharts, Plotly)
│   ├── src/app/            # App Router pages (login + 8 workflow pages)
│   ├── src/components/     # Reusable components (charts, AI panels, layout, shared)
│   │   ├── forecast/       # Decomposition, comparison, constrained forecast panels
│   │   ├── governance/     # Model cards, lineage, BI export panels
│   │   ├── pipeline/       # Multi-file analysis, pipeline execution panels
│   │   └── sku/            # SKU mapping, override management panels
│   ├── src/hooks/          # React Query hooks + useAsyncOperation utility
│   └── src/lib/            # API client, auth, types, constants
├── tests/                  # 2225 cases (pytest)
├── configs/                # YAML configuration files
├── requirements/           # Hash-locked CPU/API/dev/AI/Fabric/Spark profiles per Python minor
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
- PEP 8 conventions; existing Ruff E/F/W/I rules are configured in `pyproject.toml`

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

### Constraint contract

`forecasting.constrained.apply_point_constraints` is shared by the wrapper and
`POST /forecast/constrain`. Validate finite values/bounds, unique non-null keys,
complete per-series capacities, `cap >= floor`, and per-period
`budget >= row_count * floor`. Clip row bounds first; proportional allocation
scales only headroom above floors. Largest-first removes excess by descending
clipped value, then ascending string series ID, preserving input order and all
nonvalue columns. Never duplicate arithmetic in the route or round away a budget
violation. Full-population diagnostics precede the 200-row preview.

Refitting clears capacity state; a configured missing capacity column/map is an
error, not permission to retain previous limits. Quantiles support the same
monotone element-wise clipping without a budget, with finite/order validation
before and after. Enabled aggregate-plus-quantile requests fail explicitly as
`unsupported_aggregate_quantiles` before publication. Disabled wrappers remain
no-ops. Do not sort crossed quantiles, invent a joint distribution, or force a
general point to P50. See `docs/NUMERICAL_RELIABILITY.md`.

### Immutable run storage

`OutputConfig.runs_path` explicitly selects the local authority. Without it,
direct/custom forecast pipelines use `<forecast_path>/runs`, and backtests use
`<metrics_path>/runs`; never infer a shared parent or silently redirect to cwd.
The API/standard CLI/demo deployment pass a shared explicit root
(`API_RUNS_DIR` / `--runs-dir`, default `data/runs`). The YAML loader constructs
`OutputConfig` explicitly, so new schema fields require explicit parser wiring.

A fresh UUID32 at each `run()` entry is shared by context, engine metric rows,
manifest and response. A run is visible only after a no-replace hard link
publishes fixed `manifest.json` in its exclusive directory. All required members
must already be closed, fsynced and hashed. Never catch a metadata write failure
and return completed, mirror to a second authoritative store, delete a committed
run after an error, or use stale output from the previous invocation.
`RunPublicationError` distinguishes `not_committed`, `committed` and `uncertain`.
Latest is attempt-start UTC plus ID within LOB/type; retry means a new attempt.

Exact backtest history must return stored `leaderboard.parquet` unchanged.
`MetricStore.leaderboard_result` resolves its frame and manifest together;
`leaderboard_from_records` is the explicit current scorer, not a read adapter.
New runs persist `execution.scoring_contract` (policy/version, criteria, scope)
in the manifest and canonical config. Old missing policy stays unknown. Preserve
stored row order/rank, cohort identity, auth/path checks and config hashes through
API/notebook/BI/UI consumers; never re-rank history by today's WMAPE.

Readers verify identities, fixed member names, checksums and typed schemas;
corruption/unknown versions must be explicit, not healthy empty/fallback.
Pass the same explicit root to MetricStore/notebook readers. Legacy reads are
read-only/unknown lineage. Keep credential/private-path settings out of
canonical config/model provenance; input byte hashes do not claim semantic
equivalence or deterministic fitted-model replay. Managed Spark/Fabric Delta
storage remains a separate contract. See docs/DEPLOYMENT.md for local filesystem,
fsync/Windows and offline recovery limitations.

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

`get_frequency_profile(freq)` returns defaults. `utils.calendar.PeriodCalendar`
validates the explicitly configured frequency and preserves observed weekday,
month-start/end and quarter-start/end anchors. It drives model, feature, gap-fill
and CV dates. `freq_timedelta` is only valid for fixed D/W periods; M/Q raise
instead of approximating months as 30 days. Explicit seasonal parameters are
never replaced by frequency defaults.

New selection and newly computed leaderboards use `metrics.aggregation` on complete raw observation
keys, not means of cached metric columns. Default `wmape` is volume-weighted;
`macro_wmape` explicitly requests equal series/fold weighting. Bias is positive
for over-forecasting and ranks by absolute magnitude. See
[the numerical contract](docs/NUMERICAL_RELIABILITY.md), including undefined
scores, gap lead times, preprocessing restrictions and historical-run migration.

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

Install the matching committed `requirements/py311`, `py312` or `py313` dev
profile with `--require-hashes --only-binary=:all:`. Python 3.10 is EOL.
`scripts/lock_dependencies.py --check` verifies generated provenance offline;
CI adds `--upstream` to reject yanked/non-upstream artifacts independently of CVEs.
The hash-export pass must preserve the resolver's exact versions and markers;
never let a shared wheel cache promote an unrequested prerelease.
regenerate using the pinned pip/pip-tools and a private Windows/Linux wheel cache.
Polars 2.0 is the supported application floor. The CPU profile intentionally
uses `xgboost-cpu`, not a GPU/NCCL distribution; provenance must record the actual
distribution version. Never replace locks with unconstrained installs to make CI pass.

- Framework: pytest
- Test files mirror source structure with `test_` prefix
- Helper fixtures use `_make_*` factory functions (e.g., `_make_weekly_actuals`)
- 2225 cases across 78 test modules; run the full suite without module exclusions
- Constraint coverage: `test_constrained_demand.py` (125 cases) and
  `test_constraint_api_contracts.py` (69 real authenticated API/domain/publication cases)
  assert distinct hand allocations, full populations, exact keys and explicit
  infeasible/unsupported errors without changing existing immutable runs.
- Key test modules: `test_platform.py` (85 tests), `test_sku_mapping.py` (81), `test_ai_*.py` (73), `test_forecast_explainability.py` (59), `test_intermittent_demand.py` (55), `test_observability.py` (41), `test_file_classifier.py` (26), `test_file_merger.py` (20)
- Focused forecast-intent coverage: `test_forecast_intent.py` (108 real ASGI cases),
  plus frontend `e2e/pipeline-execution.test.tsx` and `e2e/data-onboarding-state.test.tsx`
  (21 handoff/state cases). These require only core/API and existing frontend test dependencies.
- Run safety: `test_run_store.py`, `test_run_artifacts.py`,
  `test_run_store_process.py` and `test_run_cli.py` cover publication faults on
  both sides of the marker, exact/legacy/auth/path contracts, actual multiprocess
  completion races, owned-process interruption/restart, isolated restore and
  explicit root precedence. Required Linux/Windows CI jobs do not skip these.
  `e2e/run-selection.test.tsx` and the private production Playwright case exercise
  exact older-run values/config, stale response isolation and no POST retries.
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
- Authentication: 61 JWT/RBAC/private-output cases, 18 real startup cases and
  11 seeder cases run in a required Linux/Windows job. Frontend coverage includes
  20 memory/session cases, eight explicit-demo browser cases and an isolated
  production browser/API flow with private synthetic credentials and capture
  disabled. Never upload that fixture's credentials or enable tracing.

## Key Dependencies

Core CPU/scientific: polars, numpy, scipy, pandas, pyarrow, statsforecast, mlforecast, hierarchicalforecast, lightgbm, xgboost-cpu, scikit-learn, pyyaml, rapidfuzz
API: fastapi, starlette, uvicorn, python-multipart, duckdb, pyjwt; Anthropic SDK included but calls require explicit configuration
Frontend: Node 24.22.1, npm 11.19.1, Next 16.4.2, React 19, TypeScript, Tailwind 3, Recharts, @tanstack/react-query, Plotly 4
Optional: neuralforecast, pyspark, shap, holidays, delta-spark, azure-identity

Keep the supported Webpack flag during the Next 16 migration; no bundler rewrite
is implied. `npm run lint` is noninteractive. `npm run test:runtime-browser`
checks actual fan/sunburst React pages in the production standalone server using
synthetic API interception, without provider calls or external map resources.

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
