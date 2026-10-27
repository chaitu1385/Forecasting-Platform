# Deployment Guide

How to run the Forecasting Product — from local development to production deployment.

---

## Quick Start: Docker Compose

The fastest path to a running platform:

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose up
```

This starts two services:

| Service | Port | URL |
|---------|------|-----|
| REST API (FastAPI) | 8000 | http://localhost:8000/docs (Swagger UI) |
| Frontend (Next.js) | 3000 | http://localhost:3000 |

The API persists data in the named `api-data` volume mounted at
`/app/forecasting-product/data`; the frontend retrieves it through the API.
First boot seeds the demo backtest and forecast. Subsequent containers use the
same volume and skip seeding only after the completion marker has been written.
Do not delete that volume to troubleshoot a restart.

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_DATA_DIR` | `forecasting-product/data/` | Root data directory for the API |
| `API_METRICS_DIR` | `forecasting-product/data/metrics/` | Metric store location |
| `POLARS_MAX_THREADS` | `2` in the image | Native Polars threads per process |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` | `1` in the image | Native pool budgets; set before process startup |
| `ANTHROPIC_API_KEY` | (empty) | Claude API key for AI features (optional) |
| `NEXTAUTH_SECRET` | `docker-dev-secret` | Secret for NextAuth.js session encryption |
| `NEXTAUTH_URL` | `http://localhost:3000` | Canonical URL of the Next.js frontend |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend URL for the frontend |

Set AI features:
```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up
```

### Docker Health Check

The API service has a built-in health check hitting `GET /health` every 30 seconds.
Compose starts the frontend after the API container has started; this does not
itself guarantee that demo seeding has finished. The seeder resolves storage
paths once and passes identical absolute data/metrics paths on cold and restart
launches. A failed backtest, missing champion, failed/empty forecast or child
startup failure does not write the marker or leave an apparently seeded server
running. The child is stopped and reaped on startup failure or termination.

CI checks more than HTTP liveness: it waits up to 240 seconds for seeded
readiness, retrieves all 10 demo series / 120 forecasts and the leaderboard,
then starts a **new container with the same test volume** and compares persisted
artifact hashes. An exit trap stops test containers even on failure and removes
only the temporary volume created by that smoke run. This does not reset
deployment data.

---

## Local Python Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
# Full install (all features)
pip install -r forecasting-product/requirements.txt

# Fabric-compatible subset (no DuckDB, PySpark, neuralforecast)
pip install -r forecasting-product/requirements-fabric.txt
```

### Start the API Server

```bash
python forecasting-product/scripts/serve.py --port 8000 --data-dir data/
```

CLI arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--data-dir` | `data/` | Root data directory |
| `--metrics-dir` | `data/metrics/` | Metric store directory |
| `--reload` | off | Enable hot-reload (development only) |
| `--workers` | `1` | Number of Uvicorn worker processes |

Environment variables `API_DATA_DIR` and `API_METRICS_DIR` override their CLI counterparts.

### Start the Next.js Frontend

Requires Node.js 18+. The frontend connects to the FastAPI backend over REST.

```bash
cd forecasting-product/frontend
npm install
npm run dev
```

Opens at http://localhost:3000.

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI backend URL |
| `NEXTAUTH_SECRET` | (required for production) | Secret for NextAuth.js session encryption |
| `NEXTAUTH_URL` | `http://localhost:3000` | Canonical URL of the frontend |

For production builds:

```bash
npm run build
npm start          # starts production server on port 3000
```

---

## Running Pipelines

### Backtest Pipeline

Evaluates all configured models via walk-forward cross-validation and selects a champion:

```bash
python forecasting-product/scripts/run_backtest.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob retail
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to platform config YAML |
| `--lob-override` | No | LOB-specific config override YAML |
| `--data` | Yes | Path to actuals (Parquet or CSV) |
| `--product-master` | No | Path to product master table |
| `--mapping-table` | No | Path to SKU mapping table |

### Forecast Pipeline

Generates forecasts using the champion model:

```bash
python forecasting-product/scripts/run_forecast.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob retail
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to platform config YAML |
| `--lob-override` | No | LOB-specific config override YAML |
| `--data` | Yes | Path to actuals (Parquet or CSV) |
| `--champion` | No | Champion model name (default: `naive_seasonal`) |
| `--product-master` | No | Path to product master table |
| `--mapping-table` | No | Path to SKU mapping table |

---

## Configuration

The platform uses a YAML config system with layered overrides.

### Config Files

| File | Purpose |
|------|---------|
| `configs/base_config.yaml` | Platform defaults |
| `configs/platform_config.yaml` | Standard deployment config |
| `configs/fabric_config.yaml` | Microsoft Fabric settings |
| `configs/lob/<name>.yaml` | Line-of-business overrides (inherit from base) |

### Key Settings to Customize

```yaml
# What to forecast
forecast:
  frequency: W                          # D | W | M | Q
  horizon_weeks: 39                     # forecast horizon (periods)
  forecasters: [lgbm_direct, auto_ets, seasonal_naive]
  quantiles: [0.1, 0.5, 0.9]           # prediction intervals

# How to evaluate
backtest:
  n_folds: 3                            # walk-forward CV folds
  val_weeks: 13                          # validation window
  champion_granularity: lob              # lob | product_group | series
  selection_strategy: champion           # champion | weighted_ensemble

# Data quality
data_quality:
  validation:
    enabled: true                        # schema + duplicate checks
    strict: false                        # true = halt on errors
  cleansing:
    enabled: true                        # outlier + stockout correction
    outlier_method: iqr                  # iqr | zscore

# Performance
parallelism:
  backend: local                         # local | spark
  n_workers: -1                          # -1 = all CPU cores
  batch_size: 0                          # 0 = all series at once
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete config schema details.

### Post-Validation and Forecast Publication

The Python `ForecastPipeline` validates both standard and stitched multi-horizon
forecasts before creating or writing a forecast file or its manifest. To block
publication when a check reports a `BLOCKER`, configure:

```yaml
post_validation:
  enabled: true
  halt_on_blocker: true
  min_grade: D
```

With `halt_on_blocker: true`, a blocker raises `ValidationError`. An unexpected
exception in any enabled validation layer also fails the forecast attempt,
regardless of `halt_on_blocker`. Neither failure publishes new forecast/manifest
files or overwrites files from an earlier successful attempt, including a rerun
for the same forecast origin.

The default `halt_on_blocker: false` keeps reported blockers diagnostic; it does
not suppress unexpected checker errors. `min_grade` only controls a warning and
is not a publication threshold. Explicitly setting `enabled: false` skips
post-validation. Successful validated runs include `post_validation_grade` and
`post_validation_score` in the manifest; disabled runs leave these fields null.

Forecast primary keys include the configured `forecast.series_id_column` and
`forecast.time_column`, so multiple periods for one series are valid. Backtest
results retain their canonical `series_id`, `model_id`, `fold`, and `target_week`
keys. These publication safeguards apply to `ForecastPipeline`, not the separate
distributed Spark/Fabric execution paths.

---

## Authentication Setup

Authentication is disabled by default for development. To enable:

### 1. Enable Auth in Config

The API checks if auth is enabled. When disabled, all requests are treated as admin.

### 2. Create Tokens

```bash
# Via the API
curl -X POST "http://localhost:8000/auth/token?username=analyst&role=data_scientist"
# Returns: {"access_token": "eyJ...", "token_type": "bearer"}
```

### 3. Use Tokens

```bash
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/forecast/retail
```

### Role Permissions

| Action | admin | data_scientist | planner | manager | viewer |
|--------|-------|----------------|---------|---------|--------|
| View forecasts/metrics | Y | Y | Y | Y | Y |
| Create overrides | Y | Y | Y | - | - |
| Approve overrides | Y | - | - | Y | - |
| Run backtest/pipeline | Y | Y | - | - | - |
| Promote champion model | Y | Y | - | - | - |
| View audit log | Y | Y | - | Y | - |
| Manage users | Y | - | - | - | - |

Required package: `pip install PyJWT`

---

## Observability Setup

### Logging

```yaml
observability:
  log_format: json       # "text" for human-readable, "json" for structured
  log_level: INFO         # DEBUG | INFO | WARNING | ERROR
```

JSON logs include `run_id`, `lob`, and timestamps for every pipeline operation.

### Metrics

```yaml
observability:
  metrics_backend: log      # "log" = JSON lines, "statsd" = UDP to StatsD
  statsd_host: localhost
  statsd_port: 8125
  metrics_prefix: forecast_platform
```

### Drift Alerts

```yaml
observability:
  alerts:
    channels: [log, webhook]
    webhook_url: "https://hooks.slack.com/services/..."    # Slack, Teams, PagerDuty
    min_severity: warning    # "warning" | "critical"
```

Alerts fire when forecast accuracy drifts beyond configured thresholds. Start with `min_severity: critical` and lower to `warning` once you've tuned thresholds for your data.

### Cost Tracking

```yaml
observability:
  cost_per_second: 0.0001    # $/second for compute cost estimation
```

Tracks per-model compute time and estimates cloud costs per run.

---

## Scaling

### Local (ProcessPool)

```yaml
parallelism:
  backend: local
  n_workers: -1              # -1 = all CPU cores
  batch_size: 500            # chunk series into batches of 500
```

- Reduce `n_workers` if hitting memory limits
- Reduce `batch_size` to reduce memory per worker (each worker loads its batch + model)

Local backtest and inference pools explicitly use `spawn`, which starts a fresh
interpreter instead of inheriting native-library locks through `fork`. Parallel
workers receive the configured model object, including non-default parameters
and frequency, and pools never allocate more workers than useful tasks. Custom
forecasters must be importable and pickleable; use a guarded script entry point
(`if __name__ == "__main__":`) when launching local multiprocessing.

LightGBM/XGBoost point and quantile learners inherit the forecaster's
`num_threads` budget as `n_jobs`; an explicit learner `n_jobs` override takes
precedence. Configure native pool environment variables before importing Polars
or numerical libraries. Do not compensate for an unsafe fork with fewer threads.
Backtest model failures remain available through `engine.failures`; batch
inference raises on a failed worker rather than returning an incomplete forecast.

### CI and branch cleanup

Require the stable `CI gate` status in repository branch protection. It evaluates
every lint, focused, process, full-backend-matrix, frontend, Docker and automation
job, including failure/cancellation/skip states. Administrators configure that
required context separately from these workflows.

`Cleanup merged branch` is **manual and dry-run by default**, never triggered
by a push. Run it from the actual default branch with one merged PR number.
It retains forks, default/conventional main/current/protected branches, branches
used by an open PR, and any branch whose live head differs from the merged PR
head. Unchecking `dry_run` explicitly opts into deletion after fresh checks.
The exact expected-SHA Git lease rejects a concurrent head advance; it is a
conditional deletion, not permission to force-update history. API/Git failures
are surfaced, not reported as deletion success.

The SHA lease is not an atomic transaction with GitHub PR metadata: an open PR
can be created after the last check, and a branch deleted/recreated at the
identical SHA is indistinguishable by Git. Inspect a dry run and coordinate
branch reuse before opting in. Do not reactivate the legacy broad push-triggered
workflow; enable only the reviewed manual replacement after it has been merged.

### PySpark (Distributed)

For large-scale runs on Databricks or Fabric:

```bash
python forecasting-product/scripts/spark_forecast.py \
  --config forecasting-product/configs/platform_config.yaml
```

Uses `SparkForecastPipeline` for distributed series processing. Configure via:

```yaml
parallelism:
  backend: spark
```

---

## Microsoft Fabric Deployment

### Prerequisites

- Fabric workspace with a Lakehouse
- `requirements-fabric.txt` installed in notebook environment

### Limitations

Fabric's Spark runtime does **not** include:
- DuckDB (override store auto-falls back to Parquet)
- neuralforecast / PyTorch (neural models unavailable)
- FastAPI (API runs externally, not in Fabric)

### Notebook Deployment

```python
from src.fabric.deployment import DeploymentOrchestrator, DeploymentConfig

cfg = DeploymentConfig(
    lob="retail",
    workspace="my-workspace",
    lakehouse="my-lakehouse",
    max_staleness_days=14,     # actuals must be within 14 days
    write_mode="upsert",       # "upsert" | "overwrite_partition" | "append"
)

orch = DeploymentOrchestrator(spark, config=platform_config, deploy_config=cfg)
result = orch.run(actuals_sdf=actuals_spark_df)
```

The orchestrator runs pre-flight checks (data freshness, schema), backtests (optional), forecasts, writes to Delta tables, and logs to the deploy audit table.

### Fabric Config

```yaml
# configs/fabric_config.yaml
fabric:
  workspace: my-workspace
  lakehouse: my-lakehouse
```

---

## Production Checklist

Before going to production, verify:

- [ ] **Auth enabled** — JWT authentication active, secrets rotated
- [ ] **HTTPS** — TLS termination configured (reverse proxy or load balancer)
- [ ] **Logging** — `log_format: json` for structured log aggregation
- [ ] **Alerts** — Webhook configured for drift alerts (Slack/Teams/PagerDuty)
- [ ] **Data validation** — `validation.enabled: true` with `strict: true`
- [ ] **Demand cleansing** — `cleansing.enabled: true` for outlier/stockout handling
- [ ] **Parallelism** — `n_workers` tuned for available CPU/memory
- [ ] **Batch size** — Set `batch_size > 0` for memory-constrained environments
- [ ] **Monitoring** — Health check endpoint (`/health`) monitored
- [ ] **Backups** — Metric store and audit log directories backed up
- [ ] **Cost tracking** — `cost_per_second` set if tracking cloud spend
- [ ] **API key** — `ANTHROPIC_API_KEY` set if using AI features
