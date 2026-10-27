# Deployment Guide

How to run the Forecasting Product — from local development to production deployment.

---

## Quick Start: Docker Compose

For an explicitly unauthenticated local synthetic demo, use a fresh shell without
production signing credentials:

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build
```

This starts two services:

| Service | Port | URL |
|---------|------|-----|
| REST API (FastAPI) | 8000 | http://localhost:8000/docs (Swagger UI) |
| Frontend (Next.js) | 3000 | http://localhost:3000 |

The demo API persists data in the separate `api-demo-data` volume mounted at
`/app/forecasting-product/data`; the frontend retrieves it through the API.
First boot seeds the demo backtest and forecast. Subsequent containers use the
same volume and skip seeding only after the completion marker has been written.
Do not delete that volume to troubleshoot a restart.

The base `docker-compose.yml` instead defaults to production authentication,
stores data in `api-data`, and never seeds demo data. Inject a valid
`JWT_SECRET_KEY` and follow [Authentication Setup](#authentication-setup) before
running `docker compose up --build`. There is no default secret or anonymous
production login. Both configurations bind host ports to loopback.

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_DATA_DIR` | `forecasting-product/data/` | Root data directory for the API |
| `API_METRICS_DIR` | `forecasting-product/data/metrics/` | Metric store location |
| `POLARS_MAX_THREADS` | `2` in the image | Native Polars threads per process |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` | `1` in the image | Native pool budgets; set before process startup |
| `ANTHROPIC_API_KEY` | (empty) | Claude API key for AI features (optional) |
| `AUTH_ENABLED` | `true` | Authentication required; disabling requires explicit demo mode |
| `API_DEV_MODE` | `false` | Explicit unauthenticated simulation; never production |
| `JWT_SECRET_KEY` | unset (startup fails) | Dedicated production signing key from secure environment |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Public frontend build argument; never a secret |

AI provider credentials are separate from sign-in configuration. Inject them
explicitly into the API service only when enabling those providers; they are not
needed for startup or the synthetic authentication/demo checks.

### Docker Health Check

The API service has a built-in health check hitting `GET /health` every 30 seconds.
Compose starts the frontend after API health succeeds; this does not
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

CI also verifies that the production image exits on missing/weak/contradictory
authentication settings, does not seed a demo, rejects anonymous/invalid/wrong-role
requests, accepts privately provisioned operator tokens, and enforces authentication
after a fresh-container restart. Only fresh smoke volumes and synthetic credentials
are used. No signing material or browser traces are uploaded.

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
| `--host` | `0.0.0.0` production; `127.0.0.1` demo | Bind address; explicit flag overrides `API_HOST` |
| `--port` | `8000` | Bind port |
| `--data-dir` | `data/` | Root data directory |
| `--metrics-dir` | `data/metrics/` | Metric store directory |
| `--reload` | off | Enable hot-reload (development only) |
| `--workers` | `1` | Number of Uvicorn worker processes |
| `--demo` | off | Explicit local simulation; rejects conflicting auth settings |

The server requires production authentication configuration before binding. For a
local synthetic demo only, add `--demo`. Explicit path flags override
`API_DATA_DIR` / `API_METRICS_DIR`; metrics otherwise default under the data root.

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

The factory, `serve.py`, and container entrypoint share one fail-closed contract.
Production authentication is the default, not something inferred from whether a
secret happens to exist. PyJWT is a core API dependency.

| Mode | `AUTH_ENABLED` | `API_DEV_MODE` | `JWT_SECRET_KEY` |
|------|----------------|----------------|------------------|
| Production (default) | `true` | `false` | Dedicated key generated from at least 32 cryptographically random bytes |
| Explicit local demo | `false` | `true` | Unset/empty; production signing credentials are rejected |

Booleans accept only `true`, `false`, `1`, or `0` (case-insensitive). Invalid
values, conflicting modes, missing/short/placeholder/repeated signing keys fail
startup before the API binds. Key checks cannot prove randomness: use a secure
generator, not a memorable password. `serve.py --demo` supplies demo defaults but
rejects contradictory environment settings. Hot reload and multiple workers do
not disable authentication. Programmatic tests must explicitly pass
`auth_enabled=False, dev_mode=True, jwt_secret=""` when they need a demo client.

### Trusted operator provisioning (operational MVP)

This is **not password login, SSO, or an account/tenant directory**. A trusted
local operator holding the signing key assigns an identity and role. There is no
HTTP token issuer: `/auth/token` is absent in both modes. Keep the key outside
source control and frontend configuration; protect the deployment environment
and Docker administrative access.

For a first local production-mode setup, choose a protected directory outside
the checkout and two new filenames. The following PowerShell example creates a
random key directly into a restricted file, loads it without displaying it, and
issues a viewer token. Run from `forecasting-product`:

```powershell
# Create/select your protected directory first. Never reuse an existing output file.
$env:PRIVATE_KEY_PATH = "$HOME\forecast-private\signing-key"
$env:PRIVATE_TOKEN_PATH = "$HOME\forecast-private\analyst.jwt"
python -c "import os,secrets; from pathlib import Path; from src.auth.issuer import write_private_token; write_private_token(Path(os.environ['PRIVATE_KEY_PATH']), secrets.token_urlsafe(48))"
$env:JWT_SECRET_KEY = (Get-Content -Raw -LiteralPath $env:PRIVATE_KEY_PATH).Trim()
$env:AUTH_ENABLED = "true"
$env:API_DEV_MODE = "false"
python scripts/issue_token.py --user-id analyst --role viewer --expiry-hours 1 --output $env:PRIVATE_TOKEN_PATH
python scripts/serve.py --host 127.0.0.1 --port 8000
```

In deployments, inject the same dedicated key through the existing protected
secret-management environment instead. On POSIX, the equivalent issuer command
is `python scripts/issue_token.py --user-id analyst --role viewer --expiry-hours 1
--output /protected/new-analyst.jwt`, with `JWT_SECRET_KEY` already injected.
Do not pass a signing key or token as a command argument, URL, or shell-history
literal, and do not enable shell tracing. The API and issuer must use the same
key. There is no default secret.

The issuer accepts only the five role-enum values, defaults to a one-hour
lifetime, and allows integer lifetimes from 1 to 24 hours. It writes only a new
file: POSIX mode 0600, or a protected current-user-only Windows DACL established
atomically and verified before credential writing. Existing files, symlinks,
stdout, device paths, alternate streams, or unavailable secure output fail;
Windows does not rely on `chmod`. A failed write may leave an empty or private
partial file; inspect it locally and choose a new output path. Never publish
credential files, even as test artifacts.

### Browser sign-in and deployment

Open the frontend and paste only the token from the private file into its
password-style **Access token** field. Share it only through a trusted private
channel. The browser discovers the public `/auth/status` mode, then verifies
identity, role, permissions and expiry through authenticated `/auth/me` before
signing in. Unavailable/malformed discovery disables sign-in; it never becomes a
demo. Tokens remain only in the current tab's memory, not local/session storage,
URLs, query caches or telemetry. Logout, expiry, 401 and page reload clear the
session and protected cache. Old localStorage credentials are deleted, never
restored. Demo-only role selection is a simulation, not identity assurance.

Compose binds both services to loopback. For non-local production traffic, use
HTTPS with a same-origin reverse proxy: expose the UI and proxy an `/api/` prefix
to the API, removing the prefix upstream. Set the browser-facing
`NEXT_PUBLIC_API_URL` (for example `https://forecast.example/api`) **at frontend
build time** and rebuild; Compose passes it as a build argument. Do not put
signing keys in any `NEXT_PUBLIC_*` setting. A custom cross-origin factory
deployment must explicitly configure its allowed `cors_origins`; CORS is not an
authentication substitute. `NEXTAUTH_SECRET` and `NEXTAUTH_URL` are unused.

### Expiry, inactivity and rotation

These are stateless HS256 tokens with fixed issuer/audience and validated identity,
role, activity and time claims. Missing, invalid, expired or tampered tokens get
401; an inactive signed identity or insufficient role gets 403. Old tokens without
the required claims must be reissued. A signed `is_active=false` claim is denied,
but there is **no live account lookup or immediate per-user revocation**.
Client logout does not revoke a copied token. Use short expiry; for emergency
revocation rotate the signing key on every API worker/replica, restart them, and
reissue tokens. Rolling mixed-key replicas can temporarily disagree, so coordinate
rotation. Operators must manage key protection and user/role assignment outside
this MVP.

### Role Permissions

| Action | admin | data_scientist | planner | manager | viewer |
|--------|-------|----------------|---------|---------|--------|
| View forecasts/metrics | Y | Y | Y | Y | Y |
| Create overrides | Y | Y | Y | - | - |
| Delete overrides | Y | Y | - | - | - |
| Approve overrides | Y | - | - | Y | - |
| Run backtest/pipeline | Y | Y | - | - | - |
| Promote model / modify config | Y | Y | - | - | - |
| View audit log | Y | Y | - | Y | - |
| Manage users (permission only; no user registry) | Y | - | - | - | - |

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
