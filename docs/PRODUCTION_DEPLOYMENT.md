# Production Deployment Guide

This guide covers deploying the Forecasting Platform to production environments beyond the basic Docker setup in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Container Deployment

### Docker Compose (included)

The repo ships with production-default `docker-compose.yml` at the project root.
First inject `JWT_SECRET_KEY` from protected secret storage and provision a
private operator token as described in [Authentication Setup](DEPLOYMENT.md#authentication-setup).
The image refuses to start without valid signing configuration and does not seed
demo data. Password authentication, SSO and an account directory are not provided.

```bash
docker compose up --build -d
```

| Service | Port | Resources | Healthcheck |
|---------|------|-----------|-------------|
| `api` | 8000 | 2 CPU / 2 GB | `GET /health` |
| `frontend` | 3000 | 0.5 CPU / 512 MB | — |

Host ports bind to loopback. Use an HTTPS same-origin reverse proxy for remote
access; compile its public API URL into the frontend build. The separate
`docker-compose.demo.yml` is only for local synthetic demonstrations and uses a
different volume. Do not mix demo data or credentials into production.

The API image uses digest-pinned `python:3.12.16-slim-bookworm`, `libgomp1`,
and the hash-locked CPU/API profile (no GPU/NCCL installation). The frontend
uses digest-pinned Node.js 24.22.1 / Alpine 3.24 stages, `npm ci`, and a non-root
`nextjs` user. See [supported runtime profiles](DEPLOYMENT.md#supported-runtimes-and-locked-profiles)
for exact package resolution, lifecycle boundaries and regeneration.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_DATA_DIR` | No | Root data path (default: `data/`) |
| `API_METRICS_DIR` | No | Metrics store path (default: `data/metrics/`) |
| `API_VERSION` | No | Version string for `/health` |
| `API_RATE_LIMIT` | No | Requests/min/IP (default: `100`) |
| `ANTHROPIC_API_KEY` | For AI features | Claude API key |
| `NIXTLA_API_KEY` | For TimeGPT | Nixtla API key |
| `JWT_SECRET_KEY` | Production | Dedicated key from at least 32 cryptographically random bytes |
| `AUTH_ENABLED` | Defaults true | Only `true`/`false`/`1`/`0`; false requires explicit demo |
| `API_DEV_MODE` | Defaults false | Must remain false in production |
| `NEXT_PUBLIC_API_URL` | Frontend build | Public API URL; rebuild to change (default: `http://localhost:8000`) |

### Secrets Management

- **Never** bake API keys into images. Pass via environment variables or a secrets manager.
- For Kubernetes, use `Secret` objects mounted as env vars.
- For Docker Compose, inject a protected environment variable. A file-mounted
  Docker secret is not automatically read by the application; use your trusted
  deployment bootstrap to inject it without logging it.
- Never copy the key into frontend settings. Protect Docker daemon access, which
  can inspect container environments. Coordinate key rotation across every
  replica and reissue short-lived tokens; there is no immediate per-user revocation.

---

## Kubernetes Deployment

No K8s manifests are included. Here is a reference deployment:

### API Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forecast-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: forecast-api
  template:
    metadata:
      labels:
        app: forecast-api
    spec:
      containers:
      - name: api
        image: your-registry/forecast-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: API_DATA_DIR
          value: "/data"
        - name: AUTH_ENABLED
          value: "true"
        - name: API_DEV_MODE
          value: "false"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: forecast-secrets
              key: jwt-signing-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: forecast-secrets
              key: anthropic-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: data-volume
          mountPath: /data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: forecast-data-pvc
```

### Service & Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: forecast-api
spec:
  selector:
    app: forecast-api
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: forecast-api
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
  - host: forecast.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: forecast-api
            port:
              number: 80
```

### Resource Sizing

| Component | CPU Request | Memory Request | Notes |
|-----------|-------------|----------------|-------|
| API (serving) | 1 CPU | 1 Gi | Handles REST queries and leaderboards |
| API (pipeline) | 2 CPU | 4 Gi | Backtest/forecast jobs are CPU-intensive |
| Frontend | 0.25 CPU | 256 Mi | Static Next.js |
| Chronos (GPU) | 1 CPU + 1 GPU | 8 Gi | Foundation model inference |

For pipeline workloads, consider running backtest/forecast as Kubernetes Jobs or via an orchestrator (see below) rather than through the API.

---

## Orchestration (Airflow / Dagster)

### Airflow DAG (example)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_backtest(**context):
    import polars as pl
    from forecasting_product.src.config.loader import load_config
    from forecasting_product.src.pipeline.backtest import BacktestPipeline

    config = load_config("/config/platform.yaml")
    actuals = pl.read_parquet("/data/actuals/latest.parquet")
    pipeline = BacktestPipeline(config)
    results = pipeline.run(actuals)
    # Write results for downstream tasks
    results["backtest_results"].write_parquet("/data/backtest/latest.parquet")

def run_forecast(**context):
    import polars as pl
    from forecasting_product.src.config.loader import load_config
    from forecasting_product.src.pipeline.forecast import ForecastPipeline

    config = load_config("/config/platform.yaml")
    actuals = pl.read_parquet("/data/actuals/latest.parquet")
    pipeline = ForecastPipeline(config)
    forecast = pipeline.run(actuals, champion_model="lgbm_direct")

with DAG(
    "weekly_forecast",
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 6 * * MON",    # Every Monday at 6 AM
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    backtest = PythonOperator(task_id="backtest", python_callable=run_backtest)
    forecast = PythonOperator(task_id="forecast", python_callable=run_forecast)
    backtest >> forecast
```

### Dagster (example)

```python
from dagster import asset, define_asset_job, ScheduleDefinition
import polars as pl

@asset
def backtest_results():
    from forecasting_product.src.config.loader import load_config
    from forecasting_product.src.pipeline.backtest import BacktestPipeline

    config = load_config("config.yaml")
    actuals = pl.read_parquet("data/actuals/latest.parquet")
    return BacktestPipeline(config).run(actuals)

@asset
def production_forecast(backtest_results):
    from forecasting_product.src.config.loader import load_config
    from forecasting_product.src.pipeline.forecast import ForecastPipeline

    config = load_config("config.yaml")
    actuals = pl.read_parquet("data/actuals/latest.parquet")
    pipeline = ForecastPipeline(config)
    pipeline.set_conformal_residuals(backtest_results["conformal_residuals"])
    return pipeline.run(actuals, champion_model=backtest_results["champions"])

weekly_job = define_asset_job("weekly_forecast", selection="*")
weekly_schedule = ScheduleDefinition(job=weekly_job, cron_schedule="0 6 * * 1")
```

---

## Monitoring

### Observability Config

The platform has built-in observability via `ObservabilityConfig`:

```yaml
observability:
  log_format: json          # "text" or "json" (use json in production)
  log_level: INFO
  metrics_backend: statsd   # "log" or "statsd"
  statsd_host: localhost
  statsd_port: 8125
  metrics_prefix: forecast_platform
  cost_per_second: 0.001    # For cost tracking in pipeline manifests
  alerts:
    channels: [webhook]
    webhook_url: https://hooks.slack.com/services/...
    min_severity: warning
    webhook_timeout: 10
```

### Key Metrics to Monitor

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| WMAPE by LOB | `/metrics/leaderboard/{lob}` | > 0.25 (weekly) |
| Drift alerts | `/metrics/drift/{lob}` | Any `critical` severity |
| FVA cascade | `/metrics/{lob}/fva` | Negative value-add |
| Pipeline duration | Manifests (`/pipeline/manifests`) | > 2x historical baseline |
| API latency | FastAPI middleware / APM | p95 > 2s |
| Error rate | Structured logs | > 1% of requests |

### Drift Detection Automation

Set up a cron or scheduled task to poll the drift endpoint:

```bash
curl -s http://localhost:8000/metrics/drift/retail | \
  jq '.n_critical' | \
  xargs -I{} test {} -gt 0 && \
  curl -X POST https://hooks.slack.com/services/... \
    -d '{"text": "Forecast drift: critical alerts detected"}'
```

---

## Scaling Considerations

### Parallelism

Configure via `parallelism` in your YAML:

```yaml
parallelism:
  backend: local            # "local" or future distributed backends
  n_workers: -1             # CPU workers (-1 = all cores)
  n_jobs_statsforecast: -1  # StatsForecast parallelism
  num_threads_mlforecast: -1 # ML model parallelism
  batch_size: 0             # 0 = auto batch
  gpu: false                # Enable for Chronos on GPU
```

### Data Volume Guidelines

| Series Count | Recommended Setup |
|-------------|-------------------|
| < 500 | Single container, default config |
| 500–5,000 | 2 CPU / 4 GB, max parallelism |
| 5,000–50,000 | Dedicated pipeline worker pods, API pod separate |
| > 50,000 | Batch by LOB, consider Spark/Ray for preprocessing |

### Storage

- **Metric store:** Hive-partitioned Parquet. Grows ~1 MB per 1,000 series per backtest run.
- **Forecasts:** One Parquet per pipeline run. Clean up old forecasts periodically.
- **Manifests:** Small JSON files. Keep for audit trail.

Use a shared filesystem (NFS, EFS, GCS FUSE) or object storage for multi-pod deployments.
