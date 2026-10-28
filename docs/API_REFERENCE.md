# API Reference

The Forecasting Platform exposes a REST API built with FastAPI. Interactive docs are available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the server is running.

**Base URL:** `http://localhost:8000`

---

## System

### `GET /health`

Readiness probe — returns dependency health checks.

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "checks": {
    "data_dir": true,
    "metrics_dir": true
  }
}
```

### `GET /auth/status`

Public, nonsensitive mode discovery. Returns `{"mode":"production","auth_required":true}`
or `{"mode":"demo","auth_required":false}` only after valid startup configuration.

### `GET /auth/me`

Authenticated identity verification. Returns `mode`, `username`, `email`, `role`,
server-mapped `permissions`, and integer Unix `expires_at` (`null` in explicit
demo). It never returns a token or key. Caller-supplied production role/identity
fields are not accepted.

The former `POST /auth/token` route is absent, including in demo mode.

### `GET /audit`

Query the audit log. **Requires** `VIEW_AUDIT_LOG` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | query, optional | Filter by action type |
| `resource_type` | query, optional | Filter by resource type |
| `limit` | query, default `100` | Max results (1–1000) |

---

## Forecasts

### `GET /forecast/{lob}`

Return an exact committed run or deterministic latest for the LOB. New runs take
precedence over explicitly labelled legacy files; failed/partial runs are invisible.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | path, required | Line of business name |
| `series_id` | query, optional | Filter to a single series |
| `horizon` | query, optional | Limit output to first N periods |
| `run_id` | query, optional | Exact server UUID32; wrong LOB/type or missing commit returns 404 |
| `legacy` | query, default false | Explicit old-file adapter; cannot be combined with a run ID |

**Response:** `ForecastResponse`

```json
{
  "lob": "retail",
  "series_count": 150,
  "forecast_origin": "2026-03-15",
  "points": [
    {
      "series_id": "sku_001",
      "week": "2026-03-22",
      "forecast": 1250.5,
      "model": "lgbm_direct",
      "forecast_p10": 980.0,
      "forecast_p50": 1250.5,
      "forecast_p90": 1520.0
    }
  ]
}
```

### `GET /forecast/{lob}/{series_id}`

Return the latest or selected `run_id` for a single series within a LOB.
Responses include `run_id`, `artifact_source`, `config_hash` and `input_data_hash`;
legacy identity/hashes are null. Nonfinite numeric transport values are JSON null,
not zero; stored artifact bytes and scientific metric formulas are unchanged.

---

## Metrics

### `GET /metrics/leaderboard/{lob}`

Return a committed run's stored ranking and scoring provenance, or a separately
labelled current analysis of unbundled raw metrics. Stored history is never
re-scored or re-sorted using current formulas.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | path, required | LOB name |
| `run_type` | query, default `"backtest"` | `"backtest"` or `"live"` |
| `run_id` | query, optional | Exact committed backtest UUID32; otherwise latest committed run for the LOB |

**Response:** `LeaderboardResponse`

```json
{
  "lob": "retail",
  "run_type": "backtest",
  "entries": [
    {
      "model": "lgbm_direct",
      "wmape": 0.142,
      "normalized_bias": -0.003,
      "rank": 1,
      "n_series": 150
    }
  ]
}
```

Responses additionally include `run_id`, `artifact_source` and
`scoring_provenance`: `source` (`stored`/`computed`), `status`
(`recorded`/`unknown`), metadata version, policy, primary/secondary metric and
competition scope. Missing historical metadata stays unknown. Entries retain
stored `rank`, add `scores`, `primary_score`, `group_key` and `competition_scope`,
and keep `wmape`/`normalized_bias` compatibility fields nullable. For a stored
macro-WMAPE ranking, an absent volume-WMAPE value is null, never substituted with
the macro score. Multi-cohort results have no inferred global winner.

### `GET /metrics/drift/{lob}`

Drift alerts for all series in a LOB. Runs `ForecastDriftDetector` on accuracy, bias, and volume signals.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | path, required | LOB name |
| `run_type` | query, default `"backtest"` | `"backtest"` or `"live"` |
| `baseline_weeks` | query, default `26` | Baseline window (≥4) |
| `recent_weeks` | query, default `8` | Recent window (≥2) |

**Response:** `DriftResponse`

```json
{
  "lob": "retail",
  "n_critical": 2,
  "n_warning": 8,
  "alerts": [
    {
      "series_id": "sku_001",
      "metric": "accuracy",
      "severity": "critical",
      "current_value": 0.35,
      "baseline_value": 0.15,
      "message": "WMAPE increased by 133%"
    }
  ]
}
```

---

## Analytics

### `POST /analyze`

Upload a CSV/Parquet file for automated schema detection, forecastability scoring, hierarchy detection, and config recommendation. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | CSV or Parquet upload |
| `lob_name` | query, default `"analyzed"` | Name for this analysis |
| `llm_enabled` | query, default `false` | Use Claude for narrative interpretation |

**Response:** `AnalysisResponse` — includes `recommended_config_yaml`, `forecastability_distribution`, `demand_classes`, `detected_hierarchies`, and optionally `llm_narrative`.

### `GET /metrics/{lob}/fva`

Compute Forecast Value Add (FVA) cascade — shows which model layer adds or destroys value. **Requires** `VIEW_METRICS` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | path, required | LOB name |
| `run_type` | query, default `"backtest"` | `"backtest"` or `"live"` |

### `GET /metrics/{lob}/calibration`

Compute prediction interval calibration report — compares nominal vs. empirical coverage at 50% and 80% intervals. **Requires** `VIEW_METRICS` permission.

### `POST /metrics/{lob}/shap`

Compute SHAP feature attribution for tree-based models. Accepts an optional file upload; otherwise loads from data directory. **Requires** `VIEW_METRICS` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | path, required | LOB name |
| `file` | body, optional | Actuals CSV/Parquet |
| `model_name` | query, default `"lgbm_direct"` | Model to explain |
| `season_length` | query, default `52` | Season length |
| `top_k` | query, default `10` | Number of top features |

### `POST /forecast/decompose`

Run STL decomposition on historical + forecast data. Returns trend, seasonal, and residual components plus per-series narratives.

| Parameter | Type | Description |
|-----------|------|-------------|
| `history_file` | body, required | Historical actuals |
| `forecast_file` | body, required | Forecast data |
| `id_col` | query, default `"series_id"` | Series ID column |
| `time_col` | query, default `"week"` | Time column |
| `target_col` | query, default `"quantity"` | Target column (actuals) |
| `value_col` | query, default `"forecast"` | Value column (forecast) |
| `season_length` | query, default `52` | Season length |

### `POST /forecast/compare`

Compare model forecast against an external/uploaded forecast. Returns per-series deltas and summary statistics.

### `POST /forecast/constrain`

Apply capacity and budget constraints to forecast. Supports per-series caps, aggregate caps, and proportional redistribution. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | Forecast CSV/Parquet |
| `min_demand` | query, default `0.0` | Floor (non-negativity) |
| `max_capacity` | query, optional | Per-series-per-period cap |
| `aggregate_max` | query, optional | Total cap across all series |
| `proportional` | query, default `true` | Use proportional redistribution |

---

## Series

### `GET /series/{lob}`

List all series with SBC demand classification — ADI (Average Demand Interval), CV² (coefficient of variation squared), demand class (smooth/erratic/intermittent/lumpy), and observation count.

### `GET /series/{lob}/{series_id}/history`

Return raw time series data points for a single series.

### `POST /series/breaks`

Detect structural breaks in uploaded or server-side time series data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, optional | CSV/Parquet upload |
| `lob` | query, optional | LOB name (loads from data directory) |
| `method` | query, default `"cusum"` | `"cusum"` or `"pelt"` |
| `penalty` | query, default `3.0` | PELT penalty (higher = fewer breaks) |
| `min_segment_length` | query, default `13` | Minimum periods between breaks |
| `max_breakpoints` | query, default `5` | Maximum breakpoints to detect |

### `POST /series/cleansing-audit`

Run outlier detection and stockout imputation audit on uploaded data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, optional | CSV/Parquet upload |
| `lob` | query, optional | LOB name |
| `outlier_method` | query, default `"iqr"` | `"iqr"` or `"zscore"` |
| `iqr_multiplier` | query, default `1.5` | IQR fence multiplier |
| `zscore_threshold` | query, default `3.0` | Z-score threshold |

### `POST /series/regressor-screen`

Screen external regressors for quality (variance, correlation, mutual information).

---

## Pipeline

### `POST /pipeline/backtest`

Run the backtest pipeline on uploaded data. Returns champion model and leaderboard. **Requires** `RUN_BACKTEST` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | Actuals CSV/Parquet |
| `config_file` | body, optional | YAML config file |
| `lob` | query, default `"default"` | LOB name |

### `POST /pipeline/forecast`

Run the forecast pipeline. Fits the champion model on full history and generates horizon forecasts. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | Actuals CSV/Parquet |
| `config_file` | body, optional | YAML config file |
| `lob` | query, default `"default"` | LOB name |
| `model_id` | query, optional | Specific model to use |
| `horizon` | query, optional | Explicit override; otherwise YAML then platform default |

Both operations return the committed `run_id`, `run_type`, safe relative `links`
and the existing required `execution` settings. Pre-commit errors do not publish.
A post-commit confirmation/response error is HTTP 409 with `outcome=committed`
or `uncertain` and its run ID; inspect it rather than assuming rollback or retry
idempotency.

### Exact-run read API

| Endpoint | Purpose | Permission |
|----------|---------|------------|
| `GET /pipeline/runs?lob=...&run_type=forecast` | Complete bundles ordered by attempt start + ID; limit 1-200 | `VIEW_METRICS` |
| `GET /pipeline/runs/{lob}/{run_id}` | Verified safe manifest and links | `VIEW_METRICS` |
| `GET /pipeline/runs/{lob}/{run_id}/config` | Canonical saved settings and config identity | `VIEW_METRICS` |
| `GET /pipeline/runs/{lob}/{run_id}/input` | Raw/prepared input hashes, schema, stages and history link | `VIEW_FORECASTS` |

Forecast/series-history selectors, leaderboard/FVA/calibration/validation and
governance exports accept exact run scope. History also labels `input` versus
`prepared` stage. Forecast-actual BI comparison can explicitly select a later
`actuals_run_id`; otherwise it uses the selected forecast's input, not a quietly
substituted latest dataset. Exports are unique derived files, not a cross-store
run transaction. Invalid UUID/path selectors return 400, unauthenticated reads
401, wrong roles 403, missing/cross-type runs 404, and corrupt/unknown committed
schemas explicit integrity errors. LOB is a storage namespace, not tenant RLS.

### `POST /pipeline/analyze-multi-file`

Upload multiple files for auto-classification (actuals, dimensions, regressors) and merging. **Requires** `RUN_PIPELINE` permission.

### `GET /pipeline/manifests`

List verified bundles first and legacy sidecars separately labelled unknown.
Legacy entries do not receive invented run IDs. **Requires** `VIEW_METRICS`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | query, optional | Filter by LOB |
| `limit` | query, default `20` | Max results (1–100) |

### `GET /pipeline/costs`

Get cost tracking data from pipeline manifests (timing, seconds per series). **Requires** `VIEW_METRICS` permission.

---

## Hierarchy

### `POST /hierarchy/build`

Build a hierarchy tree from uploaded data. Returns level stats, node counts, summing matrix sample, and full tree structure for visualization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | CSV/Parquet with hierarchy columns |
| `levels` | query, required | Comma-separated hierarchy levels, root to leaf |
| `id_column` | query, default `"series_id"` | Leaf-level ID column |
| `name` | query, default `"product"` | Hierarchy name |

### `POST /hierarchy/aggregate`

Aggregate leaf-level data to a target hierarchy level.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | CSV/Parquet |
| `levels` | query, required | Comma-separated hierarchy levels |
| `target_level` | query, required | Level to aggregate to |
| `value_columns` | query, default `"quantity"` | Columns to aggregate |
| `agg` | query, default `"sum"` | `"sum"` or `"mean"` |
| `top_n` | query, default `10` | Return top N nodes by total value |

### `POST /hierarchy/reconcile`

Run hierarchical reconciliation and return before/after comparison. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | body, required | Forecast CSV/Parquet |
| `levels` | query, required | Comma-separated hierarchy levels |
| `method` | query, default `"bottom_up"` | `bottom_up`, `top_down`, `ols`, `wls`, `mint` |
| `value_columns` | query, default `"forecast"` | Columns to reconcile |

---

## SKU Mapping

### `POST /sku-mapping/phase1`

Run Phase 1 SKU mapping — attribute + naming matching to find predecessor products. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_master` | body, required | Product master CSV |
| `launch_window_days` | query, default `180` | Window for new product candidates |
| `min_base_similarity` | query, default `0.70` | Minimum similarity threshold |
| `min_confidence` | query, default `"Low"` | `"Low"`, `"Medium"`, or `"High"` |

### `POST /sku-mapping/phase2`

Run Phase 2 SKU mapping — attribute + naming + demand curve fitting. **Requires** `RUN_PIPELINE` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_master` | body, required | Product master CSV |
| `sales_history` | body, optional | Sales history CSV for curve fitting |
| `window_weeks` | query, default `13` | Curve fitting window |

---

## Overrides

### `GET /overrides`

List all planner overrides, optionally filtered by SKU. **Requires** `VIEW_FORECASTS` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `old_sku` | query, optional | Filter by predecessor SKU |
| `new_sku` | query, optional | Filter by successor SKU |

### `POST /overrides`

Create a new planner override. **Requires** `CREATE_OVERRIDE` permission.

**Request body:**

```json
{
  "old_sku": "SKU_OLD_001",
  "new_sku": "SKU_NEW_001",
  "proportion": 0.8,
  "scenario": "manual",
  "ramp_shape": "linear",
  "effective_date": "2026-04-01",
  "notes": "Product transition approved"
}
```

### `PUT /overrides/{override_id}`

Update an existing override. **Requires** `CREATE_OVERRIDE` permission.

This is a patch of the existing ID, not a delete/create replacement. Supply at
least one of `proportion`, `scenario`, `ramp_shape`, `effective_date`, or `notes`.
Omitted fields remain unchanged. Only `effective_date` and `notes` accept explicit
`null` to clear a value. `proportion` must be finite and in `[0, 1]`, including
zero; dates use valid `YYYY-MM-DD` strings. Scenario and ramp shape remain strings,
not a new enumeration.

```json
{"proportion": 0, "notes": null}
```

The response is `{"override_id": "<same ID>", "status": "updated"}`. SKU keys,
creator/creation time, status and approval metadata remain exactly as stored.
Preserving approval metadata does **not** mean the edited decision was reapproved;
approval lifecycle/history and stale-edit detection remain separate work.

Unknown/immutable fields and malformed typed values return **422**. An empty
patch or invalid date string returns **400**; a missing ID returns **404**.
Detected local file-store contention returns **409**, including a read that
overlaps another file operation; retry explicitly. Unexpected storage errors
return a logged **500** without exposing private paths. No mutation is retried
automatically. Independent-field patches retain both changes after a busy retry;
same-field changes are serialized last-writer-wins, not ETag-protected edits.

### `DELETE /overrides/{override_id}`

Delete an override. **Requires** `DELETE_OVERRIDE` permission. A missing ID
returns **404** on every backend rather than claiming a successful deletion.

These CRUD operations do not modify immutable forecast/run artifacts. See
[local persistence limits](DEPLOYMENT.md#mutable-local-overrides).

---

## Governance

### `GET /governance/model-cards`

List all registered model cards.

### `GET /governance/model-cards/{model_name}`

Get a specific model card by name.

### `GET /governance/lineage`

Get forecast lineage history.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lob` | query, optional | Filter by LOB |
| `model_id` | query, optional | Filter by model |

### `POST /governance/export/{report_type}`

Export BI report as Parquet. **Requires** `VIEW_METRICS` permission.

| Parameter | Type | Description |
|-----------|------|-------------|
| `report_type` | path, required | `"forecast-actual"`, `"leaderboard"`, or `"bias-report"` |
| `lob` | query, required | LOB name |
| `run_type` | query, default `"backtest"` | `"backtest"` or `"live"` |
| `model_id` | query, optional | Filter by model (for bias report) |

---

## AI Features

All AI endpoints require `ANTHROPIC_API_KEY` environment variable. If not set, Claude features return graceful fallback responses.

### `POST /ai/explain`

Answer a natural-language question about a specific series forecast. **Requires** `VIEW_FORECASTS` permission.

**Request body:**

```json
{
  "series_id": "sku_001",
  "question": "Why did this forecast increase 30%?",
  "lob": "retail"
}
```

**Response:** `NLQueryResponse`

```json
{
  "answer": "The forecast for sku_001 increased due to...",
  "supporting_data": {"trend_direction": "increasing", "seasonal_peak": true},
  "confidence": "medium",
  "sources_used": ["history_stats", "forecast_summary", "decomposition"]
}
```

### `POST /ai/triage`

Triage drift alerts by business impact with suggested actions. **Requires** `VIEW_METRICS` permission.

**Request body:**

```json
{
  "lob": "retail",
  "run_type": "backtest",
  "severity_filter": "critical",
  "max_alerts": 50
}
```

**Response:** `TriageResponse` — ranked alerts with `business_impact_score`, `suggested_action`, and `reasoning`.

### `POST /ai/recommend-config`

Recommend configuration changes based on backtest performance. **Requires** `RUN_PIPELINE` permission.

**Request body:**

```json
{
  "lob": "retail",
  "run_type": "backtest"
}
```

**Response:** `ConfigTuneResponse` — list of recommendations, each with `field_path`, `current_value`, `recommended_value`, `reasoning`, `expected_impact`, and `risk`.

### `POST /ai/commentary`

Generate executive forecast commentary for S&OP meetings. **Requires** `VIEW_METRICS` permission.

**Request body:**

```json
{
  "lob": "retail",
  "run_type": "backtest",
  "period_start": "2026-01-01",
  "period_end": "2026-03-31"
}
```

**Response:** `CommentaryResponse` — `executive_summary`, `key_metrics[]`, `exceptions[]`, `action_items[]`.

---

## Authentication & RBAC

The factory and shipped launchers require authentication by default. All business
endpoints use verified bearer JWTs; `/health`, `/auth/status` and API documentation
are public and expose no signing configuration. `/auth/token` does not exist.
`/auth/me` returns the server-verified identity and role, never a token.

Use [trusted operator provisioning](DEPLOYMENT.md#authentication-setup), not
caller-selected production roles. Missing/invalid/expired/tampered tokens return
401. Inactive signed identities and insufficient permissions return 403.
`AUTH_ENABLED=false` is allowed only with explicit `API_DEV_MODE=true` and no
signing key; demo headers simulate roles but provide no identity assurance.

### Roles and Permissions

| Role | Permissions |
|------|-------------|
| `admin` | All permissions |
| `manager` | VIEW_FORECASTS, VIEW_METRICS, APPROVE_OVERRIDE, VIEW_AUDIT_LOG |
| `data_scientist` | VIEW_FORECASTS, VIEW_METRICS, VIEW_AUDIT_LOG, CREATE_OVERRIDE, DELETE_OVERRIDE, RUN_BACKTEST, RUN_PIPELINE, PROMOTE_MODEL, MODIFY_CONFIG |
| `planner` | VIEW_FORECASTS, VIEW_METRICS, CREATE_OVERRIDE |
| `viewer` | VIEW_FORECASTS, VIEW_METRICS |

### Rate Limiting

100 requests per 60 seconds per client IP (configurable via `API_RATE_LIMIT` env var). Returns HTTP 429 when exceeded.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_DATA_DIR` | Root data directory | `data/` |
| `API_METRICS_DIR` | Metrics store directory | `data/metrics/` |
| `API_VERSION` | Version in `/health` | `1.0.0` |
| `API_RATE_LIMIT` | Requests per minute per IP | `100` |
| `AUTH_ENABLED` | Strict boolean; false requires explicit demo | `true` |
| `API_DEV_MODE` | Strict boolean; explicit unauthenticated simulation | `false` |
| `JWT_SECRET_KEY` | Dedicated random production key; missing/weak values fail startup | unset |
| `ANTHROPIC_API_KEY` | Claude API key for AI features | *(optional)* |
