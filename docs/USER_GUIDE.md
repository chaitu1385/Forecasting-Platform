# User Guide

An end-to-end guide for using the Forecasting Product — from choosing models to interpreting results.

---

## Workflow Overview

```
1. Prepare data (CSV/Parquet)
2. Upload via the Next.js frontend or run pipeline script
3. Backtest: evaluate candidate models
4. Review results: leaderboard, FVA, champion map
5. Forecast: generate production forecasts
6. Monitor: drift alerts, data quality, cost
```

---

## Model Selection

The platform includes 18 registered models across 6 families. Use this decision tree:

### Quick Reference

| Family | Models | Best For | Data Needs | Speed |
|--------|--------|----------|-----------|-------|
| **Naive** | `naive_seasonal` | Baseline benchmark | Any series length | Instant |
| **Statistical** | `auto_arima`, `auto_ets`, `auto_theta`, `mstl` | Stable demand with clear trend/seasonality | 52+ weeks (weekly) | Fast |
| **ML** | `lgbm_direct`, `xgboost_direct` | Complex patterns, external regressors | 52+ weeks + features | Fast |
| **Neural** | `nbeats`, `nhits`, `tft` | Large datasets, complex interactions | 100+ series, 100+ weeks | Slow |
| **Foundation** | `chronos`, `timegpt` | Cold start, limited history | Any (zero-shot) | Medium |
| **Intermittent** | `croston`, `croston_sba`, `tsb` | Sparse/lumpy demand | 10+ non-zero observations | Fast |

### Decision Tree

1. **Is demand intermittent (many zeros)?**
   - Yes -> Enable sparse detection (`sparse_detection: true`). The platform auto-routes sparse series to intermittent models.
   - No -> Continue.

2. **Do you have external features (promotions, holidays, price)?**
   - Yes -> Use ML models (`lgbm_direct`, `xgboost_direct`). They automatically pick up feature columns.
   - Statistical and naive models silently ignore external features.

3. **How much history do you have?**
   - < 52 weeks -> Foundation models (`chronos`), or `naive_seasonal` with cyclic fallback.
   - 52-104 weeks -> Statistical (`auto_ets`, `auto_arima`) or ML (`lgbm_direct`).
   - 104+ weeks -> All models viable. Use backtest to compare.

4. **How many series?**
   - < 100 -> Any model. Neural models may overfit.
   - 100-10,000 -> ML models shine. Run backtest with 3-4 candidates.
   - 10,000+ -> Enable `batch_size` and `n_workers` for parallelism.

### Recommended Starting Configuration

```yaml
forecast:
  forecasters: [naive_seasonal, auto_ets, lgbm_direct]
  sparse_detection: true
  intermittent_forecasters: [croston_sba, tsb]
```

This gives you a baseline (naive), a strong statistical model (ETS), and an ML model (LightGBM). The backtest will pick the best per your data.

---

## Running a Backtest

Backtesting evaluates models using walk-forward cross-validation: train on past data, forecast the next N weeks, measure accuracy, slide the window forward.

### Configuration

```yaml
backtest:
  n_folds: 3                    # number of CV folds
  val_weeks: 13                 # validation window (one quarter)
  gap_weeks: 0                  # gap between train and test (for delayed data)
  champion_granularity: lob     # pick one champion for the whole LOB
  primary_metric: wmape          # rank by Weighted MAPE
  selection_strategy: champion   # "champion" (single best) or "weighted_ensemble"
```

### Run

```bash
python forecasting-product/scripts/run_backtest.py \
  --config configs/platform_config.yaml --lob retail
```

### Champion Selection Strategies

- **`champion`** — Picks the single model with lowest WMAPE across all folds. Simple, interpretable.
- **`weighted_ensemble`** — Blends top models using inverse-WMAPE weights. Often 1-3% more accurate, but harder to explain.

### Granularity Options

| Granularity | What It Does | When to Use |
|-------------|-------------|-------------|
| `lob` | One champion for all series in the LOB | Default — simple and reliable |
| `product_group` | Different champion per product group | When product categories have very different demand patterns |
| `series` | Different champion per series | Large datasets (1000+ series) where per-series tuning matters |

---

## Interpreting Backtest Results

### Metrics

| Metric | Formula | What It Means | Good Range |
|--------|---------|-------------|-----------|
| **WMAPE** | sum(\|error\|) / sum(\|actual\|) | Overall accuracy weighted by volume | < 20% is good, < 10% is excellent |
| **Normalized Bias** | sum(error) / sum(\|actual\|) | Systematic over/under-forecast | -5% to +5% is acceptable |
| **MAE** | mean(\|error\|) | Average error in original units | Domain-dependent |
| **RMSE** | sqrt(mean(error²)) | Penalizes large errors more | Domain-dependent |

### Leaderboard

The leaderboard ranks models by the primary metric (WMAPE by default), aggregated across all folds and series.

```
Rank | Model          | WMAPE  | Bias   | MAE
-----|----------------|--------|--------|-----
1    | lgbm_direct    | 12.3%  | -1.2%  | 234
2    | auto_ets       | 14.1%  | +0.8%  | 267
3    | naive_seasonal | 18.7%  | +2.1%  | 312
```

### Forecast Value Added (FVA)

FVA measures how much accuracy each model layer contributes compared to the naive baseline.

**Layers:**

| Layer | Models | Purpose |
|-------|--------|---------|
| L1: Naive | `naive_seasonal` | Baseline — always computed |
| L2: Statistical | `auto_arima`, `auto_ets`, `croston`, etc. | Best statistical per series |
| L3: ML | `lgbm_direct`, `xgboost_direct` | Best ML per series |
| L4: Override | Planner-adjusted | If planner override exists |

**FVA Classification:**

| Classification | WMAPE Change | Meaning |
|---------------|-------------|---------|
| `ADDS_VALUE` | > 2 percentage points improvement | Model is helping |
| `NEUTRAL` | Within ±2 pp | No meaningful difference |
| `DESTROYS_VALUE` | > 2 pp degradation | Model is hurting — consider removing |

**Reading the FVA cascade chart (Backtest Results page):**
- Bars show WMAPE at each layer
- Green arrows = layers that add value
- Red arrows = layers that destroy value
- If L3 (ML) destroys value vs L2 (Statistical), the ML model isn't learning anything useful beyond what ETS/ARIMA already captures

---

## Generating Forecasts

After backtesting, generate production forecasts:

```bash
python forecasting-product/scripts/run_forecast.py \
  --config configs/platform_config.yaml \
  --lob retail \
  --champion lgbm_direct
```

### Output

- **Run ID** — Fresh server-generated UUID32, also returned by the API and printed by the CLI
- **Immutable bundle** — Forecast, normalized input, prepared series, safe configuration, validation and provenance under one run directory

See [DATA_FORMAT.md](DATA_FORMAT.md) for complete output schemas.

### Retrieve a Previous Run

The Forecast Viewer offers latest and individual saved run IDs.
Selecting a prior run changes the actual forecast values, history and stored
configuration; it does not merely relabel latest. The page shows input/config
hashes and an expandable data table. Switching LOB clears the old selection.
Legacy files remain an explicit option with unknown lineage.

Programmatically, keep the `run_id` and relative `links` in a successful pipeline
response. Use `/pipeline/runs` to list completed attempts,
`/pipeline/runs/{lob}/{run_id}/config` for safe saved settings, and
`/forecast/{lob}?run_id=<UUID32>` for exact output. History and metric queries
also support run scope. These endpoints require the existing authenticated read
permissions; a LOB name is not tenant isolation.

Latest is ordered by attempt start time, not completion time. An ambiguous
HTTP/commit outcome should be resolved using the returned ID before starting
another attempt; a retry is a new run, not exactly-once recovery. For custom
storage roots, backup and inspection see
[DEPLOYMENT.md](DEPLOYMENT.md#immutable-local-runs-and-recovery).

---

## Hierarchical Forecasting

When you have a hierarchy (e.g., SKU -> Category -> Total), hierarchical reconciliation ensures forecasts sum correctly at every level.

### Setup

```yaml
hierarchies:
  - name: product
    levels: [total, category, sku]
    reconciliation:
      method: mint     # bottom_up | top_down | ols | wls | mint
```

Your dimension data must include the hierarchy columns (e.g., `category` for each `sku`).

### Reconciliation Methods

| Method | When to Use |
|--------|-------------|
| `bottom_up` | Leaf (SKU) forecasts are accurate; upper levels are just sums |
| `top_down` | Top-level forecast is accurate; disaggregate by historical proportions |
| `ols` | No prior knowledge about which level is most accurate |
| `wls` | Some levels have more data (weights by leaf count or residual variance) |
| `mint` | Best overall accuracy — uses covariance of forecast errors (recommended) |

---

## External Regressors

Add external features (promotions, holidays, price) to improve ML model accuracy.

### End-to-End Example

1. **Prepare features file** (see [DATA_FORMAT.md](DATA_FORMAT.md) for schema):

```csv
week,store_id,promotion_flag,holiday_flag,price_index
2024-01-07,1,1,0,1.05
...
```

2. **Configure:**

```yaml
forecast:
  external_regressors:
    enabled: true
    feature_columns: [promotion_flag, holiday_flag, price_index]
    future_features_path: data/future_features.parquet
    screen:
      enabled: true              # auto-drop low-quality features
      variance_threshold: 1.0e-6
      correlation_threshold: 0.95
```

3. **Run pipeline** — ML models (`lgbm_direct`, `xgboost_direct`) automatically use the features. Statistical models ignore them.

### Regressor Screening

When `screen.enabled: true`, the platform automatically drops:
- **Near-zero variance** features (threshold: 1e-6)
- **Highly correlated** feature pairs (threshold: 0.95, keeps first)
- **Low mutual information** features (optional, `mi_enabled: true`)

Dropped features are logged in the pipeline manifest.

---

## AI Features

Four Claude-powered endpoints provide AI-native analysis. Requires `ANTHROPIC_API_KEY` environment variable.

### Natural Language Query (`POST /ai/explain`)

Ask questions about forecasts in plain English:

```json
{
  "lob": "retail",
  "series_id": "STORE_001",
  "question": "Why did sales drop last week?"
}
```

Returns an answer with supporting data, confidence level, and sources used.

### Anomaly Triage (`POST /ai/triage`)

Prioritizes drift alerts by business impact:

```json
{
  "lob": "retail",
  "severity_filter": "critical",
  "max_alerts": 50
}
```

Returns ranked alerts with impact scores, suggested actions, and an executive summary.

### Config Recommendation (`POST /ai/recommend-config`)

Analyzes backtest results and recommends config changes:

```json
{ "lob": "retail" }
```

Returns specific recommendations (e.g., "switch from `naive_seasonal` to `lgbm_direct`") with expected impact and risk level.

### Executive Commentary (`POST /ai/commentary`)

Generates S&OP-ready executive summaries:

```json
{ "lob": "retail" }
```

Returns a 3-5 sentence summary, key metrics with trends, exceptions, and action items.

### Graceful Degradation

All AI features work without Claude — they fall back to template-based responses using the same data structures. The UI and downstream consumers don't need to handle a different schema.

---

## Planner Overrides

Demand planners can manually adjust forecasts through the override store.

```python
from src.overrides import OverrideStore

store = OverrideStore()

# Add an override
store.add_override(
    lob="retail",
    old_sku="PROD_A",
    new_sku="PROD_B",
    proportion=0.6,
    scenario="replacement",
    ramp_shape="linear"
)

# Retrieve overrides
overrides = store.get_overrides(lob="retail", week="2024-09-07")
```

The override store uses DuckDB by default, with automatic fallback to Parquet in environments where DuckDB is unavailable (e.g., Microsoft Fabric).

---

## Next.js Frontend

The frontend lives at `forecasting-product/frontend/`. It connects to the FastAPI backend over REST and provides dark mode, responsive layout, and role-based navigation.

### Getting Started

```bash
# Explicit local synthetic demo (no production signing credentials)
python forecasting-product/scripts/serve.py --demo --port 8000

# In a separate terminal, start the frontend
cd forecasting-product/frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:3000` by default. Set `NEXT_PUBLIC_API_URL` in `.env.local` to point to your API server (defaults to `http://localhost:8000`).

### Pages

| Page | Route | Description |
|------|-------|-------------|
| **Data Onboarding** | `/data-onboarding` | Upload CSV files, auto-classify columns, preview cleansing, generate config |
| **Series Explorer** | `/series-explorer` | SBC classification, structural break detection, cleansing audit, AI Q&A |
| **SKU Transitions** | `/sku-transitions` | New/discontinued SKU mapping, planner overrides, transition visualization |
| **Hierarchy Manager** | `/hierarchy` | Build hierarchy trees, aggregate, reconcile (MinT/OLS/WLS) |
| **Backtest Results** | `/backtest` | Model leaderboard, FVA cascade, champion map, calibration, SHAP |
| **Forecast Viewer** | `/forecast` | Fan charts, decomposition, AI natural language query, comparison, constraints |
| **Platform Health** | `/health` | Pipeline manifests, drift alerts, AI anomaly triage, audit log, cost tracking |
| **S&OP Meeting** | `/sop` | AI commentary, cross-run comparison, model governance, BI export |

### Running the Selected Forecast Plan

After analyzing a file in **Data Onboarding**, the **Pipeline Execution** panel uses
the recommended YAML by default. Optionally upload your own `.yaml` or `.yml` under
**Pipeline Configuration** to replace that recommendation; **Use recommended
configuration** switches back. The browser sends the YAML unchanged for validation
by the API.

Run **Backtest**, then **Forecast** to carry the current backtest champion and its
configured production horizon into forecasting. For example, a Croston winner with
`forecast.horizon_periods: 5` produces five Croston forecast periods per retained
series. **Effective model**, **Horizon (periods)**, **Frequency**, and the expandable
**Effective configuration** show what actually ran. A backtest validation window
(`backtest.val_periods`) is separate from the production forecast horizon.

Standalone forecasting remains available before backtesting: it uses the first
entry in `forecast.forecasters`, not an inferred backtest winner. Without a config,
the platform defaults are `naive_seasonal`, weekly frequency, and 39 periods.
Changing the data, LOB, or configuration clears dependent results; late responses
cannot restore an old winner. A new backtest also clears the previous selection and
blocks forecasting while it is running. If no single champion is selected, the
panel does not silently pick one. Composite selections such as a weighted ensemble
remain valid backtest results, but cannot be reconstructed by sending only a model
name. Their automatic forecast handoff is disabled; choose a single-model plan.
Directly requesting a model that requires configured components (for example
`weighted_ensemble` or `constrained_demand`) returns HTTP 400. Constraints configured
through `forecast.constraints` on a standalone model remain supported.

For API callers, precedence is **explicit query override > valid YAML > platform
defaults**. Omit `horizon` to use `forecast.horizon_periods` (or the legacy
`horizon_weeks` alias); when both YAML keys are present, `horizon_periods` wins.
Both pipeline endpoints return HTTP 400 for malformed YAML, non-mapping/empty
uploads, wrong known-field types, or invalid configuration values before any
pipeline computation or publication. Use `{}` for an explicit defaults-only YAML.
Unknown models return HTTP 400; non-positive or non-integer query horizons return
HTTP 422. An override does not make a malformed config valid. Existing
unknown-key handling and post-validation opt-out/nonblocking policies are unchanged.
Backtest strategies are `champion` and `weighted_ensemble`; the legacy
`best_metric` alias resolves to `champion`. Granularity must be `lob`,
`product_group`, or `series`. Selection metrics must be distinct, registered
metrics (including context-dependent `mase`), and included in the top-level
`metrics` list so they are actually computed before champion selection.

### Authentication

Production `/login` accepts a short-lived access token issued by a trusted local
operator, not a password or a self-selected role. The API verifies identity and
role through `/auth/me` before sign-in. This operational MVP does not provide SSO
or a user directory. Follow [operator provisioning](DEPLOYMENT.md#authentication-setup).

Tokens stay only in the current tab's memory. Logout, expiry, an invalid session
or full reload clears protected data and requires sign-in again; old localStorage
tokens are never restored. Missing API mode discovery disables sign-in instead
of falling back to anonymous access. Username/role selection is shown only in
explicitly unauthenticated demo mode; it is a simulation, not identity assurance.

### Configuration

Key environment variables for `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000    # Backend API URL
```

The public API URL is a build-time setting. Never put signing keys or access
tokens into frontend environment files; `NEXTAUTH_*` settings are unused.

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment instructions.
