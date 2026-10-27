# Data Format Specification

This document describes the input data the platform expects, how files are auto-classified, and the output formats produced by the forecast pipeline.

---

## Input: Time Series File

The primary input is a CSV or Parquet file containing historical demand data.

### Required Columns

| Column | Default Name | Type | Description |
|--------|-------------|------|-------------|
| Series ID | `series_id` | String / Categorical | Unique identifier per time series (e.g., SKU, store, product) |
| Time | `week` | Date / Datetime | Period end-date for each observation |
| Target | `quantity` | Numeric (int or float) | Demand value (sales, units, revenue) |

Column names are configurable in YAML:

```yaml
forecast:
  series_id_column: series_id    # or "Store", "sku", etc.
  time_column: week              # or "date", "ds", "Date", etc.
  target_column: quantity        # or "Sales", "demand", "revenue", etc.
```

### Minimal Example

```csv
series_id,week,quantity
STORE_001,2024-01-07,1250
STORE_001,2024-01-14,1340
STORE_001,2024-01-21,1180
STORE_002,2024-01-07,890
STORE_002,2024-01-14,920
STORE_002,2024-01-21,870
```

### Validation Rules

When `data_quality.validation.enabled: true`, the platform checks:

| Check | Level | What It Does |
|-------|-------|-------------|
| Schema | ERROR | Required columns must exist with correct types |
| Duplicates | ERROR | No duplicate `(series_id, time)` pairs allowed |
| Frequency | WARNING | Date gaps must be consistent (weekly = 7 days, monthly ~ 30 days) |
| Non-negative | ERROR | Target values must be >= 0 (configurable via `min_value`) |
| Completeness | WARNING | Series with > `max_missing_pct` missing weeks are flagged |

Set `strict: true` to halt the pipeline on any validation error. Default is `false` (warnings are logged but execution continues).

### Recognized Column Name Patterns

The platform auto-detects columns even if they don't match the configured names:

- **Time columns:** `week`, `date`, `ds`, `time`, `timestamp`, `period`, `day`
- **Target columns:** `quantity`, `sales`, `demand`, `revenue`, `volume`, `units`, `target`, `value`, `amount`, `qty`, `count`
- **ID columns:** Any string/categorical column with moderate cardinality (< 10,000 unique values)

### Supported Frequencies

| Code | Period | Min History | Default Horizon | Season Length |
|------|--------|------------|-----------------|---------------|
| `D` | Daily | 90 days | 90 days | 7 |
| `W` | Weekly | 52 weeks | 39 weeks | 52 |
| `M` | Monthly | 24 months | 12 months | 12 |
| `Q` | Quarterly | 8 quarters | 8 quarters | 4 |

Set frequency in config:
```yaml
forecast:
  frequency: "W"    # "D" | "W" | "M" | "Q"
```

---

## Input: Dimension File (Optional)

A lookup table with attributes for each series. No date column needed.

### Example

```csv
store_id,region,store_type,assortment
1,North,a,basic
2,South,b,extended
3,North,c,basic
```

Used for:
- Building hierarchy levels (e.g., store -> region -> total)
- Enriching series with categorical attributes for ML models

---

## Input: External Regressor File (Optional)

Numeric features aligned to time periods, used by ML models (LightGBM, XGBoost).

### Example

```csv
week,store_id,promotion_flag,holiday_flag,price_index
2024-01-07,1,1,0,1.05
2024-01-14,1,0,0,1.00
2024-01-21,1,0,1,0.95
```

### Requirements

- Must include a time column matching the actuals
- Numeric feature columns only (binary flags, continuous values)
- **Must cover the forecast horizon** — if you're forecasting 39 weeks ahead, the regressor file needs future values for those 39 weeks
- If `series_id` is present, features are series-specific; otherwise they broadcast globally

### Feature Types

Configure in YAML:
```yaml
forecast:
  external_regressors:
    enabled: true
    feature_columns: [promotion_flag, holiday_flag, price_index]
    future_features_path: data/future_features.parquet
    feature_types:
      promotion_flag: known_ahead     # plannable — safe for forecasting
      holiday_flag: known_ahead
      price_index: contemporaneous    # requires explicit future values
```

- **`known_ahead`** — Features you know in advance (holidays, scheduled promotions). Default.
- **`contemporaneous`** — Features observed in real time (foot traffic, weather actuals). Must provide explicit future values or they'll be dropped at prediction time.

### Holiday Calendar Generation

Generate holiday flags automatically (requires `pip install holidays`):

```python
from src.data.regressors import generate_holiday_calendar

holidays_df = generate_holiday_calendar(
    country="US",
    start_date="2023-01-01",
    end_date="2025-12-31",
    frequency="W"
)
# Returns: [week, holiday_flag, holiday_count]
```

---

## Multi-File Upload

When uploading multiple files via the Data Onboarding page, the platform auto-classifies each file into a role.

### Classification Roles

| Role | Description | Detected By |
|------|-------------|-------------|
| `time_series` | Primary demand data | Date column + numeric target + repeating IDs |
| `dimension` | Lookup/attribute table | ID columns overlapping with primary, no date column, mostly categorical |
| `regressor` | External features | Date column + numeric features, joinable to primary |
| `unknown` | Not recognized | Score below thresholds |

### Confidence Thresholds

- Time series: score >= 0.40
- Dimension: score >= 0.30
- Regressor: score >= 0.30

You can override auto-detected roles in the interactive confirmation step.

### Merge Strategy

1. Start with the primary time-series file
2. Left-join each dimension table on shared ID columns
3. Left-join each regressor table on shared time + ID columns
4. Duplicate column names get a `_<filename>` suffix (e.g., `quantity_stores`)
5. Null values in regressor columns are filled with 0

**Warning:** If key overlap between files is below 50%, a warning is shown in the merge preview.

---

## Output: Forecast Parquet

The forecast pipeline writes Parquet files with this schema:

| Column | Type | Description |
|--------|------|-------------|
| `series_id` | String | Series identifier (matches input) |
| `week` | Date | Forecast period date |
| `forecast` | Float64 | Point forecast (P50) |
| `forecast_p10` | Float64 | 10th percentile (if quantiles configured) |
| `forecast_p50` | Float64 | 50th percentile (if quantiles configured) |
| `forecast_p90` | Float64 | 90th percentile (if quantiles configured) |

Quantile columns are controlled by:
```yaml
forecast:
  quantiles: [0.1, 0.5, 0.9]    # generates forecast_p10, forecast_p50, forecast_p90
```

### Example

```
series_id  | week       | forecast | forecast_p10 | forecast_p90
-----------|------------|----------|--------------|-------------
STORE_001  | 2024-09-07 | 1234.5   | 1050.2       | 1418.8
STORE_001  | 2024-09-14 | 1250.0   | 1065.5       | 1434.5
STORE_002  | 2024-09-07 | 5600.3   | 4200.1       | 7000.5
```

---

## Output: Pipeline Manifest (JSON)

New local forecast and backtest attempts use schema-version-1 run bundles:

```text
runs/<lob>/<server-generated UUID32>/
  input.parquet              normalized raw values + configured series ID
  prepared.parquet           SeriesBuilder output
  config.json                canonical allowlisted settings + effective execution
  validation.json            actual validation result, or explicit opt-out
  forecast.parquet            forecast runs
  metrics.parquet             backtest runs
  champions.parquet           backtest runs
  leaderboard.parquet         backtest runs
  manifest.json               authoritative atomic commit marker
```

Optional supplied inputs (mapping/product/override/external/future-feature data,
conformal residuals or multi-horizon champion tables) are separately named
members when used. Internal candidate/attempt/failure files are not commit markers.

| Manifest field | Contract |
|----------------|----------|
| `schema_version`, `status`, `run_type` | `1`, `completed`, and `forecast` or `backtest` |
| `run_id`, `lob` | Same server UUID32 and validated LOB as context, metric rows, response and directory |
| `started_at`, `completed_at` | Canonical UTC timestamps; latest uses attempt start plus ID, never mtime |
| `artifacts` | Fixed allowlisted relative filenames, byte counts, SHA256 checksums, and frame row/schema descriptors |
| `input_data_hash`, `prepared_data_hash` | SHA256 of actual stored Parquet bytes; **not** semantic data-equivalence hashes |
| `config_hash`, `config_digest_kind` | SHA256 of canonical typed safe JSON, `sha256:canonical-json-v1` |
| `models`, `execution`, `source` | Actual selected model parameters/frequency/horizon, runtime versions and observed source hashes/build revision |
| `reproducibility` | Snapshot stages, supplied optional inputs, omitted/external state and retraining limitations |
| Validation/cleansing fields | Per-attempt reports, including configured opt-out and post-validation grade/score |

New `source.runtime_distributions` records allowlisted installed distribution
names and versions, including `xgboost-cpu` when it supplies the existing logical
`source.runtime_versions.xgboost` value. `source.polars_runtime` records the
loaded Polars variant, its distribution name and version; it does not guess the
standard runtime. Ambiguous XGBoost installations or unidentified Polars runtimes
block new publication: use a clean locked environment rather than mixing
distributions. These observations are not native-platform qualification or a
promise of deterministic retraining. Historical source metadata is never
backfilled from the current environment.

For scoped backtests, `validation.json` also retains paradox-comparison
applicability: `evaluated`, `status`, `comparisons_run`, performed comparisons and
an `unavailable` list with reasons/population counts. Unavailable evidence is a
WARNING with an explicitly neutral confidence component, not a clean comparison
or a substituted zero. Raw metric observations remain unchanged.

New backtests also record this typed `execution.scoring_contract` in the
manifest and hashed configuration:

```json
{
  "version": 1,
  "policy": "raw-observations-v1",
  "primary_metric": "wmape",
  "secondary_metric": "normalized_bias",
  "competition_scope": "lob"
}
```

An absent historical contract stays unknown. It is not filled from current
defaults. `leaderboard.parquet` is a result artifact: exact-run reads and exports
preserve its values/order/rank rather than running the current scorer on old
metric rows. New rankings retain `(competition_scope, group_key, model_id)` and
within-competition `rank`; no comparison across distinct competitions is implied.

Normalized raw input is not the original CSV/Parquet upload byte stream. Prepared
data is SeriesBuilder output; forecasters may do additional internal preparation.
Fitted models/pickles, remote provider responses and external model weights are
not captured or promised deterministic. Configuration omits descriptions,
credentials, webhook/network settings and private server paths. No secret-bearing
raw YAML or model `__dict__` dump is saved. Canonical JSON uses sorted keys and
explicit tags for dates/nonfinite metadata, not arbitrary object repr.
Supported weighted ensembles and constrained wrappers also record ordered,
recursively sanitized component parameters in `models[].components`, including
nested omissions. Each constrained node records its own declared constraint
configuration separately from its child, so stacked caps cannot overwrite their
parent's provenance. This changes their canonical effective configuration identity
when a constituent changes, without serializing fitted state or changing predictions.

Historical `forecast_*_manifest.json` and `manifest_*.json` sidecars remain legacy
schema zero, with unknown/unverified lineage. They are not upgraded into verified
run identities. See [local storage/recovery](DEPLOYMENT.md#immutable-local-runs-and-recovery)
for visibility, filesystem and durability boundaries.

Derived BI Parquet exports add `source_run_id` and `actuals_run_id` string columns
to the report data. Null identifies unbound/legacy provenance; these are not new
forecast run IDs. Each export has a short date-plus-UUID filename and is an atomic
standalone file, not a second authoritative run store.

---

## Output: Pipeline Execution Response

`POST /pipeline/backtest` and `POST /pipeline/forecast` return a typed
`PipelineRunResponse` (also documented in OpenAPI). Required `run_id`, `run_type`
and relative `links` identify the committed bundle; no filesystem paths are returned.
Existing fields such as
`champion_model`, `leaderboard`, `model_id`, `horizon`, `forecast_rows`, and
`forecast_preview` remain available where applicable. The required `execution`
object makes resolved intent explicit:

| Field | Meaning |
|-------|---------|
| `model_id` | Forecast model, or the single selected backtest result (including a composite); null for multiple/no winners |
| `model_source` | `request`, `config`, `default`, or `backtest` |
| `horizon_periods` | Effective production horizon, not the backtest fold length |
| `horizon_source` | `request`, `config`, or `default` |
| `config_source` | `upload` (including inherited defaults) or `default` |
| `config` | Allowlisted frequency, season length, column names, candidate models, quantiles, preprocessing/validation switches and backtest settings |

The top-level `champion_model` is supplied only for a single winner that this
name-based forecast endpoint can construct. A weighted-ensemble backtest still
reports `execution.model_id: "weighted_ensemble"` and its selection strategy, but
omits `champion_model` because its fitted components and weights are not carried
by a model-name request. The UI blocks that unsupported handoff rather than
substituting a baseline.

For example, forecasting with `?model_id=croston` and YAML
`forecast: {horizon_periods: 5, forecasters: [naive_seasonal, croston]}` reports
`execution.model_id: "croston"`, `model_source: "request"`,
`horizon_periods: 5`, and `horizon_source: "config"`. Sending `?horizon=3`
instead reports 3 and `horizon_source: "request"`. With neither file nor overrides,
the weekly platform default is 39 periods of `naive_seasonal`.

This is a safe settings summary, not a raw configuration dump or persistent run
catalog. It omits paths, credentials, webhook URLs, and arbitrary config metadata.
Configuration parse/type/value errors are HTTP 400 responses with a `detail`
message; query horizon validation uses HTTP 422. Invalid config uploads never
start a pipeline or replace existing output.

---

## Output: Metric Store (Parquet)

Pipeline backtests store metric records inside their committed run bundle.
Standalone appends retain explicit legacy partitions and unique atomic filenames.
Raw observations, not cached metric-column means, drive new scoring.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | String | Unique backtest run ID |
| `run_type` | String | `"backtest"` or `"live"` |
| `model_id` | String | Model name (e.g., `"lgbm_direct"`) |
| `fold` | Int | Walk-forward fold number |
| `series_id` | String | Series identifier |
| `target_week` | Date | Actual target period label |
| `forecast_step` | Int | Actual lead from training end, including any gap |
| `actual`, `forecast` | Float | Paired finite observations used by the scorer |
| `mase_scale` | Float / null | Training-only seasonal-naive error scale |
| `competition_scope` | String | Declared comparison scope; equal coverage is required within it |
| `wmape` | Float / null | Per-observation ratio; do not average to reconstruct volume WMAPE |
| `normalized_bias` | Float / null | Per-observation signed ratio; aggregate from raw values |
| `mae` | Float | Mean Absolute Error |
| `rmse` | Float | Root Mean Squared Error |

New leaderboard `wmape` is pooled absolute error divided by pooled absolute
actual volume. Explicit `macro_wmape` instead equally weights series/fold ratios.
Undefined denominators remain null. See
[numerical reliability](NUMERICAL_RELIABILITY.md) for formulas and policy limits.

---

## Sample Dataset: Rossmann

The bundled Rossmann dataset (`data/rossmann/`) provides a ready-to-use demo:

| File | Role | Rows | Key Columns |
|------|------|------|-------------|
| `train.csv` | Time series | 1M+ | `Store` (ID), `Date` (time), `Sales` (target), `Customers`, `Open`, `Promo` |
| `store.csv` | Dimension | 1,115 | `Store` (join key), `StoreType`, `Assortment`, `CompetitionDistance` |
| `test.csv` | Future features | 41K+ | `Store`, `Date`, `Open`, `Promo` (no `Sales` — forecast target) |

Load the demo via the Data Onboarding page or programmatically:

```python
import polars as pl
actuals = pl.read_csv("data/rossmann/train.csv")
stores = pl.read_csv("data/rossmann/store.csv")
```
