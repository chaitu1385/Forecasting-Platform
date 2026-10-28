# Python API Guide

Use the Forecasting Platform's Python classes directly for scripting, notebook workflows, or integration into orchestration tools like Airflow/Dagster.

---

## Quick Start

```python
import polars as pl
from forecasting_product.src.config.schema import PlatformConfig
from forecasting_product.src.pipeline.backtest import BacktestPipeline
from src.pipeline.forecast import ForecastPipeline

# Load config
config = PlatformConfig.from_yaml("config.yaml")

# Load data
actuals = pl.read_csv("actuals.csv", try_parse_dates=True)

# Backtest
bt = BacktestPipeline(config)
results = bt.run(actuals)
print(f"Champion: {results['champions']}")
print(f"WMAPE: {results['leaderboard'][0]['wmape']:.3f}")

# Forecast
fp = ForecastPipeline(config)
fp.set_conformal_residuals(results["conformal_residuals"])
forecast = fp.run(actuals, champion_model=results["champions"])
```

---

## Core Classes

### `BacktestPipeline`

**Module:** `src.pipeline.backtest`

End-to-end backtest: builds series, runs expanding-window backtests across all registered models, selects champion, writes metrics.

```python
from forecasting_product.src.pipeline.backtest import BacktestPipeline

pipeline = BacktestPipeline(config)
results = pipeline.run(
    actuals=actuals,                    # pl.DataFrame — required
    product_master=None,                # pl.DataFrame — for SKU transitions
    mapping_table=None,                 # pl.DataFrame — SKU mapping
    forecast_origin=None,               # date — default: max date in actuals
    overrides=None,                     # pl.DataFrame — planner overrides
    external_features=None,             # pl.DataFrame — regressors
)
```

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `backtest_results` | `pl.DataFrame` | Per-fold, per-series, per-model metrics |
| `champions` | `pl.DataFrame` | Champion model(s) per horizon bucket |
| `leaderboard` | `list[dict]` | Ranked model summary |
| `ensemble` | `BaseForecaster` | Fitted weighted ensemble |
| `failures` | `list` | Series/model failures |
| `calibration_report` | `dict` | Prediction interval coverage |
| `conformal_residuals` | `pl.DataFrame` | Residuals for conformal calibration |
| `data_quality_report` | `dict` | Quality assessment from SeriesBuilder |

---

### `ForecastPipeline`

**Module:** `src.pipeline.forecast`

Production forecast: builds series, fits champion model(s) on full history, generates point + probabilistic forecasts, writes Parquet with provenance manifest.

```python
from src.pipeline.forecast import ForecastPipeline

pipeline = ForecastPipeline(config)

# Optional: inject backtest residuals for conformal PI correction
pipeline.set_conformal_residuals(residuals_df)

forecast = pipeline.run(
    actuals=actuals,
    champion_model="lgbm_direct",       # str, BaseForecaster, or pl.DataFrame
    product_master=None,
    mapping_table=None,
    forecast_origin=None,
    overrides=None,
    external_features=None,
)
```

The `champion_model` parameter accepts:
- **String** — model name from registry (e.g., `"lgbm_direct"`)
- **BaseForecaster instance** — a pre-built model
- **DataFrame** — multi-horizon champion table for stitched forecasts

**Returns:** `pl.DataFrame` with the configured identity/time columns and `forecast`;
single-champion inference can also return requested quantiles.

Enabled point constraints use the shared allocator for both single champions and
the **complete stitched output**, not each bucket independently. Capacity
preparation is shared with the wrapper, and stored row/model provenance is
preserved. Infeasible or missing-capacity inputs fail before publication.
The existing stitched path produces points only: with enabled constraints,
aggregate-plus-quantile configuration raises `unsupported_aggregate_quantiles`;
element-only quantile configuration raises `unsupported_stitched_quantiles`.
Disabled constraints retain the existing point-only stitched behavior, not new
probabilistic support.

---

### `MetricStore`

**Module:** `src.metrics.store`

Immutable committed-run retrieval plus explicit standalone Parquet metric
partitions (`run_type=.../lob=.../`). Use the same run authority as the producer.

```python
from src.metrics.store import MetricStore

store = MetricStore(base_path="data/metrics/", runs_path="data/runs/")

# Write backtest metrics
store.write(records=metrics_df, run_type="backtest", lob="retail")

# Read with filters
all_metrics = store.read(run_type="backtest", lob="retail")

# Exact saved ranking; run_id comes from the pipeline response
lb = store.leaderboard(
    run_type="backtest",
    lob="retail",
    run_id=results["run_id"],
)

# Or select latest within this LOB, preserving its stored criteria and order
latest = store.leaderboard(lob="retail", latest=True)
board, manifest = store.leaderboard_result(lob="retail", run_id=results["run_id"])
scoring = manifest.execution.get("scoring_contract")  # None means unknown, not today's policy

# Accuracy over time for monitoring
trend = store.accuracy_over_time(
    model_id="lgbm_direct",
    run_type="live",
    lob="retail",
    metric="wmape",
)
```

Stored leaderboard reads never rescore metric rows. Multiple committed runs
without an exact/latest selection raise an actionable scope error, and criteria
overrides cannot re-rank immutable history. Notebook `model_leaderboard` accepts
the same `run_id`/`latest` selectors. BI leaderboard exports preserve the selected
rows and add export/source identities without changing their scores or order.

For an explicitly separate current analysis of unbundled raw metrics, select
`store.read(..., run_id=legacy_id)` and pass those rows to
`MetricStore.leaderboard_from_records(...)`. Its `scoring_policy` labels the
computation; it does not invent an original historical ranking.

**Metric schema:** `run_id`, `run_type`, `run_date`, `lob`, `model_id`, `fold`,
`grain_level`, `series_id`, `channel`, `target_week`, `forecast_step`, `actual`,
`forecast`, observation metric columns, nullable training-only `mase_scale`, and
the producer's `competition_scope`. See [DATA_FORMAT.md](DATA_FORMAT.md).

---

### `DataValidator`

**Module:** `src.data.validator`

Five-layer input validation: schema → duplicates → frequency → value range → completeness.

```python
from forecasting_product.src.data.validator import DataValidator
from forecasting_product.src.config.schema import ValidationConfig

validator = DataValidator(ValidationConfig())
report = validator.validate(df, target_col="quantity", time_col="week", id_col="series_id")

if not report.passed:
    for issue in report.errors:
        print(f"[{issue.check}] {issue.message}")
```

**`ValidationReport` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | True if no errors (warnings allowed) |
| `issues` | `list[ValidationIssue]` | All findings |
| `n_rows` | `int` | Total rows |
| `n_series` | `int` | Unique series count |
| `duplicate_count` | `int` | Duplicate `(id, time)` pairs |
| `negative_count` | `int` | Negative target values |
| `frequency_violations` | `int` | Inconsistent time gaps |
| `missing_column_names` | `list[str]` | Missing required columns |

Individual checks can also be called directly:

```python
issues, missing = validator.check_schema(df, "quantity", "week", "series_id")
issues, dups = validator.check_duplicates(df, "week", "series_id")
issues, freq = validator.check_frequency(df, "week", "series_id", frequency="W")
issues, negs = validator.check_value_range(df, "quantity")
issues = validator.check_completeness(df, "week", "series_id")
```

---

### `SeriesBuilder`

**Module:** `src.series.builder`

Constructs model-ready time series from raw actuals through a 10-step pipeline: schema validation → transition stitching → gap fill → structural break detection → demand cleansing → quality report → short/zero series filtering → regressor join/screening.

```python
from forecasting_product.src.series.builder import SeriesBuilder

builder = SeriesBuilder(config)
clean_df = builder.build(
    actuals=actuals,
    external_features=regressors_df,
    product_master=product_master_df,
    mapping_table=mapping_df,
    forecast_origin=None,
    overrides=None,
)

# Access intermediate reports after build()
print(builder._last_validation_report)
print(builder._last_cleansing_report)
print(builder._last_quality_report)
```

---

### `Reconciler`

**Module:** `src.hierarchy.reconciler`

Hierarchical forecast reconciliation — ensures parent = sum(children) at every level of the hierarchy.

```python
from forecasting_product.src.hierarchy.reconciler import Reconciler
from forecasting_product.src.hierarchy.tree import HierarchyTree

# Build tree
tree = HierarchyTree.build(actuals, levels=["category", "subcategory", "series_id"])

# Reconcile
reconciler = Reconciler(
    trees={"product": tree},
    config=config.reconciliation,
)
reconciled = reconciler.reconcile(
    forecasts={"product": forecast_df},
    actuals={"product": actuals},
    residuals={"product": residuals_df},   # needed for WLS/MinT
    value_columns=["forecast", "forecast_p10", "forecast_p90"],
    time_column="week",
)
```

**Supported methods:**

| Method | Description |
|--------|-------------|
| `bottom_up` | Leaf forecasts are authoritative; aggregate up |
| `top_down` | Disaggregate root forecast to leaves by historical proportions |
| `middle_out` | Forecast at mid level, disaggregate down |
| `ols` | Ordinary least squares optimal reconciliation |
| `wls` | Weighted least squares (structural/variance weights) |
| `mint` | Minimum trace with Ledoit-Wolf shrinkage covariance |

---

### `apply_point_constraints` and `ConstrainedDemandEstimator`

```python
from src.config.schema import ConstraintConfig
from src.forecasting.constrained import apply_point_constraints, ConstrainedDemandEstimator

constraints = ConstraintConfig(
    enabled=True, min_demand=0, aggregate_max=50,
    proportional_redistribution=False,
)
bounded = apply_point_constraints(
    forecasts, constraints,
    id_col="series_id", time_col="week", value_col="forecast",
)
model = ConstrainedDemandEstimator(base_forecaster, constraints)
model.fit(actuals)
point_forecast = model.predict(horizon=4)
```

The pure helper returns a new frame, preserving exact identity/time keys,
input order and nonvalue columns. Disabled constraints are a no-op. Bounds and
values must be finite; bounds must be nonnegative and feasible. After row clipping,
proportional allocation scales only headroom above `min_demand`. Largest-first
removes excess in descending clipped-value order, with ascending string-ID ties.
Thus `[100,100]` for IDs `a,b` and budget `50` yields `[25,25]` proportionally or
`[0,50]` largest-first.

An optional `capacity_map` must cover every output series with finite capacities
at least as large as the floor. Scalar and per-series caps use the tighter bound.
When `capacity_column` is configured on the wrapper, **every fit** requires that
column and complete finite training values; its per-series maximum becomes the
capacity. A refit clears old capacity state even when the new fit fails. To remove
that policy, explicitly set `capacity_column=None` and refit.
`prepare_constraint_capacity_map(prepared_history, constraints, id_col)` exposes
this same validation/extraction for the stitched pipeline; it never retains a
previous fit's state.

`predict_quantiles` and quantile-bearing point frames support element-wise
scalar/per-series bounds only. Requested values must already be finite and ordered;
clipping never sorts or drops quantiles or forces a general model's point to P50.
Enabled aggregate budgets raise `ValueError("unsupported_aggregate_quantiles: ...")`
before probabilistic publication. Invalid/infeasible inputs also raise `ValueError`;
no existing immutable run is rewritten. See [the numerical contract](NUMERICAL_RELIABILITY.md).

### `ForecasterRegistry`

**Module:** `src.forecasting.registry`

Central registry for all model implementations. Uses a decorator pattern.

```python
from forecasting_product.src.forecasting.registry import registry

# List available models
print(registry.available)
# ['naive_seasonal', 'ses', 'ets', 'arima', 'theta', 'lgbm_direct', ...]

# Instantiate by name
model = registry.build("lgbm_direct", horizon=12, season_length=52)

# Batch instantiate from config
models = registry.build_from_config(
    names=["lgbm_direct", "naive_seasonal", "ets"],
    params={"lgbm_direct": {"n_estimators": 500}},
)
```

See [Adding Custom Models](ADDING_MODELS.md) for how to register your own forecaster.

---

## Working with Polars

The entire platform uses [Polars](https://pola.rs) DataFrames — not pandas. Key differences:

```python
import polars as pl

# Read
df = pl.read_csv("actuals.csv", try_parse_dates=True)
df = pl.read_parquet("forecast.parquet")

# Filter
retail = df.filter(pl.col("lob") == "retail")

# Group + aggregate
summary = df.group_by("series_id").agg(
    pl.col("quantity").sum().alias("total"),
    pl.col("quantity").count().alias("n_weeks"),
)
```

All pipeline inputs and outputs are `pl.DataFrame`. Convert to/from pandas with `df.to_pandas()` / `pl.from_pandas(pdf)`.
