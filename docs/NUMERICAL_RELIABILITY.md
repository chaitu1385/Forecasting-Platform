# Numerical reliability and score migration

## Calendars and forecast leads

Configure `forecast.frequency` explicitly as `D`, `W`, `M`, or `Q`. The calendar
helper validates, rather than guesses, the grid:

- Daily and weekly periods preserve their date/weekday anchor.
- Monthly and quarterly model grids use consistent start or end labels.
  Fiscal-quarter anchors are supported. Month-end survives February and leap years.
- Thirty-day pseudo-months and mixed start/end grids are rejected. Fill missing
  periods explicitly before CV; gap filling never drops off-grid observations.
- `freq_timedelta` only supports D/W. Use `PeriodCalendar` for series dates or
  `advance_date` for scalar event-date offsets.
- The existing `horizon_weeks`, `val_weeks`, and `gap_weeks` configuration names
  still mean **periods**. The `*_periods` aliases retain the same meaning.
- Models forecast `gap_periods + val_periods` steps. Only the validation dates
  are scored. `forecast_step` is the lead from training end: with gap 2, the first
  scored observation is step 3. A requested bucket without complete coverage fails.
- Explicit season lengths are preserved; only omitted values use profile defaults.
- Backend aliases come from installed pandas public offsets, preserving fiscal
  anchors without version-string parsing. Cleansing shifts retain Date/Datetime
  units, timezone and nanosecond labels using native expressions.
- Spark ML expectations use the same prepared panel as model fitting. Under ML's
  existing forward-fill policy, shorter series are padded to the panel end;
  forecast origins follow that declared policy, not independently guessed endpoints.

## Comparable observations

The serial engine, spawned worker and pure Spark worker share the same fold
evaluator. It validates nonempty, finite actuals and predictions, duplicate keys,
and exact series/date coverage **before** joining. Output order is irrelevant.
Quantiles must cover the same keys as point forecasts.

Each record identifies its run, LOB, fold, series, target date and forecast lead.
Raw `actual` and `forecast` are authoritative. Per-record metric columns describe
that observation; they are not repeated whole-fold scores and must not be averaged
to reconstruct volume WMAPE or RMSE.

A model failing any fold is excluded from comparable local results in its entirety,
with failures exposed by the engine. If no model completes all folds, the run fails.
Different candidate populations cannot be compared as a single LOB competition;
use an appropriate shared population or series-level selection. Spark worker
errors fail their task instead of returning deceptively successful partial rows.
The pure-worker contract is testable locally; it does not certify live Fabric.

The producer declares `competition_scope` (`lob`, `series_id`, or the configured
grouping column). A leaderboard retains `competition_scope` and `group_key`;
equal coverage is required **within** each declared competition, never inferred
from which model happened to succeed. Its unique key is
`(competition_scope, group_key, model_id)`, not merely `model_id`. Multi-group API
responses do not publish the easiest group's score as a global `best_wmape`.
Immutable publication must retain this scope and the stored leaderboard.

## Point forecasts, naive intervals and calibration

Seasonal naive retains its existing **point forecast = P50** contract, including
the plain-naive case (`season_length=1`). For each forecast's existing seasonal
residual population, its requested percentile is now:

`point + (empirical residual quantile(q) - empirical residual median)`

P50 is returned as the unchanged point exactly. Previously, nonmedian bounds
included the residual location while P50 did not, so biased residuals could put
the labeled median outside the bounds. Subtracting the same median from every
offset intentionally shifts the bounds without changing point predictions,
residual membership, model defaults or interval widths. It is not sorting,
clipping or widening. A single residual yields zero-width intervals at the point,
not evidence of certainty. Ordered point-centered intervals do **not** by
themselves establish empirical coverage; evaluate that on held-out observations.

Production quantiles must be finite and nondecreasing across **requested**
levels, including P50 when requested. A raw point need not lie inside an
upper-tail-only range such as P75–P90. Conformal correction keeps its existing
order-statistic and symmetric-bound formulas; it never changes the point/P50.
Valid negative corrections still tighten intervals. If correction would cross
any requested percentiles (including nested interval levels), it fails explicitly
before publication rather than clipping bounds or modifying the correction.
Malformed/nonfinite values fail explicitly; empty calibration residuals remain
a no-op for valid quantiles. Both named and prebuilt champions use only their
own model's residuals. Existing archived forecasts are not rewritten.

## Metric definitions and configuration

`src/metrics/definitions.py` defines the formulas and ranking directions.
`src/metrics/aggregation.py` applies them to the requested raw population.

| Criterion | Population calculation | Direction |
|---|---|---|
| `wmape` (default) | sum absolute error / sum absolute actual volume | lower |
| `macro_wmape` (explicit alternative) | arithmetic mean of each series/fold WMAPE | lower |
| `normalized_bias` | sum (forecast − actual) / sum absolute actual volume | closer to zero |
| `mae` | mean absolute error | lower |
| `rmse` | square root of mean squared error | lower |
| `mape` | mean absolute percentage error over nonzero actual observations | lower |
| `mase` | mean absolute error scaled by each series/fold's training-only seasonal-naive MAE | lower |

Both `backtest.primary_metric` and `secondary_metric` are usable independently
of cached record columns. Signed bias uses absolute magnitude in either ranking
position. Ties are deterministic by model ID. Ensemble weights use the same
primary criterion; exact zero-error models share all weight.

For actuals `[1000, 1]`, forecasts A=`[900, 0]` and B=`[800, 1]`, volume WMAPE
chooses A (`101/1001 < 200/1001`); macro WMAPE chooses B (`0.10 < 0.55`).
This is an intentional correction to previously mislabeled macro aggregation,
not a new business objective. Request `primary_metric: macro_wmape` for explicit
equal-series/fold weighting. It matches the former weighting on complete,
equal-length, positive-volume folds.

For YAML/API plans, list the selection criteria in `metrics` as required by the
existing configuration contract:

```yaml
backtest:
  primary_metric: macro_wmape
  secondary_metric: normalized_bias
metrics: [macro_wmape, normalized_bias]
```

Zero-volume observations' errors still count in aggregate volume WMAPE.
WMAPE and normalized bias are undefined (`null`), not perfect, if total volume
is zero. Macro WMAPE is undefined if any constituent ratio is undefined; it cannot
drop hard cases silently. MASE stores a nullable training-only `mase_scale`;
missing or zero scales do not yield a comparable MASE score. Undefined primary
criteria cause an explicit insufficient-data error, not an arbitrary champion.

### FVA and downstream reports

FVA first validates full per-run/series model populations, then recomputes each
model's complete fold score, then chooses a single model within each layer.
It never takes the best observation or combines different models' best periods.
FVA summaries explicitly average **series/fold layer scores**, not observation
ratios, retain run identity, distinguish series counts from evaluation counts,
and report undefined evaluations. Undefined layers cannot receive a `Keep`
recommendation. Missing model coverage produces an explicit API conflict.

Notebook/BI reports, model cards and AI numeric context also recompute from
paired raw observations. Explicit descriptive pooling is labelled with its
population and counts; it is not champion selection. `avg_*` reporting names
remain compatibility aliases alongside canonical weighted metrics. Legacy
score-only summaries cannot reconstruct volume-weighted errors.

### Spark/Fabric boundaries

The public Spark model identifier remains `model`; workers use `model_id`
internally. Primary and secondary criteria reach the notebook adapter.
In-memory leaderboards expose canonical metric names. Existing notebook, CLI
and deployment Delta write sites use explicit compatibility projections:
backtests retain the old `(series, model, fold)` table grain; leaderboards retain
the `model` merge key and historical `mean_*` column names as deprecated wire
aliases for newly computed metrics, **not** mean-of-observation formulas.
This avoids an implicit table-schema migration or changes to the Delta writer.
These projections never re-score an existing stored leaderboard. Validation
covers pure adapter/write-boundary contracts, not a live Fabric deployment.

## Preprocessing and validation targets

Cleansing, structural-break fitting/truncation and enabled regressor screening
are not yet fold-local. Backtesting rejects these enabled combinations before
the whole-history series builder runs. Production forecasting retains these
options. Defaults remain disabled and are not characterized as leaky.
Validation scores refer to the input target after the explicitly configured
non-learned series preparation, not retrospectively cleansed demand.

`POST /series/cleansing-audit` accepts explicit `frequency=D|W|M|Q` and returns
the effective frequency. Its existing default remains `W`; daily callers must
send `D` rather than relying on inference. The Python series builder passes its
configured forecast frequency to the cleanser. A diagnostic seasonal lookup uses
date offsets on raw/gapped data; it does not impose a forecast grid on that data.
Month/quarter lookups retain calendar anchors rather than fixed-day offsets.

## Exact M5 reference and empirical baseline qualification

`tests/integration/baselines/m5_daily_reference.json` is an input-only arithmetic
reference, not a learned-model golden. The independent standard-library oracle
uses the complete 105-series, 76,755-row fixture and both 28-period validation
folds. Seasonal-naive absolute error is 7,921 over absolute actual volume 7,945;
signed error is +479. Thus WMAPE is exactly `7921/7945` and bias `479/7945`.
These totals include zero-demand observations. Input identities normalize only
CRLF/LF checkout differences; model-run provenance also records raw byte hashes.

The full M5 fixture still fits all three configured models once. It checks every
raw key and actual against the independent reference, checks naive predictions
point-for-point, and recomputes model metrics with scalar arithmetic. Its JUnit
property `m5_daily_unapproved_candidate` records actual Python/library versions,
runner information and observed scores. Capture is not approval or locked-profile
qualification; no seed override or accepted lock is invented.

The original empirical per-model regression and FVA gates remain separate.
They must be completed using reviewed results from accepted native locked profiles;
the exact naive reference does not replace either empirical gate. An emulated
runtime loader/CPU-feature failure is not evidence of incorrect model mathematics.

## Historical rankings are immutable evidence

New leaderboards are computed from the current run's frame and carry
`scoring_policy: raw-observations-v1`. The aggregation helper rejects pooling
different run IDs. Summary-only legacy results lack enough information for
comparable rescoring and must be re-backtested.

An exact historical-run reader must return its **stored leaderboard unchanged**.
Recomputing older raw records with the current scorer is a separate, explicitly
versioned analysis, not the original historical ranking. Do not substitute a
recomputed leaderboard during immutable-run retrieval. Run publication/retrieval
owns this boundary; the numerical helper does not rewrite stored bundles.
