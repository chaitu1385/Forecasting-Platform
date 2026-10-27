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

Zero-volume observations' errors still count in aggregate volume WMAPE.
WMAPE and normalized bias are undefined (`null`), not perfect, if total volume
is zero. Macro WMAPE is undefined if any constituent ratio is undefined; it cannot
drop hard cases silently. MASE stores a nullable training-only `mase_scale`;
missing or zero scales do not yield a comparable MASE score. Undefined primary
criteria cause an explicit insufficient-data error, not an arbitrary champion.

## Preprocessing and validation targets

Cleansing, structural-break fitting/truncation and enabled regressor screening
are not yet fold-local. Backtesting rejects these enabled combinations before
the whole-history series builder runs. Production forecasting retains these
options. Defaults remain disabled and are not characterized as leaky.
Validation scores refer to the input target after the explicitly configured
non-learned series preparation, not retrospectively cleansed demand.

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
