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

## Point constraints and probabilistic budgets

`forecasting.constrained.apply_point_constraints` is the shared allocator for the
constraint upload API and forecaster wrapper. For each period, let `L` be the
nonnegative floor, `U_i` the tighter supplied scalar/per-series cap (or no cap),
and `x_i = clip(input_i, L, U_i)`. Every input value and supplied bound must be
finite, every cap must be at least `L`, and a budget `B` must satisfy `B >= n*L`.
Missing/null/duplicate keys, missing configured capacities and infeasible periods
fail explicitly. The helper never edits the input frame or stored runs.

If `sum(x_i) <= B`, preserve the clipped values. Otherwise:

- **Proportional:** `L + (x_i-L) * (B-n*L) / sum_j(x_j-L)`. The implementation
  normalizes headroom before summing to avoid overflow, without rounding rows.
- **Largest-first:** remove excess down to `L`, descending by clipped value and
  ascending by string series ID for ties. Retaining headroom in reverse removal
  order avoids catastrophic cancellation and restores the original row order.

For IDs `a,b`, `[100,100]`, floor `0`, budget `50`, the two results are `[25,25]`
and `[0,50]`. `[100,2]`, floor `10`, budget `30` becomes `[20,10]`; two rows with
floor `40` and budget `50` are infeasible, not a successful total of `80`.
All output rows/periods are checked before any 200-row preview. Bounds/budget
checks allow only `1e-9 + 1e-12*abs(bound)` arithmetic error; measured excess and
totals are not rounded away. The API requires representable finite summary totals.
Polars keys retain nanosecond precision; the API does not date-infer CSV period
strings, which must use a consistent representation for the same period.

Without an aggregate budget, scalar/per-series clipping is the same monotone
function for the point and every supplied quantile. Validate finite ordering
before and after; do not sort invalid input or infer point containment in an
upper-tail interval. With an enabled budget, quantile-bearing input or
`predict_quantiles` raises `unsupported_aggregate_quantiles` (API 400/library
`ValueError`). These are marginal bounds, not a calibrated joint-budget
distribution. Disabled constraints bypass this restriction and remain a no-op.
Quantile algorithms, calibration formulas, scoring and historical artifacts are
otherwise unchanged.

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

### Post-validation comparison applicability

Backtest Simpson diagnostics use the declared competition scope, not arbitrary
row halves or models from different cohorts. Eligible models must have the same
complete observation population within the competition and each segment.
Explicit `simpsons_segment_columns` are honored after input/coverage validation;
the 2-to-50-cardinality heuristic applies only to automatic discovery.
The adapter first uses the authoritative raw-observation aggregator at the
model/segment comparison grain, then applies the existing volume-weighted
comparison/directional-reversal check. Zero-actual errors remain in the numerator:
mixed zero/positive arms can be evaluated even when cached observation ratios
contain nulls.

Single-candidate or insufficient-segment populations are explicitly
`not_applicable`; complete zero-volume arms are `not_evaluable`. Reports retain
affected groups, candidates, counts, undefined arms and a WARNING rather than a
PASS. The unavailable paradox-confidence component uses the existing neutral50
contribution, labelled unavailable/evaluated=false; its weight is not removed and
the number is not presented as a measured result. Other enabled layers still run.
Malformed eligible observations and unexpected checker errors remain failures.
A genuine reported paradox BLOCKER still halts publication when
`halt_on_blocker` is enabled. No ranking or weighted-mean formula is changed.

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
They now fail explicitly for missing/unapproved references, incomplete models,
or incompatible native profiles instead of skipping. Their original 5% champion,
10% per-model and 80% FVA-retention thresholds are unchanged. No authoritative
prior learned-model reference was found in the repository/path history; captured
runs are candidates for a first corrected-policy empirical reference, not proof
of historical no-regression.

The initial Linux x86-64 py311/py312/py313 references from native run
`40587318909` were subsequently approved in this
[independent coordinator audit](https://github.com/chaitu1385/Forecasting-Platform/pull/124#issuecomment-6455007868).
The full Linux matrix selects its exact `m5_daily_baseline_<profile>_linux.json`
through `M5_BASELINE_PATH`. Captured provenance, unapproved-candidate flags,
original source identity, null seed and full-precision observations remain
unchanged inside each approved reference. This is an AI coordinator audit
assertion, not a human review, cryptographic signature, historical regression
claim or Windows M5 qualification.

Both empirical gates consume the independently rescored winner's WMAPE, never
the API summary as the scoring authority. The API `best_wmape` must additionally
match that score at the existing `1e-12` relative/absolute arithmetic precision.

The full fixture can bind its candidate and complete raw predictions to actual
lock/manifest bytes, installed versions, selected wheel hashes and native profile
probe evidence. The legacy capture script only prepares an **unapproved** review
draft; independent approval must bind the entire captured record. Input hashes
normalize only CRLF/LF; dependency/wheel hashes remain raw. References are specific
to the exact Python patch, qualified OS family, native architecture, interpreter
platform/ABI, packages, wheels and lock/manifest identities. Full host/kernel
descriptors remain audit provenance, not compatibility equality keys; an
incidental kernel patch does not change the declared profile. References are never
silently reused across OS families or dependency changes. See the [empirical qualification commands and evidence
contract](../forecasting-product/tests/integration/baselines/README.md).

Final references still require reviewed results from accepted native locked
profiles; the exact naive oracle replaces neither empirical gate. An emulated
runtime loader/CPU-feature failure is not evidence of incorrect model mathematics.

The required Linux full-suite CI matrix captures each Python profile in a fresh
run-ID/attempt/profile directory. Its actual hash-locked dev restore produces a
pip install report, and the same interpreter produces the native dev
qualification before the full suite runs. Candidate JSON, complete raw
predictions, lock/manifest copies, install/check logs and native reports are
uploaded even when the two unapproved-reference gates fail; JUnit is retained
separately. Capture does not turn those failures into success or approve a
reference. The Docker cold/restart smoke also checks all 120 real demo forecast
rows for finite ordered P10/P50/P90, and preserves the seasonal-naive point/P50
contract when that model is selected. After the unchanged default two-run flow,
it explicitly requests a new seasonal-naive run using the same weekly input and
configuration with a 12-period horizon. All 120 rows must identify that model
and match the independent last-season point oracle. A further restart must
preserve its exact forecast/configuration/artifacts and both previous runs.

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

New backtest bundles record `execution.scoring_contract` in both the manifest and
canonical configuration: metadata version1, `policy: raw-observations-v1`, primary
and secondary criteria, and the declared competition scope. The configuration
digest includes this contract. The stored leaderboard contains the scorer's
values, competition identity and within-competition rank.

`MetricStore.leaderboard(lob="retail", run_id=run_id)` and
`ForecastAnalytics.model_leaderboard(lob="retail", run_id=run_id)` return that saved
frame without re-sorting or recomputation. Use `latest=True` within one LOB for a
deterministic latest committed selection. Multiple committed runs without a
selection raise an actionable scope error. Metric overrides cannot re-rank a
stored result; omit them for historical retrieval. `leaderboard_result(...)`
returns the frame and selected manifest together when provenance is needed.

Old bundles without scoring metadata remain **unknown**, not retroactively
labelled with today's policy. Existing recorded older policies are retained.
The API exposes `scoring_provenance` separately from values, and preserves
`group_key`/`competition_scope`. The UI uses stored row order/rank and the recorded
criterion, labels macro WMAPE distinctly, and does not invent a global winner
across cohorts or unknown scope. Leaderboard BI exports preserve those same rows.

Unbundled raw metrics remain readable. An explicit current analysis can use
`MetricStore.leaderboard_from_records(store.read(..., run_id=legacy_id))`; its
`scoring_policy` labels the newly computed result, not historical lineage.
For multiple legacy runs, `latest=True` requires stored `run_date` and uses
date/run-ID ordering; without that information select an explicit run ID.
FVA, calibration and descriptive reports remain separate analyses over selected
raw observations, not replacements for the immutable saved ranking.
