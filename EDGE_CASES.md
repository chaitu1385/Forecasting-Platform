# Edge Cases

A catalog of the failure modes that break forecasting platforms — what goes wrong, how this platform defends against it, and what to watch for. Written for data scientists and engineers who need to understand why forecasts fail and how to prevent it.

---

## 1. All-Zero Series

**What happens:** A time series where every demand value is zero. Models either fail to fit (division by zero in variance calculations) or produce meaningless forecasts. Metrics like WMAPE and CV become undefined.

**How the platform handles it:** The data quality report (`src/data/quality_report.py`) counts zero series in `zero_series_count`. The series builder (`src/series/builder.py`) drops them before model training when `drop_zero_series=True` — it sums absolute values per series and filters out any with total = 0. The forecastability module (`src/analytics/forecastability.py`) returns CV=0.0 for constant/zero series to prevent downstream division errors.

**What to watch for:** This check runs on the *full* series. A series that was active but became all-zero recently (post-discontinuation trailing zeros) will be dropped even though the historical portion had real demand. If you need to preserve recent-zero series for transition mapping, disable `drop_zero_series` and handle them in the SKU transition pipeline instead.

---

## 2. Series Shorter Than the Seasonal Period

**What happens:** A series with 30 weekly observations when models need at least 52 to estimate a full seasonal cycle. Seasonal ARIMA fails (needs d+D+s observations). Seasonal decomposition produces artifacts. Lag features reference nonexistent periods.

**How the platform handles it:** `FREQUENCY_PROFILES` in `src/config/schema.py` defines frequency-aware minimum lengths (weekly: 52, monthly: 24, daily: 90, quarterly: 8). The series builder filters by `min_series_length_weeks` before training. The data quality report counts and surfaces `short_series_count`. The sparse detector returns `demand_class="insufficient_data"` for series below `min_periods` (default 10), routing them to standard models rather than intermittent ones. The seasonal naive forecaster uses cyclic index fallback for series shorter than `season_length`.

**What to watch for:** Borderline series (e.g., 53 weeks for a weekly model with season_length=52) pass the filter but may still produce unstable seasonal estimates. Consider setting `min_series_length` to 1.5× the seasonal period for the configured frequency for better reliability. Also, foundation models (Chronos, TimeGPT) can produce reasonable forecasts on shorter series since they don't need to learn seasonality from scratch.

---

## 3. Sudden Level Shifts / Structural Breaks

**What happens:** A store renovation, competitor entry, or market disruption creates a permanent shift in demand level. Models trained on full history average the old and new regimes, forecasting neither correctly — systematically biased in both directions.

**How the platform handles it:** The `StructuralBreakDetector` (`src/series/break_detector.py`) identifies changepoints using CUSUM (zero-dependency) or PELT (via `ruptures` library). CUSUM uses binary segmentation with within-segment standard deviation to avoid inflation from the breaks themselves. Once breaks are detected, the platform truncates history to keep only post-last-break data, ensuring models train on the current regime. Results are surfaced in a `BreakReport` with per-series break dates and counts.

**What to watch for:** Break detection is sensitive to the penalty parameter — too low finds spurious breaks in noisy data, too high misses real shifts. Gradual trends can be misidentified as breaks. Truncation is aggressive: if the last break is recent, very little training data remains. Consider combining break detection with the `min_series_length` filter to ensure post-truncation series are still long enough to model. The PELT method is more accurate than CUSUM but requires the optional `ruptures` dependency.

---

## 4. Holiday & Promotional Spikes

**What happens:** Black Friday, Christmas, or a promotional event creates demand 5-10× the baseline. If untreated, the model memorizes the spike and either (a) expects it every week, inflating the baseline, or (b) treats it as noise, widening prediction intervals unnecessarily.

**How the platform handles it:** Two complementary mechanisms: (1) The `DemandCleanser` (`src/data/cleanser.py`) detects spikes as outliers via IQR or z-score and can clip or interpolate them, removing the distortion from baseline estimation. (2) The `generate_holiday_calendar()` function (`src/data/regressors.py`) creates holiday flag and count columns that ML models use as known-ahead regressors, letting them *learn* the event effect explicitly rather than treating it as unexplained noise. The external regressor config distinguishes `"known_ahead"` features (holidays, planned promos) from `"contemporaneous"` ones.

**What to watch for:** Cleansing and regressor encoding serve different purposes. *Recurring* events (Christmas, Thanksgiving) should be encoded as regressors so models can predict them. *One-time* events (a competitor's liquidation sale, a data entry error) should be cleansed. Applying both simultaneously to the same event risks removing the signal the regressor is supposed to capture. The platform doesn't automatically distinguish between these — the user must configure which events are regressors and which are outliers.

---

## 5. New Products with No History (Cold Start)

**What happens:** A new SKU launched last week has zero or near-zero historical data points. No traditional time series model can fit — there's simply nothing to learn from. This is the hardest problem in retail forecasting.

**How the platform handles it:** The `SKUMappingPipeline` (`src/sku_mapping/pipeline.py`) searches for predecessor SKUs using four methods: attribute matching (same category/size/brand), fuzzy name matching (rapidfuzz), sales curve fitting (S-curve proportion transition), and temporal comovement (cross-correlation). When a predecessor is found, the `TransitionEngine` (`src/series/transition.py`) stitches the old SKU's demand history onto the new one across three scenarios: (A) already launched — full history stitch, (B) launches within forecast horizon — ramp down old, ramp up new, (C) beyond horizon — forecast old only. For genuinely novel products with no match, foundation models (Chronos, TimeGPT) provide zero-shot forecasts from pre-trained knowledge. Manual overrides via the `OverrideStore` let planners inject judgment.

**What to watch for:** Predecessor matching quality determines forecast quality — a wrong match produces worse forecasts than no match. The `CandidateFusion` module blends multiple matching methods using Bayesian proportions, but the user should review high-impact mappings. Foundation model zero-shot forecasts vary in quality depending on how representative the pre-training corpus is of your domain. Always validate cold-start forecasts manually for the first few cycles.

---

## 6. Late or Out-of-Order Data

**What happens:** Data arrives after the forecast has already been generated, or the same (series_id, date) pair appears twice with different values (e.g., a revised actual replaces a preliminary estimate).

**How the platform handles it:** The `DataValidator` (`src/data/validator.py`) catches duplicate (id, date) pairs via `check_duplicates()` — this is an ERROR-level issue that blocks the pipeline. The `check_frequency()` method flags series with inconsistent date gaps, which can indicate out-of-order or missing periods.

**What to watch for:** The platform catches duplicates at pipeline *input* time, but it does not detect data that arrives *after* a forecast run has already consumed stale values. This is a pipeline scheduling concern, not a data quality concern — the solution is to ensure data pipelines complete before forecast jobs trigger. Additionally, "soft" duplicates (same key, different values from a revised actuals feed) will be caught as duplicates, but the validator can't determine which value is correct. Pre-process revised actuals upstream to resolve conflicts before loading.

---

## 7. Negative Demand Values (Returns)

**What happens:** Net returns create negative demand values in the data. WMAPE becomes undefined when actuals are negative. Log transforms fail. Models that assume non-negative demand (Croston, Poisson-based) crash or produce nonsense.

**How the platform handles it:** The `DataValidator` (`src/data/validator.py`) has a `check_non_negative` flag (enabled by default) that raises an ERROR on any negative values. The value range check also supports custom `min_value` and `max_value` bounds for domain-specific constraints.

**What to watch for:** Some businesses legitimately have net-negative demand periods (e.g., a return wave after holiday season). In these cases, set `check_non_negative=False` in the validation config and handle the data upstream — either separate returns from gross demand, or use models robust to negative values (tree-based models like LightGBM handle negatives natively). The intermittent demand models (Croston, TSB) will not work correctly with negative values regardless of the validation setting.

---

## 8. Mixed Frequencies in the Same Dataset

**What happens:** Weekly and monthly data mixed in one input file, or monthly data with varying month lengths (28-31 days) that look like inconsistent gaps. The model treats all rows as the same frequency, creating misaligned lags and broken seasonal patterns.

**How the platform handles it:** The `DataValidator` (`src/data/validator.py`) accepts a `frequency` parameter (``"D"``, ``"W"``, ``"M"``, ``"Q"``) and passes it to `check_frequency()`, which compares observed date gaps per series against the expected interval. For daily and weekly data, gaps must match exactly. For monthly and quarterly data, the validator uses a tolerance range (28–31 days for monthly, 89–92 days for quarterly) to accommodate variable calendar month lengths. Series with gaps outside the expected range are flagged in `frequency_violations`. The `FREQUENCY_PROFILES` system (`src/config/schema.py`) defines a single frequency per pipeline run — you can't mix frequencies in one execution.

**What to watch for:** The `SeriesBuilder` threads `ForecastConfig.frequency` into `DataValidator` automatically. If you construct a `DataValidator` directly (e.g. in scripts or tests), pass the correct `frequency` — it defaults to ``"W"`` for backward compatibility. If you have genuinely mixed-frequency data, split it into separate pipeline runs (one weekly, one monthly) and combine the outputs downstream. The platform does not support mixed-frequency reconciliation within a single run.

---

## 9. Missing Values / Gaps in Time Series

**What happens:** Weeks with no data points — caused by data collection failures, store closures, or system outages. Lag features reference null values. ARIMA can't handle internal gaps. Aggregation over periods with gaps undercounts.

**How the platform handles it:** The data quality report (`src/data/quality_report.py`) surfaces `missing_week_pct` and `series_with_gaps`. The base forecaster class (`src/forecasting/base.py`) provides `fill_weekly_gaps()` with two strategies: `"zero"` (fill gaps with 0.0 — appropriate when missing = no demand) and `"forward_fill"` (propagate last known value — appropriate when missing = unobserved but demand likely continued). ML models default to forward-fill to avoid zero contamination in lag features. Gap filling is configurable via `DataQualityConfig(fill_gaps, fill_value)`.

**What to watch for:** Long gaps (longer than one seasonal cycle) filled with zeros create artificial sparse patterns that may trigger the intermittent demand detector, routing the series to Croston when it's actually a regular-demand series with bad data. Long gaps filled with forward-fill create artificially flat regions that distort trend estimation. For gaps longer than ~4 periods (frequency-dependent), consider excluding the pre-gap data entirely (similar to structural break truncation) rather than imputing.

---

## 10. Extreme Outliers (Data Entry Errors)

**What happens:** A single value 100× the baseline — a typo adding extra zeros, an inventory correction coded as a sale, or a bulk order that doesn't represent recurring demand. The outlier inflates variance estimates, shifts the mean, destabilizes model coefficients, and widens prediction intervals for months.

**How the platform handles it:** The `DemandCleanser` (`src/data/cleanser.py`) offers two detection methods: IQR (flags values outside [Q1 - k·IQR, Q3 + k·IQR], default k=1.5) and z-score (flags |z| > threshold, default 3.0). Both are computed per-series to avoid cross-series contamination. Correction actions: `"clip"` winsorizes to the fence values, `"interpolate"` replaces with linear interpolation between adjacent valid points. Every correction is logged in the `CleansingReport` with before/after values.

**What to watch for:** The IQR method is robust (outliers don't inflate the IQR itself) but conservative — it may miss moderate outliers. The z-score method is more sensitive but suffers from masking: the very outliers it's trying to detect inflate σ, making them look less extreme. For heavy-tailed demand distributions (common in retail), IQR is generally the better choice. Neither method distinguishes between data errors and legitimate rare events — a genuine bulk order looks identical to a typo. Review the cleansing report before accepting corrections.

---

## 11. Constant-Value Series

**What happens:** Every observation has the same value (e.g., a placeholder, a fixed allocation, or genuinely flat demand). Standard deviation = 0. Z-score calculation divides by zero. Coefficient of variation is undefined. Seasonal decomposition finds no pattern.

**How the platform handles it:** The forecastability module (`src/analytics/forecastability.py`) returns CV=0.0 when std < 1e-12, preventing division errors. The cleanser's z-score method (`src/data/cleanser.py`) returns z=0 for zero-variance series, treating all values as non-outliers. The regressor screening module (`src/data/regressor_screen.py`) drops features with variance below a threshold (default 1e-6), preventing constant-value features from reaching models.

**What to watch for:** A constant series might indicate a data quality issue — placeholder values (e.g., all 999s), a frozen data feed, or an allocation that replaced actual demand. Before accepting a constant series as genuine, verify with the data source. If it is genuine (a fixed contractual allocation), a simple repeat forecast may be more appropriate than any statistical model, and the series could be excluded from the main pipeline.

---

## 12. Duplicate Records

**What happens:** The same (series_id, date) pair appears multiple times. Aggregations double-count demand. Backtest splits include the same observation in both train and test. Metric calculations are biased.

**How the platform handles it:** The `DataValidator` (`src/data/validator.py`) `check_duplicates()` computes `unique (id, date)` count vs total rows. Any duplicates raise an ERROR-level validation issue that blocks the pipeline. The count is reported in `ValidationReport.duplicate_count`.

**What to watch for:** This catches exact key duplicates but not "soft" duplicates — rows with the same (id, date) but different values, which can occur when preliminary actuals are revised. The validator blocks on the duplicate key but can't determine which value is correct. Pre-process your data to resolve conflicts (keep latest revision, sum if appropriate, etc.) before feeding it to the platform. Also, be aware that some data sources produce duplicates by design (e.g., one row per transaction rather than aggregated demand per period) — these need pre-aggregation.

---

## 13. Discontinued Products

**What happens:** A product reaches end-of-life. Without detection, the platform continues generating forecasts for a product that will never sell again, wasting compute and creating phantom demand in the S&OP plan.

**How the platform handles it:** The `TransitionEngine` (`src/series/transition.py`) handles discontinuation through the SKU mapping pipeline. When a new SKU replaces an old one with `launch_date` beyond the forecast horizon (Scenario C), the old SKU is forecasted to the end of horizon and the new SKU is marked as "pending transition." The transition proportions and ramp shapes control how demand transfers from old to new.

**What to watch for:** The platform has no explicit "discontinued" flag — discontinuation is inferred from the presence of a successor mapping. A series that simply stops receiving new data will *not* be automatically flagged as discontinued; it will just have trailing zeros that eventually trigger the all-zero filter (if enabled) or produce stale forecasts based on historical patterns. Consider adding a product lifecycle status to your master data and filtering discontinued SKUs before pipeline input.

---

## 14. Capacity / Budget Constraint Violations

**What happens:** The unconstrained statistical forecast says demand will be 2 million units, but the factory can only produce 1.5 million. Or the category forecast exceeds the allocated budget for the quarter. The S&OP plan becomes infeasible.

**How the platform handles it:** The `ConstrainedDemandEstimator` (`src/forecasting/constrained.py`) wraps any base forecaster and applies post-prediction constraints. Element-wise bounds enforce per-SKU floors (`min_demand`) and ceilings (`max_capacity`, either from a data column or a global scalar). Aggregate constraints enforce per-period budget caps, redistributing excess proportionally across series. Quantile intervals are also constrained, with a monotonicity pass ensuring p10 ≤ p50 ≤ p90 after clipping.

**What to watch for:** Proportional redistribution assumes all series are equally elastic — it scales every SKU by the same factor to meet the aggregate constraint. In practice, some SKUs have higher priority (higher margin, contractual obligations) and should absorb less of the cut. Priority-based allocation is not yet supported; for now, model high-priority SKUs with tighter `min_demand` floors to protect their forecasts. Also, heavily constrained forecasts lose their statistical interpretation — a forecast capped at capacity tells you "we'll be capacity-limited," not "demand will equal capacity."

---

## 15. Model Fitting Failures

**What happens:** A forecaster throws an exception during fitting — numerical instability (singular matrix in ARIMA), missing optional dependency (neuralforecast not installed), insufficient data after all the filtering, or convergence failure in optimization.

**How the platform handles it:** Multiple defense layers: (1) Optional dependency guards (`_HAS_STATSFORECAST`, `_HAS_NEURALFORECAST`, `_HAS_MLFORECAST`) check at import time and raise clear errors at fit time. (2) The structural break detector falls back from PELT to CUSUM when `ruptures` is unavailable. (3) The ensemble forecaster falls back to uniform weights when inverse-WMAPE weights are invalid (zero/negative totals). (4) The seasonal naive forecaster uses cyclic index fallback for series shorter than the seasonal period. (5) All forecasters return an empty DataFrame with correct schema when no valid series remain, preventing downstream schema errors.

**What to watch for:** Silent fallbacks can mask problems — a model that falls back from PELT to CUSUM, or from inverse-WMAPE to uniform weights, will still produce output, but potentially of lower quality. Check the pipeline manifest to verify which model actually ran and whether any fallbacks were triggered. If a specific model consistently fails on certain series, investigate whether those series should be routed to a different model class (e.g., intermittent demand series failing in ARIMA should be routed to Croston via the sparse detector).

---

## 16. Fabric Environment Restrictions

**What happens:** Microsoft Fabric's default Spark runtime doesn't include DuckDB, neuralforecast/PyTorch, or certain Python packages. `pip install` in notebook cells is limited. The override store fails to initialize. Neural models can't be instantiated.

**How the platform handles it:** The `get_override_store()` factory auto-detects DuckDB availability and falls back to `ParquetOverrideStore` (Polars read/write Parquet). The `requirements-fabric.txt` pins only packages safe to install in Fabric notebooks — it excludes `pyspark` (provided by runtime), `delta-spark` (provided), `duckdb`, `neuralforecast`, `shap`, `fastapi`. The `FabricNotebookAdapter` wraps all boilerplate (SparkSession, FabricLakehouse, config loading) into a one-liner. Optional dependency guards (`_HAS_STATSFORECAST`, `_HAS_NEURALFORECAST`) prevent import-time crashes.

**What to watch for:** The Parquet override store doesn't support concurrent writes — if multiple notebooks write overrides simultaneously, last-write-wins. In production Fabric environments, use a single orchestrator notebook for override management. Also, `%pip install` in Fabric adds packages for the session only — they don't persist across cluster restarts. Pin your library env in the Fabric workspace settings for durable installs.

---

## 17. Parallel Execution Failures

**What happens:** A worker process in `BatchInferenceRunner` or parallel backtest crashes from out-of-memory, convergence or serialization errors. Forking after Polars has initialized its native threads can also deadlock workers in IPC reads while the parent waits indefinitely.

**How the platform handles it:** Both executors use an explicit `spawn` context without changing the embedding application's global start method. Workers receive configured model objects and Polars IPC data, preserving parameters and frequency; pools are capped by useful task count. Failed forecast batches are logged and raised, never silently concatenated into a partial success. The backtest engine retains its per-model/fold failure report through `engine.failures` and `get_failure_summary()`. Bounded subprocess tests reproduce the former Linux deadlock and verify real ASGI execution and serialization-failure reporting.

**What to watch for:** A silently failed batch means some series are missing from the output. Always check `engine.failures` after a parallel backtest run and compare the number of output series against the input. Memory pressure is the most common cause — each worker loads a full copy of the model plus its batch data. If you see OOM errors, reduce `batch_size` or `n_workers`. For models that aren't pickle-safe (e.g., those with live database connections), the parallel path will fail — use `n_workers=1` as a safe fallback.

Previously fitted ML models can also carry cached quantile learners trained on a
different date range. Each refit clears that training-dependent cache while
retaining model configuration. Otherwise point predictions can land on validation
dates while cached intervals target dates beyond the full history, silently
leaving missing intervals or using information from outside the training fold.
Real LightGBM/XGBoost refit and previously fitted parallel-fold tests compare
against fresh training on the same fold, including non-null interval checks.

---

## 18. Stale Alerts from Drift Detection

**What happens:** The `AlertDispatcher` fires drift alerts based on `ForecastDriftDetector` output, but the alerts reflect a temporary data anomaly (late data delivery, one-off holiday effect) rather than genuine model degradation. The operations team gets paged for a non-issue, learns to ignore alerts, and then misses a real drift event.

**How the platform handles it:** The `AlertConfig` has a `min_severity` filter — set to `"critical"` to suppress warning-level alerts. The `ForecastDriftDetector` uses configurable `baseline_weeks` and `recent_weeks` windows, so transient spikes in a 1-week window don't trigger alerts if the baseline window is 12+ weeks. The alert payload includes `current_value` and `baseline_value`, enabling downstream filtering rules. The `AlertDispatcher` counts dispatched alerts via `dispatched_count` for monitoring alert volume.

**What to watch for:** Alert fatigue is a serious operational risk. Start with `min_severity: critical` and only drop to `warning` once you've tuned the drift detection thresholds for your data. Set `baseline_weeks` to at least 8-12 to absorb seasonal variation. Monitor `dispatched_count` — if it exceeds a few per week, your thresholds are too sensitive. Consider adding a cooldown period (not yet built-in) where the same series can't trigger the same alert type within N days.
## 19. Claude API Unavailability (AI Features)

---

## 19. Claude API / LLM Unavailability

**What happens:** The Anthropic API is unreachable — no API key configured, network timeout, rate limiting, or the `anthropic` package is not installed. All four AI endpoints (`/ai/explain`, `/ai/triage`, `/ai/recommend-config`, `/ai/commentary`) would fail if they depended on a live API call. Similarly, the `LLMAnalyzer` in the analytics module would fail if it assumes the LLM is always available.

**How the platform handles it:** Two layers of graceful degradation: (1) AI API features (`src/ai/`): Every AI feature class inherits from `AIFeatureBase`, which checks for client availability via the `.available` property. When unavailable, `NaturalLanguageQueryEngine` returns "AI analysis unavailable", `AnomalyTriageEngine` returns alerts in original severity order with a template summary, `ConfigTunerEngine` returns an empty recommendations list, and `CommentaryEngine` generates a template-based summary from raw metrics. The `AIConfig.enabled` flag provides a master switch. (2) Analytics LLM integration (`src/analytics/llm_analyzer.py`): The `LLMAnalyzer` checks for API key availability at initialization and returns a stub response with `available=False`. No pipeline stage hard-depends on LLM output — all statistical analysis runs independently.

**What to watch for:** Fallback responses lack the business-context reasoning that Claude provides — a triaged alert list without impact scoring is just the raw drift output. Monitor the `sources_used` field in NL query responses and the presence of `suggested_action` in triage results to detect when fallbacks are active. For `LLMAnalyzer`, always gate content display on the `available` flag. If Claude rate limiting is frequent, consider adding a response cache keyed on (lob, series_id, question_hash).

---

## 20. Schema Auto-Detection Failures

**What happens:** The `DataAnalyzer` uses heuristics (column names, dtypes, cardinality) to auto-detect time columns, target columns, and series ID columns. Ambiguous schemas — multiple date columns, multiple numeric columns with similar names, or non-standard column naming — can produce incorrect mappings, leading to wrong config recommendations.

**How the platform handles it:** The `DataAnalyzer` (`src/analytics/analyzer.py`) returns detection results with confidence scores and warnings. Ambiguous detections are flagged in the `warnings` list. The auto-generated `PlatformConfig` is a recommendation, not a binding contract — users should review and override column mappings in their YAML config when auto-detection produces incorrect results.

**What to watch for:** Auto-detection works well for datasets with conventional column names (`date`, `sales`, `sku_id`) but struggles with domain-specific naming (`wk_end_dt`, `qty_shipped`, `material_number`). Datasets with multiple date columns (e.g., `order_date`, `ship_date`, `invoice_date`) will trigger ambiguity warnings — the analyzer picks the most likely candidate but may choose wrong. Always validate the `SchemaDetection` output before running a full pipeline on a new dataset.

---

## 21. Forecastability Score Edge Cases

**What happens:** The `ForecastabilityAnalyzer` computes entropy, SNR, and CV metrics per series. Series with fewer than ~20 observations produce unreliable entropy and SNR estimates. Constant series (zero variance) return NaN for CV and spectral entropy. Very short or degenerate series can produce misleading forecastability scores.

**How the platform handles it:** The `ForecastabilityAnalyzer` (`src/analytics/forecastability.py`) handles constant series by returning CV=0.0 when standard deviation is below 1e-12, preventing division-by-zero errors. For very short series, entropy calculations use whatever data is available but may produce unstable estimates. The forecastability score defaults to low values for degenerate cases, which conservatively routes these series to simpler models.

**What to watch for:** Fallback responses lack the business-context reasoning that Claude provides — a triaged alert list without impact scoring is just the raw drift output. Monitor the `sources_used` field in NL query responses and the presence of `suggested_action` in triage results to detect when fallbacks are active. If Claude rate limiting is frequent, consider adding a response cache keyed on (lob, series_id, question_hash).

---

## 20. Large File Uploads in Frontend

**What happens:** A user uploads a CSV with millions of rows via the Data Onboarding page. The FastAPI `/analyze` endpoint runs `DataAnalyzer.analyze()` synchronously, which can time out for very large files. Memory consumption grows with file size.

**How the platform handles it:** The analysis endpoint processes data server-side and returns a structured response. The Next.js frontend renders charts client-side using Recharts and Plotly, avoiding server-side image generation bottlenecks. Each page shows graceful empty/loading states when no data is loaded. The Rossmann sample dataset (18 MB, 1M rows) is bundled as a demo to bypass upload friction.

**What to watch for:** For datasets larger than the Rossmann sample, API request timeouts may need to be increased. The Docker API container should be provisioned with at least 2 GB RAM for datasets with 1000+ series. Consider adding file size validation on the frontend to warn users before uploading very large files.


---

## 21. Multi-File Upload: No Time-Series File Detected

**What happens:** A user uploads multiple CSV files (e.g., a stores lookup table and a promotions calendar) but none contains a recognizable time-series structure — no date column paired with a numeric target and repeating identifiers.

**How the platform handles it:** The `FileClassifier.classify_files()` method scores each file for the `time_series` role using heuristics: date column detection (type + name pattern), target column name matching, ID column repetition, and row count. If no file scores above the 0.4 confidence threshold, `ClassificationResult.primary_file` is `None` and a clear warning is emitted. The Data Onboarding page shows an `st.error()` explaining exactly what format is expected (date column, numeric target, identifier columns) and stops the flow before any merge or analysis is attempted.

**What to watch for:** Files with date columns but no clear target (e.g., a holiday calendar with `date` and `holiday_name`) might score near the threshold. If a user's time-series file uses an unusual target column name (e.g., `y`, `total`), it may not match the known patterns and get classified as a regressor instead. Users can override the auto-detected role via the interactive confirmation step.

## 22. Multi-File Upload: Duplicate Column Names Across Files

**What happens:** Two files share a non-key column name — e.g., both `sales.csv` and `stores.csv` have a column called `quantity`. A naive join would silently overwrite one column or create ambiguous `_right` suffixes.

**How the platform handles it:** `MultiFileMerger._resolve_column_conflicts()` detects non-key columns that exist in both the left (merged-so-far) and right DataFrames before each join. Conflicting columns in the right DataFrame are renamed with a `_<filename_stem>` suffix (e.g., `quantity` from `stores.csv` becomes `quantity_stores`). The list of renames is recorded in `MergePreview.column_name_conflicts` and displayed to the user in an expandable section.

**What to watch for:** If many files have overlapping column names, the renamed columns (e.g., `temperature_weather`, `temperature_climate`) can be confusing. Review the conflict list in the merge preview. Column renames only apply to non-key columns — join keys are never renamed.

## 23. Multi-File Upload: Mismatched Granularity

**What happens:** The primary time series is at store-level granularity (`store_id × week`), but an external regressor file is at region-level (`region × week`). A direct join on `[week]` would broadcast region-level features to all stores, which may be correct for weather data but misleading for store-specific regressors.

**How the platform handles it:** The `MultiFileMerger` performs a left join, so all primary rows are preserved regardless of granularity mismatch. When a regressor has no ID column overlap with the primary, it joins on `[time_col]` only — effectively broadcasting global features. When it has partial ID overlap (e.g., `region` but not `store_id`), the join uses the available keys. Low key overlap percentages (< 50%) trigger warnings in `JoinSpec.warnings`, displayed in the merge preview.

**What to watch for:** Broadcast features can introduce multicollinearity if the same value appears across many series. The `RegressorScreen` (run during model fitting) catches near-constant columns via variance threshold, but it's worth reviewing the merge preview to ensure the join makes business sense.
**What to watch for:** A low forecastability score doesn't always mean "don't forecast" — it means the signal-to-noise ratio is poor with available methods. Foundation models (Chronos, TimeGPT) may still produce useful forecasts on low-forecastability series since they bring pre-trained knowledge. Also, forecastability is assessed on historical data; a series that became forecastable after a structural break may score low due to the pre-break noise. Consider running forecastability assessment on post-break data only if structural breaks are detected.

---

## 24. Frontend: API Timeout on Large Datasets

**What happens:** The Next.js frontend sends a request (e.g., backtest or forecast) that takes longer than the browser's default fetch timeout. The user sees a "Network Error" or the request silently fails, even though the backend is still processing.

**How the product handles it:** The API client (`frontend/src/lib/api-client.ts`) uses React Query with configurable `staleTime` and retry logic. Long-running endpoints like `/pipeline/backtest` and `/pipeline/forecast` return immediately with a run ID; the frontend polls for completion. For synchronous endpoints handling large data, the FastAPI backend streams results where possible.

**What to watch for:** If a user uploads a very large file or runs a backtest with many models and series, the initial POST may itself be slow. Consider increasing the proxy timeout if running behind nginx/Caddy. The frontend shows a loading spinner but has no progress bar for synchronous operations.

---

## 25. Frontend: Stale React Query Cache After Config Change

**What happens:** A user changes the backend configuration (e.g., switches frequency from weekly to monthly) but the frontend still shows cached data from the previous configuration — stale leaderboard results, wrong horizon in forecast charts, or outdated series counts.

**How the product handles it:** React Query caches are keyed by endpoint URL and query parameters. When the configuration changes, the API returns different data for the same endpoints, but previously cached responses may persist until `staleTime` expires (default 5 minutes). Critical pages like Backtest Results and Forecast Viewer include the `lob` and config hash in their query keys to differentiate cache entries.

**What to watch for:** If you change config and immediately navigate to a results page, you may see old data. A hard refresh (`Ctrl+Shift+R`) clears the React Query cache. In production, consider reducing `staleTime` for config-sensitive endpoints or adding a "Refresh Data" button that calls `queryClient.invalidateQueries()`.

---

## 26. Frontend: Large Dataset Rendering in Recharts

**What happens:** A forecast chart with thousands of data points (e.g., daily data over 3 years for 50+ series) causes the browser to lag or freeze. Recharts renders SVG elements for each data point, and SVG performance degrades beyond ~5,000 elements.

**How the product handles it:** The Forecast Viewer page downsamples data for chart rendering when the point count exceeds a threshold. The API's `/forecast/compare` endpoint supports a `max_points` parameter that returns pre-aggregated data. Individual series charts are paginated — only the selected series renders at full resolution.

**What to watch for:** Comparison views overlaying multiple series are the worst case for rendering performance. If the browser becomes unresponsive, reduce the number of series shown simultaneously. For very large datasets, the Plotly-based charts (fan chart, sunburst) handle large point counts more efficiently than SVG-based Recharts components.

---

## 27. Post-Validation Failure During Forecast Publication

**What happens:** An enabled post-validation check reports a blocker, or a checker raises an unexpected exception.

**How the product handles it:** `ForecastPipeline` validates before any forecast or manifest write, for both standard and multi-horizon runs. A blocker halts publication when `post_validation.halt_on_blocker` is true; unexpected checker exceptions always propagate when that layer is enabled. Failed validation leaves existing files unchanged, even on a repeat attempt for the same forecast origin. Each run clears its previous validation result and forecast-path state so a failed attempt cannot appear to have succeeded using stale state.

**What to watch for:** `halt_on_blocker` defaults to false, and `min_grade` is warning-only. Set `halt_on_blocker: true` for blocking quality checks; `post_validation.enabled: false` is an explicit opt-out. Forecast keys use both the configured series ID and period, not just the series ID. Local filesystem failures have the separate pre/post-commit contract below; managed Spark/Fabric pipelines do not inherit local bundle atomicity.

---

## 28. Invalid or Stale Forecast Plans

**What happens:** YAML has invalid syntax, a wrong field type (such as `"false"`
instead of a boolean), or an invalid horizon/model. Alternatively, a planner
changes inputs while an earlier backtest or analysis is still running.

**How the product handles it:** Both pipeline upload endpoints validate the config
before constructing the pipeline or its stores and return a useful HTTP 400 error
instead of substituting defaults. Explicit query horizons must be positive
integers (HTTP 422). Valid explicit overrides take precedence over YAML, then
platform defaults. The execution panel clears dependent results on input changes,
ignores stale completions, and forwards only the current winner and horizon.
Responses identify the resolved model, horizon, settings, and their sources.

**What to watch for:** An absent config means defaults; an empty/null uploaded
document is invalid (use `{}` for defaults explicitly). Standalone forecasting
chooses the first configured model, not a remembered or guessed winner.
Invalidation prevents stale UI selection, not cancellation of server work.
Immutable run publication is a separate server-side guarantee. Multi-model backtest results are not
silently reduced to their first row for this single-model handoff.

---

## 29. Missing Authentication Configuration and Stale Browser Identity

**What happens:** A production launcher lacks a signing key, an environment value
such as `"false"` is interpreted as truthy, a caller chooses their own admin
role, or an old request completes after logout or a different login.

**How the product handles it:** One shared `AuthSettings` contract defaults to
authenticated production and rejects missing/weak keys, invalid booleans and
conflicting demo settings before startup. The HTTP token issuer is removed;
only a trusted local operator with the key can issue a short-lived token into a
new restricted file. `/auth/me` verifies signed identity, role, activity and
expiry. The browser holds credentials only in memory, clears protected caches on
replacement/logout/expiry/401, and rejects stale session-owned responses.
Unavailable mode discovery never opens a demo fallback.

**What to watch for:** Explicit demo is intentionally unauthenticated and must
not receive production data or keys. Browser reload requires sign-in again.
Windows private output uses a verified protected DACL, not `chmod`; failed secure
output may leave an empty or private partial file. Stateless tokens have no live
per-user revocation: a copied token remains valid until expiry or coordinated
key rotation/restart. Production containers never seed demo data; the separate
demo Compose override uses its own volume and retains cold/restart safeguards.

---

## 30. Concurrent Runs, Incomplete Writes and Lost Completion Receipts

**What happens:** Two runs share an origin/second, an older attempt completes
last, the process is interrupted while staging, or a disk/response error occurs
just after the run becomes visible.

**How the product handles it:** Each attempt exclusively reserves its own UUID32
directory; all required members are hashed before the fixed manifest commit
marker. Pre-commit errors leave no visible completed run, and prior bytes stay
unchanged. Post-commit errors expose a `committed` or `uncertain` outcome and ID
instead of deleting committed data or falsely promising rollback. Latest uses
attempt start plus ID, so late completion does not regress it; exact-run reads,
stored input/config and prior-run UI selection survive service restart.

**What to watch for:** Inspect the exact ID before repeating an ambiguous write;
retries create new attempts, not exactly-once operations. Do not remove a
markerless directory merely because it looks old: it may have a live owner.
Unknown versions, checksum mismatch or corrupt metadata are explicit integrity
errors, never a healthy empty list or silent old-run fallback. Use the documented
offline backup/restore procedure and the supported local hard-link filesystem
profile; Windows power-loss and managed Delta atomicity are not certified here.

---

## 31. Undefined or Incomparable Validation Arms

**What happens:** Backtests contain different candidate sets in separate declared
competitions, or complete comparison arms have zero actual volume. Observation
WMAPE cells may also be null even when their pooled population score is defined.

**How the product handles it:** Post-validation compares eligible models only
within the same competition and common observation population. It aggregates
raw errors at the actual comparison grain before judging applicability, retaining
zero-actual errors. Single candidates and genuinely undefined complete arms are
reported as unavailable with reasons, counts and a WARNING; the existing neutral
confidence contribution is explicitly labelled unmeasured, not PASS/100.

**What to watch for:** Missing or malformed eligible observations are still
actionable failures. Unexpected checker errors and configured blockers still
prevent publication. No rows are silently dropped, no formula is changed and no
historical artifact is rewritten. A previously saved leaderboard is always read
as stored, not recomputed to make old and new rankings agree.

---

## 32. Incomplete or Ambiguous Hierarchy Reconciliation Inputs

**What happens:** A missing leaf/period, conflicting parent, repeated node key
across levels or absent historical volume can produce a plausible-looking but
incomplete reconciliation. Nanosecond timestamps can also collapse into the same
microsecond wire key even when the original stored keys differ.

**How the product handles it:** The hierarchy API rejects invalid mappings,
duplicate/null keys, incomplete observed-period panels and nonfinite requested
values before calling the existing reconciler. Linear methods reject ambiguous
cross-level keys. Top-down requires earlier, complete history with positive
leaf-group volumes and matching root forecasts; no zero padding or equal-share
fallback is used. CSV/Parquet submicrosecond timestamps are rejected, not rounded.
Full output keys, quantile order and scoped coherence are checked before preview
truncation, and unexpected checker failures return a logged error.

**What to watch for:** Coverage is defined by the periods and hierarchy actually
supplied, not an inferred missing calendar or unknown external SKU catalogue.
Structural WLS/no-residual MinT does not estimate residual covariance; derived
parent totals do not establish independent forecast accuracy. Changing UI inputs
clears old results and invalidates late responses. Stored runs remain unchanged.
