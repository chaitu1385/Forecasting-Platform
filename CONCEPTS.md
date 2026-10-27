# Platform Concepts

A "why this exists and when you'd use it" guide for the forecasting product. Each concept is 3-4 sentences — enough to orient a new team member without becoming a textbook.

---

## Forecasting Fundamentals

### Trusted Operator Authentication

A production forecasting API must not give an anonymous caller administrator access
just because signing configuration is missing. This platform instead fails startup
and accepts short-lived JWTs issued by a trusted local operator who controls the
signing key and role assignment; it is not SSO or a user directory. The browser
verifies identity with the API and holds the token only in tab memory, while an
explicitly separate demo can simulate roles with no production credentials.
Stateless tokens cannot be revoked individually here: use expiry or coordinated
key rotation, not client logout, when a copied token must stop working.

*Implementation: `src/auth/settings.py`, `src/auth/issuer.py`, `scripts/issue_token.py`, `frontend/src/providers/auth-provider.tsx`*

### Walk-Forward Backtesting

A random train/test split on time series leaks future information into training — the model sees patterns it shouldn't know yet. Walk-forward validation avoids this by using an expanding window: each fold trains on all data up to a cutoff, then predicts the next *n* periods forward. The platform's `WalkForwardCV` generates multiple folds walking backward from the most recent date, optionally inserting a gap to simulate production lag. This is the only honest way to estimate how a model will perform on genuinely unseen future data.

*Implementation: `src/backtesting/cross_validator.py` — `WalkForwardCV`*

### Champion Model Selection

No single forecasting model wins on every series. AutoARIMA might excel on smooth trends while LightGBM captures promotion effects better. The `ChampionSelector` runs all candidate models through walk-forward backtesting, then picks the best performer per series (or per horizon bucket) by primary metric (typically WMAPE). It also computes inverse-WMAPE ensemble weights, so if no single champion dominates, a blend of top models can be used instead.

*Implementation: `src/backtesting/champion.py` — `ChampionSelector`*

### Forecast Value Added (FVA)

Adding complexity doesn't always improve forecasts — an expensive ML model can actually perform *worse* than a simple seasonal naive baseline. FVA measures the error reduction (or increase) at each layer of the forecasting cascade: Naive → Statistical → ML → Human Override. A layer that consistently shows FVA < -2 percentage points is classified as "DESTROYS_VALUE" and should be removed. This prevents organizations from paying for model layers that make forecasts worse.

*Implementation: `src/metrics/fva.py` — `compute_fva_cascade()`, `classify_fva()`*

### Ensemble Forecasting

Individual models have different failure modes — ARIMA struggles with regime changes, tree models struggle with extrapolation. Blending multiple forecasters reduces variance and hedges against any single model's weaknesses. The `WeightedEnsembleForecaster` combines models using inverse-WMAPE weights: models that performed better in backtesting get more influence. The weights are normalized to sum to 1, and both point forecasts and quantile intervals are blended.

*Implementation: `src/forecasting/ensemble.py` — `WeightedEnsembleForecaster`*

---

## Hierarchical & Reconciliation

### Hierarchical Forecasting

When you forecast at multiple levels of a product hierarchy (total company → category → subcategory → SKU), the forecasts won't be coherent — SKU-level forecasts don't sum to the category total, and the category totals don't sum to the company total. This creates contradictory plans across the organization. Hierarchical forecasting solves this by building a tree structure, generating base forecasts at each level, then reconciling them so all levels are mathematically consistent. The summing matrix *S* encodes which leaves roll up to which parents.

*Implementation: `src/hierarchy/tree.py` — `HierarchyTree`; `src/hierarchy/aggregator.py` — `HierarchyAggregator`*

### MinT Reconciliation

Top-down allocation loses SKU-level signal. Bottom-up aggregation loses top-level stability. Simple averaging splits the difference but ignores correlation structure. MinT (Minimum Trace) reconciliation uses Ledoit-Wolf shrinkage estimation of the forecast error covariance matrix to find the statistically optimal linear combination of base forecasts across all hierarchy levels. It dominates simpler methods (OLS, WLS) especially in large hierarchies where sample covariance is ill-conditioned. The platform also supports bottom-up, top-down, middle-out, OLS, and WLS for comparison.

*Implementation: `src/hierarchy/reconciler.py` — `Reconciler.mint()`*

---

## Intermittent Demand

### Sparse / Intermittent Demand Detection

Standard forecasting models (ARIMA, ETS, LightGBM) assume roughly continuous demand. When a series is 60-80% zeros — common for spare parts, slow-moving SKUs, or long-tail products — these models produce nonsensical results: negative forecasts, wildly oscillating predictions, or flat zeros. The platform uses the Syntetos-Boylan-Croston (SBC) classification matrix, which plots Average Demand Interval (ADI) against squared Coefficient of Variation (CV²) to classify each series as smooth, intermittent, erratic, or lumpy. Series classified as intermittent or lumpy are automatically routed to specialized forecasters.

*Implementation: `src/series/sparse_detector.py` — `SparseDetector` (thresholds: ADI=1.32, CV²=0.49)*

### Croston & TSB Methods

You can't fit an ARIMA model to a series that's 80% zeros — the model wastes its degrees of freedom trying to explain the zero/non-zero pattern instead of the demand magnitude. Croston's method separates the problem: one exponential smoothing model for demand *size* (when demand occurs) and another for the *inter-arrival interval* (how often demand occurs). The forecast is size divided by interval. TSB (Teunter-Syntetos-Babai) improves on this by updating the demand probability at *every* period, not just when demand occurs — this lets it detect obsolescence (a product going from slow to dead) much faster than standard Croston.

*Implementation: `src/forecasting/intermittent.py` — `CrostonForecaster`, `CrostonSBAForecaster`, `TSBForecaster`*

---

## Prediction Intervals & Constraints

### Conformal Prediction / Calibration

A model that claims "80% prediction interval" but actually covers only 65% of outcomes will cause systematic safety stock miscalculation — you'll understock because you trusted the interval. Conformal prediction fixes this without distributional assumptions: it computes nonconformity scores on a calibration set (how wrong was the model?), then widens the prediction intervals symmetrically until empirical coverage matches the nominal target. The platform's calibration report checks whether actual values fall within stated intervals, and `apply_conformal_correction()` adjusts intervals that are too narrow or too wide.

*Implementation: `src/evaluation/calibration.py` — `compute_calibration_report()`, `apply_conformal_correction()`*

### Constrained Demand Estimation

Raw statistical forecasts ignore physical reality: a factory can't produce 2 million units if capacity is 1.5 million, and a category can't exceed its budget allocation. The `ConstrainedDemandEstimator` wraps any base forecaster and applies post-prediction constraints: element-wise bounds (per-SKU min/max capacity) and aggregate budget caps (total across all SKUs per period). When aggregate demand exceeds the budget, it redistributes proportionally. Quantile intervals are also constrained, with a monotonicity pass ensuring p10 ≤ p50 ≤ p90 after clipping.

*Implementation: `src/forecasting/constrained.py` — `ConstrainedDemandEstimator`*

---

## Data Quality

### Demand Cleansing

Raw point-of-sale data is not true demand. Stockouts suppress demand (zero sales ≠ zero demand), data entry errors create outlier spikes, and promotional periods inflate baselines. The `DemandCleanser` applies a multi-step pipeline: (1) exclude known bad periods before computing statistics, (2) detect outliers via IQR or z-score methods, (3) correct outliers by clipping to fence values or interpolating, (4) detect stockout patterns (consecutive zeros followed by recovery) and impute using seasonal or interpolation methods. Every correction is logged in a `CleansingReport` for auditability.

*Implementation: `src/data/cleanser.py` — `DemandCleanser`*

### Regressor Screening

Adding every available feature (weather, price, promotions, macroeconomic indicators) to a model sounds helpful but often makes forecasts worse. Near-zero-variance features add noise. Highly correlated features create multicollinearity that destabilizes coefficients. Irrelevant features dilute the signal. The regressor screening pipeline applies three filters before training: (1) variance threshold to drop flat features, (2) pairwise correlation check to warn about redundant features, and (3) optional mutual information scoring to assess nonlinear relevance. Only features that pass all screens reach the model.

*Implementation: `src/data/regressor_screen.py` — `screen_regressors()`*

### Automated Data Profiling

When onboarding a new dataset, you need to answer a dozen configuration questions: which column is the time axis? Which is the target? What's the data frequency? Are there hierarchies? What's the data quality? The `DataAnalyzer` automates this by running schema detection (column name and dtype heuristics), hierarchy discovery (cardinality analysis), frequency inference, and quality profiling — then recommending a `PlatformConfig` that can be used as-is or fine-tuned. This eliminates manual config guesswork and catches data issues before they reach the forecasting models.

*Implementation: `src/analytics/analyzer.py` — `DataAnalyzer`*

### Data Validation

Garbage in, garbage out is the single most common forecasting failure mode. A single duplicated row can double-count a week's sales. A missing frequency check lets monthly data slip into a weekly pipeline. Negative values from return adjustments can flip model coefficients. The `DataValidator` runs sequential checks — schema enforcement, duplicate detection, frequency consistency, value range validation, and completeness assessment — before any data reaches the forecasting models. Critical issues (duplicates, schema violations) raise errors that block the pipeline; minor issues (small gaps, warnings) are logged but don't stop execution.

*Implementation: `src/data/validator.py` — `DataValidator`*

---

## Product Lifecycle

### SKU Transitions & New Product Mapping

New products with no sales history are the hardest forecasting problem in retail. You can't fit a model to data that doesn't exist. The platform addresses this through predecessor identification: attribute matching (same category, size, brand), fuzzy name matching (rapidfuzz), sales curve fitting (S-curve proportion transition), and temporal comovement (cross-correlation of sales patterns). Once a predecessor is identified, the `TransitionEngine` stitches the old SKU's history onto the new one, handling three scenarios: already launched (stitch), launching within forecast horizon (ramp down old, ramp up new), and beyond horizon (forecast old only). For truly novel products with no predecessor, foundation models (Chronos, TimeGPT) provide zero-shot forecasts.

*Implementation: `src/sku_mapping/pipeline.py` — `SKUMappingPipeline`; `src/series/transition.py` — `TransitionEngine`*

### Structural Break Detection

A store renovation, a competitor opening nearby, or a pandemic can create a permanent level shift in demand. If the model trains on pre-break data, it will forecast the old regime. The `StructuralBreakDetector` identifies these breaks using CUSUM (zero-dependency cumulative sum method) or PELT (Pruned Exact Linear Time, via the `ruptures` library). Once detected, the platform truncates history to keep only post-break data, ensuring models train on the current regime. The within-segment standard deviation is used for significance testing to avoid break-inflated variance.

*Implementation: `src/series/break_detector.py` — `StructuralBreakDetector`*

---

## Governance & Observability

### Drift Detection

A model that was 95% accurate last quarter can silently degrade as demand patterns shift, new competitors enter, or data pipelines break. Without monitoring, you won't know until the business complains about bad forecasts — weeks or months too late. The `ForecastDriftDetector` monitors three signals continuously: accuracy drift (current WMAPE vs baseline ratio), bias drift (systematic over/under-forecasting), and volume anomaly (actual demand z-score). Configurable warning and critical thresholds trigger `DriftAlert` objects that can drive automated retraining or human review.

*Implementation: `src/metrics/drift.py` — `ForecastDriftDetector`*

### Pipeline Provenance

When a forecast looks wrong, the first question is "what produced this?" Without provenance, debugging is archaeology. The `PipelineManifest` is a JSON sidecar written alongside every forecast output, capturing: input data hash, cleansing summary, validation results, regressor screening decisions, config hash, champion model name, backtest WMAPE, and forecast metadata. This creates a complete audit trail from raw data to final forecast, enabling reproducibility and root-cause analysis.

*Implementation: `src/pipeline/manifest.py` — `PipelineManifest`, `build_manifest()`*

### Model Governance & Audit

Regulated industries (financial services, pharma, food safety) require documentation of what model was used, when it was trained, on what data, and how it performed. The governance module provides three primitives: `DriftDetector` (compares live accuracy to backtest baseline), `ModelCard` (captures training window, series count, feature set, metrics, config hash), and `ForecastLineage` (append-only log of which model produced each forecast run). Together, these create the audit infrastructure needed for model risk management.

*Implementation: `src/analytics/governance.py` — `ModelCard`, `ModelCardRegistry`, `ForecastLineage`*

### Planner Overrides

Automated forecasts can't know about next month's product launch, the competitor's store closure, or the supply chain disruption that will redirect demand. Human planners carry context that no model can learn from historical data alone. The `OverrideStore` (backed by DuckDB) lets planners record manual adjustments with full metadata: old SKU → new SKU proportion, scenario tags, ramp shape, and approval status. FVA analysis then measures whether these human overrides actually improved the forecast — closing the feedback loop between judgment and outcomes.

*Implementation: `src/overrides/store.py` — `OverrideStore`*

### Multi-Frequency Support

A weekly-only platform can't serve a monthly S&OP process or daily replenishment cycle. But frequency changes everything: season length (52 for weekly, 12 for monthly), default lags, minimum series length, and date arithmetic all depend on the data frequency. Rather than scattering frequency-specific logic across 20+ modules, the platform uses `FREQUENCY_PROFILES` as a single source of truth — a dict mapping `"D"`, `"W"`, `"M"`, `"Q"` to all frequency-dependent parameters. Every model, backtester, validator, and data processor reads from this profile, so switching frequency is a single YAML config change.

*Implementation: `src/config/schema.py` — `FREQUENCY_PROFILES`, `get_frequency_profile()`, `freq_timedelta()`*

---

## Analytics & Interpretation

### Forecastability Assessment

Not all series can be forecast equally well — a stable seasonal product is inherently more predictable than a lumpy spare part or a trend-driven fashion item. The `ForecastabilityAnalyzer` computes per-series statistical signals: coefficient of variation (CV), approximate entropy (ApEn), spectral entropy, signal-to-noise ratio (SNR), and trend/seasonal strength. These are combined into a forecastability score that informs model selection, prediction interval width, and expectations setting. Low-forecastability series get wider intervals and simpler models; high-forecastability series justify more complex approaches.

*Implementation: `src/analytics/forecastability.py` — `ForecastabilityAnalyzer`*

### Causal & Econometric Analysis

Correlation isn't causation — a price change that coincides with a seasonal shift produces misleading elasticity estimates. The `CausalAnalyzer` addresses this by detrending and deseasonalizing before computing price elasticity, detecting cannibalization via lagged cross-correlation between competing products, and measuring promotional lift through controlled before/after comparison. These econometric insights help planners understand *why* demand changed, not just *that* it changed.

*Implementation: `src/analytics/causal.py` — `CausalAnalyzer`*

### LLM-Powered Interpretation

Raw statistical output (p-values, entropy scores, trend coefficients) isn't actionable for business users who need to make S&OP decisions. The `LLMAnalyzer` sends analysis reports to Claude for plain-language interpretation: key findings, hypotheses about demand drivers, model selection rationale, and risk identification. It gracefully degrades to a stub response when `ANTHROPIC_API_KEY` is unset, ensuring pipelines never hard-depend on LLM availability.

*Implementation: `src/analytics/llm_analyzer.py` — `LLMAnalyzer`*

---

## Distributed Execution & Portability

### Batch Inference & Parallelism

Per-series sequential execution doesn't scale — fitting 10,000 series one-at-a-time on a single core wastes available compute. The `BatchInferenceRunner` partitions series into groups and dispatches them to a `ProcessPoolExecutor`, using Polars IPC serialization for zero-copy inter-process transfer. ML models (LightGBM, XGBoost) naturally train one model across all series in a batch, while statistical models benefit from I/O batching. The `ParallelismConfig` controls worker count, batch size, and backend (`"local"`, `"spark"`, `"ray"`), and flows through to Nixtla's `n_jobs` and `num_threads` parameters.

*Implementation: `src/pipeline/batch_runner.py` — `BatchInferenceRunner`; `src/config/schema.py` — `ParallelismConfig`*

### Pipeline Observability

When a production forecast pipeline runs weekly across thousands of series, "something went wrong" is not actionable. Structured logging with correlation IDs (`run_id`) lets you trace every log line, metric, and alert back to a specific pipeline execution. The `MetricsEmitter` records timing (model fit/predict duration), counts (series processed, errors), and gauges (forecast rows) to a pluggable backend (log or StatsD). The `AlertDispatcher` routes drift alerts to Slack/Teams webhooks. Together, these make the difference between "the forecast is wrong" and "model X failed on fold 2 of LOB Y at 3:42am, here's the error."

*Implementation: `src/observability/context.py` — `PipelineContext`; `src/observability/metrics.py` — `MetricsEmitter`; `src/observability/alerts.py` — `AlertDispatcher`*

### Compute Portability

The platform's business logic (forecasting, backtesting, reconciliation) should be deployable on a laptop, a Spark cluster, or Microsoft Fabric without code changes. This requires separating the compute layer (how models are distributed) from the data layer (where data lives) from the business logic (what models do). The `FabricNotebookAdapter` wraps the Spark + Fabric infrastructure into a one-liner setup class. The `ParquetOverrideStore` replaces DuckDB when it's unavailable in constrained runtimes. The `requirements-fabric.txt` pins only packages available in Fabric's default environment. Each layer can be swapped independently.

*Implementation: `src/fabric/notebook_adapter.py` — `FabricNotebookAdapter`; `src/overrides/store.py` — `get_override_store()`*
## AI & Intelligence

### AI-Native Features

Forecasting platforms generate data (metrics, drift alerts, leaderboards) that planners must manually interpret. AI-native features use Claude to automate this interpretation: answering natural-language questions about forecasts, triaging drift alerts by business impact, recommending config changes from backtest results, and generating S&OP executive commentary. All four capabilities are API endpoints that send structured platform data to Claude with domain-specific prompts, then parse the response into actionable dataclasses. When Claude is unavailable (no API key or network issue), every feature gracefully degrades to a template-based fallback so the platform never breaks.

*Implementation: `src/ai/` — `NaturalLanguageQueryEngine`, `AnomalyTriageEngine`, `ConfigTunerEngine`, `CommentaryEngine`*

## Product & UX

### Next.js Frontend

A forecasting product that only data scientists can operate is incomplete — demand planners need to see forecasts, compare models, and review exceptions; platform admins need to monitor health and drift. The Next.js frontend provides a browser-based interface with eight pages mapped to four user personas: Data Onboarding (upload data, assess forecastability, get a recommended config), Series Explorer (SBC classification, structural breaks, cleansing audit, AI Q&A), SKU Transitions (predecessor matching, planner overrides), Hierarchy Manager (tree visualization with sunburst, reconciliation), Backtest Results (model leaderboard with FVA cascade), Forecast Viewer (interactive fan chart with seasonal decomposition), Platform Health (drift alerts, pipeline manifests, compute cost), and S&OP Meeting (AI commentary, cross-run comparison, governance, BI export). It communicates with the FastAPI backend over REST, using TypeScript, Tailwind CSS, React Query, and role-based authentication. All features are backed by live API endpoints.

*Implementation: `frontend/` — Next.js 15 App Router, `src/app/` (pages), `src/components/` (charts, AI panels, layout, domain panels for forecast/governance/pipeline/sku), `src/hooks/` (React Query + useAsyncOperation), `src/lib/` (API client, auth, types)*

### Multi-File Upload Classification

Real-world forecasting projects rarely have a single tidy CSV — sales history lives in one extract, store attributes in another, promotions in a third, and weather data from an external API. Asking users to pre-join everything is error-prone and kills the onboarding experience. The `FileClassifier` auto-detects each file's role (time series, dimension lookup, external regressor) using heuristics on column types, naming patterns, and cardinality, then `MultiFileMerger` detects join keys and produces a merged DataFrame for analysis. The Data Onboarding page exposes this as an interactive confirmation step: users upload N files, review the auto-detected roles, see a merge preview with join stats, and then run the full analysis on the combined data.

*Implementation: `src/data/file_classifier.py` — `FileClassifier`, `src/data/file_merger.py` — `MultiFileMerger`*
