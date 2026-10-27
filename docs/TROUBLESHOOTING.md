# Troubleshooting

Common errors, frequently asked questions, and debugging tips.

---

## Common Errors

### "Missing columns: ['week', 'quantity']"

```
ValueError: [series_id, week, quantity] missing: ['week', 'quantity']
```

**Cause:** Your CSV column names don't match what the platform expects.

**Fix:** Either rename your columns or configure the expected names in YAML:

```yaml
forecast:
  time_column: Date           # match your CSV's date column
  target_column: Sales        # match your CSV's target column
  series_id_column: Store     # match your CSV's ID column
```

---

### "Forecaster 'xxx' not registered"

```
KeyError: "Forecaster 'unknown_model' not registered. Available: [...]"
```

**Cause:** Misspelled model name in config or the model's dependency isn't installed.

**Fix:** Check the available model names:

| Name | Requires |
|------|----------|
| `naive_seasonal` | (built-in) |
| `auto_arima`, `auto_ets`, `auto_theta`, `mstl` | `statsforecast` |
| `lgbm_direct` | `lightgbm`, `mlforecast` |
| `xgboost_direct` | `xgboost`, `mlforecast` |
| `nbeats`, `nhits`, `tft` | `neuralforecast` |
| `chronos` | `chronos` |
| `timegpt` | `nixtla` |
| `croston`, `croston_sba`, `tsb` | `statsforecast` |

---

### "Unsupported frequency 'X'"

```
ValueError: "Unsupported frequency 'X'. Choose from: ['D', 'W', 'M', 'Q']"
```

**Cause:** Invalid frequency code in config.

**Fix:** Use one of: `D` (daily), `W` (weekly), `M` (monthly), `Q` (quarterly).

---

### "No forecast data found for LOB 'retail'"

```
HTTPException 404: "No forecast data found for LOB 'retail'. Expected directory: ..."
```

**Cause:** You're querying the API before running the forecast pipeline.

**Fix:** Run the forecast pipeline first:
```bash
python forecasting-product/scripts/run_forecast.py \
  --config configs/platform_config.yaml --lob retail
```

---

### "Authentication required" / "Invalid or expired token"

**Cause:** The production API has no valid, active operator-signed token. Missing,
invalid or expired tokens return 401; inactive identity and wrong-role requests
return 403. There is no HTTP token issuer.

**Fix:** Ask your trusted operator to issue a fresh token with the required role
using `scripts/issue_token.py`, then paste it into the production login field.
The CLI defaults to one hour (1-24 hours supported). Do not put the token in
command arguments, URLs, logs or browser storage. Reload deliberately signs out;
it does not restore an old localStorage session. See
[Authentication Setup](DEPLOYMENT.md#authentication-setup).

For startup rejection, check the exact production/demo settings and a dedicated
random `JWT_SECRET_KEY`. Missing keys never enable anonymous access. Use an
explicit demo only for isolated synthetic data, not to bypass a production error.

---

### "Permission 'xxx' required"

```
HTTPException 403: "Permission 'run_backtest' required. Your role 'viewer' does not have this permission."
```

**Cause:** Your user role doesn't have the required permission.

**Fix:** Request a token with the appropriate role. See [DEPLOYMENT.md](DEPLOYMENT.md) for the role-permission matrix.

---

### "Unsupported file format: .xlsx"

```
ValueError: "Unsupported file format: .xlsx. Use .parquet or .csv"
```

**Cause:** The platform only accepts CSV and Parquet files.

**Fix:** Save your Excel file as CSV first.

---

### "Claude client not available" / "anthropic package not installed"

```
RuntimeError: "Claude client not available"
RuntimeError: "anthropic package not installed"
```

**Cause:** AI features require the `anthropic` package and an API key.

**Fix:**
```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

AI features degrade gracefully when Claude is unavailable — they return template-based responses using the same data structures. No data is lost.

---

### "PyJWT is required for token operations"

```
RuntimeError: "PyJWT is required for token operations. Install with: pip install PyJWT"
```

**Fix:** `pip install PyJWT`

---

## Frequently Asked Questions

### How many weeks of history do I need?

Minimum history depends on frequency:

| Frequency | Minimum | Recommended |
|-----------|---------|-------------|
| Daily | 90 days | 180+ days |
| Weekly | 52 weeks | 104+ weeks |
| Monthly | 24 months | 36+ months |
| Quarterly | 8 quarters | 12+ quarters |

Foundation models (Chronos, TimeGPT) can work with less history because they don't learn seasonality from scratch.

---

### Can I use daily data?

Yes. Set `frequency: "D"` in your config. The platform auto-adjusts season lengths (7 for daily), lag features, and minimum history requirements.

```yaml
forecast:
  frequency: "D"
  horizon_weeks: 90    # 90-day horizon
```

---

### How do I add a new model?

Register it with the `ForecasterRegistry`:

```python
from src.forecasting.registry import registry
from src.forecasting.base import BaseForecaster

class MyForecaster(BaseForecaster):
    def fit(self, data, config):
        # your training logic
        return self

    def predict(self, horizon, **kwargs):
        # your prediction logic
        return forecast_df

registry.register("my_model", MyForecaster)
```

Then reference it in config:
```yaml
forecast:
  forecasters: [my_model, lgbm_direct]
```

---

### Why are my forecasts flat?

Common causes:
1. **Constant series** — If your data has no variation, models produce flat forecasts. Check for placeholder data.
2. **Insufficient history** — With very short series, models default to the mean. Provide more history or use foundation models.
3. **All-zero series after cleansing** — Aggressive outlier cleansing may remove real demand signals. Check the cleansing report.
4. **Wrong frequency** — Monthly data parsed as weekly creates gaps that get filled with zeros.

---

### Why are some series missing from the forecast output?

Series can be dropped during processing:
1. **Too short** — Below `min_series_length_weeks` (default: 52 for weekly)
2. **All zeros** — Dropped when `drop_zero_series: true`
3. **Model failure** — The model failed to fit a specific series. Check `engine.failures` for details.
4. **Parallel batch failure** — A worker process crashed. Check logs and compare input vs output series counts.

---

### How do I handle sparse/intermittent demand?

Enable sparse detection:

```yaml
forecast:
  sparse_detection: true
  intermittent_forecasters: [croston_sba, tsb]
```

The platform auto-classifies each series using the Syntetos-Boylan-Croston (SBC) matrix:

| Classification | Condition | Routed To |
|---------------|-----------|-----------|
| Smooth | CV² < 0.49, ADI < 1.32 | Regular models |
| Erratic | CV² >= 0.49, ADI < 1.32 | Regular models |
| Intermittent | CV² < 0.49, ADI >= 1.32 | Intermittent models |
| Lumpy | CV² >= 0.49, ADI >= 1.32 | Intermittent models |

---

### Can I use the platform without the dashboard?

Yes. The platform is fully usable via:
- **CLI scripts** — `run_backtest.py`, `run_forecast.py`
- **Python API** — Import classes directly (see Quick Start in README)
- **REST API** — All endpoints available via HTTP

The Next.js frontend is a convenience layer, not a requirement.

---

## Debugging Tips

### Enable structured logging

```yaml
observability:
  log_format: json
  log_level: DEBUG
```

JSON logs include `run_id` for correlating events across a pipeline run.

### Check the pipeline manifest

Every forecast run produces a manifest JSON file with full provenance — what data, config, and model produced the forecast. Look for:
- `validation_warnings` — Data quality issues that were non-blocking
- `outliers_clipped` — How many values were modified by cleansing
- `regressors_dropped` — Features that failed screening
- `champion_model_id` — Which model actually ran

### Use drift alerts for monitoring

Set up webhook alerts to catch accuracy degradation early:

```yaml
observability:
  alerts:
    channels: [webhook]
    webhook_url: "https://hooks.slack.com/services/..."
    min_severity: critical
```

Start with `critical` severity only. Lower to `warning` after tuning thresholds.

### Check for model failures

After a backtest:

```python
failures = engine.get_failure_summary()
if len(failures) > 0:
    print(failures)   # DataFrame with model_name, fold, error_type, message
```

---

## Interpreting Data Quality Reports

The `DataValidator` and `SeriesBuilder` produce quality reports at each pipeline run. Here's how to interpret them.

### Validation Report

The 5-layer validation runs automatically on every dataset:

| Check | What It Catches | Action When Failed |
|-------|----------------|-------------------|
| **Schema** | Missing/wrong-type columns | Fix column names or update config mapping |
| **Duplicates** | Repeated `(series_id, week)` pairs | Deduplicate source data — duplicates cause double-counting |
| **Frequency** | Inconsistent time gaps (e.g., missing weeks) | Enable `fill_gaps: true` in data quality config, or investigate source gaps |
| **Value Range** | Negative demand, extreme outliers | Enable cleansing (`cleansing.enabled: true`) or fix source data |
| **Completeness** | Too many missing weeks per series, too few series | Lower `min_series_length_weeks` or provide more data |

**Severity levels:**
- `error` — Blocks the pipeline if `strict: true`; otherwise logged as warning
- `warning` — Non-blocking; captured in manifest for review

### Cleansing Report

When `cleansing.enabled: true`, the report includes:

```python
# Access after pipeline run
report = builder._last_cleansing_report
# {
#   "outliers_detected": 42,
#   "outliers_clipped": 38,
#   "stockouts_detected": 15,
#   "stockouts_imputed": 15,
#   "series_affected": ["sku_001", "sku_042", ...]
# }
```

| Field | What It Means |
|-------|--------------|
| `outliers_detected` | Values outside IQR/z-score bounds |
| `outliers_clipped` | Values actually modified (clipped to fence) |
| `stockouts_detected` | Zero-runs identified as stockouts (not real zero demand) |
| `stockouts_imputed` | Stockout periods replaced with seasonal average |

**If outliers_clipped is very high (>5% of rows):** Your data may have structural issues — check for unit changes, returns, or data entry errors before relying on automatic cleansing.

### Structural Break Report

When `structural_breaks.enabled: true`:

| Field | What It Means |
|-------|--------------|
| `series_with_breaks` | Number of series with detected structural breaks |
| `total_breaks` | Total breakpoints found across all series |
| `truncated_series` | If `truncate_to_last_break: true`, series cut to post-break data only |

**When to truncate:** If a product was reformulated, repackaged, or repriced, pre-break history may hurt forecasts. Set `truncate_to_last_break: true` for these cases.

### Data Quality Analyzer Report

The overall quality assessment (when `report.enabled: true`):

```python
report = builder._last_quality_report
# {
#   "n_series": 150,
#   "n_short": 12,
#   "n_zero": 3,
#   "n_sparse": 28,
#   "demand_classes": {"smooth": 80, "erratic": 28, "intermittent": 22, "lumpy": 20},
#   "overall_quality": "good"     # good / fair / poor
# }
```

| Quality Rating | Meaning |
|---------------|---------|
| `good` | <5% short series, <10% zero series, <10% missing data |
| `fair` | 5–15% short/zero series, 10–25% missing |
| `poor` | >15% short/zero, >25% missing — results may be unreliable |

### Common Data Quality Patterns

| Pattern | Symptom | Fix |
|---------|---------|-----|
| New product launches | Many short series | Lower `min_series_length_weeks` or use foundation models |
| Seasonal stockouts | Zero-runs in peak periods | Enable stockout imputation |
| Price changes | Structural breaks in volume | Enable break detection + truncation |
| Mixed frequencies | Frequency check failures | Separate daily/weekly/monthly into distinct LOBs |
| Hierarchy mismatches | Orphan series in aggregation | Check product master completeness |

---

## Performance Issues

### Backtests are slow

Distinguish a stalled worker from expensive optimization. A parent waiting in
`as_completed` is not sufficient evidence of a deadlock: inspect worker stacks
and CPU use. Old Linux pools could fork after Polars initialization and block in
`read_ipc`/`collect`; current pools use explicit `spawn`. Active ARIMA likelihood
optimization is a different case and can legitimately take minutes on the full
daily sample. Do not disable tests, models or assertions to hide either case.
CI emits stack diagnostics and uses bounded test/job deadlines; the expensive
M5 setup runs once with isolated copies for its consumers.

- **Reduce models** — Start with 2-3 candidates, not all 18
- **Reduce folds** — `n_folds: 2` instead of 5 (less robust but faster)
- **Skip neural models** — `nbeats`, `nhits`, `tft` are orders of magnitude slower than statistical/ML
- **Enable parallelism** — `n_workers: -1` uses all CPU cores
- **Use batch processing** — `batch_size: 500` to chunk large datasets

### Out of memory

- **Reduce `n_workers`** — Each worker loads a full model + data copy
- **Reduce `batch_size`** — Smaller chunks per worker
- **Use Spark** — `backend: spark` for datasets that don't fit in memory
- **Drop neural models** — They have the largest memory footprint

### Docker container crashes

Check the seeder logs before deleting or recreating data. Cold and restart
launches must use the same absolute data/metrics paths, and `.seeded` is written
only after both pipelines succeed with nonempty forecast output. A failed seed
now exits and reaps its child instead of leaving an unseeded service running.
Keep the named data volume intact; CI separately exercises a fresh-container
restart against an unchanged temporary volume.

- Provision at least **2 GB RAM** for datasets with 1000+ series
- The Rossmann demo dataset (1M rows) runs comfortably in 2 GB
- For larger datasets, increase container memory limits

---

## Environment-Specific Issues

### Microsoft Fabric

**Issue:** DuckDB, neuralforecast, and PySpark are not available in the Fabric Spark runtime.

**What happens:**
- Override store auto-falls back to Parquet (no concurrent write support)
- Neural models (`nbeats`, `nhits`, `tft`) are unavailable
- `pip install` only persists for the session, not across cluster restarts

**Fix:**
- Use `requirements-fabric.txt` (excludes incompatible packages)
- Pin libraries in Fabric workspace settings for persistent installs
- Use a single orchestrator notebook for override management

### Docker volume mounts

**Issue:** Forecast/metric files aren't visible between API and frontend containers.

**Fix:** Both services must share the same data volume:
```yaml
volumes:
  - ./forecasting-product/data:/app/forecasting-product/data
```

### Import errors for optional dependencies

**Issue:** `ImportError` or `ModuleNotFoundError` for packages like `neuralforecast`, `shap`, `holidays`.

**What happens:** The platform uses optional dependency guards — it won't crash on import, but the feature will be unavailable at runtime.

**Fix:** Install the specific package:
```bash
pip install neuralforecast    # for N-BEATS, N-HiTS, TFT
pip install shap              # for SHAP explainability
pip install holidays          # for holiday calendar generation
pip install PyJWT             # for JWT authentication
pip install anthropic         # for AI features
```

---

## Next.js Frontend Issues

### Frontend can't connect to API

**Issue:** Pages show "Network Error" or data doesn't load.

**Fix:**
1. Ensure the FastAPI backend is running (`python forecasting-product/scripts/serve.py --port 8000`)
2. Check `.env.local` has `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. If running on different hosts/ports, check for CORS — the API must allow the frontend's origin

### `npm install` fails

**Issue:** Dependency installation errors.

**Fix:** Ensure Node.js 18+ is installed (`node --version`). Delete `node_modules` and `package-lock.json`, then retry:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Login doesn't work

**Issue:** Login form submits but redirects back or shows error.

**Fix:**
1. Check public `/auth/status`. Unavailable or inconsistent mode discovery deliberately disables sign-in.
2. Production requires a fresh trusted-operator-issued token verified by `/auth/me`; there is no `/auth/token`, password login or SSO provider.
3. Explicit demo requires `AUTH_ENABLED=false`, `API_DEV_MODE=true` and no signing key. A missing key never enables anonymous production access. `NEXTAUTH_*` settings do nothing.

### `npm run build` fails

**Issue:** TypeScript or build errors when running `npm run build`.

**Fix:**
1. Ensure all dependencies are installed: `npm install`
2. Check for TypeScript errors: `npx tsc --noEmit`
3. Clear the Next.js cache: `rm -rf .next` and rebuild
4. Verify Node.js 18+ is installed

### API returns 401 Unauthorized

**Issue:** Pages load but show "Unauthorized" errors when fetching data.

**Fix:**
1. The operator CLI defaults to one-hour tokens (1-24 hours supported). Ask for a fresh token; reloading the browser intentionally loses in-memory credentials.
2. Check that the API and trusted operator use the same protected `JWT_SECRET_KEY`; do not print it while diagnosing.
3. After key rotation, restart every API worker/replica and reissue tokens. Client logout alone does not revoke a copied stateless token.
