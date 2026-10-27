# Quick Start

Get the forecasting product running in your browser. Pick one path:

## Path A — Docker (recommended)

No Python setup needed. Just Docker.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose up
```

Open **http://localhost:3000** in your browser. That's it.

On the first boot, wait for the API log's **Seed complete** message before using
the preloaded demo results. Restarts reuse the `api-data` volume without
retraining the demo. If seeding fails, inspect the container logs; do not delete
the volume as a workaround.

## Path B — Local Python

Requires Python 3.8+.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r forecasting-product/requirements.txt
python forecasting-product/scripts/serve.py --port 8000 --data-dir data/
```

Then, in a separate terminal:

```bash
cd forecasting-product/frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

## What to do next

Once the app is open, follow these eight pages in order:

### 1. Data Onboarding

Click **Data Onboarding** in the sidebar. Then click **Use sample data** — no download needed, the platform generates a built-in retail dataset automatically.

You'll see: schema detection (columns, frequency, date range), a forecastability gauge, demand classification, and a recommended model configuration. Click **Download Config YAML** to save it.

<!-- screenshot: Data Onboarding page with forecastability gauge and config -->

### 2. Series Explorer

Click **Series Explorer** to see demand classification (SBC scatter: smooth, intermittent, erratic, lumpy), structural break detection, data quality audit, and cleansing before/after views. Use the AI Q&A panel to ask questions about individual series.

### 3. SKU Transitions

Click **SKU Transitions** to manage new/discontinued product mapping. The pipeline finds predecessors using attribute matching, naming conventions, curve fitting, and temporal co-movement. Planners can review and override mappings.

### 4. Hierarchy Manager

Click **Hierarchy Manager** to visualize the product/location hierarchy tree, configure aggregation levels, and select reconciliation methods (bottom-up, top-down, MinT, OLS, WLS).

### 5. Run a backtest

After onboarding, run a backtest to evaluate model performance. The command depends on how you started the platform:

**Docker (Path A):**
```bash
docker compose exec api python forecasting-product/scripts/run_backtest.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob uploaded
```

**Local Python (Path B):**
```bash
python forecasting-product/scripts/run_backtest.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob uploaded
```

Then switch to the **Backtest Results** page (page 5) in the sidebar to see the model leaderboard, FVA cascade chart, and per-series champion map.

<!-- screenshot: FVA cascade bar chart with ADDS_VALUE / DESTROYS_VALUE annotations -->

### 6. Run a forecast

Similarly, generate predictions:

**Docker (Path A):**
```bash
docker compose exec api python forecasting-product/scripts/run_forecast.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob uploaded
```

**Local Python (Path B):**
```bash
python forecasting-product/scripts/run_forecast.py \
  --config forecasting-product/configs/platform_config.yaml \
  --lob uploaded
```

Then switch to the **Forecast Viewer** page (page 6) to see an interactive chart with P10/P90 confidence intervals. Add actuals to overlay historical data and see the seasonal decomposition (trend, seasonal, residual).

<!-- screenshot: Fan chart with actuals overlay -->

### 7. Platform Health

Monitor pipeline runs, drift alerts, data quality, and compute cost. Drift alerts are colour-coded by severity (warning, critical).

<!-- screenshot: Drift alerts table with severity colouring -->

### 8. S&OP Meeting

The final page generates AI executive commentary, enables cross-run forecast comparison, shows model governance (model cards, lineage), and provides BI export for downstream tools.

## Optional: AI features

Set `ANTHROPIC_API_KEY` to enable Claude-powered features (natural-language explanations, config recommendations, executive commentary). Everything works without it — AI features degrade gracefully.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Next steps

- [README.md](README.md) — full architecture and module reference
- [CONCEPTS.md](CONCEPTS.md) — why each component exists
- [EDGE_CASES.md](EDGE_CASES.md) — failure modes and how the platform handles them
