# Quick Start

Get the forecasting product running in your browser. Pick one path:

## Path A — Explicit local Docker demo

No Python setup needed. Just Docker. Use a fresh shell with no production
`JWT_SECRET_KEY`: this mode is unauthenticated and is only for synthetic data.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build
```

Open **http://localhost:3000** and use the explicitly labelled demo username/role
selector. Both ports are loopback-only. For production authentication and private
operator-issued token login, use [Authentication Setup](docs/DEPLOYMENT.md#authentication-setup).
Plain `docker compose up` is production mode and refuses startup without a valid key.

On the first boot, wait for the API log's **Seed complete** message before using
the preloaded demo results. Restarts reuse the separate `api-demo-data` volume without
retraining the demo. If seeding fails, inspect the container logs; do not delete
the volume as a workaround.

## Path B — Explicit local Python demo

Use **CPython 3.12.16** and **Node.js 24.22.1 / npm 11.19.1**.
The [runtime profiles](docs/DEPLOYMENT.md#supported-runtimes-and-locked-profiles)
also define explicit Python 3.11/3.13 installations; do not mix their lock files.

```bash
git clone https://github.com/chaitu1385/Forecasting-Platform.git
cd Forecasting-Platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --require-hashes --only-binary=:all: -r forecasting-product/requirements/py312/api.txt
python forecasting-product/scripts/serve.py --demo --port 8000 --data-dir data/
```

Then, in a separate terminal:

```bash
cd forecasting-product/frontend
npm ci
npm run dev
```

Open **http://localhost:3000** in your browser.

`--demo` binds loopback by default and rejects conflicting authentication
environment settings or a supplied signing key. Without it, the API requires
production signing configuration. Reloading the browser deliberately signs out;
credentials and identities are never restored from localStorage.

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
docker compose -f docker-compose.yml -f docker-compose.demo.yml exec api python forecasting-product/scripts/run_backtest.py \
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
docker compose -f docker-compose.yml -f docker-compose.demo.yml exec api python forecasting-product/scripts/run_forecast.py \
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
