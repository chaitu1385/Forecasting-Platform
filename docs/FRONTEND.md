# Frontend Architecture

Next.js 15 frontend for the Forecasting Product, providing a workflow-driven UI for S&OP forecasting operations.

## Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Framework | Next.js (App Router) | 15.1 |
| Language | TypeScript | 5.7 |
| UI | React | 19 |
| Styling | Tailwind CSS | 3.4 |
| Primitives | Radix UI | Various |
| Charts | Recharts + Plotly.js | 2.15 / 2.35 |
| Server State | TanStack React Query | 5.62 |
| Tables | TanStack React Table | 8.20 |
| Auth | Server-verified operator JWT / explicit demo | Per-tab memory only |
| Icons | Lucide React | 0.468 |
| Unit Tests | Vitest + Testing Library | 4.1 / 16.3 |
| E2E Tests | Playwright | 1.58 |

## Provider Hierarchy

The root layout (`src/app/layout.tsx`) wraps the entire app in three providers:

```
QueryProvider          -- TanStack React Query (staleTime: 5min, retry: 1)
  AuthProvider         -- JWT auth state, login/logout, permission checks
    LobProvider        -- Selected LOB (persisted to localStorage)
      AuthBoundary     -- No protected page mount before sign-in
        <Sidebar />
        <Header />
        <main>{children}</main>
```

## Pages (10 routes)

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Home page |
| `/login` | Login | Production operator-token entry; role simulation only in explicit demo |
| `/data-onboarding` | Data Onboarding | File upload, auto-analysis, config generation, backtest-to-forecast execution |
| `/series-explorer` | Series Explorer | SBC classification, break detection, cleansing audit, regressor screening |
| `/hierarchy` | Hierarchy | Tree building, sunburst visualization, aggregation, reconciliation |
| `/backtest` | Backtest Results | Pipeline execution, leaderboard, FVA, calibration, SHAP, AI config tuner |
| `/forecast` | Forecast Viewer | Fan chart, decomposition, comparison, constraints, NL query |
| `/sku-transitions` | SKU Transitions | Phase 1/2 SKU mapping, planner override management |
| `/sop` | S&OP Meeting | AI commentary, model cards, lineage, BI export |
| `/health` | Platform Health | Drift monitoring, alert triage, audit log, cost tracking |

## Component Structure

```
src/components/
  ai/                  # 5 components — AI-powered panels
    nl-query-panel     # Natural language Q&A about series
    triage-panel       # Drift alert triage with business impact scoring
    config-tuner-panel # AI-recommended config changes
    commentary-panel   # Executive S&OP commentary generation
    confidence-badge   # High/Medium/Low confidence indicator
  charts/              # 12 components — Data visualization
    time-series-line   # Basic time series line chart (Recharts)
    fan-chart          # Forecast with prediction intervals
    leaderboard-bar    # Model accuracy comparison bars
    fva-cascade        # FVA waterfall chart
    drift-histogram    # Drift alert distribution
    calibration-plot   # Prediction interval calibration
    hierarchy-sunburst # Hierarchy tree sunburst (Plotly)
    demand-class-donut # SBC demand class distribution
    forecastability-gauge # Forecastability score gauge
    break-timeline     # Structural break visualization
    cleansing-overlay  # Before/after cleansing overlay
    ramp-shape         # SKU ramp shape visualization
  data/                # 3 components — Data management
    file-upload        # Drag-and-drop file upload
    data-table         # TanStack Table with sorting/filtering
    config-viewer      # YAML config display
  forecast/            # 4 components — Forecast analysis
    decomposition-panel      # STL trend/seasonal/residual
    comparison-panel         # Model vs external forecast
    constrained-forecast-panel # Capacity/budget constraint application
    cross-run-comparison-panel # Cross-run forecast comparison
  governance/          # 3 components — Model governance
    model-cards-panel  # Model card registry viewer
    lineage-panel      # Forecast lineage history
    bi-export-panel    # BI report export (CSV/Parquet)
  layout/              # 3 components — App shell and auth boundary
    auth-boundary      # Wait for verified identity before protected content
    sidebar            # Navigation sidebar with workflow steps
    header             # Top bar with LOB selector, user menu
  pipeline/            # 2 components — Pipeline operations
    multi-file-panel   # Multi-file upload with auto-classification
    pipeline-execution-panel # Backtest/forecast pipeline runner
  shared/              # 6 components — Reusable utilities
    error-boundary     # React error boundary
    loading-skeleton   # Skeleton loading states
    metric-card        # KPI card with trend indicator
    no-data-guide      # Empty state with next-step guidance
    coming-soon        # Placeholder for unreleased features
    workflow-nav       # Step-by-step workflow navigation
  sku/                 # 2 components — SKU management
    sku-mapping-panel  # Phase 1/2 SKU transition mapping
    override-management-panel # Override CRUD with ramp shapes
```

## API Client

The centralized API client (`src/lib/api-client.ts`) provides typed methods for all 41 backend endpoints.

**Features:**
- Verified JWT injection from per-tab memory, never persistent browser storage
- Server-verified expiry, logout/401 cache clearing, and stale-session response rejection
- Retry logic: 3 retries with exponential backoff (1s, 2s, 4s)
- Retryable status codes: 502, 503, 504, 429
- FormData handling for file uploads (auto Content-Type boundary)
- Custom `ApiError` class with HTTP status code

**Usage:**
```typescript
import { api } from "@/lib/api-client";

// Typed response — ForecastResponse
const forecasts = await api.getForecasts("retail", { horizon: 12 });

// File upload — FormData handled automatically
const analysis = await api.analyze(file, "my_lob", true);

// AI endpoint
const answer = await api.aiExplain({ series_id: "SKU_001", question: "Why is this forecast high?", lob: "retail" });
```

## State Management

| Concern | Mechanism | Details |
|---------|-----------|---------|
| Server state | React Query | `staleTime: 5min`, `retry: 1`, `refetchOnWindowFocus: false` |
| Auth state | `AuthProvider` context | Verified identity and in-memory credential; no restore/persistence |
| LOB selection | `LobProvider` context | Persisted to `localStorage`, `useLob()` hook |
| Async operations | `useAsyncOperation` hook | Loading/error/result state; only the latest operation may update state, and reset/unmount invalidates pending completions |

### Pipeline Intent

`PipelineExecutionPanel` sends the current backtest's `champion_model` and
`execution.horizon_periods` with the same YAML to `api.runForecast`. Without a
backtest it omits these overrides, allowing the API to resolve the first configured
model and configured horizon. The optional YAML upload replaces the generated
config without parsing or merging it in JavaScript. Empty uploaded YAML is still
sent and rejected by the API rather than treated as absent.

Changes to the actuals file, LOB, generated YAML, or uploaded config invalidate both
operations. Starting another backtest invalidates the old winner and forecast.
`useAsyncOperation` uses generation ownership to ignore out-of-order successes,
errors, and loading updates after reset. This does not cancel already-running
server computation. Data Onboarding associates analysis results with the
submitted file/LOB/AI settings, so stale analysis cannot supply a new plan.

`PipelineRunResponse.execution` is a required typed contract matching the API's
`PipelineExecutionSettings` and `EffectivePipelineConfig`. Forecast results show
the effective model, horizon and frequency; both results expose an expandable,
allowlisted settings summary. Configured paths, webhook addresses, AI credentials,
and arbitrary metadata are not included. Contract and state regressions live in
`e2e/pipeline-execution.test.tsx` and `e2e/data-onboarding-state.test.tsx`.

### Isolated Real-API Regression

`playwright/integration/forecast-intent.spec.ts` exercises real analysis, a
Croston-winning backtest, a five-period forecast, visible effective settings, and
invalid-config no-write behavior. It skips unless `FORECAST_INTENT_API_URL` and
`FORECAST_INTENT_DATA_DIR` explicitly identify the dedicated loopback fixture.
It never calls provider models or uses real datasets.

Start the fixture with core/API dependencies (plus `uvicorn`) using a **new**
temporary output directory:

```text
python playwright/fixtures/forecast_intent_server.py --data-dir <new-temp-directory> --frontend-origin http://localhost:3000
```

In a separate frontend terminal, set `NEXT_PUBLIC_API_URL` and
`FORECAST_INTENT_API_URL` to `http://127.0.0.1:18118`, and
`FORECAST_INTENT_DATA_DIR` to that same absolute temporary directory. Run
`npm run test:pw -- playwright/integration/forecast-intent.spec.ts`. The existing
Playwright config starts the frontend on port 3000; ensure any reused dev server
has that API URL. A Playwright Chromium installation is required. The test permits
only the UI and fixture API origins and verifies the fixture marker and API title.
Stop the fixture afterward and delete only its dedicated temporary output directory.
Screenshots, traces and request/response evidence are written to Playwright results.

## Custom Hooks

| Hook | File | Purpose |
|------|------|---------|
| `useAsyncOperation<T>` | `hooks/use-async-operation.ts` | Generic async state (loading, error, result, run, reset) |
| `useForecast` | `hooks/use-forecast.ts` | Forecast data fetching |
| `useLeaderboard` | `hooks/use-leaderboard.ts` | Model leaderboard data |
| `useDrift` | `hooks/use-drift.ts` | Drift alert data |
| `useAnalyze` | `hooks/use-analyze.ts` | File upload analysis |
| `useAudit` | `hooks/use-audit.ts` | Audit log querying |

## Auth Flow

Production sign-in is an operational MVP for trusted-operator-issued JWTs, not
NextAuth, password authentication, SSO or a user directory. See
[operator provisioning](DEPLOYMENT.md#authentication-setup) for private output,
expiry, rotation and the absence of immediate per-user revocation.

1. Public `/auth/status` declares production or explicit demo mode. Invalid or unavailable discovery disables sign-in.
2. Production accepts a token in a password-style field; only explicit demo shows a username/role selector.
3. `/auth/me` verifies the candidate before `AuthProvider` commits server-provided identity, permissions and expiry. The UI never chooses production claims.
4. Credentials live only in per-tab module memory. The input is cleared on submission; no credential is persisted, sent in a URL, or placed in React Query.
5. Replacement login, logout, expiry, pagehide and current-session 401 clear identity and protected caches. Legacy localStorage auth is deleted without restoration.
6. API requests capture their session owner. Stale responses cannot populate a newer identity's cache or log that identity out. Neither API nor query retry treats 401/403 as transient.
7. `AuthBoundary` prevents protected page mounting before sign-in. SPA navigation preserves memory; full reload intentionally requires a new sign-in.

Server RBAC remains authoritative. Permissions are lowercase enum values:

| Role | Permissions |
|------|-------------|
| `admin` | All 11 permissions |
| `data_scientist` | `view_forecasts`, `view_metrics`, `view_audit_log`, `create_override`, `delete_override`, `run_backtest`, `run_pipeline`, `promote_model`, `modify_config` |
| `planner` | `view_forecasts`, `view_metrics`, `create_override` |
| `manager` | `view_forecasts`, `view_metrics`, `view_audit_log`, `approve_override` |
| `viewer` | `view_forecasts`, `view_metrics` |

Use `useAuth().hasPermission("run_pipeline")` for UI hints, never as the API
authorization boundary. Explicit demo uses `X-Demo-User` / `X-Demo-Role`, which
production ignores; demo refuses bearer credentials.

`e2e/authentication.test.tsx` covers 20 identity/cache/race/expiry cases. The
separate `npm run test:auth-browser` check requires `AUTH_E2E_ROOT` to be a fresh
absolute temporary directory outside the repository. It starts only isolated
loopback servers, creates private synthetic credentials, and verifies real
production API/browser behavior. Use `AUTH_E2E_PYTHON` and
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` for explicit local executables if needed. Its
dedicated configuration disables traces, screenshots, video, debug logging and
retries; never include its credential directory in artifacts. Normal Playwright
tests exclude this directory and use explicit demo login, not storage injection.

## Type System

All TypeScript interfaces (`src/lib/types.ts`) mirror the FastAPI Pydantic schemas (`src/api/schemas.py`):

- 46 exported interfaces/types covering all API request and response shapes
- Strict typing for roles (`Role` union type), severity levels, trends, confidence levels
- `Record<string, unknown>` used for flexible/dynamic data (e.g., per-series stats)

## Styling

- **Tailwind CSS 3.4** for utility-first styling
- **Radix UI** primitives for accessible interactive components (dialogs, dropdowns, tabs, tooltips, toasts, accordions, progress bars)
- **`class-variance-authority`** + **`clsx`** + **`tailwind-merge`** for variant-based component styling
- Color constants defined in `src/lib/constants.ts` (severity, demand class, model layer, FVA, confidence, risk)

## Testing

- **Unit tests:** Vitest + React Testing Library (`npm run test:e2e` — note: script name is misleading, runs Vitest)
- **E2E tests:** Playwright (`@playwright/test` in devDependencies)
- **Test runner:** `jsdom` environment for component tests

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

## Development

```bash
cd forecasting-product/frontend
npm install
npm run dev          # Start dev server (port 3000)
npm run build        # Production build
npm run lint         # ESLint
npm run test:e2e     # Run Vitest unit tests
```
