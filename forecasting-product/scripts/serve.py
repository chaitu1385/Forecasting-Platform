"""
serve.py — Start the forecasting product REST API server.

Usage
-----
python scripts/serve.py
python scripts/serve.py --port 8080 --data-dir data/ --metrics-dir data/metrics/
python scripts/serve.py --host 0.0.0.0 --port 8000 --reload   # authenticated hot reload
python scripts/serve.py --demo --host 127.0.0.1             # explicit unauthenticated demo

Environment variable overrides
-------------------------------
API_DATA_DIR      Path to data directory.
API_METRICS_DIR   Path to metrics store.
API_VERSION       Version string embedded in /health.
AUTH_ENABLED      Strict boolean; defaults to true.
JWT_SECRET_KEY    Required production signing key; never passed as a CLI argument.
API_DEV_MODE      Strict boolean; defaults to false.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("serve")


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting Product REST API server")
    p.add_argument("--host",        default=os.environ.get("API_HOST"), help="Bind host (demo defaults to loopback)")
    p.add_argument("--port",        type=int, default=8000, help="Bind port")
    p.add_argument("--data-dir",    default=os.environ.get("API_DATA_DIR", "data/"),
                   help="Root data directory for forecasts")
    p.add_argument("--metrics-dir", default=os.environ.get("API_METRICS_DIR"),
                   help="Metrics store directory")
    p.add_argument("--runs-dir", default=os.environ.get("API_RUNS_DIR"),
                   help="Shared immutable run authority (default: data-dir/runs)")
    p.add_argument("--demo", action="store_true", help="Explicit unauthenticated local demo; do not use in production")
    p.add_argument("--reload",      action="store_true", help="Enable hot-reload (dev mode)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Number of Uvicorn worker processes (production)")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["API_DATA_DIR"] = str(Path(args.data_dir).resolve())
    os.environ["API_METRICS_DIR"] = str(
        Path(args.metrics_dir).resolve() if args.metrics_dir else Path(os.environ["API_DATA_DIR"]) / "metrics"
    )
    if args.runs_dir is not None and (not args.runs_dir.strip() or "://" in args.runs_dir):
        raise ValueError("--runs-dir must be a nonblank local filesystem path")
    os.environ["API_RUNS_DIR"] = str(
        Path(args.runs_dir).resolve() if args.runs_dir else Path(os.environ["API_DATA_DIR"]) / "runs"
    )
    from src.auth.settings import AuthSettings

    try:
        environment = None
        if args.demo:
            environment = {
                "AUTH_ENABLED": os.environ.get("AUTH_ENABLED", "false"),
                "API_DEV_MODE": os.environ.get("API_DEV_MODE", "true"),
                "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY", ""),
            }
        settings = AuthSettings.from_env(environment=environment)
        if args.demo and not settings.dev_mode:
            raise ValueError("--demo conflicts with production authentication settings")
    except (ValueError, RuntimeError) as exc:
        logger.error("Authentication configuration rejected: %s", exc)
        raise SystemExit(2) from None
    if args.demo:
        os.environ["AUTH_ENABLED"] = "false"
        os.environ["API_DEV_MODE"] = "true"
    Path(os.environ["API_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["API_METRICS_DIR"]).mkdir(parents=True, exist_ok=True)

    import uvicorn

    host = args.host or ("127.0.0.1" if settings.dev_mode else "0.0.0.0")
    logger.info("Starting Forecasting Product API on %s:%d", host, args.port)
    logger.info("  mode        = %s", settings.mode)
    logger.info("  data_dir    = %s", os.environ["API_DATA_DIR"])
    logger.info("  metrics_dir = %s", os.environ["API_METRICS_DIR"])
    logger.info("  Swagger UI  = http://%s:%d/docs", host, args.port)

    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
