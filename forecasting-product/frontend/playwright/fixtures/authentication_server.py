"""Isolated production-mode API with synthetic data and private ephemeral credentials."""

import argparse
import json
import os
import secrets
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import polars as pl
import uvicorn

from src.api.app import create_app
from src.auth.issuer import write_private_token
from src.auth.token import create_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--credentials-dir", type=Path, required=True)
    parser.add_argument("--frontend-origin", required=True)
    args = parser.parse_args()
    origin = urlparse(args.frontend_origin)
    if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Authentication fixture requires a loopback frontend origin")
    data = args.data_dir.resolve()
    credentials = args.credentials_dir.resolve()
    if credentials.is_relative_to(ROOT.parent) or credentials.is_relative_to(data):
        raise ValueError("Synthetic credentials must be outside the repository and API data")
    for directory in (data, credentials):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Authentication fixture requires fresh empty directories")
        directory.mkdir(parents=True, exist_ok=True)

    key = secrets.token_urlsafe(48)
    os.environ["AUTH_ENABLED"] = "true"
    os.environ["API_DEV_MODE"] = "false"
    os.environ["JWT_SECRET_KEY"] = key
    app = create_app(
        data_dir=str(data), metrics_dir=str(data / "metrics"),
        title="Authentication integration test fixture",
        cors_origins=[args.frontend_origin],
    )
    write_private_token(credentials / "signing-key", key)
    for role in ("admin", "viewer"):
        write_private_token(
            credentials / f"{role}.jwt",
            create_token(f"browser-{role}", "", role, key, expiry_hours=1),
        )
    forecast = data / "forecasts" / "retail"
    forecast.mkdir(parents=True)
    pl.DataFrame({
        "series_id": ["SYNTHETIC_AUTH_SERIES"] * 4,
        "week": [date(2024, 1, 1) + timedelta(weeks=i) for i in range(4)],
        "forecast": [10.0, 11.0, 12.0, 13.0],
        "model": ["naive_seasonal"] * 4,
    }).write_parquet(forecast / "forecast_retail_2024-01-01.parquet")
    (data / ".authentication-fixture.json").write_text(json.dumps({
        "api_url": f"http://127.0.0.1:{args.port}",
        "frontend_origin": args.frontend_origin,
        "mode": "production",
    }), encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
