"""Loopback-only API fixture; refuses to reuse an existing output directory."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.api.app import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18118)
    parser.add_argument("--frontend-origin", default="http://127.0.0.1:13118")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    if root.exists():
        parser.error("--data-dir must be a new directory dedicated to this fixture")
    frontend = urlparse(args.frontend_origin)
    if frontend.scheme != "http" or frontend.hostname not in {"localhost", "127.0.0.1"}:
        parser.error("--frontend-origin must use the loopback address")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    root.mkdir(parents=True)
    (root / "metrics").mkdir()
    (root / ".forecast-intent-fixture.json").write_text(
        json.dumps({"api_url": f"http://127.0.0.1:{args.port}"}),
        encoding="utf-8",
    )
    app = create_app(
        data_dir=str(root),
        metrics_dir=str(root / "metrics"),
        audit_log_path=str(root / "audit"),
        title="Forecast intent test fixture",
        auth_enabled=False,
        cors_origins=[args.frontend_origin],
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
