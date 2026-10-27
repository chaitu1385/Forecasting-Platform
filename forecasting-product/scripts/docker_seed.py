"""Production API supervisor; seed backtest + forecast only in explicit demo mode.

Production validates signing configuration and never seeds demo data.
On the first explicit-demo boot, this script:
  1. Generates the weekly M5 sample (daily → weekly aggregation)
  2. Starts the API server in the background
  3. Calls POST /pipeline/backtest with the weekly data + config
  4. Calls POST /pipeline/forecast with the champion model
  5. Writes a sentinel file so subsequent boots skip seeding
  6. Supervises the same API server command on cold and subsequent boots

Usage (inside Docker):
    python forecasting-product/scripts/docker_seed.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("docker_seed")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # forecasting-product/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
DATA_DIR = Path(os.environ.get("API_DATA_DIR", str(_ROOT / "data"))).resolve()
METRICS_DIR = Path(os.environ.get("API_METRICS_DIR", str(DATA_DIR / "metrics"))).resolve()
SENTINEL = DATA_DIR / ".seeded"
DEMO_LOB = os.environ.get("DEMO_LOB", "walmart_m5_weekly")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))


def wait_for_api(
    host: str = "127.0.0.1",
    port: int = API_PORT,
    timeout: int = 120,
    process: subprocess.Popen | None = None,
) -> bool:
    """Poll the /health endpoint until it responds or timeout."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"API server exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def seed_via_api(csv_path: Path, config_path: Path, lob: str) -> str:
    """Call the backtest + forecast API endpoints. Returns champion model name."""
    import urllib.request
    import urllib.error

    base = f"http://127.0.0.1:{API_PORT}"

    # ── Backtest ──────────────────────────────────────────────────────────
    logger.info("Seeding backtest for LOB=%s …", lob)
    csv_bytes = csv_path.read_bytes()
    config_bytes = config_path.read_bytes()

    # Build multipart form data
    boundary = "----DockerSeedBoundary"
    body = bytearray()

    # file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'.encode()
    body += b"Content-Type: text/csv\r\n\r\n"
    body += csv_bytes
    body += b"\r\n"

    # config_file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="config_file"; filename="{config_path.name}"\r\n'.encode()
    body += b"Content-Type: application/x-yaml\r\n\r\n"
    body += config_bytes
    body += b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base}/pipeline/backtest?lob={lob}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        champion = result.get("champion_model")
        if result.get("status") != "completed" or not isinstance(champion, str) or not champion:
            raise RuntimeError("Backtest did not complete with a champion model")
        best_wmape = result.get("best_wmape")
        logger.info("Backtest done — champion=%s, WMAPE=%s", champion, best_wmape)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Backtest API call failed: {exc}") from exc

    # ── Forecast ──────────────────────────────────────────────────────────
    logger.info("Seeding forecast with champion=%s …", champion)
    body2 = bytearray()
    body2 += f"--{boundary}\r\n".encode()
    body2 += f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'.encode()
    body2 += b"Content-Type: text/csv\r\n\r\n"
    body2 += csv_bytes
    body2 += b"\r\n"

    body2 += f"--{boundary}\r\n".encode()
    body2 += f'Content-Disposition: form-data; name="config_file"; filename="{config_path.name}"\r\n'.encode()
    body2 += b"Content-Type: application/x-yaml\r\n\r\n"
    body2 += config_bytes
    body2 += b"\r\n"

    body2 += f"--{boundary}--\r\n".encode()

    model = champion
    req2 = urllib.request.Request(
        f"{base}/pipeline/forecast?lob={lob}&model_id={model}",
        data=bytes(body2),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req2, timeout=600) as resp:
            result2 = json.loads(resp.read())
        rows = result2.get("forecast_rows", 0)
        if result2.get("status") != "completed" or not isinstance(rows, int) or rows <= 0:
            raise RuntimeError("Forecast did not complete with nonempty results")
        logger.info("Forecast done — %d rows", rows)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Forecast API call failed: {exc}") from exc

    return champion


def _server_command() -> list[str]:
    return [
        sys.executable, str(_HERE / "serve.py"),
        "--host", API_HOST, "--port", str(API_PORT),
        "--data-dir", str(DATA_DIR), "--metrics-dir", str(METRICS_DIR),
    ]


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("API server did not stop gracefully; killing child process")
            process.kill()
            process.wait(timeout=10)


def _exit_on_signal(signum, _frame) -> None:
    raise SystemExit(128 + signum)


def _run_server(seed_paths: tuple[Path, Path] | None = None) -> None:
    """Keep process ownership and cleanup consistent, including on Windows."""
    logger.info("Starting API server on %s:%d …", API_HOST, API_PORT)
    api_proc = subprocess.Popen(_server_command(), cwd=str(_ROOT))
    previous_sigterm = signal.signal(signal.SIGTERM, _exit_on_signal)
    try:
        if seed_paths is not None:
            if not wait_for_api(timeout=120, process=api_proc):
                raise RuntimeError("API server did not become healthy within 120s")
            demo_csv, config_path = seed_paths
            champion = seed_via_api(demo_csv, config_path, DEMO_LOB)
            SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            SENTINEL.write_text(f"seeded:{DEMO_LOB}:champion={champion}\n", encoding="utf-8")
            logger.info("Seed complete — sentinel written to %s", SENTINEL)
        returncode = api_proc.wait()
        if returncode:
            raise SystemExit(returncode)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        _stop_server(api_proc)


def main() -> None:
    from src.auth.settings import AuthSettings

    settings = AuthSettings.from_env()
    if not settings.dev_mode:
        logger.info("Production authentication enabled; demo seeding is disabled.")
        _run_server()
        return
    if SENTINEL.exists():
        logger.info("Sentinel found (%s) — skipping seed, starting API directly.", SENTINEL)
        _run_server()
        return

    # Step 1: Generate weekly M5 sample
    demo_csv = DATA_DIR / "demo" / "m5_weekly.csv"
    config_path = _ROOT / "configs" / "m5_weekly_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Demo configuration not found: {config_path}")

    if not demo_csv.exists():
        logger.info("Generating weekly M5 sample …")
        from scripts.prepare_m5_weekly import prepare_weekly
        daily_csv = _ROOT / "tests" / "integration" / "fixtures" / "m5_daily_sample.csv"
        if not daily_csv.exists():
            raise FileNotFoundError(f"Daily fixture not found: {daily_csv}")
        prepare_weekly(daily_csv, demo_csv, n_series=10)

    _run_server((demo_csv, config_path))


if __name__ == "__main__":
    main()
