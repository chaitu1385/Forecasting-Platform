"""Bless a new M5 integration baseline.

Runs the same pipeline that the integration tests exercise, then writes
the observed champion WMAPE, per-model WMAPE, FVA vs naive, and fixture
hashes to ``tests/integration/baselines/m5_<frequency>_baseline.json``.

The resulting JSON is the ground truth against which subsequent PRs are
gated — see ``tests/integration/baseline.py`` for the tolerance policy.

Usage
-----
    python scripts/bless_m5_baseline.py --frequency daily
    python scripts/bless_m5_baseline.py --frequency daily --notes "+ xgboost_direct"
    python scripts/bless_m5_baseline.py --frequency daily --dry-run

The ``--dry-run`` flag prints the captured numbers without writing the
baseline file, useful for previewing before committing.

Only ``daily`` is wired today; weekly/monthly hooks are in place but
skipped until their pytest suites are added.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make ``src`` importable when this script is invoked directly.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.integration.baseline import baseline_path, sha256_file  # noqa: E402

# ── Supported configurations ──────────────────────────────────────────────────
FIXTURES_DIR = _REPO / "tests" / "integration" / "fixtures"

SUPPORTED: dict[str, dict[str, Path | str]] = {
    "daily": {
        "fixture": FIXTURES_DIR / "m5_daily_sample.csv",
        "config": FIXTURES_DIR / "m5_daily_config.yaml",
        "lob": "walmart_m5_daily",
    },
    # Stubs — enable once pytest integration suites exist.
    "weekly": {
        "fixture": FIXTURES_DIR / "m5_weekly_sample.csv",
        "config": FIXTURES_DIR / "m5_weekly_config.yaml",
        "lob": "walmart_m5_weekly",
    },
    "monthly": {
        "fixture": FIXTURES_DIR / "m5_monthly_sample.csv",
        "config": FIXTURES_DIR / "m5_monthly_config.yaml",
        "lob": "walmart_m5_monthly",
    },
}

DEFAULT_SEED = 42


def _run_backtest(fixture: Path, config: Path, lob: str) -> dict:
    """Drive ``POST /pipeline/backtest`` the same way the integration tests do.

    Returns the parsed JSON response containing the leaderboard + champion.
    """
    import tempfile

    from fastapi.testclient import TestClient
    import polars as pl

    from src.api.app import create_app

    tmpdir = Path(tempfile.mkdtemp(prefix="bless_m5_"))
    data_dir = tmpdir / "data"
    metrics_dir = tmpdir / "metrics"
    data_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)

    # Pre-populate history (mirrors the test's ``setUpClass``).
    df = pl.read_csv(fixture, try_parse_dates=True)
    hist_dir = data_dir / "history" / lob
    hist_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(hist_dir / "actuals.parquet")

    app = create_app(
        data_dir=str(data_dir),
        metrics_dir=str(metrics_dir),
        auth_enabled=False,
        dev_mode=True,
        jwt_secret="",
    )
    client = TestClient(app)

    resp = client.post(
        f"/pipeline/backtest?lob={lob}",
        files={
            "file": (fixture.name, fixture.read_bytes(), "text/csv"),
            "config_file": (config.name, config.read_bytes(), "application/x-yaml"),
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Backtest failed: {resp.status_code} {resp.text}")
    return resp.json()


def _extract_metrics(backtest_result: dict) -> dict:
    """Distil the backtest JSON into baseline fields."""
    leaderboard = backtest_result.get("leaderboard") or []
    per_model: dict[str, float] = {}
    for entry in leaderboard:
        model_id = entry.get("model_id") or entry.get("model")
        wmape = entry.get("wmape")
        if model_id and wmape is not None:
            per_model[str(model_id)] = float(wmape)

    champion_model = backtest_result.get("champion_model")
    champion_wmape = backtest_result.get("best_wmape")

    # Naive WMAPE drives the FVA gate. Match by model id containing "naive"
    # so the script is resilient to ``naive_seasonal`` vs ``seasonal_naive``.
    naive_wmape = None
    for mid, w in per_model.items():
        if "naive" in mid.lower():
            naive_wmape = w
            break

    fva = None
    if naive_wmape is not None and champion_wmape is not None:
        fva = round(float(naive_wmape) - float(champion_wmape), 6)

    return {
        "per_model_wmape": {k: round(v, 6) for k, v in per_model.items()},
        "champion_model": champion_model,
        "champion_wmape": (
            round(float(champion_wmape), 6) if champion_wmape is not None else None
        ),
        "naive_wmape": round(float(naive_wmape), 6) if naive_wmape is not None else None,
        "fva_vs_naive": fva,
    }


def bless(frequency: str, notes: str, dry_run: bool) -> int:
    if frequency not in SUPPORTED:
        print(f"ERROR: unsupported frequency '{frequency}'. "
              f"Choose one of: {', '.join(SUPPORTED)}.", file=sys.stderr)
        return 2

    spec = SUPPORTED[frequency]
    fixture = Path(spec["fixture"])
    config = Path(spec["config"])
    lob = str(spec["lob"])

    if not fixture.exists():
        print(f"ERROR: fixture not found at {fixture}.", file=sys.stderr)
        if frequency == "daily":
            print("  Run: python -m tests.integration.prepare_m5_daily",
                  file=sys.stderr)
        else:
            print(f"  No fixture preparation pipeline wired for '{frequency}' yet.",
                  file=sys.stderr)
        return 2
    if not config.exists():
        print(f"ERROR: config not found at {config}.", file=sys.stderr)
        return 2

    print(f"Running backtest for '{frequency}' (lob={lob})…")
    backtest_result = _run_backtest(fixture, config, lob)
    metrics = _extract_metrics(backtest_result)

    if metrics["champion_wmape"] is None:
        print("ERROR: pipeline did not return a champion WMAPE; refusing to bless.",
              file=sys.stderr)
        print(json.dumps(backtest_result, indent=2, default=str)[:2000], file=sys.stderr)
        return 3

    baseline = {
        "frequency": frequency,
        "fixture": str(fixture.relative_to(_REPO)).replace("\\", "/"),
        "config": str(config.relative_to(_REPO)).replace("\\", "/"),
        "fixture_sha256": sha256_file(fixture),
        "config_sha256": sha256_file(config),
        "seed": DEFAULT_SEED,
        "blessed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "notes": notes,
        **metrics,
    }

    print("\nCaptured baseline:")
    print(json.dumps(baseline, indent=2))

    if dry_run:
        print("\n[dry-run] not writing baseline file.")
        return 0

    out = baseline_path(frequency)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--frequency",
        choices=sorted(SUPPORTED),
        required=True,
        help="Which M5 baseline to bless.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form note recorded in the baseline JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print captured numbers without writing the baseline file.",
    )
    args = parser.parse_args()
    return bless(args.frequency, args.notes, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
