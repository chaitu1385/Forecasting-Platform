"""Read-only local run inspection; never repairs, deletes or migrates history."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.run_store import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True, help="Explicit local run authority")
    parser.add_argument("--lob", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-type", choices=["forecast", "backtest"], default=None)
    parser.add_argument("--config", action="store_true", help="Print safe stored config for one exact run")
    args = parser.parse_args()
    if args.run_id and not args.lob:
        parser.error("--run-id requires --lob")
    if args.config and not args.run_id:
        parser.error("--config requires --run-id")
    store = RunStore(args.runs_dir)
    if args.run_id:
        manifest = store.get(args.lob, args.run_id, args.run_type)
        value = store.json_member(args.lob, args.run_id, "config") if args.config else asdict(manifest)
    else:
        value = [asdict(item) for item in store.list(args.lob, args.run_type)]
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
