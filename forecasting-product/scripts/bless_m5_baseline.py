"""Prepare an UNAPPROVED M5 empirical-reference draft from native evidence.

The legacy filename is retained, but this command no longer runs models,
inventively assigns a seed, or blesses current output. It preserves full
captured precision and cannot write into the approved baselines directory.

Example (after the full native M5 fixture writes M5_CANDIDATE_OUTPUT):
    python scripts/bless_m5_baseline.py --frequency daily \
        --candidate artifacts/m5-candidate.json --output artifacts/m5-review-draft.json

Independent review must bind approval to the complete candidate digest and
its native lock, wheel and raw-prediction artifacts. See the baselines README.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.integration.baseline import (  # noqa: E402
    BASELINES_DIR,
    baseline_document,
    baseline_path,
    candidate_sha256,
)


def bless(
    frequency: str,
    notes: str,
    dry_run: bool,
    *,
    candidate: Path,
    output: Path | None = None,
) -> int:
    """Compatibility entry point: prepare a draft only; never grant approval."""
    if frequency != "daily":
        raise ValueError("Only the complete daily M5 fixture is supported")
    snapshot = json.loads(candidate.read_text(encoding="utf-8"))
    document = baseline_document(snapshot, notes)
    if not dry_run:
        if output is None:
            raise ValueError("An explicit draft --output or --dry-run is required")
        destination = output.resolve()
        if (
            destination == baseline_path(frequency).resolve()
            or destination.is_relative_to(BASELINES_DIR.resolve())
        ):
            raise ValueError("An unapproved draft cannot overwrite an empirical reference")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print("UNAPPROVED empirical-reference draft; capture is not independent approval.")
    print(f"Candidate SHA-256 for review: {candidate_sha256(snapshot)}")
    print(json.dumps(document, indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frequency", choices=["daily"], required=True)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Complete native M5_CANDIDATE_OUTPUT JSON, not a naive-only reference")
    parser.add_argument("--output", type=Path, help="New unapproved review draft outside baselines/")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return bless(
            args.frequency, args.notes, args.dry_run,
            candidate=args.candidate, output=args.output,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
