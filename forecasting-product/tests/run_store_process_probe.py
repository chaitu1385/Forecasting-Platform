"""Owned child process for local run-store barrier/crash tests; no production hooks."""

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_store
from test_run_store import _bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["write", "barrier", "crash", "verify", "case-race"], default="write")
    parser.add_argument("--lob", default="retail")
    parser.add_argument("--value", type=float, default=10.0)
    parser.add_argument("--started-at", default="2026-01-01T12:00:00.000000+00:00")
    args = parser.parse_args()
    args.control.mkdir(parents=True, exist_ok=True)
    store = run_store.RunStore(args.root)

    def report(suffix, payload):
        destination = args.control / f"{args.name}.{suffix}.json"
        temporary = destination.with_suffix(".pending")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(destination)

    if args.mode == "verify":
        rows = [{
            "run_id": item.run_id, "input_hash": item.input_data_hash, "config_hash": item.config_hash,
            "forecast": store.frame(item.lob, item.run_id, "forecast")["forecast"].to_list(),
        } for item in store.list()]
        report("result", rows)
        return
    manifest, config, frames = _bundle(args.value, started_at=args.started_at, lob=args.lob)
    original_link, original_write = run_store.os.link, run_store._write_new

    def wait_for_release():
        report("ready", {"run_id": manifest.run_id})
        deadline = time.monotonic() + 60
        while not (args.control / f"{args.name}.release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Parent did not release the bounded test writer")
            time.sleep(0.02)

    def barrier(source, destination):
        if args.mode == "barrier" and destination.name == run_store.MANIFEST_NAME:
            wait_for_release()
        return original_link(source, destination)

    def stop_after_input(path, writer):
        result = original_write(path, writer)
        if args.mode == "crash" and path.name == "input.parquet":
            wait_for_release()
        return result

    if args.mode == "case-race":
        wait_for_release()
    try:
        with patch.object(run_store.os, "link", barrier), patch.object(run_store, "_write_new", stop_after_input):
            committed = store.publish(manifest, config, frames, {"skipped": True})
    except run_store.RunPublicationError as exc:
        if args.mode != "case-race":
            raise
        report("result", {
            "run_id": manifest.run_id, "lob": args.lob, "outcome": exc.outcome,
            "stage": exc.stage, "cause": type(exc.__cause__).__name__,
            "namespace_rejected": isinstance(exc.__cause__, ValueError)
            and str(exc.__cause__) == "LOB spelling must match the existing filesystem partition",
        })
    else:
        report("result", {"run_id": committed.run_id, "value": args.value, "lob": args.lob, "outcome": "committed"})


if __name__ == "__main__":
    main()
