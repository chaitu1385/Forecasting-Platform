"""Real multiprocess writers and interrupted staging on the current local filesystem."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.pipeline.run_store import RunStore
from test_auth_startup import _stop
from test_run_store import _publish, _snapshot

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("run_store_process_probe.py")
pytestmark = pytest.mark.timeout(150)


def _start(root, control, name, mode, value=10.0, started_at="2026-01-01T12:00:00.000000+00:00", lob="retail"):
    environment = {
        **os.environ, "POLARS_MAX_THREADS": "2", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    }
    return subprocess.Popen(
        [sys.executable, str(PROBE), "--root", str(root), "--control", str(control),
         "--name", name, "--mode", mode, "--value", str(value), "--started-at", started_at, "--lob", lob],
        cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )


def _wait(path, processes, timeout=60):
    deadline = time.monotonic() + timeout
    while not path.exists():
        for process in processes:
            if process.poll() is not None and process.returncode:
                _, error = process.communicate(timeout=5)
                pytest.fail("Owned run-store child failed: " + error.decode(errors="replace"))
        if time.monotonic() > deadline:
            pytest.fail("Run-store child exceeded its bounded readiness deadline")
        time.sleep(0.03)
    return json.loads(path.read_text())


def _finish(process):
    output, error = process.communicate(timeout=60)
    assert process.returncode == 0, (output + error).decode(errors="replace")


def test_real_concurrent_writers_commit_distinct_values_and_latest_never_regresses(tmp_path):
    root, control = tmp_path / "runs", tmp_path / "control"
    old = _start(root, control, "old", "barrier", 10.0)
    new = _start(root, control, "new", "barrier", 20.0, "2026-01-01T12:00:01.000000+00:00")
    try:
        old_id = _wait(control / "old.ready.json", [old, new])["run_id"]
        new_id = _wait(control / "new.ready.json", [old, new])["run_id"]
        assert old_id != new_id
        store = RunStore(root)
        assert store.list() == []
        (control / "new.release").touch()
        _finish(new)
        assert store.latest("retail", "forecast").run_id == new_id
        (control / "old.release").touch()
        _finish(old)
        assert store.latest("retail", "forecast").run_id == new_id
        assert store.frame("retail", old_id, "forecast")["forecast"].to_list() == [10.0]
        assert store.frame("retail", new_id, "forecast")["forecast"].to_list() == [20.0]
        assert {item.run_id for item in store.list()} == {old_id, new_id}
    finally:
        _stop(old)
        _stop(new)


def test_killed_staging_writer_is_not_visible_after_process_restart(tmp_path):
    root, control = tmp_path / "runs", tmp_path / "control"
    store = RunStore(root)
    first = _publish(store, 10.0)
    before = _snapshot(store.directory("retail", first.run_id))
    interrupted = _start(root, control, "interrupted", "crash", 20.0)
    later = restarted = None
    try:
        abandoned_id = _wait(control / "interrupted.ready.json", [interrupted])["run_id"]
        _stop(interrupted)
        assert [item.run_id for item in store.list()] == [first.run_id]
        assert _snapshot(store.directory("retail", first.run_id)) == before
        restarted = _start(root, control, "restart", "verify")
        _finish(restarted)
        rows = _wait(control / "restart.result.json", [restarted])
        assert [row["run_id"] for row in rows] == [first.run_id]
        assert rows[0]["forecast"] == [10.0]
        later = _start(root, control, "later", "write", 30.0, "2026-01-01T12:00:02.000000+00:00")
        _finish(later)
        later_id = _wait(control / "later.result.json", [later])["run_id"]
        assert store.latest("retail", "forecast").run_id == later_id
        assert abandoned_id not in {item.run_id for item in store.list()}
        assert _snapshot(store.directory("retail", first.run_id)) == before
    finally:
        _stop(interrupted)
        if later is not None:
            _stop(later)
        if restarted is not None:
            _stop(restarted)


def test_real_initial_case_variant_race_never_publishes_an_unlistable_run(tmp_path, record_property):
    root, control = tmp_path / "runs", tmp_path / "control"
    root.mkdir()
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    case_insensitive = (tmp_path / "caseprobe").exists()
    record_property("case_alias_race_filesystem", case_insensitive)
    probe.rmdir()
    lower = _start(root, control, "lower", "case-race", 10.0, lob="retail")
    upper = _start(root, control, "upper", "case-race", 20.0, lob="RETAIL")
    try:
        _wait(control / "lower.ready.json", [lower, upper])
        _wait(control / "upper.ready.json", [lower, upper])
        assert list(root.iterdir()) == []
        (control / "lower.release").touch()
        (control / "upper.release").touch()
        _finish(lower)
        _finish(upper)
        results = [_wait(control / f"{name}.result.json", [lower, upper]) for name in ("lower", "upper")]
        outcomes = sorted(result["outcome"] for result in results)
        assert outcomes == (["committed", "not_committed"] if case_insensitive else ["committed", "committed"])
        for result in results:
            if result["outcome"] != "committed":
                assert result["stage"] == "reserve" and result["namespace_rejected"], result
        store = RunStore(root)
        successful = [result for result in results if result["outcome"] == "committed"]
        assert {item.run_id for item in store.list()} == {result["run_id"] for result in successful}
        for result in successful:
            assert store.get(result["lob"], result["run_id"]).lob == result["lob"]
            assert store.frame(result["lob"], result["run_id"], "forecast")["forecast"].to_list() == [result["value"]]
            assert store.latest(result["lob"], "forecast").run_id == result["run_id"]
    finally:
        _stop(lower)
        _stop(upper)
