"""Bounded subprocess regressions for native-pool/process-pool interaction."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["backtest", "batch"])
def test_parallel_execution_after_polars_initialization(scenario, tmp_path):
    env = {
        **os.environ,
        "POLARS_MAX_THREADS": "2",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "tests.process_execution_probe", scenario, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
    )
    try:
        output, _ = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=True, capture_output=True,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=10)
        pytest.fail(f"{scenario} process pool stalled for 90 seconds:\n{output}")
    assert process.returncode == 0, output
    assert f'"scenario": "{scenario}", "status": "passed"' in output, output
