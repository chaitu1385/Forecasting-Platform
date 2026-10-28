"""Test-only observations; never persist credentials, streams, paths or SIDs."""

import argparse
import json
import math
import os
import platform
import re
import stat
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROPERTY = "windows_writer_observation"
APPLICATION_SHA = "f025cb82dabe44b3da6eb7ab68dce36d48f01ae9"
PHASES = (
    "script_entry", "before_create", "file_created", "acl_verified",
    "stdin_eof", "bytes_written", "flushed", "disposed", "script_done",
)
CASES = (
    "test_private_output_is_new_owner_only_and_cannot_overwrite",
    "test_private_output_verifier_rejects_a_broader_read_rule",
)
EVENT = re.compile(
    r"^F8_WRITER_V1\|([a-z_]+)\|([0-9]+)\|"
    r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{7}Z)$"
)


def _marker(phase):
    return (
        "    [Console]::Error.WriteLine(('F8_WRITER_V1|" + phase
        + "|{0}|{1:o}' -f $PID, [DateTime]::UtcNow))"
    )


def instrument_writer(original):
    anchors = (
        ("$stream = $null", "script_entry", True),
        ("    $stream = [System.IO.FileStream]::new(", "before_create", False),
        ("    $actual = $stream.GetAccessControl()", "file_created", False),
        ("    $bytes = [System.Text.Encoding]::UTF8.GetBytes([Console]::In.ReadToEnd())", "acl_verified", False),
        ("    $stream.Write($bytes, 0, $bytes.Length)", "stdin_eof", False),
        ("    $stream.Flush($true)", "bytes_written", False),
        ("    $stream.Flush($true)", "flushed", True),
        ("    if ($null -ne $stream) { $stream.Dispose() }", "disposed", True),
    )
    result = original
    for anchor, phase, after in anchors:
        if result.count(anchor) != 1:
            raise AssertionError("The reviewed writer observation anchor changed")
        insertion = f"{anchor}\n{_marker(phase)}" if after else f"{_marker(phase)}\n{anchor}"
        result = result.replace(anchor, insertion)
    return result + _marker("script_done") + "\n"


def parse_events(stream, started):
    if isinstance(stream, bytes):
        stream = stream.decode("utf-8", errors="replace")
    events, rejected = [], 0
    for line in (stream or "").splitlines():
        match = EVENT.fullmatch(line)
        if match and match[1] in PHASES and 0 < int(match[2]) <= 0xFFFFFFFF:
            try:
                stamp = datetime.fromisoformat(match[3].replace("Z", "+00:00"))
            except ValueError:
                rejected += 1
                continue
            events.append({
                "phase": match[1], "pid": int(match[2]),
                "seconds_since_call": (stamp - started).total_seconds(),
            })
        else:
            rejected += 1
    return events, rejected


def install_observer(monkeypatch, record_property):
    if os.name != "nt":
        return
    from src.auth import issuer

    marked = instrument_writer(issuer._WINDOWS_WRITE)
    real_run, real_popen = subprocess.run, subprocess.Popen
    active = None

    def matches(command):
        return isinstance(command, (list, tuple)) and bool(command) and command[-1] == marked

    def observed_popen(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        process = real_popen(*args, **kwargs)
        if active is not None and matches(command):
            active["pid"] = process.pid
            active["launch_seconds"] = time.perf_counter() - active["clock"]
        return process

    def observed_run(*args, **kwargs):
        nonlocal active
        command = args[0] if args else kwargs.get("args")
        if not matches(command):
            return real_run(*args, **kwargs)
        started = datetime.now(timezone.utc)
        active = {"pid": None, "launch_seconds": None, "clock": time.perf_counter()}
        result, stderr, outcome = None, None, "exception"
        try:
            result = real_run(*args, **kwargs)
            stderr, outcome = result.stderr, "returned"
            return result
        except subprocess.TimeoutExpired as error:
            stderr, outcome = error.stderr, "timeout"
            raise
        finally:
            events, rejected = parse_events(stderr, started)
            observation = {
                "outcome": outcome, "pid": active["pid"],
                "launch_seconds": active["launch_seconds"],
                "elapsed_seconds": time.perf_counter() - active["clock"],
                "timeout_seconds": kwargs.get("timeout"),
                "returncode": result.returncode if result is not None else None,
                "events": events, "unparsed_stderr_lines": rejected,
                "file_state": "unavailable", "file_size": None,
            }
            output = kwargs.get("env", {}).get("FORECAST_PRIVATE_TOKEN_OUTPUT")
            if output is not None:
                try:
                    info = os.stat(output, follow_symlinks=False)
                except FileNotFoundError:
                    observation["file_state"] = "absent"
                except OSError:
                    observation["file_state"] = "unavailable"
                else:
                    observation["file_state"] = "regular" if stat.S_ISREG(info.st_mode) else "other"
                    observation["file_size"] = info.st_size
            active = None
            record_property(PROPERTY, json.dumps(observation, allow_nan=False))

    monkeypatch.setattr(issuer, "_WINDOWS_WRITE", marked)
    monkeypatch.setattr(subprocess, "Popen", observed_popen)
    monkeypatch.setattr(subprocess, "run", observed_run)


def _safe_observation(value):
    keys = {
        "outcome", "pid", "launch_seconds", "elapsed_seconds", "timeout_seconds",
        "returncode", "events", "unparsed_stderr_lines", "file_state", "file_size",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Unexpected observation shape")
    if value["outcome"] not in ("returned", "timeout", "exception"):
        raise ValueError("Unexpected outcome")
    if value["file_state"] not in ("regular", "absent", "other", "unavailable"):
        raise ValueError("Unexpected file state")
    for key in ("pid", "returncode", "unparsed_stderr_lines", "file_size"):
        if value[key] is not None and (type(value[key]) is not int or abs(value[key]) > 2**63 - 1):
            raise ValueError("Unexpected integer observation")
    if value["pid"] is not None and not 0 < value["pid"] <= 0xFFFFFFFF:
        raise ValueError("Unexpected process identity")
    for key in ("launch_seconds", "elapsed_seconds", "timeout_seconds"):
        if value[key] is not None and (type(value[key]) not in (int, float) or not math.isfinite(value[key])):
            raise ValueError("Unexpected timing observation")
    if value["timeout_seconds"] != 15 or not isinstance(value["events"], list):
        raise ValueError("The original writer deadline or event shape changed")
    for event in value["events"]:
        if not isinstance(event, dict) or set(event) != {"phase", "pid", "seconds_since_call"}:
            raise ValueError("Unexpected event shape")
        if event["phase"] not in PHASES or type(event["pid"]) is not int or event["pid"] != value["pid"]:
            raise ValueError("Unexpected event identity")
        if type(event["seconds_since_call"]) not in (int, float) or not math.isfinite(event["seconds_since_call"]):
            raise ValueError("Unexpected event timing")
    return value


def summarize(junit, output, expected_cases=None):
    root = ET.parse(junit)
    cases = list(root.iter("testcase"))
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError("The original suite population changed")
    observations = {}
    for name in CASES:
        selected = [case for case in cases if case.attrib.get("name") == name]
        if len(selected) != 1:
            raise ValueError("The original observed case is missing or duplicated")
        records = [item.attrib["value"] for item in selected[0].findall("./properties/property")
                   if item.attrib.get("name") == PROPERTY]
        if len(records) != 1:
            raise ValueError("The original writer observation is missing or duplicated")
        observations[name] = {
            "test_failed": selected[0].find("failure") is not None,
            "test_error": selected[0].find("error") is not None,
            "test_skipped": selected[0].find("skipped") is not None,
            "writer": _safe_observation(json.loads(records[0])),
        }
    reload_name = "test_default_production_and_worker_launches_use_verified_identity[extra_args2]"
    reload_cases = [case for case in cases if case.attrib.get("name") == reload_name]
    if len(reload_cases) > 1:
        raise ValueError("The original reload case is duplicated")
    if expected_cases == 1032 and not reload_cases:
        raise ValueError("The original reload case is missing")
    diagnostic_sha = os.environ.get("GITHUB_SHA", "")
    report = {
        "application_source_sha": APPLICATION_SHA,
        "diagnostic_source_sha": diagnostic_sha if re.fullmatch(r"[0-9a-f]{40}", diagnostic_sha) else None,
        "python": platform.python_version(),
        "suite": {
            "cases": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
            "errors": sum(c.find("error") is not None for c in cases),
            "skips": sum(c.find("skipped") is not None for c in cases),
        },
        "observations": observations,
        "original_reload_case": {
            "present": bool(reload_cases),
            "failed": bool(reload_cases) and reload_cases[0].find("failure") is not None,
            "error": bool(reload_cases) and reload_cases[0].find("error") is not None,
            "skipped": bool(reload_cases) and reload_cases[0].find("skipped") is not None,
        },
        "scope": "Test-only timing observation, not final qualification or a correction.",
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=1032)
    options = parser.parse_args()
    try:
        summarize(options.junit, options.output, options.expected_cases)
    except (OSError, ET.ParseError, ValueError, KeyError, TypeError, OverflowError):
        raise SystemExit("The allowlisted writer diagnostic could not be produced; no raw data was published.") from None
