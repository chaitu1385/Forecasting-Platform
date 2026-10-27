"""Reproducible profile boundaries and generated-artifact freshness."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
MINORS = ("py311", "py312", "py313")
PROFILES = ("core", "api", "dev", "ai", "fabric", "spark")


def _requirements(path: Path) -> dict[str, Requirement]:
    content = path.read_text(encoding="utf-8").replace("\\\n", "")
    requirements = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert not line.startswith("-"), f"Nonportable installer option in {path.name}: {line}"
        requirement = Requirement(line.split("--hash=", 1)[0].strip())
        assert requirement.url is None, "Session-local or authenticated URLs must never enter a lock"
        pins = list(requirement.specifier)
        assert len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version
        assert re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)", line)
        requirements[requirement.name.lower().replace("_", "-")] = requirement
    assert requirements, f"Empty generated profile: {path}"
    return requirements


@pytest.mark.parametrize("minor", MINORS)
@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_are_hashed_portable_and_cpu_only(minor, profile):
    packages = _requirements(ROOT / "requirements" / minor / f"{profile}.txt")
    assert {"numpy", "polars", "scipy", "statsforecast", "mlforecast", "lightgbm", "xgboost-cpu"} <= packages.keys()
    assert not any(name.startswith(("nvidia-", "torch", "tensorflow")) for name in packages)
    if profile in ("core", "ai", "fabric", "spark"):
        assert not {"fastapi", "uvicorn", "duckdb", "pytest"} & packages.keys()
    if profile == "fabric":
        assert not {"pyspark", "delta-spark", "neuralforecast", "azure-identity"} & packages.keys()
    if profile in ("api", "dev"):
        assert {"fastapi", "starlette", "uvicorn", "python-multipart", "pyjwt", "duckdb", "anthropic"} <= packages.keys()
    if profile == "spark":
        assert {"pyspark", "delta-spark", "py4j"} <= packages.keys()
    if profile != "dev":
        dev = _requirements(ROOT / "requirements" / minor / "dev.txt")
        if profile != "spark":
            assert packages.keys() <= dev.keys(), "The audited dev closure must cover this profile"
        record = json.loads((ROOT / "requirements" / "manifest.json").read_text(encoding="utf-8"))
        version = record["profiles"][f"{minor}/{profile}.txt"]["python"]
        for name in packages.keys() & dev.keys():
            assert packages[name].specifier == dev[name].specifier, (minor, profile, name)
            for system, os_name, machine in (("Windows", "nt", "AMD64"), ("Linux", "posix", "x86_64")):
                environment = {
                    **default_environment(),
                    "os_name": os_name,
                    "sys_platform": "win32" if system == "Windows" else "linux",
                    "platform_system": system,
                    "platform_machine": machine,
                    "python_full_version": version,
                    "python_version": ".".join(version.split(".")[:2]),
                    "implementation_version": version,
                    "implementation_name": "cpython",
                    "platform_python_implementation": "CPython",
                }
                active = packages[name].marker is None or packages[name].marker.evaluate(environment)
                audited = dev[name].marker is None or dev[name].marker.evaluate(environment)
                assert not active or audited, (minor, profile, system, name)


def test_generated_profiles_match_manifest_and_recorded_content():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "lock_dependencies.py"), "--check"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_lock_check_rejects_modified_content(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("lock_dependencies", ROOT / "scripts" / "lock_dependencies.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOCK_ROOT", tmp_path)
    monkeypatch.setattr(module, "RECORD", ROOT / "requirements" / "manifest.json")
    target = tmp_path / "py311" / "dev.txt"
    target.parent.mkdir()
    target.write_text("# corrupted generated content\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated content changed"):
        module.verify()
