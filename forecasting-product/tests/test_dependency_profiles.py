"""Reproducible profile boundaries and generated-artifact freshness."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
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


def test_full_native_matrix_captures_unapproved_m5_evidence_without_hiding_failures():
    workflow = yaml.safe_load((ROOT.parent / ".github" / "workflows" / "ci.yml").read_text())
    job = workflow["jobs"]["test"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"]["include"] == [
        {"python-version": "3.11.18", "profile": "py311"},
        {"python-version": "3.12.16", "profile": "py312"},
        {"python-version": "3.13.18", "profile": "py313"},
    ]
    env = job["env"]
    steps = {step["name"]: step for step in job["steps"] if "name" in step}
    preparation = steps["Prepare fresh native M5 evidence"]
    evidence = preparation["env"]["M5_EVIDENCE_DIR"]
    assert all(value in evidence for value in (
        "${{ github.run_id }}", "${{ github.run_attempt }}", "${{ matrix.profile }}",
    ))
    assert 'mkdir "$M5_EVIDENCE_DIR"' in preparation["run"]
    assert all(value in preparation["run"] for value in (
        "M5_EVIDENCE_DIR=$M5_EVIDENCE_DIR",
        "M5_PROFILE_INSTALL_REPORT=$M5_EVIDENCE_DIR/install.json",
        "M5_PROFILE_QUALIFICATION=$M5_EVIDENCE_DIR/native-profiles/dev.json",
        "M5_CANDIDATE_OUTPUT=$M5_EVIDENCE_DIR/candidate.json",
        '>> "$GITHUB_ENV"',
    ))
    assert env["M5_PROFILE_LOCK"].endswith("/requirements/${{ matrix.profile }}/dev.txt")
    assert not {"GITHUB_RUN_ID", "GITHUB_SHA", "RUNNER_ARCH"} & env.keys()
    restore = steps["Install dependencies"]["run"]
    assert all(value in restore for value in (
        "set -euo pipefail", "--require-hashes", "--only-binary=:all:", "--force-reinstall",
        '--report "$M5_PROFILE_INSTALL_REPORT"', '-r "$M5_PROFILE_LOCK"', "python -m pip check",
    ))
    qualifier = steps["Verify clean native Linux profiles and model intervals"]["run"]
    assert qualifier == (
        'python scripts/check_dependency_profile.py --all --output "$M5_EVIDENCE_DIR/native-profiles"'
    )
    tests = steps["Run tests"]["run"]
    assert "python -m pytest" in tests and "--junitxml=test-results.xml" in tests
    arguments = tests.split("python -m pytest", 1)[1]
    assert not any(flag in arguments for flag in ("--ignore", "--deselect", " -k ", " -m ", "|| true"))
    assert not job.get("continue-on-error", False)
    assert not any(step.get("continue-on-error", False) for step in job["steps"])
    assert steps["Upload backend results"]["if"] == "always()"
    capture = steps["Upload unapproved native M5 candidate and raw predictions"]
    assert capture["if"] == "always()"
    assert capture["with"]["path"] == evidence + "/"
    assert steps["Upload native Linux profile evidence"]["with"]["path"] == evidence + "/native-profiles/"
    assert capture["with"]["if-no-files-found"] == "error"
    assert all("bless_m5_baseline" not in step.get("run", "") for step in job["steps"])


@pytest.fixture
def lock_module():
    spec = importlib.util.spec_from_file_location("lock_dependencies", ROOT / "scripts" / "lock_dependencies.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_check_rejects_modified_content(tmp_path, monkeypatch, lock_module):
    monkeypatch.setattr(lock_module, "LOCK_ROOT", tmp_path)
    monkeypatch.setattr(lock_module, "RECORD", ROOT / "requirements" / "manifest.json")
    target = tmp_path / "py311" / "dev.txt"
    target.parent.mkdir()
    target.write_text("# corrupted generated content\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated content changed"):
        lock_module.verify()


@pytest.mark.parametrize("version", ["2.5.2rc1", "2.6.0.dev1"])
def test_release_policy_rejects_prerelease_pins(tmp_path, lock_module, version):
    path = tmp_path / "prerelease.txt"
    path.write_text(f"wrapt=={version}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unqualified prerelease"):
        lock_module._read_pins(path)


@pytest.mark.parametrize("include_valid_alternative", [False, True])
def test_release_policy_rejects_yanked_artifacts(lock_module, include_valid_alternative):
    urls = [{"digests": {"sha256": "0" * 64}, "yanked": True}]
    if include_valid_alternative:
        urls.append({"digests": {"sha256": "1" * 64}, "yanked": False})
    with pytest.raises(ValueError, match="yanked"):
        lock_module._check_release_artifacts("example", "1.0", ("0" * 64,), urls)


def test_release_policy_rejects_non_upstream_hash(lock_module):
    urls = [{"digests": {"sha256": "0" * 64}, "yanked": False}]
    with pytest.raises(ValueError, match="not an upstream artifact"):
        lock_module._check_release_artifacts("example", "1.0", ("1" * 64,), urls)


def test_release_policy_accepts_published_non_yanked_hash(lock_module):
    urls = [{"digests": {"sha256": "0" * 64}, "yanked": False}]
    lock_module._check_release_artifacts("example", "1.0", ("0" * 64,), urls)


@pytest.mark.parametrize("replacement", ["example==1.1", 'example==1.0; sys_platform == "win32"'])
def test_hash_export_cannot_promote_cached_versions_or_change_markers(tmp_path, lock_module, replacement):
    resolved = tmp_path / "resolved.txt"
    exported = tmp_path / "hashed.txt"
    resolved.write_text("example==1.0\n", encoding="utf-8")
    exported.write_text(replacement + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed the resolved versions or platform markers"):
        lock_module._check_resolution_preserved(resolved, exported)


def test_hash_export_preserves_resolution_while_adding_hashes(tmp_path, lock_module):
    resolved = tmp_path / "resolved.txt"
    exported = tmp_path / "hashed.txt"
    resolved.write_text('example==1.0; sys_platform != "darwin"\n', encoding="utf-8")
    exported.write_text('example==1.0; sys_platform != "darwin" --hash=sha256:' + "0" * 64 + "\n", encoding="utf-8")
    lock_module._check_resolution_preserved(resolved, exported)
