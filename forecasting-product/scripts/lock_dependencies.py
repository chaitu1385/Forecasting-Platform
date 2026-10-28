"""Compile and verify hash-locked profiles from the canonical pyproject manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
LOCK_ROOT = ROOT / "requirements"
RECORD = LOCK_ROOT / "manifest.json"
SUPPORTED_MINORS = ("3.11", "3.12", "3.13")
PIP_VERSION = "26.2.2"
PIP_TOOLS_VERSION = "7.6.2"
RELEASE_POLICY = "stable-non-yanked-pins-preserved"
PROFILES = {
    "dev": ("api", "ai", "holidays", "visualization", "dev"),
    "core": (),
    "api": ("api", "ai"),
    "ai": ("ai",),
    "fabric": ("holidays", "visualization"),
    "spark": ("spark",),
}
WHEEL_PLATFORMS = (
    ("win_amd64",),
    (
        "manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_26_x86_64",
        "manylinux_2_24_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64",
        "manylinux_2_12_x86_64", "manylinux2010_x86_64", "manylinux_2_5_x86_64",
        "manylinux1_x86_64", "linux_x86_64",
    ),
)


def manifest_digest() -> str:
    """Hash resolution inputs, excluding unrelated pytest and lint settings."""
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    inputs = {
        "build-system": manifest["build-system"],
        "project": manifest["project"],
        "profiles": PROFILES,
        "wheel-platforms": WHEEL_PLATFORMS,
        "pip": PIP_VERSION,
        "pip-tools": PIP_TOOLS_VERSION,
        "release-policy": RELEASE_POLICY,
    }
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_pins(path: Path) -> dict[str, tuple[str, str, tuple[str, ...]]]:
    pins = {}
    for line in path.read_text(encoding="utf-8").replace("\\\n", "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line.split("--hash=", 1)[0].strip())
        specifiers = list(requirement.specifier)
        if requirement.url or len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(f"{path.name}: expected an exact registry pin: {requirement}")
        version = Version(specifiers[0].version)
        if version.is_prerelease or version.is_devrelease:
            raise ValueError(f"Unqualified prerelease: {requirement.name}=={version}")
        name = canonicalize_name(requirement.name)
        if name in pins:
            raise ValueError(f"{path.name}: duplicate pin for {name}")
        pins[name] = (
            str(version), str(requirement.marker or ""),
            tuple(re.findall(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)", line)),
        )
    if not pins:
        raise ValueError(f"{path.name}: empty resolution")
    return pins


def _check_release_artifacts(name: str, version: str, hashes: tuple[str, ...], urls: list[dict]) -> None:
    artifacts = {entry["digests"]["sha256"]: entry for entry in urls}
    if not any(not entry["yanked"] for entry in urls):
        raise ValueError(f"{name}=={version}: upstream release is yanked")
    for digest in hashes:
        if digest not in artifacts:
            raise ValueError(f"{name}=={version}: hash is not an upstream artifact")
        if artifacts[digest]["yanked"]:
            raise ValueError(f"{name}=={version}: selected upstream artifact is yanked")


def _check_upstream(path: Path, cache: dict[tuple[str, str], list[dict]]) -> None:
    for name, (version, _marker, hashes) in _read_pins(path).items():
        key = (name, version)
        if key not in cache:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
                release = json.load(response)
            if canonicalize_name(release["info"]["name"]) != name or Version(release["info"]["version"]) != Version(version):
                raise ValueError(f"{name}=={version}: unexpected upstream release metadata")
            cache[key] = release["urls"]
        _check_release_artifacts(name, version, hashes, cache[key])


def _check_resolution_preserved(resolved: Path, exported: Path) -> None:
    before = {name: pin[:2] for name, pin in _read_pins(resolved).items()}
    after = {name: pin[:2] for name, pin in _read_pins(exported).items()}
    if before != after:
        raise ValueError("Hash export changed the resolved versions or platform markers")


def verify(upstream: bool = False) -> None:
    """Reject missing, stale or manually altered generated profiles."""
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    expected = manifest_digest()
    cache: dict[tuple[str, str], list[dict]] = {}
    for minor in SUPPORTED_MINORS:
        for profile in PROFILES:
            relative = f"py{minor.replace('.', '')}/{profile}.txt"
            entry = record["profiles"][relative]
            if entry["manifest_sha256"] != expected:
                raise ValueError(f"{relative}: manifest changed; regenerate this profile")
            actual = hashlib.sha256((LOCK_ROOT / relative).read_bytes()).hexdigest()
            if entry["sha256"] != actual:
                raise ValueError(f"{relative}: generated content changed; regenerate this profile")
            if entry.get("release_policy") != RELEASE_POLICY:
                raise ValueError(f"{relative}: regenerate with the current release policy")
            _read_pins(LOCK_ROOT / relative)
            if upstream:
                _check_upstream(LOCK_ROOT / relative, cache)
    print("All supported dependency profiles match their manifest and generated hashes.")


def compile_profiles(selected: list[str], wheelhouse: Path) -> None:
    """Resolve profiles with the matching interpreter and pinned pip-tools."""
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if minor not in SUPPORTED_MINORS:
        raise ValueError(f"Use one of the supported Python minors: {SUPPORTED_MINORS}")
    for package, version in (("pip", PIP_VERSION), ("pip-tools", PIP_TOOLS_VERSION)):
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Regeneration requires {package}=={version}")

    wheelhouse = wheelhouse.resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PIP_CACHE_DIR"] = str(wheelhouse.parent / "pip-cache")
    environment["PIP_TOOLS_CACHE_DIR"] = str(wheelhouse.parent / "pip-tools-cache")
    record = (
        json.loads(RECORD.read_text(encoding="utf-8"))
        if RECORD.exists()
        else {"profiles": {}}
    )
    digest = manifest_digest()
    cache: dict[tuple[str, str], list[dict]] = {}
    directory = Path("requirements") / f"py{minor.replace('.', '')}"
    (ROOT / directory).mkdir(parents=True, exist_ok=True)
    if "dev" not in selected:
        relative = f"py{minor.replace('.', '')}/dev.txt"
        dev_entry = record["profiles"].get(relative)
        if dev_entry is None or dev_entry["manifest_sha256"] != digest:
            raise ValueError("Regenerate the dev profile first to constrain shared dependencies")

    for profile in PROFILES:
        if profile not in selected:
            continue
        output = directory / f"{profile}.txt"
        command = [
            sys.executable, "-m", "piptools", "compile", "pyproject.toml",
            "--resolver=backtracking", "--allow-unsafe", "--no-build-isolation",
            "--strip-extras", "--no-emit-index-url", "--no-emit-trusted-host",
            "--no-emit-find-links", "--newline=lf", "--quiet",
        ]
        for extra in PROFILES[profile]:
            command.extend(("--extra", extra))
        if profile != "dev":
            command.extend(("--constraint", str(directory / "dev.txt")))
        environment["CUSTOM_COMPILE_COMMAND"] = (
            f"python scripts/lock_dependencies.py --profile {profile} --wheelhouse <private-wheel-cache>"
        )
        with tempfile.TemporaryDirectory(prefix="resolve-", dir=wheelhouse.parent) as temp:
            resolved = None
            if profile in ("dev", "spark"):
                # Stage only supported Windows/Linux artifacts, not every upstream wheel.
                unresolved = Path(temp) / "resolved.txt"
                if (ROOT / output).exists():
                    unresolved.write_bytes((ROOT / output).read_bytes())
                print(f"Resolving Python {minor} {profile}", flush=True)
                subprocess.run(
                    [
                        *command, "--find-links", str(wheelhouse),
                        "--output-file", str(unresolved),
                    ],
                    cwd=ROOT, env=environment, check=True,
                )
                _check_upstream(unresolved, cache)
                resolved = unresolved
                for platforms in WHEEL_PLATFORMS:
                    download = [
                        sys.executable, "-m", "pip", "download", "--no-deps",
                        # Exact artifacts only: avoid mirrors' missing PEP 658 sidecars.
                        "--use-deprecated=legacy-resolver",
                        "--only-binary=:all:", "--disable-pip-version-check", "--quiet",
                        "--python-version", minor, "--implementation", "cp",
                        "--abi", f"cp{minor.replace('.', '')}",
                        "--dest", str(wheelhouse), "--requirement", str(unresolved),
                    ]
                    if profile == "spark":
                        download.append("--no-binary=pyspark")
                    for platform in platforms:
                        download.extend(("--platform", platform))
                    print(f"Staging {minor} {profile}: {platforms[0]}", flush=True)
                    subprocess.run(download, cwd=ROOT, env=environment, check=True)
            print(f"Hash-locking Python {minor} {profile}", flush=True)
            exported = Path(temp) / "hashed.txt"
            export_command = [
                *command, "--no-index", "--find-links", str(wheelhouse),
                "--no-annotate", "--generate-hashes", "--output-file", str(exported),
            ]
            if resolved is not None:
                export_command.extend(("--constraint", str(resolved)))
            subprocess.run(export_command, cwd=ROOT, env=environment, check=True)
            if resolved is not None:
                _check_resolution_preserved(resolved, exported)
            _check_upstream(exported, cache)
            (ROOT / output).write_bytes(exported.read_bytes())
        relative = output.relative_to("requirements").as_posix()
        record["profiles"][relative] = {
            "manifest_sha256": digest,
            "sha256": hashlib.sha256((ROOT / output).read_bytes()).hexdigest(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "pip": PIP_VERSION,
            "pip-tools": PIP_TOOLS_VERSION,
            "release_policy": RELEASE_POLICY,
        }
        RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Run explicit regeneration, or an offline freshness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--upstream", action="store_true", help="Also verify live upstream release/yank policy")
    parser.add_argument("--profile", choices=tuple(PROFILES), action="append")
    parser.add_argument("--wheelhouse", type=Path)
    arguments = parser.parse_args()
    if arguments.check:
        if arguments.profile:
            parser.error("--check verifies every profile; do not combine it with --profile")
        verify(upstream=arguments.upstream)
    else:
        if arguments.upstream:
            parser.error("--upstream is only used with --check")
        if arguments.wheelhouse is None:
            parser.error("--wheelhouse must name a private cache directory for regeneration")
        compile_profiles(arguments.profile or list(PROFILES), arguments.wheelhouse)


if __name__ == "__main__":
    main()
