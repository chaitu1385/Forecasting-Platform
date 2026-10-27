"""Local immutable run bundles with one no-replace manifest commit marker.

Files are closed, fsynced and hashed before the manifest hard link makes a run
visible. This profile requires local hard-link semantics (tested on NTFS/Linux),
not a network/OneLake transaction. Windows cannot promise directory-fsync or
power-loss durability. A post-link failure is explicitly committed/uncertain.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import stat
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import polars as pl

from ..config.schema import PlatformConfig
from ..observability.context import new_run_id
from .manifest import PipelineManifest, canonical_json, configuration_snapshot

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
_CANDIDATE_NAME = ".manifest.pending.json"
MAX_METADATA_BYTES = 2 * 1024 * 1024
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_LOB = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_KINDS = frozenset({"forecast", "backtest"})
FRAME_MEMBERS = {
    name: f"{name}.parquet" for name in (
        "input", "prepared", "forecast", "metrics", "champions", "leaderboard",
        "product_master", "mapping_table", "overrides", "external_features",
        "future_features", "conformal_residuals", "champion_table",
    )
}
JSON_MEMBERS = {"config": "config.json", "validation": "validation.json"}
MEMBERS = {**FRAME_MEMBERS, **JSON_MEMBERS}
_REQUIRED = {
    "forecast": frozenset({"input", "prepared", "forecast", "config", "validation"}),
    "backtest": frozenset({"input", "prepared", "metrics", "champions", "leaderboard", "config", "validation"}),
}


class RunIntegrityError(RuntimeError):
    """A committed artifact cannot be trusted; never substitute an older run."""

    def __init__(self, code: str, run_id: Optional[str] = None):
        self.code = code
        self.run_id = run_id
        super().__init__(f"Run integrity check failed: {code}" + (f" (run_id={run_id})" if run_id else ""))


class RunNotFoundError(FileNotFoundError):
    """The requested committed run does not exist in this authority."""


class RunPublicationError(RuntimeError):
    """Publication failed, with an explicit result on either side of commit."""

    def __init__(self, run_id: str, stage: str, outcome: str):
        self.run_id = run_id
        self.stage = stage
        self.outcome = outcome
        super().__init__(f"Run publication could not be confirmed (run_id={run_id}, stage={stage}, outcome={outcome})")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be 32 lowercase hexadecimal characters")
    return run_id


def validate_lob(lob: str) -> str:
    if not isinstance(lob, str) or not _LOB.fullmatch(lob):
        raise ValueError("LOB must be a safe 1-128 character identifier")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if lob.endswith(".") or lob.split(".")[0].lower() in reserved:
        raise ValueError("LOB must not be a device name or filesystem alias")
    return lob


def resolve_run_root(config: PlatformConfig, run_type: str) -> Path:
    """Respect explicit authority; otherwise stay inside this operation's root."""
    if run_type not in _KINDS:
        raise ValueError("Unsupported run type")
    if config.output.runs_path is not None:
        if not str(config.output.runs_path).strip() or "://" in str(config.output.runs_path):
            raise ValueError("output.runs_path must be a nonblank local filesystem root")
        return Path(config.output.runs_path)
    operation_root = config.output.forecast_path if run_type == "forecast" else config.output.metrics_path
    if not str(operation_root).strip() or "://" in str(operation_root):
        raise ValueError("Run bundles require a local operation output root")
    return Path(operation_root) / "runs"


def utc_timestamp(value: Optional[datetime] = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _entry_info(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise RunIntegrityError("symlink_or_reparse_point")
    return info


def checked_child(root: Path, *parts: str) -> Path:
    """Resolve only fixed safe children; reject symlinks/junctions at each level."""
    root = root.resolve()
    current = root
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part or ":" in part:
            raise ValueError("Invalid artifact path component")
        current = current / part
        _entry_info(current)
        if not current.resolve().is_relative_to(root):
            raise RunIntegrityError("outside_store_root")
    return current


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, writer: Callable[[Any], None]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def _read_file(path: Path, maximum: Optional[int] = None) -> bytes:
    info = _entry_info(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise RunIntegrityError("missing_or_nonregular_member")
    if maximum is not None and info.st_size > maximum:
        raise RunIntegrityError("metadata_too_large")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        data = stream.read() if maximum is None else stream.read(maximum + 1)
    if maximum is not None and len(data) > maximum:
        raise RunIntegrityError("metadata_too_large")
    return data


def _file_hash(path: Path) -> str:
    info = _entry_info(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise RunIntegrityError("missing_or_nonregular_member")
    result = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_frame_append(frame: pl.DataFrame, directory: Path, prefix: str = "metrics") -> Path:
    """Publish a standalone frame atomically without overwriting another append."""
    directory.mkdir(parents=True, exist_ok=True)
    _entry_info(directory)
    identity = new_run_id()
    pending = checked_child(directory, f".pending-{identity}")
    destination = checked_child(directory, f"{prefix}_{identity}.parquet")
    _write_new(pending, lambda stream: frame.write_parquet(stream))
    _fsync_directory(directory)
    try:
        os.link(pending, destination)
    except OSError as exc:
        if destination.exists():
            raise RunPublicationError(identity, "standalone_commit", "uncertain") from exc
        raise RunPublicationError(identity, "standalone_commit", "not_committed") from exc
    try:
        _fsync_directory(directory)
    except OSError as exc:
        raise RunPublicationError(identity, "standalone_directory_fsync", "committed") from exc
    try:
        pending.unlink()
    except OSError:
        logger.warning("Committed metric append retained a private staging link: append_id=%s", identity)
    return destination


def atomic_bytes_create(destination: Path, payload: bytes) -> Path:
    """Create one immutable standalone JSON/marker file without replacement."""
    directory = destination.parent.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = checked_child(directory, destination.name)
    if _entry_info(destination) is not None:
        raise FileExistsError("The destination already exists and will not be overwritten")
    identity = new_run_id()
    pending = checked_child(directory, f".pending-{identity}")
    _write_new(pending, lambda stream: stream.write(payload))
    _fsync_directory(directory)
    try:
        os.link(pending, destination)
    except FileExistsError:
        raise
    except OSError as exc:
        outcome = "uncertain" if destination.exists() else "not_committed"
        raise RunPublicationError(identity, "standalone_commit", outcome) from exc
    try:
        _fsync_directory(directory)
    except OSError as exc:
        raise RunPublicationError(identity, "standalone_directory_fsync", "committed") from exc
    try:
        pending.unlink()
    except OSError:
        logger.warning("Committed standalone artifact retained a private staging link: append_id=%s", identity)
    return destination


class RunStore:
    """One local filesystem authority for both forecast and backtest bundles."""

    def __init__(self, root: Path | str):
        if "://" in str(root):
            raise ValueError("Run bundles require a local filesystem root, not an object-store URI")
        self.root = Path(root).resolve()
        info = _entry_info(self.root)
        if info is not None and not stat.S_ISDIR(info.st_mode):
            raise RunIntegrityError("store_root_is_not_directory")

    def directory(self, lob: str, run_id: str) -> Path:
        return checked_child(self.root, validate_lob(lob), validate_run_id(run_id))

    def publish(
        self,
        manifest: PipelineManifest,
        config: PlatformConfig,
        frames: Mapping[str, pl.DataFrame],
        validation: Mapping[str, Any],
    ) -> PipelineManifest:
        """Publish all required members or raise an explicit commit outcome.

        A completed marker is immutable. Failures after link creation never
        delete it or masquerade as an unpublished attempt.
        """
        validate_run_id(manifest.run_id)
        validate_lob(manifest.lob)
        if manifest.run_type not in _KINDS or manifest.lob != config.lob:
            raise ValueError("Invalid run type or LOB")
        if set(frames) - FRAME_MEMBERS.keys():
            raise ValueError("Unknown frame artifact kind")
        if not (_REQUIRED[manifest.run_type] - JSON_MEMBERS.keys()).issubset(frames):
            raise ValueError("Run is missing required frame members")
        if manifest.run_type == "backtest":
            metrics = frames["metrics"]
            for column, value in (("run_id", manifest.run_id), ("run_type", "backtest"), ("lob", manifest.lob)):
                if column not in metrics.columns or metrics.filter(pl.col(column).is_null() | (pl.col(column) != value)).height:
                    raise ValueError("Backtest metric identity does not match the run")
        directory = self.directory(manifest.lob, manifest.run_id)
        stage = "reserve"
        reserved = False
        marker_bytes = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            parent = checked_child(self.root, manifest.lob)
            parent.mkdir(exist_ok=True)
            _entry_info(parent)
            directory.mkdir(mode=0o700)
            reserved = True
            attempt = {
                "schema_version": SCHEMA_VERSION, "run_id": manifest.run_id,
                "lob": manifest.lob, "run_type": manifest.run_type,
                "started_at": manifest.started_at,
            }
            _write_new(directory / ".attempt.json", lambda stream: stream.write(canonical_json(attempt)))
            artifacts: Dict[str, Dict[str, Any]] = {}
            for kind, frame in frames.items():
                stage = kind
                member = checked_child(directory, FRAME_MEMBERS[kind])
                _write_new(member, lambda stream, value=frame: value.write_parquet(stream))
                artifacts[kind] = {
                    "file": FRAME_MEMBERS[kind], "sha256": _file_hash(member),
                    "bytes": member.stat().st_size, "rows": frame.height,
                    "schema": {name: str(dtype).replace("Utf8", "String") for name, dtype in frame.schema.items()},
                }
            stage = "config"
            snapshot = configuration_snapshot(config, manifest.execution)
            self._validate_settings_and_validation(snapshot, dict(validation), manifest)
            for kind, value in (("config", snapshot), ("validation", dict(validation))):
                stage = kind
                payload = canonical_json(value)
                if len(payload) > MAX_METADATA_BYTES:
                    raise ValueError("Run metadata exceeds the supported size")
                member = checked_child(directory, JSON_MEMBERS[kind])
                _write_new(member, lambda stream, data=payload: stream.write(data))
                artifacts[kind] = {
                    "file": JSON_MEMBERS[kind], "sha256": _file_hash(member), "bytes": len(payload),
                }
            committed = replace(
                manifest, schema_version=SCHEMA_VERSION, status="completed",
                completed_at=utc_timestamp(), artifacts=artifacts,
                input_snapshot_stage="normalized_raw_dataframe",
                input_digest_kind="sha256:stored-parquet-bytes",
                config_digest_kind="sha256:canonical-json-v1",
                input_data_hash=artifacts["input"]["sha256"],
                prepared_data_hash=artifacts["prepared"]["sha256"],
                config_hash=artifacts["config"]["sha256"],
                forecast_file=FRAME_MEMBERS["forecast"] if manifest.run_type == "forecast" else "",
            )
            stage = "manifest"
            marker_bytes = canonical_json(asdict(committed))
            if len(marker_bytes) > MAX_METADATA_BYTES:
                raise ValueError("Run metadata exceeds the supported size")
            candidate = checked_child(directory, _CANDIDATE_NAME)
            _write_new(candidate, lambda stream: stream.write(marker_bytes))
            self._validate_manifest(json.loads(marker_bytes), manifest.lob, manifest.run_id)
            _fsync_directory(directory)
            _fsync_directory(parent)
            stage = "commit"
            os.link(candidate, checked_child(directory, MANIFEST_NAME))
            stage = "commit_directory_fsync"
            _fsync_directory(directory)
            return committed
        except Exception as exc:
            outcome = self._commit_outcome(directory, marker_bytes) if reserved else "not_committed"
            if reserved and outcome == "not_committed":
                self._record_failure(directory, manifest.run_id, stage, type(exc).__name__)
            logger.error("Run publication error: run_id=%s stage=%s outcome=%s error_type=%s",
                         manifest.run_id, stage, outcome, type(exc).__name__)
            raise RunPublicationError(manifest.run_id, stage, outcome) from exc

    @staticmethod
    def _commit_outcome(directory: Path, expected: Optional[bytes]) -> str:
        try:
            marker = directory / MANIFEST_NAME
            if _entry_info(marker) is None:
                return "not_committed"
            data = _read_file(marker, MAX_METADATA_BYTES)
            return "committed" if expected is not None and data == expected else "uncertain"
        except (OSError, RunIntegrityError):
            return "uncertain"

    @staticmethod
    def _record_failure(directory: Path, run_id: str, stage: str, error_type: str) -> None:
        payload = canonical_json({
            "run_id": run_id, "status": "not_committed", "stage": stage,
            "error_type": error_type, "timestamp": utc_timestamp(),
        })
        try:
            _write_new(directory / "failure.json", lambda stream: stream.write(payload))
        except OSError:
            logger.error("Run failure diagnostics could not be persisted: run_id=%s", run_id)

    def get(self, lob: str, run_id: str, run_type: Optional[str] = None) -> PipelineManifest:
        """Read a committed manifest and verify every declared member's bytes."""
        directory = self.directory(lob, run_id)
        marker = checked_child(directory, MANIFEST_NAME)
        if _entry_info(marker) is None:
            raise RunNotFoundError("No committed run exists for this LOB and run ID")
        try:
            data = json.loads(_read_file(marker, MAX_METADATA_BYTES))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RunIntegrityError("invalid_manifest_json", run_id) from exc
        manifest = self._validate_manifest(data, lob, run_id)
        if run_type is not None and manifest.run_type != run_type:
            raise RunNotFoundError("The run has a different operation type")
        for artifact in manifest.artifacts.values():
            member = checked_child(directory, artifact["file"])
            info = _entry_info(member)
            if info is None or info.st_size != artifact["bytes"] or _file_hash(member) != artifact["sha256"]:
                raise RunIntegrityError("artifact_checksum_mismatch", run_id)
        if manifest.input_data_hash != manifest.artifacts["input"]["sha256"]:
            raise RunIntegrityError("input_identity_mismatch", run_id)
        if manifest.config_hash != manifest.artifacts["config"]["sha256"]:
            raise RunIntegrityError("config_identity_mismatch", run_id)
        if manifest.prepared_data_hash != manifest.artifacts["prepared"]["sha256"]:
            raise RunIntegrityError("prepared_identity_mismatch", run_id)
        settings = self._json_payload(directory, "config", manifest)
        validation = self._json_payload(directory, "validation", manifest)
        self._validate_settings_and_validation(settings, validation, manifest)
        return manifest

    @staticmethod
    def _json_payload(directory: Path, kind: str, manifest: PipelineManifest) -> Dict[str, Any]:
        payload = _read_file(checked_child(directory, JSON_MEMBERS[kind]), MAX_METADATA_BYTES)
        if hashlib.sha256(payload).hexdigest() != manifest.artifacts[kind]["sha256"]:
            raise RunIntegrityError("artifact_checksum_mismatch", manifest.run_id)
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RunIntegrityError("invalid_json_member", manifest.run_id) from exc
        if not isinstance(data, dict):
            raise RunIntegrityError("invalid_json_member_type", manifest.run_id)
        return data

    @staticmethod
    def _validate_settings_and_validation(
        settings: Mapping[str, Any], validation: Mapping[str, Any], manifest: PipelineManifest,
    ) -> None:
        try:
            configured = settings["settings"]
            frequency = configured["forecast"]["frequency"]
            horizon = configured["forecast"]["horizon_weeks"]
            enabled = configured["post_validation"]["enabled"]
            if (
                settings["schema_version"] != SCHEMA_VERSION
                or settings["normalization"] != "canonical-json-v1"
                or configured["lob"] != manifest.lob
                or frequency not in {"D", "W", "M", "Q"}
                or type(horizon) is not int or horizon < 1 or horizon != manifest.forecast_horizon
                or type(enabled) is not bool
                or type(validation.get("skipped")) is not bool
                or validation["skipped"] == enabled
            ):
                raise RunIntegrityError("configuration_or_validation_mismatch", manifest.run_id)
            if settings.get("execution") != json.loads(canonical_json(manifest.execution)):
                raise RunIntegrityError("execution_identity_mismatch", manifest.run_id)
            if enabled and (
                validation.get("grade") not in {"A", "B", "C", "D", "F"}
                or type(validation.get("score")) is not int
                or not isinstance(validation.get("layers"), dict)
                or not isinstance(validation.get("confidence"), dict)
                or not isinstance(validation.get("badge"), str)
            ):
                raise RunIntegrityError("invalid_validation_result", manifest.run_id)
            if (
                manifest.post_validation_grade != (validation.get("grade") if enabled else None)
                or manifest.post_validation_score != (validation.get("score") if enabled else None)
            ):
                raise RunIntegrityError("validation_summary_mismatch", manifest.run_id)
        except (KeyError, TypeError) as exc:
            raise RunIntegrityError("invalid_configuration_snapshot", manifest.run_id) from exc

    @staticmethod
    def _validate_manifest(data: Any, lob: str, run_id: str) -> PipelineManifest:
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise RunIntegrityError("invalid_manifest_schema", run_id)
        if data["schema_version"] != SCHEMA_VERSION:
            raise RunIntegrityError("unsupported_schema_version", run_id)
        if data.get("run_id") != run_id or data.get("lob") != lob:
            raise RunIntegrityError("run_identity_mismatch", run_id)
        run_type = data.get("run_type")
        if not isinstance(run_type, str) or run_type not in _KINDS or data.get("status") != "completed":
            raise RunIntegrityError("invalid_commit_state", run_id)
        if (
            data.get("input_snapshot_stage") != "normalized_raw_dataframe"
            or data.get("input_digest_kind") != "sha256:stored-parquet-bytes"
            or data.get("config_digest_kind") != "sha256:canonical-json-v1"
            or data.get("forecast_file") != ("forecast.parquet" if run_type == "forecast" else "")
        ):
            raise RunIntegrityError("invalid_provenance_schema", run_id)
        for field in ("execution", "source", "reproducibility"):
            if not isinstance(data.get(field), dict):
                raise RunIntegrityError("invalid_provenance_fields", run_id)
        models = data.get("models")
        if not isinstance(models, list) or not models or any(
            not isinstance(model, dict) or not isinstance(model.get("name"), str)
            or not isinstance(model.get("parameters"), dict) for model in models
        ):
            raise RunIntegrityError("invalid_model_metadata", run_id)
        for field in ("started_at", "completed_at"):
            value = data.get(field)
            if not isinstance(value, str):
                raise RunIntegrityError("invalid_run_timestamp", run_id)
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise RunIntegrityError("invalid_run_timestamp", run_id) from exc
            if utc_timestamp(parsed) != value:
                raise RunIntegrityError("noncanonical_run_timestamp", run_id)
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict) or not _REQUIRED[run_type].issubset(artifacts):
            raise RunIntegrityError("missing_artifact_descriptors", run_id)
        for kind, artifact in artifacts.items():
            if kind not in MEMBERS or not isinstance(artifact, dict) or artifact.get("file") != MEMBERS[kind]:
                raise RunIntegrityError("unsafe_artifact_member", run_id)
            if type(artifact.get("bytes")) is not int or artifact["bytes"] < 0:
                raise RunIntegrityError("invalid_artifact_size", run_id)
            checksum = artifact.get("sha256")
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise RunIntegrityError("invalid_artifact_digest", run_id)
            if kind in FRAME_MEMBERS and (
                type(artifact.get("rows")) is not int or artifact["rows"] < 0
                or not isinstance(artifact.get("schema"), dict)
                or not all(isinstance(name, str) and isinstance(dtype, str)
                           for name, dtype in artifact["schema"].items())
            ):
                raise RunIntegrityError("invalid_frame_descriptor", run_id)
        try:
            manifest = PipelineManifest(**data)
        except TypeError as exc:
            raise RunIntegrityError("invalid_manifest_fields", run_id) from exc
        output = "forecast" if run_type == "forecast" else "metrics"
        if (
            type(manifest.input_row_count) is not int
            or manifest.input_row_count != artifacts["input"]["rows"]
            or type(manifest.forecast_row_count) is not int
            or manifest.forecast_row_count != artifacts[output]["rows"]
        ):
            raise RunIntegrityError("manifest_row_count_mismatch", run_id)
        return manifest

    def list(self, lob: Optional[str] = None, run_type: Optional[str] = None) -> list[PipelineManifest]:
        """List only committed bundles, ordered by attempt start then ID."""
        if run_type is not None and run_type not in _KINDS:
            raise ValueError("Unsupported run type")
        if _entry_info(self.root) is None:
            return []
        lobs: Iterable[Path] = [checked_child(self.root, validate_lob(lob))] if lob else sorted(self.root.iterdir())
        manifests = []
        for partition in lobs:
            if partition.name.startswith("."):
                continue
            info = _entry_info(partition)
            if info is None or not stat.S_ISDIR(info.st_mode):
                continue
            validate_lob(partition.name)
            for directory in partition.iterdir():
                if not _RUN_ID.fullmatch(directory.name):
                    continue
                info = _entry_info(directory)
                if info is None or not stat.S_ISDIR(info.st_mode):
                    raise RunIntegrityError("invalid_run_directory", directory.name)
                if _entry_info(checked_child(directory, MANIFEST_NAME)) is None:
                    continue
                manifest = self.get(partition.name, directory.name)
                if run_type is None or manifest.run_type == run_type:
                    manifests.append(manifest)
        return sorted(manifests, key=lambda manifest: (manifest.started_at, manifest.run_id), reverse=True)

    def latest(self, lob: str, run_type: str) -> Optional[PipelineManifest]:
        manifests = self.list(lob=lob, run_type=run_type)
        return manifests[0] if manifests else None

    def frame(self, lob: str, run_id: str, kind: str) -> pl.DataFrame:
        if kind not in FRAME_MEMBERS:
            raise ValueError("Unknown frame member")
        manifest = self.get(lob, run_id)
        if kind not in manifest.artifacts:
            raise RunNotFoundError("The requested member is not part of this run")
        member = checked_child(self.directory(lob, run_id), FRAME_MEMBERS[kind])
        payload = _read_file(member)
        if hashlib.sha256(payload).hexdigest() != manifest.artifacts[kind]["sha256"]:
            raise RunIntegrityError("artifact_checksum_mismatch", run_id)
        try:
            frame = pl.read_parquet(io.BytesIO(payload))
        except Exception as exc:
            raise RunIntegrityError("unreadable_parquet_member", run_id) from exc
        descriptor = manifest.artifacts[kind]
        schema = {name: str(dtype).replace("Utf8", "String") for name, dtype in frame.schema.items()}
        if frame.height != descriptor.get("rows") or schema != descriptor.get("schema"):
            raise RunIntegrityError("frame_schema_or_rows_mismatch", run_id)
        if kind == "metrics":
            for column, value in (("run_id", run_id), ("run_type", manifest.run_type), ("lob", lob)):
                if column not in frame.columns or frame.filter(pl.col(column).is_null() | (pl.col(column) != value)).height:
                    raise RunIntegrityError("metric_identity_mismatch", run_id)
        return frame

    def json_member(self, lob: str, run_id: str, kind: str) -> Dict[str, Any]:
        if kind not in JSON_MEMBERS:
            raise ValueError("Unknown JSON member")
        manifest = self.get(lob, run_id)
        return self._json_payload(self.directory(lob, run_id), kind, manifest)
