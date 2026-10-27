"""One committed-run resolver with an explicit read-only legacy adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl

from .manifest import PipelineManifest, read_manifest
from .run_store import (
    MAX_METADATA_BYTES,
    RunIntegrityError,
    RunNotFoundError,
    RunStore,
    checked_child,
    validate_lob,
)


@dataclass
class ArtifactRead:
    frame: pl.DataFrame
    run_id: Optional[str]
    source: str
    id_column: str
    time_column: str
    target_column: str
    manifest: Optional[PipelineManifest] = None
    forecast_origin: Optional[str] = None
    input_stage: Optional[str] = None


class RunReader:
    """Resolve exact/latest data without mixing a run with unrelated history."""

    def __init__(
        self, runs_path: Path | str, forecast_path: Path | str,
        history_roots: tuple[Path, ...] = (),
    ):
        self.store = RunStore(runs_path)
        self.forecast_path = Path(forecast_path).resolve()
        self.history_roots = tuple(Path(root).resolve() for root in history_roots)

    def _columns(self, manifest: PipelineManifest) -> tuple[str, str, str]:
        configuration = self.store.json_member(manifest.lob, manifest.run_id, "config")
        try:
            forecast = configuration["settings"]["forecast"]
            columns = tuple(forecast[name] for name in ("series_id_column", "time_column", "target_column"))
        except (KeyError, TypeError) as exc:
            raise RunIntegrityError("invalid_config_columns", manifest.run_id) from exc
        if not all(isinstance(column, str) and column for column in columns) or len(set(columns)) != 3:
            raise RunIntegrityError("invalid_config_columns", manifest.run_id)
        return columns

    @staticmethod
    def _legacy_columns(frame: pl.DataFrame) -> tuple[str, str, str]:
        identifiers = ("series_id", "sku_id", "product_id", "item_id", "sku")
        dates = ("week", "date", "ds", "timestamp", "month", "quarter", "period", "period_end")
        targets = ("quantity", "target", "value", "sales", "demand", "y")
        identifier = next((name for name in identifiers if name in frame.columns), None)
        time = next((name for name in dates if name in frame.columns), None)
        target = next((name for name in targets if name in frame.columns), "quantity")
        if identifier is None or time is None:
            raise RunIntegrityError("legacy_columns_unknown")
        return identifier, time, target

    @staticmethod
    def _legacy_frame(path: Path) -> pl.DataFrame:
        try:
            return pl.read_parquet(path, hive_partitioning=False)
        except Exception as exc:
            raise RunIntegrityError("unreadable_legacy_artifact") from exc

    def forecast(self, lob: str, run_id: Optional[str] = None, *, legacy: bool = False) -> ArtifactRead:
        validate_lob(lob)
        if legacy and run_id is not None:
            raise ValueError("A legacy artifact is not a verified run ID")
        manifest = None
        if not legacy:
            manifest = self.store.get(lob, run_id, "forecast") if run_id is not None else self.store.latest(lob, "forecast")
        if manifest is not None:
            frame = self.store.frame(lob, manifest.run_id, "forecast")
            identifier, time, target = self._columns(manifest)
            if not {identifier, time, "forecast"}.issubset(frame.columns):
                raise RunIntegrityError("forecast_columns_mismatch", manifest.run_id)
            return ArtifactRead(
                frame, manifest.run_id, "committed", identifier, time, target,
                manifest, manifest.forecast_origin,
            )
        directory = checked_child(self.forecast_path, lob)
        files = [
            checked_child(self.forecast_path, lob, file.name)
            for file in directory.glob("forecast_*.parquet")
        ]
        if not files:
            raise RunNotFoundError("No committed or legacy forecast exists for this LOB")
        try:
            path = max(files, key=lambda file: (
                re.sub(r"(?:_[0-9a-f]{32}){1,2}$", "", file.stem),
                file.stat().st_mtime_ns,
                file.name,
            ))
        except OSError as exc:
            raise RunIntegrityError("unreadable_legacy_artifact") from exc
        frame = self._legacy_frame(path)
        identifier, time, target = self._legacy_columns(frame)
        origin = path.stem.removeprefix(f"forecast_{lob}_")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:_[0-9a-f]{32}){1,2}", origin):
            origin = origin[:10]
        try:
            origin = date.fromisoformat(origin).isoformat()
        except ValueError:
            origin = None
        return ArtifactRead(
            frame, None, "legacy", identifier, time, target,
            forecast_origin=origin,
        )

    def history(
        self, lob: str, run_id: Optional[str] = None, *, stage: str = "input", legacy: bool = False,
    ) -> ArtifactRead:
        validate_lob(lob)
        if stage not in {"input", "prepared"}:
            raise ValueError("History stage must be input or prepared")
        if legacy and run_id is not None:
            raise ValueError("A legacy artifact is not a verified run ID")
        manifest = None
        if not legacy:
            if run_id is not None:
                manifest = self.store.get(lob, run_id)
            else:
                runs = self.store.list(lob=lob)
                manifest = runs[0] if runs else None
        if manifest is not None:
            frame = self.store.frame(lob, manifest.run_id, stage)
            identifier, time, target = self._columns(manifest)
            if not {identifier, time, target}.issubset(frame.columns):
                raise RunIntegrityError("history_columns_mismatch", manifest.run_id)
            return ArtifactRead(
                frame, manifest.run_id, "committed", identifier, time, target,
                manifest=manifest, input_stage=stage,
            )
        for root in self.history_roots:
            directory = checked_child(root, lob)
            files = sorted(directory.glob("*.parquet"))
            if files:
                path = checked_child(root, lob, files[-1].name)
                frame = self._legacy_frame(path)
                identifier, time, target = self._legacy_columns(frame)
                return ArtifactRead(frame, None, "legacy", identifier, time, target, input_stage="unknown")
        raise RunNotFoundError("No committed input snapshot or legacy history exists for this LOB")

    def legacy_manifests(self, lob: Optional[str] = None) -> list[PipelineManifest]:
        """Read both historical sidecar names, without promoting their IDs."""
        if lob is not None:
            validate_lob(lob)
        partitions = [checked_child(self.forecast_path, lob)] if lob else sorted(self.forecast_path.glob("*"))
        manifests = []
        for partition in partitions:
            if not partition.is_dir():
                continue
            checked_child(self.forecast_path, partition.name)
            files = sorted(set(partition.glob("forecast_*_manifest.json")) | set(partition.glob("manifest_*.json")))
            for path in files:
                checked_child(self.forecast_path, partition.name, path.name)
                try:
                    if path.stat().st_size > MAX_METADATA_BYTES:
                        raise RunIntegrityError("legacy_metadata_too_large")
                    manifest = read_manifest(str(path))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise RunIntegrityError("unreadable_legacy_manifest") from exc
                if manifest.lob and manifest.lob != partition.name:
                    raise RunIntegrityError("legacy_lob_mismatch")
                if manifest.schema_version != 0:
                    raise RunIntegrityError("unsupported_legacy_schema_version")
                manifest.lob = partition.name
                manifest.status = "legacy"
                manifests.append(manifest)
        return sorted(manifests, key=lambda manifest: manifest.timestamp or "", reverse=True)
