"""Shared FastAPI dependencies for router endpoints."""

from __future__ import annotations

import re
from typing import NoReturn

from fastapi import HTTPException, Request, UploadFile

# Regex for safe path-segment identifiers (LOB names, series IDs, model names).
# Allows alphanumeric, hyphens, underscores, and dots (no slashes, no ..).
_SAFE_PATH_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")

# Maximum upload file size: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def validate_path_param(value: str, name: str = "parameter") -> str:
    """Validate a path parameter is a safe filesystem identifier.

    Rejects path traversal attempts (../, /, etc.) and empty strings.
    Returns the validated value unchanged.
    """
    if not value or not _SAFE_PATH_SEGMENT.match(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {name}: '{value}'. "
                f"Must be 1-128 characters, alphanumeric/hyphens/underscores/dots, "
                f"and cannot start with a dot or hyphen."
            ),
        )
    return value


async def validate_upload_size(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file and enforce a size limit.

    Returns the file content as bytes.
    """
    content = await file.read()
    if len(content) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / (1024*1024):.1f} MB). Maximum allowed: {mb:.0f} MB.",
        )
    return content


def run_reader(request: Request):
    """Use only the server-selected authority and its explicit legacy roots."""
    from ..pipeline.readers import RunReader

    data = request.app.state.data_dir
    return RunReader(
        request.app.state.runs_dir, data / "forecasts",
        (data / "history", data / "actuals"),
    )


def metric_store(request: Request):
    from ..metrics.store import MetricStore

    return MetricStore(str(request.app.state.metrics_dir), str(request.app.state.runs_dir))


def metric_records(
    request: Request, lob: str, run_type: str = "backtest",
    run_id: str | None = None, *, latest: bool = False,
):
    """Resolve run scope before exposing metric rows; legacy identity stays unknown."""
    from ..pipeline.run_store import RunIntegrityError, RunNotFoundError, validate_lob

    try:
        validate_lob(lob)
        if run_type not in {"backtest", "live"}:
            raise ValueError("Unsupported metric run type")
        authority = run_reader(request).store
        selected = authority.get(lob, run_id, run_type) if run_id is not None else (
            authority.latest(lob, "backtest") if latest and run_type == "backtest" else None
        )
        frame = metric_store(request).read(
            lob=lob, run_type=run_type,
            run_id=selected.run_id if selected else None, latest=latest,
        )
        return frame, selected
    except (RunIntegrityError, RunNotFoundError, ValueError, OSError, KeyError) as exc:
        raise_run_error(exc)


def run_links(lob: str, run_id: str, run_type: str):
    from ..pipeline.run_store import validate_lob, validate_run_id
    from .schemas import RunLinks

    validate_lob(lob)
    validate_run_id(run_id)
    base = f"/pipeline/runs/{lob}/{run_id}"
    return RunLinks(
        metadata=base, config=base + "/config", input_lineage=base + "/input",
        forecast=f"/forecast/{lob}?run_id={run_id}" if run_type == "forecast" else None,
        leaderboard=f"/metrics/leaderboard/{lob}?run_id={run_id}&run_type=backtest" if run_type == "backtest" else None,
    )


def raise_run_error(error: Exception) -> NoReturn:
    """Bounded transport diagnostics, never filesystem paths or arbitrary values."""
    from ..pipeline.run_store import RunIntegrityError, RunNotFoundError

    if isinstance(error, RunNotFoundError):
        raise HTTPException(404, detail="No committed run or legacy artifact matches this request") from error
    if isinstance(error, RunIntegrityError):
        raise HTTPException(500, detail={
            "code": "run_integrity_error", "reason": error.code, "run_id": error.run_id,
        }) from error
    if isinstance(error, ValueError):
        raise HTTPException(400, detail="Invalid run selector or artifact request") from error
    raise HTTPException(500, detail="Run storage could not be read") from error
