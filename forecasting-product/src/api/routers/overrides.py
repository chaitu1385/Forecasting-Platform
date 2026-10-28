"""Planner override CRUD endpoints."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from ...auth.models import Permission, User
from ...auth.rbac import require_permission

if TYPE_CHECKING:
    from ...overrides.store import OverrideStore, ParquetOverrideStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overrides", tags=["overrides"])


def _finite_proportion(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        # Do not echo a non-JSON number in FastAPI's validation-error payload.
        raise HTTPException(status_code=422, detail="proportion must be a finite number in [0, 1].")
    return value


OverrideProportion = Annotated[float, BeforeValidator(_finite_proportion)]


class CreateOverrideRequest(BaseModel):
    old_sku: str
    new_sku: str
    proportion: OverrideProportion = Field(ge=0.0, le=1.0)
    scenario: str = "manual"
    ramp_shape: str = "linear"
    effective_date: Optional[str] = None
    notes: Optional[str] = None


class UpdateOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proportion: Optional[OverrideProportion] = Field(None, ge=0.0, le=1.0, allow_inf_nan=False)
    scenario: Optional[str] = None
    ramp_shape: Optional[str] = None
    effective_date: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("proportion", "scenario", "ramp_shape")
    @classmethod
    def reject_explicit_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("This field cannot be null; omit it to keep the stored value.")
        return value


@contextmanager
def _override_store(
    request: Request, operation: str,
) -> Iterator[OverrideStore | ParquetOverrideStore]:
    from ...overrides.store import (
        OverrideStoreBusy,
        OverrideValidationError,
        get_override_store,
    )

    store = None
    try:
        try:
            store = get_override_store(path=str(request.app.state.data_dir / "overrides"))
            yield store
        finally:
            if store is not None:
                store.close()
    except OverrideStoreBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OverrideValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to %s overrides", operation)
        raise HTTPException(status_code=500, detail=f"Failed to {operation} overrides.") from exc


@router.get("")
def list_overrides(
    request: Request,
    old_sku: Optional[str] = Query(None),
    new_sku: Optional[str] = Query(None),
    user: User = Depends(require_permission(Permission.VIEW_FORECASTS)),
):
    """List all planner overrides, optionally filtered by SKU."""
    with _override_store(request, "read") as store:
        if old_sku or new_sku:
            df = store.get_overrides(old_sku=old_sku, new_sku=new_sku)
        else:
            df = store.get_all()
        return {"count": df.height, "overrides": df.to_dicts()}


@router.post("")
def create_override(
    http_request: Request,
    body: CreateOverrideRequest,
    user: User = Depends(require_permission(Permission.CREATE_OVERRIDE)),
):
    """Create a new planner override."""
    with _override_store(http_request, "create") as store:
        override_id = store.add_override(
            old_sku=body.old_sku,
            new_sku=body.new_sku,
            proportion=body.proportion,
            scenario=body.scenario,
            ramp_shape=body.ramp_shape,
            effective_date=body.effective_date,
            created_by=user.user_id,
            notes=body.notes,
        )
    return {"override_id": override_id, "status": "created"}


@router.put("/{override_id}")
def update_override(
    override_id: str,
    http_request: Request,
    body: UpdateOverrideRequest,
    user: User = Depends(require_permission(Permission.CREATE_OVERRIDE)),
):
    """Update an existing planner override."""
    with _override_store(http_request, "update") as store:
        if not store.update_override(override_id, body.model_dump(exclude_unset=True)):
            raise HTTPException(status_code=404, detail=f"Override '{override_id}' not found.")
    return {"override_id": override_id, "status": "updated"}


@router.delete("/{override_id}")
def delete_override(
    override_id: str,
    request: Request,
    user: User = Depends(require_permission(Permission.DELETE_OVERRIDE)),
):
    """Delete a planner override."""
    with _override_store(request, "delete") as store:
        deleted = store.delete_override(override_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Override '{override_id}' not found.")

    return {"override_id": override_id, "status": "deleted"}
