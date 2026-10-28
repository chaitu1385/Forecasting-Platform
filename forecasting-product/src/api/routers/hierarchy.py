"""Hierarchy management endpoints — build tree, aggregate, reconcile."""

from __future__ import annotations

import io
import logging
import math
import re
from typing import TYPE_CHECKING, Optional

import polars as pl
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from ...auth.models import Permission, User
from ...auth.rbac import get_current_user, require_permission
from ...evaluation.calibration import validate_quantile_forecasts
from ..deps import validate_upload_size
from ..schemas import (
    HierarchyCoherenceResult,
    HierarchyEffectiveMethod,
    HierarchyReconcileResponse,
    HierarchyReconciliationMethod,
)

if TYPE_CHECKING:
    from ...hierarchy.tree import HierarchyTree

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hierarchy", tags=["hierarchy"])

_TIMESTAMP_PRECISION_ERROR = (
    "unsupported_timestamp_precision: timestamps must be exactly representable "
    "in microseconds; submicrosecond values are not supported"
)


@router.post("/build")
async def build_hierarchy(
    file: UploadFile = File(...),
    levels: str = Query(..., description="Comma-separated hierarchy levels, root to leaf"),
    id_column: str = Query("series_id", description="Leaf-level ID column"),
    name: str = Query("product", description="Hierarchy name"),
    user: User = Depends(get_current_user),
):
    """Build a hierarchy tree and return structure stats."""
    from ...config.schema import HierarchyConfig
    from ...hierarchy.tree import HierarchyTree

    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    level_list = [level.strip() for level in levels.split(",")]
    missing = [level for level in level_list if level not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Hierarchy levels not found in data: {missing}. Available columns: {df.columns}",
        )

    config = HierarchyConfig(name=name, levels=level_list, id_column=id_column)
    tree = HierarchyTree(config, df)

    # Build stats per level
    level_stats = []
    for level in level_list:
        nodes = tree.get_nodes(level)
        level_stats.append({
            "level": level,
            "node_count": len(nodes),
        })

    leaves = tree.get_leaves()
    s_matrix = tree.summing_matrix()
    s_sample = s_matrix.head(20).to_dicts() if not s_matrix.is_empty() else []

    # Build flat tree_nodes list for sunburst visualization
    tree_nodes = []
    for level in level_list:
        for node in tree.get_nodes(level):
            tree_nodes.append({
                "key": node.key,
                "level": level,
                "parent": node.parent.key if node.parent else "",
                "is_leaf": node.is_leaf,
            })

    return {
        "name": name,
        "levels": level_list,
        "level_stats": level_stats,
        "total_nodes": sum(s["node_count"] for s in level_stats),
        "leaf_count": len(leaves),
        "s_matrix_shape": [s_matrix.height, s_matrix.width],
        "s_matrix_sample": s_sample,
        "tree_nodes": tree_nodes,
    }


@router.post("/aggregate")
async def aggregate_hierarchy(
    file: UploadFile = File(...),
    levels: str = Query(..., description="Comma-separated hierarchy levels"),
    id_column: str = Query("series_id"),
    target_level: str = Query(..., description="Level to aggregate to"),
    value_columns: str = Query("quantity", description="Comma-separated columns to aggregate"),
    time_column: str = Query("week"),
    agg: str = Query("sum", description="Aggregation function: 'sum', 'mean'"),
    top_n: int = Query(10, description="Return top N nodes by total value"),
    user: User = Depends(get_current_user),
):
    """Aggregate leaf-level data to a target hierarchy level."""
    from ...config.schema import HierarchyConfig
    from ...hierarchy.aggregator import HierarchyAggregator
    from ...hierarchy.tree import HierarchyTree

    content = await validate_upload_size(file)
    filename = file.filename or ""
    try:
        if filename.endswith(".parquet"):
            df = pl.read_parquet(io.BytesIO(content))
        else:
            df = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    level_list = [level.strip() for level in levels.split(",")]
    val_cols = [c.strip() for c in value_columns.split(",")]

    config = HierarchyConfig(name="hierarchy", levels=level_list, id_column=id_column)
    tree = HierarchyTree(config, df)
    aggregator = HierarchyAggregator(tree)

    aggregated = aggregator.aggregate_to(
        df, target_level=target_level,
        value_columns=val_cols, time_column=time_column, agg=agg,
    )

    # Top N nodes by total value
    if val_cols and val_cols[0] in aggregated.columns:
        top_nodes = (
            aggregated
            .group_by(target_level)
            .agg(pl.col(val_cols[0]).sum().alias("total"))
            .sort("total", descending=True)
            .head(top_n)
        )
        top_node_ids = top_nodes[target_level].to_list()
        top_data = aggregated.filter(pl.col(target_level).is_in(top_node_ids))
    else:
        top_data = aggregated.head(top_n * 52)

    return {
        "target_level": target_level,
        "total_rows": aggregated.height,
        "unique_nodes": aggregated[target_level].n_unique() if target_level in aggregated.columns else 0,
        "top_n_data": top_data.to_dicts(),
    }


@router.post("/reconcile", response_model=HierarchyReconcileResponse)
async def reconcile_hierarchy(
    file: UploadFile = File(...),
    actuals_file: Optional[UploadFile] = File(None, description="Historical leaf data for top_down"),
    levels: str = Query(..., description="Comma-separated hierarchy levels"),
    id_column: str = Query("series_id"),
    method: str = Query("bottom_up", description="Reconciliation method: bottom_up, top_down, ols, wls, mint"),
    value_columns: str = Query("forecast", description="Comma-separated value columns"),
    time_column: str = Query("week"),
    history_value_column: str = Query("quantity", description="Historical volume column for top_down"),
    validate: bool = Query(False, description="Run structural, logical and business checks; no model-comparison paradox is evaluated"),
    user: User = Depends(require_permission(Permission.RUN_PIPELINE)),
) -> HierarchyReconcileResponse:
    """Reconcile complete leaf panels, or root forecasts using earlier leaf history."""
    from ...config.schema import HierarchyConfig, ReconciliationConfig
    from ...hierarchy.reconciler import Reconciler
    from ...hierarchy.tree import HierarchyTree

    try:
        requested_method = HierarchyReconciliationMethod(method)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="unsupported_reconciliation_mode: use bottom_up, top_down, ols, wls or mint; middle_out requires a separate middle-level input contract.",
        ) from exc

    df = await _read_reconciliation_upload(file, time_column)
    history = await _read_reconciliation_upload(actuals_file, time_column) if actuals_file is not None else None
    try:
        level_list = _column_names(levels, "levels")
        val_cols = _column_names(value_columns, "value_columns")
        leaf = level_list[-1]
        if not id_column.strip() or not time_column.strip():
            raise ValueError("id_column and time_column must be nonempty column names")
        if time_column in level_list or time_column == id_column:
            raise ValueError("Time and hierarchy identity columns must be distinct")
        if set(val_cols).intersection([*level_list, id_column, time_column]):
            raise ValueError("Value columns must be distinct from time and identity columns")
        if id_column in level_list[:-1]:
            raise ValueError("id_column must identify leaves, not an upper hierarchy level")

        top_down = requested_method == HierarchyReconciliationMethod.TOP_DOWN
        source_level = level_list[0] if top_down else leaf
        if top_down:
            if len(level_list) < 2:
                raise ValueError("top_down requires a root and a distinct leaf level")
            if history is None:
                raise ValueError("top_down requires actuals_file with earlier leaf-level history")
            if not history_value_column.strip():
                raise ValueError("history_value_column must be nonempty")
            if history_value_column in [*level_list, id_column, time_column]:
                raise ValueError("Historical values must be distinct from time and identity columns")
            reserved = {"proportion", "_src", "_tgt", "_target_total", "_source_total"}
            if reserved.intersection([*level_list, id_column, time_column, *val_cols]):
                raise ValueError("top_down column names conflict with reserved historical-proportion columns")
            _check_reconciliation_frame(
                history, [*level_list, id_column], time_column, [history_value_column],
            )
            mapping_data = history
            _check_reconciliation_frame(df, [source_level], time_column, val_cols)
        else:
            if history is not None:
                raise ValueError("actuals_file is supported only for top_down")
            _check_reconciliation_frame(df, [*level_list, id_column], time_column, val_cols)
            mapping_data = df

        _check_hierarchy_mapping(mapping_data, level_list, id_column)
        leaf_ids = mapping_data[leaf].unique(maintain_order=True)
        mapping = mapping_data.select(level_list).unique().with_columns(
            pl.col(level).cast(pl.String) for level in level_list
        )
        source = df.select([source_level, time_column, *val_cols]).with_columns(
            pl.col(source_level).cast(pl.String),
        )
        _check_complete_panel(source, source_level, time_column, mapping[source_level])
        actuals = None
        if top_down:
            _check_complete_panel(history, leaf, time_column, leaf_ids)
            last_history = history[time_column].cast(pl.Datetime("us")).cast(pl.Int64).max()
            first_forecast = df[time_column].cast(pl.Datetime("us")).cast(pl.Int64).min()
            if last_history >= first_forecast:
                raise ValueError("All historical observations must be strictly before the forecast window")
            volumes = history.group_by(leaf).agg(
                pl.col(history_value_column).cast(pl.Float64).sum(),
            )[history_value_column]
            if not volumes.is_finite().all() or (volumes <= 0).any():
                raise ValueError("Every historical leaf group must have finite positive total volume")
            # The domain uses its leaf-ID alias and the first requested value name.
            actuals = history.select(
                pl.col(leaf).cast(pl.String).alias(id_column),
                pl.col(time_column),
                pl.col(history_value_column).alias(val_cols[0]),
            )
        elif requested_method != HierarchyReconciliationMethod.BOTTOM_UP:
            keys: set[str] = set()
            for level in level_list:
                level_keys = set(mapping[level].to_list())
                if keys.intersection(level_keys):
                    raise ValueError("Linear methods require unambiguous node keys across hierarchy levels")
                keys.update(level_keys)
            if {"node_key", "node_level"}.intersection(mapping[leaf].to_list()):
                raise ValueError("Linear leaf keys cannot use the reserved summing-matrix metadata names")

        tree = HierarchyTree(
            HierarchyConfig(name="hierarchy", levels=level_list, id_column=id_column), mapping,
        )
    except (ValueError, pl.exceptions.PolarsError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid_hierarchy_input: {exc}") from exc

    reconciler = Reconciler(
        trees={"hierarchy": tree}, config=ReconciliationConfig(method=requested_method.value),
    )
    try:
        reconciled = reconciler.reconcile(
            forecasts={source_level: source},
            actuals=actuals,
            value_columns=val_cols,
            time_column=time_column,
        ).select([leaf, time_column, *val_cols])
        _check_reconciliation_frame(reconciled, [leaf], time_column, val_cols)
        _check_complete_panel(reconciled, leaf, time_column, mapping[leaf])
        if set(reconciled[time_column].to_list()) != set(source[time_column].to_list()):
            raise ValueError("Reconciled periods differ from the requested forecast window")
        coherence = _check_reconciliation_coherence(
            tree, reconciled, mapping, source, val_cols, time_column, top_down,
        )
        reconciled = reconciled.with_columns(pl.col(leaf).replace_strict(
            old=leaf_ids.cast(pl.String), new=leaf_ids, return_dtype=leaf_ids.dtype,
        ))
        validation = _run_reconciliation_validation(
            reconciled, validate, id_column=leaf, time_column=time_column,
        )
    except (ValueError, RuntimeError, KeyError, TypeError, OverflowError, pl.exceptions.PolarsError):
        logger.exception("Hierarchy reconciliation or full-frame output validation failed")
        raise HTTPException(
            status_code=500, detail="reconciliation_failed: unable to produce valid complete hierarchy output",
        ) from None

    warnings = [] if top_down else [
        "Parent totals are derived from leaves; no independent upper-level forecasts were supplied.",
    ]
    effective_method = HierarchyEffectiveMethod(requested_method.value) if requested_method not in {
        HierarchyReconciliationMethod.WLS, HierarchyReconciliationMethod.MINT,
    } else HierarchyEffectiveMethod.WLS_STRUCTURAL
    if requested_method == HierarchyReconciliationMethod.WLS:
        warnings.append("Residuals were not supplied; WLS used structural weights.")
    elif requested_method == HierarchyReconciliationMethod.MINT:
        warnings.append("Residuals were not supplied; MinT used structural WLS, not residual-covariance MinT.")

    return HierarchyReconcileResponse(
        method=requested_method,
        requested_method=requested_method,
        effective_method=effective_method,
        source_level=source_level,
        output_level=leaf,
        input_rows=df.height,
        output_rows=reconciled.height,
        before_total=round(float(source[val_cols[0]].cast(pl.Float64).sum()), 2),
        after_total=round(float(reconciled[val_cols[0]].cast(pl.Float64).sum()), 2),
        rows=reconciled.height,
        reconciled_preview=reconciled.head(200).to_dicts(),
        warnings=warnings,
        coherence=coherence,
        validation=validation,
    )


async def _read_reconciliation_upload(file: UploadFile, time_column: str) -> pl.DataFrame:
    filename = file.filename or ""
    if "/" in filename or "\\" in filename or not filename.lower().endswith((".csv", ".parquet")):
        raise HTTPException(
            status_code=400, detail="invalid_hierarchy_file: upload a CSV or Parquet file with a plain filename",
        )
    content = await validate_upload_size(file)
    try:
        if filename.lower().endswith(".parquet"):
            return pl.read_parquet(io.BytesIO(content))
        frame = pl.read_csv(io.BytesIO(content), try_parse_dates=True)
        if time_column in frame.columns:
            # CSV inference already truncates fractional seconds beyond microseconds.
            raw_times = pl.read_csv(
                io.BytesIO(content), columns=[time_column],
                schema_overrides={time_column: pl.String},
            )[time_column]
            if raw_times.str.contains(r":\d{2}\.\d{6}\d*[1-9]").any():
                raise HTTPException(status_code=400, detail=f"invalid_hierarchy_input: {_TIMESTAMP_PRECISION_ERROR}")
        return frame
    except (ValueError, UnicodeDecodeError, OSError, pl.exceptions.PolarsError):
        raise HTTPException(
            status_code=400, detail="invalid_hierarchy_file: unable to parse the CSV or Parquet upload",
        ) from None


def _column_names(value: str, label: str) -> list[str]:
    columns = [column.strip() for column in value.split(",")]
    if any(not column for column in columns) or len(columns) != len(set(columns)):
        raise ValueError(f"{label} must contain distinct, nonempty column names")
    return columns


def _check_reconciliation_values(frame: pl.DataFrame, columns: list[str]) -> None:
    quantiles = []
    for column in columns:
        values = frame[column]
        if not values.dtype.is_numeric() or values.null_count() or not values.is_finite().all():
            raise ValueError(f"Requested values in {column!r} must be finite numeric observations")
        if not math.isfinite(float(values.cast(pl.Float64).sum())):
            raise ValueError(f"Requested values in {column!r} must have a finite total")
        percentile = re.fullmatch(r"forecast_p(\d+)", column)
        if percentile:
            percent = int(percentile[1])
            if not 0 <= percent <= 100 or column != f"forecast_p{percent}":
                raise ValueError("Quantile columns must use forecast_p0 through forecast_p100 without leading zeros")
            quantiles.append(percent / 100)
    validate_quantile_forecasts(frame, quantiles, "Hierarchy reconciliation")


def _check_reconciliation_frame(
    frame: pl.DataFrame, identities: list[str], time_column: str, values: list[str],
) -> None:
    required = set([*identities, time_column, *values])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError("Hierarchy inputs must contain observations")
    for column in set(identities):
        ids = frame[column]
        if ids.null_count() or (ids.cast(pl.String).str.strip_chars() == "").any():
            raise ValueError(f"Identity column {column!r} must contain nonempty, nonnull keys")
        if ids.dtype.is_numeric() and not ids.is_finite().all():
            raise ValueError(f"Identity column {column!r} must contain finite keys")
    times = frame[time_column]
    if times.null_count() or times.dtype.base_type() not in {pl.Date, pl.Datetime}:
        raise ValueError(f"Time column {time_column!r} must contain nonnull dates or timestamps")
    if times.dtype.base_type() == pl.Datetime and times.dtype.time_unit == "ns":
        if (times.cast(pl.Int64) % 1000 != 0).any():
            raise ValueError(_TIMESTAMP_PRECISION_ERROR)
    _check_reconciliation_values(frame, values)


def _check_hierarchy_mapping(frame: pl.DataFrame, levels: list[str], id_column: str) -> None:
    pairs = list(zip(levels[:-1], levels[1:]))
    if id_column != levels[-1]:
        pairs.extend([(levels[-1], id_column), (id_column, levels[-1])])
    for parent, child in pairs:
        counts = frame.group_by(child).agg(pl.col(parent).n_unique())[parent]
        if (counts != 1).any():
            raise ValueError(f"Each {child!r} node must map to exactly one {parent!r} node")


def _check_complete_panel(
    frame: pl.DataFrame, node_column: str, time_column: str, expected_nodes: pl.Series,
) -> None:
    if frame.select([node_column, time_column]).is_duplicated().any():
        raise ValueError("Duplicate node/time observations are not supported")
    expected = set(expected_nodes.cast(pl.String).to_list())
    if set(frame[node_column].cast(pl.String).to_list()) != expected:
        raise ValueError("The forecast panel contains unknown or missing hierarchy nodes")
    if frame.height != len(expected) * frame[time_column].n_unique():
        raise ValueError("Incomplete node/time coverage; supply every expected node in every period")


def _check_reconciliation_coherence(
    tree: HierarchyTree,
    output: pl.DataFrame,
    mapping: pl.DataFrame,
    source: pl.DataFrame,
    values: list[str],
    time_column: str,
    top_down: bool,
) -> HierarchyCoherenceResult:
    from ...hierarchy.aggregator import HierarchyAggregator

    aggregator = HierarchyAggregator(tree)
    leaf = tree.leaf_level
    output_with_id = output.with_columns(pl.col(leaf).alias(tree.config.id_column))
    levels = tree.levels[:1] if top_down else tree.levels[:-1]
    max_error = 0.0
    checked_groups = 0
    for level in levels:
        keys = [level, time_column]
        observed = aggregator.aggregate_to(
            output_with_id, level, values, time_column,
        ).sort(keys)
        expected = source.sort(keys) if top_down else (
            output.join(mapping.select([leaf, level]), on=leaf, how="left")
            .group_by(keys).agg(pl.col(column).sum() for column in values).sort(keys)
        )
        if not observed.select(keys).equals(expected.select(keys)):
            raise ValueError("Reconciled parent/time keys differ from expected keys")
        _check_reconciliation_values(observed, values)
        _check_reconciliation_values(expected, values)
        for column in values:
            actual = observed[column].cast(pl.Float64)
            target = expected[column].cast(pl.Float64)
            error = (actual - target).abs()
            if (error > 1e-8 + 1e-6 * target.abs()).any():
                raise ValueError("Reconciled values do not conserve the declared parent totals")
            max_error = max(max_error, float(error.max()))
        checked_groups += observed.height
    return HierarchyCoherenceResult(
        scope="root_forecast_conservation" if top_down else "leaf_derived_parent_totals",
        passed=True,
        checked_rows=output.height,
        checked_periods=output[time_column].n_unique(),
        checked_groups=checked_groups,
        value_columns=values,
        max_abs_error=max_error,
    )


def _run_reconciliation_validation(
    reconciled: pl.DataFrame, enabled: bool, *,
    id_column: str = "series_id", time_column: str = "week",
) -> Optional[dict]:
    """Run optional post-reconciliation validation."""
    if not enabled:
        return None
    from ...config.schema import PostValidationConfig
    from ...pipeline.validation_step import run_post_validation

    config = PostValidationConfig(simpsons_paradox_checks=False)
    return run_post_validation(
        reconciled, config, lob="reconciliation", id_col=id_column, time_col=time_column,
    )
