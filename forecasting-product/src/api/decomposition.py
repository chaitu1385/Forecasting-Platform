"""Admission, full-result checks and provenance for the decomposition API.

The domain's moving-average arithmetic is unchanged. Only finite, losslessly
representable panels with defined seasonal positions reach that computation.
"""

from __future__ import annotations

import io
import math

import polars as pl

from ..analytics.explainer import ForecastExplainer
from ..pipeline.readers import ArtifactRead
from .schemas import DecompositionComponent, DecompositionSummaryResponse

_MAX_EXACT_INTEGER = 2**53 - 1
_COMPONENTS = ("trend", "seasonal", "residual")
_KEYS = ("series_id", "week", "is_forecast")
_ISO_DATETIME = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"


class DecompositionInputError(ValueError):
    """An unsupported or incomplete panel; messages contain no uploaded values."""


class UnsupportedDecompositionWindow(DecompositionInputError):
    """A recent trend fit would sample an entirely undefined last-three anchor."""


class DecompositionResultError(RuntimeError):
    """The complete domain result violates the admitted panel contract."""


def _time_column(values: pl.Series) -> pl.Series:
    if values.null_count():
        raise DecompositionInputError("Time keys must not be null")
    dtype = values.dtype
    if dtype == pl.String:
        if values.str.contains(r"^\d{4}-\d{2}-\d{2}$").all():
            values = values.str.to_date("%Y-%m-%d", strict=True)
        else:
            if not values.str.contains(_ISO_DATETIME).all():
                raise DecompositionInputError("Use uniform ISO dates or ISO datetimes for time keys")
            fractions = values.str.extract(r"\.(\d+)", 1).fill_null("")
            if fractions.str.slice(6).str.contains("[1-9]").any():
                raise DecompositionInputError("Submicrosecond time keys are unsupported; no rounding is performed")
            aware = values.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$")
            if aware.any() and not aware.all():
                raise DecompositionInputError("Do not mix timezone-aware and naive time keys")
            values = values.str.replace(r"(\.\d{6})\d+", "$1").str.to_datetime(
                time_unit="us", time_zone="UTC" if aware.all() else None, strict=True,
            )
        dtype = values.dtype
    if isinstance(dtype, pl.Datetime):
        if dtype.time_unit == "ns" and (values.cast(pl.Int64) % 1000 != 0).any():
            raise DecompositionInputError("Submicrosecond time keys are unsupported; no rounding is performed")
        if not values.dt.year().is_between(1, 9999).all():
            raise DecompositionInputError("Time keys must be representable as Python datetimes")
        values = values.cast(pl.Datetime("us", dtype.time_zone))
        if dtype.time_zone is not None:
            values = values.dt.convert_time_zone("UTC")
    elif dtype == pl.Date:
        if not values.dt.year().is_between(1, 9999).all():
            raise DecompositionInputError("Time keys must be representable as Python dates")
    else:
        raise DecompositionInputError("Time keys must be dates or datetimes")
    return values


def read_decomposition_upload(content: bytes, filename: str, target_column: str) -> pl.DataFrame:
    """Preserve raw CSV identifiers and fractional time text before admission."""
    try:
        if filename.lower().endswith(".parquet"):
            return pl.read_parquet(io.BytesIO(content))
        frame = pl.read_csv(io.BytesIO(content), infer_schema=False, try_parse_dates=False)
        if target_column not in frame.columns:
            raise DecompositionInputError("Uploaded history is missing the requested target column")
        return frame.with_columns(pl.col(target_column).cast(pl.Float64, strict=True))
    except (pl.exceptions.PolarsError, UnicodeDecodeError, OSError) as exc:
        raise DecompositionInputError("Cannot read the history upload as CSV or Parquet") from exc


def _panel(artifact: ArtifactRead, value_column: str) -> pl.DataFrame:
    columns = (artifact.id_column, artifact.time_column, value_column)
    if len(set(columns)) != 3 or not set(columns).issubset(artifact.frame.columns):
        raise DecompositionInputError("Each panel needs distinct identifier, time and value columns")
    if artifact.frame.is_empty():
        raise DecompositionInputError("History and forecast panels must both contain observations")
    frame = artifact.frame.select(
        pl.col(columns[0]).alias("series_id"),
        pl.col(columns[1]).alias("week"),
        pl.col(columns[2]).alias("value"),
    )
    ids = frame["series_id"]
    if ids.null_count():
        raise DecompositionInputError("Series identifiers must not be null")
    if ids.dtype in (pl.String, pl.Categorical) or isinstance(ids.dtype, pl.Enum):
        ids = ids.cast(pl.String)
        if (ids.str.strip_chars().str.len_chars() == 0).any():
            raise DecompositionInputError("Series identifiers must not be blank")
    elif ids.dtype.is_integer():
        if ids.min() < -_MAX_EXACT_INTEGER or ids.max() > _MAX_EXACT_INTEGER:
            raise DecompositionInputError("Integer series identifiers must be exactly representable in JSON")
        ids = ids.cast(pl.Int64)
    else:
        raise DecompositionInputError("Series identifiers must be strings or exact JSON integers")
    values = frame["value"]
    if not (values.dtype.is_integer() or values.dtype.is_float()):
        raise DecompositionInputError("Observed and forecast values must be numeric")
    if values.null_count():
        raise DecompositionInputError("Observed and forecast values must be finite and non-null")
    if values.dtype.is_integer() and (values.min() < -_MAX_EXACT_INTEGER or values.max() > _MAX_EXACT_INTEGER):
        raise DecompositionInputError("Integer observations must be exactly representable as numbers")
    values = values.cast(pl.Float64)
    if values.null_count() or not values.is_finite().all():
        raise DecompositionInputError("Observed and forecast values must be finite and non-null")
    try:
        times = _time_column(frame["week"])
    except pl.exceptions.PolarsError as exc:
        raise DecompositionInputError("Invalid or unsupported time keys") from exc
    frame = frame.with_columns(ids, times, values)
    if frame.select("series_id", "week").is_duplicated().any():
        raise DecompositionInputError("Duplicate series/time keys are unsupported")
    return frame


def _check_result(result: pl.DataFrame, history: pl.DataFrame, forecast: pl.DataFrame, half: int) -> None:
    historical = history.sort("series_id", "week").with_columns(
        pl.int_range(pl.len()).over("series_id").alias("_position"),
        pl.len().over("series_id").alias("_count"),
    ).with_columns(
        ((pl.col("_position") < half) | (pl.col("_position") >= pl.col("_count") - half)).alias("_undefined"),
        pl.lit(False).alias("is_forecast"),
    ).drop("_position", "_count")
    expected = pl.concat([
        historical,
        forecast.with_columns(pl.lit(False).alias("_undefined"), pl.lit(True).alias("is_forecast")),
    ])
    required = {*_KEYS, "value", *_COMPONENTS}
    if set(result.columns) != required or result.height != expected.height:
        raise DecompositionResultError("Incomplete decomposition result")
    if result["is_forecast"].dtype != pl.Boolean or result["is_forecast"].null_count():
        raise DecompositionResultError("Invalid decomposition population flags")
    if result.select(*_KEYS).is_duplicated().any():
        raise DecompositionResultError("Duplicate decomposition keys")
    if result.schema["series_id"] != expected.schema["series_id"] or result.schema["week"] != expected.schema["week"]:
        raise DecompositionResultError("Decomposition changed key types or precision")
    checked = result.join(expected, on=list(_KEYS), how="inner", suffix="_expected")
    if checked.height != expected.height or not (checked["value"] == checked["value_expected"]).all():
        raise DecompositionResultError("Decomposition changed input keys or values")
    for name in ("value", *_COMPONENTS):
        values = checked[name]
        if not (values.dtype.is_float() or values.dtype.is_integer()) or not values.drop_nulls().is_finite().all():
            raise DecompositionResultError("Nonfinite decomposition values")
        undefined = checked["_undefined"] if name in ("trend", "residual") else pl.Series([False] * checked.height)
        if not (values.is_null() == undefined).all():
            raise DecompositionResultError("Unexpected undefined decomposition values")
    defined = checked.filter(~pl.col("_undefined"))
    reconstructed = defined["trend"] + defined["seasonal"] + defined["residual"]
    if not reconstructed.is_finite().all() or (
        (reconstructed - defined["value"]).abs() > 1e-8 + 1e-10 * defined["value"].abs()
    ).any():
        raise DecompositionResultError("Components do not reconstruct the supplied observations")


def decompose_artifacts(
    lob: str, selected: ArtifactRead, history: ArtifactRead, season_length: int,
) -> DecompositionSummaryResponse:
    """Describe only forecast-series history; validate all rows before previewing."""
    if season_length < 1:
        raise DecompositionInputError("season_length must be a positive integer")
    forecast = _panel(selected, "forecast")
    historical = _panel(history, history.target_column)
    if forecast.schema["series_id"] != historical.schema["series_id"]:
        raise DecompositionInputError("History and forecast identifier types must match")
    if forecast.schema["week"] != historical.schema["week"]:
        raise DecompositionInputError("History and forecast time types and timezone awareness must match")
    forecast_ids = forecast.select("series_id").unique()
    if forecast_ids.join(historical.select("series_id").unique(), on="series_id", how="anti").height:
        raise DecompositionInputError("Every forecast series requires historical observations")
    scoped = historical.join(forecast_ids, on="series_id", how="semi")
    half = season_length // 2
    minimum = 2 * half + max(season_length, 2)
    lengths = scoped.group_by("series_id").len()["len"]
    if lengths.min() < minimum:
        raise DecompositionInputError(
            f"Each forecast series needs at least {minimum} history rows to define every seasonal position and trend",
        )
    explainer = ForecastExplainer(season_length=season_length)
    for count in lengths:
        recent_window = min(explainer.trend_window, count)
        # Defined centered trends occupy [half, count - half).
        trend_stop = count - half
        recent_defined = max(0, trend_stop - max(half, count - recent_window))
        anchor_defined = max(0, trend_stop - max(half, count - 3))
        if recent_window >= 2 and recent_defined >= 2 and anchor_defined == 0:
            raise UnsupportedDecompositionWindow(
                "The requested seasonal period leaves no defined last-three trend anchor "
                "after the recent trend fit. This decomposition method does not support "
                "that window combination; no substitute trend is computed.",
            )
    bounds = scoped.group_by("series_id").agg(pl.col("week").max().alias("last_history")).join(
        forecast.group_by("series_id").agg(pl.col("week").min().alias("first_forecast")), on="series_id",
    )
    if (bounds["first_forecast"] <= bounds["last_history"]).any():
        raise DecompositionInputError("Forecast times must follow the complete history for each series")
    result = explainer.decompose(
        history=scoped, forecast=forecast.rename({"value": "forecast"}), target_col="value",
    )
    _check_result(result, scoped, forecast, half)
    components = []
    historical_result = result.filter(~pl.col("is_forecast"))
    for name in _COMPONENTS:
        values = historical_result[name].drop_nulls()
        mean_abs = values.abs().mean()
        std = values.std(ddof=1) if len(values) >= 2 else None
        if not len(values) or any(value is not None and not math.isfinite(value) for value in (mean_abs, std)):
            raise DecompositionResultError("Undefined or nonfinite component statistics")
        components.append(DecompositionComponent(
            component=name, mean_abs_value=mean_abs, std=std,
            defined_count=len(values), undefined_count=scoped.height - len(values),
        ))
    model = selected.manifest.champion_model_id if selected.manifest else None
    model_source = "manifest"
    if not isinstance(model, str) or not model.strip():
        model, model_source = None, "unknown"
        if "model" in selected.frame.columns and selected.frame["model"].dtype == pl.String:
            models = selected.frame["model"].unique().to_list()
            if len(models) == 1 and isinstance(models[0], str) and models[0].strip():
                model, model_source = models[0], "forecast_frame"
    preview = result.head(50)
    return DecompositionSummaryResponse(
        lob=lob, run_id=selected.run_id, artifact_source=selected.source,
        forecast_model=model, forecast_model_source=model_source,
        history_source=history.source, history_stage=history.input_stage,
        history_run_id=history.run_id, season_length=season_length, trend_window=2 * half + 1,
        series_count=forecast_ids.height, history_rows=scoped.height,
        excluded_history_rows=historical.height - scoped.height, forecast_rows=forecast.height,
        checked_rows=result.height, preview_rows=preview.height, preview_truncated=result.height > preview.height,
        components=components, decomposition_preview=preview.to_dicts(),
    )
