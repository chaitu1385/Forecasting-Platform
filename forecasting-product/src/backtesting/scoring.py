"""Pure fold-evaluation contract used by serial, spawned and Spark workers."""

import math
from datetime import date, time
from typing import List, Sequence

import polars as pl

from ..config.schema import PlatformConfig, get_frequency_profile
from ..forecasting.base import BaseForecaster
from ..metrics.definitions import _naive_seasonal_mae, metric_expression
from ..utils.calendar import PeriodCalendar


def require_fold_local_preprocessing(config: PlatformConfig) -> None:
    """Fail before whole-history learned transforms can leak into validation."""
    unsafe = []
    if config.data_quality.cleansing.enabled:
        unsafe.append("data_quality.cleansing")
    if config.data_quality.structural_breaks.enabled:
        unsafe.append("data_quality.structural_breaks")
    if config.forecast.external_regressors.enabled and config.forecast.external_regressors.screen.enabled:
        unsafe.append("forecast.external_regressors.screen")
    if unsafe:
        raise ValueError(
            "Backtesting requires fold-local fit of learned preprocessing. "
            f"Unsupported enabled settings: {', '.join(unsafe)}. "
            "Disable these settings for backtesting; production forecasting is unchanged."
        )


def _validated_frame(
    df: pl.DataFrame, id_col: str, time_col: str, value_cols: Sequence[str], label: str
) -> pl.DataFrame:
    required = [id_col, time_col] + list(value_cols)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} coverage missing columns: {sorted(missing)}")
    if df.is_empty():
        raise ValueError(f"Insufficient data: empty {label}")
    if any(df[col].null_count() for col in required):
        raise ValueError(f"{label} contains null keys or values")
    if isinstance(df[time_col].dtype, pl.Datetime):
        if df.filter(pl.col(time_col).dt.time() != time(0)).height:
            raise ValueError(f"{label} requires date labels or midnight datetimes")
        df = df.with_columns(pl.col(time_col).cast(pl.Date))
    if df[time_col].dtype != pl.Date:
        raise ValueError(f"{label} requires date-typed observation keys")
    if df.select([id_col, time_col]).is_duplicated().any():
        raise ValueError(f"{label} contains duplicate series/date keys")
    for column in value_cols:
        values = df[column]
        if not values.dtype.is_numeric() or not values.is_finite().all():
            raise ValueError(f"{label} requires finite numeric {column}")
    return df


def require_exact_coverage(
    expected: pl.DataFrame,
    observed: pl.DataFrame,
    id_col: str,
    time_col: str,
    value_cols: Sequence[str],
    label: str,
) -> pl.DataFrame:
    """Validate keys before any join; row ordering never determines alignment."""
    observed = _validated_frame(observed, id_col, time_col, value_cols, label)
    keys = [id_col, time_col]
    missing = expected.select(keys).join(observed.select(keys), on=keys, how="anti").height
    extra = observed.select(keys).join(expected.select(keys), on=keys, how="anti").height
    if missing or extra:
        raise ValueError(f"{label} coverage mismatch: {missing} missing and {extra} unexpected series/date keys")
    return observed


def evaluate_model_fold(
    train: pl.DataFrame,
    val: pl.DataFrame,
    forecaster: BaseForecaster,
    *,
    fold_index: int,
    run_id: str,
    run_date: date,
    lob: str,
    target_col: str,
    time_col: str,
    id_col: str,
    metrics_list: List[str],
    val_periods: int,
    gap_periods: int,
    frequency: str,
    quantiles: List[float],
    calibration_enabled: bool,
) -> pl.DataFrame:
    """Fit on training data and score every requested validation key or fail.

    ``forecast_step`` is the actual lead from training end. With a two-period
    gap, the first scored observation is step 3, not step 1.
    """
    train = _validated_frame(train, id_col, time_col, [target_col], "Training data")
    calendar = PeriodCalendar.from_dates(train[time_col].unique().to_list(), frequency)
    horizon = gap_periods + val_periods
    if val_periods < 1 or gap_periods < 0:
        raise ValueError("Validation periods must be positive and gap periods nonnegative")
    expected_rows = []
    scales = {}
    season_length = get_frequency_profile(frequency)["season_length"]
    for group in train.partition_by(id_col):
        group = group.sort(time_col)
        sid = group[id_col][0]
        dates = group[time_col].to_list()
        if dates != calendar.date_range(dates[0], dates[-1]):
            raise ValueError(f"Training calendar for series {sid!r} has missing periods")
        scale = _naive_seasonal_mae(group[target_col], season_length)
        scales[sid] = scale if math.isfinite(scale) and scale > 0 else None
        expected_rows.extend(
            {id_col: sid, time_col: label, "forecast_step": step}
            for step, label in enumerate(calendar.future_dates(dates[-1], horizon), start=1)
        )
    expected = pl.DataFrame(expected_rows)
    expected_val = expected.filter(pl.col("forecast_step") > gap_periods)
    val = require_exact_coverage(expected_val, val, id_col, time_col, [target_col], "Validation actuals")

    forecaster.fit(train, target_col=target_col, time_col=time_col, id_col=id_col)
    predictions = require_exact_coverage(
        expected, forecaster.predict(horizon, id_col=id_col, time_col=time_col),
        id_col, time_col, ["forecast"], "Predictions",
    ).select(id_col, time_col, "forecast")
    q_cols = []
    if quantiles and calibration_enabled:
        q_cols = [f"forecast_p{int(round(q * 100))}" for q in quantiles]
        qdf = require_exact_coverage(
            expected,
            forecaster.predict_quantiles(horizon, quantiles, id_col=id_col, time_col=time_col),
            id_col, time_col, q_cols, "Quantile predictions",
        )
        predictions = predictions.join(qdf.select([id_col, time_col] + q_cols), on=[id_col, time_col])

    # Exact coverage and uniqueness were checked above. Polars 0.20 cannot use
    # join(validate="1:1") on composite keys; correctness must not depend on it.
    merged = (
        expected_val.join(val.select(id_col, time_col, target_col), on=[id_col, time_col])
        .join(predictions, on=[id_col, time_col])
        .sort([id_col, time_col])
    )
    records = merged.select(
        pl.lit(run_id).alias("run_id"), pl.lit("backtest").alias("run_type"),
        pl.lit(run_date).alias("run_date"), pl.lit(lob).alias("lob"),
        pl.lit(forecaster.name).alias("model_id"), pl.lit(fold_index).alias("fold"),
        pl.lit("series").alias("grain_level"), pl.col(id_col).alias("series_id"),
        pl.lit("").alias("channel"), pl.col(time_col).alias("target_week"),
        pl.col("forecast_step"), pl.col(target_col).cast(pl.Float64).alias("actual"),
        pl.col("forecast").cast(pl.Float64), *[pl.col(qc) for qc in q_cols],
    )
    scale_frame = pl.DataFrame({id_col: list(scales), "mase_scale": list(scales.values())},
                              schema_overrides={"mase_scale": pl.Float64})
    records = records.join(scale_frame.rename({id_col: "series_id"}), on="series_id", validate="m:1")
    names = list(dict.fromkeys(name for name in metrics_list if name != "macro_wmape"))
    return records.with_columns([metric_expression(name, aggregate=False) for name in names])
