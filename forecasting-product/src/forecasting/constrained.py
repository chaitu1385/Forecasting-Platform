"""
Constrained demand estimator.

Wraps any base forecaster and applies hard business constraints —
non-negativity, per-series capacity limits, and aggregate budgets —
to point forecasts. Quantiles support element-wise bounds only; aggregate
probabilistic budgets are explicitly unsupported.

Example
-------
>>> from src.forecasting.constrained import ConstrainedDemandEstimator
>>> from src.config.schema import ConstraintConfig
>>> cde = ConstrainedDemandEstimator(
...     base_forecaster=naive,
...     constraints=ConstraintConfig(enabled=True, max_capacity=500),
... )
>>> cde.fit(train_df)
>>> forecast = cde.predict(horizon=13)
"""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Dict, List

import polars as pl

from ..config.schema import ConstraintConfig
from ..evaluation.calibration import validate_quantile_forecasts
from .base import BaseForecaster
from .registry import registry


def _nonnegative_limit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"invalid_constraint_limit: {name} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"invalid_constraint_limit: {name} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid_constraint_limit: {name} must be a finite nonnegative number")
    return number


def _constraint_limits(
    constraints: ConstraintConfig,
) -> tuple[float, float | None, float | None]:
    floor = _nonnegative_limit(constraints.min_demand, "min_demand")
    cap = (
        _nonnegative_limit(constraints.max_capacity, "max_capacity")
        if constraints.max_capacity is not None else None
    )
    budget = (
        _nonnegative_limit(constraints.aggregate_max, "aggregate_max")
        if constraints.aggregate_max is not None else None
    )
    if cap is not None and cap < floor:
        raise ValueError("infeasible_constraints: max_capacity must be at least min_demand")
    if not isinstance(constraints.proportional_redistribution, bool):
        raise ValueError("invalid_constraint_limit: proportional_redistribution must be boolean")
    if constraints.capacity_column is not None and (
        not isinstance(constraints.capacity_column, str)
        or not constraints.capacity_column.strip()
    ):
        raise ValueError("invalid_capacity_data: capacity_column must be a nonempty column name")
    return floor, cap, budget


def _finite_values(frame: pl.DataFrame, column: str) -> list[float]:
    values = frame[column]
    if not values.dtype.is_numeric() or values.null_count():
        raise ValueError(f"invalid_constraint_frame: {column} must contain finite numeric values")
    values = values.cast(pl.Float64)
    if not values.is_finite().all():
        raise ValueError(f"invalid_constraint_frame: {column} must contain finite numeric values")
    return values.to_list()


def _validate_key_values(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column]
        if values.null_count() or values.dtype.is_nested() or values.dtype in (pl.Object, pl.Binary):
            raise ValueError("invalid_constraint_frame: identity and time keys must be non-null supported scalars")
        if values.dtype.is_numeric() and not values.cast(pl.Float64).is_finite().all():
            raise ValueError("invalid_constraint_frame: identity and time keys must be finite")
        if values.dtype == pl.String and (values.str.strip_chars() == "").any():
            raise ValueError("invalid_constraint_frame: identity and time keys must be nonempty")


def _validate_keys(frame: pl.DataFrame, id_col: str, time_col: str) -> None:
    _validate_key_values(frame, (id_col, time_col))
    if frame.select(id_col, time_col).is_duplicated().any():
        raise ValueError("invalid_constraint_frame: duplicate identity/time keys")


def _quantile_levels(frame: pl.DataFrame, id_col: str, time_col: str) -> list[float]:
    levels = []
    for column in frame.columns:
        if column in (id_col, time_col) or not column.startswith("forecast_p"):
            continue
        suffix = column.removeprefix("forecast_p")
        if not suffix.isascii() or not suffix.isdigit():
            raise ValueError("invalid_constraint_frame: quantile columns must use forecast_p0 through forecast_p100")
        percentile = int(suffix)
        if not 0 <= percentile <= 100 or column != f"forecast_p{percentile}":
            raise ValueError("invalid_constraint_frame: quantile columns must use forecast_p0 through forecast_p100")
        levels.append(percentile / 100)
    return sorted(levels)


def _validate_input_quantiles(frame: pl.DataFrame, quantiles: list[float]) -> None:
    try:
        validate_quantile_forecasts(frame, quantiles, "Constraint input")
    except (TypeError, pl.exceptions.InvalidOperationError) as exc:
        raise ValueError(
            "invalid_constraint_frame: quantile columns must use supported integer/floating-point "
            "types and quantile levels must be finite numbers"
        ) from exc


def _period_rows(frame: pl.DataFrame, time_col: str) -> list[list[int]]:
    # Group exact Polars keys, not Python datetimes which truncate nanoseconds.
    return (
        frame.select(
            pl.col(time_col).alias("period"),
            pl.int_range(0, pl.len()).alias("rows"),
        )
        .group_by("period", maintain_order=True)
        .agg("rows")["rows"]
        .to_list()
    )


def _finite_sum(values: list[float]) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("invalid_constraint_frame: aggregate totals must be finite") from exc
    if not math.isfinite(total):
        raise ValueError("invalid_constraint_frame: aggregate totals must be finite")
    return total


def apply_point_constraints(
    frame: pl.DataFrame,
    constraints: ConstraintConfig,
    id_col: str = "series_id",
    time_col: str = "week",
    value_col: str = "forecast",
    capacity_map: Mapping[object, float] | None = None,
) -> pl.DataFrame:
    """Apply validated bounds and per-period point budgets without changing keys.

    Proportional allocation preserves each floor and scales only headroom.
    Largest-first removes headroom in descending clipped-value order, breaking
    ties by ascending string series ID. Original row/column order and all
    nonvalue columns are preserved. Quantile-bearing frames support the same
    monotone element-wise clipping, but cannot be combined with a budget.

    Parameters
    ----------
    frame : pl.DataFrame
        Nonempty forecast panel with unique, non-null identity/time keys.
    constraints : ConstraintConfig
        Nonnegative finite bounds and an optional feasible per-period budget.
        Disabled constraints return the input unchanged.
    id_col, time_col, value_col : str
        Distinct identity, period and point-value columns.
    capacity_map : Mapping, optional
        Complete per-series capacities. Required when capacity_column is set.

    Returns
    -------
    pl.DataFrame
        Constrained finite values in the original population and order.

    Raises
    ------
    ValueError
        Invalid inputs, infeasible bounds, or unsupported aggregate quantiles.
    ArithmeticError
        Computed values fail the full-population bounds/budget checks.
    """
    if not constraints.enabled:
        return frame
    floor, cap, budget = _constraint_limits(constraints)
    names = (id_col, time_col, value_col)
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != 3:
        raise ValueError("invalid_constraint_frame: identity, time and value column names must be distinct and nonempty")
    if any(name not in frame.columns for name in names):
        raise ValueError("invalid_constraint_frame: required identity, time or value column is missing")
    if frame.is_empty():
        raise ValueError("invalid_constraint_frame: forecast input must not be empty")
    _validate_keys(frame, id_col, time_col)
    quantiles = _quantile_levels(frame, id_col, time_col)
    if budget is not None and quantiles:
        raise ValueError(
            "unsupported_aggregate_quantiles: aggregate budgets support point forecasts only; "
            "remove aggregate_max or the quantile columns"
        )
    _validate_input_quantiles(frame, quantiles)
    columns = list(dict.fromkeys([value_col, *[f"forecast_p{int(round(q * 100))}" for q in quantiles]]))
    values = {column: _finite_values(frame, column) for column in columns}

    identities = frame[id_col].to_list()
    if constraints.capacity_column is not None and capacity_map is None:
        raise ValueError("invalid_capacity_data: configured capacity_column requires a complete capacity map")
    upper = [cap] * frame.height
    if capacity_map is not None:
        if not isinstance(capacity_map, Mapping):
            raise ValueError("invalid_capacity_data: capacity_map must be a mapping")
        capacities = {
            identity: _nonnegative_limit(value, "per-series capacity")
            for identity, value in capacity_map.items()
        }
        if any(value < floor for value in capacities.values()):
            raise ValueError("infeasible_constraints: per-series capacity must be at least min_demand")
        if any(identity not in capacities for identity in identities):
            raise ValueError("invalid_capacity_data: capacity map must cover every output series")
        upper = [
            min(cap, capacities[identity]) if cap is not None else capacities[identity]
            for identity in identities
        ]

    period_rows = _period_rows(frame, time_col)
    if budget is not None and any(budget < len(rows) * floor for rows in period_rows):
        raise ValueError("infeasible_constraints: aggregate_max is below row count * min_demand for a period")
    constrained_values = {
        column: [
            min(max(value, floor), ceiling) if ceiling is not None else max(value, floor)
            for value, ceiling in zip(column_values, upper)
        ]
        for column, column_values in values.items()
    }
    points = constrained_values[value_col]
    if budget is not None:
        labels = frame[id_col].cast(pl.String).to_list()
        for rows in period_rows:
            try:
                total = math.fsum(points[index] for index in rows)
            except OverflowError:
                # An overflowing nonnegative total exceeds every finite budget.
                total = math.inf
            if total <= budget:
                continue
            headroom = budget - len(rows) * floor
            if constraints.proportional_redistribution:
                scale = max(points[index] - floor for index in rows)
                denominator = math.fsum((points[index] - floor) / scale for index in rows)
                for index in rows:
                    points[index] = floor + headroom * (((points[index] - floor) / scale) / denominator)
            else:
                ordered = sorted(rows, key=lambda index: (-points[index], labels[index]))
                # Retain headroom in reverse removal order to avoid cancellation
                # when subtracting a large excess from nearly equal forecasts.
                for index in reversed(ordered):
                    retained = min(points[index] - floor, headroom)
                    points[index] = floor + retained
                    headroom -= retained

    for column_values in constrained_values.values():
        for value, ceiling in zip(column_values, upper):
            if not math.isfinite(value) or value < floor - (1e-9 + 1e-12 * floor):
                raise ArithmeticError("Constraint output failed the finite lower-bound check")
            if ceiling is not None and value > ceiling + (1e-9 + 1e-12 * ceiling):
                raise ArithmeticError("Constraint output failed the upper-bound check")
    if budget is not None:
        for rows in period_rows:
            total = _finite_sum([points[index] for index in rows])
            if total - budget > 1e-9 + 1e-12 * budget:
                raise ArithmeticError("Constraint output failed the per-period budget check")
    result = frame.with_columns(
        [pl.Series(column, column_values, dtype=pl.Float64) for column, column_values in constrained_values.items()]
    )
    validate_quantile_forecasts(result, quantiles, "Constraint output")
    return result


@registry.register("constrained_demand")
class ConstrainedDemandEstimator(BaseForecaster):
    """
    Forecaster wrapper that enforces capacity and business-rule constraints.

    Delegates fit/predict to an inner *base_forecaster*, then clips or
    redistributes the resulting forecasts to satisfy:

    - **Element-wise bounds**: ``min_demand ≤ forecast ≤ capacity`` per row.
    - **Aggregate budget**: ``sum(forecast) ≤ aggregate_max`` per period,
      with proportional scaling or clip-largest-first redistribution.

    Quantiles support scalar/per-series bounds without an aggregate budget.
    Invalid input ordering is rejected, never repaired by sorting.

    Parameters
    ----------
    base_forecaster : BaseForecaster
        Any fitted-or-unfitted forecaster to wrap.
    constraints : ConstraintConfig
        Constraint thresholds and toggle flags.
    """

    name = "constrained_demand"

    def __init__(
        self,
        base_forecaster: BaseForecaster,
        constraints: ConstraintConfig,
    ) -> None:
        self.base = base_forecaster
        self.constraints = constraints
        self._capacity_map: Dict[object, float] = {}

    # ── BaseForecaster interface ───────────────────────────────────────────

    def validate_and_prepare(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """Extract per-series capacity from data if configured; delegate to base."""
        self._capacity_map = {}
        if self.constraints.enabled:
            floor, _, _ = _constraint_limits(self.constraints)
            column = self.constraints.capacity_column
            if column is not None:
                if df.is_empty() or column not in df.columns or id_col not in df.columns:
                    raise ValueError("invalid_capacity_data: configured capacity column and series IDs are required at every fit")
                if column == id_col or df[id_col].null_count():
                    raise ValueError("invalid_capacity_data: capacity needs a distinct, non-null series identity")
                _validate_key_values(df, (id_col,))
                capacities = _finite_values(df, column)
                if any(value < floor for value in capacities):
                    raise ValueError("infeasible_constraints: training capacities must be nonnegative and at least min_demand")
                cap = df.select(
                    pl.col(id_col).alias("identity"),
                    pl.col(column).cast(pl.Float64).alias("capacity"),
                ).group_by("identity").agg(pl.col("capacity").max())
                self._capacity_map = dict(cap.iter_rows())

        return self.base.validate_and_prepare(
            df, target_col, time_col, id_col
        )

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        """Fit the base forecaster (constraints are applied at predict time)."""
        df = self.validate_and_prepare(df, target_col, time_col, id_col)
        self.base.fit(df, target_col, time_col, id_col)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
        value_col: str = "forecast",
    ) -> pl.DataFrame:
        """Generate constrained point forecasts."""
        forecast = self.base.predict(horizon, id_col=id_col, time_col=time_col)

        return apply_point_constraints(
            forecast, self.constraints, id_col, time_col, value_col,
            self._capacity_map if self.constraints.capacity_column is not None else None,
        )

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """Generate constrained quantile forecasts."""
        if self.constraints.enabled and self.constraints.aggregate_max is not None:
            raise ValueError("unsupported_aggregate_quantiles: predict_quantiles does not support aggregate budgets")
        qf = self.base.predict_quantiles(
            horizon, quantiles, id_col=id_col, time_col=time_col
        )

        if not self.constraints.enabled:
            return qf

        if not quantiles:
            raise ValueError("Constraint quantiles require at least one requested level")
        _validate_input_quantiles(qf, quantiles)
        value_col = "forecast" if "forecast" in qf.columns else f"forecast_p{int(round(quantiles[0] * 100))}"
        return apply_point_constraints(
            qf, self.constraints, id_col, time_col, value_col,
            self._capacity_map if self.constraints.capacity_column is not None else None,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "base": self.base.name,
            "min_demand": self.constraints.min_demand,
            "max_capacity": self.constraints.max_capacity,
            "aggregate_max": self.constraints.aggregate_max,
            **self.base.get_params(),
        }

    def __repr__(self) -> str:
        return (
            f"ConstrainedDemandEstimator("
            f"base={self.base.name!r}, "
            f"enabled={self.constraints.enabled})"
        )
