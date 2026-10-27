"""
Forecast accuracy metric definitions.

Primary metrics:
  - WMAPE  (Weighted Mean Absolute Percentage Error)
  - Normalized Bias  (directional error — over vs. under forecasting)

Secondary metrics (available for deeper analysis):
  - MAPE, MAE, RMSE, MASE

All functions operate on Polars Series for vectorised computation.
"""

from typing import Any, Callable, Dict, List, Optional

import polars as pl

# ── Metric functions ──────────────────────────────────────────────────────────

def wmape(actual: pl.Series, forecast: pl.Series) -> Optional[float]:
    """
    Weighted Mean Absolute Percentage Error.

    WMAPE = Σ|actual - forecast| / Σ|actual|

    - Volume-weighted: high-movers naturally dominate.
    - Handles zeros better than MAPE (no per-row division).
    - Range: [0, ∞), lower is better.

    Returns ``None`` when the actuals are entirely zero on the evaluation
    window — WMAPE is undefined in that case (any forecast error divided by
    zero volume is meaningless). Callers/aggregators skip nulls.
    """
    abs_error = (actual - forecast).abs().sum()
    abs_actual = actual.abs().sum()
    if abs_actual == 0:
        return None
    return float(abs_error / abs_actual)


def normalized_bias(actual: pl.Series, forecast: pl.Series) -> Optional[float]:
    """
    Normalized Bias — measures systematic over/under-forecasting.

    Bias = Σ(forecast - actual) / Σ|actual|

    - Positive → over-forecasting (forecast > actual on average)
    - Negative → under-forecasting (forecast < actual on average)
    - Range: (-∞, +∞), closer to 0 is better.
    - Undefined (``None``) when total absolute actual volume is zero.
    """
    signed_error = (forecast - actual).sum()
    abs_actual = actual.abs().sum()
    if abs_actual == 0:
        return None
    return float(signed_error / abs_actual)


def mape(actual: pl.Series, forecast: pl.Series) -> Optional[float]:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|actual - forecast| / |actual|)

    Excluded where actual == 0 to avoid division by zero.
    Returns ``None`` if every row has actual == 0 (MAPE undefined).
    """
    mask = actual != 0
    if mask.sum() == 0:
        return None
    pct_errors = ((actual - forecast).abs() / actual.abs()).filter(mask)
    return float(pct_errors.mean())


def mae(actual: pl.Series, forecast: pl.Series) -> float:
    """Mean Absolute Error."""
    return float((actual - forecast).abs().mean())


def rmse(actual: pl.Series, forecast: pl.Series) -> float:
    """Root Mean Squared Error."""
    return float(((actual - forecast) ** 2).mean() ** 0.5)


# ── MASE helpers ─────────────────────────────────────────────────────────────

def _naive_seasonal_mae(insample: pl.Series, m: int = 52) -> float:
    """MAE of the naive seasonal forecast on in-sample data."""
    if len(insample) <= m:
        return float("inf")
    diffs = (insample[m:] - insample[:len(insample) - m]).abs()
    return float(diffs.mean())


def make_mase(
    insample: pl.Series, seasonal_period: int = 52
) -> Callable[[pl.Series, pl.Series], float]:
    """
    Factory: returns a 2-arg MASE function with frozen denominator.

    MASE = mean(|actual - forecast|) / mean(|y_t - y_{t-m}|)

    The denominator is the MAE of the naive seasonal forecast on the
    *training* (in-sample) data, so it must be provided at construction time.

    Parameters
    ----------
    insample : pl.Series
        Training data used to compute the naive seasonal MAE denominator.
    seasonal_period : int
        Seasonal period in the same units as the data (default 52 for weekly).

    Returns
    -------
    Callable that takes (actual, forecast) and returns the MASE score.
    """
    denom = _naive_seasonal_mae(insample, seasonal_period)

    def mase(actual: pl.Series, forecast: pl.Series) -> float:
        if denom == 0 or denom == float("inf"):
            return float("inf")
        return float((actual - forecast).abs().mean() / denom)

    return mase


# ── Registry ──────────────────────────────────────────────────────────────────

METRIC_REGISTRY: Dict[str, Callable[[pl.Series, pl.Series], float]] = {
    "wmape": wmape,
    "normalized_bias": normalized_bias,
    "mape": mape,
    "mae": mae,
    "rmse": rmse,
}

# Aggregation semantics and ranking direction are shared by all selectors.
# ``macro_wmape`` is the explicitly named equal-(series, fold) alternative;
# it is not an alias for the volume-weighted WMAPE used by default.
METRIC_DIRECTIONS = {
    "wmape": "minimize",
    "macro_wmape": "minimize",
    "normalized_bias": "minimize_absolute",
    "mape": "minimize",
    "mae": "minimize",
    "rmse": "minimize",
    "mase": "minimize",
}


def metric_expression(name: str, *, aggregate: bool = True) -> pl.Expr:
    """Canonical Polars expression over raw ``actual``/``forecast`` rows.

    The nonaggregate form describes one observation, not a repeated full-fold
    score. MASE additionally requires its training-only ``mase_scale`` column.
    """
    actual = pl.col("actual")
    error = pl.col("forecast") - actual
    absolute = error.abs()
    volume = actual.abs()
    if name in ("wmape", "normalized_bias"):
        numerator = absolute if name == "wmape" else error
        if aggregate:
            numerator, volume = numerator.sum(), volume.sum()
        expression = pl.when(volume > 0).then(numerator / volume).otherwise(None)
    elif name == "mape":
        expression = pl.when(volume > 0).then(absolute / volume).otherwise(None)
        if aggregate:
            expression = expression.mean()
    elif name == "mae":
        expression = absolute.mean() if aggregate else absolute
    elif name == "rmse":
        expression = (error.pow(2).mean()).sqrt() if aggregate else absolute
    elif name == "mase":
        scale = pl.col("mase_scale")
        valid = scale.is_not_null() & scale.is_finite() & (scale > 0)
        expression = pl.when(valid).then(absolute / scale).otherwise(None)
        if aggregate:
            expression = pl.when(valid.all()).then(expression.mean()).otherwise(None)
    else:
        raise ValueError(f"Unknown or non-additive metric {name!r}")
    return expression.cast(pl.Float64).alias(name)


def validate_metric_inputs(actual: pl.Series, forecast: pl.Series) -> None:
    """Reject empty, misaligned, null or nonfinite observations before scoring."""
    if len(actual) == 0 or len(actual) != len(forecast):
        raise ValueError("Scoring requires nonempty actual/forecast arrays of equal length")
    for label, values in (("actual", actual), ("forecast", forecast)):
        if not values.dtype.is_numeric() or values.null_count() or not values.is_finite().all():
            raise ValueError(f"Scoring requires finite, non-null numeric {label} values")

# Metrics that require extra context (e.g. in-sample data) beyond actual/forecast.
# Key = metric name, value = context key expected in the ``context`` dict.
CONTEXT_METRICS: Dict[str, str] = {
    "mase": "insample",
}


def compute_all_metrics(
    actual: pl.Series,
    forecast: pl.Series,
    metric_names: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute multiple metrics at once.

    Parameters
    ----------
    actual, forecast:
        Polars Series of equal length.
    metric_names:
        Which metrics to compute.  Defaults to all registered.
    context:
        Optional dict of extra data for context-dependent metrics.
        For MASE, pass ``{"insample": <training series>}``.

    Returns
    -------
    Dict mapping metric name → computed value.
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())
    validate_metric_inputs(actual, forecast)

    results = {}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is not None:
            results[name] = fn(actual, forecast)
            continue

        # Context-dependent metric (e.g. MASE)
        if name in CONTEXT_METRICS:
            ctx_key = CONTEXT_METRICS[name]
            if context and ctx_key in context:
                fn = make_mase(context[ctx_key], seasonal_period=context.get("seasonal_period", 52))
                results[name] = fn(actual, forecast)
            else:
                results[name] = float("nan")
            continue

        raise KeyError(
            f"Unknown metric {name!r}. "
            f"Available: {list(METRIC_REGISTRY.keys()) + list(CONTEXT_METRICS.keys())}"
        )
    return results
