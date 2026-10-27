"""
Prediction interval calibration — coverage checks and conformal correction.

Provides:
  - Empirical coverage computation (do 80% of actuals fall within P10–P90?)
  - Calibration report with per-model and per-series breakdown
  - Conformal prediction correction to adjust miscalibrated intervals
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import polars as pl

# --------------------------------------------------------------------------- #
#  Result types
# --------------------------------------------------------------------------- #

@dataclass
class IntervalCoverage:
    """Coverage statistics for one interval level."""
    label: str              # e.g. "80"
    nominal: float          # e.g. 0.80
    empirical: float        # e.g. 0.65
    miscalibration: float   # nominal - empirical (positive = too narrow)
    sharpness: float        # mean interval width
    n_observations: int


@dataclass
class CalibrationReport:
    """Full calibration report across models and intervals."""
    model_reports: Dict[str, List[IntervalCoverage]]
    per_series: pl.DataFrame  # [model_id, series_id, label, nominal, empirical, sharpness]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _coverage_quantile_pair(label: str) -> Tuple[float, float]:
    """
    Map a coverage label to a (lower, upper) quantile pair.

    "80" → (0.10, 0.90)   — symmetric around median
    "90" → (0.05, 0.95)
    "50" → (0.25, 0.75)
    """
    pct = float(label) / 100.0
    tail = (1.0 - pct) / 2.0
    return (tail, 1.0 - tail)


def _quantile_col(q: float) -> str:
    """Quantile float → column name: 0.1 → 'forecast_p10'."""
    return f"forecast_p{int(round(q * 100))}"


def validate_quantile_forecasts(
    forecast_df: pl.DataFrame,
    quantiles: List[float],
    context: str = "Forecast",
) -> None:
    """Require finite, ordered requested percentiles, not point containment."""
    if any(not math.isfinite(q) or not 0 <= q <= 1 for q in quantiles):
        raise ValueError(f"{context} quantile levels must be finite and between 0 and 1")
    columns = [_quantile_col(q) for q in sorted(quantiles)]
    if len(set(columns)) != len(columns):
        raise ValueError(f"{context} quantile levels must have distinct percentile column names")
    for column in columns:
        if column not in forecast_df.columns:
            raise ValueError(f"{context} quantiles missing column {column}")
        values = forecast_df[column]
        if not values.dtype.is_numeric() or values.null_count() or not values.is_finite().all():
            raise ValueError(f"{context} quantiles require finite numeric {column}")
    for lower, upper in zip(columns, columns[1:]):
        if (forecast_df[lower] > forecast_df[upper]).any():
            raise ValueError(
                f"{context} requires ordered quantiles: {lower} exceeds {upper}"
            )


# --------------------------------------------------------------------------- #
#  Core metric
# --------------------------------------------------------------------------- #

def compute_interval_coverage(
    actuals: pl.Series,
    lower: pl.Series,
    upper: pl.Series,
) -> float:
    """
    Fraction of actuals that fall within [lower, upper].

    Parameters
    ----------
    actuals, lower, upper : pl.Series of equal length.

    Returns
    -------
    Coverage rate in [0, 1].
    """
    if actuals.len() == 0:
        return 0.0
    covered = (actuals >= lower) & (actuals <= upper)
    return float(covered.mean())


# --------------------------------------------------------------------------- #
#  Calibration report
# --------------------------------------------------------------------------- #

def compute_calibration_report(
    backtest_results: pl.DataFrame,
    quantiles: List[float],
    coverage_targets: Dict[str, float],
) -> CalibrationReport:
    """
    Compute empirical coverage for each model and coverage target.

    Parameters
    ----------
    backtest_results:
        Backtest output with columns: model_id, series_id, actual,
        forecast_p{q} for each configured quantile.
    quantiles:
        The quantile levels used (e.g. [0.1, 0.5, 0.9]).
    coverage_targets:
        Label → nominal coverage, e.g. {"80": 0.80}.

    Returns
    -------
    CalibrationReport with per-model and per-series breakdowns.
    """
    model_reports: Dict[str, List[IntervalCoverage]] = {}
    per_series_rows: List[dict] = []

    {round(q, 4) for q in quantiles}
    models = backtest_results["model_id"].unique().to_list()

    for model_id in models:
        model_df = backtest_results.filter(pl.col("model_id") == model_id)
        coverages: List[IntervalCoverage] = []

        for label, nominal in coverage_targets.items():
            lower_q, upper_q = _coverage_quantile_pair(label)

            # Check that the required quantile columns exist
            lower_col = _quantile_col(lower_q)
            upper_col = _quantile_col(upper_q)
            if lower_col not in model_df.columns or upper_col not in model_df.columns:
                continue

            # Filter out rows where quantile columns are null
            valid = model_df.filter(
                pl.col(lower_col).is_not_null() & pl.col(upper_col).is_not_null()
            )
            if valid.is_empty():
                continue

            # Overall coverage for this model
            empirical = compute_interval_coverage(
                valid["actual"],
                valid[lower_col],
                valid[upper_col],
            )
            sharpness = float((valid[upper_col] - valid[lower_col]).mean())

            coverages.append(IntervalCoverage(
                label=label,
                nominal=nominal,
                empirical=round(empirical, 4),
                miscalibration=round(nominal - empirical, 4),
                sharpness=round(sharpness, 2),
                n_observations=valid.height,
            ))

            # Per-series breakdown
            for sid in valid["series_id"].unique().to_list():
                s = valid.filter(pl.col("series_id") == sid)
                s_cov = compute_interval_coverage(
                    s["actual"], s[lower_col], s[upper_col],
                )
                s_sharp = float((s[upper_col] - s[lower_col]).mean())
                per_series_rows.append({
                    "model_id": model_id,
                    "series_id": sid,
                    "label": label,
                    "nominal": nominal,
                    "empirical": round(s_cov, 4),
                    "sharpness": round(s_sharp, 2),
                })

        model_reports[model_id] = coverages

    per_series = pl.DataFrame(per_series_rows) if per_series_rows else pl.DataFrame(
        schema={"model_id": pl.Utf8, "series_id": pl.Utf8, "label": pl.Utf8,
                "nominal": pl.Float64, "empirical": pl.Float64, "sharpness": pl.Float64}
    )

    return CalibrationReport(model_reports=model_reports, per_series=per_series)


# --------------------------------------------------------------------------- #
#  Conformal residuals
# --------------------------------------------------------------------------- #

def compute_conformal_residuals(
    backtest_results: pl.DataFrame,
    quantiles: List[float],
    coverage_targets: Dict[str, float],
    id_col: str = "series_id",
) -> pl.DataFrame:
    """
    Compute nonconformity scores for conformal prediction.

    For each interval, the nonconformity score is:
        s_i = max(lower - actual_i, actual_i - upper)

    Positive scores mean the actual fell outside the interval.

    Returns
    -------
    DataFrame with columns: [model_id, residual_{label}] — one row per
    (model, series, week) observation with the nonconformity score.
    """
    result_cols = [id_col, "model_id"]
    df = backtest_results.clone()

    for label, nominal in coverage_targets.items():
        lower_q, upper_q = _coverage_quantile_pair(label)
        lower_col = _quantile_col(lower_q)
        upper_col = _quantile_col(upper_q)

        if lower_col not in df.columns or upper_col not in df.columns:
            continue

        resid_col = f"residual_{label}"
        df = df.with_columns(
            pl.max_horizontal(
                pl.col(lower_col) - pl.col("actual"),
                pl.col("actual") - pl.col(upper_col),
            ).alias(resid_col),
        )
        result_cols.append(resid_col)

    return df.select(result_cols)


# --------------------------------------------------------------------------- #
#  Conformal correction
# --------------------------------------------------------------------------- #

def apply_conformal_correction(
    forecast_df: pl.DataFrame,
    conformal_residuals: pl.DataFrame,
    quantiles: List[float],
    coverage_targets: Dict[str, float],
    id_col: str = "series_id",
    model_id: Optional[str] = None,
) -> pl.DataFrame:
    """
    Apply split-conformal prediction correction to forecast intervals.

    For each coverage target:
      1. From residuals, compute the correction quantile q_hat at level
         ceil((n+1) * nominal) / n
      2. Adjust intervals: lower -= q_hat, upper += q_hat

    Negative corrections may tighten intervals, but a correction that crosses
    requested quantiles raises ValueError rather than clipping or changing the
    correction. The point forecast and any median are unchanged.

    Parameters
    ----------
    forecast_df:
        Production forecast with quantile columns (forecast_p10, etc.).
    conformal_residuals:
        Residuals from ``compute_conformal_residuals``.
    quantiles:
        The quantile levels used.
    coverage_targets:
        Label → nominal coverage.
    id_col:
        Series identifier column.
    model_id:
        If provided, filter residuals to this model only.

    Returns
    -------
    Forecast DataFrame with adjusted quantile columns.
    """
    validate_quantile_forecasts(forecast_df, quantiles, "Conformal input")
    if conformal_residuals.is_empty():
        return forecast_df

    resids = conformal_residuals
    if model_id and "model_id" in resids.columns:
        resids = resids.filter(pl.col("model_id") == model_id)

    for label, nominal in coverage_targets.items():
        lower_q, upper_q = _coverage_quantile_pair(label)
        lower_col = _quantile_col(lower_q)
        upper_col = _quantile_col(upper_q)
        resid_col = f"residual_{label}"

        if lower_col not in forecast_df.columns or upper_col not in forecast_df.columns:
            continue
        if resid_col not in resids.columns:
            continue

        # Compute the conformal quantile (global, not per-series)
        scores = resids[resid_col]
        if not scores.dtype.is_numeric() or scores.null_count() or not scores.is_finite().all():
            raise ValueError(f"Conformal residuals require finite numeric {resid_col}")
        scores = scores.sort()
        n = scores.len()
        if n == 0:
            continue

        # Conformal quantile level: ceil((n+1) * nominal) / n
        idx = min(math.ceil((n + 1) * nominal) - 1, n - 1)
        idx = max(idx, 0)
        q_hat = float(scores[idx])

        forecast_df = forecast_df.with_columns(
            (pl.col(lower_col) - q_hat).alias(lower_col),
            (pl.col(upper_col) + q_hat).alias(upper_col),
        )

    validate_quantile_forecasts(forecast_df, quantiles, "Conformal correction")
    return forecast_df
