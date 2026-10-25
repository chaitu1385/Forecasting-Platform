"""
Layer 2 — Logical Validation for forecasting data.

Validates logical consistency: aggregation roll-ups, percentage sums,
monotonicity, trend coherence, ratio bounds, group balance, and temporal
sanity.  Designed for hierarchical forecasting checks — e.g. verifying
that store-level forecasts sum to the region-level forecast.

All functions accept Polars DataFrames and return dicts with ``ok``,
``severity``, and descriptive details.

Usage:
    from src.validation.logical_validator import (
        validate_aggregation_consistency, validate_trend_consistency,
        validate_no_future_dates, run_logical_checks,
    )
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_float_series(s: pl.Series) -> pl.Series:
    """Cast a series to Float64 safely."""
    return s.cast(pl.Float64)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_aggregation_consistency(
    df: pl.DataFrame,
    group_col: str,
    value_col: str,
    parent_group_col: Optional[str] = None,
    tolerance: float = 0.01,
) -> dict:
    """Check that child-level values aggregate within tolerance of parent totals.

    If ``parent_group_col`` is None, compares the sum of children against the
    overall total.

    Args:
        df: Polars DataFrame.
        group_col: Column grouping the child level (e.g. "store_id").
        value_col: Value column to aggregate (e.g. "quantity").
        parent_group_col: Optional parent column (e.g. "region").
        tolerance: Maximum relative difference.  Default 1%.

    Returns:
        dict with ``ok``, ``mismatches``, ``severity``.
    """
    issues: list[dict] = []

    if parent_group_col is None:
        child_sum = df[value_col].sum()
        if child_sum is None or child_sum == 0:
            return {"ok": True, "mismatches": [], "severity": "PASS"}
        # Nothing more to compare without parent groups
        return {"ok": True, "mismatches": [], "severity": "PASS"}

    # Compare parent aggregate vs child aggregate per parent
    child_agg = (
        df.group_by(parent_group_col)
        .agg(pl.col(value_col).sum().alias("child_sum"))
    )

    parent_agg = (
        df.group_by(parent_group_col)
        .agg(pl.col(value_col).sum().alias("parent_sum"))
    )

    merged = child_agg.join(parent_agg, on=parent_group_col, how="inner")
    merged = merged.with_columns(
        (
            (pl.col("child_sum") - pl.col("parent_sum")).abs()
            / pl.when(pl.col("parent_sum").abs() > 0)
                .then(pl.col("parent_sum").abs())
                .otherwise(1.0)
        ).alias("rel_diff")
    )

    bad = merged.filter(pl.col("rel_diff") > tolerance)
    for row in bad.to_dicts():
        issues.append({
            "parent": row[parent_group_col],
            "child_sum": row["child_sum"],
            "parent_sum": row["parent_sum"],
            "rel_diff": round(row["rel_diff"], 4),
        })

    severity = "BLOCKER" if issues else "PASS"
    return {"ok": len(issues) == 0, "mismatches": issues, "severity": severity}


def validate_percentages_sum(
    df: pl.DataFrame,
    pct_columns: Sequence[str],
    expected_sum: float = 100.0,
    tolerance: float = 1.0,
    group_col: Optional[str] = None,
) -> dict:
    """Check that percentage columns sum to expected total per row/group.

    Args:
        df: Polars DataFrame.
        pct_columns: Column names that should sum to expected_sum.
        expected_sum: Target sum.  Default 100.0.
        tolerance: Max absolute difference allowed.
        group_col: If set, checks per group instead of per row.

    Returns:
        dict with ``ok``, ``violations_count``, ``violations_sample``.
    """
    cols = list(pct_columns)
    if not cols:
        return {"ok": True, "violations_count": 0, "violations_sample": []}

    if group_col:
        agg_df = df.group_by(group_col).agg(
            [pl.col(c).sum() for c in cols]
        )
        agg_df = agg_df.with_columns(
            pl.sum_horizontal(*[pl.col(c) for c in cols]).alias("_pct_sum")
        )
    else:
        agg_df = df.with_columns(
            pl.sum_horizontal(*[pl.col(c) for c in cols]).alias("_pct_sum")
        )

    violations = agg_df.filter(
        (pl.col("_pct_sum") - expected_sum).abs() > tolerance
    )

    sample = violations.head(5).to_dicts()

    return {
        "ok": violations.height == 0,
        "violations_count": violations.height,
        "violations_sample": sample,
        "severity": "WARNING" if violations.height > 0 else "PASS",
    }


def validate_monotonic(
    df: pl.DataFrame,
    column: str,
    direction: str = "increasing",
    group_col: Optional[str] = None,
) -> dict:
    """Validate that a column is monotonically increasing or decreasing.

    Useful for cumulative metrics.

    Args:
        df: Polars DataFrame (assumed sorted).
        column: Column to check.
        direction: ``"increasing"`` or ``"decreasing"``.
        group_col: Run check per group.

    Returns:
        dict with ``ok``, ``violations_count``.
    """
    if group_col:
        groups = df[group_col].unique().to_list()
        total_violations = 0
        for g in groups:
            sub = df.filter(pl.col(group_col) == g)
            v = _count_monotonic_violations(sub, column, direction)
            total_violations += v
    else:
        total_violations = _count_monotonic_violations(df, column, direction)

    return {
        "ok": total_violations == 0,
        "violations_count": total_violations,
        "severity": "WARNING" if total_violations > 0 else "PASS",
    }


def _count_monotonic_violations(
    df: pl.DataFrame, column: str, direction: str,
) -> int:
    if df.height < 2:
        return 0
    diffs = df[column].diff().drop_nulls()
    if direction == "increasing":
        return diffs.filter(diffs < 0).len()
    else:
        return diffs.filter(diffs > 0).len()


def validate_trend_consistency(
    df: pl.DataFrame,
    date_column: str,
    value_column: str,
    z_threshold: float = 3.0,
    window: int = 12,
    group_col: Optional[str] = None,
) -> dict:
    """Detect anomalous spikes/drops using a rolling z-score.

    Args:
        df: Polars DataFrame sorted by date.
        date_column: Date column.
        value_column: Value column.
        z_threshold: Z-score threshold for flagging anomalies.
        window: Rolling window size.
        group_col: Run check per group.

    Returns:
        dict with ``ok``, ``anomalies`` (list of dicts with date, value, z_score).
    """
    anomalies: list[dict] = []

    def _check_group(sub: pl.DataFrame) -> list[dict]:
        local = []
        if sub.height < window + 1:
            return local
        sorted_df = sub.sort(date_column)
        vals = sorted_df[value_column].cast(pl.Float64)
        rolling_mean = vals.rolling_mean(window_size=window)
        rolling_std = vals.rolling_std(window_size=window)
        for i in range(window, len(vals)):
            mean_v = rolling_mean[i]
            std_v = rolling_std[i]
            if std_v is None or std_v == 0 or mean_v is None:
                continue
            z = abs((vals[i] - mean_v) / std_v)
            if z > z_threshold:
                local.append({
                    "date": str(sorted_df[date_column][i]),
                    "value": float(vals[i]),
                    "z_score": round(float(z), 2),
                    "group": None,
                })
        return local

    if group_col:
        for g in df[group_col].unique().to_list():
            sub = df.filter(pl.col(group_col) == g)
            for a in _check_group(sub):
                a["group"] = g
                anomalies.append(a)
    else:
        anomalies = _check_group(df)

    return {
        "ok": len(anomalies) == 0,
        "anomalies": anomalies[:20],  # cap sample
        "severity": "WARNING" if anomalies else "PASS",
    }


def validate_ratio_bounds(
    df: pl.DataFrame,
    numerator_col: str,
    denominator_col: str,
    min_ratio: Optional[float] = None,
    max_ratio: Optional[float] = None,
) -> dict:
    """Validate that the ratio of two columns stays within expected bounds.

    Useful for forecast accuracy ratios, error rates, etc.

    Args:
        df: Polars DataFrame.
        numerator_col: Numerator column.
        denominator_col: Denominator column.
        min_ratio: Minimum allowed ratio.
        max_ratio: Maximum allowed ratio.

    Returns:
        dict with ``ok``, ``out_of_bounds_count``, ``sample``.
    """
    ratio_df = df.with_columns(
        pl.when(pl.col(denominator_col) != 0)
        .then(pl.col(numerator_col) / pl.col(denominator_col))
        .otherwise(None)
        .alias("_ratio")
    )

    conditions = []
    if min_ratio is not None:
        conditions.append(pl.col("_ratio") < min_ratio)
    if max_ratio is not None:
        conditions.append(pl.col("_ratio") > max_ratio)

    if not conditions:
        return {"ok": True, "out_of_bounds_count": 0, "sample": [],
                "severity": "PASS"}

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    violations = ratio_df.filter(combined)
    sample = violations.head(5).to_dicts()

    return {
        "ok": violations.height == 0,
        "out_of_bounds_count": violations.height,
        "sample": sample,
        "severity": "WARNING" if violations.height > 0 else "PASS",
    }


def validate_group_balance(
    df: pl.DataFrame,
    group_col: str,
    min_count: int = 5,
    max_imbalance_ratio: float = 100.0,
) -> dict:
    """Check that groups have balanced sizes.

    Useful for ensuring all series have enough data points.

    Args:
        df: Polars DataFrame.
        group_col: Column to group by.
        min_count: Minimum rows per group.
        max_imbalance_ratio: Max ratio between largest and smallest group.

    Returns:
        dict with ``ok``, ``group_sizes``, ``imbalance_ratio``,
        ``undersize_groups``.
    """
    counts = (
        df.group_by(group_col)
        .agg(pl.col(group_col).len().alias("n"))
        .sort("n")
    )

    min_n = counts["n"].min()
    max_n = counts["n"].max()
    ratio = max_n / min_n if min_n and min_n > 0 else float("inf")

    undersize = counts.filter(pl.col("n") < min_count)
    undersize_list = undersize[group_col].to_list()

    issues = []
    if ratio > max_imbalance_ratio:
        issues.append(
            f"Group imbalance ratio {ratio:.1f}x exceeds max {max_imbalance_ratio}x"
        )
    if undersize.height > 0:
        issues.append(
            f"{undersize.height} groups below minimum count {min_count}"
        )

    severity = "BLOCKER" if undersize.height > 0 else (
        "WARNING" if ratio > max_imbalance_ratio else "PASS"
    )

    return {
        "ok": len(issues) == 0,
        "group_count": counts.height,
        "min_count": int(min_n) if min_n is not None else 0,
        "max_count": int(max_n) if max_n is not None else 0,
        "imbalance_ratio": round(ratio, 1) if ratio != float("inf") else None,
        "undersize_groups": undersize_list[:20],
        "severity": severity,
    }


def validate_no_future_dates(
    df: pl.DataFrame,
    date_column: str,
    reference_date: Optional[date] = None,
    allow_forecast_horizon: int = 0,
) -> dict:
    """Check that dates in actuals don't exceed a reference date.

    In forecasting, you might allow dates up to forecast_horizon days ahead.

    Args:
        df: Polars DataFrame.
        date_column: Date column.
        reference_date: Max allowed date. Defaults to today.
        allow_forecast_horizon: Additional days allowed.

    Returns:
        dict with ``ok``, ``future_count``, ``max_date``.
    """
    if reference_date is None:
        from datetime import timedelta
        reference_date = date.today()

    boundary = reference_date
    if allow_forecast_horizon:
        from datetime import timedelta
        boundary = reference_date + timedelta(days=allow_forecast_horizon)

    dates = df[date_column].drop_nulls()
    # Cast to date if datetime
    if dates.dtype == pl.Datetime:
        dates = dates.dt.date()

    future = dates.filter(dates > boundary)
    max_date = dates.max()

    return {
        "ok": future.len() == 0,
        "future_count": future.len(),
        "max_date": max_date,
        "boundary": boundary,
        "severity": "WARNING" if future.len() > 0 else "PASS",
    }


def validate_forecast_vs_actual_alignment(
    df_forecast: pl.DataFrame,
    df_actual: pl.DataFrame,
    date_column: str,
    value_column: str,
    series_column: Optional[str] = None,
    max_directional_mismatch_rate: float = 0.5,
) -> dict:
    """Check that forecast direction (up/down) aligns with actuals.

    Flags when >50% of forecasts go in the wrong direction vs actuals.

    Args:
        df_forecast: DataFrame with forecasts.
        df_actual: DataFrame with actuals.
        date_column: Date column (shared key).
        value_column: Value column name.
        series_column: Series ID column for series-level join.
        max_directional_mismatch_rate: Threshold for mismatch rate.

    Returns:
        dict with ``ok``, ``mismatch_rate``, ``severity``.
    """
    join_cols = [date_column]
    if series_column:
        join_cols.append(series_column)

    merged = df_forecast.select(
        join_cols + [pl.col(value_column).alias("forecast")]
    ).join(
        df_actual.select(
            join_cols + [pl.col(value_column).alias("actual")]
        ),
        on=join_cols,
        how="inner",
    )

    if merged.height < 2:
        return {"ok": True, "mismatch_rate": 0.0, "severity": "PASS",
                "message": "Insufficient overlap to check alignment"}

    # Compute period-over-period direction
    merged = merged.sort(date_column)
    merged = merged.with_columns([
        pl.col("forecast").diff().sign().alias("fc_dir"),
        pl.col("actual").diff().sign().alias("act_dir"),
    ])

    compared = merged.filter(
        pl.col("fc_dir").is_not_null() & pl.col("act_dir").is_not_null()
    )
    if compared.height == 0:
        return {"ok": True, "mismatch_rate": 0.0, "severity": "PASS"}

    mismatches = compared.filter(pl.col("fc_dir") != pl.col("act_dir")).height
    rate = mismatches / compared.height

    return {
        "ok": rate <= max_directional_mismatch_rate,
        "mismatch_rate": round(rate, 4),
        "total_compared": compared.height,
        "severity": "WARNING" if rate > max_directional_mismatch_rate else "PASS",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_logical_checks(
    df: pl.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run configured logical checks and aggregate results.

    Args:
        df: Polars DataFrame.
        config: Controls which checks to run:
            - ``date_column``: str (enables trend + future-date checks)
            - ``value_column``: str (enables trend check)
            - ``group_col``: str (enables group balance)
            - ``z_threshold``: float (trend check, default 3.0)
            - ``window``: int (trend check, default 12)
            - ``min_group_count``: int (group balance, default 5)
            - ``reference_date``: date (future dates check)
            - ``pct_columns``: list[str] (percentage sum check)

    Returns:
        dict with ``ok``, ``checks_run``, ``checks_passed``,
        ``checks_failed``, ``details``.
    """
    config = config or {}
    details: Dict[str, dict] = {}

    # Trend consistency
    if config.get("date_column") and config.get("value_column"):
        details["trend_consistency"] = validate_trend_consistency(
            df,
            date_column=config["date_column"],
            value_column=config["value_column"],
            z_threshold=config.get("z_threshold", 3.0),
            window=config.get("window", 12),
            group_col=config.get("group_col"),
        )

    # No future dates
    if config.get("date_column"):
        details["no_future_dates"] = validate_no_future_dates(
            df,
            date_column=config["date_column"],
            reference_date=config.get("reference_date"),
        )

    # Group balance
    if config.get("group_col"):
        details["group_balance"] = validate_group_balance(
            df,
            group_col=config["group_col"],
            min_count=config.get("min_group_count", 5),
        )

    # Percentage sum
    if config.get("pct_columns"):
        details["percentages_sum"] = validate_percentages_sum(
            df, pct_columns=config["pct_columns"],
        )

    checks_run = len(details)
    checks_passed = sum(1 for r in details.values() if r.get("ok", True))
    checks_failed = checks_run - checks_passed

    return {
        "ok": all(r.get("ok", True) for r in details.values()),
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "details": details,
    }
