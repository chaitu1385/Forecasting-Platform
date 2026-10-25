"""
Layer 1 — Structural Validation for forecasting data.

Validates data structure: schema, primary keys, completeness, date ranges,
referential integrity, value domains, and row counts.  All functions accept
Polars DataFrames (the platform standard) and return dicts with a consistent
``ok`` (bool) + ``severity`` ("PASS" / "WARNING" / "BLOCKER") pattern.

This module complements ``src.data.validator.DataValidator`` — use DataValidator
for pipeline-integrated schema enforcement, use this module for standalone
pre-analysis structural checks and for feeding into confidence scoring.

Usage:
    from src.validation.structural_validator import (
        validate_schema, validate_primary_key, validate_completeness,
        validate_date_range, validate_row_count, run_structural_checks,
    )

    result = validate_schema(df, expected_columns=["week", "series_id", "quantity"])
    if not result["ok"]:
        print(result["issues"])
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# Type compatibility sets
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

_DATETIME_DTYPES = {pl.Date, pl.Datetime}

_STRING_DTYPES = {pl.Utf8, pl.Categorical}


def _dtypes_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check fuzzy type compatibility (e.g. Int64 ~ Float64)."""
    if actual == expected:
        return True
    if actual in _NUMERIC_DTYPES and expected in _NUMERIC_DTYPES:
        return True
    if actual in _DATETIME_DTYPES and expected in _DATETIME_DTYPES:
        return True
    if actual in _STRING_DTYPES and expected in _STRING_DTYPES:
        return True
    return False


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_schema(
    df: pl.DataFrame,
    expected_columns: Optional[Sequence[str]] = None,
    expected_types: Optional[Dict[str, pl.DataType]] = None,
) -> dict:
    """Check that DataFrame has the expected columns and types.

    Args:
        df: Polars DataFrame to validate.
        expected_columns: List of column names that must be present.
        expected_types: Dict mapping ``column_name -> expected Polars dtype``.

    Returns:
        dict with keys: ``ok``, ``missing_columns``, ``extra_columns``,
        ``dtype_mismatches``, ``issues``, ``severity``.
    """
    issues: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    dtype_mismatches: list[dict] = []

    actual_cols = set(df.columns)

    if expected_columns:
        expected_set = set(expected_columns)
        missing = sorted(expected_set - actual_cols)
        extra = sorted(actual_cols - expected_set)
        for col in missing:
            issues.append(f"Missing required column: '{col}'")

    if expected_types:
        for col, expected_dt in expected_types.items():
            if col not in actual_cols:
                continue  # already caught by missing check
            actual_dt = df[col].dtype
            if not _dtypes_compatible(actual_dt, expected_dt):
                dtype_mismatches.append({
                    "column": col,
                    "expected": str(expected_dt),
                    "actual": str(actual_dt),
                })
                issues.append(
                    f"Column '{col}' has type {actual_dt}, expected {expected_dt}"
                )

    severity = "BLOCKER" if missing else ("WARNING" if dtype_mismatches else "PASS")

    return {
        "ok": len(issues) == 0,
        "missing_columns": missing,
        "extra_columns": extra,
        "dtype_mismatches": dtype_mismatches,
        "issues": issues,
        "severity": severity,
    }


def validate_primary_key(
    df: pl.DataFrame,
    key_columns: Sequence[str],
) -> dict:
    """Check that composite key columns have no nulls and no duplicates.

    Args:
        df: Polars DataFrame.
        key_columns: Columns that form the primary key.

    Returns:
        dict with keys: ``ok``, ``null_count``, ``duplicate_count``,
        ``duplicate_sample``, ``severity``.
    """
    if df.is_empty():
        return {"ok": True, "null_count": 0, "duplicate_count": 0,
                "duplicate_sample": [], "severity": "PASS"}

    key_cols = list(key_columns)

    # Null check
    null_count = 0
    for col in key_cols:
        if col in df.columns:
            null_count += df[col].null_count()

    # Duplicate check
    total = len(df)
    unique = df.select(key_cols).unique().height
    dup_count = total - unique

    dup_sample: list[dict] = []
    if dup_count > 0:
        dupes = (
            df.group_by(key_cols)
            .agg(pl.col(key_cols[0]).len().alias("_cnt"))
            .filter(pl.col("_cnt") > 1)
            .head(5)
        )
        dup_sample = dupes.drop("_cnt").to_dicts()

    ok = null_count == 0 and dup_count == 0
    if dup_count > 0 or null_count > 0:
        severity = "BLOCKER"
    else:
        severity = "PASS"

    return {
        "ok": ok,
        "null_count": null_count,
        "duplicate_count": dup_count,
        "duplicate_sample": dup_sample,
        "severity": severity,
    }


def validate_completeness(
    df: pl.DataFrame,
    required_columns: Optional[Sequence[str]] = None,
    threshold: float = 0.95,
) -> dict:
    """Check null rate per column against a completeness threshold.

    Args:
        df: Polars DataFrame.
        required_columns: Columns to check. If None, checks all columns.
        threshold: Minimum completeness (1 - null_rate). Default: 0.95 (5% max nulls).

    Returns:
        dict with keys: ``ok``, ``column_stats``, ``overall_null_rate``,
        ``overall_severity``.
    """
    columns = list(required_columns) if required_columns else df.columns
    max_null_rate = 1.0 - threshold
    total_rows = len(df)
    column_stats: list[dict] = []
    worst_severity = "PASS"

    for col in columns:
        if col not in df.columns:
            column_stats.append({
                "name": col, "null_count": total_rows,
                "null_rate": 1.0, "severity": "BLOCKER",
            })
            worst_severity = "BLOCKER"
            continue

        null_count = df[col].null_count()
        null_rate = null_count / total_rows if total_rows > 0 else 0.0

        if null_rate > 0.20:
            sev = "BLOCKER"
        elif null_rate > max_null_rate:
            sev = "WARNING"
        else:
            sev = "PASS"

        if sev == "BLOCKER":
            worst_severity = "BLOCKER"
        elif sev == "WARNING" and worst_severity != "BLOCKER":
            worst_severity = "WARNING"

        column_stats.append({
            "name": col,
            "null_count": null_count,
            "null_rate": round(null_rate, 4),
            "severity": sev,
        })

    overall_null_rate = (
        sum(s["null_count"] for s in column_stats) / (total_rows * len(columns))
        if total_rows > 0 and len(columns) > 0 else 0.0
    )

    return {
        "ok": worst_severity == "PASS",
        "column_stats": column_stats,
        "overall_null_rate": round(overall_null_rate, 4),
        "overall_severity": worst_severity,
    }


def validate_date_range(
    df: pl.DataFrame,
    date_column: str,
    expected_start=None,
    expected_end=None,
    max_gap_days: Optional[int] = None,
) -> dict:
    """Validate temporal coverage and detect date gaps.

    Args:
        df: Polars DataFrame.
        date_column: Name of the date column.
        expected_start: Expected start date (optional).
        expected_end: Expected end date (optional).
        max_gap_days: Maximum allowed gap in days between consecutive dates.

    Returns:
        dict with keys: ``ok``, ``actual_start``, ``actual_end``, ``gaps``,
        ``issues``.
    """
    issues: list[str] = []
    gaps: list[dict] = []

    if date_column not in df.columns:
        return {"ok": False, "actual_start": None, "actual_end": None,
                "gaps": [], "issues": [f"Column '{date_column}' not found"]}

    dates = df[date_column].drop_nulls().unique().sort()
    if dates.is_empty():
        return {"ok": False, "actual_start": None, "actual_end": None,
                "gaps": [], "issues": ["No non-null dates found"]}

    actual_start = dates[0]
    actual_end = dates[-1]

    if expected_start is not None and actual_start > expected_start:
        issues.append(
            f"Data starts at {actual_start}, expected {expected_start}"
        )

    if expected_end is not None and actual_end < expected_end:
        issues.append(
            f"Data ends at {actual_end}, expected {expected_end}"
        )

    # Gap detection
    if max_gap_days is not None and len(dates) > 1:
        diffs = dates.diff().drop_nulls()
        for i, d in enumerate(diffs):
            gap_days = d.days if hasattr(d, "days") else d.total_seconds() / 86400
            if gap_days > max_gap_days:
                gaps.append({
                    "start": str(dates[i]),
                    "end": str(dates[i + 1]),
                    "gap_days": int(gap_days),
                })

    if gaps:
        issues.append(f"Found {len(gaps)} gaps exceeding {max_gap_days} days")

    return {
        "ok": len(issues) == 0,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "gaps": gaps,
        "issues": issues,
    }


def validate_referential_integrity(
    df_child: pl.DataFrame,
    df_parent: pl.DataFrame,
    child_key: str,
    parent_key: str,
) -> dict:
    """Check that all child key values exist in the parent table.

    Args:
        df_child: Child DataFrame.
        df_parent: Parent DataFrame.
        child_key: Column name in child.
        parent_key: Column name in parent.

    Returns:
        dict with keys: ``ok``, ``orphan_count``, ``orphan_rate``,
        ``orphan_sample``, ``severity``.
    """
    child_vals = df_child[child_key].unique()
    parent_vals = set(df_parent[parent_key].unique().to_list())

    orphans = [v for v in child_vals.to_list() if v not in parent_vals]
    orphan_count = len(orphans)
    total = child_vals.len()
    orphan_rate = orphan_count / total if total > 0 else 0.0

    if orphan_rate > 0.05:
        severity = "BLOCKER"
    elif orphan_count > 0:
        severity = "WARNING"
    else:
        severity = "PASS"

    return {
        "ok": orphan_count == 0,
        "orphan_count": orphan_count,
        "orphan_rate": round(orphan_rate, 4),
        "orphan_sample": orphans[:10],
        "severity": severity,
    }


def validate_row_count(
    df: pl.DataFrame,
    min_rows: int = 1,
    max_rows: Optional[int] = None,
) -> dict:
    """Check that the DataFrame has an acceptable number of rows.

    Args:
        df: Polars DataFrame.
        min_rows: Minimum required rows.
        max_rows: Maximum allowed rows (optional).

    Returns:
        dict with keys: ``ok``, ``row_count``, ``message``.
    """
    n = len(df)
    issues = []

    if n < min_rows:
        issues.append(f"Row count {n} below minimum {min_rows}")
    if max_rows is not None and n > max_rows:
        issues.append(f"Row count {n} above maximum {max_rows}")

    return {
        "ok": len(issues) == 0,
        "row_count": n,
        "message": issues[0] if issues else "OK",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_structural_checks(
    df: pl.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run all configured structural checks and aggregate results.

    Args:
        df: Polars DataFrame.
        config: Dict controlling which checks to run. Supported keys:
            - ``expected_columns``: list[str]
            - ``expected_types``: dict[str, pl.DataType]
            - ``primary_key``: list[str]
            - ``required_columns``: list[str]
            - ``completeness_threshold``: float
            - ``date_column``: str
            - ``expected_start``, ``expected_end``: date-like
            - ``max_gap_days``: int
            - ``min_rows``, ``max_rows``: int

    Returns:
        dict with keys: ``ok``, ``checks_run``, ``checks_passed``,
        ``checks_failed``, ``details``.
    """
    config = config or {}
    details: Dict[str, dict] = {}

    # Schema check (always runs)
    details["schema"] = validate_schema(
        df,
        expected_columns=config.get("expected_columns"),
        expected_types=config.get("expected_types"),
    )

    # Primary key
    if "primary_key" in config:
        details["primary_key"] = validate_primary_key(
            df, config["primary_key"],
        )

    # Completeness
    details["completeness"] = validate_completeness(
        df,
        required_columns=config.get("required_columns"),
        threshold=config.get("completeness_threshold", 0.95),
    )

    # Date range
    if "date_column" in config:
        details["date_range"] = validate_date_range(
            df,
            date_column=config["date_column"],
            expected_start=config.get("expected_start"),
            expected_end=config.get("expected_end"),
            max_gap_days=config.get("max_gap_days"),
        )

    # Row count
    details["row_count"] = validate_row_count(
        df,
        min_rows=config.get("min_rows", 1),
        max_rows=config.get("max_rows"),
    )

    checks_run = len(details)
    checks_passed = sum(1 for r in details.values() if r.get("ok", True))
    checks_failed = checks_run - checks_passed
    overall_ok = all(r.get("ok", True) for r in details.values())

    return {
        "ok": overall_ok,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "details": details,
    }
