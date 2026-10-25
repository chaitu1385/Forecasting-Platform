"""
Layer 3 — Business Rules Validation for forecasting data.

Domain-specific checks: value ranges, metric relationships, temporal
consistency (max period-over-period change), segment coverage, cardinality
limits, and configurable rule sets.

Ships with forecasting-specific defaults (e.g. WMAPE ∈ [0, 2], bias ∈
[-1, 1], capacity ceilings) that can be overridden per dataset.

All functions accept Polars DataFrames.

Usage:
    from src.validation.business_rules import (
        validate_ranges, validate_metric_relationships,
        validate_temporal_consistency, validate_business_rules,
        get_default_rules,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import polars as pl

# ---------------------------------------------------------------------------
# Default forecasting business rules
# ---------------------------------------------------------------------------

# Range rules: column → (min, max)
DEFAULT_RANGE_RULES: Dict[str, tuple] = {
    "wmape": (0.0, 2.0),
    "mape": (0.0, 10.0),
    "bias": (-1.0, 1.0),
    "quantity": (0.0, None),  # non-negative
    "forecast": (0.0, None),
    "price": (0.0, None),
}

# Metric relationship rules: (col_a, operator, col_b)
DEFAULT_RELATIONSHIP_RULES: list[dict] = [
    # safety_stock should not exceed average demand
    {"description": "safety_stock <= avg_demand",
     "col_a": "safety_stock", "op": "<=", "col_b": "avg_demand"},
]

# Maximum period-over-period change (relative)
DEFAULT_TEMPORAL_RULES: Dict[str, float] = {
    "quantity": 5.0,   # 500% max change
    "forecast": 5.0,
    "revenue": 10.0,
}

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_ranges(
    df: pl.DataFrame,
    rules: Optional[Dict[str, tuple]] = None,
) -> dict:
    """Check that column values fall within expected ranges.

    Args:
        df: Polars DataFrame.
        rules: Dict mapping ``column_name -> (min_val, max_val)``.
               None in either bound means unbounded.

    Returns:
        dict with ``ok``, ``violations`` (list of per-column dicts).
    """
    rules = rules or DEFAULT_RANGE_RULES
    violations: list[dict] = []

    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue

        vals = df[col].drop_nulls()
        if vals.is_empty():
            continue

        below = vals.filter(vals < lo).len() if lo is not None else 0
        above = vals.filter(vals > hi).len() if hi is not None else 0

        if below > 0 or above > 0:
            violations.append({
                "column": col,
                "min_rule": lo,
                "max_rule": hi,
                "below_count": below,
                "above_count": above,
                "actual_min": float(vals.min()),
                "actual_max": float(vals.max()),
            })

    severity = "BLOCKER" if violations else "PASS"
    return {"ok": len(violations) == 0, "violations": violations,
            "severity": severity}


def validate_metric_relationships(
    df: pl.DataFrame,
    rules: Optional[list[dict]] = None,
) -> dict:
    """Check inter-column relationships.

    Args:
        df: Polars DataFrame.
        rules: List of dicts with ``col_a``, ``op``, ``col_b``, ``description``.

    Returns:
        dict with ``ok``, ``violations``.
    """
    rules = rules or DEFAULT_RELATIONSHIP_RULES
    violations: list[dict] = []

    for rule in rules:
        col_a = rule["col_a"]
        col_b = rule["col_b"]
        op = rule["op"]
        desc = rule.get("description", f"{col_a} {op} {col_b}")

        if col_a not in df.columns or col_b not in df.columns:
            continue

        op_fn = _OPS.get(op)
        if op_fn is None:
            continue

        # Drop rows where either is null
        valid = df.filter(
            pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null()
        )

        # Evaluate using Polars expressions for the comparison
        if op == "<=":
            bad = valid.filter(pl.col(col_a) > pl.col(col_b))
        elif op == "<":
            bad = valid.filter(pl.col(col_a) >= pl.col(col_b))
        elif op == ">=":
            bad = valid.filter(pl.col(col_a) < pl.col(col_b))
        elif op == ">":
            bad = valid.filter(pl.col(col_a) <= pl.col(col_b))
        elif op == "==":
            bad = valid.filter(pl.col(col_a) != pl.col(col_b))
        elif op == "!=":
            bad = valid.filter(pl.col(col_a) == pl.col(col_b))
        else:
            continue

        if bad.height > 0:
            violations.append({
                "rule": desc,
                "violation_count": bad.height,
                "sample": bad.head(3).to_dicts(),
            })

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "severity": "WARNING" if violations else "PASS",
    }


def validate_temporal_consistency(
    df: pl.DataFrame,
    date_column: str,
    value_columns: Optional[Dict[str, float]] = None,
    group_col: Optional[str] = None,
) -> dict:
    """Check for unrealistic period-over-period changes.

    Args:
        df: Polars DataFrame sorted by date.
        date_column: Date column.
        value_columns: Dict ``column → max_relative_change``.  Default uses
            ``DEFAULT_TEMPORAL_RULES``.
        group_col: Run per-group.

    Returns:
        dict with ``ok``, ``spikes`` (list of dicts).
    """
    value_columns = value_columns or DEFAULT_TEMPORAL_RULES
    spikes: list[dict] = []

    def _check(sub: pl.DataFrame, grp_name=None):
        sorted_df = sub.sort(date_column)
        for col, max_change in value_columns.items():
            if col not in sorted_df.columns:
                continue
            vals = sorted_df[col].cast(pl.Float64)
            diffs = vals.diff().drop_nulls()
            shifted = vals.shift(1).drop_nulls()
            for i in range(len(diffs)):
                base = shifted[i]
                if base is None or base == 0:
                    continue
                change = abs(diffs[i] / base)
                if change > max_change:
                    spikes.append({
                        "group": grp_name,
                        "column": col,
                        "period_index": i + 1,
                        "date": str(sorted_df[date_column][i + 1]),
                        "value": float(vals[i + 1]),
                        "prev_value": float(base),
                        "relative_change": round(float(change), 2),
                    })

    if group_col:
        for g in df[group_col].unique().to_list():
            _check(df.filter(pl.col(group_col) == g), grp_name=g)
    else:
        _check(df)

    return {
        "ok": len(spikes) == 0,
        "spikes": spikes[:30],
        "severity": "WARNING" if spikes else "PASS",
    }


def validate_segment_coverage(
    df: pl.DataFrame,
    segment_col: str,
    expected_segments: Optional[Sequence[str]] = None,
    min_rows_per_segment: int = 1,
) -> dict:
    """Check that expected segments are present with sufficient data.

    Args:
        df: Polars DataFrame.
        segment_col: Segment column.
        expected_segments: Optional list of expected segment values.
        min_rows_per_segment: Minimum rows per segment.

    Returns:
        dict with ``ok``, ``missing_segments``, ``undersized_segments``.
    """
    if segment_col not in df.columns:
        return {
            "ok": False,
            "missing_segments": expected_segments or [],
            "undersized_segments": [],
            "severity": "BLOCKER",
        }

    actual = df.group_by(segment_col).agg(pl.col(segment_col).len().alias("n"))
    actual_set = set(actual[segment_col].to_list())

    missing = []
    if expected_segments:
        missing = [s for s in expected_segments if s not in actual_set]

    undersized = actual.filter(pl.col("n") < min_rows_per_segment)
    undersized_list = undersized[segment_col].to_list()

    ok = len(missing) == 0 and undersized.height == 0
    severity = "BLOCKER" if missing else ("WARNING" if undersized_list else "PASS")

    return {
        "ok": ok,
        "missing_segments": missing,
        "undersized_segments": undersized_list,
        "severity": severity,
    }


def validate_no_negative(
    df: pl.DataFrame,
    columns: Sequence[str],
) -> dict:
    """Check that specified columns contain no negative values.

    Args:
        df: Polars DataFrame.
        columns: Columns to check.

    Returns:
        dict with ``ok``, ``violations``.
    """
    violations: list[dict] = []
    for col in columns:
        if col not in df.columns:
            continue
        neg = df.filter(pl.col(col) < 0)
        if neg.height > 0:
            violations.append({
                "column": col,
                "negative_count": neg.height,
                "min_value": float(df[col].min()),
            })

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "severity": "BLOCKER" if violations else "PASS",
    }


def validate_cardinality(
    df: pl.DataFrame,
    column: str,
    max_cardinality: Optional[int] = None,
    min_cardinality: int = 1,
) -> dict:
    """Check that column has expected cardinality (distinct values).

    Args:
        df: Polars DataFrame.
        column: Column to check.
        max_cardinality: Max allowed distinct values.
        min_cardinality: Min required distinct values.

    Returns:
        dict with ``ok``, ``cardinality``.
    """
    if column not in df.columns:
        return {"ok": False, "cardinality": 0, "severity": "BLOCKER",
                "message": f"Column '{column}' not found"}

    card = df[column].n_unique()
    issues = []
    if card < min_cardinality:
        issues.append(f"Cardinality {card} below minimum {min_cardinality}")
    if max_cardinality is not None and card > max_cardinality:
        issues.append(f"Cardinality {card} above maximum {max_cardinality}")

    return {
        "ok": len(issues) == 0,
        "cardinality": card,
        "issues": issues,
        "severity": "WARNING" if issues else "PASS",
    }


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_default_rules() -> dict:
    """Return a copy of all default forecasting business rules."""
    return {
        "range_rules": dict(DEFAULT_RANGE_RULES),
        "relationship_rules": list(DEFAULT_RELATIONSHIP_RULES),
        "temporal_rules": dict(DEFAULT_TEMPORAL_RULES),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def validate_business_rules(
    df: pl.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run all configured business rule checks.

    Args:
        df: Polars DataFrame.
        config: Controls which checks to run:
            - ``range_rules``: dict[str, tuple]
            - ``relationship_rules``: list[dict]
            - ``date_column``: str (for temporal consistency)
            - ``temporal_rules``: dict[str, float]
            - ``group_col``: str
            - ``segment_col``: str
            - ``expected_segments``: list[str]
            - ``non_negative_columns``: list[str]

    Returns:
        dict with ``ok``, ``checks_run``, ``checks_passed``,
        ``checks_failed``, ``details``.
    """
    config = config or {}
    details: Dict[str, dict] = {}

    # Range checks (always)
    details["ranges"] = validate_ranges(
        df, rules=config.get("range_rules"),
    )

    # Metric relationships
    if config.get("relationship_rules"):
        details["metric_relationships"] = validate_metric_relationships(
            df, rules=config["relationship_rules"],
        )

    # Temporal consistency
    if config.get("date_column"):
        details["temporal_consistency"] = validate_temporal_consistency(
            df,
            date_column=config["date_column"],
            value_columns=config.get("temporal_rules"),
            group_col=config.get("group_col"),
        )

    # Segment coverage
    if config.get("segment_col"):
        details["segment_coverage"] = validate_segment_coverage(
            df,
            segment_col=config["segment_col"],
            expected_segments=config.get("expected_segments"),
        )

    # Non-negative
    if config.get("non_negative_columns"):
        details["no_negative"] = validate_no_negative(
            df, columns=config["non_negative_columns"],
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
