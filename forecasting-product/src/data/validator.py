"""
Data validation — centralized schema enforcement for forecasting input data.

Validates column presence, types, duplicates, frequency, value ranges,
and completeness before any processing occurs in the pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from ..config.schema import ValidationConfig
from ..utils.calendar import PeriodCalendar

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """A single validation finding."""
    level: str                               # "error" | "warning"
    check: str                               # e.g. "schema", "duplicates", "frequency"
    message: str
    series_id: Optional[str] = None          # None = dataset-wide issue
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Aggregated result of all validation checks."""
    passed: bool
    issues: List[ValidationIssue]
    n_rows: int
    n_series: int
    duplicate_count: int = 0
    negative_count: int = 0
    missing_column_names: List[str] = field(default_factory=list)
    frequency_violations: int = 0

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


class DataValidator:
    """
    Validates input DataFrames against the expected forecasting schema.

    Checks run in order: schema → duplicates → frequency → value range →
    completeness.  Each check appends issues to the report.  An issue is
    an *error* (blocks pipeline) or a *warning* (informational).  When
    ``strict=True`` in the config, all warnings are promoted to errors.

    Parameters
    ----------
    config : ValidationConfig
        Toggles for individual checks and thresholds.
    frequency : str
        Expected data frequency (``"D"``, ``"W"``, ``"M"``, ``"Q"``).
        Used by :meth:`check_frequency` to determine the expected gap
        between observations.  Defaults to ``"W"`` for backward
        compatibility, but callers should pass the configured frequency.
    """

    def __init__(self, config: ValidationConfig, frequency: str = "W"):
        self.config = config
        self.frequency = frequency

    def validate(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> ValidationReport:
        """Run all enabled checks and return a consolidated report."""
        issues: List[ValidationIssue] = []
        n_rows = len(df)
        n_series = df[id_col].n_unique() if id_col in df.columns else 0
        duplicate_count = 0
        negative_count = 0
        missing_cols: List[str] = []
        freq_violations = 0

        # 1. Schema check (always runs)
        schema_issues, missing_cols = self.check_schema(
            df, target_col, time_col, id_col
        )
        issues.extend(schema_issues)

        # If critical columns are missing or have wrong types, skip remaining checks
        has_schema_errors = any(i.level == "error" for i in schema_issues)
        if not has_schema_errors:
            # 2. Duplicates
            if self.config.check_duplicates:
                dup_issues, duplicate_count = self.check_duplicates(
                    df, time_col, id_col
                )
                issues.extend(dup_issues)

            # 3. Frequency
            if self.config.check_frequency:
                freq_issues, freq_violations = self.check_frequency(
                    df, time_col, id_col, frequency=self.frequency
                )
                issues.extend(freq_issues)

            # 4. Value range
            range_issues, negative_count = self.check_value_range(
                df, target_col
            )
            issues.extend(range_issues)

            # 5. Completeness
            comp_issues = self.check_completeness(df, time_col, id_col, target_col)
            issues.extend(comp_issues)

        # Strict mode: promote warnings → errors
        if self.config.strict:
            for issue in issues:
                if issue.level == "warning":
                    issue.level = "error"

        passed = all(i.level != "error" for i in issues)

        return ValidationReport(
            passed=passed,
            issues=issues,
            n_rows=n_rows,
            n_series=n_series,
            duplicate_count=duplicate_count,
            negative_count=negative_count,
            missing_column_names=missing_cols,
            frequency_violations=freq_violations,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_schema(
        self,
        df: pl.DataFrame,
        target_col: str,
        time_col: str,
        id_col: str,
    ) -> Tuple[List[ValidationIssue], List[str]]:
        """Validate required columns exist and have correct types.

        Returns (issues, missing_column_names).
        """
        issues: List[ValidationIssue] = []
        missing: List[str] = []

        # Required columns: time, target, id + any extras from config
        required = [time_col, target_col, id_col] + list(
            self.config.require_columns
        )

        for col in required:
            if col not in df.columns:
                missing.append(col)
                issues.append(ValidationIssue(
                    level="error",
                    check="schema",
                    message=f"Required column '{col}' not found. Available: {df.columns}",
                ))

        # Type checks (only if columns exist)
        type_map = {
            time_col: (pl.Date, pl.Datetime),
            target_col: (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8),
            id_col: (pl.Utf8, pl.Categorical),
        }
        for col, expected_types in type_map.items():
            if col in df.columns and df[col].dtype not in expected_types:
                issues.append(ValidationIssue(
                    level="error",
                    check="schema",
                    message=(
                        f"Column '{col}' has type {df[col].dtype}, "
                        f"expected one of {expected_types}"
                    ),
                ))

        return issues, missing

    def check_duplicates(
        self,
        df: pl.DataFrame,
        time_col: str,
        id_col: str,
    ) -> Tuple[List[ValidationIssue], int]:
        """Detect duplicate (id_col, time_col) pairs.

        Returns (issues, duplicate_count).
        """
        issues: List[ValidationIssue] = []
        total = len(df)
        unique = df.select([id_col, time_col]).unique().height
        dup_count = total - unique

        if dup_count > 0:
            issues.append(ValidationIssue(
                level="error",
                check="duplicates",
                message=(
                    f"Found {dup_count} duplicate ({id_col}, {time_col}) "
                    f"rows out of {total} total"
                ),
                details={"duplicate_count": dup_count},
            ))

        return issues, dup_count

    def check_frequency(
        self,
        df: pl.DataFrame,
        time_col: str,
        id_col: str,
        frequency: str = "W",
    ) -> Tuple[List[ValidationIssue], int]:
        """Validate consistent intervals per series for the given frequency.

        Parameters
        ----------
        frequency:
            Expected data frequency — ``"D"``, ``"W"``, ``"M"``, ``"Q"``.

        Returns (issues, violation_count).
        """
        issues: List[ValidationIssue] = []

        violating_series = []
        for group in df.partition_by(id_col):
            dates = group[time_col].sort().to_list()
            try:
                calendar = PeriodCalendar.from_dates(dates, frequency)
                regular = dates == calendar.date_range(dates[0], dates[-1])
            except ValueError:
                regular = False
            if not regular:
                violating_series.append(group[id_col][0])
        if not violating_series:
            return issues, 0

        violation_count = len(violating_series)
        issues.append(ValidationIssue(
            level="warning",
            check="frequency",
            message=(
                f"{violation_count} series have inconsistent gaps "
                f"(expected an exact anchored calendar grid for frequency={frequency!r})"
            ),
            details={
                "violation_count": violation_count,
                "example_series": violating_series[:5],
            },
        ))

        return issues, violation_count

    def check_value_range(
        self,
        df: pl.DataFrame,
        target_col: str,
    ) -> Tuple[List[ValidationIssue], int]:
        """Check for values outside allowed range.

        Returns (issues, negative_count).
        """
        issues: List[ValidationIssue] = []
        negative_count = 0

        # Non-negative check
        if self.config.check_non_negative and self.config.min_value is None:
            neg = df.filter(pl.col(target_col) < 0)
            negative_count = len(neg)
            if negative_count > 0:
                issues.append(ValidationIssue(
                    level="error",
                    check="value_range",
                    message=f"Found {negative_count} negative values in '{target_col}'",
                    details={"negative_count": negative_count},
                ))

        # Custom min bound
        if self.config.min_value is not None:
            below = df.filter(pl.col(target_col) < self.config.min_value)
            count = len(below)
            if count > 0:
                negative_count = count
                issues.append(ValidationIssue(
                    level="error",
                    check="value_range",
                    message=(
                        f"Found {count} values below min_value "
                        f"{self.config.min_value} in '{target_col}'"
                    ),
                    details={"below_min_count": count},
                ))

        # Custom max bound
        if self.config.max_value is not None:
            above = df.filter(pl.col(target_col) > self.config.max_value)
            count = len(above)
            if count > 0:
                issues.append(ValidationIssue(
                    level="warning",
                    check="value_range",
                    message=(
                        f"Found {count} values above max_value "
                        f"{self.config.max_value} in '{target_col}'"
                    ),
                    details={"above_max_count": count},
                ))

        return issues, negative_count

    def check_completeness(
        self,
        df: pl.DataFrame,
        time_col: str,
        id_col: str,
        target_col: Optional[str] = None,
    ) -> List[ValidationIssue]:
        """Check series count and per-series missing-week percentage."""
        issues: List[ValidationIssue] = []

        n_series = df[id_col].n_unique()

        # Minimum series count
        if n_series < self.config.min_series_count:
            issues.append(ValidationIssue(
                level="error",
                check="completeness",
                message=(
                    f"Found {n_series} series, "
                    f"minimum required is {self.config.min_series_count}"
                ),
            ))

        counts = df.group_by(id_col).agg(
            pl.len().alias("_n_periods"),
        )
        if self.config.min_history_periods is not None:
            short = counts.filter(
                pl.col("_n_periods") < self.config.min_history_periods
            )
            if not short.is_empty():
                issues.append(ValidationIssue(
                    level="warning",
                    check="completeness",
                    message=(
                        f"{len(short)} series have fewer than "
                        f"{self.config.min_history_periods} history periods"
                    ),
                ))

        if self.config.max_zero_pct < 100.0:
            zero_limit = (
                self.config.max_zero_pct * 100.0
                if self.config.max_zero_pct <= 1.0
                else self.config.max_zero_pct
            )
            zero_rates = df.group_by(id_col).agg(
                (pl.col(target_col).eq(0).mean() * 100.0).alias("_zero_pct")
            ) if target_col and target_col in df.columns else None
            if zero_rates is not None:
                zero_violators = zero_rates.filter(pl.col("_zero_pct") > zero_limit)
                if not zero_violators.is_empty():
                    issues.append(ValidationIssue(
                        level="warning",
                        check="completeness",
                        message=f"{len(zero_violators)} series exceed {zero_limit}% zero values",
                    ))

        # Per-series missing-week percentage
        if self.config.max_missing_pct < 100.0:
            min_date = df[time_col].min()
            max_date = df[time_col].max()
            if min_date is not None and max_date is not None:
                elapsed_days = (max_date - min_date).days
                if self.frequency == "D":
                    total_periods = elapsed_days + 1
                elif self.frequency == "W":
                    total_periods = elapsed_days // 7 + 1
                elif self.frequency == "M":
                    total_periods = (max_date.year - min_date.year) * 12 + max_date.month - min_date.month + 1
                else:
                    total_periods = ((max_date.year - min_date.year) * 12 + max_date.month - min_date.month) // 3 + 1
                if total_periods > 0:
                    series_counts = df.group_by(id_col).agg(
                        pl.col(time_col).count().alias("_n_weeks")
                    )
                    series_counts = series_counts.with_columns(
                        ((1.0 - pl.col("_n_weeks") / total_periods) * 100.0)
                        .alias("_missing_pct")
                    )
                    violators = series_counts.filter(
                        pl.col("_missing_pct") > (
                            self.config.max_missing_pct * 100.0
                            if self.config.max_missing_pct <= 1.0
                            else self.config.max_missing_pct
                        )
                    )
                    if len(violators) > 0:
                        worst_pct = violators["_missing_pct"].max()
                        issues.append(ValidationIssue(
                            level="warning",
                            check="completeness",
                            message=(
                                f"{len(violators)} series exceed "
                                f"{self.config.max_missing_pct}% missing weeks "
                                f"(worst: {worst_pct:.1f}%)"
                            ),
                            details={
                                "violating_series_count": len(violators),
                                "worst_missing_pct": round(float(worst_pct), 1),
                            },
                        ))

        return issues
