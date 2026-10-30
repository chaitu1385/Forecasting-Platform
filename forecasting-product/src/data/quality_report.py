"""
Pre-training data quality report.

Consolidates completeness, distribution, and demand-pattern diagnostics
into a single ``DataQualityReport`` that surfaces actionable warnings
before model training begins.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import polars as pl

from ..config.schema import PlatformConfig
from ..utils.calendar import PeriodCalendar

# --------------------------------------------------------------------------- #
#  Report dataclass
# --------------------------------------------------------------------------- #

@dataclass
class DataQualityReport:
    """Data quality summary; legacy week-named fields count configured periods."""

    # Overall
    total_rows: int = 0
    total_series: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    total_weeks: int = 0

    # Completeness
    missing_week_pct: float = 0.0
    series_with_gaps: int = 0
    short_series_count: int = 0
    zero_series_count: int = 0

    # Value distribution
    zero_inflation_rate: float = 0.0
    outlier_count: int = 0
    outlier_pct: float = 0.0

    # Demand patterns (SBC classification)
    demand_classes: Dict[str, int] = field(default_factory=dict)

    # Structural breaks
    series_with_breaks: int = 0
    total_breaks: int = 0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Per-series detail (optional)
    per_series: Optional[pl.DataFrame] = None


# --------------------------------------------------------------------------- #
#  Analyzer
# --------------------------------------------------------------------------- #

class DataQualityAnalyzer:
    """
    Runs pre-training quality checks and returns a ``DataQualityReport``.

    Designed to run **after** gap-fill and cleansing but **before** the
    short-series / zero-series drops so the report captures what will be
    filtered out.
    """

    def __init__(self, config: PlatformConfig):
        self.config = config

    def analyze(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
        cleansing_report=None,
        break_report=None,
    ) -> DataQualityReport:
        """
        Run all quality checks.

        Parameters
        ----------
        df : pl.DataFrame
            Gap-filled (and optionally cleansed) panel data.
        time_col, value_col, sid_col :
            Column names for time, target value, and series id.
        cleansing_report :
            Optional ``CleansingReport`` from a prior cleansing step.
            When provided, outlier counts are pulled from it instead of
            re-running detection.
        break_report :
            Optional ``BreakReport`` from structural break detection.
            When provided, break counts are included in the report.
        """
        if df.is_empty():
            return DataQualityReport(warnings=["Empty dataset"])

        dq = self.config.data_quality
        report = DataQualityReport()

        # ---- Step 1: Basic stats ----------------------------------------- #
        report.total_rows = len(df)
        report.total_series = df[sid_col].n_unique()
        report.date_range_start = df[time_col].min()
        report.date_range_end = df[time_col].max()

        if report.date_range_start is not None and report.date_range_end is not None:
            calendar = PeriodCalendar.from_dates(
                df[time_col].unique().to_list(), self.config.forecast.frequency,
            )
            report.total_weeks = calendar.index(report.date_range_end) + 1

        # ---- Step 2: Completeness ---------------------------------------- #
        per_series_weeks = (
            df.group_by(sid_col)
            .agg(pl.col(time_col).n_unique().cast(pl.Int64).alias("n_weeks"))
        )

        expected_weeks = report.total_weeks
        if expected_weeks > 0:
            gaps = per_series_weeks.with_columns(
                (expected_weeks - pl.col("n_weeks")).clip(lower_bound=0).alias("missing")
            )
            total_expected = expected_weeks * report.total_series
            total_missing = gaps["missing"].sum()
            report.missing_week_pct = (
                (total_missing / total_expected * 100) if total_expected > 0 else 0.0
            )
            report.series_with_gaps = int(gaps.filter(pl.col("missing") > 0).height)

        # Short series
        min_len = dq.effective_min_series_length(self.config.forecast.frequency)
        if min_len > 0:
            report.short_series_count = int(
                per_series_weeks.filter(pl.col("n_weeks") < min_len).height
            )

        # All-zero series
        zero_sums = (
            df.group_by(sid_col)
            .agg(pl.col(value_col).abs().sum().alias("_total"))
        )
        report.zero_series_count = int(
            zero_sums.filter(pl.col("_total") == 0).height
        )

        # ---- Step 3: Value distribution ---------------------------------- #
        report.zero_inflation_rate = (
            df.filter(pl.col(value_col) == 0).height / report.total_rows * 100
            if report.total_rows > 0 else 0.0
        )

        if cleansing_report is not None:
            report.outlier_count = cleansing_report.total_outliers
            report.outlier_pct = cleansing_report.outlier_pct
        else:
            report.outlier_count, report.outlier_pct = self._detect_outliers_iqr(
                df, value_col, sid_col
            )

        # ---- Step 4: Demand classification ------------------------------- #
        report_cfg = dq.report
        if report_cfg.sparse_classification and report.total_series > 0:
            report.demand_classes = self._classify_demand(df, value_col, sid_col)

        # ---- Step 4b: Structural breaks ---------------------------------- #
        if break_report is not None:
            report.series_with_breaks = break_report.series_with_breaks
            report.total_breaks = break_report.total_breaks

        # ---- Step 5: Per-series detail ----------------------------------- #
        if report_cfg.include_series_detail:
            report.per_series = self._build_per_series(
                df, time_col, value_col, sid_col, expected_weeks,
                report.demand_classes,
            )

        # ---- Step 6: Warnings -------------------------------------------- #
        report.warnings = self._generate_warnings(report, min_len)

        return report

    # ----- helpers -------------------------------------------------------- #

    @staticmethod
    def _detect_outliers_iqr(
        df: pl.DataFrame, value_col: str, sid_col: str
    ) -> Tuple[int, float]:
        """Lightweight per-series IQR outlier count (detect only)."""
        stats = (
            df.group_by(sid_col)
            .agg([
                pl.col(value_col).quantile(0.25).alias("q1"),
                pl.col(value_col).quantile(0.75).alias("q3"),
            ])
        )
        stats = stats.with_columns(
            ((pl.col("q3") - pl.col("q1")) * 1.5).alias("fence")
        )
        enriched = df.join(stats, on=sid_col, how="left")
        outlier_mask = (
            (pl.col(value_col) < (pl.col("q1") - pl.col("fence")))
            | (pl.col(value_col) > (pl.col("q3") + pl.col("fence")))
        )
        count = int(enriched.filter(outlier_mask).height)
        total = len(df)
        pct = count / total * 100 if total > 0 else 0.0
        return count, pct

    def _classify_demand(
        self, df: pl.DataFrame, value_col: str, sid_col: str
    ) -> Dict[str, int]:
        """Run SBC demand classification via SparseDetector."""
        from ..series.sparse_detector import SparseDetector

        fc = self.config.forecast
        detector = SparseDetector(
            adi_threshold=fc.sparse_adi_threshold,
            cv2_threshold=fc.sparse_cv2_threshold,
        )
        classified = detector.classify(df, target_col=value_col, id_col=sid_col)
        counts = (
            classified.group_by("demand_class")
            .agg(pl.len().alias("count"))
        )
        return {
            row["demand_class"]: row["count"]
            for row in counts.iter_rows(named=True)
        }

    @staticmethod
    def _build_per_series(
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
        expected_weeks: int,
        demand_classes: Dict[str, int],
    ) -> pl.DataFrame:
        """Build per-series diagnostic table."""
        agg = (
            df.group_by(sid_col)
            .agg([
                pl.col(time_col).n_unique().cast(pl.Int64).alias("n_weeks"),
                pl.col(value_col).mean().alias("mean"),
                pl.col(value_col).std().alias("std"),
                (pl.col(value_col) == 0).mean().alias("zero_pct"),
            ])
        )
        if expected_weeks > 0:
            agg = agg.with_columns(
                ((expected_weeks - pl.col("n_weeks")).clip(lower_bound=0) / expected_weeks * 100)
                .alias("missing_pct")
            )
        else:
            agg = agg.with_columns(pl.lit(0.0).alias("missing_pct"))

        # Coefficient of variation
        agg = agg.with_columns(
            pl.when(pl.col("mean") > 0)
            .then(pl.col("std") / pl.col("mean"))
            .otherwise(0.0)
            .alias("cv")
        )

        # Add demand class if classification was run
        if demand_classes:

            # Re-classify per series to get per-row labels
            # (demand_classes dict only has totals)
            # We join via the sparse detector output
            pass  # demand_class is added below if available

        return agg

    @staticmethod
    def _generate_warnings(report: DataQualityReport, min_len: int) -> List[str]:
        """Generate actionable warning messages."""
        warnings: List[str] = []

        if report.missing_week_pct > 20:
            warnings.append(
                f"High gap rate: {report.missing_week_pct:.1f}% of series\u00d7period cells were missing"
            )
        if report.zero_inflation_rate > 30:
            warnings.append(
                f"High zero-inflation: {report.zero_inflation_rate:.1f}% of values are zero"
            )
        if report.short_series_count > 0:
            warnings.append(
                f"{report.short_series_count} series shorter than {min_len} periods will be dropped"
            )
        if report.zero_series_count > 0:
            warnings.append(
                f"{report.zero_series_count} all-zero series detected"
            )
        if report.outlier_pct > 5:
            warnings.append(
                f"High outlier rate: {report.outlier_pct:.1f}% of values flagged as outliers"
            )
        if report.series_with_breaks > 0:
            warnings.append(
                f"{report.series_with_breaks} series have structural breaks "
                f"({report.total_breaks} total break points)"
            )
        return warnings
