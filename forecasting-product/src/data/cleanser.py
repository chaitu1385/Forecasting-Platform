"""
Demand cleansing — outlier detection, stockout imputation, period exclusion.

Operates on Polars DataFrames with per-series logic.  All detection
thresholds and correction actions are driven by ``CleansingConfig``.
"""

from dataclasses import dataclass
from datetime import date
from typing import Tuple

import polars as pl

from ..config.schema import CleansingConfig, get_frequency_profile
from ..utils.calendar import PeriodCalendar, advance_date

# --------------------------------------------------------------------------- #
#  Result types
# --------------------------------------------------------------------------- #

@dataclass
class CleansingReport:
    """Summary of what the cleanser found and changed."""
    total_series: int
    series_with_outliers: int
    total_outliers: int
    outlier_pct: float
    series_with_stockouts: int
    total_stockout_periods: int
    total_stockout_weeks: int
    excluded_period_weeks: int
    rows_modified: int
    per_series: pl.DataFrame  # [series_id, outliers, stockouts, modified_rows]


@dataclass
class CleansingResult:
    """Cleaned DataFrame plus audit report."""
    df: pl.DataFrame
    report: CleansingReport


# --------------------------------------------------------------------------- #
#  DemandCleanser
# --------------------------------------------------------------------------- #

class DemandCleanser:
    """
    Config-driven demand cleansing for weekly time series.

    Usage
    -----
    >>> from src.config.schema import CleansingConfig
    >>> cfg = CleansingConfig(enabled=True, outlier_method="iqr")
    >>> cleanser = DemandCleanser(cfg)
    >>> result = cleanser.cleanse(df, "week", "quantity", "series_id")
    >>> cleaned = result.df
    >>> print(result.report)
    """

    def __init__(self, config: CleansingConfig, frequency: str = "W"):
        self.config = config
        get_frequency_profile(frequency)
        self.frequency = frequency

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def cleanse(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> CleansingResult:
        """Run all enabled cleansing steps and return cleaned data + report."""
        if df.is_empty():
            return CleansingResult(
                df=df,
                report=self._empty_report(sid_col),
            )

        original = df.clone()
        df = df.sort([sid_col, time_col])

        # Step 1: Period exclusion (before outlier detection so excluded
        # periods don't skew IQR / z-score statistics).
        excluded_weeks = 0
        if self.config.exclude_periods:
            df, excluded_weeks = self._apply_period_exclusion(
                df, time_col, value_col, sid_col,
            )

        # Step 2: Outlier detection + correction
        df = self.detect_outliers(df, time_col, value_col, sid_col)
        df = self._correct_outliers(df, time_col, value_col, sid_col)

        # Step 3: Stockout detection + imputation
        if self.config.stockout_detection:
            df = self.detect_stockouts(df, time_col, value_col, sid_col)
            df = self._impute_stockouts(df, time_col, value_col, sid_col)

        # Build report
        report = self._build_report(original, df, sid_col, value_col, excluded_weeks)

        # Optionally strip flag columns
        if not self.config.add_flag_columns:
            flag_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_flag")]
            df = df.drop(flag_cols)

        return CleansingResult(df=df, report=report)

    # ------------------------------------------------------------------ #
    #  Outlier detection
    # ------------------------------------------------------------------ #

    def detect_outliers(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> pl.DataFrame:
        """Flag outliers per series.  Adds ``_outlier_flag`` column."""
        method = self.config.outlier_method

        if method == "iqr":
            return self._detect_outliers_iqr(df, value_col, sid_col)
        elif method == "zscore":
            return self._detect_outliers_zscore(df, value_col, sid_col)
        else:
            raise ValueError(f"Unknown outlier_method: {method!r}")

    def _detect_outliers_iqr(
        self, df: pl.DataFrame, value_col: str, sid_col: str,
    ) -> pl.DataFrame:
        k = self.config.iqr_multiplier
        stats = df.group_by(sid_col).agg(
            pl.col(value_col).quantile(0.25).alias("_q1"),
            pl.col(value_col).quantile(0.75).alias("_q3"),
        ).with_columns(
            (pl.col("_q3") - pl.col("_q1")).alias("_iqr"),
        ).with_columns(
            (pl.col("_q1") - k * pl.col("_iqr")).alias("_lower"),
            (pl.col("_q3") + k * pl.col("_iqr")).alias("_upper"),
        )

        df = df.join(stats, on=sid_col, how="left")
        df = df.with_columns(
            ((pl.col(value_col) < pl.col("_lower"))
             | (pl.col(value_col) > pl.col("_upper"))).alias("_outlier_flag"),
        )
        return df.drop(["_q1", "_q3", "_iqr", "_lower", "_upper"])

    def _detect_outliers_zscore(
        self, df: pl.DataFrame, value_col: str, sid_col: str,
    ) -> pl.DataFrame:
        threshold = self.config.zscore_threshold
        stats = df.group_by(sid_col).agg(
            pl.col(value_col).mean().alias("_mean"),
            pl.col(value_col).std().alias("_std"),
        )
        df = df.join(stats, on=sid_col, how="left")
        df = df.with_columns(
            pl.when(pl.col("_std") > 0)
            .then(((pl.col(value_col) - pl.col("_mean")) / pl.col("_std")).abs())
            .otherwise(0.0)
            .alias("_zscore"),
        )
        df = df.with_columns(
            (pl.col("_zscore") > threshold).alias("_outlier_flag"),
        )
        return df.drop(["_mean", "_std", "_zscore"])

    # ------------------------------------------------------------------ #
    #  Outlier correction
    # ------------------------------------------------------------------ #

    def _correct_outliers(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> pl.DataFrame:
        action = self.config.outlier_action
        if action == "flag_only" or "_outlier_flag" not in df.columns:
            return df

        if action == "clip":
            return self._clip_outliers(df, value_col, sid_col)
        elif action == "interpolate":
            return self._interpolate_flagged(
                df, time_col, value_col, sid_col, "_outlier_flag",
            )
        else:
            raise ValueError(f"Unknown outlier_action: {action!r}")

    def _clip_outliers(
        self, df: pl.DataFrame, value_col: str, sid_col: str,
    ) -> pl.DataFrame:
        """Winsorize outliers to IQR fence bounds."""
        k = self.config.iqr_multiplier
        stats = df.group_by(sid_col).agg(
            pl.col(value_col).quantile(0.25).alias("_q1"),
            pl.col(value_col).quantile(0.75).alias("_q3"),
        ).with_columns(
            (pl.col("_q3") - pl.col("_q1")).alias("_iqr"),
        ).with_columns(
            (pl.col("_q1") - k * pl.col("_iqr")).alias("_lower"),
            (pl.col("_q3") + k * pl.col("_iqr")).alias("_upper"),
        )
        df = df.join(stats, on=sid_col, how="left")
        df = df.with_columns(
            pl.when(pl.col("_outlier_flag"))
            .then(pl.col(value_col).clip(pl.col("_lower"), pl.col("_upper")))
            .otherwise(pl.col(value_col))
            .alias(value_col),
        )
        return df.drop(["_q1", "_q3", "_iqr", "_lower", "_upper"])

    # ------------------------------------------------------------------ #
    #  Stockout detection
    # ------------------------------------------------------------------ #

    def detect_stockouts(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> pl.DataFrame:
        """
        Flag stockout periods: consecutive-zero runs followed by recovery.

        A run of zeros at the tail of the series is NOT flagged (could be
        end-of-life or seasonal closure).
        """
        min_run = self.config.min_zero_run
        df = df.sort([sid_col, time_col])

        # Mark zeros
        df = df.with_columns(
            (pl.col(value_col) == 0).alias("_is_zero"),
        )

        # Build run-length group ids per series
        df = df.with_columns(
            (pl.col("_is_zero") != pl.col("_is_zero").shift(1).over(sid_col))
            .fill_null(True)
            .cum_sum()
            .over(sid_col)
            .alias("_run_group"),
        )

        # Compute run lengths and whether each run is a zero-run
        run_info = df.group_by([sid_col, "_run_group"]).agg(
            pl.col("_is_zero").first().alias("_is_zero_run"),
            pl.col("_is_zero").count().alias("_run_len"),
            pl.col(time_col).max().alias("_run_end"),
        )

        # For each series, find the last time point
        series_max = df.group_by(sid_col).agg(
            pl.col(time_col).max().alias("_series_end"),
        )
        run_info = run_info.join(series_max, on=sid_col, how="left")

        # A stockout run: is_zero, length >= min_run, and doesn't end at series tail
        run_info = run_info.with_columns(
            (
                pl.col("_is_zero_run")
                & (pl.col("_run_len") >= min_run)
                & (pl.col("_run_end") < pl.col("_series_end"))
            ).alias("_is_stockout"),
        )

        stockout_groups = run_info.filter(pl.col("_is_stockout")).select(
            sid_col, "_run_group",
        ).with_columns(pl.lit(True).alias("_stockout_flag"))

        df = df.join(stockout_groups, on=[sid_col, "_run_group"], how="left")
        df = df.with_columns(
            pl.col("_stockout_flag").fill_null(False),
        )
        return df.drop(["_is_zero", "_run_group"])

    # ------------------------------------------------------------------ #
    #  Stockout imputation
    # ------------------------------------------------------------------ #

    def _impute_stockouts(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> pl.DataFrame:
        method = self.config.stockout_imputation
        if method == "none" or "_stockout_flag" not in df.columns:
            return df

        if method == "seasonal":
            return self._impute_seasonal(df, time_col, value_col, sid_col, self.frequency)
        elif method == "interpolate":
            return self._interpolate_flagged(
                df, time_col, value_col, sid_col, "_stockout_flag",
            )
        else:
            raise ValueError(f"Unknown stockout_imputation: {method!r}")

    def _impute_seasonal(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
        frequency: str = "W",
    ) -> pl.DataFrame:
        """Look up the configured prior season by date, including on raw/gapped data."""
        sl = get_frequency_profile(frequency)["season_length"]
        if df.is_empty():
            return df
        dates = df[time_col].unique().sort().to_list()
        if frequency in ("M", "Q"):
            PeriodCalendar.from_dates(dates, frequency)
        # A fixed D/W lookup need not impose a model grid on diagnostic data.
        # For example, the legacy audit uses a weekly lookback on daily rows.
        lookup_columns = {
            time_col: dates,
            "_lookup_date": [advance_date(value, frequency, -sl) for value in dates],
            "_previous_date": [advance_date(value, frequency, -1) for value in dates],
            "_next_date": [advance_date(value, frequency, 1) for value in dates],
        }
        offsets = pl.DataFrame(
            lookup_columns,
            schema_overrides={column: df[time_col].dtype for column in lookup_columns},
        )
        df = df.join(offsets, on=time_col, validate="m:1")

        lookup = df.select(
            pl.col(sid_col),
            pl.col(time_col).alias("_lookup_date"),
            pl.col(value_col).alias("_prior_year_val"),
        )

        df = df.join(lookup, on=[sid_col, "_lookup_date"], how="left")

        # Fall back to ±1 period neighbors from prior year
        lookup_minus1 = df.select(
            pl.col(sid_col),
            pl.col("_next_date").alias("_lookup_date"),
            pl.col(value_col).alias("_py_minus1"),
        )
        lookup_plus1 = df.select(
            pl.col(sid_col),
            pl.col("_previous_date").alias("_lookup_date"),
            pl.col(value_col).alias("_py_plus1"),
        )

        df = df.join(
            lookup_minus1.select(sid_col, "_lookup_date", "_py_minus1"),
            on=[sid_col, "_lookup_date"],
            how="left",
        ).join(
            lookup_plus1.select(sid_col, "_lookup_date", "_py_plus1"),
            on=[sid_col, "_lookup_date"],
            how="left",
        )

        # Use prior year, else average of neighbors, else keep original
        df = df.with_columns(
            pl.when(pl.col("_stockout_flag") & pl.col("_prior_year_val").is_not_null() & (pl.col("_prior_year_val") > 0))
            .then(pl.col("_prior_year_val"))
            .when(pl.col("_stockout_flag") & (pl.col("_py_minus1").is_not_null() | pl.col("_py_plus1").is_not_null()))
            .then(
                (pl.col("_py_minus1").fill_null(0.0) + pl.col("_py_plus1").fill_null(0.0))
                / (pl.col("_py_minus1").is_not_null().cast(pl.Float64) + pl.col("_py_plus1").is_not_null().cast(pl.Float64)).clip(1, 2)
            )
            .otherwise(pl.col(value_col))
            .alias(value_col),
        )

        return df.drop(["_previous_date", "_next_date", "_lookup_date", "_prior_year_val", "_py_minus1", "_py_plus1"])

    # ------------------------------------------------------------------ #
    #  Shared interpolation helper
    # ------------------------------------------------------------------ #

    def _interpolate_flagged(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
        flag_col: str,
    ) -> pl.DataFrame:
        """Replace flagged values with linear interpolation per series."""
        df = df.sort([sid_col, time_col])
        df = df.with_columns(
            pl.when(pl.col(flag_col))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(value_col))
            .alias("_interp_src"),
        )
        df = df.with_columns(
            pl.col("_interp_src").interpolate().over(sid_col).alias("_interp_val"),
        )
        # Forward/backward fill for edges
        df = df.with_columns(
            pl.col("_interp_val").forward_fill().over(sid_col).alias("_interp_val"),
        )
        df = df.with_columns(
            pl.col("_interp_val").backward_fill().over(sid_col).alias("_interp_val"),
        )
        df = df.with_columns(
            pl.when(pl.col(flag_col))
            .then(pl.col("_interp_val"))
            .otherwise(pl.col(value_col))
            .alias(value_col),
        )
        return df.drop(["_interp_src", "_interp_val"])

    # ------------------------------------------------------------------ #
    #  Period exclusion
    # ------------------------------------------------------------------ #

    def _apply_period_exclusion(
        self,
        df: pl.DataFrame,
        time_col: str,
        value_col: str,
        sid_col: str,
    ) -> Tuple[pl.DataFrame, int]:
        """Apply configured period exclusions.  Returns (df, excluded_week_count)."""
        total_excluded = 0

        for period in self.config.exclude_periods:
            start = date.fromisoformat(period["start"])
            end = date.fromisoformat(period["end"])
            action = period.get("action", "flag")

            mask = (pl.col(time_col) >= start) & (pl.col(time_col) <= end)
            n_excluded = df.filter(mask).height
            total_excluded += n_excluded

            if action == "drop":
                df = df.filter(~mask)

            elif action == "interpolate":
                df = df.with_columns(
                    mask.alias("_excluded_flag"),
                )
                df = self._interpolate_flagged(
                    df, time_col, value_col, sid_col, "_excluded_flag",
                )

            elif action == "flag":
                if "_excluded_flag" not in df.columns:
                    df = df.with_columns(pl.lit(False).alias("_excluded_flag"))
                df = df.with_columns(
                    pl.when(mask)
                    .then(pl.lit(True))
                    .otherwise(pl.col("_excluded_flag"))
                    .alias("_excluded_flag"),
                )

        return df, total_excluded

    # ------------------------------------------------------------------ #
    #  Report generation
    # ------------------------------------------------------------------ #

    def _build_report(
        self,
        original: pl.DataFrame,
        cleaned: pl.DataFrame,
        sid_col: str,
        value_col: str,
        excluded_weeks: int,
    ) -> CleansingReport:
        total_series = cleaned[sid_col].n_unique()

        # Outlier stats
        if "_outlier_flag" in cleaned.columns:
            outlier_count = cleaned.filter(pl.col("_outlier_flag")).height
            series_w_outliers = cleaned.filter(pl.col("_outlier_flag"))[sid_col].n_unique()
        else:
            outlier_count = 0
            series_w_outliers = 0

        # Stockout stats
        if "_stockout_flag" in cleaned.columns:
            stockout_weeks = cleaned.filter(pl.col("_stockout_flag")).height
            series_w_stockouts = cleaned.filter(pl.col("_stockout_flag"))[sid_col].n_unique()
            # Count distinct stockout periods (contiguous runs)
            cleaned.filter(pl.col("_stockout_flag")).sort([sid_col, "week"] if "week" in cleaned.columns else [sid_col])
            stockout_periods = series_w_stockouts  # approximate: at least 1 per series
        else:
            stockout_weeks = 0
            series_w_stockouts = 0
            stockout_periods = 0

        # Rows modified (value changed)
        rows_modified = 0
        if original.height == cleaned.height:
            orig_vals = original.sort([sid_col])[value_col]
            clean_vals = cleaned.sort([sid_col])[value_col]
            if orig_vals.dtype == pl.Float64 and clean_vals.dtype == pl.Float64:
                rows_modified = int((orig_vals != clean_vals).sum())

        total_rows = max(cleaned.height, 1)
        outlier_pct = round(outlier_count / total_rows * 100, 2)

        # Per-series breakdown
        [pl.col(sid_col)]
        if "_outlier_flag" in cleaned.columns:
            agg_exprs_inner = [pl.col("_outlier_flag").sum().alias("outliers")]
        else:
            agg_exprs_inner = [pl.lit(0).alias("outliers")]
        if "_stockout_flag" in cleaned.columns:
            agg_exprs_inner.append(pl.col("_stockout_flag").sum().alias("stockouts"))
        else:
            agg_exprs_inner.append(pl.lit(0).alias("stockouts"))

        per_series = cleaned.group_by(sid_col).agg(agg_exprs_inner)

        return CleansingReport(
            total_series=total_series,
            series_with_outliers=series_w_outliers,
            total_outliers=outlier_count,
            outlier_pct=outlier_pct,
            series_with_stockouts=series_w_stockouts,
            total_stockout_periods=stockout_periods,
            total_stockout_weeks=stockout_weeks,
            excluded_period_weeks=excluded_weeks,
            rows_modified=rows_modified,
            per_series=per_series,
        )

    def _empty_report(self, sid_col: str) -> CleansingReport:
        return CleansingReport(
            total_series=0,
            series_with_outliers=0,
            total_outliers=0,
            outlier_pct=0.0,
            series_with_stockouts=0,
            total_stockout_periods=0,
            total_stockout_weeks=0,
            excluded_period_weeks=0,
            rows_modified=0,
            per_series=pl.DataFrame({sid_col: [], "outliers": [], "stockouts": []}),
        )
