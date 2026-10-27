"""
Series builder — constructs model-ready time series from raw actuals.

Responsibilities:
  1. Validate and align raw data to the expected schema.
  2. Apply product transition stitching (via TransitionEngine).
  3. Build composite series IDs from hierarchy columns.
  4. Fill gaps (missing weeks) with zeros or interpolation.
  5. Return a clean DataFrame ready for forecasting.
"""

import logging
from datetime import date
from typing import Optional

import polars as pl

from ..config.schema import PlatformConfig
from ..utils.gap_fill import fill_gaps as _shared_fill_gaps
from .transition import TransitionEngine

logger = logging.getLogger(__name__)


class SeriesBuilder:
    """
    Constructs model-ready time series.

    Usage
    -----
    >>> builder = SeriesBuilder(config)
    >>> series = builder.build(actuals, product_master, mapping_table)
    """

    def __init__(self, config: PlatformConfig):
        self.config = config
        self._frequency = config.forecast.frequency
        self._transition_engine = TransitionEngine(config.transition, frequency=config.forecast.frequency)
        self._last_validation_report = None
        self._last_cleansing_report = None
        self._last_break_report = None
        self._last_quality_report = None
        self._last_regressor_screen_report = None

    def normalized_input_snapshot(self, actuals: pl.DataFrame) -> pl.DataFrame:
        """Preserve raw values and add only the configured composite series ID."""
        return self._ensure_series_id(actuals.clone(), self.config.forecast.series_id_column)

    def build(
        self,
        actuals: pl.DataFrame,
        external_features: Optional[pl.DataFrame] = None,
        product_master: Optional[pl.DataFrame] = None,
        mapping_table: Optional[pl.DataFrame] = None,
        forecast_origin: Optional[date] = None,
        overrides: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        Build model-ready time series from raw actuals.

        Parameters
        ----------
        actuals:
            Raw weekly actuals.  Must contain the time column, value column,
            and hierarchy identifier columns.
        product_master:
            Product metadata with launch_date.  Required for transitions.
        mapping_table:
            SKU mapping output.  Required for transitions.
        forecast_origin:
            The forecast start date.  Required for transitions.
        overrides:
            Planner overrides for transitions.

        Returns
        -------
        Model-ready DataFrame with columns: series_id, week, quantity
        (plus any extra feature columns).
        """
        fc = self.config.forecast
        time_col = fc.time_column
        value_col = fc.target_column
        sid_col = fc.series_id_column

        # Step 0: Schema validation (before anything else)
        dq = self.config.data_quality
        self._last_cleansing_report = None
        self._last_validation_report = None
        if dq.validation.enabled:
            from ..data.validator import DataValidator
            validator = DataValidator(dq.validation, frequency=fc.frequency)
            self._last_validation_report = validator.validate(
                actuals, value_col, time_col, sid_col
            )
            for issue in self._last_validation_report.issues:
                if issue.level == "error":
                    logger.error("Validation: %s", issue.message)
                else:
                    logger.warning("Validation: %s", issue.message)
            if not self._last_validation_report.passed:
                raise ValueError(
                    f"Data validation failed with "
                    f"{len(self._last_validation_report.errors)} error(s). "
                    f"First: {self._last_validation_report.errors[0].message}"
                )

        # Step 1: Build composite series ID if not present
        df = self._ensure_series_id(actuals, sid_col)

        # Step 2: Apply transitions
        if (
            mapping_table is not None
            and product_master is not None
            and forecast_origin is not None
        ):
            plans = self._transition_engine.compute_plans(
                mapping_table=mapping_table,
                product_master=product_master,
                forecast_origin=forecast_origin,
                horizon_weeks=fc.horizon_weeks,
                overrides=overrides,
            )
            df = self._transition_engine.stitch_series(
                actuals=df,
                plans=plans,
                time_column=time_col,
                series_id_column=sid_col,
                value_column=value_col,
            )

        # Step 3: Fill missing weeks (controlled by data_quality config)
        if dq.fill_gaps:
            df = self._fill_gaps(df, time_col, sid_col, value_col, dq.fill_value)

        # Step 3-breaks: Structural break detection (after gap-fill, before cleansing)
        self._last_break_report = None
        if dq.structural_breaks.enabled:
            from .break_detector import StructuralBreakDetector
            detector = StructuralBreakDetector(dq.structural_breaks)
            self._last_break_report = detector.detect(df, value_col, time_col, sid_col)
            for w in self._last_break_report.warnings:
                logger.warning("Structural break: %s", w)
            if dq.structural_breaks.truncate_to_last_break:
                df = detector.truncate(
                    df, self._last_break_report, time_col, sid_col
                )
                logger.info(
                    "Truncated %d series to post-break data",
                    self._last_break_report.series_with_breaks,
                )

        # Step 3-cleanse: Demand cleansing (after gap-fill, before filtering)
        if dq.cleansing.enabled:
            from ..data.cleanser import DemandCleanser
            cleanser = DemandCleanser(dq.cleansing, frequency=fc.frequency)
            result = cleanser.cleanse(df, time_col, value_col, sid_col)
            df = result.df
            self._last_cleansing_report = result.report
            logger.info(
                "Cleansing: %d outliers in %d series, %d stockout periods",
                result.report.total_outliers,
                result.report.series_with_outliers,
                result.report.total_stockout_periods,
            )

        # Step 3-report: Data quality report (after gap-fill + cleanse, before drops)
        self._last_quality_report = None
        if dq.report.enabled:
            from ..data.quality_report import DataQualityAnalyzer
            analyzer = DataQualityAnalyzer(self.config)
            self._last_quality_report = analyzer.analyze(
                df, time_col, value_col, sid_col,
                cleansing_report=self._last_cleansing_report,
                break_report=self._last_break_report,
            )
            for w in self._last_quality_report.warnings:
                logger.warning("Data quality: %s", w)

        # Step 3a: Drop short series
        effective_min_len = dq.effective_min_series_length(fc.frequency)
        if effective_min_len > 0:
            series_lengths = (
                df.group_by(sid_col)
                .agg(pl.col(time_col).count().alias("_len"))
            )
            valid_ids = series_lengths.filter(
                pl.col("_len") >= effective_min_len
            )[sid_col].to_list()
            df = df.filter(pl.col(sid_col).is_in(valid_ids))

        # Step 3b: Drop all-zero series
        if dq.drop_zero_series:
            nonzero = (
                df.group_by(sid_col)
                .agg(pl.col(value_col).abs().sum().alias("_total"))
                .filter(pl.col("_total") > 0)
            )[sid_col].to_list()
            df = df.filter(pl.col(sid_col).is_in(nonzero))

        # Step 3c: Join external features (if provided and enabled)
        if external_features is not None and self.config.forecast.external_regressors.enabled:
            feature_cols = self.config.forecast.external_regressors.feature_columns
            available_cols = [c for c in feature_cols if c in external_features.columns]
            if available_cols:
                join_cols = [time_col]
                select_cols = [time_col] + available_cols

                # If features have series_id, join on both; otherwise broadcast
                if sid_col in external_features.columns:
                    join_cols = [sid_col, time_col]
                    select_cols = [sid_col, time_col] + available_cols

                df = df.join(
                    external_features.select(select_cols),
                    on=join_cols,
                    how="left",
                )

                # Fill nulls in feature columns with 0
                for col in available_cols:
                    df = df.with_columns(
                        pl.col(col).fill_null(0).alias(col)
                    )

        # Step 3d: Regressor screening (after join, before model fit)
        self._last_regressor_screen_report = None
        if (
            external_features is not None
            and self.config.forecast.external_regressors.enabled
            and self.config.forecast.external_regressors.screen.enabled
        ):
            from ..data.regressor_screen import screen_regressors
            screen_cfg = self.config.forecast.external_regressors.screen
            screened_cols = [
                c for c in self.config.forecast.external_regressors.feature_columns
                if c in df.columns
            ]
            if screened_cols:
                report = screen_regressors(df, screened_cols, value_col, screen_cfg)
                self._last_regressor_screen_report = report
                for w in report.warnings:
                    logger.warning("Regressor screen: %s", w)
                if screen_cfg.auto_drop and report.dropped_columns:
                    cols_to_drop = [c for c in report.dropped_columns if c in df.columns]
                    if cols_to_drop:
                        df = df.drop(cols_to_drop)
                        logger.info(
                            "Dropped %d low-quality regressors: %s",
                            len(cols_to_drop), cols_to_drop,
                        )

        # Step 4: Sort
        df = df.sort([sid_col, time_col])

        return df

    def _ensure_series_id(
        self, df: pl.DataFrame, sid_col: str
    ) -> pl.DataFrame:
        """
        If series_id column doesn't exist, build it from the output grain
        columns defined in config.
        """
        if sid_col in df.columns:
            return df

        grain = self.config.output.grain
        grain_cols = list(grain.values())
        missing = [c for c in grain_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Cannot build series_id: columns {missing} not in data. "
                f"Available: {df.columns}"
            )

        # Concatenate grain columns with "|" separator
        concat_expr = pl.concat_str(
            [pl.col(c).cast(pl.Utf8) for c in grain_cols],
            separator="|",
        ).alias(sid_col)

        return df.with_columns(concat_expr)

    def _fill_gaps(
        self,
        df: pl.DataFrame,
        time_col: str,
        sid_col: str,
        value_col: str,
        fill_value: float = 0.0,
    ) -> pl.DataFrame:
        """Fill missing periods for each series using shared utility."""
        freq = getattr(self, "_frequency", "W")
        return _shared_fill_gaps(
            df,
            time_col=time_col,
            id_col=sid_col,
            target_col=value_col,
            fill_value=fill_value,
            strategy="zero",
            freq=freq,
        )
