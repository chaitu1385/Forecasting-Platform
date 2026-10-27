"""Champion and ensemble selection on validated, comparable raw observations."""

import logging
from typing import Dict, List, Optional

import polars as pl

from ..config.schema import BacktestConfig, HorizonBucket
from ..metrics.aggregation import aggregate_metrics, rank_metrics, validate_comparable_records
from ..metrics.definitions import METRIC_DIRECTIONS

logger = logging.getLogger(__name__)


class ChampionSelector:
    """Rank volume errors (or an explicitly configured alternative) per group."""

    def __init__(self, config: BacktestConfig):
        self.primary_metric = config.primary_metric
        self.secondary_metric = config.secondary_metric
        self.granularity = config.champion_granularity
        for name in (self.primary_metric, self.secondary_metric):
            if name not in METRIC_DIRECTIONS:
                raise ValueError(f"Unknown selection metric {name!r}")

    def _group_column(self, records: pl.DataFrame, override: Optional[str]) -> Optional[str]:
        if override is not None:
            if override not in records.columns:
                raise ValueError(f"Champion grouping column {override!r} is missing")
            return override
        column = {"lob": "lob", "series": "series_id", "product_group": "product_group"}.get(self.granularity)
        if column in records.columns:
            return column
        if self.granularity == "product_group":
            raise ValueError("Product-group selection requires a product_group column or granularity_col")
        return None

    def select(
        self, backtest_results: pl.DataFrame, granularity_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """Select champions from complete raw keys, not averaged cached scores.

        WMAPE pools error and absolute volume across the selected series/folds.
        For equal-(series, fold) weighting, explicitly use ``macro_wmape``.
        Signed bias is ranked by distance from zero in either metric position.
        """
        if backtest_results.is_empty():
            return pl.DataFrame(schema={
                "group_key": pl.String, "model_id": pl.String,
                self.primary_metric: pl.Float64, self.secondary_metric: pl.Float64,
            })
        group_col = self._group_column(backtest_results, granularity_col)
        groups = [group_col] if group_col else []
        board = aggregate_metrics(
            backtest_results, [self.primary_metric, self.secondary_metric], groups,
        )
        if board[self.primary_metric].null_count() or not board[self.primary_metric].is_finite().all():
            raise ValueError(
                f"Insufficient data: {self.primary_metric} is undefined for the requested population"
            )
        board = rank_metrics(board, self.primary_metric, self.secondary_metric, groups)
        if group_col:
            champions = board.group_by(group_col, maintain_order=True).first().rename({group_col: "group_key"})
        else:
            champions = board.head(1).with_columns(pl.lit("global").alias("group_key"))
        logger.info("Champion selection (%s): %d group(s)", self.granularity, champions.height)
        return champions

    def select_by_horizon(
        self,
        backtest_results: pl.DataFrame,
        horizon_buckets: List[HorizonBucket],
        granularity_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """Recompute every bucket from its own errors and complete lead coverage."""
        if not horizon_buckets:
            return self.select(backtest_results, granularity_col)
        group_col = self._group_column(backtest_results, granularity_col)
        validate_comparable_records(backtest_results, [group_col] if group_col else [])
        seen_names, seen_steps = set(), set()
        champions = []
        for bucket in horizon_buckets:
            steps = set(range(bucket.start_step, bucket.end_step + 1))
            if bucket.start_step < 1 or not steps or bucket.name in seen_names or steps & seen_steps:
                raise ValueError("Horizon buckets require unique names and nonoverlapping positive step ranges")
            seen_names.add(bucket.name)
            seen_steps.update(steps)
            subset = backtest_results.filter(pl.col("forecast_step").is_between(bucket.start_step, bucket.end_step))
            if subset.is_empty():
                raise ValueError(f"Insufficient data: no forecast coverage for horizon bucket {bucket.name!r}")
            units = ["run_id", "lob", "fold", "series_id", "model_id"]
            original_units = backtest_results.select(units).unique().height
            counts = subset.group_by(units).agg(pl.col("forecast_step").n_unique().alias("_steps"))
            if counts.height != original_units or counts.filter(pl.col("_steps") != len(steps)).height:
                raise ValueError(f"Incomplete forecast coverage for horizon bucket {bucket.name!r}")
            chosen = self.select(subset, granularity_col).with_columns(
                pl.lit(bucket.name).alias("horizon_bucket"),
                pl.lit(bucket.start_step).alias("start_step"),
                pl.lit(bucket.end_step).alias("end_step"),
            )
            champions.append(chosen)
        return pl.concat(champions)

    def compute_ensemble_weights(self, backtest_results: pl.DataFrame) -> Dict[str, float]:
        """Inverse-primary-error weights over the same validated population.

        Bias uses absolute aggregate bias; perfect-error models share all weight.
        Undefined primary metrics cannot silently turn into perfect scores.
        """
        if backtest_results.is_empty():
            return {}
        board = aggregate_metrics(backtest_results, [self.primary_metric])
        values = {}
        for row in board.iter_rows(named=True):
            value = row[self.primary_metric]
            if value is None:
                raise ValueError(f"Insufficient data: {self.primary_metric} is undefined for ensemble weighting")
            values[row["model_id"]] = abs(value)
        perfect = [name for name, value in values.items() if value == 0.0]
        if perfect:
            return {name: 1.0 / len(perfect) if name in perfect else 0.0 for name in values}
        raw = {name: 1.0 / value for name, value in values.items()}
        total = sum(raw.values())
        return {name: weight / total for name, weight in raw.items()}
