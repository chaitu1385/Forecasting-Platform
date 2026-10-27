"""
FVA aggregation and reporting.

Takes per-series, per-fold FVA detail data and produces summary tables
for BI consumption and stakeholder reporting.
"""

import logging
from typing import List, Optional

import polars as pl

from ..metrics.aggregation import aggregate_metrics, rank_metrics, validate_comparable_records
from ..metrics.fva import classify_fva

logger = logging.getLogger(__name__)


class FVAAnalyzer:
    """
    Aggregate FVA results across series, folds, and LOBs.

    Usage
    -----
    >>> analyzer = FVAAnalyzer()
    >>> detail = analyzer.compute_fva_detail(backtest_results)
    >>> summary = analyzer.summarize(detail)
    >>> leaderboard = analyzer.layer_leaderboard(detail)
    """

    LAYER_NAMES = {
        "naive_seasonal": "naive",
        "seasonal_naive": "naive",  # Historical metric ID, not a new forecasting plugin.
        "auto_arima": "statistical",
        "auto_ets": "statistical",
        "croston": "statistical",
        "croston_sba": "statistical",
        "tsb": "statistical",
        "lgbm_direct": "ml",
        "xgboost_direct": "ml",
    }

    def classify_model_layer(self, model_id: str) -> str:
        """Map a model_id to its FVA layer name."""
        return self.LAYER_NAMES.get(model_id, "ml")

    def compute_fva_detail(
        self,
        backtest_results: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "target_week",
    ) -> pl.DataFrame:
        """
        Compute per-series, per-fold FVA from backtest results.

        The backtest_results DataFrame must have columns:
        model_id, fold, series_id, target_week, actual, forecast, wmape

        Returns a DataFrame with FVA metrics per (series, fold, layer).
        """
        if backtest_results.is_empty():
            return pl.DataFrame()

        required_cols = {"model_id", "fold", id_col, time_col, "actual", "forecast"}
        missing = required_cols - set(backtest_results.columns)
        if missing:
            raise ValueError(f"Missing columns for FVA: {missing}")

        aliases = {source: canonical for source, canonical in
                   ((id_col, "series_id"), (time_col, "target_week")) if source != canonical}
        df = backtest_results.rename(aliases)
        if "run_id" not in df.columns:
            raise ValueError("FVA requires raw run-scoped observations; re-run summary-only backtests")
        competition = ["lob", "series_id"] + [
            column for column in ("grain_level", "channel") if column in df.columns
        ]
        unit = ["run_id"] + competition + ["fold"]
        results = []
        for run in df.partition_by("run_id"):
            # Validate entire model populations before selecting within a layer.
            # A missing hard period/fold cannot become the best "statistical" score.
            validate_comparable_records(run, competition)
            models = aggregate_metrics(run, ["wmape", "normalized_bias"], unit)
            volumes = run.group_by(unit + ["model_id"]).agg(
                pl.col("actual").abs().sum().alias("total_volume")
            )
            models = models.join(volumes, on=unit + ["model_id"]).with_columns(
                pl.col("model_id").map_elements(
                    self.classify_model_layer, return_dtype=pl.String,
                ).alias("forecast_layer")
            )
            best = rank_metrics(models, "wmape", "normalized_bias", unit + ["forecast_layer"])
            best = best.group_by(unit + ["forecast_layer"], maintain_order=True).first()
            for identity, group in best.group_by(unit, maintain_order=True):
                common = dict(zip(unit, identity))
                layers = {row["forecast_layer"]: row for row in group.iter_rows(named=True)}
                available = [name for name in ("naive", "statistical", "ml") if name in layers]
                for index, layer in enumerate(available):
                    row = layers[layer]
                    parent = available[index - 1] if index else None
                    parent_wmape = layers[parent]["wmape"] if parent else None
                    value = row["wmape"]
                    if value is None or (parent is not None and parent_wmape is None):
                        fva, classification = None, "UNDEFINED"
                    elif parent is None:
                        fva, classification = 0.0, "BASELINE"
                    else:
                        fva = parent_wmape - value
                        classification = classify_fva(fva)
                    results.append({
                        **common, "forecast_layer": layer, "parent_layer": parent,
                        "model_id": row["model_id"] if value is not None else None,
                        "wmape": value, "parent_wmape": parent_wmape,
                        "normalized_bias": row["normalized_bias"],
                        "fva_wmape": fva, "fva_class": classification,
                        "total_volume": row["total_volume"],
                        "n_observations": row["n_observations"],
                        "scoring_policy": row["scoring_policy"],
                    })

        if not results:
            return pl.DataFrame()

        detail = pl.DataFrame(results, schema_overrides={
            "wmape": pl.Float64, "parent_wmape": pl.Float64,
            "normalized_bias": pl.Float64, "fva_wmape": pl.Float64,
            "model_id": pl.String, "parent_layer": pl.String,
        })
        if id_col != "series_id":
            detail = detail.with_columns(pl.col("series_id").alias(id_col))
        return detail

    def summarize(
        self,
        fva_detail: pl.DataFrame,
        group_by: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        Aggregate FVA detail into a summary table.

        Default grouping is by forecast_layer. Optionally group by
        additional columns (e.g. LOB, sparse_class).

        Returns one row per (group x layer) with:
        - mean_wmape, mean_fva_wmape
        - pct_adds_value, pct_neutral, pct_destroys_value
        - total_volume
        """
        if fva_detail.is_empty():
            return pl.DataFrame()

        group_cols = list(dict.fromkeys(
            [column for column in ("run_id", "lob") if column in fva_detail.columns]
            + (group_by or []) + ["forecast_layer"]
        ))

        summary = (
            fva_detail
            .group_by(group_cols)
            .agg([
                pl.col("series_id").n_unique().alias("n_series"),
                pl.col("forecast_layer").count().alias("n_evaluations"),
                (pl.col("fva_class") == "UNDEFINED").sum().alias("n_undefined"),
                pl.when(pl.col("wmape").is_not_null().all())
                    .then(pl.col("wmape").mean()).otherwise(None).alias("mean_wmape"),
                pl.when(pl.col("fva_wmape").is_not_null().all())
                    .then(pl.col("fva_wmape").mean()).otherwise(None).alias("mean_fva_wmape"),
                pl.when(pl.col("fva_wmape").is_not_null().all())
                    .then(pl.col("fva_wmape").sum()).otherwise(None).alias("total_fva_wmape"),
                (pl.col("fva_class") == "ADDS_VALUE")
                    .sum().alias("n_adds_value"),
                (pl.col("fva_class") == "NEUTRAL")
                    .sum().alias("n_neutral"),
                (pl.col("fva_class") == "DESTROYS_VALUE")
                    .sum().alias("n_destroys_value"),
                pl.col("total_volume").sum().alias("total_volume"),
            ])
        )

        # Compute percentages
        summary = summary.with_columns([
            (pl.col("n_adds_value") / pl.col("n_evaluations") * 100)
                .round(1).alias("pct_adds_value"),
            (pl.col("n_neutral") / pl.col("n_evaluations") * 100)
                .round(1).alias("pct_neutral"),
            (pl.col("n_destroys_value") / pl.col("n_evaluations") * 100)
                .round(1).alias("pct_destroys_value"),
        ])

        return summary.sort("forecast_layer")

    def layer_leaderboard(
        self, fva_detail: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Rank layers by aggregate FVA contribution.

        Returns one row per layer with:
        - rank (1 = most value added)
        - cumulative WMAPE reduction
        - robustness score (% of series improved)
        - recommendation (Keep / Review / Remove)
        """
        if fva_detail.is_empty():
            return pl.DataFrame()

        # Skip baseline layer for ranking
        ranked_data = fva_detail.filter(pl.col("parent_layer").is_not_null())

        if ranked_data.is_empty():
            return pl.DataFrame()

        population = [column for column in ("run_id", "lob") if column in fva_detail.columns]
        board = (
            ranked_data
            .group_by(population + ["forecast_layer"])
            .agg([
                pl.when(pl.col("fva_wmape").is_not_null().all())
                    .then(pl.col("fva_wmape").mean()).otherwise(None).alias("mean_fva_wmape"),
                pl.when(pl.col("fva_wmape").is_not_null().all())
                    .then(pl.col("fva_wmape").sum()).otherwise(None).alias("cumulative_wmape_reduction"),
                pl.col("series_id").n_unique().alias("n_series"),
                pl.col("forecast_layer").count().alias("n_evaluations"),
                (pl.col("fva_class") == "UNDEFINED").sum().alias("n_undefined"),
                (pl.col("fva_class") == "ADDS_VALUE")
                    .sum().alias("n_improved"),
            ])
            .with_columns([
                (pl.col("n_improved") / pl.col("n_evaluations") * 100)
                    .round(1).alias("robustness_score"),
            ])
            .sort(population + ["mean_fva_wmape", "forecast_layer"],
                  descending=[False] * len(population) + [True, False], nulls_last=True)
        )

        # Add rank and recommendation
        board = (
            board.with_columns(pl.col("forecast_layer").cum_count().over(population).alias("rank"))
            if population else board.with_row_index("rank", offset=1)
        )

        board = board.with_columns(
            pl.when(pl.col("n_undefined") > 0)
            .then(pl.lit("Insufficient data"))
            .when(pl.col("robustness_score") >= 60)
            .then(pl.lit("Keep"))
            .when(pl.col("robustness_score") >= 30)
            .then(pl.lit("Review"))
            .otherwise(pl.lit("Remove"))
            .alias("recommendation")
        )

        return board
