"""Comparable-population metric aggregation, shared by ranking and reporting."""

from typing import List, Sequence

import polars as pl

from .definitions import METRIC_DIRECTIONS, metric_expression, validate_metric_inputs

OBSERVATION_KEYS = ["run_id", "lob", "fold", "series_id", "target_week", "forecast_step"]
SCORING_POLICY = "raw-observations-v1"


def validate_comparable_records(records: pl.DataFrame, group_cols: Sequence[str] = ()) -> None:
    """Require one run, unique full keys and identical actual populations per model.

    A summary-only table cannot establish comparable coverage or reconstruct a
    volume ratio. Historical score-only records must be re-backtested.
    """
    required = set(OBSERVATION_KEYS + ["model_id", "actual", "forecast"] + list(group_cols))
    missing = required - set(records.columns)
    if missing:
        raise ValueError(
            f"Comparable scoring requires raw observation keys/values; missing {sorted(missing)}. "
            "Re-run legacy summary-only backtests."
        )
    if records.is_empty():
        raise ValueError("Insufficient data: no observations to score")
    validate_metric_inputs(records["actual"], records["forecast"])
    keys = list(dict.fromkeys(list(group_cols) + OBSERVATION_KEYS))
    keys += [name for name in ("grain_level", "channel") if name in records.columns and name not in keys]
    if any(records[col].null_count() for col in keys + ["model_id"]):
        raise ValueError("Null observation keys cannot establish scoring coverage")
    if records["run_id"].n_unique() != 1:
        raise ValueError("Comparable scoring must target exactly one run_id, not pool historical runs")
    if records.select(keys + ["model_id"]).is_duplicated().any():
        raise ValueError("Duplicate model/series/date/fold/horizon observation keys")

    populations = records.partition_by(list(group_cols), maintain_order=True) if group_cols else [records]
    for population in populations:
        expected = population.select(keys).unique().height
        counts = population.group_by("model_id").len()
        if counts.filter(pl.col("len") != expected).height:
            raise ValueError("Model coverage differs: all candidates must score the exact same observation population")
        disagreement = population.group_by(keys).agg(pl.col("actual").n_unique().alias("_actuals"))
        if disagreement.filter(pl.col("_actuals") != 1).height:
            raise ValueError("Models disagree on actual values for the same observation keys")


def aggregate_metrics(
    records: pl.DataFrame,
    metric_names: Sequence[str],
    group_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Recompute metrics on the requested raw rows, retaining each row's weight.

    WMAPE is sum absolute error / sum absolute actual volume across all selected
    series and folds. ``macro_wmape`` instead equally weights each series/fold
    ratio and is undefined if any constituent has zero actual volume.
    """
    names = list(dict.fromkeys(metric_names))
    unknown = set(names) - set(METRIC_DIRECTIONS)
    if unknown:
        raise ValueError(f"Unknown selection metrics: {sorted(unknown)}")
    validate_comparable_records(records, group_cols)
    keys = list(group_cols) + ["model_id"]
    expressions: List[pl.Expr] = [
        metric_expression(name) for name in names if name != "macro_wmape"
    ]
    expressions += [
        pl.col("series_id").n_unique().alias("n_series"),
        pl.len().alias("n_observations"),
        pl.col("fold").n_unique().alias("n_folds"),
    ]
    if "mase" in names and "mase_scale" not in records.columns:
        raise ValueError("MASE scoring requires the training-only mase_scale; re-run this backtest")
    board = records.group_by(keys).agg(expressions)
    if "macro_wmape" in names:
        unit_keys = list(dict.fromkeys(keys + ["run_id", "lob", "series_id", "fold"]))
        units = records.group_by(unit_keys).agg(metric_expression("wmape"))
        macro = units.group_by(keys).agg(
            pl.when(pl.col("wmape").is_not_null().all())
            .then(pl.col("wmape").mean()).otherwise(None).alias("macro_wmape")
        )
        board = board.join(macro, on=keys, validate="1:1")
    return board.with_columns(pl.lit(SCORING_POLICY).alias("scoring_policy"))


def rank_metrics(
    board: pl.DataFrame,
    primary_metric: str,
    secondary_metric: str,
    group_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Sort errors ascending and signed bias by distance from zero, nulls last."""
    ranks = []
    for name, alias in ((primary_metric, "_primary_rank"), (secondary_metric, "_secondary_rank")):
        direction = METRIC_DIRECTIONS.get(name)
        if direction is None:
            raise ValueError(f"Unknown selection metric {name!r}")
        column = pl.col(name)
        ranks.append((column.abs() if direction == "minimize_absolute" else column).alias(alias))
    return (
        board.with_columns(ranks)
        .sort(list(group_cols) + ["_primary_rank", "_secondary_rank", "model_id"], nulls_last=True)
        .drop("_primary_rank", "_secondary_rank")
    )
