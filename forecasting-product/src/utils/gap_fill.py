"""Shared gap-filling utility for time series DataFrames.

Consolidates the repeated pattern of:
  date_range → cross join with series IDs → left join → fill nulls.

Used by ``series.builder.SeriesBuilder`` and ``forecasting.base.BaseForecaster``.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from ..config.schema import get_frequency_profile
from .calendar import PeriodCalendar


def fill_gaps(
    df: pl.DataFrame,
    *,
    time_col: str = "week",
    id_col: str = "series_id",
    target_col: str = "quantity",
    fill_value: float = 0.0,
    strategy: Literal["zero", "forward_fill"] = "zero",
    freq: str = "W",
) -> pl.DataFrame:
    """Fill missing periods for each series to produce a contiguous date grid.

    Parameters
    ----------
    df:
        Input DataFrame with at least ``time_col``, ``id_col``, and ``target_col``.
    time_col:
        Name of the date column.
    id_col:
        Name of the series identifier column.
    target_col:
        Name of the numeric target column to fill.
    fill_value:
        Value used when ``strategy="zero"`` (default 0.0).
    strategy:
        ``"zero"`` fills gaps with *fill_value*;
        ``"forward_fill"`` carries the last observation forward,
        then back-fills any remaining leading nulls.
    freq:
        Frequency code: ``"D"`` | ``"W"`` | ``"M"`` | ``"Q"``.

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per (id, period) combination, sorted by
        ``[id_col, time_col]``.
    """
    get_frequency_profile(freq)
    if df.is_empty():
        return df

    min_date = df[time_col].min()
    max_date = df[time_col].max()
    if min_date is None or max_date is None:
        return df

    calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), freq)
    if df.select([id_col, time_col]).is_duplicated().any():
        raise ValueError("Cannot fill gaps with duplicate series/date keys")
    all_periods = calendar.date_range(min_date, max_date)
    periods_df = pl.DataFrame({time_col: pl.Series(time_col, all_periods, dtype=df[time_col].dtype)})

    series_ids = df.select(id_col).unique()
    complete_grid = series_ids.join(periods_df, how="cross")

    filled = complete_grid.join(df, on=[id_col, time_col], how="left")

    if target_col in filled.columns:
        if strategy == "forward_fill":
            filled = filled.sort([id_col, time_col])
            filled = filled.with_columns(
                pl.col(target_col).forward_fill().over(id_col).alias(target_col)
            )
            filled = filled.with_columns(
                pl.col(target_col).backward_fill().over(id_col).alias(target_col)
            )
        else:
            filled = filled.with_columns(
                pl.col(target_col).fill_null(fill_value)
            )

    return filled.sort([id_col, time_col])
