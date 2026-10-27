"""
Seasonal Naive forecaster — the mandatory baseline.

Repeats the last observed seasonal cycle forward.  For weekly data with
yearly seasonality, it copies weeks from the same period one year ago.

This serves as the "beat this" benchmark for all other models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..config.schema import get_frequency_profile
from ..utils.calendar import PeriodCalendar
from .base import BaseForecaster
from .registry import registry


@registry.register("naive_seasonal")
class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive: repeat last year's values."""

    name = "naive_seasonal"

    def __init__(self, season_length: Optional[int] = None, frequency: str = "W"):
        """
        Parameters
        ----------
        season_length:
            Number of periods in one seasonal cycle.  Defaults to the
            value from ``FREQUENCY_PROFILES[frequency]`` when omitted.
            Explicit values, including 52, are always preserved.
        frequency:
            Data frequency — ``"D"``, ``"W"``, ``"M"``, or ``"Q"``.
        """
        self.frequency = frequency
        profile = get_frequency_profile(frequency)
        if season_length is None:
            season_length = profile["season_length"]
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self._fitted_data: pl.DataFrame = pl.DataFrame()
        self._target_col: str = "quantity"
        self._time_col: str = "week"
        self._id_col: str = "series_id"
        self._calendar: Optional[PeriodCalendar] = None

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)
        self._fitted_data = df.select([id_col, time_col, target_col])
        self._target_col = target_col
        self._time_col = time_col
        self._id_col = id_col

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        sl = self.season_length

        def _predict_group(group_df: pl.DataFrame) -> pl.DataFrame:
            """Vectorized predict for a single series group."""
            sorted_df = group_df.sort(self._time_col)
            values = sorted_df[self._target_col].to_list()
            max_date = sorted_df[self._time_col].max()
            series_id_val = sorted_df[self._id_col][0]

            if len(values) == 0 or max_date is None:
                return pl.DataFrame(schema={
                    id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
                })

            n = len(values)
            # Pre-compute all forecast values using numpy for speed
            h_range = np.arange(1, horizon + 1)
            indices = n - sl + ((h_range - 1) % sl)

            # Handle short series: where idx < 0, use cyclic fallback
            short_mask = indices < 0
            indices[short_mask] = (h_range[short_mask] - 1) % n

            # Clip to valid range and extract values
            indices = np.clip(indices, 0, n - 1)
            vals_array = np.array(values, dtype=np.float64)
            forecast_vals = vals_array[indices]

            # Build date array
            dates = self._calendar.future_dates(max_date, horizon)

            return pl.DataFrame({
                id_col: [series_id_val] * horizon,
                time_col: dates,
                "forecast": forecast_vals.tolist(),
            })

        # Use map_groups for vectorized per-series prediction
        if len(self._fitted_data) == 0:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
            })

        results = self._fitted_data.group_by(self._id_col).map_groups(_predict_group)
        return results

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Interval estimation from year-over-year empirical residuals.

        For each seasonal position, collect historical ``value[t] - value[t - season_length]``
        residuals, then use their empirical quantiles as additive offsets over the P50 point
        forecast.  If fewer than two seasons of data are available, falls back to the
        base-class degenerate intervals.
        """
        results = []
        point = self.predict(horizon, id_col=id_col, time_col=time_col)

        for series_id in self._fitted_data[self._id_col].unique().to_list():
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == series_id)
                .sort(self._time_col)
            )
            values = series[self._target_col].to_list()
            max_date = series[self._time_col].max()

            if len(values) < self.season_length + 1 or max_date is None:
                # Not enough history — fall back to degenerate intervals
                point_rows = [
                    {id_col: series_id, time_col: row[time_col],
                     **{f"forecast_p{int(round(q * 100))}": row["forecast"] for q in quantiles}}
                    for row in point.filter(pl.col(id_col) == series_id).iter_rows(named=True)
                ]
                results.extend(point_rows)
                continue

            n = len(values)
            sl = self.season_length

            # Collect YoY residuals per seasonal position
            residuals_by_pos: Dict[int, List[float]] = {}
            for i in range(sl, n):
                pos = i % sl
                residuals_by_pos.setdefault(pos, []).append(values[i] - values[i - sl])

            for h in range(1, horizon + 1):
                idx = n - sl + ((h - 1) % sl)
                if idx < 0:
                    idx = (h - 1) % n
                point_val = float(values[idx] if 0 <= idx < n else 0.0)
                forecast_date = self._calendar.shift(max_date, h)

                pos = idx % sl
                pos_residuals = residuals_by_pos.get(pos, [0.0])

                row: Dict = {id_col: series_id, time_col: forecast_date}
                for q in quantiles:
                    col = f"forecast_p{int(round(q * 100))}"
                    if abs(q - 0.5) < 1e-6:
                        row[col] = point_val
                    else:
                        row[col] = point_val + float(np.quantile(pos_residuals, q))
                results.append(row)

        if not results:
            schema: Dict = {id_col: pl.Utf8, time_col: pl.Date}
            for q in quantiles:
                schema[f"forecast_p{int(round(q * 100))}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(results)

    def get_params(self) -> Dict[str, Any]:
        return {"season_length": self.season_length, "frequency": self.frequency}
