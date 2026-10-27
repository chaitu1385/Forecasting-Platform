"""
Statistical forecasters via statsforecast (Nixtla).

Wraps AutoARIMA and AutoETS to provide the BaseForecaster interface.
Falls back to a simple seasonal repeat if statsforecast is not installed.

These models output the full horizon vector at once (direct multi-step),
which aligns with the platform's seq2seq requirement.
"""

from typing import Any, Dict, List, Optional

import polars as pl

from ..config.schema import get_frequency_profile
from ..utils.calendar import PeriodCalendar
from .base import BaseForecaster
from .registry import registry

# Attempt to import statsforecast; fall back gracefully
try:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL as _MSTL
    from statsforecast.models import AutoARIMA as _AutoARIMA
    from statsforecast.models import AutoETS as _AutoETS
    from statsforecast.models import AutoTheta as _AutoTheta
    from statsforecast.models import SeasonalNaive as _SeasonalNaive  # noqa: F401
    _HAS_STATSFORECAST = True
except ImportError:
    _HAS_STATSFORECAST = False


class _StatsforecastBase(BaseForecaster):
    """
    Shared logic for statsforecast-backed models.

    Handles the Polars ↔ pandas conversion that statsforecast requires,
    and maps columns to the expected ``unique_id / ds / y`` schema.
    """

    def __init__(self, season_length: Optional[int] = None, frequency: str = "W", n_jobs: int = 1):
        self.frequency = frequency
        profile = get_frequency_profile(frequency)
        if season_length is None:
            season_length = profile["season_length"]
        if season_length < 1:
            raise ValueError("season_length must be positive")
        self.season_length = season_length
        self.n_jobs = n_jobs
        self._sf: Optional[Any] = None
        self._id_col: str = "series_id"
        self._time_col: str = "week"
        self._target_col: str = "quantity"

    def _get_model(self) -> Any:
        raise NotImplementedError

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        if not _HAS_STATSFORECAST:
            raise ImportError(
                f"{self.name} requires 'statsforecast'. "
                "Install with: pip install statsforecast"
            )

        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col

        # statsforecast expects pandas with columns: unique_id, ds, y
        pdf = (
            df.select([id_col, time_col, target_col])
            .rename({id_col: "unique_id", time_col: "ds", target_col: "y"})
            .to_pandas()
        )
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")

        calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)
        model = self._get_model()
        self._sf = StatsForecast(
            models=[model],
            freq=calendar.pandas_frequency,
            n_jobs=self.n_jobs,
        )
        self._sf.fit(pdf)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        if self._sf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

        result_pdf = self._sf.predict(h=horizon)

        # Bring index columns (unique_id, ds) back if they're in the index
        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()

        # Map back to original column names
        result = pl.from_pandas(result_pdf)

        # statsforecast names the prediction column after the model class
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        if pred_cols:
            result = result.rename({pred_cols[0]: "forecast"})

        result = result.rename({"unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])

        # Ensure date type
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        return result

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Native statsforecast prediction intervals.

        statsforecast returns symmetric PI at ``level`` % coverage.
        A ``level=80`` interval covers [P10, P90]; ``level=90`` covers [P5, P95].
        For each lower quantile q < 0.5 the required coverage level is
        ``(1 - 2*q) * 100``.
        """
        if self._sf is None:
            raise RuntimeError(f"{self.name}: call fit() before predict_quantiles()")

        # Determine the coverage levels we need
        lower_qs = sorted(q for q in quantiles if q < 0.5 - 1e-6)
        levels = sorted({int(round((1 - 2 * q) * 100)) for q in lower_qs}) if lower_qs else [80]

        result_pdf = self._sf.predict(h=horizon, level=levels)

        # Bring index columns back if needed
        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)
        result = result.rename({"unique_id": id_col, "ds": time_col})
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        # Identify point forecast column (no "-lo-" / "-hi-" in name)
        point_col = next(
            c for c in result.columns
            if c not in (id_col, time_col) and "-lo-" not in c and "-hi-" not in c
        )

        output = result.select([id_col, time_col])
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                output = output.with_columns(result[point_col].alias(col))
            elif q < 0.5:
                level = int(round((1 - 2 * q) * 100))
                lo = next((c for c in result.columns if f"-lo-{level}" in c), None)
                src = result[lo] if lo else result[point_col]
                output = output.with_columns(src.alias(col))
            else:  # q > 0.5 — mirror lower bound
                mirror_q = 1.0 - q
                level = int(round((1 - 2 * mirror_q) * 100))
                hi = next((c for c in result.columns if f"-hi-{level}" in c), None)
                src = result[hi] if hi else result[point_col]
                output = output.with_columns(src.alias(col))

        return output


@registry.register("auto_arima")
class AutoARIMAForecaster(_StatsforecastBase):
    """AutoARIMA via statsforecast."""

    name = "auto_arima"

    def _get_model(self) -> Any:
        return _AutoARIMA(season_length=self.season_length)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "AutoARIMA", "season_length": self.season_length, "frequency": self.frequency}


@registry.register("auto_ets")
class AutoETSForecaster(_StatsforecastBase):
    """AutoETS via statsforecast."""

    name = "auto_ets"

    def _get_model(self) -> Any:
        return _AutoETS(season_length=self.season_length)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "AutoETS", "season_length": self.season_length, "frequency": self.frequency}


@registry.register("auto_theta")
class AutoThetaForecaster(_StatsforecastBase):
    """
    AutoTheta via statsforecast (Nixtla).

    Automatic Theta method selection (Standard Theta, Optimised Theta,
    Dynamic Optimised Theta, Dynamic Standard Theta) with multiplicative
    or additive seasonality.  Typically competitive with ETS/ARIMA on
    monthly and weekly retail data.

    Parameters
    ----------
    season_length:
        Seasonal period.  52 for weekly data with yearly seasonality.
    decomposition_type:
        ``"multiplicative"`` (default) or ``"additive"``.
    """

    name = "auto_theta"

    def __init__(
        self,
        season_length: Optional[int] = None,
        decomposition_type: str = "multiplicative",
        frequency: str = "W",
    ):
        super().__init__(season_length=season_length, frequency=frequency)
        self.decomposition_type = decomposition_type

    def _get_model(self) -> Any:
        return _AutoTheta(
            season_length=self.season_length,
            decomposition_type=self.decomposition_type,
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "AutoTheta",
            "season_length": self.season_length,
            "decomposition_type": self.decomposition_type,
            "frequency": self.frequency,
        }


@registry.register("mstl")
class MSTLForecaster(_StatsforecastBase):
    """
    MSTL (Multiple Seasonal-Trend decomposition using LOESS) via statsforecast.

    Decomposes the series into multiple seasonal components (e.g. weekly +
    yearly) plus trend/remainder, then forecasts each component separately.
    Particularly useful for data with multiple overlapping seasonal patterns.

    The trend component is forecast using AutoETS by default.  Seasonal
    components are projected forward using the last observed cycle (SeasonalNaive).

    Parameters
    ----------
    season_length:
        Primary seasonal period for the outer STL pass (52 for yearly).
    secondary_season_length:
        Optional additional seasonal period (e.g. 13 for quarterly).
        If ``None``, only one seasonal component is extracted.
    """

    name = "mstl"

    def __init__(
        self,
        season_length: Optional[int] = None,
        secondary_season_length: Optional[int] = None,
        frequency: str = "W",
    ):
        super().__init__(season_length=season_length, frequency=frequency)
        self.secondary_season_length = secondary_season_length

    def _get_model(self) -> Any:
        season_lengths = [self.season_length]
        if self.secondary_season_length is not None:
            season_lengths.append(self.secondary_season_length)
        return _MSTL(season_length=season_lengths)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model": "MSTL",
            "season_length": self.season_length,
            "secondary_season_length": self.secondary_season_length,
            "frequency": self.frequency,
        }
