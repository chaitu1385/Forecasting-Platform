"""
Intermittent / sparse demand forecasters.

Implements three methods suitable for slow-moving, lumpy, or intermittent
demand patterns:

  CrostonForecaster:    Classic Croston (1972) — separate SES for size and interval.
  CrostonSBAForecaster: Syntetos-Boylan Approximation — bias-corrected Croston.
  TSBForecaster:        Teunter-Syntetos-Babai — handles demand obsolescence.

All three follow the same BaseForecaster contract (fit / predict / predict_quantiles).
``fit()`` stores the final smoothed state; ``predict()`` projects a constant
forecast forward (the standard approach for intermittent demand).

Quantile intervals
------------------
Intermittent demand distributions are zero-inflated.  The quantile forecast
partitions the probability mass between zero and positive demand:

  - Quantiles below (1 - p_demand) map to 0
  - Quantiles above (1 - p_demand) are drawn from the empirical distribution
    of historical non-zero demands, re-scaled to the remaining probability mass.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ..config.schema import get_frequency_profile
from ..utils.calendar import PeriodCalendar
from .base import BaseForecaster
from .registry import registry

# ─────────────────────────────────────────────────────────────────────────────
# Core fitting routines (pure Python for testability)
# ─────────────────────────────────────────────────────────────────────────────

def _croston_fit(
    values: List[float],
    alpha: float = 0.1,
    sba_correction: bool = False,
) -> Tuple[float, float]:
    """
    Fit Croston's method on one series.

    Returns
    -------
    (z, x): smoothed demand size and smoothed inter-demand interval.
    Forecast rate = z / x.
    """
    z: Optional[float] = None
    x: Optional[float] = None
    last_demand_t: Optional[int] = None

    for t, demand in enumerate(values):
        if demand > 0:
            if z is None:
                # First non-zero: initialize
                z = float(demand)
                x = 1.0
                last_demand_t = t
            else:
                interval = t - last_demand_t  # type: ignore[operator]
                z = alpha * demand + (1 - alpha) * z
                x = alpha * interval + (1 - alpha) * x
                last_demand_t = t

    if z is None:
        # Series is all zeros
        return 0.0, 1.0

    if sba_correction:
        # Syntetos-Boylan: (1 - alpha/2) corrects Croston's upward bias
        z = (1.0 - alpha / 2.0) * z

    return float(z), float(x)  # type: ignore[arg-type]


def _tsb_fit(
    values: List[float],
    alpha_z: float = 0.1,
    alpha_p: float = 0.1,
) -> Tuple[float, float]:
    """
    Fit the Teunter-Syntetos-Babai (TSB) method on one series.

    Returns
    -------
    (p, z): demand probability per period and mean non-zero demand size.
    Forecast = p * z.
    """
    if not values:
        return 0.0, 0.0

    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return 0.0, 0.0

    # Initialise with overall proportion and first non-zero demand
    p = len(nonzero) / len(values)
    z = float(nonzero[0])

    for demand in values[1:]:
        if demand > 0:
            p = alpha_p * 1.0 + (1.0 - alpha_p) * p
            z = alpha_z * demand + (1.0 - alpha_z) * z
        else:
            p = alpha_p * 0.0 + (1.0 - alpha_p) * p
            # z unchanged when there is no demand

    return float(p), float(z)


# ─────────────────────────────────────────────────────────────────────────────
# Forecasters
# ─────────────────────────────────────────────────────────────────────────────

@registry.register("croston")
class CrostonForecaster(BaseForecaster):
    """
    Croston's method (1972) for intermittent demand.

    Maintains separate exponential smoothing for:
      - z : mean non-zero demand size
      - x : mean inter-demand interval

    Point forecast = z / x  (constant over the horizon).
    """

    name = "croston"

    def __init__(self, alpha: float = 0.1, frequency: str = "W"):
        """
        Parameters
        ----------
        alpha:
            Smoothing parameter applied to both demand size and interval.
        """
        self.alpha = alpha
        get_frequency_profile(frequency)
        self.frequency = frequency
        self._calendar: Optional[PeriodCalendar] = None
        # series_id → (z, x)
        self._states: Dict[str, Tuple[float, float]] = {}
        self._last_dates: Dict[str, Any] = {}
        self._history: Dict[str, List[float]] = {}
        self._target_col = "quantity"
        self._time_col = "week"
        self._id_col = "series_id"

    @property
    def _sba(self) -> bool:
        """Subclass hook — True enables Syntetos-Boylan correction."""
        return False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._target_col = target_col
        self._time_col = time_col
        self._id_col = id_col
        self._states = {}
        self._last_dates = {}
        self._history = {}
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)

        for series_id in df[id_col].unique().to_list():
            series = df.filter(pl.col(id_col) == series_id).sort(time_col)
            values = series[target_col].to_list()
            max_date = series[time_col].max()

            z, x = _croston_fit(values, alpha=self.alpha, sba_correction=self._sba)
            self._states[series_id] = (z, x)
            self._last_dates[series_id] = max_date
            self._history[series_id] = [float(v) for v in values]

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        results = []

        for series_id, (z, x) in self._states.items():
            max_date = self._last_dates.get(series_id)
            if max_date is None:
                continue

            forecast_val = max(0.0, z / x if x > 0 else 0.0)
            for h in range(1, horizon + 1):
                results.append({
                    id_col: series_id,
                    time_col: self._calendar.shift(max_date, h),
                    "forecast": forecast_val,
                })

        if not results:
            return pl.DataFrame(
                schema={id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64}
            )
        return pl.DataFrame(results)

    # ── predict_quantiles ─────────────────────────────────────────────────────

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Zero-inflated quantile intervals.

        The demand distribution is modelled as a two-part mixture:
          - A point mass at 0 with probability (1 - p_demand)
          - A continuous part drawn from empirical historical non-zero demands

        where p_demand = 1 / x  (x = smoothed inter-demand interval).

        Quantile q:
          - If q < 1 - p_demand  → forecast = 0
          - Otherwise            → rescaled quantile of non-zero empirical dist.
        """
        results = []

        for series_id, (z, x) in self._states.items():
            max_date = self._last_dates.get(series_id)
            if max_date is None:
                continue

            values = self._history.get(series_id, [])
            nonzero = [v for v in values if v > 0]
            point_forecast = max(0.0, z / x if x > 0 else 0.0)
            p_demand = min(1.0, 1.0 / x) if x > 0 else 1.0

            for h in range(1, horizon + 1):
                row: Dict[str, Any] = {
                    id_col: series_id,
                    time_col: self._calendar.shift(max_date, h),
                }
                for q in quantiles:
                    col = f"forecast_p{int(round(q * 100))}"
                    if abs(q - 0.5) < 1e-6:
                        row[col] = point_forecast
                    elif not nonzero:
                        row[col] = 0.0
                    else:
                        p_zero = 1.0 - p_demand
                        if q <= p_zero:
                            row[col] = 0.0
                        else:
                            q_nz = (q - p_zero) / p_demand
                            q_nz = max(0.0, min(1.0, q_nz))
                            row[col] = float(np.quantile(nonzero, q_nz))
                results.append(row)

        if not results:
            schema: Dict[str, Any] = {id_col: pl.Utf8, time_col: pl.Date}
            for q in quantiles:
                schema[f"forecast_p{int(round(q * 100))}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(results)

    def get_params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "frequency": self.frequency}


@registry.register("croston_sba")
class CrostonSBAForecaster(CrostonForecaster):
    """
    Syntetos-Boylan Approximation (SBA) — bias-corrected Croston.

    Applies the (1 - alpha/2) correction factor to the demand size estimate,
    addressing the upward bias documented by Syntetos & Boylan (2001).
    """

    name = "croston_sba"

    @property
    def _sba(self) -> bool:
        return True


@registry.register("tsb")
class TSBForecaster(BaseForecaster):
    """
    Teunter-Syntetos-Babai (TSB) method for intermittent demand.

    Unlike Croston, TSB updates the demand probability at every period
    (not only at demand occurrences), making it better at detecting
    demand obsolescence — i.e. series that have permanently dropped to zero.

    Point forecast = p × z  (probability × mean non-zero demand).
    """

    name = "tsb"

    def __init__(self, alpha_z: float = 0.1, alpha_p: float = 0.1, frequency: str = "W"):
        """
        Parameters
        ----------
        alpha_z:
            Smoothing for demand size (z).
        alpha_p:
            Smoothing for demand probability (p).
        """
        self.alpha_z = alpha_z
        self.alpha_p = alpha_p
        get_frequency_profile(frequency)
        self.frequency = frequency
        self._calendar: Optional[PeriodCalendar] = None
        # series_id → (p, z)
        self._states: Dict[str, Tuple[float, float]] = {}
        self._last_dates: Dict[str, Any] = {}
        self._history: Dict[str, List[float]] = {}
        self._target_col = "quantity"
        self._time_col = "week"
        self._id_col = "series_id"

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._target_col = target_col
        self._time_col = time_col
        self._id_col = id_col
        self._states = {}
        self._last_dates = {}
        self._history = {}
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)

        for series_id in df[id_col].unique().to_list():
            series = df.filter(pl.col(id_col) == series_id).sort(time_col)
            values = series[target_col].to_list()
            max_date = series[time_col].max()

            p, z = _tsb_fit(values, alpha_z=self.alpha_z, alpha_p=self.alpha_p)
            self._states[series_id] = (p, z)
            self._last_dates[series_id] = max_date
            self._history[series_id] = [float(v) for v in values]

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        results = []

        for series_id, (p, z) in self._states.items():
            max_date = self._last_dates.get(series_id)
            if max_date is None:
                continue

            forecast_val = max(0.0, p * z)
            for h in range(1, horizon + 1):
                results.append({
                    id_col: series_id,
                    time_col: self._calendar.shift(max_date, h),
                    "forecast": forecast_val,
                })

        if not results:
            return pl.DataFrame(
                schema={id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64}
            )
        return pl.DataFrame(results)

    # ── predict_quantiles ─────────────────────────────────────────────────────

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Zero-inflated quantile intervals using TSB demand probability.

        p_zero = 1 - p  (probability of zero demand)

        Quantile q:
          - If q <= p_zero → forecast = 0
          - Otherwise      → rescaled quantile of historical non-zero demands.
        """
        results = []

        for series_id, (p, z) in self._states.items():
            max_date = self._last_dates.get(series_id)
            if max_date is None:
                continue

            values = self._history.get(series_id, [])
            nonzero = [v for v in values if v > 0]
            point_forecast = max(0.0, p * z)
            p_zero = 1.0 - p

            for h in range(1, horizon + 1):
                row: Dict[str, Any] = {
                    id_col: series_id,
                    time_col: self._calendar.shift(max_date, h),
                }
                for q in quantiles:
                    col = f"forecast_p{int(round(q * 100))}"
                    if abs(q - 0.5) < 1e-6:
                        row[col] = point_forecast
                    elif not nonzero:
                        row[col] = 0.0
                    else:
                        if q <= p_zero:
                            row[col] = 0.0
                        else:
                            q_nz = (q - p_zero) / p if p > 0 else 1.0
                            q_nz = max(0.0, min(1.0, q_nz))
                            row[col] = float(np.quantile(nonzero, q_nz))
                results.append(row)

        if not results:
            schema: Dict[str, Any] = {id_col: pl.Utf8, time_col: pl.Date}
            for q in quantiles:
                schema[f"forecast_p{int(round(q * 100))}"] = pl.Float64
            return pl.DataFrame(schema=schema)

        return pl.DataFrame(results)

    def get_params(self) -> Dict[str, Any]:
        return {"alpha_z": self.alpha_z, "alpha_p": self.alpha_p, "frequency": self.frequency}
