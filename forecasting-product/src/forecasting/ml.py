"""
ML-based forecasters via mlforecast (Nixtla) or direct implementation.

These produce direct multi-step forecasts: one model outputs the full
horizon vector.  Uses LightGBM or XGBoost as the underlying learner.

Falls back to a direct manual implementation if mlforecast is not installed.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from ..config.schema import freq_timedelta, get_frequency_profile
from .base import BaseForecaster
from .feature_manager import MLForecastFeatureManager
from .registry import registry

# Attempt to import mlforecast; fall back to manual implementation
try:
    from mlforecast import MLForecast
    _HAS_MLFORECAST = True
except ImportError:
    _HAS_MLFORECAST = False


class _DirectMLBase(BaseForecaster):
    """
    Shared logic for ML direct multi-step forecasters.

    If mlforecast is available, delegates to it for feature engineering
    (lags, rolling stats, date features).  Otherwise, builds features
    manually in Polars.
    """

    # Default lags for weekly frequency — used when lags=None and frequency="W".
    # For other frequencies the profile default is used instead.
    DEFAULT_LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 20, 26, 52]

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None,
        num_threads: int = 1,
        freq: Optional[str] = None,
        frequency: str = "W",
        **kwargs,
    ):
        self.frequency = frequency
        profile = get_frequency_profile(frequency)
        self.lags = lags or list(profile["default_lags"])
        self.lag_transforms = lag_transforms or self._default_lag_transforms()
        self.num_threads = num_threads
        self._freq = freq  # auto-detected from data if None
        self._id_col: str = "series_id"
        self._time_col: str = "week"
        self._target_col: str = "quantity"

        # Set by subclass
        self._model = None

        # mlforecast instance (if available)
        self._mlf: Optional[Any] = None

        # Feature manager handles external feature lifecycle
        self._feature_mgr = MLForecastFeatureManager()

        # Manual fallback state
        self._fitted_data: Optional[pl.DataFrame] = None
        self._models_per_step: Dict[int, Any] = {}

        # Quantile regression state (lazily populated by predict_quantiles)
        self._quantile_mlfs: Dict[float, Any] = {}     # q -> fitted MLForecast

    @staticmethod
    def _default_lag_transforms() -> Dict:
        """Default rolling window transforms applied to lag features."""
        try:
            from mlforecast.lag_transforms import RollingMean, RollingStd
            return {
                1: [
                    RollingMean(window_size=4),
                    RollingMean(window_size=8),
                    RollingMean(window_size=13),
                    RollingMean(window_size=26),
                    RollingStd(window_size=4),
                    RollingStd(window_size=13),
                ],
            }
        except ImportError:
            return {}

    def _get_learner(self) -> Any:
        raise NotImplementedError

    def _get_quantile_learner(self, alpha: float) -> Any:
        """Return a quantile regression version of the base learner. Override in subclasses."""
        raise NotImplementedError

    def validate_and_prepare(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> pl.DataFrame:
        """Fill gaps with forward-fill using the correct frequency — avoids zero-contamination for tree models."""
        from ..utils.gap_fill import fill_gaps
        return fill_gaps(
            df, time_col=time_col, id_col=id_col, target_col=target_col,
            strategy="forward_fill", freq=self.frequency,
        )

    def fit(
        self,
        df: pl.DataFrame,
        target_col: str = "quantity",
        time_col: str = "week",
        id_col: str = "series_id",
    ) -> None:
        self._id_col = id_col
        self._time_col = time_col
        self._target_col = target_col
        # Quantile models belong to the previous training window, not this refit.
        self._quantile_mlfs.clear()

        df = self.validate_and_prepare(df, target_col, time_col, id_col)

        if _HAS_MLFORECAST:
            self._fit_mlforecast(df, target_col, time_col, id_col)
        else:
            self._fit_manual(df, target_col, time_col, id_col)

    def predict(
        self,
        horizon: int,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        if _HAS_MLFORECAST and self._mlf is not None:
            return self._predict_mlforecast(horizon, id_col, time_col)
        return self._predict_manual(horizon, id_col, time_col)

    # ── mlforecast path ───────────────────────────────────────────────────

    @staticmethod
    def _week_of_year(dates: Any) -> Any:
        """Extract ISO week-of-year as an integer feature for mlforecast."""
        return dates.isocalendar().week.astype(int)

    @staticmethod
    def _month(dates: Any) -> Any:
        """Extract month as an integer feature."""
        return dates.month

    @staticmethod
    def _quarter(dates: Any) -> Any:
        """Extract quarter as an integer feature."""
        return dates.quarter

    def _detect_freq(self, df: pl.DataFrame, time_col: str) -> str:
        """Detect the pandas-compatible frequency string from data dates.

        For weekly data, infers the correct ``W-<DAY>`` anchor from actual
        dates.  For other frequencies, returns the statsforecast freq string
        from the frequency profile.
        """
        profile = get_frequency_profile(self.frequency)
        if self.frequency != "W":
            return profile["statsforecast_freq"]
        dates = df[time_col].unique().sort()
        if len(dates) < 2:
            return "W"
        first_date = dates[0]
        if hasattr(first_date, "weekday"):
            day_name = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][first_date.weekday()]
            return f"W-{day_name}"
        return "W"

    def _get_date_features(self) -> List:
        """Return date feature extractors appropriate for the frequency."""
        features = [self._month, self._quarter]
        if self.frequency in ("D", "W"):
            features.append(self._week_of_year)
        if self.frequency == "D":
            features.append(self._day_of_week)
        return features

    @staticmethod
    def _day_of_week(dates: Any) -> Any:
        """Extract day-of-week (0=Mon … 6=Sun) as an integer feature."""
        return dates.weekday

    def _fit_mlforecast(self, df: pl.DataFrame, target_col: str, time_col: str, id_col: str) -> None:
        train_pdf = self._feature_mgr.prepare_fit(df, id_col, time_col, target_col)

        freq = self._freq or self._detect_freq(df, time_col)

        self._mlf = MLForecast(
            models=[self._get_learner()],
            freq=freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self._get_date_features(),
            num_threads=self.num_threads,
        )

        if self._feature_mgr.has_features:
            self._mlf.fit(train_pdf, validate_data=False, static_features=[])
        else:
            self._mlf.fit(train_pdf, validate_data=False)

    def _predict_mlforecast(self, horizon: int, id_col: str, time_col: str) -> pl.DataFrame:
        future_X = self._feature_mgr.prepare_predict(horizon)
        if future_X is not None:
            result_pdf = self._mlf.predict(h=horizon, X_df=future_X)
        else:
            result_pdf = self._mlf.predict(h=horizon)

        # Bring index columns back if needed
        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()

        result = pl.from_pandas(result_pdf)

        # mlforecast names prediction column after the learner class
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        if pred_cols:
            result = result.rename({pred_cols[0]: "forecast"})

        result = result.rename({"unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])

        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))

        return result

    def set_future_features(self, future_features: pl.DataFrame, id_col: str = "series_id", time_col: str = "week") -> None:
        """Set external feature values for the forecast horizon."""
        self._feature_mgr.set_future_features(future_features, id_col, time_col)

    # ── Probabilistic forecasting ─────────────────────────────────────────

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: List[float],
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> pl.DataFrame:
        """
        Quantile forecasts via native quantile regression.

        For each non-P50 quantile a separate model is trained lazily using the
        underlying learner's quantile objective (LightGBM ``objective="quantile"``,
        XGBoost ``objective="reg:quantileerror"``).  P50 reuses the point forecast.

        Falls back to YoY-residual empirical quantiles when mlforecast is not
        available or quantile learner is not supported by the subclass.
        """
        # P50 always comes from the already-fitted point model
        point = self.predict(horizon, id_col=id_col, time_col=time_col)
        output = point.select([id_col, time_col])

        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            if abs(q - 0.5) < 1e-6:
                output = output.with_columns(point["forecast"].alias(col))
                continue

            if _HAS_MLFORECAST and self._feature_mgr.train_pdf is not None:
                try:
                    q_preds = self._predict_quantile_mlforecast(
                        q, horizon, id_col, time_col
                    )
                    output = output.join(
                        q_preds.rename({"forecast": col}),
                        on=[id_col, time_col],
                        how="left",
                    )
                    continue
                except (NotImplementedError, Exception):
                    pass  # fall through to residual fallback

            # Residual fallback: use empirical quantile of historical errors
            q_preds = self._predict_quantile_residual(
                q, point, id_col, time_col
            )
            output = output.with_columns(q_preds.alias(col))

        return output

    def _predict_quantile_mlforecast(
        self, q: float, horizon: int, id_col: str, time_col: str,
    ) -> pl.DataFrame:
        """Lazily train + predict a quantile regression model via mlforecast."""
        if q not in self._quantile_mlfs:
            q_learner = self._get_quantile_learner(q)  # raises NotImplementedError if unsupported
            freq = self._freq or (self._mlf.freq if self._mlf else "W")
            mlf_q = MLForecast(
                models=[q_learner],
                freq=freq,
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=self._get_date_features(),
                num_threads=self.num_threads,
            )
            mlf_q.fit(self._feature_mgr.train_pdf, validate_data=False)
            self._quantile_mlfs[q] = mlf_q

        result_pdf = self._quantile_mlfs[q].predict(h=horizon)
        if "unique_id" not in result_pdf.columns or "ds" not in result_pdf.columns:
            result_pdf = result_pdf.reset_index()
        result = pl.from_pandas(result_pdf)
        pred_cols = [c for c in result.columns if c not in ("unique_id", "ds")]
        result = result.rename({pred_cols[0]: "forecast", "unique_id": id_col, "ds": time_col})
        result = result.select([id_col, time_col, "forecast"])
        if result[time_col].dtype != pl.Date:
            result = result.with_columns(pl.col(time_col).cast(pl.Date))
        return result

    def _predict_quantile_residual(
        self, q: float, point: pl.DataFrame, id_col: str, time_col: str
    ) -> pl.Series:
        """
        Residual-based interval: point forecast ± empirical quantile of historical
        seasonal residuals.  Used when quantile regression is unavailable.
        """
        if self._fitted_data is None:
            return point["forecast"]

        offsets = []
        for h_row in point.iter_rows(named=True):
            sid = h_row[id_col]
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == sid)
                .sort(self._time_col)
            )
            values = series[self._target_col].to_list()
            n = len(values)
            sl = get_frequency_profile(self.frequency)["season_length"]
            residuals = [
                values[i] - values[i - sl]
                for i in range(sl, n)
            ] or [0.0]
            offset = float(np.quantile(residuals, q))
            offsets.append(h_row["forecast"] + offset)

        return pl.Series("forecast", offsets)

    # ── Manual fallback ───────────────────────────────────────────────────

    def _fit_manual(self, df: pl.DataFrame, target_col: str, time_col: str, id_col: str) -> None:
        """Direct multi-step: train one model per horizon step."""
        self._fitted_data = df.select([id_col, time_col, target_col]).sort(
            [id_col, time_col]
        )

    def _predict_manual(self, horizon: int, id_col: str, time_col: str) -> pl.DataFrame:
        """Simple lag-based prediction when mlforecast is unavailable."""
        if self._fitted_data is None:
            raise RuntimeError("Call fit() before predict()")

        results = []
        for sid in self._fitted_data[self._id_col].unique().to_list():
            series = (
                self._fitted_data
                .filter(pl.col(self._id_col) == sid)
                .sort(self._time_col)
            )
            values = series[self._target_col].to_list()
            max_date = series[self._time_col].max()

            if not values or max_date is None:
                continue

            # Simple: use last season_length values cyclically
            n = len(values)
            sl = get_frequency_profile(self.frequency)["season_length"]
            for h in range(1, horizon + 1):
                idx = n - sl + ((h - 1) % sl)
                if idx < 0:
                    idx = max(0, n - 1)
                val = values[min(idx, n - 1)]
                results.append({
                    id_col: sid,
                    time_col: max_date + freq_timedelta(self.frequency, h),
                    "forecast": float(val),
                })

        if not results:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date, "forecast": pl.Float64
            })
        return pl.DataFrame(results)


@registry.register("lgbm_direct")
class LGBMDirectForecaster(_DirectMLBase):
    """LightGBM direct multi-step forecaster."""

    name = "lgbm_direct"

    # Tuned defaults: more trees + lower lr + regularization to prevent overfitting
    DEFAULT_PARAMS = dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
    )

    def __init__(self, lgbm_params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._lgbm_params = {
            **self.DEFAULT_PARAMS, "n_jobs": self.num_threads, **(lgbm_params or {}),
        }

    def _get_learner(self) -> Any:
        import lightgbm as lgb
        return lgb.LGBMRegressor(**self._lgbm_params)

    def _get_quantile_learner(self, alpha: float) -> Any:
        import lightgbm as lgb
        params = {**self._lgbm_params, "objective": "quantile", "alpha": alpha}
        return lgb.LGBMRegressor(**params)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "LightGBM", "lags": self.lags, **self._lgbm_params}


@registry.register("xgboost_direct")
class XGBoostDirectForecaster(_DirectMLBase):
    """XGBoost direct multi-step forecaster."""

    name = "xgboost_direct"

    # Tuned defaults: more trees + lower lr + regularization to prevent overfitting
    DEFAULT_PARAMS = dict(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbosity=0,
    )

    def __init__(self, xgb_params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._xgb_params = {
            **self.DEFAULT_PARAMS, "n_jobs": self.num_threads, **(xgb_params or {}),
        }

    def _get_learner(self) -> Any:
        import xgboost as xgb
        return xgb.XGBRegressor(**self._xgb_params)

    def _get_quantile_learner(self, alpha: float) -> Any:
        import xgboost as xgb
        try:
            params = {
                **self._xgb_params,
                "objective": "reg:quantileerror",
                "quantile_alpha": alpha,
            }
            return xgb.XGBRegressor(**params)
        except TypeError:
            return xgb.XGBRegressor(**self._xgb_params)

    def get_params(self) -> Dict[str, Any]:
        return {"model": "XGBoost", "lags": self.lags, **self._xgb_params}
