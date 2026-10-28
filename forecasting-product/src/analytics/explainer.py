"""
ForecastExplainer — business and technical forecast attribution.

Two explanation paths:

  Statistical (moving-average decomposition, not LOESS STL)
  ──────────────────────────────────────
  Works for any model.  Decomposes the history into trend + seasonal +
  residual components, then attributes the forecast to the same components
  using a linear trend extrapolation and the stored seasonal pattern.

  Components:
    trend      — centered moving average (window = 2 * floor(season_length / 2) + 1)
    seasonal   — average (value - trend) per seasonal position
    residual   — value - trend - seasonal
    forecast_trend    — linear trend extrapolated from last trend_window obs
    forecast_seasonal — historical seasonal pattern applied forward

  ML feature attribution (SHAP)
  ──────────────────────────────
  Requires ``shap`` (pip install shap) and a fitted LightGBM / XGBoost
  model that exposes ``.feature_names_in_`` or ``.feature_name_``.
  Returns per-prediction SHAP values as a tidy DataFrame.

  If ``shap`` is not installed, ``explain_ml()`` returns an empty
  DataFrame with a warning — the rest of the platform is unaffected.

Narrative generator
────────────────────
``narrative()`` combines decomposition and comparison outputs into a
templated natural-language string per series, suitable for BI tools or
planner review UIs.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


class ForecastExplainer:
    """
    Decompose and explain forecasts for planners and data scientists.

    Parameters
    ----------
    season_length:
        Number of periods in one seasonal cycle (52 for weekly-yearly).
    trend_window:
        Number of recent observations used to estimate the linear trend
        that is extrapolated into the forecast horizon.
    """

    def __init__(self, season_length: int = 52, trend_window: int = 12):
        self.season_length = season_length
        self.trend_window = trend_window

    # ─────────────────────────────────────────────────────────────────────────
    # Statistical decomposition
    # ─────────────────────────────────────────────────────────────────────────

    def decompose(
        self,
        history: pl.DataFrame,
        forecast: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
        target_col: str = "quantity",
        value_col: str = "forecast",
    ) -> pl.DataFrame:
        """
        Decompose history and forecast into trend + seasonal + residual.

        Parameters
        ----------
        history:
            Historical panel data [id_col, time_col, target_col].
        forecast:
            Forecast panel data [id_col, time_col, value_col].

        Returns
        -------
        DataFrame with columns:
          [id_col, time_col, "value", "trend", "seasonal", "residual",
           "is_forecast"]
        where "value" = target_col for history, value_col for forecast.
        """
        results = []

        all_series = (
            set(history[id_col].unique().to_list())
            | set(forecast[id_col].unique().to_list())
        )

        for sid in all_series:
            hist_s = (
                history.filter(pl.col(id_col) == sid)
                .sort(time_col)
            )
            fc_s = (
                forecast.filter(pl.col(id_col) == sid)
                .sort(time_col)
            )

            if hist_s.is_empty():
                continue

            hist_vals = np.array(hist_s[target_col].to_list(), dtype=float)
            hist_times = hist_s[time_col].to_list()
            fc_vals = np.array(fc_s[value_col].to_list(), dtype=float) if not fc_s.is_empty() else np.array([])
            fc_times = fc_s[time_col].to_list()

            # Decompose history
            trend_hist, seasonal_hist, resid_hist = self._decompose_series(hist_vals)

            # Extrapolate trend into forecast horizon
            n = len(hist_vals)
            win = min(self.trend_window, n)
            if win >= 2:
                x = np.arange(win, dtype=float)
                y = trend_hist[n - win:]
                # Filter out NaN (can appear at edges of moving average)
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    slope = float(np.polyfit(x[mask], y[mask], 1)[0])
                    trend_last = float(np.nanmean(trend_hist[max(0, n - 3):]))
                else:
                    slope = 0.0
                    trend_last = float(np.nanmean(trend_hist))
            else:
                slope = 0.0
                trend_last = float(np.nanmean(trend_hist)) if len(trend_hist) > 0 else 0.0

            # Build historical component rows
            for i, (t, val) in enumerate(zip(hist_times, hist_vals)):
                results.append({
                    id_col: sid,
                    time_col: t,
                    "value": float(val),
                    "trend": float(trend_hist[i]) if not np.isnan(trend_hist[i]) else None,
                    "seasonal": float(seasonal_hist[i]),
                    "residual": float(resid_hist[i]) if not np.isnan(resid_hist[i]) else None,
                    "is_forecast": False,
                })

            # Build forecast component rows
            for h, (t, val) in enumerate(zip(fc_times, fc_vals), start=1):
                fc_trend = trend_last + slope * h
                pos = (n + h - 1) % self.season_length
                fc_seasonal = seasonal_hist[pos] if pos < len(seasonal_hist) else 0.0
                fc_resid = float(val) - fc_trend - fc_seasonal
                results.append({
                    id_col: sid,
                    time_col: t,
                    "value": float(val),
                    "trend": fc_trend,
                    "seasonal": float(fc_seasonal),
                    "residual": fc_resid,
                    "is_forecast": True,
                })

        if not results:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date,
                "value": pl.Float64, "trend": pl.Float64,
                "seasonal": pl.Float64, "residual": pl.Float64,
                "is_forecast": pl.Boolean,
            })

        return pl.DataFrame(results).sort([id_col, time_col])

    # ─────────────────────────────────────────────────────────────────────────
    # ML attribution (SHAP)
    # ─────────────────────────────────────────────────────────────────────────

    def explain_ml(
        self,
        model: Any,
        features_df: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
        top_k: int = 5,
    ) -> pl.DataFrame:
        """
        Per-prediction SHAP attribution for a fitted LightGBM/XGBoost model.

        Parameters
        ----------
        model:
            Fitted model with a ``predict`` method.  Must expose feature
            names via ``.feature_names_in_`` (sklearn convention) or
            ``.feature_name_`` (LightGBM native).
        features_df:
            DataFrame with [id_col, time_col, feature_1, ...].
        top_k:
            Keep only the top-K features by absolute SHAP value per row.

        Returns
        -------
        Tidy DataFrame with columns:
          [id_col, time_col, "feature", "shap_value", "rank"]
        or empty DataFrame with a ``shap_unavailable`` column if
        ``shap`` is not installed.
        """
        try:
            import shap  # type: ignore
        except ImportError:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date,
                "feature": pl.Utf8, "shap_value": pl.Float64,
                "rank": pl.Int32,
                "shap_unavailable": pl.Boolean,
            })

        # Resolve feature names
        if hasattr(model, "feature_names_in_"):
            feature_names: List[str] = list(model.feature_names_in_)
        elif hasattr(model, "feature_name_"):
            feature_names = list(model.feature_name_())
        else:
            feature_names = [
                c for c in features_df.columns
                if c not in (id_col, time_col)
            ]

        feat_cols = [c for c in feature_names if c in features_df.columns]
        if not feat_cols:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date,
                "feature": pl.Utf8, "shap_value": pl.Float64,
                "rank": pl.Int32,
            })

        X = features_df.select(feat_cols).to_numpy()
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X).values  # (n_rows × n_features)

        ids = features_df[id_col].to_list()
        times = features_df[time_col].to_list()

        rows = []
        for i, (sid, t) in enumerate(zip(ids, times)):
            sv = shap_values[i]
            ranked_idx = np.argsort(np.abs(sv))[::-1][:top_k]
            for rank, fi in enumerate(ranked_idx, start=1):
                rows.append({
                    id_col: sid,
                    time_col: t,
                    "feature": feat_cols[fi],
                    "shap_value": float(sv[fi]),
                    "rank": rank,
                })

        if not rows:
            return pl.DataFrame(schema={
                id_col: pl.Utf8, time_col: pl.Date,
                "feature": pl.Utf8, "shap_value": pl.Float64,
                "rank": pl.Int32,
            })

        return pl.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Natural language narrative
    # ─────────────────────────────────────────────────────────────────────────

    def narrative(
        self,
        decomposition: pl.DataFrame,
        comparison: Optional[pl.DataFrame] = None,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> Dict[str, str]:
        """
        Generate a one-sentence natural language explanation per series.

        Parameters
        ----------
        decomposition:
            Output of ``decompose()``.
        comparison:
            Optional output of ``ForecastComparator.compare()``.  When
            provided, adds a sentence about the gap vs field/financial.

        Returns
        -------
        Dict mapping series_id → narrative string.
        """
        narratives: Dict[str, str] = {}

        for sid in decomposition[id_col].unique().to_list():
            s = decomposition.filter(pl.col(id_col) == sid)

            hist = s.filter(~pl.col("is_forecast"))
            fc = s.filter(pl.col("is_forecast"))

            if fc.is_empty():
                narratives[sid] = f"Series {sid}: no forecast available."
                continue

            # Total forecast (sum over horizon)
            fc_total = fc["value"].sum()
            fc_trend_mean = fc["trend"].drop_nulls().mean() if "trend" in fc.columns else None
            fc_seasonal_mean = fc["seasonal"].drop_nulls().mean() if "seasonal" in fc.columns else None

            # YoY comparison (last season_length periods of history)
            if not hist.is_empty() and len(hist) >= self.season_length:
                prior_year = hist.tail(self.season_length)["value"].sum()
                yoy_pct = (fc_total - prior_year) / abs(prior_year) * 100 if prior_year != 0 else None
            else:
                yoy_pct = None

            # Trend contribution to narrative
            if fc_trend_mean is not None and fc_seasonal_mean is not None and fc_total != 0:
                trend_share = fc_trend_mean / fc_total * 100
                seasonal_share = fc_seasonal_mean / fc_total * 100
                driver = "trend" if abs(trend_share) >= abs(seasonal_share) else "seasonality"
            else:
                driver = "historical pattern"

            # Build sentence
            if yoy_pct is not None:
                direction = "above" if yoy_pct > 0 else "below"
                line = (
                    f"Series {sid}: forecast is {abs(yoy_pct):.0f}% {direction} "
                    f"last year, primarily driven by {driver}."
                )
            else:
                line = f"Series {sid}: forecast driven by {driver}."

            # Add comparison context if available
            if comparison is not None and not comparison.is_empty():
                comp_s = comparison.filter(pl.col(id_col) == sid)
                gap_pct_cols = [c for c in comp_s.columns if c.endswith("_gap_pct")]
                for gc in gap_pct_cols:
                    source = gc.replace("_gap_pct", "")
                    avg_gap = comp_s[gc].drop_nulls().mean()
                    if avg_gap is not None:
                        gap_dir = "above" if avg_gap > 0 else "below"
                        line += (
                            f" System is {abs(avg_gap):.0f}% {gap_dir} "
                            f"{source} forecast on average."
                        )

            # Add uncertainty if available
            if comparison is not None and "uncertainty_ratio" in comparison.columns:
                comp_s = comparison.filter(pl.col(id_col) == sid)
                avg_unc = comp_s["uncertainty_ratio"].drop_nulls().mean()
                if avg_unc is not None:
                    if avg_unc > 0.75:
                        line += " Model uncertainty is HIGH — review P10/P90 range."
                    elif avg_unc > 0.40:
                        line += " Model uncertainty is moderate."
                    else:
                        line += " Model uncertainty is low."

            narratives[sid] = line

        return narratives

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _decompose_series(
        self,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classical additive decomposition: trend + seasonal + residual.

        Returns three arrays of len(values), with NaN at edge positions
        where the centered moving average cannot be computed.
        """
        n = len(values)
        sl = self.season_length

        # Trend: centered moving average
        trend = np.full(n, np.nan)
        half = sl // 2
        for i in range(half, n - half):
            trend[i] = np.mean(values[i - half: i + half + 1])

        # Seasonal: average de-trended value per seasonal position
        de_trended = values - trend
        seasonal_avg = np.zeros(sl)
        counts = np.zeros(sl)
        for i in range(n):
            if not np.isnan(de_trended[i]):
                pos = i % sl
                seasonal_avg[pos] += de_trended[i]
                counts[pos] += 1
        with np.errstate(invalid="ignore"):
            seasonal_avg = np.where(counts > 0, seasonal_avg / counts, 0.0)

        seasonal = np.array([seasonal_avg[i % sl] for i in range(n)])

        # Residual
        residual = values - trend - seasonal

        return trend, seasonal, residual
