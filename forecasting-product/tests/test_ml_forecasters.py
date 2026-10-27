"""
Tests for ML-based forecasters (LightGBM, XGBoost) — src/forecasting/ml.py.

Covers:
  - fit / predict shape and columns
  - predict_quantiles ordering
  - manual fallback when mlforecast unavailable
  - get_params, set_future_features
  - error paths (predict before fit)
"""

import random
import unittest
from datetime import timedelta
from unittest.mock import patch

import polars as pl

from conftest import make_weekly_series as _make_weekly_series
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_name", ["lgbm_direct", "xgboost_direct"])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_native_learners_respect_forecaster_thread_budget(model_name, num_threads):
    from src.forecasting.registry import registry

    model = registry.build(model_name, num_threads=num_threads)
    assert model._get_learner().get_params()["n_jobs"] == num_threads
    assert model._get_quantile_learner(0.1).get_params()["n_jobs"] == num_threads


@pytest.mark.parametrize(
    "model_name,params_name",
    [("lgbm_direct", "lgbm_params"), ("xgboost_direct", "xgb_params")],
)
def test_explicit_native_thread_budget_is_preserved(model_name, params_name):
    from src.forecasting.registry import registry

    model = registry.build(model_name, num_threads=1, **{params_name: {"n_jobs": 2}})
    assert model._get_learner().get_params()["n_jobs"] == 2
    assert model._get_quantile_learner(0.1).get_params()["n_jobs"] == 2


# --------------------------------------------------------------------------- #
#  LightGBM
# --------------------------------------------------------------------------- #

class TestLGBMDirectForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.ml import LGBMDirectForecaster
        return LGBMDirectForecaster()

    def test_fit_predict_shape(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=8)
        self.assertIn("series_id", out.columns)
        self.assertIn("week", out.columns)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)
        self.assertFalse(out["forecast"].is_null().any())

    def test_predict_quantiles(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)
        # p10 <= p50 <= p90 (row-wise)
        for row in out.iter_rows(named=True):
            self.assertLessEqual(row["forecast_p10"], row["forecast_p90"])

    def test_get_params(self):
        f = self._get_forecaster()
        p = f.get_params()
        self.assertEqual(p["model"], "LightGBM")
        self.assertIn("lags", p)

    def test_set_future_features(self):
        """set_future_features stores future features without error."""
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        # build future features for horizon=4
        future_rows = []
        max_date = df["week"].max()
        for s in ["SKU-000", "SKU-001"]:
            for h in range(1, 5):
                future_rows.append({
                    "series_id": s,
                    "week": max_date + timedelta(weeks=h),
                    "promo": 0.0,
                })
        future_df = pl.DataFrame(future_rows)
        f.set_future_features(future_df)
        # Verify it was stored in the feature manager (converted to pandas)
        self.assertIsNotNone(f._feature_mgr._future_features)
        self.assertEqual(len(f._feature_mgr._future_features), 8)


# --------------------------------------------------------------------------- #
#  XGBoost
# --------------------------------------------------------------------------- #

class TestXGBoostDirectForecaster(unittest.TestCase):

    def _get_forecaster(self):
        from src.forecasting.ml import XGBoostDirectForecaster
        return XGBoostDirectForecaster()

    def test_fit_predict_shape(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict(horizon=8)
        self.assertIn("forecast", out.columns)
        self.assertEqual(out.shape[0], 2 * 8)

    def test_predict_quantiles(self):
        f = self._get_forecaster()
        df = _make_weekly_series()
        f.fit(df)
        out = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
        self.assertIn("forecast_p10", out.columns)
        self.assertIn("forecast_p50", out.columns)
        self.assertIn("forecast_p90", out.columns)

    def test_get_params(self):
        f = self._get_forecaster()
        p = f.get_params()
        self.assertEqual(p["model"], "XGBoost")


# --------------------------------------------------------------------------- #
#  Manual fallback (mlforecast unavailable)
# --------------------------------------------------------------------------- #

class TestManualFallback(unittest.TestCase):

    def test_manual_fallback_predict(self):
        """When _HAS_MLFORECAST is False, fit/predict should still work."""
        import src.forecasting.ml as ml_mod
        original = ml_mod._HAS_MLFORECAST
        try:
            ml_mod._HAS_MLFORECAST = False
            from src.forecasting.ml import LGBMDirectForecaster
            f = LGBMDirectForecaster()
            df = _make_weekly_series()
            f.fit(df)
            out = f.predict(horizon=4)
            self.assertIn("forecast", out.columns)
            self.assertEqual(out.shape[0], 2 * 4)
        finally:
            ml_mod._HAS_MLFORECAST = original

    def test_predict_before_fit_raises(self):
        import src.forecasting.ml as ml_mod
        original = ml_mod._HAS_MLFORECAST
        try:
            ml_mod._HAS_MLFORECAST = False
            from src.forecasting.ml import LGBMDirectForecaster
            f = LGBMDirectForecaster()
            with self.assertRaises(RuntimeError):
                f.predict(horizon=4)
        finally:
            ml_mod._HAS_MLFORECAST = original

    def test_manual_fallback_quantiles_residual(self):
        """Quantile prediction falls back to residual-based when mlforecast off."""
        import src.forecasting.ml as ml_mod
        original = ml_mod._HAS_MLFORECAST
        try:
            ml_mod._HAS_MLFORECAST = False
            from src.forecasting.ml import LGBMDirectForecaster
            f = LGBMDirectForecaster()
            df = _make_weekly_series()
            f.fit(df)
            out = f.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])
            self.assertIn("forecast_p10", out.columns)
            self.assertIn("forecast_p50", out.columns)
            self.assertIn("forecast_p90", out.columns)
        finally:
            ml_mod._HAS_MLFORECAST = original

    def test_manual_empty_series(self):
        """Empty input should produce empty output with correct schema."""
        import src.forecasting.ml as ml_mod
        original = ml_mod._HAS_MLFORECAST
        try:
            ml_mod._HAS_MLFORECAST = False
            from src.forecasting.ml import LGBMDirectForecaster
            f = LGBMDirectForecaster()
            empty = pl.DataFrame({
                "series_id": pl.Series([], dtype=pl.Utf8),
                "week": pl.Series([], dtype=pl.Date),
                "quantity": pl.Series([], dtype=pl.Float64),
            })
            f.fit(empty)
            out = f.predict(horizon=4)
            self.assertEqual(out.shape[0], 0)
            self.assertIn("forecast", out.columns)
        finally:
            ml_mod._HAS_MLFORECAST = original


if __name__ == "__main__":
    unittest.main()
