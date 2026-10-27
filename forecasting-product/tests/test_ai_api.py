"""Integration tests for AI API endpoints using FastAPI TestClient."""

import json
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from src.api.app import create_app

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_metrics_parquet(metrics_dir: Path, lob="retail", n_series=3, n_weeks=30):
    """Write a synthetic metric store Parquet file."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2024, 1, 1)
    for i in range(n_series):
        for w in range(n_weeks):
            actual = float(rng.normal(100, 20))
            forecast = actual * (1 + rng.normal(0, 0.15))
            rows.append({
                "series_id": f"series_{i}",
                "model_id": "auto_arima",
                "target_week": base + timedelta(weeks=w),
                "actual": actual,
                "forecast": forecast,
                "wmape": abs(actual - forecast) / max(abs(actual), 1e-9),
                "normalized_bias": (forecast - actual) / max(abs(actual), 1e-9),
                "lob": lob,
                "run_type": "backtest",
            })
    df = pl.DataFrame(rows)
    out_dir = metrics_dir / "backtest" / f"lob={lob}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "metrics.parquet")
    return df


def _make_forecast_parquet(data_dir: Path, lob="retail", n_series=3, n_weeks=13):
    """Write a synthetic forecast Parquet file."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2024, 7, 1)
    for i in range(n_series):
        for w in range(n_weeks):
            rows.append({
                "series_id": f"series_{i}",
                "week": base + timedelta(weeks=w),
                "forecast": float(rng.normal(100, 15)),
                "model": "auto_arima",
            })
    df = pl.DataFrame(rows)
    fc_dir = data_dir / "forecasts" / lob
    fc_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(fc_dir / f"forecast_{lob}_2024-07-01.parquet")
    return df


# --------------------------------------------------------------------------- #
#  Mock AI engines
# --------------------------------------------------------------------------- #

class MockNLQueryResult:
    answer = "Test answer for the question"
    supporting_data = {"wmape": 0.12}
    confidence = "high"
    sources_used = ["history", "forecast"]


class MockTriageResult:
    class Alert:
        series_id = "series_0"
        metric = "accuracy"
        severity = "critical"
        business_impact_score = 90.0
        suggested_action = "Retrain model"
        reasoning = "High impact"
        original_message = "WMAPE degraded"

    ranked_alerts = [Alert()]
    executive_summary = "Test summary"
    total_alerts = 1
    critical_count = 1
    warning_count = 0


class MockConfigResult:
    class Rec:
        field_path = "forecast.forecasters"
        current_value = ["naive_seasonal"]
        recommended_value = ["naive_seasonal", "auto_arima"]
        reasoning = "AutoARIMA outperforms"
        expected_impact = "5% WMAPE improvement"
        risk = "low"

    recommendations = [Rec()]
    overall_assessment = "Test assessment"
    risk_summary = "Low risk overall"


class MockCommentaryResult:
    class Metric:
        name = "WMAPE"
        value = 0.15
        unit = "%"
        trend = "stable"

    executive_summary = "Test executive summary"
    key_metrics = [Metric()]
    exceptions = ["series_0: accuracy drift"]
    action_items = ["Retrain models"]


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #

class TestAIEndpointsBase(unittest.TestCase):
    """Base class with shared setUp for AI endpoint tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = Path(self.tmpdir) / "data"
        self.metrics_dir = Path(self.tmpdir) / "metrics"
        self.data_dir.mkdir(parents=True)
        self.metrics_dir.mkdir(parents=True)

        self.app = create_app(
            data_dir=str(self.data_dir),
            metrics_dir=str(self.metrics_dir),
            auth_enabled=False,
            dev_mode=True,
            jwt_secret="",
        )
        self.client = TestClient(self.app)


class TestAIExplainEndpoint(TestAIEndpointsBase):

    @patch("src.ai.nl_query.NaturalLanguageQueryEngine")
    def test_explain_success(self, MockEngine):
        _make_forecast_parquet(self.data_dir)
        mock_instance = MagicMock()
        mock_instance.query.return_value = MockNLQueryResult()
        MockEngine.return_value = mock_instance

        response = self.client.post("/ai/explain", json={
            "series_id": "series_0",
            "question": "Why did forecast change?",
            "lob": "retail",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)

    def test_explain_without_mock_returns_200(self):
        """Even without Claude, endpoint should return (graceful degradation)."""
        _make_forecast_parquet(self.data_dir)
        response = self.client.post("/ai/explain", json={
            "series_id": "series_0",
            "question": "Why?",
            "lob": "retail",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("unavailable", data["answer"].lower())


class TestAITriageEndpoint(TestAIEndpointsBase):

    @patch("src.ai.anomaly_triage.AnomalyTriageEngine")
    def test_triage_success(self, MockEngine):
        _make_metrics_parquet(self.metrics_dir)
        mock_instance = MagicMock()
        mock_instance.query.return_value = MockTriageResult()
        MockEngine.return_value = mock_instance

        response = self.client.post("/ai/triage", json={
            "lob": "retail",
            "run_type": "backtest",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("executive_summary", data)
        self.assertIn("ranked_alerts", data)

    def test_triage_lob_not_found(self):
        response = self.client.post("/ai/triage", json={
            "lob": "nonexistent",
            "run_type": "backtest",
        })
        self.assertIn(response.status_code, [404, 500])


class TestAIRecommendConfigEndpoint(TestAIEndpointsBase):

    @patch("src.ai.config_tuner.ConfigTunerEngine")
    def test_recommend_config_success(self, MockEngine):
        _make_metrics_parquet(self.metrics_dir)
        mock_instance = MagicMock()
        mock_instance.recommend.return_value = MockConfigResult()
        MockEngine.return_value = mock_instance

        response = self.client.post("/ai/recommend-config", json={
            "lob": "retail",
            "run_type": "backtest",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)
        self.assertIn("overall_assessment", data)


class TestAICommentaryEndpoint(TestAIEndpointsBase):

    @patch("src.ai.commentary.CommentaryEngine")
    def test_commentary_success(self, MockEngine):
        _make_metrics_parquet(self.metrics_dir)
        mock_instance = MagicMock()
        mock_instance.generate.return_value = MockCommentaryResult()
        MockEngine.return_value = mock_instance

        response = self.client.post("/ai/commentary", json={
            "lob": "retail",
            "run_type": "backtest",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("executive_summary", data)
        self.assertIn("key_metrics", data)

    def test_commentary_lob_not_found(self):
        response = self.client.post("/ai/commentary", json={
            "lob": "nonexistent",
            "run_type": "backtest",
        })
        self.assertIn(response.status_code, [404, 500])


if __name__ == "__main__":
    unittest.main()
