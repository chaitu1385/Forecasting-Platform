"""Tests for API security hardening: path traversal prevention and upload size limits."""
import io
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import MAX_UPLOAD_BYTES, validate_path_param, validate_upload_size

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------- #
#  validate_path_param unit tests
# --------------------------------------------------------------------------- #


class TestValidatePathParam:
    """Unit tests for the path traversal prevention function."""

    @pytest.mark.parametrize(
        "value",
        [
            "retail",
            "grocery",
            "my-lob",
            "my_lob",
            "lob123",
            "v2.0",
            "LOB-Retail_v1.2",
        ],
    )
    def test_valid_values_pass(self, value):
        assert validate_path_param(value, "lob") == value

    @pytest.mark.parametrize(
        "value",
        [
            "../etc/passwd",
            "../../secrets",
            "foo/bar",
            "foo\\bar",
            "",
            ".hidden",
            "-starts-with-dash",
            "a" * 200,  # exceeds 128 char limit
        ],
    )
    def test_traversal_attempts_rejected(self, value):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_path_param(value, "lob")
        assert exc_info.value.status_code == 400

    def test_none_rejected(self):
        from fastapi import HTTPException

        with pytest.raises((HTTPException, TypeError)):
            validate_path_param(None, "lob")


# --------------------------------------------------------------------------- #
#  validate_upload_size unit tests
# --------------------------------------------------------------------------- #


class TestValidateUploadSize:
    """Unit tests for file upload size limiting."""

    def test_small_file_passes(self):
        import asyncio
        from fastapi import UploadFile

        content = b"hello world"
        file = UploadFile(file=io.BytesIO(content), filename="test.csv")
        result = asyncio.run(
            validate_upload_size(file, max_bytes=1024)
        )
        assert result == content

    def test_oversized_file_rejected(self):
        import asyncio
        from fastapi import HTTPException, UploadFile


        content = b"x" * 2000
        file = UploadFile(file=io.BytesIO(content), filename="big.csv")
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                validate_upload_size(file, max_bytes=1000)
            )
        assert exc_info.value.status_code == 413


# --------------------------------------------------------------------------- #
#  Integration tests — path traversal via API
# --------------------------------------------------------------------------- #


def _make_data(data_dir: Path, lob="retail"):
    """Write minimal forecast + actuals for a LOB."""
    rng = np.random.RandomState(42)
    rows = []
    base = date(2024, 1, 1)
    for i in range(3):
        for w in range(13):
            rows.append({
                "series_id": f"sku_{i}",
                "week": base + timedelta(weeks=w),
                "forecast": float(rng.normal(100, 15)),
                "model": "auto_arima",
            })
    df = pl.DataFrame(rows)
    fc_dir = data_dir / "forecasts" / lob
    fc_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(fc_dir / f"forecast_{lob}_2024-01-01.parquet")

    act_rows = []
    base_act = date(2023, 1, 2)
    for i in range(3):
        for w in range(52):
            act_rows.append({
                "series_id": f"sku_{i}",
                "week": base_act + timedelta(weeks=w),
                "quantity": float(max(0, rng.normal(100, 30))),
            })
    act_df = pl.DataFrame(act_rows)
    hist_dir = data_dir / "history" / lob
    hist_dir.mkdir(parents=True, exist_ok=True)
    act_df.write_parquet(hist_dir / "actuals.parquet")


@pytest.fixture
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        _make_data(data_dir, "retail")
        (data_dir / "metrics").mkdir()
        app = create_app(
            data_dir=str(data_dir), metrics_dir=str(data_dir / "metrics"),
            auth_enabled=False, dev_mode=True, jwt_secret="",
        )
        yield TestClient(app)


class TestPathTraversalPrevention:
    """Ensure path traversal attempts get 400, not 404/500."""

    def test_forecast_hidden_dir_rejected(self, client):
        resp = client.get("/forecast/.hidden")
        assert resp.status_code == 400


# --------------------------------------------------------------------------- #
#  Rate limiter memory management tests
# --------------------------------------------------------------------------- #


class TestRateLimiterMemory:
    """Verify that the rate limiter evicts stale IP entries."""

    def test_stale_ips_evicted_from_hits_dict(self):
        """IPs with no recent requests should not linger in memory."""
        import time
        from unittest.mock import AsyncMock, MagicMock

        import asyncio

        from src.api.app import _RateLimitMiddleware

        app_mock = MagicMock()
        mw = _RateLimitMiddleware(app_mock, max_requests=10, window_seconds=1)

        # Simulate a past request from a now-stale IP
        stale_time = time.monotonic() - 10  # 10 seconds ago
        mw._hits["old-ip"] = [stale_time]

        assert "old-ip" in mw._hits

        # Simulate old-ip making a new request — old timestamps pruned
        request = MagicMock()
        request.client.host = "old-ip"
        call_next = AsyncMock()

        asyncio.run(mw.dispatch(request, call_next))

        # old-ip should still exist (it just made a new request)
        assert "old-ip" in mw._hits
        # But only with the new timestamp, not the stale one
        assert len(mw._hits["old-ip"]) == 1
        assert mw._hits["old-ip"][0] > stale_time

    def test_new_ip_not_precreated(self):
        """Accessing _hits for an unknown IP should not create an empty entry."""
        import time

        from src.api.app import _RateLimitMiddleware
        from unittest.mock import MagicMock

        app_mock = MagicMock()
        mw = _RateLimitMiddleware(app_mock, max_requests=10, window_seconds=60)

        # dict.get() should not auto-create keys (unlike defaultdict)
        _ = mw._hits.get("never-seen", [])
        assert "never-seen" not in mw._hits

    def test_forecast_space_rejected(self, client):
        resp = client.get("/forecast/%20")
        assert resp.status_code == 400

    def test_valid_lob_works(self, client):
        resp = client.get("/forecast/retail")
        assert resp.status_code == 200

    def test_valid_lob_with_hyphens_works(self, client):
        # Not found, but validated (no traversal)
        resp = client.get("/forecast/my-new-lob")
        assert resp.status_code == 404  # valid format, just no data

    def test_bi_export_lob_traversal_via_query(self, client):
        resp = client.post("/governance/export/leaderboard?lob=..&run_type=backtest")
        assert resp.status_code == 400

    def test_bi_export_hidden_lob_via_query(self, client):
        resp = client.post("/governance/export/leaderboard?lob=.hidden&run_type=backtest")
        assert resp.status_code == 400

    def test_pipeline_backtest_lob_rejected(self, client):
        csv_content = b"series_id,week,quantity\nsku_0,2024-01-01,100\n"
        resp = client.post(
            "/pipeline/backtest?lob=..",
            files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 400

    def test_pipeline_backtest_hidden_lob_rejected(self, client):
        csv_content = b"series_id,week,quantity\nsku_0,2024-01-01,100\n"
        resp = client.post(
            "/pipeline/backtest?lob=.hidden",
            files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 400
