"""
Tests for observability modules: PipelineContext, StructuredLogger,
MetricsEmitter, AlertDispatcher, CostEstimator, and PipelineScheduler.
"""
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# ── PipelineContext ──────────────────────────────────────────────────────────

class TestPipelineContext:

    def test_auto_generated_run_id(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext()
        assert len(ctx.run_id) == 32

    def test_custom_run_id(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext(run_id="custom123")
        assert ctx.run_id == "custom123"

    def test_lob_and_tags(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext(lob="retail", tags={"env": "prod"})
        assert ctx.lob == "retail"
        assert ctx.tags == {"env": "prod"}

    def test_child_context(self):
        from src.observability.context import PipelineContext
        parent = PipelineContext(run_id="abc", lob="surface")
        child = parent.child("backtest")
        assert child.run_id == "abc-backtest"
        assert child.parent_run_id == "abc"
        assert child.lob == "surface"

    def test_child_tags_independent(self):
        from src.observability.context import PipelineContext
        parent = PipelineContext(tags={"a": "1"})
        child = parent.child("sub")
        child.tags["b"] = "2"
        assert "b" not in parent.tags

    def test_elapsed_seconds(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext()
        assert ctx.elapsed_seconds >= 0

    def test_as_dict(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext(run_id="test", lob="retail")
        d = ctx.as_dict()
        assert d["run_id"] == "test"
        assert d["lob"] == "retail"
        assert "started_at" in d

    def test_str_representation(self):
        from src.observability.context import PipelineContext
        ctx = PipelineContext(run_id="x", lob="retail")
        assert "x" in str(ctx)
        assert "retail" in str(ctx)


# ── StructuredLogger ─────────────────────────────────────────────────────────

class TestStructuredLogger:

    def test_info_includes_context(self, caplog):
        from src.observability.context import PipelineContext
        from src.observability.logging import StructuredLogger

        ctx = PipelineContext(run_id="log_test", lob="retail")
        slog = StructuredLogger("test_logger", context=ctx)

        with caplog.at_level(logging.INFO, logger="test_logger"):
            slog.info("hello world", extra_key="val")

        assert len(caplog.records) == 1
        record_text = caplog.records[0].message
        parsed = json.loads(record_text)
        assert parsed["run_id"] == "log_test"
        assert parsed["lob"] == "retail"
        assert parsed["msg"] == "hello world"
        assert parsed["extra_key"] == "val"

    def test_without_context(self, caplog):
        from src.observability.logging import StructuredLogger

        slog = StructuredLogger("test_no_ctx")
        with caplog.at_level(logging.WARNING, logger="test_no_ctx"):
            slog.warning("no context")

        record = json.loads(caplog.records[0].message)
        assert record["msg"] == "no context"
        assert "run_id" not in record

    def test_set_context(self):
        from src.observability.context import PipelineContext
        from src.observability.logging import StructuredLogger

        slog = StructuredLogger("test_set_ctx")
        assert slog.context is None
        ctx = PipelineContext(run_id="new_ctx")
        slog.set_context(ctx)
        assert slog.context.run_id == "new_ctx"

    def test_setup_logging_json(self):
        from src.observability.logging import setup_logging
        setup_logging(format="json", level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_text(self):
        from src.observability.logging import setup_logging
        setup_logging(format="text", level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO


# ── MetricsEmitter ───────────────────────────────────────────────────────────

class TestMetricsEmitter:

    def test_counter(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log")
        emitter.counter("events", 5)

        assert len(emitter.recorded) == 1
        assert emitter.recorded[0]["type"] == "counter"
        assert emitter.recorded[0]["value"] == 5.0

    def test_gauge(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log")
        emitter.gauge("series_count", 1200.0)

        assert len(emitter.recorded) == 1
        assert emitter.recorded[0]["type"] == "gauge"
        assert emitter.recorded[0]["value"] == 1200.0

    def test_timer(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log")

        with emitter.timer("model_fit"):
            time.sleep(0.01)

        assert len(emitter.recorded) == 1
        assert "model_fit_duration_seconds" in emitter.recorded[0]["metric"]
        assert emitter.recorded[0]["value"] >= 0.01

    def test_prefix(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log", prefix="fp")
        emitter.counter("x")
        assert emitter.recorded[0]["metric"] == "fp.x"

    def test_no_prefix(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log", prefix="")
        emitter.counter("x")
        assert emitter.recorded[0]["metric"] == "x"

    def test_context_tags(self):
        from src.observability.context import PipelineContext
        from src.observability.metrics import MetricsEmitter

        ctx = PipelineContext(run_id="r1", lob="surface")
        emitter = MetricsEmitter(backend="log", context=ctx)
        emitter.gauge("count", 10)

        assert emitter.recorded[0]["run_id"] == "r1"
        assert emitter.recorded[0]["lob"] == "surface"

    def test_get_timers(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log")

        with emitter.timer("fit"):
            pass
        with emitter.timer("predict"):
            pass

        timers = emitter.get_timers()
        assert "forecast_platform.fit_duration_seconds" in timers
        assert "forecast_platform.predict_duration_seconds" in timers

    def test_reset(self):
        from src.observability.metrics import MetricsEmitter
        emitter = MetricsEmitter(backend="log")
        emitter.counter("x")
        assert len(emitter.recorded) == 1
        emitter.reset()
        assert len(emitter.recorded) == 0

    def test_set_context(self):
        from src.observability.context import PipelineContext
        from src.observability.metrics import MetricsEmitter

        emitter = MetricsEmitter(backend="log")
        emitter.gauge("x", 1)
        assert "run_id" not in emitter.recorded[0]

        ctx = PipelineContext(run_id="new")
        emitter.set_context(ctx)
        emitter.gauge("y", 2)
        assert emitter.recorded[1]["run_id"] == "new"


# ── AlertDispatcher ──────────────────────────────────────────────────────────

class _Severity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class _FakeAlert:
    """Minimal alert-like object for testing."""
    series_id: str = "SKU_001"
    metric: str = "wmape"
    severity: _Severity = _Severity.WARNING
    current_value: float = 0.35
    baseline_value: float = 0.15
    message: str = "WMAPE spike detected"


class TestAlertDispatcher:

    def test_dispatch_to_log(self, caplog):
        from src.observability.alerts import AlertConfig, AlertDispatcher

        config = AlertConfig(channels=["log"])
        dispatcher = AlertDispatcher(config)
        alerts = [_FakeAlert()]

        with caplog.at_level(logging.WARNING):
            count = dispatcher.dispatch(alerts)

        assert count == 1
        assert dispatcher.dispatched_count == 1

    def test_severity_filtering(self, caplog):
        from src.observability.alerts import AlertConfig, AlertDispatcher

        config = AlertConfig(channels=["log"], min_severity="critical")
        dispatcher = AlertDispatcher(config)

        # Warning alert should be filtered out
        warning_alert = _FakeAlert(severity=_Severity.WARNING)
        count = dispatcher.dispatch([warning_alert])
        assert count == 0

        # Critical alert should pass
        critical_alert = _FakeAlert(severity=_Severity.CRITICAL)
        with caplog.at_level(logging.WARNING):
            count = dispatcher.dispatch([critical_alert])
        assert count == 1

    def test_multiple_alerts(self, caplog):
        from src.observability.alerts import AlertConfig, AlertDispatcher

        config = AlertConfig(channels=["log"])
        dispatcher = AlertDispatcher(config)
        alerts = [_FakeAlert(series_id=f"S{i}") for i in range(5)]

        with caplog.at_level(logging.WARNING):
            count = dispatcher.dispatch(alerts)
        assert count == 5
        assert dispatcher.dispatched_count == 5

    def test_empty_alerts(self):
        from src.observability.alerts import AlertConfig, AlertDispatcher

        dispatcher = AlertDispatcher(AlertConfig())
        count = dispatcher.dispatch([])
        assert count == 0

    def test_webhook_channel_without_url(self, caplog):
        from src.observability.alerts import AlertConfig, AlertDispatcher

        config = AlertConfig(channels=["webhook"], webhook_url="")
        dispatcher = AlertDispatcher(config)
        alerts = [_FakeAlert()]

        # Should not crash even without a URL
        with caplog.at_level(logging.WARNING):
            count = dispatcher.dispatch(alerts)
        assert count == 1  # Alert dispatched but webhook skipped


# ── CostEstimator ────────────────────────────────────────────────────────────

class TestCostEstimator:

    def test_record_and_estimate(self):
        from src.observability.cost import CostEstimator
        est = CostEstimator(cost_per_second=0.001)
        est.record_model("lgbm", fit_seconds=10.0, predict_seconds=2.0)
        est.record_model("arima", fit_seconds=30.0, predict_seconds=1.0)

        result = est.build_estimate(run_id="test", series_count=100)
        assert result.total_seconds == 43.0
        assert abs(result.estimated_cost - 0.043) < 1e-6
        assert result.cost_per_series == result.estimated_cost / 100

    def test_zero_cost_rate(self):
        from src.observability.cost import CostEstimator
        est = CostEstimator(cost_per_second=0.0)
        est.record_model("naive", fit_seconds=1.0)
        result = est.build_estimate()
        assert result.estimated_cost == 0.0

    def test_zero_series(self):
        from src.observability.cost import CostEstimator
        est = CostEstimator(cost_per_second=0.01)
        est.record_model("x", fit_seconds=5.0)
        result = est.build_estimate(series_count=0)
        assert result.cost_per_series == 0.0

    def test_as_dict(self):
        from src.observability.cost import CostEstimator
        est = CostEstimator(cost_per_second=0.001)
        est.record_model("m1", fit_seconds=10)
        d = est.build_estimate(run_id="abc", series_count=50).as_dict()
        assert d["run_id"] == "abc"
        assert "total_seconds" in d
        assert "estimated_cost" in d
        assert "model_seconds" in d

    def test_reset(self):
        from src.observability.cost import CostEstimator
        est = CostEstimator()
        est.record_model("m", fit_seconds=5)
        est.reset()
        result = est.build_estimate()
        assert result.total_seconds == 0.0

    def test_record_from_emitter(self):
        from src.observability.cost import CostEstimator
        from src.observability.metrics import MetricsEmitter

        emitter = MetricsEmitter(backend="log")
        with emitter.timer("model_fit", model="lgbm"):
            time.sleep(0.01)

        est = CostEstimator(cost_per_second=0.001)
        est.record_from_emitter(emitter)
        result = est.build_estimate()
        assert result.total_seconds >= 0.01


# ── PipelineScheduler ────────────────────────────────────────────────────────

class TestPipelineScheduler:

    def test_run_once_success(self, tmp_path):
        from src.pipeline.scheduler import PipelineScheduler

        call_count = {"n": 0}
        def _mock_pipeline():
            call_count["n"] += 1

        scheduler = PipelineScheduler(
            pipeline_fn=_mock_pipeline,
            dead_letter_path=str(tmp_path / "dead"),
        )
        result = scheduler.run_once()
        assert result.status == "success"
        assert result.attempts == 1
        assert call_count["n"] == 1

    def test_run_once_retry_then_success(self, tmp_path):
        from src.pipeline.scheduler import PipelineScheduler

        attempts = {"n": 0}
        def _flaky_pipeline():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient error")

        scheduler = PipelineScheduler(
            pipeline_fn=_flaky_pipeline,
            max_retries=3,
            dead_letter_path=str(tmp_path / "dead"),
        )
        result = scheduler.run_once()
        assert result.status == "success"
        assert result.attempts == 2

    def test_run_once_dead_letter(self, tmp_path):
        from src.pipeline.scheduler import PipelineScheduler

        def _always_fail():
            raise RuntimeError("permanent failure")

        dl_path = tmp_path / "dead"
        scheduler = PipelineScheduler(
            pipeline_fn=_always_fail,
            max_retries=2,
            dead_letter_path=str(dl_path),
        )
        result = scheduler.run_once()
        assert result.status == "dead_letter"
        assert result.attempts == 2
        assert result.error == "permanent failure"

        # Dead-letter file should exist
        dl_files = list(dl_path.glob("dead_letter_*.json"))
        assert len(dl_files) == 1

    def test_history(self, tmp_path):
        from src.pipeline.scheduler import PipelineScheduler

        scheduler = PipelineScheduler(
            pipeline_fn=lambda: None,
            dead_letter_path=str(tmp_path / "dead"),
        )
        scheduler.run_once()
        scheduler.run_once()
        assert len(scheduler.history) == 2

    def test_stop(self, tmp_path):
        from src.pipeline.scheduler import PipelineScheduler
        scheduler = PipelineScheduler(
            pipeline_fn=lambda: None,
            dead_letter_path=str(tmp_path / "dead"),
        )
        scheduler.stop()
        assert scheduler._running is False


# ── ObservabilityConfig / AlertConfig schema ─────────────────────────────────

class TestObservabilityConfig:

    def test_defaults(self):
        from src.config.schema import ObservabilityConfig
        cfg = ObservabilityConfig()
        assert cfg.log_format == "text"
        assert cfg.log_level == "INFO"
        assert cfg.metrics_backend == "log"
        assert cfg.cost_per_second == 0.0

    def test_alert_config_defaults(self):
        from src.config.schema import AlertConfig
        cfg = AlertConfig()
        assert cfg.channels == ["log"]
        assert cfg.webhook_url == ""
        assert cfg.min_severity == "warning"
        assert cfg.webhook_timeout == 10

    def test_in_platform_config(self):
        from src.config.schema import PlatformConfig

        cfg = PlatformConfig()
        assert hasattr(cfg, "observability")
        assert cfg.observability.log_format == "text"
        assert cfg.observability.alerts.channels == ["log"]
