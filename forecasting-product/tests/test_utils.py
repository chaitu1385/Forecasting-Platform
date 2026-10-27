"""Tests for src/utils/ — config.py, logger.py, gap_fill.py."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.utils.config import load_config
from src.utils.gap_fill import fill_gaps
from src.utils.logger import get_logger

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("key: value\nnumber: 42\n")
        result = load_config(str(cfg_file))
        assert result == {"key": "value", "number": 42}

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_returns_dict(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("a: 1\nb: [1, 2, 3]\n")
        result = load_config(str(cfg_file))
        assert isinstance(result, dict)

    def test_nested_config(self, tmp_path):
        cfg_file = tmp_path / "nested.yaml"
        cfg_file.write_text("outer:\n  inner: 99\n  list:\n    - x\n    - y\n")
        result = load_config(str(cfg_file))
        assert result["outer"]["inner"] == 99
        assert result["outer"]["list"] == ["x", "y"]

    def test_empty_yaml_returns_none(self, tmp_path):
        """Empty YAML file parses to None via yaml.safe_load."""
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        result = load_config(str(cfg_file))
        assert result is None


# ---------------------------------------------------------------------------
# logger.py
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger("test.logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_preserved(self):
        logger = get_logger("my.module.name")
        assert logger.name == "my.module.name"

    def test_level_set_when_provided(self):
        logger = get_logger("test.level.debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_default_level_set_when_no_handlers(self):
        # Use a unique name to avoid picking up an already-configured logger
        logger = get_logger("test.fresh.no.handlers.xyz123")
        assert logger.level == logging.INFO

    def test_does_not_override_level_when_already_has_handlers(self):
        # Get a logger and set it up once
        logger1 = get_logger("test.handlers.already.set")
        original_level = logger1.level
        # Second call without explicit level should not reset level
        logger2 = get_logger("test.handlers.already.set")
        assert logger1 is logger2  # same object
        assert logger2.level == original_level

    def test_explicit_level_overrides_regardless(self):
        logger = get_logger("test.explicit.level.override", level=logging.WARNING)
        assert logger.level == logging.WARNING


# ---------------------------------------------------------------------------
# gap_fill.py
# ---------------------------------------------------------------------------

def _make_weekly_df(series_ids, dates, values, time_col="week", id_col="series_id"):
    """Helper: build a simple weekly DataFrame."""
    rows = []
    for sid, d, v in zip(series_ids, dates, values):
        rows.append({id_col: sid, time_col: d, "quantity": v})
    return pl.DataFrame(rows)


class TestFillGaps:
    def test_contiguous_series_unchanged(self):
        start = date(2023, 1, 2)
        dates = [start + timedelta(weeks=i) for i in range(5)]
        df = _make_weekly_df(
            ["A"] * 5, dates, [10.0, 20.0, 30.0, 40.0, 50.0]
        )
        result = fill_gaps(df, time_col="week", id_col="series_id", target_col="quantity")
        assert result.shape[0] == 5

    def test_fills_missing_week_with_zero(self):
        start = date(2023, 1, 2)
        # Skip week 3 (index 2)
        dates = [start, start + timedelta(weeks=1), start + timedelta(weeks=3)]
        df = _make_weekly_df(["A"] * 3, dates, [10.0, 20.0, 40.0])
        result = fill_gaps(df, strategy="zero")
        # Should have 4 rows (weeks 0, 1, 2, 3)
        a_vals = result.filter(pl.col("series_id") == "A").sort("week")["quantity"].to_list()
        assert len(a_vals) == 4
        assert a_vals[2] == 0.0  # filled gap

    def test_forward_fill_strategy(self):
        start = date(2023, 1, 2)
        dates = [start, start + timedelta(weeks=2)]
        df = _make_weekly_df(["A"] * 2, dates, [10.0, 30.0])
        result = fill_gaps(df, strategy="forward_fill")
        a_vals = result.filter(pl.col("series_id") == "A").sort("week")["quantity"].to_list()
        # Gap at week 1 should be forward-filled from week 0 value
        assert len(a_vals) == 3
        assert a_vals[1] == 10.0  # forward-filled

    def test_multiple_series_each_filled(self):
        start = date(2023, 1, 2)
        # Series A: weeks 0, 2 (gap at 1)
        # Series B: weeks 0, 2 (gap at 1)
        dates = [start, start + timedelta(weeks=2), start, start + timedelta(weeks=2)]
        sids = ["A", "A", "B", "B"]
        df = _make_weekly_df(sids, dates, [10.0, 30.0, 100.0, 300.0])
        result = fill_gaps(df, strategy="zero")
        assert result.filter(pl.col("series_id") == "A").shape[0] == 3
        assert result.filter(pl.col("series_id") == "B").shape[0] == 3

    def test_empty_dataframe_returned_as_is(self):
        df = pl.DataFrame({"series_id": [], "week": [], "quantity": []}).cast(
            {"week": pl.Date, "quantity": pl.Float64}
        )
        result = fill_gaps(df)
        assert result.is_empty()

    def test_sorted_output(self):
        start = date(2023, 1, 2)
        dates = [start + timedelta(weeks=4), start, start + timedelta(weeks=2)]
        df = _make_weekly_df(["A"] * 3, dates, [40.0, 10.0, 30.0])
        result = fill_gaps(df)
        weeks = result["week"].to_list()
        assert weeks == sorted(weeks)

    def test_custom_fill_value(self):
        start = date(2023, 1, 2)
        dates = [start, start + timedelta(weeks=2)]
        df = _make_weekly_df(["A"] * 2, dates, [10.0, 30.0])
        result = fill_gaps(df, fill_value=-1.0, strategy="zero")
        a_vals = result.filter(pl.col("series_id") == "A").sort("week")["quantity"].to_list()
        assert a_vals[1] == -1.0

    def test_monthly_frequency(self):
        """fill_gaps should produce contiguous monthly grid."""
        dates = [date(2023, 1, 1), date(2023, 3, 1)]  # skip February
        df = _make_weekly_df(["A"] * 2, dates, [10.0, 30.0], time_col="week")
        result = fill_gaps(df, freq="M", strategy="zero")
        assert result.filter(pl.col("series_id") == "A").shape[0] == 3

    def test_daily_frequency(self):
        dates = [date(2023, 1, 1), date(2023, 1, 3)]  # gap on Jan 2
        df = _make_weekly_df(["A"] * 2, dates, [5.0, 15.0], time_col="week")
        result = fill_gaps(df, freq="D", strategy="zero")
        assert result.filter(pl.col("series_id") == "A").shape[0] == 3

    def test_unknown_frequency_is_not_guessed_as_weekly(self):
        """An invalid configured frequency must not silently become weekly."""
        start = date(2023, 1, 2)
        dates = [start, start + timedelta(weeks=1)]
        df = _make_weekly_df(["A"] * 2, dates, [1.0, 2.0])
        with pytest.raises(ValueError, match="Unsupported frequency"):
            fill_gaps(df, freq="UNKNOWN")
        with pytest.raises(ValueError, match="Unsupported frequency"):
            fill_gaps(df.head(0), freq="UNKNOWN")
