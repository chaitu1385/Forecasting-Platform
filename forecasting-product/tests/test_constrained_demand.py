"""Tests for ConstrainedDemandEstimator — capacity and business-rule constraints."""

import math
from datetime import date, timedelta
from typing import List

import polars as pl
import pytest
from src.config.schema import ConstraintConfig
from src.forecasting.base import BaseForecaster
from src.forecasting.constrained import (
    ConstrainedDemandEstimator,
    apply_point_constraints,
)
from src.forecasting.registry import registry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake base forecaster for deterministic testing
# ---------------------------------------------------------------------------


class _StubForecaster(BaseForecaster):
    """Returns pre-set forecast values for testing."""

    name = "stub"

    def __init__(self, forecast_values: pl.DataFrame):
        self._forecast = forecast_values

    def fit(self, df, target_col="quantity", time_col="week", id_col="series_id"):
        pass  # no-op

    def predict(self, horizon, id_col="series_id", time_col="week"):
        return self._forecast

    def predict_quantiles(self, horizon, quantiles, id_col="series_id", time_col="week"):
        df = self._forecast.clone()
        for q in quantiles:
            col = f"forecast_p{int(round(q * 100))}"
            # Scale forecast by quantile for deterministic spread
            df = df.with_columns(
                (pl.col("forecast") * (0.5 + q)).alias(col)
            )
        return df.drop("forecast")


def _make_forecast_df(
    series: List[str],
    n_weeks: int = 4,
    base_value: float = 100.0,
    start_date: date = date(2024, 1, 1),
) -> pl.DataFrame:
    """Build a stub forecast DataFrame."""
    rows = []
    for sid in series:
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": start_date + timedelta(weeks=w),
                "forecast": base_value,
            })
    return pl.DataFrame(rows).with_columns(
        pl.col("week").cast(pl.Date),
        pl.col("forecast").cast(pl.Float64),
    )


# ===========================================================================
# TestElementWiseConstraints
# ===========================================================================


class TestElementWiseConstraints:
    """Test per-row clipping: non-negativity and capacity."""

    def test_non_negativity_clipping(self):
        """Negative forecasts are clipped to min_demand (default 0)."""
        fdf = _make_forecast_df(["a"], base_value=-50.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=True, min_demand=0.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        assert result["forecast"].min() >= 0.0

    def test_global_max_capacity(self):
        """Forecasts exceeding max_capacity are clipped down."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=True, max_capacity=150.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        assert result["forecast"].max() <= 150.0

    def test_per_series_capacity_column(self):
        """Capacity extracted from data column applies per-series."""
        # Training data with capacity column
        train = pl.DataFrame({
            "series_id": ["a", "a", "b", "b"],
            "week": [date(2023, 1, 2)] * 4,
            "quantity": [10.0, 10.0, 10.0, 10.0],
            "max_cap": [80.0, 80.0, 120.0, 120.0],
        }).with_columns(
            pl.col("week").cast(pl.Date),
            pl.col("quantity").cast(pl.Float64),
        )

        fdf = _make_forecast_df(["a", "b"], base_value=100.0)
        stub = _StubForecaster(fdf)
        cfg = ConstraintConfig(enabled=True, capacity_column="max_cap")
        cde = ConstrainedDemandEstimator(stub, cfg)
        cde.validate_and_prepare(train, "quantity", "week", "series_id")
        cde.base.fit(pl.DataFrame())

        result = cde.predict(horizon=4)
        a_max = result.filter(pl.col("series_id") == "a")["forecast"].max()
        b_max = result.filter(pl.col("series_id") == "b")["forecast"].max()
        assert a_max <= 80.0
        assert b_max <= 100.0  # b's cap is 120, forecast is 100

    def test_disabled_passthrough(self):
        """When enabled=False, forecasts pass through unchanged."""
        fdf = _make_forecast_df(["a"], base_value=-50.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=False)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=4)
        # Negatives should remain
        assert result["forecast"].min() < 0.0


# ===========================================================================
# TestAggregateConstraints
# ===========================================================================


class TestAggregateConstraints:
    """Test per-period aggregate budget constraints."""

    def test_proportional_redistribution(self):
        """When sum exceeds budget, forecasts scale down proportionally."""
        # 4 series × 100 = 400 per period, budget = 300
        fdf = _make_forecast_df(["a", "b", "c", "d"], base_value=100.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                aggregate_max=300.0,
                proportional_redistribution=True,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        period_sum = result["forecast"].sum()
        assert abs(period_sum - 300.0) < 0.01

    def test_clip_largest_strategy(self):
        """Largest-first removes excess from a before touching tied series b."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                aggregate_max=300.0,
                proportional_redistribution=False,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        period_sum = result["forecast"].sum()
        assert abs(period_sum - 300.0) < 0.01
        assert result["forecast"].to_list() == [100.0, 200.0]

    def test_under_budget_unchanged(self):
        """No adjustment when total is within budget."""
        fdf = _make_forecast_df(["a", "b"], base_value=100.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, aggregate_max=500.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        # 200 < 500, so no change
        assert result["forecast"].sum() == 200.0

    def test_aggregate_plus_element_wise(self):
        """Aggregate and element-wise constraints work together."""
        fdf = _make_forecast_df(["a", "b"], base_value=200.0, n_weeks=1)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(
                enabled=True,
                max_capacity=180.0,
                aggregate_max=300.0,
            ),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict(horizon=1)
        # Element-wise clips to 180 first (360 total), then aggregate scales to 300
        assert result["forecast"].max() <= 180.0
        assert result["forecast"].sum() <= 300.01


# ===========================================================================
# TestQuantileConstraints
# ===========================================================================


class TestQuantileConstraints:
    """Test quantile clipping and monotonicity."""

    def test_quantiles_clipped_to_bounds(self):
        """All quantile values respect [min_demand, max_capacity]."""
        fdf = _make_forecast_df(["a"], base_value=200.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, min_demand=0.0, max_capacity=250.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])

        for col in ["forecast_p10", "forecast_p50", "forecast_p90"]:
            assert result[col].min() >= 0.0
            assert result[col].max() <= 250.0

    def test_monotonicity_preserved(self):
        """After clipping, p10 ≤ p50 ≤ p90 holds for every row."""
        fdf = _make_forecast_df(["a"], base_value=200.0)
        stub = _StubForecaster(fdf)
        # Tight cap forces clipping that could break order
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, min_demand=0.0, max_capacity=250.0),
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.5, 0.9])

        for row in result.iter_rows(named=True):
            assert row["forecast_p10"] <= row["forecast_p50"]
            assert row["forecast_p50"] <= row["forecast_p90"]

    def test_quantile_disabled_passthrough(self):
        """When disabled, quantiles pass through unchanged."""
        fdf = _make_forecast_df(["a"], base_value=1000.0)
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub, ConstraintConfig(enabled=False, max_capacity=100.0)
        )
        cde.fit(pl.DataFrame())
        result = cde.predict_quantiles(horizon=4, quantiles=[0.1, 0.9])
        # Should exceed 100 since constraints are disabled
        assert result["forecast_p90"].max() > 100.0


# ===========================================================================
# TestRegistryIntegration
# ===========================================================================


class TestRegistryIntegration:
    """Verify the forecaster is registered."""

    def test_in_registry(self):
        assert "constrained_demand" in registry.available

    def test_get_from_registry(self):
        cls = registry.get("constrained_demand")
        assert cls is ConstrainedDemandEstimator


# ===========================================================================
# TestGetParams
# ===========================================================================


class TestGetParams:
    """Verify parameter reporting."""

    def test_params_include_base_and_constraints(self):
        fdf = _make_forecast_df(["a"])
        stub = _StubForecaster(fdf)
        cde = ConstrainedDemandEstimator(
            stub,
            ConstraintConfig(enabled=True, max_capacity=500.0),
        )
        params = cde.get_params()
        assert params["base"] == "stub"
        assert params["max_capacity"] == 500.0


# ===========================================================================
# TestWithDifferentBases
# ===========================================================================


class TestWithDifferentBases:
    """Verify composition works with real forecasters."""

    def test_with_seasonal_naive(self):
        from src.forecasting.naive import SeasonalNaiveForecaster

        naive = SeasonalNaiveForecaster()
        cde = ConstrainedDemandEstimator(
            naive, ConstraintConfig(enabled=True, max_capacity=500.0)
        )
        # Just verify construction — no fit needed for this test
        assert cde.base.name == "naive_seasonal"
        assert repr(cde).startswith("ConstrainedDemandEstimator")

    def test_with_lgbm(self):
        from src.forecasting.ml import LGBMDirectForecaster

        lgbm = LGBMDirectForecaster()
        cde = ConstrainedDemandEstimator(
            lgbm, ConstraintConfig(enabled=True, min_demand=10.0)
        )
        assert cde.base.name == "lgbm_direct"
        params = cde.get_params()
        assert params["min_demand"] == 10.0


def _point_frame(values=(100.0, 100.0)):
    return _make_forecast_df(
        [chr(ord("a") + index) for index in range(len(values))], n_weeks=1,
    ).with_columns(pl.Series("forecast", values))


class _QuantileFrameForecaster(_StubForecaster):
    def predict_quantiles(self, horizon, quantiles, id_col="series_id", time_col="week"):
        return self._forecast


class TestPointConstraintContract:
    @pytest.mark.parametrize("proportional,expected", [(True, [25.0, 25.0]), (False, [0.0, 50.0])])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_distinct_hand_allocations_and_input_order(self, proportional, expected, reverse):
        frame = _point_frame()
        if reverse:
            frame = frame.reverse()
        original = frame.clone()
        config = ConstraintConfig(enabled=True, aggregate_max=50, proportional_redistribution=proportional)
        result = apply_point_constraints(frame, config)
        assert dict(zip(result["series_id"], result["forecast"])) == dict(zip(["a", "b"], expected))
        assert result.drop("forecast").equals(frame.drop("forecast"))
        assert frame.equals(original)
        wrapper = ConstrainedDemandEstimator(_StubForecaster(frame), config)
        wrapper.fit(pl.DataFrame())
        assert wrapper.predict(1).equals(result)

    @pytest.mark.parametrize("proportional", [False, True])
    @pytest.mark.parametrize(
        "values,floor,budget,expected",
        [
            ([100.0, 2.0], 10.0, 30.0, [20.0, 10.0]),
            ([100.0, 2.0], 10.0, 20.0, [10.0, 10.0]),
            ([100.0, 2.0], 0.0, 0.0, [0.0, 0.0]),
            ([-5.0, 2.0], 0.0, 10.0, [0.0, 2.0]),
            ([10.0, 15.0], 10.0, 25.0, [10.0, 15.0]),
            ([0.0, 0.0], 0.0, 0.0, [0.0, 0.0]),
        ],
    )
    def test_feasible_floors_zero_and_under_budget(self, proportional, values, floor, budget, expected):
        result = apply_point_constraints(
            _point_frame(values),
            ConstraintConfig(
                enabled=True, min_demand=floor, aggregate_max=budget,
                proportional_redistribution=proportional,
            ),
        )
        assert result["forecast"].to_list() == pytest.approx(expected, abs=1e-9, rel=1e-12)
        assert math.fsum(result["forecast"]) <= budget + 1e-9 + 1e-12 * budget

    def test_largest_first_descending_values_then_string_identity(self):
        result = apply_point_constraints(
            _point_frame([120.0, 80.0, 20.0]),
            ConstraintConfig(enabled=True, min_demand=10, aggregate_max=100, proportional_redistribution=False),
        )
        assert result["forecast"].to_list() == [10.0, 70.0, 20.0]
        numeric_ids = _point_frame().with_columns(pl.Series("series_id", [2, 10]))
        result = apply_point_constraints(
            numeric_ids, ConstraintConfig(enabled=True, aggregate_max=50, proportional_redistribution=False),
        )
        assert result["series_id"].to_list() == [2, 10]
        assert result["forecast"].to_list() == [50.0, 0.0]

    @pytest.mark.parametrize("budget", [0.0, 1.0])
    def test_largest_first_does_not_lose_small_remaining_values(self, budget):
        result = apply_point_constraints(
            _point_frame([1e16, 1.0]),
            ConstraintConfig(enabled=True, aggregate_max=budget, proportional_redistribution=False),
        )
        assert result["forecast"].to_list() == [0.0, budget]

    @pytest.mark.parametrize("proportional,expected", [(True, [25.0, 25.0]), (False, [0.0, 50.0])])
    def test_finite_inputs_with_overflowing_preallocation_sum(self, proportional, expected):
        result = apply_point_constraints(
            _point_frame([1e308, 1e308]),
            ConstraintConfig(enabled=True, aggregate_max=50, proportional_redistribution=proportional),
        )
        assert result["forecast"].to_list() == expected

    @pytest.mark.parametrize("field", ["min_demand", "max_capacity", "aggregate_max"])
    @pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), float("-inf"), "10", True])
    def test_invalid_limits_are_explicit(self, field, invalid):
        config = ConstraintConfig(enabled=True)
        setattr(config, field, invalid)
        with pytest.raises(ValueError, match="invalid_constraint_limit"):
            apply_point_constraints(_point_frame(), config)

    @pytest.mark.parametrize(
        "settings",
        [
            {"min_demand": 40, "aggregate_max": 50},
            {"min_demand": 10, "max_capacity": 9},
        ],
    )
    def test_infeasible_limits_do_not_return_violated_success(self, settings):
        frame = _point_frame()
        before = frame.clone()
        with pytest.raises(ValueError, match="infeasible_constraints"):
            apply_point_constraints(frame, ConstraintConfig(enabled=True, **settings))
        assert frame.equals(before)

    @pytest.mark.parametrize("column", ["series_id", "week", "forecast"])
    def test_missing_required_columns(self, column):
        with pytest.raises(ValueError, match="required identity, time or value column"):
            apply_point_constraints(_point_frame().drop(column), ConstraintConfig(enabled=True))

    @pytest.mark.parametrize(
        "frame",
        [
            _point_frame().head(0),
            pl.concat([_point_frame(), _point_frame().head(1)]),
            _point_frame().with_columns(pl.Series("series_id", [None, "b"])),
            _point_frame().with_columns(pl.Series("series_id", [" ", "b"])),
            _point_frame().with_columns(pl.Series("series_id", [float("nan"), 2.0])),
            _point_frame().with_columns(pl.Series("week", [None, date(2024, 1, 1)])),
            _point_frame().with_columns(pl.Series("week", [float("inf"), 1.0])),
            _point_frame().with_columns(pl.Series("series_id", [["a"], ["b"]])),
        ],
        ids=["empty", "duplicate", "null-id", "blank-id", "nonfinite-id", "null-time", "nonfinite-time", "nested-id"],
    )
    def test_invalid_populations(self, frame):
        with pytest.raises(ValueError, match="invalid_constraint_frame"):
            apply_point_constraints(frame, ConstraintConfig(enabled=True))

    @pytest.mark.parametrize(
        "values",
        [
            [float("nan"), 1.0], [float("inf"), 1.0], [float("-inf"), 1.0],
            [None, 1.0], ["100", "1"], [True, False],
        ],
    )
    def test_values_must_be_finite_numeric(self, values):
        with pytest.raises(ValueError, match="finite numeric"):
            apply_point_constraints(_point_frame(values), ConstraintConfig(enabled=True))

    @pytest.mark.parametrize("names", [("x", "x", "forecast"), ("", "week", "forecast")])
    def test_column_names_must_be_distinct_nonempty(self, names):
        with pytest.raises(ValueError, match="distinct and nonempty"):
            apply_point_constraints(_point_frame(), ConstraintConfig(enabled=True), *names)

    @pytest.mark.parametrize("proportional", [True, False])
    def test_custom_columns_and_internal_name_collisions(self, proportional):
        frame = _point_frame().rename({"series_id": "_cap", "week": "_period_total", "forecast": "_q_list"})
        frame = frame.with_columns(
            pl.Series("_q_sorted", [["unchanged"], ["also unchanged"]]),
            pl.Series("rows", [17, 91]), pl.Series("period", ["metadata-a", "metadata-b"]),
        )
        settings = ConstraintConfig(enabled=True, aggregate_max=50, proportional_redistribution=proportional)
        result = apply_point_constraints(frame, settings, "_cap", "_period_total", "_q_list")
        assert result.columns == frame.columns
        assert result.drop("_q_list").equals(frame.drop("_q_list"))
        wrapper = ConstrainedDemandEstimator(_StubForecaster(frame), settings)
        assert wrapper.predict(1, "_cap", "_period_total", value_col="_q_list").equals(result)

    @pytest.mark.parametrize("proportional", [True, False])
    def test_full_population_multiple_periods_and_preserved_metadata(self, proportional):
        frame = _make_forecast_df([f"item-{i:03}" for i in range(210)], n_weeks=2)
        frame = frame.with_columns(pl.int_range(0, pl.len()).alias("_cap"))
        result = apply_point_constraints(
            frame, ConstraintConfig(
                enabled=True, min_demand=5, max_capacity=80,
                aggregate_max=1100, proportional_redistribution=proportional,
            ),
        )
        assert result.height == 420
        assert result.drop("forecast").equals(frame.drop("forecast"))
        assert result["forecast"].is_between(5, 80).all()
        for values in result.group_by("week").agg("forecast")["forecast"]:
            assert math.fsum(values) == pytest.approx(1100, abs=1e-9, rel=1e-12)
        if proportional:
            assert result["forecast"].to_list() == pytest.approx([1100 / 210] * 420, abs=1e-9, rel=1e-12)
        else:
            assert result.tail(2)["forecast"].to_list() == [55.0, 55.0]
            assert result.head(418)["forecast"].to_list() == [5.0] * 418

    def test_invalid_value_beyond_preview_is_not_ignored(self):
        frame = _make_forecast_df([f"item-{i}" for i in range(201)], n_weeks=1)
        frame = frame.with_columns(pl.Series("forecast", [100.0] * 200 + [float("nan")]))
        with pytest.raises(ValueError, match="finite numeric"):
            apply_point_constraints(frame, ConstraintConfig(enabled=True, aggregate_max=50))

    def test_exact_nanosecond_periods_are_not_merged(self):
        frame = pl.DataFrame({
            "series_id": ["a", "a"],
            "week": pl.Series([1, 2], dtype=pl.Datetime("ns")),
            "forecast": [100.0, 100.0],
        })
        result = apply_point_constraints(frame, ConstraintConfig(enabled=True, aggregate_max=50))
        assert result["week"].equals(frame["week"])
        assert result["forecast"].to_list() == [50.0, 50.0]

    def test_disabled_constraints_are_a_true_noop(self):
        frame = _point_frame().with_columns(pl.lit(float("nan")).alias("forecast_p90"))
        config = ConstraintConfig(
            enabled=False, min_demand=-1, max_capacity=-2, aggregate_max=1, capacity_column="absent",
        )
        assert apply_point_constraints(frame, config) is frame
        wrapper = ConstrainedDemandEstimator(_QuantileFrameForecaster(frame), config)
        wrapper.fit(pl.DataFrame())
        assert wrapper.predict(1) is frame
        assert wrapper.predict_quantiles(1, [0.9]) is frame


class TestCapacityLifecycle:
    @staticmethod
    def _train(capacities=(80.0, 120.0)):
        return pl.DataFrame({
            "series_id": ["a", "b"], "week": [date(2023, 1, 2)] * 2,
            "quantity": [10.0, 10.0], "_cap": capacities,
        })

    def test_scalar_and_per_series_caps_use_tighter_bound(self):
        frame = _point_frame()
        config = ConstraintConfig(enabled=True, min_demand=10, max_capacity=90, capacity_column="_cap")
        result = apply_point_constraints(frame, config, capacity_map={"a": 80, "b": 120})
        assert result["forecast"].to_list() == [80.0, 90.0]
        wrapper = ConstrainedDemandEstimator(_StubForecaster(frame), config)
        wrapper.fit(self._train())
        assert wrapper.predict(1).equals(result)

    @pytest.mark.parametrize(
        "capacity_map",
        [None, {}, {"a": 20}, {"a": None, "b": 20}, {"a": float("nan"), "b": 20},
         {"a": float("inf"), "b": 20}, {"a": -1, "b": 20}, {"a": 9, "b": 20}],
    )
    def test_configured_capacity_map_is_complete_finite_and_feasible(self, capacity_map):
        with pytest.raises(ValueError):
            apply_point_constraints(
                _point_frame(), ConstraintConfig(enabled=True, min_demand=10, capacity_column="_cap"),
                capacity_map=capacity_map,
            )

    @pytest.mark.parametrize("capacity", [None, float("nan"), float("inf"), -1.0, 9.0, "80", True])
    def test_bad_training_capacity_does_not_survive_fit(self, capacity):
        wrapper = ConstrainedDemandEstimator(
            _StubForecaster(_point_frame()),
            ConstraintConfig(enabled=True, min_demand=10, capacity_column="_cap"),
        )
        wrapper.fit(self._train())
        invalid = self._train().with_columns(pl.Series("_cap", [capacity, capacity]))
        with pytest.raises(ValueError):
            wrapper.fit(invalid)
        with pytest.raises(ValueError, match="capacity map must cover"):
            wrapper.predict(1)

    def test_missing_refit_capacity_clears_old_map_then_disabled_option_matches_fresh(self):
        settings = ConstraintConfig(enabled=True, max_capacity=90, capacity_column="_cap")
        frame = _point_frame()
        wrapper = ConstrainedDemandEstimator(_StubForecaster(frame), settings)
        wrapper.fit(self._train())
        assert wrapper.predict(1)["forecast"].to_list() == [80.0, 90.0]
        no_capacity = self._train().drop("_cap")
        with pytest.raises(ValueError, match="required at every fit"):
            wrapper.fit(no_capacity)
        with pytest.raises(ValueError, match="capacity map must cover"):
            wrapper.predict(1)
        settings.capacity_column = None
        wrapper.fit(no_capacity)
        fresh = ConstrainedDemandEstimator(_StubForecaster(frame), settings)
        fresh.fit(no_capacity)
        assert wrapper.predict(1).equals(fresh.predict(1))
        assert wrapper.predict(1)["forecast"].to_list() == [90.0, 90.0]

    def test_successful_refit_replaces_capacity_and_requires_every_prediction_identity(self):
        wrapper = ConstrainedDemandEstimator(
            _StubForecaster(_point_frame()),
            ConstraintConfig(enabled=True, capacity_column="_cap"),
        )
        wrapper.fit(self._train())
        wrapper.fit(self._train([30.0, 40.0]))
        assert wrapper.predict(1)["forecast"].to_list() == [30.0, 40.0]
        wrapper.fit(self._train().head(1))
        with pytest.raises(ValueError, match="capacity map must cover"):
            wrapper.predict(1)

    @pytest.mark.parametrize("identities", [[None, "b"], ["", "b"], [float("nan"), 2.0], [["a"], ["b"]]])
    def test_training_capacity_identity_is_validated(self, identities):
        wrapper = ConstrainedDemandEstimator(
            _StubForecaster(_point_frame()),
            ConstraintConfig(enabled=True, capacity_column="_cap"),
        )
        with pytest.raises(ValueError):
            wrapper.fit(self._train().with_columns(pl.Series("series_id", identities)))


class TestQuantileSupportBoundary:
    @staticmethod
    def _frame():
        return _point_frame([50.0, 70.0]).with_columns(
            pl.Series("forecast_p10", [-2.0, 60.0]),
            pl.Series("forecast_p50", [40.0, 80.0]),
            pl.Series("forecast_p90", [100.0, 120.0]),
        )

    def test_monotone_clipping_respects_each_series_and_does_not_force_point_to_median(self):
        frame = self._frame()
        settings = ConstraintConfig(enabled=True, min_demand=10, max_capacity=90, capacity_column="_cap")
        result = apply_point_constraints(frame, settings, capacity_map={"a": 60, "b": 120})
        assert result["forecast"].to_list() == [50.0, 70.0]
        assert result["forecast_p10"].to_list() == [10.0, 60.0]
        assert result["forecast_p50"].to_list() == [40.0, 80.0]
        assert result["forecast_p90"].to_list() == [60.0, 90.0]
        wrapper = ConstrainedDemandEstimator(_QuantileFrameForecaster(frame), settings)
        wrapper.fit(TestCapacityLifecycle._train([60.0, 120.0]))
        assert wrapper.predict_quantiles(1, [0.9, 0.1, 0.5]).equals(result)
        assert wrapper.predict(1).equals(result)

    def test_upper_tail_quantiles_need_not_contain_the_point(self):
        frame = _point_frame([5.0]).with_columns(
            pl.Series("forecast_p80", [10.0]), pl.Series("forecast_p90", [30.0]),
        )
        result = apply_point_constraints(frame, ConstraintConfig(enabled=True, max_capacity=20))
        assert result["forecast"].to_list() == [5.0]
        assert result["forecast_p80"].to_list() == [10.0]
        assert result["forecast_p90"].to_list() == [20.0]

    @pytest.mark.parametrize("proportional", [True, False])
    def test_aggregate_quantile_frames_and_both_wrapper_entrypoints_are_rejected(self, proportional):
        frame = self._frame()
        config = ConstraintConfig(enabled=True, aggregate_max=1000, proportional_redistribution=proportional)
        with pytest.raises(ValueError, match="unsupported_aggregate_quantiles"):
            apply_point_constraints(frame, config)
        wrapper = ConstrainedDemandEstimator(_QuantileFrameForecaster(frame), config)
        with pytest.raises(ValueError, match="unsupported_aggregate_quantiles"):
            wrapper.predict(1)
        with pytest.raises(ValueError, match="unsupported_aggregate_quantiles"):
            wrapper.predict_quantiles(1, [0.1, 0.5, 0.9])

    def test_aggregate_rejection_precedes_base_quantile_prediction(self):
        class NoQuantileCall(_StubForecaster):
            def predict_quantiles(self, *args, **kwargs):
                raise AssertionError("Unsupported quantile prediction must not be called")

        wrapper = ConstrainedDemandEstimator(
            NoQuantileCall(_point_frame()), ConstraintConfig(enabled=True, aggregate_max=50),
        )
        with pytest.raises(ValueError, match="unsupported_aggregate_quantiles"):
            wrapper.predict_quantiles(1, [0.1, 0.9])

    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None, "100"])
    def test_nonfinite_quantiles_are_rejected_before_clipping(self, invalid):
        frame = self._frame().with_columns(pl.Series("forecast_p90", [invalid, invalid]))
        wrapper = ConstrainedDemandEstimator(
            _QuantileFrameForecaster(frame), ConstraintConfig(enabled=True, max_capacity=0),
        )
        with pytest.raises(ValueError, match="finite numeric"):
            wrapper.predict_quantiles(1, [0.1, 0.5, 0.9])

    def test_crossed_quantiles_are_not_repaired_by_clipping_or_sorting(self):
        frame = self._frame().with_columns(pl.Series("forecast_p10", [110.0, 130.0]))
        settings = ConstraintConfig(enabled=True, min_demand=10, max_capacity=10)
        with pytest.raises(ValueError, match="ordered quantiles"):
            apply_point_constraints(frame, settings)
        wrapper = ConstrainedDemandEstimator(_QuantileFrameForecaster(frame), settings)
        with pytest.raises(ValueError, match="ordered quantiles"):
            wrapper.predict_quantiles(1, [0.1, 0.5, 0.9])

    @pytest.mark.parametrize("quantiles", [[0.25], [-0.1], [1.1], [float("nan")], [0.1, 0.101], []])
    def test_missing_invalid_or_ambiguous_requested_quantiles(self, quantiles):
        wrapper = ConstrainedDemandEstimator(
            _QuantileFrameForecaster(self._frame()), ConstraintConfig(enabled=True),
        )
        with pytest.raises(ValueError):
            wrapper.predict_quantiles(1, quantiles)

    @pytest.mark.parametrize("column", ["forecast_p101", "forecast_p-1", "forecast_p010", "forecast_pbad"])
    def test_invalid_quantile_column_names_are_not_returned_unconstrained(self, column):
        frame = _point_frame().with_columns(pl.lit(1000.0).alias(column))
        with pytest.raises(ValueError, match="quantile columns"):
            apply_point_constraints(frame, ConstraintConfig(enabled=True, max_capacity=50))
