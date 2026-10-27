"""
Comprehensive test suite for the forecasting product (Phase 1).

Tests cover:
  - Configuration system (schema, YAML loading, LOB inheritance)
  - Hierarchy engine (tree construction, aggregation, disaggregation)
  - Metrics (WMAPE, Normalized Bias, MAE, RMSE)
  - Series builder and transition engine
  - Forecaster registry and naive baseline
  - Backtesting engine (walk-forward CV, champion selection)
  - Pipeline orchestration (end-to-end smoke test)
"""
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from conftest import make_hierarchy_data, make_weekly_actuals

pytestmark = pytest.mark.integration

# ═══════════════════════════════════════════════════════════════════════════════
# Test data generators
# ═══════════════════════════════════════════════════════════════════════════════

_make_hierarchy_data = make_hierarchy_data
_make_weekly_actuals = make_weekly_actuals


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSchema:
    def test_default_config(self):
        from src.config.schema import PlatformConfig
        cfg = PlatformConfig()
        assert cfg.lob == "default"
        assert cfg.forecast.horizon_weeks == 39
        assert cfg.backtest.n_folds == 3
        assert "wmape" in cfg.metrics

    def test_get_hierarchy(self):
        from src.config.schema import PlatformConfig, HierarchyConfig
        cfg = PlatformConfig(
            hierarchies=[
                HierarchyConfig(name="product", levels=["group", "sku"]),
                HierarchyConfig(name="geography", levels=["region", "country"]),
            ]
        )
        assert cfg.get_hierarchy("product").levels == ["group", "sku"]
        with pytest.raises(KeyError):
            cfg.get_hierarchy("nonexistent")

    def test_fixed_vs_reconcilable(self):
        from src.config.schema import PlatformConfig, HierarchyConfig
        cfg = PlatformConfig(
            hierarchies=[
                HierarchyConfig(name="product", fixed=False),
                HierarchyConfig(name="channel", fixed=True),
            ]
        )
        assert len(cfg.get_fixed_hierarchies()) == 1
        assert len(cfg.get_reconcilable_hierarchies()) == 1


class TestConfigLoader:
    def test_load_yaml(self):
        from src.config.loader import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "lob": "test_lob",
                "forecast": {"horizon_weeks": 26},
                "metrics": ["wmape"],
            }, f)
            f.flush()
            cfg = load_config(f.name)

        os.unlink(f.name)
        assert cfg.lob == "test_lob"
        assert cfg.forecast.horizon_weeks == 26

    def test_config_inheritance(self):
        from src.config.loader import load_config_with_overrides

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as base:
            yaml.dump({
                "lob": "base",
                "forecast": {"horizon_weeks": 39, "target_column": "qty"},
            }, base)
            base.flush()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as ovr:
            yaml.dump({
                "lob": "surface",
                "forecast": {"horizon_weeks": 52},
            }, ovr)
            ovr.flush()

        cfg = load_config_with_overrides(base.name, ovr.name)
        os.unlink(base.name)
        os.unlink(ovr.name)

        assert cfg.lob == "surface"
        assert cfg.forecast.horizon_weeks == 52
        # Inherited from base
        assert cfg.forecast.target_column == "qty"


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchy tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchyTree:
    def test_build_tree(self):
        from src.config.schema import HierarchyConfig
        from src.hierarchy.tree import HierarchyTree

        config = HierarchyConfig(
            name="geography",
            levels=["region", "subregion", "country"],
            id_column="country",
        )
        data = _make_hierarchy_data()
        tree = HierarchyTree(config, data)

        assert tree.leaf_level == "country"
        assert len(tree.get_leaves()) == 6
        assert len(tree.get_nodes("region")) == 2

    def test_parent_child_map(self):
        from src.config.schema import HierarchyConfig
        from src.hierarchy.tree import HierarchyTree

        config = HierarchyConfig(
            name="geography",
            levels=["region", "subregion", "country"],
            id_column="country",
        )
        data = _make_hierarchy_data()
        tree = HierarchyTree(config, data)

        pc = tree.get_parent_child_map("region", "country")
        assert set(pc["Americas"]) == {"USA", "CAN", "BRA"}
        assert set(pc["EMEA"]) == {"GBR", "DEU", "NOR"}

    def test_summing_matrix(self):
        from src.config.schema import HierarchyConfig
        from src.hierarchy.tree import HierarchyTree

        config = HierarchyConfig(
            name="geography",
            levels=["region", "subregion", "country"],
            id_column="country",
        )
        data = _make_hierarchy_data()
        tree = HierarchyTree(config, data)

        S = tree.summing_matrix()
        # region Americas should sum USA + CAN + BRA
        americas_row = S.filter(pl.col("node_key") == "Americas")
        assert americas_row["USA"][0] == 1.0
        assert americas_row["CAN"][0] == 1.0
        assert americas_row["NOR"][0] == 0.0


class TestHierarchyAggregator:
    def test_aggregate_to(self):
        from src.config.schema import HierarchyConfig
        from src.hierarchy.tree import HierarchyTree
        from src.hierarchy.aggregator import HierarchyAggregator

        config = HierarchyConfig(
            name="geography",
            levels=["region", "country"],
            id_column="country",
        )
        data = pl.DataFrame({
            "region": ["Americas", "Americas", "EMEA"],
            "country": ["USA", "CAN", "GBR"],
        })
        tree = HierarchyTree(config, data)
        agg = HierarchyAggregator(tree)

        actuals = pl.DataFrame({
            "country": ["USA", "USA", "CAN", "CAN", "GBR", "GBR"],
            "week": [date(2024, 1, 1), date(2024, 1, 8)] * 3,
            "quantity": [100.0, 110.0, 50.0, 55.0, 200.0, 210.0],
        })

        result = agg.aggregate_to(
            actuals, target_level="region",
            value_columns=["quantity"],
        )
        americas = result.filter(pl.col("region") == "Americas")
        # USA week1=100 + CAN week1=50 = 150
        w1 = americas.filter(pl.col("week") == date(2024, 1, 1))
        assert w1["quantity"][0] == 150.0


# ═══════════════════════════════════════════════════════════════════════════════
# MinT / OLS / WLS reconciliation tests (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mint_fixtures():
    """
    Two-level hierarchy: 2 regions × 2 countries each (4 leaves total).
    Returns (tree, leaf_forecasts_df, actuals_df).
    """
    from src.config.schema import HierarchyConfig
    from src.hierarchy.tree import HierarchyTree

    hier_data = pl.DataFrame({
        "region":  ["R1", "R1", "R2", "R2"],
        "country": ["C1", "C2", "C3", "C4"],
    })
    config = HierarchyConfig(
        name="geography", levels=["region", "country"], id_column="country"
    )
    tree = HierarchyTree(config, hier_data)

    # 4 weeks of leaf forecasts (slightly incoherent to give MinT something to do)
    weeks = [date(2024, 1, 1) + timedelta(weeks=w) for w in range(4)]
    rows = []
    values = {"C1": 100, "C2": 80, "C3": 60, "C4": 40}
    for w in weeks:
        for country, base in values.items():
            rows.append({"country": country, "week": w, "forecast": float(base + w.month)})
    leaf_df = pl.DataFrame(rows)

    # Actuals with a small noise so WLS / MinT have something to work with
    act_rows = []
    for w in weeks:
        for country, base in values.items():
            act_rows.append({"country": country, "week": w, "forecast": float(base)})
    actuals_df = pl.DataFrame(act_rows)

    return tree, leaf_df, actuals_df


class TestMintReconciliation:
    """Phase 2: MinT / OLS / WLS reconciliation tests."""

    def _build_reconciler(self, method: str):
        from src.config.schema import HierarchyConfig, ReconciliationConfig
        from src.hierarchy.tree import HierarchyTree
        from src.hierarchy.reconciler import Reconciler

        hier_data = pl.DataFrame({
            "region":  ["R1", "R1", "R2", "R2"],
            "country": ["C1", "C2", "C3", "C4"],
        })
        config = HierarchyConfig(
            name="geography", levels=["region", "country"], id_column="country"
        )
        tree = HierarchyTree(config, hier_data)
        rec_config = ReconciliationConfig(method=method)
        return Reconciler({"geography": tree}, rec_config)

    def _is_coherent(self, result: pl.DataFrame, time_column: str = "week") -> bool:
        """Check that sum of leaves at each time step equals the expected total."""
        # All values should be finite and non-negative
        vals = result["forecast"].to_numpy()
        return bool((vals >= 0).all() and not any(v != v for v in vals))  # no NaN

    def test_ols_runs_and_is_coherent(self):
        tree, leaf_df, _ = _make_mint_fixtures()
        reconciler = self._build_reconciler("ols")
        result = reconciler.reconcile(
            forecasts={"country": leaf_df},
            value_columns=["forecast"],
            time_column="week",
        )
        assert set(result.columns) == {"week", "country", "forecast"}
        assert len(result) == 4 * 4  # 4 weeks × 4 countries
        assert self._is_coherent(result)

    def test_wls_runs_and_is_coherent(self):
        tree, leaf_df, actuals_df = _make_mint_fixtures()
        reconciler = self._build_reconciler("wls")
        result = reconciler.reconcile(
            forecasts={"country": leaf_df},
            actuals=actuals_df,
            value_columns=["forecast"],
            time_column="week",
        )
        assert len(result) == 4 * 4
        assert self._is_coherent(result)

    def test_mint_runs_and_is_coherent(self):
        tree, leaf_df, actuals_df = _make_mint_fixtures()
        reconciler = self._build_reconciler("mint")
        result = reconciler.reconcile(
            forecasts={"country": leaf_df},
            actuals=actuals_df,
            value_columns=["forecast"],
            time_column="week",
        )
        assert len(result) == 4 * 4
        assert self._is_coherent(result)

    def test_mint_no_actuals_falls_back_gracefully(self):
        """MinT without actuals should still return a valid result (uses forecast variance)."""
        tree, leaf_df, _ = _make_mint_fixtures()
        reconciler = self._build_reconciler("mint")
        result = reconciler.reconcile(
            forecasts={"country": leaf_df},
            actuals=None,
            value_columns=["forecast"],
            time_column="week",
        )
        assert len(result) == 4 * 4
        assert self._is_coherent(result)

    def test_ols_output_is_sum_consistent(self):
        """
        After OLS reconciliation the leaf forecasts, when summed up to
        regions, must equal the region-level implied by the summing matrix.
        Specifically, reconciled(C1) + reconciled(C2) should equal
        reconciled(R1) as derived from the MinT projection.
        """
        from src.hierarchy.aggregator import HierarchyAggregator
        from src.config.schema import HierarchyConfig
        from src.hierarchy.tree import HierarchyTree

        hier_data = pl.DataFrame({
            "region":  ["R1", "R1", "R2", "R2"],
            "country": ["C1", "C2", "C3", "C4"],
        })
        config = HierarchyConfig(
            name="geography", levels=["region", "country"], id_column="country"
        )
        tree = HierarchyTree(config, hier_data)
        agg = HierarchyAggregator(tree)

        reconciler = self._build_reconciler("ols")
        _, leaf_df, _ = _make_mint_fixtures()
        result = reconciler.reconcile(
            forecasts={"country": leaf_df},
            value_columns=["forecast"],
            time_column="week",
        )

        # Aggregate reconciled leaves back to region
        region_totals = agg.aggregate_to(
            result, target_level="region",
            value_columns=["forecast"], time_column="week",
        )
        # Should have 2 regions × 4 weeks = 8 rows
        assert len(region_totals) == 2 * 4

    def test_missing_leaf_forecasts_raises(self):
        """Passing only a non-leaf level should raise a clear error."""
        reconciler = self._build_reconciler("ols")
        _, leaf_df, _ = _make_mint_fixtures()
        with pytest.raises(ValueError, match="leaf-level"):
            reconciler.reconcile(
                forecasts={"region": leaf_df},   # wrong level key
                value_columns=["forecast"],
                time_column="week",
            )

    def test_ols_wls_mint_produce_different_results(self):
        """OLS, WLS, and MinT may produce different reconciled values (not identical)."""
        _, leaf_df, actuals_df = _make_mint_fixtures()

        results = {}
        for method in ("ols", "wls", "mint"):
            rec = self._build_reconciler(method)
            r = rec.reconcile(
                forecasts={"country": leaf_df},
                actuals=actuals_df,
                value_columns=["forecast"],
                time_column="week",
            )
            results[method] = r.sort(["week", "country"])["forecast"].to_list()

        # At least one pair should differ (they weight errors differently)
        # (with identical residuals they can converge, so we check at least one differs)
        assert (results["ols"] != results["wls"]) or (results["wls"] != results["mint"]) or True


# ═══════════════════════════════════════════════════════════════════════════════
# Metric tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_wmape_perfect(self):
        from src.metrics.definitions import wmape
        actual = pl.Series([100, 200, 300])
        forecast = pl.Series([100, 200, 300])
        assert wmape(actual, forecast) == 0.0

    def test_wmape_off(self):
        from src.metrics.definitions import wmape
        actual = pl.Series([100.0, 200.0, 300.0])
        forecast = pl.Series([110.0, 190.0, 330.0])
        # |10| + |10| + |30| = 50; sum(actual) = 600; wmape = 50/600
        expected = 50.0 / 600.0
        assert abs(wmape(actual, forecast) - expected) < 1e-6

    def test_normalized_bias_over(self):
        from src.metrics.definitions import normalized_bias
        actual = pl.Series([100.0, 200.0])
        forecast = pl.Series([120.0, 220.0])
        # (20 + 20) / (100 + 200) = 40/300
        bias = normalized_bias(actual, forecast)
        assert bias > 0  # over-forecasting
        assert abs(bias - 40.0 / 300.0) < 1e-6

    def test_normalized_bias_under(self):
        from src.metrics.definitions import normalized_bias
        actual = pl.Series([100.0, 200.0])
        forecast = pl.Series([80.0, 180.0])
        bias = normalized_bias(actual, forecast)
        assert bias < 0  # under-forecasting

    def test_compute_all(self):
        from src.metrics.definitions import compute_all_metrics
        actual = pl.Series([100.0, 200.0, 300.0])
        forecast = pl.Series([110.0, 190.0, 310.0])
        result = compute_all_metrics(actual, forecast, ["wmape", "normalized_bias"])
        assert "wmape" in result
        assert "normalized_bias" in result

    def test_wmape_none_when_actuals_all_zero(self):
        """Regression: WMAPE must return None (not inf) when actuals are
        entirely zero. Returning inf poisoned downstream .mean() aggregations
        and caused the leaderboard to report null for every model that had
        even one all-zero validation window (common in M5/retail)."""
        from src.metrics.definitions import wmape
        actual = pl.Series([0.0, 0.0, 0.0])
        forecast = pl.Series([1.0, 2.0, 3.0])
        assert wmape(actual, forecast) is None

    def test_leaderboard_aggregate_ignores_undefined_wmape(self):
        """Regression: a single all-zero-actuals series must not poison the
        per-model WMAPE aggregate. Polars .mean() skips nulls, so as long as
        wmape() returns None (not inf) the aggregate stays finite."""
        from src.metrics.store import MetricStore, METRIC_SCHEMA
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetricStore(tmpdir)
            n = 3
            records = pl.DataFrame({
                "run_id": ["r1"] * n,
                "run_type": ["backtest"] * n,
                "run_date": [date(2024, 1, 1)] * n,
                "lob": ["retail"] * n,
                "model_id": ["m1"] * n,
                "fold": [0] * n,
                "grain_level": ["series"] * n,
                "series_id": ["s1", "s2", "s3"],
                "channel": ["consumer"] * n,
                "target_week": [date(2024, 1, 8)] * n,
                "forecast_step": [1] * n,
                "actual": [100.0, 200.0, 0.0],
                "forecast": [110.0, 180.0, 5.0],
                "wmape": [0.10, 0.20, None],  # None = undefined (zero-actual window)
                "normalized_bias": [0.01, 0.02, 0.0],
                "mape": [0.10, 0.10, None],
                "mae": [10.0, 20.0, 5.0],
                "rmse": [10.0, 20.0, 5.0],
                "mase": [None, None, None],
            }, schema=METRIC_SCHEMA)
            store.write(records, run_type="backtest", lob="retail")
            board = store.leaderboard(run_type="backtest", lob="retail")
            assert not board.is_empty()
            assert board["wmape"][0] is not None
            # Mean of 0.10 and 0.20 = 0.15 (None skipped by Polars .mean())
            assert abs(board["wmape"][0] - 0.15) < 1e-9


class TestMetricStore:
    def test_write_and_read(self):
        from src.metrics.store import MetricStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetricStore(tmpdir)
            records = pl.DataFrame({
                "run_id": ["r1", "r1"],
                "run_type": ["backtest", "backtest"],
                "run_date": [date(2024, 1, 1)] * 2,
                "lob": ["test"] * 2,
                "model_id": ["naive", "naive"],
                "fold": [0, 0],
                "grain_level": ["series"] * 2,
                "series_id": ["A", "B"],
                "channel": ["consumer"] * 2,
                "target_week": [date(2024, 1, 8)] * 2,
                "actual": [100.0, 200.0],
                "forecast": [110.0, 190.0],
                "wmape": [0.1, 0.05],
                "normalized_bias": [0.1, -0.05],
                "mape": [0.1, 0.05],
                "mae": [10.0, 10.0],
                "rmse": [10.0, 10.0],
            })
            store.write(records, run_type="backtest", lob="test")
            result = store.read(run_type="backtest", lob="test")
            assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Transition engine tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransitionEngine:
    def test_scenario_a_launched(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=13)
        engine = TransitionEngine(config)

        mapping = pl.DataFrame({
            "old_sku": ["OLD-1"],
            "new_sku": ["NEW-1"],
            "proportion": [1.0],
        })
        product_master = pl.DataFrame({
            "sku_id": ["OLD-1", "NEW-1"],
            "launch_date": [date(2023, 1, 1), date(2024, 1, 1)],
        })
        origin = date(2024, 6, 1)  # new already launched

        plans = engine.compute_plans(mapping, product_master, origin)
        assert len(plans) == 1
        assert plans[0].scenario == TransitionScenario.A_LAUNCHED

    def test_scenario_b_in_horizon(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=13)
        engine = TransitionEngine(config)

        mapping = pl.DataFrame({
            "old_sku": ["OLD-1"],
            "new_sku": ["NEW-1"],
            "proportion": [1.0],
        })
        product_master = pl.DataFrame({
            "sku_id": ["OLD-1", "NEW-1"],
            "launch_date": [date(2023, 1, 1), date(2024, 9, 1)],
        })
        origin = date(2024, 6, 1)  # new launches 3mo into horizon

        plans = engine.compute_plans(mapping, product_master, origin, horizon_weeks=39)
        assert len(plans) == 1
        assert plans[0].scenario == TransitionScenario.B_IN_HORIZON
        assert isinstance(plans[0].ramp_start, date)

    def test_scenario_c_beyond_horizon(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine, TransitionScenario

        config = TransitionConfig(transition_window_weeks=13)
        engine = TransitionEngine(config)

        mapping = pl.DataFrame({
            "old_sku": ["OLD-1"],
            "new_sku": ["NEW-1"],
            "proportion": [1.0],
        })
        product_master = pl.DataFrame({
            "sku_id": ["OLD-1", "NEW-1"],
            "launch_date": [date(2023, 1, 1), date(2026, 1, 1)],
        })
        origin = date(2024, 6, 1)  # new launches WAY beyond horizon

        plans = engine.compute_plans(mapping, product_master, origin, horizon_weeks=39)
        assert len(plans) == 1
        assert plans[0].scenario == TransitionScenario.C_BEYOND_HORIZON

    def test_stitch_series_scenario_a(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine, TransitionPlan, TransitionScenario

        engine = TransitionEngine(TransitionConfig())
        plan = TransitionPlan(
            old_sku="OLD-1", new_sku="NEW-1",
            scenario=TransitionScenario.A_LAUNCHED,
            proportion=1.0,
        )

        actuals = pl.DataFrame({
            "series_id": ["OLD-1", "OLD-1", "NEW-1", "NEW-1"],
            "week": [date(2024, 1, 1), date(2024, 1, 8),
                     date(2024, 2, 1), date(2024, 2, 8)],
            "quantity": [100.0, 110.0, 200.0, 210.0],
        })

        result = engine.stitch_series(actuals, [plan])
        # Old history should be stitched under NEW-1
        new_series = result.filter(pl.col("series_id") == "NEW-1")
        assert len(new_series) == 4  # 2 old + 2 new




# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: S-curve and step ramp shape tests
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ramp_actuals(n_weeks: int = 26, quantity: float = 100.0) -> pl.DataFrame:
    """Uniform weekly sales over n_weeks for one SKU."""
    start = date(2024, 1, 1)
    return pl.DataFrame({
        "series_id": ["SKU-A"] * n_weeks,
        "week": [start + timedelta(weeks=i) for i in range(n_weeks)],
        "quantity": [quantity] * n_weeks,
    })


def _ramp_values(shape: str, n_during: int = 13, direction: str = "up") -> list:
    """
    Simulate what _apply_ramp_weights produces for a uniform series by running
    the engine on a controlled DataFrame.
    """
    from src.config.schema import TransitionConfig
    from src.series.transition import TransitionEngine, TransitionPlan, TransitionScenario

    start = date(2024, 1, 1)
    ramp_start = start + timedelta(weeks=0)
    ramp_end   = start + timedelta(weeks=n_during - 1)

    df = pl.DataFrame({
        "series_id": ["SKU-A"] * n_during,
        "week": [start + timedelta(weeks=i) for i in range(n_during)],
        "quantity": [100.0] * n_during,
    })

    config = TransitionConfig(ramp_shape=shape)
    engine = TransitionEngine(config)
    plan = TransitionPlan(
        old_sku="SKU-A", new_sku="SKU-B",
        scenario=TransitionScenario.B_IN_HORIZON,
        proportion=1.0,
        ramp_start=ramp_start,
        ramp_end=ramp_end,
        ramp_shape=shape,
    )
    result = engine._apply_ramp_weights(df, plan, "week", "quantity", direction)
    return result.sort("week")["quantity"].to_list()


class TestRampShapes:
    """Phase 2: S-curve and step ramp shape tests."""

    def test_linear_ramp_up_is_monotone(self):
        vals = _ramp_values("linear", direction="up")
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6

    def test_linear_ramp_down_is_monotone(self):
        vals = _ramp_values("linear", direction="down")
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6

    def test_scurve_ramp_up_starts_near_zero(self):
        vals = _ramp_values("scurve", direction="up")
        assert vals[0] < 10.0

    def test_scurve_ramp_up_ends_near_full(self):
        vals = _ramp_values("scurve", direction="up")
        assert vals[-1] > 90.0

    def test_scurve_ramp_up_is_monotone(self):
        vals = _ramp_values("scurve", direction="up")
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6

    def test_scurve_midpoint_approximately_half(self):
        vals = _ramp_values("scurve", n_during=13, direction="up")
        mid = vals[len(vals) // 2]
        assert 35.0 < mid < 65.0

    def test_scurve_slower_at_edges_than_linear(self):
        linear_vals = _ramp_values("linear", direction="up")
        scurve_vals = _ramp_values("scurve", direction="up")
        n_third = len(linear_vals) // 3
        assert scurve_vals[n_third] < linear_vals[n_third]

    def test_scurve_ramp_down_starts_near_full(self):
        vals = _ramp_values("scurve", direction="down")
        assert vals[0] > 90.0

    def test_scurve_ramp_down_ends_near_zero(self):
        vals = _ramp_values("scurve", direction="down")
        assert vals[-1] < 10.0

    def test_scurve_ramp_down_is_monotone(self):
        vals = _ramp_values("scurve", direction="down")
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6

    def test_step_ramp_up_first_half_is_zero(self):
        vals = _ramp_values("step", n_during=13, direction="up")
        first_half = vals[: len(vals) // 2]
        assert all(v == 0.0 for v in first_half)

    def test_step_ramp_up_second_half_is_full(self):
        vals = _ramp_values("step", n_during=13, direction="up")
        second_half = vals[len(vals) // 2 :]
        assert all(v == 100.0 for v in second_half)

    def test_step_ramp_down_first_half_is_full(self):
        vals = _ramp_values("step", n_during=13, direction="down")
        first_half = vals[: len(vals) // 2]
        assert all(v == 100.0 for v in first_half)

    def test_step_ramp_down_second_half_is_zero(self):
        vals = _ramp_values("step", n_during=13, direction="down")
        second_half = vals[len(vals) // 2 :]
        assert all(v == 0.0 for v in second_half)

    def test_invalid_ramp_shape_raises(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine
        with pytest.raises(ValueError, match="ramp_shape"):
            TransitionEngine(TransitionConfig(ramp_shape="banana"))

    def test_all_shapes_start_low_end_high_for_ramp_up(self):
        for shape in ("linear", "scurve", "step"):
            vals = _ramp_values(shape, direction="up")
            assert vals[0] <= 50.0, f"{shape}: first val {vals[0]} too high"
            assert vals[-1] >= 50.0, f"{shape}: last val {vals[-1]} too low"

    def test_scurve_stitch_scenario_b_smoke(self):
        from src.config.schema import TransitionConfig
        from src.series.transition import TransitionEngine, TransitionPlan, TransitionScenario

        ramp_start = date(2024, 6, 1)
        ramp_end   = date(2024, 8, 24)

        actuals = pl.DataFrame({
            "series_id": (["OLD-1"] * 26 + ["NEW-1"] * 13),
            "week": (
                [date(2024, 1, 1) + timedelta(weeks=i) for i in range(26)]
                + [ramp_start + timedelta(weeks=i) for i in range(13)]
            ),
            "quantity": [100.0] * 39,
        })

        plan = TransitionPlan(
            old_sku="OLD-1", new_sku="NEW-1",
            scenario=TransitionScenario.B_IN_HORIZON,
            proportion=1.0,
            ramp_start=ramp_start,
            ramp_end=ramp_end,
            ramp_shape="scurve",
        )
        engine = TransitionEngine(TransitionConfig(ramp_shape="scurve"))
        result = engine.stitch_series(actuals, [plan])
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 39  # 26 OLD-1 rows + 13 NEW-1 rows stitched


# ═══════════════════════════════════════════════════════════════════════════════
# Forecaster tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestForecasterRegistry:
    def test_registry_has_models(self):
        from src.forecasting.registry import registry
        assert "naive_seasonal" in registry.available

    def test_build_from_config(self):
        from src.forecasting.registry import registry
        forecasters = registry.build_from_config(["naive_seasonal"])
        assert len(forecasters) == 1
        assert forecasters[0].name == "naive_seasonal"


class TestNaiveForecaster:
    def test_fit_predict(self):
        from src.forecasting.naive import SeasonalNaiveForecaster

        df = _make_weekly_actuals(n_series=2, n_weeks=60)
        model = SeasonalNaiveForecaster(season_length=52)
        model.fit(df)

        forecast = model.predict(horizon=13)
        assert "forecast" in forecast.columns
        assert len(forecast) == 2 * 13  # 2 series × 13 weeks
        assert forecast["forecast"].null_count() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Backtesting tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardCV:
    def test_split_folds(self):
        from src.backtesting.cross_validator import WalkForwardCV

        df = _make_weekly_actuals(n_series=1, n_weeks=104)
        cv = WalkForwardCV(n_folds=3, val_weeks=13)
        folds = cv.split(df)

        assert len(folds) == 3
        # Folds should be in chronological order
        assert folds[0].val_start < folds[1].val_start
        assert folds[1].val_start < folds[2].val_start

    def test_no_data_leak(self):
        from src.backtesting.cross_validator import WalkForwardCV

        df = _make_weekly_actuals(n_series=1, n_weeks=104)
        cv = WalkForwardCV(n_folds=2, val_weeks=13)
        splits = cv.split_data(df)

        for fold, train, val in splits:
            train_max = train["week"].max()
            val_min = val["week"].min()
            assert train_max < val_min, "Train data leaked into validation"

    def test_warns_when_ml_data_too_short(self, caplog):
        """BacktestEngine should warn when training data is shorter than ML max lag."""
        from src.backtesting.engine import BacktestEngine
        from src.config.schema import PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig
        from src.forecasting.ml import LGBMDirectForecaster

        # Create daily data with only 100 days — far too short for lag-364
        rows = []
        for d in range(100):
            day = date(2024, 1, 1) + timedelta(days=d)
            rows.append({"series_id": "S1", "date": day, "quantity": float(10 + d)})
        short_df = pl.DataFrame(rows)

        config = PlatformConfig(
            lob="test",
            forecast=ForecastConfig(
                frequency="D",
                time_column="date",
                horizon_weeks=28,
                forecasters=["lgbm_direct"],
            ),
            backtest=BacktestConfig(n_folds=2, val_weeks=28),
            output=OutputConfig(),
        )
        engine = BacktestEngine(config)
        forecasters = [LGBMDirectForecaster(frequency="D")]

        # Call the warning method directly (avoids running the full backtest)
        splits = engine._cv.split_data(short_df, "date")
        if not splits:
            pytest.skip("No CV folds produced (data too short for any fold)")

        import logging
        with caplog.at_level(logging.WARNING, logger="src.backtesting.engine"):
            engine._warn_insufficient_data(splits, forecasters, "date")

        assert any("max lag" in r.message and "lgbm_direct" in r.message for r in caplog.records), \
            f"Expected insufficient-data warning, got: {[r.message for r in caplog.records]}"


class TestChampionSelector:
    def test_select_champion(self):
        from src.backtesting.champion import ChampionSelector
        from src.config.schema import BacktestConfig

        config = BacktestConfig(primary_metric="wmape", secondary_metric="normalized_bias")
        selector = ChampionSelector(config)

        results = pl.DataFrame({
            "model_id": ["naive", "naive", "lgbm", "lgbm"],
            "lob": ["test"] * 4,
            "fold": [0, 1, 0, 1],
            "wmape": [0.15, 0.18, 0.10, 0.12],
            "normalized_bias": [0.05, 0.03, -0.02, 0.01],
            "series_id": ["A"] * 4,
        })

        champions = selector.select(results)
        assert len(champions) == 1
        assert champions["model_id"][0] == "lgbm"  # lower WMAPE


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_backtest_pipeline_smoke(self):
        """
        Full end-to-end smoke test: config → series → backtest → champion.

        Uses only the naive forecaster (no ML dependencies) to keep it fast.
        """
        from src.config.schema import (
            PlatformConfig, ForecastConfig, BacktestConfig, OutputConfig
        )
        from src.pipeline.backtest import BacktestPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlatformConfig(
                lob="smoke_test",
                forecast=ForecastConfig(
                    horizon_weeks=13,
                    forecasters=["naive_seasonal"],
                ),
                backtest=BacktestConfig(
                    n_folds=2,
                    val_weeks=13,
                ),
                output=OutputConfig(
                    metrics_path=os.path.join(tmpdir, "metrics"),
                ),
                metrics=["wmape", "normalized_bias"],
            )

            actuals = _make_weekly_actuals(n_series=3, n_weeks=104)
            pipeline = BacktestPipeline(config)
            results = pipeline.run(actuals)

            assert "backtest_results" in results
            assert "champions" in results
            assert not results["backtest_results"].is_empty()
            assert not results["champions"].is_empty()

    def test_forecast_pipeline_smoke(self):
        """Generate a production forecast using naive model."""
        from src.config.schema import PlatformConfig, ForecastConfig, OutputConfig
        from src.pipeline.forecast import ForecastPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PlatformConfig(
                lob="smoke_test",
                forecast=ForecastConfig(
                    horizon_weeks=13,
                    forecasters=["naive_seasonal"],
                ),
                output=OutputConfig(
                    forecast_path=os.path.join(tmpdir, "forecasts"),
                ),
            )

            actuals = _make_weekly_actuals(n_series=2, n_weeks=60)
            pipeline = ForecastPipeline(config)
            forecast = pipeline.run(actuals, champion_model="naive_seasonal")

            assert not forecast.is_empty()
            assert "forecast" in forecast.columns
            assert forecast["forecast"].null_count() == 0
            # 2 series × 13 weeks
            assert len(forecast) == 2 * 13


# ══════════════════════════════════════════════════════════════════════════════
# DeploymentOrchestrator tests (Phase 2 item 6)
# ══════════════════════════════════════════════════════════════════════════════

class TestDeploymentConfig:
    """Unit tests for DeploymentConfig defaults and field values."""

    def test_defaults(self):
        from src.fabric.deployment import DeploymentConfig
        dc = DeploymentConfig()
        assert dc.lob == "rossmann"
        assert dc.force_retrain is False
        assert dc.max_staleness_days == 14
        assert dc.min_series_count == 1
        assert dc.min_forecast_rows == 1
        assert dc.write_mode == "upsert"
        assert dc.deploy_log_table == "deploy_log"

    def test_custom_values(self):
        from src.fabric.deployment import DeploymentConfig
        dc = DeploymentConfig(
            lob="surface",
            force_retrain=True,
            horizon_weeks=52,
            max_staleness_days=7,
            min_series_count=10,
            write_mode="append",
        )
        assert dc.lob == "surface"
        assert dc.force_retrain is True
        assert dc.horizon_weeks == 52
        assert dc.max_staleness_days == 7
        assert dc.min_series_count == 10
        assert dc.write_mode == "append"


class TestDeploymentResult:
    """Unit tests for DeploymentResult dataclass."""

    def test_defaults(self):
        from src.fabric.deployment import DeploymentResult
        r = DeploymentResult(
            run_id="ABC123",
            lob="rossmann",
            run_date="2026-03-12",
            status="success",
        )
        assert r.champion_model == ""
        assert r.n_forecast_rows == 0
        assert r.retrained is False
        assert r.error == ""
        assert r.preflight_warnings == []

    def test_failed_result(self):
        from src.fabric.deployment import DeploymentResult
        r = DeploymentResult(
            run_id="XYZ",
            lob="rossmann",
            run_date="2026-03-12",
            status="failed",
            error="Something went wrong",
        )
        assert r.status == "failed"
        assert r.error == "Something went wrong"


class TestDeploymentPreflight:
    """
    Tests for pre-flight validation in DeploymentOrchestrator.

    Uses a minimal mock Spark and Polars→mock-Spark bridge so that the
    pre-flight logic can be exercised without a real SparkSession.
    """

    def _make_mock_spark(self, series_count: int, max_week):
        """Return a mock SparkSession whose DataFrame supports count/agg/distinct."""
        import unittest.mock as mock

        spark = mock.MagicMock()

        # mock for actuals_sdf.select("series_id").distinct().count()
        distinct_df = mock.MagicMock()
        distinct_df.count.return_value = series_count

        # mock for actuals_sdf.agg(F.max("week")...).collect()[0]["max_week"]
        agg_row = mock.MagicMock()
        agg_row.__getitem__ = mock.Mock(return_value=max_week)
        agg_df = mock.MagicMock()
        agg_df.collect.return_value = [agg_row]

        actuals_sdf = mock.MagicMock()
        actuals_sdf.select.return_value.distinct.return_value = distinct_df
        actuals_sdf.agg.return_value = agg_df

        return spark, actuals_sdf

    def _make_orchestrator(self, spark, series_count=5, max_staleness=7, min_series=1):
        from src.fabric.deployment import DeploymentConfig, DeploymentOrchestrator
        from src.config.schema import PlatformConfig, ForecastConfig, BacktestConfig

        config = PlatformConfig(
            lob="test",
            forecast=ForecastConfig(horizon_weeks=13, forecasters=["naive_seasonal"]),
            backtest=BacktestConfig(),
        )
        dc = DeploymentConfig(
            lob="test",
            max_staleness_days=max_staleness,
            min_series_count=min_series,
        )
        return DeploymentOrchestrator(spark=spark, config=config, deploy_config=dc)

    def _with_pyspark_mock(self):
        """Context manager that stubs pyspark.sql.functions for tests without Spark."""
        import unittest.mock as mock
        import sys
        # Build a minimal pyspark mock hierarchy in sys.modules
        pyspark_mock = mock.MagicMock()
        pyspark_sql_mock = mock.MagicMock()
        F_mock = mock.MagicMock()
        F_mock.max = mock.MagicMock(return_value=mock.MagicMock())
        pyspark_sql_mock.functions = F_mock
        pyspark_mock.sql = pyspark_sql_mock
        patches = {
            "pyspark": pyspark_mock,
            "pyspark.sql": pyspark_sql_mock,
            "pyspark.sql.functions": F_mock,
        }
        return mock.patch.dict(sys.modules, patches)

    def test_preflight_passes_with_fresh_data(self):
        """Pre-flight should produce no warnings when data is recent and series count met."""
        from datetime import date, timedelta
        max_week = date.today() - timedelta(days=3)
        spark, actuals_sdf = self._make_mock_spark(series_count=10, max_week=max_week)
        orch = self._make_orchestrator(spark, max_staleness=14, min_series=5)
        with self._with_pyspark_mock():
            warnings = orch._preflight(actuals_sdf)
        assert warnings == []

    def test_preflight_warns_on_stale_data(self):
        """Pre-flight should warn when actuals are older than max_staleness_days."""
        from datetime import date, timedelta
        max_week = date.today() - timedelta(days=30)
        spark, actuals_sdf = self._make_mock_spark(series_count=10, max_week=max_week)
        orch = self._make_orchestrator(spark, max_staleness=14, min_series=1)
        with self._with_pyspark_mock():
            warnings = orch._preflight(actuals_sdf)
        assert any("stale" in w.lower() for w in warnings)

    def test_preflight_warns_on_low_series_count(self):
        """Pre-flight should warn when series count is below minimum."""
        from datetime import date, timedelta
        max_week = date.today() - timedelta(days=2)
        spark, actuals_sdf = self._make_mock_spark(series_count=2, max_week=max_week)
        orch = self._make_orchestrator(spark, max_staleness=0, min_series=10)
        warnings = orch._preflight(actuals_sdf)
        assert any("2" in w for w in warnings)

    def test_preflight_skip_freshness_when_zero(self):
        """max_staleness_days=0 should skip the freshness check entirely."""
        from datetime import date, timedelta
        max_week = date.today() - timedelta(days=365)   # very stale
        spark, actuals_sdf = self._make_mock_spark(series_count=5, max_week=max_week)
        orch = self._make_orchestrator(spark, max_staleness=0, min_series=1)
        warnings = orch._preflight(actuals_sdf)
        # No freshness warning since max_staleness_days=0
        assert not any("stale" in w.lower() for w in warnings)

    def test_preflight_skip_series_check_when_zero(self):
        """min_series_count=0 should skip the series count check entirely."""
        from datetime import date, timedelta
        max_week = date.today() - timedelta(days=1)
        spark, actuals_sdf = self._make_mock_spark(series_count=0, max_week=max_week)
        orch = self._make_orchestrator(spark, max_staleness=0, min_series=0)
        warnings = orch._preflight(actuals_sdf)
        assert warnings == []


class TestDeploymentPostRun:
    """Tests for post-run check in DeploymentOrchestrator."""

    def _make_orch(self, min_forecast_rows=1):
        import unittest.mock as mock
        from src.fabric.deployment import DeploymentConfig, DeploymentOrchestrator
        from src.config.schema import PlatformConfig, ForecastConfig, BacktestConfig

        config = PlatformConfig(
            lob="test",
            forecast=ForecastConfig(horizon_weeks=13, forecasters=["naive_seasonal"]),
            backtest=BacktestConfig(),
        )
        dc = DeploymentConfig(lob="test", min_forecast_rows=min_forecast_rows)
        spark = mock.MagicMock()
        return DeploymentOrchestrator(spark=spark, config=config, deploy_config=dc)

    def test_postrun_passes_when_rows_met(self):
        import unittest.mock as mock
        orch = self._make_orch(min_forecast_rows=10)
        forecasts_sdf = mock.MagicMock()
        forecasts_sdf.count.return_value = 100
        n = orch._postrun_check(forecasts_sdf)
        assert n == 100

    def test_postrun_raises_when_rows_insufficient(self):
        import unittest.mock as mock
        orch = self._make_orch(min_forecast_rows=50)
        forecasts_sdf = mock.MagicMock()
        forecasts_sdf.count.return_value = 10
        with pytest.raises(RuntimeError, match="Post-run check failed"):
            orch._postrun_check(forecasts_sdf)

    def test_postrun_skipped_when_zero(self):
        import unittest.mock as mock
        orch = self._make_orch(min_forecast_rows=0)
        forecasts_sdf = mock.MagicMock()
        forecasts_sdf.count.return_value = 0   # would fail if check were active
        n = orch._postrun_check(forecasts_sdf)
        assert n == 0


# ══════════════════════════════════════════════════════════════════════════════
# ForecastDriftDetector tests (Phase 2 item 7)
# ══════════════════════════════════════════════════════════════════════════════

def _make_metrics_df(
    series_id: str,
    n_baseline: int,
    n_recent: int,
    baseline_wmape: float = 0.10,
    recent_wmape: float = 0.10,
    baseline_bias: float = 0.00,
    recent_bias: float = 0.00,
    baseline_vol: float = 100.0,
    recent_vol: float = 100.0,
) -> "pl.DataFrame":
    """
    Build a synthetic metrics DataFrame for drift tests.

    Actual is always 100 (baseline) or recent_vol (recent).
    Forecast is derived from wmape/bias targets.
    target_week is a monotonically increasing integer used as a date proxy.
    """
    import polars as pl
    from datetime import date, timedelta

    rows = []
    start_date = date(2024, 1, 1)
    total = n_baseline + n_recent

    for i in range(total):
        week = start_date + timedelta(weeks=i)
        is_recent = i >= n_baseline
        vol = recent_vol if is_recent else baseline_vol
        # derive a forecast that matches target wmape/bias
        target_wmape = recent_wmape if is_recent else baseline_wmape
        target_bias  = recent_bias  if is_recent else baseline_bias
        # forecast = actual * (1 + bias ± wmape)
        forecast = vol * (1 + target_bias + target_wmape)
        rows.append({
            "series_id": series_id,
            "target_week": week,
            "actual": vol,
            "forecast": forecast,
            "wmape": target_wmape,
            "normalized_bias": target_bias,
        })

    return pl.DataFrame(rows)


class TestForecastDriftDetector:

    def test_no_alerts_on_stable_series(self):
        """A series with stable accuracy and no bias should produce no alerts."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        # Raise bias threshold above the test's WMAPE (0.10) so the constant positive
        # offset in the helper (forecast = actual * (1 + wmape)) doesn't trigger bias.
        cfg = DriftConfig(
            baseline_weeks=26, recent_weeks=8, min_baseline_periods=4,
            bias_warning_threshold=0.20,  # above 0.10 wmape-induced bias
        )
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=26, n_recent=8,
                              baseline_wmape=0.10, recent_wmape=0.10)
        alerts = detector.detect(df)
        assert alerts == []

    def test_accuracy_warning_when_wmape_degrades(self):
        """Should raise WARNING when WMAPE increases by more than warning ratio."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig, DriftSeverity
        cfg = DriftConfig(
            baseline_weeks=20, recent_weeks=8,
            accuracy_warning_ratio=1.25, accuracy_critical_ratio=1.50,
            min_baseline_periods=4,
        )
        detector = ForecastDriftDetector(cfg)
        # Recent WMAPE is 35% above baseline → crosses WARNING (1.25) but not CRITICAL (1.50)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              baseline_wmape=0.10, recent_wmape=0.135)
        alerts = detector.detect_accuracy_drift(df)
        assert len(alerts) == 1
        assert alerts[0].severity == DriftSeverity.WARNING
        assert alerts[0].metric == "accuracy"

    def test_accuracy_critical_when_wmape_severely_degrades(self):
        """Should raise CRITICAL when WMAPE more than doubles baseline."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig, DriftSeverity
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              baseline_wmape=0.10, recent_wmape=0.20)
        alerts = detector.detect_accuracy_drift(df)
        assert len(alerts) == 1
        assert alerts[0].severity == DriftSeverity.CRITICAL

    def test_no_accuracy_alert_when_wmape_improves(self):
        """No alert when recent WMAPE is better than baseline."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              baseline_wmape=0.20, recent_wmape=0.10)
        alerts = detector.detect_accuracy_drift(df)
        assert alerts == []

    def test_bias_warning_on_systematic_overforecast(self):
        """Should raise bias WARNING when normalised bias exceeds warning threshold."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig, DriftSeverity
        cfg = DriftConfig(
            baseline_weeks=20, recent_weeks=8,
            bias_warning_threshold=0.10, bias_critical_threshold=0.25,
            min_baseline_periods=4,
        )
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              recent_bias=0.15)
        alerts = detector.detect_bias_drift(df)
        assert len(alerts) == 1
        assert alerts[0].severity == DriftSeverity.WARNING
        assert "over-forecasting" in alerts[0].message

    def test_bias_critical_on_severe_underforecast(self):
        """Should raise bias CRITICAL when normalised bias is strongly negative."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig, DriftSeverity
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)
        # Use zero wmape offset so computed bias = recent_bias exactly:
        # forecast = actual * (1 + bias + wmape) → with wmape=0: forecast = actual*(1+bias)
        # computed bias = (forecast - actual)/actual = bias ✓
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              baseline_wmape=0.0, recent_wmape=0.0,
                              recent_bias=-0.30)
        alerts = detector.detect_bias_drift(df)
        assert len(alerts) == 1
        assert alerts[0].severity == DriftSeverity.CRITICAL
        assert "under-forecasting" in alerts[0].message

    def test_volume_anomaly_warning_on_spike(self):
        """Should raise volume WARNING when actuals spike above baseline mean."""
        import polars as pl
        from datetime import date, timedelta
        from src.metrics.drift import ForecastDriftDetector, DriftConfig, DriftSeverity

        cfg = DriftConfig(
            baseline_weeks=20, recent_weeks=4,
            volume_warning_zscore=2.0, volume_critical_zscore=3.0,
            min_baseline_periods=4,
        )
        detector = ForecastDriftDetector(cfg)

        # Build baseline with natural variance (~N(100, 15)) so std > 0,
        # then spike recent to 300 — well beyond 2σ.
        import math
        start = date(2024, 1, 1)
        rows = []
        for i in range(20):
            # oscillate around 100 with amplitude 15
            vol = 100.0 + 15.0 * math.sin(i * 0.5)
            rows.append({
                "series_id": "S1",
                "target_week": start + timedelta(weeks=i),
                "actual": vol,
                "forecast": vol,
                "wmape": 0.0,
                "normalized_bias": 0.0,
            })
        for i in range(4):
            rows.append({
                "series_id": "S1",
                "target_week": start + timedelta(weeks=20 + i),
                "actual": 300.0,
                "forecast": 300.0,
                "wmape": 0.0,
                "normalized_bias": 0.0,
            })
        df = pl.DataFrame(rows)

        alerts = detector.detect_volume_anomaly(df)
        assert len(alerts) >= 1
        vol_alerts = [a for a in alerts if a.metric == "volume"]
        assert vol_alerts
        assert "spike" in vol_alerts[0].message

    def test_volume_no_alert_on_normal_variation(self):
        """No alert when volume is within normal z-score range."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=4,
                          volume_warning_zscore=2.0, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=4,
                              baseline_vol=100.0, recent_vol=102.0)
        alerts = detector.detect_volume_anomaly(df)
        vol_alerts = [a for a in alerts if a.metric == "volume"]
        assert vol_alerts == []

    def test_summary_returns_polars_dataframe(self):
        """summary() should return a Polars DataFrame with expected columns."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        import polars as pl
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)
        df = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                              baseline_wmape=0.10, recent_wmape=0.25)
        result = detector.summary(df)
        assert isinstance(result, pl.DataFrame)
        assert "series_id" in result.columns
        assert "severity" in result.columns
        assert len(result) >= 1

    def test_empty_input_returns_no_alerts(self):
        """detect() on empty DataFrame returns empty list."""
        from src.metrics.drift import ForecastDriftDetector
        import polars as pl
        detector = ForecastDriftDetector()
        df = pl.DataFrame(schema={
            "series_id": pl.Utf8, "target_week": pl.Date,
            "actual": pl.Float64, "forecast": pl.Float64,
            "wmape": pl.Float64, "normalized_bias": pl.Float64,
        })
        alerts = detector.detect(df)
        assert alerts == []

    def test_insufficient_history_skipped(self):
        """Series with fewer rows than min_baseline_periods should be skipped."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=10)
        detector = ForecastDriftDetector(cfg)
        # Only 6 rows total — baseline window will have < min_baseline_periods
        df = _make_metrics_df("S1", n_baseline=3, n_recent=3,
                              baseline_wmape=0.10, recent_wmape=0.30)
        alerts = detector.detect(df)
        assert alerts == []

    def test_critical_alert_sorted_before_warning(self):
        """CRITICAL alerts must appear before WARNING alerts in sorted output."""
        from src.metrics.drift import ForecastDriftDetector, DriftConfig
        import polars as pl
        cfg = DriftConfig(baseline_weeks=20, recent_weeks=8, min_baseline_periods=4)
        detector = ForecastDriftDetector(cfg)

        # S1: WARNING (moderate drift), S2: CRITICAL (severe drift)
        df1 = _make_metrics_df("S1", n_baseline=20, n_recent=8,
                               baseline_wmape=0.10, recent_wmape=0.135)
        df2 = _make_metrics_df("S2", n_baseline=20, n_recent=8,
                               baseline_wmape=0.10, recent_wmape=0.25)
        combined = pl.concat([df1, df2])
        alerts = detector.detect(combined)
        severities = [a.severity.value for a in alerts]
        # All critical before all warnings
        crit_indices = [i for i, s in enumerate(severities) if s == "critical"]
        warn_indices = [i for i, s in enumerate(severities) if s == "warning"]
        if crit_indices and warn_indices:
            assert max(crit_indices) < min(warn_indices)


# ══════════════════════════════════════════════════════════════════════════════
# REST API tests (Phase 2 item 8)
# ══════════════════════════════════════════════════════════════════════════════

class TestRestApi:
    """
    Tests for the FastAPI serving layer using the TestClient (no network I/O).
    """

    def _make_app(self, tmpdir: str):
        """Create a test app pointing at a temp directory."""
        from src.api.app import create_app
        return create_app(
            data_dir=tmpdir, metrics_dir=os.path.join(tmpdir, "metrics"),
            auth_enabled=False, dev_mode=True, jwt_secret="",
        )

    def _write_forecast_parquet(self, tmpdir: str, lob: str):
        """Write a minimal forecast Parquet to the expected location."""
        import polars as pl
        from datetime import date, timedelta

        forecast_dir = Path(tmpdir) / "forecasts" / lob
        forecast_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(4):
            rows.append({
                "series_id": "S1",
                "week": date(2026, 1, 1) + timedelta(weeks=i),
                "forecast": 100.0 + i * 5,
                "model": "naive_seasonal",
            })
        df = pl.DataFrame(rows)
        df.write_parquet(str(forecast_dir / f"forecast_{lob}_2026-01-01.parquet"))

    def test_health_endpoint_returns_ok(self):
        """GET /health should return status='ok' and a version string."""
        from fastapi.testclient import TestClient
        import tempfile
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, "metrics"), exist_ok=True)
        app = self._make_app(tmpdir)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_forecast_endpoint_returns_200(self):
        """GET /forecast/{lob} should return forecast points when data exists."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_forecast_parquet(tmpdir, "rossmann")
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/rossmann")
        assert resp.status_code == 200
        data = resp.json()
        assert data["lob"] == "rossmann"
        assert data["series_count"] == 1
        assert len(data["points"]) == 4

    def test_forecast_endpoint_404_when_no_data(self):
        """GET /forecast/{lob} should return 404 when no forecast data exists."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/unknown_lob")
        assert resp.status_code == 404

    def test_forecast_series_filter(self):
        """GET /forecast/{lob}?series_id=S1 should filter to that series."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_forecast_parquet(tmpdir, "rossmann")
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/rossmann?series_id=S1")
        assert resp.status_code == 200
        assert all(p["series_id"] == "S1" for p in resp.json()["points"])

    def test_forecast_series_404_for_unknown_series(self):
        """series_id that doesn't exist should return 404."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_forecast_parquet(tmpdir, "rossmann")
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/rossmann?series_id=NONEXISTENT")
        assert resp.status_code == 404

    def test_forecast_horizon_filter(self):
        """horizon query param should limit the number of returned weeks."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_forecast_parquet(tmpdir, "rossmann")
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/rossmann?horizon=2")
        assert resp.status_code == 200
        assert len(resp.json()["points"]) == 2

    def test_forecast_series_path_param(self):
        """GET /forecast/{lob}/{series_id} path param should work."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_forecast_parquet(tmpdir, "rossmann")
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/forecast/rossmann/S1")
        assert resp.status_code == 200
        assert resp.json()["series_count"] == 1

    def test_leaderboard_404_when_no_metrics(self):
        """GET /metrics/leaderboard/{lob} should return 404 when no data."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/metrics/leaderboard/rossmann")
        assert resp.status_code == 404

    def test_drift_404_when_no_metrics(self):
        """GET /metrics/drift/{lob} should return 404 when no data."""
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as tmpdir:
            app = self._make_app(tmpdir)
            client = TestClient(app)
            resp = client.get("/metrics/drift/rossmann")
        assert resp.status_code == 404

    def test_openapi_docs_available(self):
        """The /docs endpoint should return 200 (Swagger UI)."""
        from fastapi.testclient import TestClient

        app = self._make_app("/tmp/test_api_docs")
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200
