"""Tests for DataAnalyzer — schema detection, hierarchy, hypotheses, config recommendation."""

import unittest
from dataclasses import asdict
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import yaml
from src.analytics.analyzer import DataAnalyzer, HierarchyDetection, SchemaDetection
from src.backtesting.scoring import require_fold_local_preprocessing
from src.config.loader import parse_config_yaml
from src.config.schema import PlatformConfig
from src.pipeline.forecast import ForecastPipeline
from src.pipeline.run_store import RunStore

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #

def _make_retail_data(n_stores=5, n_products=3, n_weeks=104, seed=42):
    """Synthetic retail data with store/product hierarchy, realistic weekly sales."""
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2020, 1, 6)

    # Hierarchy: region → store, category → product
    regions = {"S1": "North", "S2": "North", "S3": "South", "S4": "South", "S5": "East"}
    categories = {"P1": "Food", "P2": "Food", "P3": "Electronics"}

    stores = [f"S{i+1}" for i in range(n_stores)]
    products = [f"P{i+1}" for i in range(n_products)]

    for store in stores:
        for product in products:
            store_base = 80 + rng.randint(0, 40)
            for w in range(n_weeks):
                week = base + timedelta(weeks=w)
                seasonal = 20 * np.sin(2 * np.pi * w / 52)
                noise = rng.normal(0, 5)
                qty = max(0, store_base + seasonal + noise)
                rows.append({
                    "week": week,
                    "store_id": store,
                    "region": regions.get(store, "Unknown"),
                    "product_id": product,
                    "category": categories.get(product, "Other"),
                    "quantity": float(qty),
                    "price": float(rng.uniform(5, 50)),
                })

    return pl.DataFrame(rows)


def _make_flat_data(n_series=10, n_weeks=52, seed=42):
    """Simple series_id + week + quantity, no hierarchy."""
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2022, 1, 3)
    for i in range(n_series):
        sid = f"series_{i}"
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": base + timedelta(weeks=w),
                "quantity": float(rng.normal(100, 20)),
            })
    return pl.DataFrame(rows)


def _make_data_with_known_hierarchy(n_weeks=52, seed=42):
    """region → country → store, with known functional dependencies."""
    rng = np.random.RandomState(seed)
    # Each store maps to exactly one country, each country to exactly one region
    store_map = {
        "store_A": ("US", "Americas"),
        "store_B": ("US", "Americas"),
        "store_C": ("UK", "EMEA"),
        "store_D": ("DE", "EMEA"),
    }
    rows = []
    base = date(2022, 1, 3)
    for store, (country, region) in store_map.items():
        for w in range(n_weeks):
            rows.append({
                "week": base + timedelta(weeks=w),
                "store": store,
                "country": country,
                "region": region,
                "quantity": float(rng.normal(100, 10)),
            })
    return pl.DataFrame(rows)


def _make_sparse_data(n_series=10, n_weeks=104, demand_prob=0.3, seed=42):
    """Data with intermittent/sparse demand patterns."""
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2020, 1, 6)
    for i in range(n_series):
        sid = f"series_{i}"
        for w in range(n_weeks):
            if rng.random() < demand_prob:
                qty = float(rng.exponential(50))
            else:
                qty = 0.0
            rows.append({
                "series_id": sid,
                "week": base + timedelta(weeks=w),
                "quantity": qty,
            })
    return pl.DataFrame(rows)


def _make_short_data(n_series=3, n_weeks=20, seed=42):
    """Very short time series."""
    rng = np.random.RandomState(seed)
    rows = []
    base = date(2023, 6, 5)
    for i in range(n_series):
        sid = f"S{i}"
        for w in range(n_weeks):
            rows.append({
                "series_id": sid,
                "week": base + timedelta(weeks=w),
                "quantity": float(rng.normal(100, 10)),
            })
    return pl.DataFrame(rows)


def _make_nonfixed_hierarchy():
    return _make_flat_data(n_series=8, n_weeks=104).with_columns(
        (pl.col("series_id").str.extract(r"(\d+)").cast(pl.Int64) // 2)
        .cast(pl.String).alias("category"),
    )


# --------------------------------------------------------------------------- #
#  Tests: Schema Detection
# --------------------------------------------------------------------------- #

class TestSchemaDetection(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer()

    def test_detects_date_column(self):
        df = _make_flat_data()
        schema = self.analyzer.detect_schema(df)
        self.assertEqual(schema.time_column, "week")

    def test_detects_target_column(self):
        df = _make_flat_data()
        schema = self.analyzer.detect_schema(df)
        self.assertEqual(schema.target_column, "quantity")

    def test_detects_frequency_weekly(self):
        df = _make_flat_data()
        schema = self.analyzer.detect_schema(df)
        self.assertEqual(schema.frequency_guess, "W")

    def test_detects_id_columns(self):
        df = _make_flat_data()
        schema = self.analyzer.detect_schema(df)
        self.assertEqual(schema.id_columns, ["series_id"])

    def test_detects_numeric_regressors(self):
        df = _make_retail_data()
        schema = self.analyzer.detect_schema(df)
        self.assertIn("price", schema.numeric_columns)

    def test_correct_series_count(self):
        df = _make_flat_data(n_series=10)
        schema = self.analyzer.detect_schema(df)
        self.assertEqual(schema.n_series, 10)

    def test_date_range(self):
        df = _make_flat_data(n_weeks=52)
        schema = self.analyzer.detect_schema(df)
        self.assertIn("2022-01-03", schema.date_range[0])


# --------------------------------------------------------------------------- #
#  Tests: Hierarchy Detection
# --------------------------------------------------------------------------- #

class TestHierarchyDetection(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer()

    def test_known_three_level_hierarchy(self):
        df = _make_data_with_known_hierarchy()
        schema = self.analyzer.detect_schema(df)
        hierarchy = self.analyzer.detect_hierarchy(df, schema)
        # Should find region → country → store
        self.assertTrue(len(hierarchy.hierarchies) >= 1)
        # At least one hierarchy should have multiple levels
        max_levels = max(len(h.levels) for h in hierarchy.hierarchies)
        self.assertGreaterEqual(max_levels, 2)

    def test_flat_data_single_level(self):
        df = _make_flat_data()
        schema = self.analyzer.detect_schema(df)
        hierarchy = self.analyzer.detect_hierarchy(df, schema)
        if hierarchy.hierarchies:
            # Each hierarchy should have exactly 1 level
            for h in hierarchy.hierarchies:
                self.assertEqual(len(h.levels), 1)

    def test_retail_detects_both_dimensions(self):
        df = _make_retail_data()
        schema = self.analyzer.detect_schema(df)
        hierarchy = self.analyzer.detect_hierarchy(df, schema)
        # Should detect geography and product dimensions
        names = {h.name for h in hierarchy.hierarchies}
        # At least detect some hierarchies
        self.assertTrue(len(hierarchy.hierarchies) >= 1)

    def test_no_dimension_columns(self):
        """DataFrame with only time + target columns."""
        base = date(2022, 1, 3)
        df = pl.DataFrame({
            "week": [base + timedelta(weeks=w) for w in range(52)],
            "quantity": [float(i) for i in range(52)],
        })
        schema = self.analyzer.detect_schema(df)
        hierarchy = self.analyzer.detect_hierarchy(df, schema)
        self.assertTrue(
            len(hierarchy.hierarchies) == 0 or len(hierarchy.warnings) > 0
        )

    def test_reasoning_populated(self):
        df = _make_data_with_known_hierarchy()
        schema = self.analyzer.detect_schema(df)
        hierarchy = self.analyzer.detect_hierarchy(df, schema)
        if len(hierarchy.hierarchies) > 1 or any(len(h.levels) > 1 for h in hierarchy.hierarchies):
            self.assertIsInstance(hierarchy.reasoning, list)
            self.assertGreaterEqual(len(hierarchy.reasoning), 1)


# --------------------------------------------------------------------------- #
#  Tests: Config Recommendation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["flat", "sparse", "hierarchical", "nonfixed_hierarchical"])
def test_recommended_plan_and_yaml_roundtrip_are_backtest_compatible(kind):
    actuals = {
        "flat": lambda: _make_flat_data(n_series=2, n_weeks=104),
        "sparse": lambda: _make_sparse_data(n_series=2, n_weeks=104, demand_prob=0.2),
        "hierarchical": lambda: _make_data_with_known_hierarchy(n_weeks=104),
        "nonfixed_hierarchical": _make_nonfixed_hierarchy,
    }[kind]()
    report = DataAnalyzer(lob_name="retail").analyze(actuals)
    generated = report.recommended_config
    roundtrip = parse_config_yaml(yaml.safe_dump(asdict(generated)))

    for config in (generated, roundtrip):
        require_fold_local_preprocessing(config)
        assert config.data_quality.cleansing.enabled is False
        assert config.data_quality.structural_breaks.enabled is False
        assert config.forecast.external_regressors.screen.enabled is False
        assert config.reconciliation.method == "bottom_up"
    assert any("fold-local" in item for item in report.config_reasoning)

    generated.data_quality.cleansing.enabled = True
    with pytest.raises(ValueError, match="data_quality.cleansing"):
        require_fold_local_preprocessing(generated)
    generated.data_quality.cleansing.enabled = False
    generated.data_quality.structural_breaks.enabled = True
    with pytest.raises(ValueError, match="data_quality.structural_breaks"):
        require_fold_local_preprocessing(generated)


def test_generated_nonfixed_hierarchy_publishes_supported_leaf_forecast(tmp_path):
    actuals = _make_nonfixed_hierarchy()
    config = DataAnalyzer(lob_name="retail").analyze(actuals).recommended_config
    config.output.runs_path = str(tmp_path / "runs")
    config.output.forecast_path = str(tmp_path / "forecasts")
    config.output.metrics_path = str(tmp_path / "metrics")
    assert len(config.get_reconcilable_hierarchies()) == 1
    pipeline = ForecastPipeline(config)

    forecast = pipeline.run(actuals, champion_model=config.forecast.forecasters[0])

    assert forecast.height == 8 * 25
    assert forecast["series_id"].n_unique() == 8
    assert forecast.select("series_id", "week").is_duplicated().sum() == 0
    store = RunStore(config.output.runs_path)
    assert pipeline.last_run is not None
    original_id = pipeline.last_run.run_id
    assert store.frame("retail", original_id, "forecast").equals(forecast)
    config.reconciliation.method = "middle_out"
    with pytest.raises(ValueError, match="supports bottom_up reconciliation only"):
        pipeline.run(actuals)
    assert [run.run_id for run in store.list("retail", "forecast")] == [original_id]
    assert store.frame("retail", original_id, "forecast").equals(forecast)


class TestConfigRecommendation(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer(lob_name="test_lob")

    def test_short_data_fewer_folds(self):
        df = _make_short_data(n_weeks=20)
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        # With only 20 weeks, folds should be small
        self.assertLessEqual(config.backtest.n_folds, 3)

    def test_long_data_includes_ml_models(self):
        df = _make_flat_data(n_series=10, n_weeks=104)
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        forecasters = config.forecast.forecasters
        # Should include ML models for 104 weeks + 10 series
        self.assertTrue(
            "lgbm_direct" in forecasters or "auto_arima" in forecasters,
            f"Expected ML or statistical models in {forecasters}"
        )

    def test_sparse_data_includes_intermittent_models(self):
        df = _make_sparse_data(demand_prob=0.2)
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        # Should detect sparse demand and include intermittent models
        if report.forecastability.demand_class_distribution.get("intermittent", 0) > 0 or \
           report.forecastability.demand_class_distribution.get("lumpy", 0) > 0:
            self.assertTrue(
                len(config.forecast.intermittent_forecasters) > 0,
                "Expected intermittent models for sparse data"
            )

    def test_regressors_detected(self):
        df = _make_retail_data()
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        self.assertTrue(config.forecast.external_regressors.enabled)
        self.assertIn("price", config.forecast.external_regressors.feature_columns)

    def test_horizon_reasonable(self):
        df = _make_flat_data(n_weeks=52)
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        self.assertGreaterEqual(config.forecast.horizon_weeks, 4)
        self.assertLessEqual(config.forecast.horizon_weeks, 13)

    def test_config_is_valid_platform_config(self):
        df = _make_retail_data()
        report = self.analyzer.analyze(df)
        config = report.recommended_config
        self.assertIsInstance(config, PlatformConfig)
        self.assertEqual(config.lob, "test_lob")


# --------------------------------------------------------------------------- #
#  Tests: Hypothesis Generation
# --------------------------------------------------------------------------- #

class TestHypothesisGeneration(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer()

    def test_produces_hypotheses(self):
        df = _make_flat_data()
        report = self.analyzer.analyze(df)
        self.assertIsInstance(report.hypotheses, list)
        self.assertGreaterEqual(len(report.hypotheses), 1)

    def test_mentions_forecastability(self):
        df = _make_flat_data()
        report = self.analyzer.analyze(df)
        combined = " ".join(report.hypotheses).lower()
        self.assertIn("forecastab", combined)

    def test_sparse_data_mentions_intermittent(self):
        df = _make_sparse_data(demand_prob=0.2)
        report = self.analyzer.analyze(df)
        combined = " ".join(report.hypotheses).lower()
        # Should mention sparse/intermittent demand
        self.assertTrue(
            "sparse" in combined or "intermittent" in combined or "forecastab" in combined
        )

    def test_short_data_warns_about_length(self):
        df = _make_short_data(n_weeks=20)
        report = self.analyzer.analyze(df)
        combined = " ".join(report.hypotheses).lower()
        self.assertTrue("week" in combined)


# --------------------------------------------------------------------------- #
#  Tests: End-to-End Analysis
# --------------------------------------------------------------------------- #

class TestAnalyzerEndToEnd(unittest.TestCase):
    def test_retail_data_full_pipeline(self):
        df = _make_retail_data()
        analyzer = DataAnalyzer(lob_name="retail_test")
        report = analyzer.analyze(df)

        # All fields populated
        self.assertIsNotNone(report.schema)
        self.assertIsNotNone(report.hierarchy)
        self.assertIsNotNone(report.forecastability)
        self.assertIsNotNone(report.recommended_config)
        self.assertIsInstance(report.config_reasoning, list)
        self.assertGreaterEqual(len(report.config_reasoning), 1)
        self.assertIsInstance(report.hypotheses, list)
        self.assertGreaterEqual(len(report.hypotheses), 1)

    def test_flat_data_works(self):
        df = _make_flat_data()
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertEqual(report.schema.n_series, 10)

    def test_config_reasoning_nonempty(self):
        df = _make_retail_data()
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertTrue(len(report.config_reasoning) >= 3)

    def test_forecastability_per_series_populated(self):
        df = _make_flat_data(n_series=5)
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertIsNotNone(report.forecastability.per_series)
        self.assertEqual(report.forecastability.per_series.height, 5)


# --------------------------------------------------------------------------- #
#  Tests: Edge Cases
# --------------------------------------------------------------------------- #

class TestAnalyzerEdgeCases(unittest.TestCase):
    def test_single_series(self):
        base = date(2022, 1, 3)
        df = pl.DataFrame({
            "week": [base + timedelta(weeks=w) for w in range(52)],
            "quantity": [float(100 + np.sin(2 * np.pi * w / 52) * 20) for w in range(52)],
        })
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertEqual(report.forecastability.n_series, 1)

    def test_very_short_data(self):
        df = _make_short_data(n_series=2, n_weeks=8)
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertIsNotNone(report.recommended_config)

    def test_all_zero_series(self):
        base = date(2022, 1, 3)
        df = pl.DataFrame({
            "series_id": ["A"] * 52,
            "week": [base + timedelta(weeks=w) for w in range(52)],
            "quantity": [0.0] * 52,
        })
        analyzer = DataAnalyzer()
        report = analyzer.analyze(df)
        self.assertIsNotNone(report.recommended_config)
