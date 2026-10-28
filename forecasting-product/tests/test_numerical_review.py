"""Reproductions for the six independent Phase IV review findings."""

import sys
from datetime import date, timedelta
from types import ModuleType, SimpleNamespace

import pandas as pd
import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient
from src.analytics.fva_analyzer import FVAAnalyzer
from src.api.app import create_app
from src.config.schema import BacktestConfig, CleansingConfig
from src.data.cleanser import DemandCleanser
from src.forecasting.ml import LGBMDirectForecaster
from src.forecasting.registry import registry
from src.metrics.store import MetricStore
from src.pipeline.backtest import BacktestPipeline
from src.spark.pipeline import forecast_partition
from src.utils.calendar import PeriodCalendar

from tests.test_numerical_reliability import _config, _panel, _rows


def _fva_records(extra_model=False):
    models = {"naive_seasonal": [120.0, 120.0], "auto_arima": [100.0, 200.0]}
    if extra_model:
        models["auto_ets"] = [120.0, 120.0]
    return _rows(models).with_columns(
        ((pl.col("forecast") - pl.col("actual")).abs() / pl.col("actual")).alias("wmape"),
        pl.lit("backtest").alias("run_type"),
    )


def test_fva_uses_whole_model_errors_not_the_best_period():
    analyzer = FVAAnalyzer()
    detail = analyzer.compute_fva_detail(_fva_records())
    statistical = detail.filter(pl.col("forecast_layer") == "statistical").row(0, named=True)
    assert statistical["wmape"] == pytest.approx(0.5, abs=1e-12)
    assert statistical["fva_wmape"] == pytest.approx(-0.3, abs=1e-12)
    assert statistical["fva_class"] == "DESTROYS_VALUE"
    assert analyzer.layer_leaderboard(detail)["recommendation"].to_list() == ["Remove"]


def test_fva_chooses_one_complete_model_and_keeps_its_identity_and_volume():
    detail = FVAAnalyzer().compute_fva_detail(_fva_records(extra_model=True))
    statistical = detail.filter(pl.col("forecast_layer") == "statistical").row(0, named=True)
    assert statistical["model_id"] == "auto_ets"
    assert statistical["wmape"] == pytest.approx(0.2, abs=1e-12)
    assert statistical["total_volume"] == 200.0


def test_fva_rejects_a_model_missing_its_hard_period():
    records = _fva_records(extra_model=True).filter(
        ~((pl.col("model_id") == "auto_arima") & (pl.col("forecast_step") == 2))
    )
    with pytest.raises(ValueError, match="coverage|population"):
        FVAAnalyzer().compute_fva_detail(records)


def test_real_fva_api_does_not_recommend_keep_for_a_worse_layer(tmp_path):
    metrics = tmp_path / "metrics"
    MetricStore(str(metrics)).write(_fva_records(), "backtest", "retail")
    app = create_app(
        data_dir=str(tmp_path / "data"), metrics_dir=str(metrics),
        auth_enabled=False, dev_mode=True, jwt_secret="", audit_log_path=str(tmp_path / "audit"),
    )
    with TestClient(app) as client:
        response = client.get("/metrics/retail/fva")
    assert response.status_code == 200, response.text
    statistical = next(row for row in response.json()["detail_preview"] if row["forecast_layer"] == "statistical")
    assert statistical["fva_wmape"] == pytest.approx(-0.3, abs=1e-12)
    assert response.json()["layer_leaderboard"][0]["recommendation"] == "Remove"


def test_fva_keeps_runs_separate_and_marks_zero_volume_undefined():
    first = _fva_records()
    second = first.with_columns(
        pl.lit("other-run").alias("run_id"),
        pl.when(pl.col("model_id") == "auto_arima").then(140.0).otherwise(pl.col("forecast")).alias("forecast"),
    )
    analyzer = FVAAnalyzer()
    detail = analyzer.compute_fva_detail(pl.concat([first, second]))
    values = detail.filter(pl.col("forecast_layer") == "statistical")
    assert dict(zip(values["run_id"], values["wmape"])) == pytest.approx({"oracle": 0.5, "other-run": 0.4})
    assert analyzer.layer_leaderboard(detail)["rank"].to_list() == [1, 1]
    undefined = analyzer.compute_fva_detail(first.with_columns(pl.lit(0.0).alias("actual")))
    board = analyzer.layer_leaderboard(undefined)
    assert board["recommendation"].to_list() == ["Insufficient data"]
    assert board["mean_fva_wmape"].to_list() == [None]


def test_real_fva_api_reports_incomplete_coverage(tmp_path):
    records = _fva_records().head(3)
    metrics = tmp_path / "metrics"
    MetricStore(str(metrics)).write(records, "backtest", "retail")
    app = create_app(
        data_dir=str(tmp_path / "data"), metrics_dir=str(metrics),
        auth_enabled=False, dev_mode=True, jwt_secret="", audit_log_path=str(tmp_path / "audit"),
    )
    with TestClient(app) as client:
        response = client.get("/metrics/retail/fva")
    assert response.status_code == 409, response.text
    assert "coverage" in response.json()["detail"]


def _unequal_volume_records():
    return _rows({"naive_seasonal": [900.0, 0.0]}, (1000.0, 1.0)).with_columns(
        ((pl.col("forecast") - pl.col("actual")).abs() / pl.col("actual")).alias("wmape"),
        pl.lit("backtest").alias("run_type"),
    )


@pytest.mark.parametrize("consumer", ["model_card", "commentary", "nl_query", "notebook", "bi_export"])
def test_reporting_consumers_recompute_raw_volume_metrics(tmp_path, consumer):
    records = _unequal_volume_records()
    expected = 101 / 1001
    if consumer == "model_card":
        from src.analytics.governance import ModelCard
        card = ModelCard.from_backtest("naive_seasonal", "retail", records)
        assert card.backtest_wmape == pytest.approx(expected, rel=1e-12)
        assert card.backtest_bias == pytest.approx(-expected, rel=1e-12)
    elif consumer == "commentary":
        from src.ai.commentary import CommentaryEngine
        stats = CommentaryEngine._compute_stats(records)
        assert stats["overall_wmape"] == round(expected, 4)
        assert stats["overall_bias"] == round(-expected, 4)
        assert stats["metric_population"] == "all supplied observations"
    elif consumer == "nl_query":
        from src.ai.nl_query import NaturalLanguageQueryEngine
        engine = NaturalLanguageQueryEngine.__new__(NaturalLanguageQueryEngine)
        context, _ = engine._gather_context("s", None, None, records, None, None, None)
        assert context["metrics"]["wmape"] == round(expected, 4)
        assert context["metrics"]["bias"] == round(-expected, 4)
    else:
        path = tmp_path / "metrics"
        store = MetricStore(str(path))
        store.write(records, "backtest", "retail")
        if consumer == "notebook":
            from src.analytics.notebook_api import ForecastAnalytics
            analytics = ForecastAnalytics(str(path))
            assert analytics.model_comparison_by_fold("retail")["wmape"].item() == pytest.approx(expected)
            assert analytics.bias_distribution(lob="retail")["avg_wmape"].item() == pytest.approx(expected)
            assert analytics.accuracy_by_grain("series_id", lob="retail")["wmape"].item() == pytest.approx(expected)
            assert analytics.transition_impact("s", "absent")["wmape"].item() == pytest.approx(expected)
        else:
            from src.analytics.bi_export import BIExporter
            report = BIExporter(str(tmp_path / "bi")).export_bias_report(store, "retail")
            assert pl.read_parquet(report)["avg_wmape"].item() == pytest.approx(expected)


def test_sparse_series_pipeline_keeps_its_competitions_scoped(tmp_path):
    config = _config(tmp_path)
    config.forecast.intermittent_forecasters = ["croston"]
    config.backtest.champion_granularity = "series"
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(60)]
    dense = _panel(dates, [10.0 + index % 7 for index in range(60)]).with_columns(
        pl.lit("dense").alias("series_id")
    )
    sparse = _panel(dates, [10.0 if index % 4 == 0 else 0.0 for index in range(60)]).with_columns(
        pl.lit("sparse").alias("series_id")
    )
    result = BacktestPipeline(config).run(pl.concat([dense, sparse]))
    expected = {"dense": "naive_seasonal", "sparse": "croston"}
    assert {row["group_key"]: row["model_id"] for row in result["champions"].iter_rows(named=True)} == expected
    assert {row["group_key"]: row["model_id"] for row in result["leaderboard"].iter_rows(named=True)} == expected
    assert result["leaderboard"]["competition_scope"].unique().to_list() == ["series_id"]
    assert result["validation"]["skipped"] is False
    assert all(result["validation"]["layers"][name] is not None for name in ("structural", "logical", "business"))
    assert result["validation"]["layers"]["paradox"]["status"] == "not_applicable"
    assert result["validation"]["confidence"]["factors"]["paradox"]["unavailable"] is True
    assert result["validation"]["confidence"]["factors"]["paradox"]["score"] == 50
    persisted = MetricStore(config.output.metrics_path).leaderboard(lob=config.lob)
    assert {row["group_key"]: row["model_id"] for row in persisted.iter_rows(named=True)} == expected


def test_scoped_leaderboard_still_rejects_missing_cases_within_a_series():
    records = _fva_records().head(3).with_columns(pl.lit("series_id").alias("competition_scope"))
    with pytest.raises(ValueError, match="coverage"):
        MetricStore.leaderboard_from_records(records)


def test_leaderboard_counting_does_not_depend_on_new_polars_len_apis(monkeypatch):
    records = _fva_records().with_columns(pl.lit("lob").alias("competition_scope"))

    def unsupported(*args, **kwargs):
        pytest.fail("This counting API is unavailable in the standalone Polars 0.20 gate")

    monkeypatch.setattr(pl, "len", unsupported, raising=False)
    monkeypatch.setattr(type(records.group_by("model_id")), "len", unsupported, raising=False)
    board = MetricStore.leaderboard_from_records(records)
    assert board["n_observations"].to_list() == [2, 2]
    assert board["model_id"].to_list() == ["naive_seasonal", "auto_arima"]


def test_sparse_series_api_has_scoped_rows_not_a_false_global_best(tmp_path):
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(60)]
    dense = _panel(dates).with_columns(pl.lit("dense").alias("series_id"))
    sparse = _panel(dates, [10.0 if index % 4 == 0 else 0.0 for index in range(60)]).with_columns(
        pl.lit("sparse").alias("series_id")
    )
    config = {
        "forecast": {"frequency": "D", "forecasters": ["naive_seasonal"], "intermittent_forecasters": ["croston"]},
        "backtest": {"n_folds": 2, "val_periods": 2, "champion_granularity": "series"},
        "data_quality": {"min_series_length_periods": 1},
        "post_validation": {"enabled": False},
    }
    app = create_app(
        data_dir=str(tmp_path / "data"), metrics_dir=str(tmp_path / "data" / "metrics"),
        auth_enabled=False, dev_mode=True, jwt_secret="", audit_log_path=str(tmp_path / "audit"),
    )
    with TestClient(app) as client:
        response = client.post("/pipeline/backtest?lob=retail", files={
            "file": ("actuals.csv", pl.concat([dense, sparse]).write_csv().encode(), "text/csv"),
            "config_file": ("config.yaml", yaml.safe_dump(config).encode(), "application/x-yaml"),
        })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["best_wmape"] is None
    assert "champion_model" not in body
    assert {row["group_key"] for row in body["leaderboard"]} == {"dense", "sparse"}
    assert {row["competition_scope"] for row in body["leaderboard"]} == {"series_id"}


def test_pure_ml_worker_uses_the_same_prepared_panel_origin_as_local(monkeypatch):
    dates = [date(2023, 10, 9) + timedelta(weeks=index) for index in range(14)]
    assert dates[-2:] == [date(2024, 1, 1), date(2024, 1, 8)]
    long = _panel(dates, [float(10 + index * 2) for index in range(14)]).with_columns(
        pl.lit("long").alias("series_id"), pl.lit(2.0).alias("planned_promo")
    )
    short = _panel(dates[:-1], [float(100 + index * 3) for index in range(13)]).with_columns(
        pl.lit("short").alias("series_id"), pl.lit(3.0).alias("planned_promo")
    )
    panel = pl.concat([long, short])

    def build_model(*args, **kwargs):
        return LGBMDirectForecaster(
            frequency=kwargs.get("frequency", "W"), lags=[1, 2], lag_transforms={1: []},
            lgbm_params={"n_estimators": 3, "min_child_samples": 1, "num_leaves": 2},
        )

    local = build_model(frequency="W")
    local.fit(panel)
    assert local._mlf is not None, "This regression must exercise real mlforecast"
    expected = local.predict(2).sort(["series_id", "week"])
    monkeypatch.setattr(registry, "build", build_model)
    observed = pl.from_pandas(forecast_partition(panel.to_pandas(), {
        "id_col": "series_id", "time_col": "week", "target_col": "quantity",
        "model_name": "lgbm_direct", "frequency": "W", "horizon": 2,
    })).with_columns(pl.col("week").cast(pl.Date)).sort(["series_id", "week"])
    assert observed.to_dicts() == expected.to_dicts()
    for group in observed.partition_by("series_id"):
        assert group["week"].to_list() == [date(2024, 1, 15), date(2024, 1, 22)]


@pytest.mark.parametrize("unit", ["ns", "ms"])
@pytest.mark.parametrize("time_zone", [None, "America/New_York"])
def test_monthly_imputation_preserves_timestamp_labels_and_units(unit, time_zone):
    dates = [date(2023, 2, 28), date(2023, 3, 31), date(2023, 4, 30), date(2023, 5, 31),
             date(2023, 6, 30), date(2023, 7, 31), date(2023, 8, 31), date(2023, 9, 30),
             date(2023, 10, 31), date(2023, 11, 30), date(2023, 12, 31), date(2024, 1, 31),
             date(2024, 2, 29)]
    dtype = pl.Datetime(unit, time_zone)
    labels = pl.Series("week", dates).cast(pl.Datetime(unit))
    if time_zone is not None:
        labels = labels.dt.replace_time_zone(time_zone)
    labels = (labels.cast(pl.Int64) + 123).cast(dtype)
    source = _panel(dates, [70.0] + [10.0] * 11 + [0.0]).with_columns(
        labels, pl.col("quantity").eq(0).alias("_stockout_flag"),
    )
    cleaned = DemandCleanser(CleansingConfig(), frequency="M")._impute_seasonal(
        source, "week", "quantity", "series_id", frequency="M",
    ).sort("week")
    assert cleaned.schema["week"] == dtype
    assert cleaned["week"].cast(pl.Int64).to_list() == labels.cast(pl.Int64).to_list()
    assert cleaned["quantity"][-1] == 70.0


def test_backend_offsets_use_the_installed_public_alias_and_keep_fiscal_anchor(monkeypatch):
    seen = []
    monkeypatch.setattr(pd.tseries.offsets, "MonthEnd", lambda: SimpleNamespace(freqstr="M"))

    def old_quarter_end(*, startingMonth):
        seen.append(startingMonth)
        return SimpleNamespace(freqstr="Q-NOV")

    monkeypatch.setattr(pd.tseries.offsets, "QuarterEnd", old_quarter_end)
    assert PeriodCalendar.from_dates([date(2024, 1, 31)], "M").pandas_frequency == "M"
    assert PeriodCalendar.from_dates([date(2023, 11, 30)], "Q").pandas_frequency == "Q-NOV"
    assert seen == [11]


class _Frame:
    """Only the public column operations needed for the real adapter contract."""
    def __init__(self, data):
        self.data = data

    @property
    def columns(self):
        return self.data.columns

    def filter(self, expression):
        return _Frame(self.data.filter(expression))

    def select(self, *columns):
        return _Frame(self.data.select(*columns))

    def first(self):
        return self.data.row(0) if self.data.height else None

    def withColumnRenamed(self, old, new):
        return _Frame(self.data.rename({old: new}))

    def toDF(self, *columns):
        return _Frame(self.data.rename(dict(zip(self.data.columns, columns))))

    def withColumn(self, name, expression):
        return _Frame(self.data.with_columns(expression.alias(name)))

    def groupby(self, *columns):
        return SimpleNamespace(agg=lambda *expressions: _Frame(self.data.group_by(columns).agg(*expressions)))

    def limit(self, count):
        return _Frame(self.data.head(count))

    def collect(self):
        return self.data.to_dicts()


def test_real_fabric_adapter_handles_canonical_and_legacy_model_columns(monkeypatch):
    from src.fabric.notebook_adapter import FabricNotebookAdapter

    functions = ModuleType("pyspark.sql.functions")
    functions.col = pl.col
    sql = ModuleType("pyspark.sql")
    sql.functions = functions
    spark = ModuleType("pyspark")
    spark.sql = sql
    monkeypatch.setitem(sys.modules, "pyspark", spark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
    board = pl.DataFrame({"model_id": ["naive_seasonal", "auto_arima"], "rank": [1, 2]})
    seen = []

    def select_champion(frame, **kwargs):
        seen.append(kwargs)
        return _Frame(board)

    adapter = FabricNotebookAdapter.__new__(FabricNotebookAdapter)
    adapter._config = SimpleNamespace(backtest=BacktestConfig(primary_metric="mae"))
    adapter._pipeline = SimpleNamespace(select_champion=select_champion)
    assert adapter.select_champion(None) == "naive_seasonal"
    assert seen[-1] == {"primary_metric": "mae", "secondary_metric": "normalized_bias"}
    board = board.rename({"model_id": "model"})
    assert adapter.select_champion(None) == "naive_seasonal"


def test_real_deployment_write_boundary_preserves_legacy_delta_columns_and_grain(monkeypatch):
    from src.fabric import delta_writer
    from src.fabric.deployment import DeploymentOrchestrator
    from src.spark.pipeline import public_spark_leaderboard

    functions = ModuleType("pyspark.sql.functions")
    functions.col, functions.lit = pl.col, pl.lit
    functions.sum = lambda expression: expression.sum()
    functions.abs = lambda expression: expression.abs()
    functions.avg = lambda expression: expression.mean()
    functions.when = lambda condition, value: pl.when(condition).then(value)
    sql = ModuleType("pyspark.sql")
    sql.functions = functions
    spark = ModuleType("pyspark")
    spark.sql = sql
    monkeypatch.setitem(sys.modules, "pyspark", spark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
    written = {}

    class Writer:
        def __init__(self, *args):
            pass

        def append(self, frame, table_name, **kwargs):
            written[table_name] = frame.data

        def upsert(self, frame, table_name, **kwargs):
            assert kwargs["merge_keys"] == ["lob", "run_date", "model"]
            written[table_name] = frame.data

    monkeypatch.setattr(delta_writer, "DeltaWriter", Writer)
    board = public_spark_leaderboard(_Frame(pl.DataFrame({
        "model_id": ["naive_seasonal", "auto_arima"], "wmape": [0.2, 0.5],
        "normalized_bias": [0.2, 0.5], "mae": [20.0, 50.0], "n_evals": [1, 1],
        "n_observations": [2, 2], "rank": [1, 2],
    })))
    raw = _Frame(_fva_records().rename({"model_id": "model"}))
    deployment = DeploymentOrchestrator.__new__(DeploymentOrchestrator)
    deployment.spark = None
    deployment.config = SimpleNamespace(forecast=SimpleNamespace(series_id_column="series_id"))
    deployment.dc = SimpleNamespace(workspace="test", lakehouse="test", lob="retail")
    deployment._write_leaderboard(raw, board)
    assert set(written["leaderboard"].columns) == {
        "model", "mean_wmape", "mean_normalized_bias", "mean_mae", "n_evals", "rank",
        "lob", "run_date", "champion_model",
    }
    assert set(written["backtest_results"].columns) == {
        "series_id", "model", "fold", "wmape", "normalized_bias", "mae", "lob", "run_date",
    }
    assert written["backtest_results"].height == 2  # one row per model/series/fold, not four observations
    assert written["backtest_results"].sort("model")["wmape"].to_list() == pytest.approx([0.5, 0.2])
    assert written["leaderboard"]["champion_model"].unique().to_list() == ["naive_seasonal"]
