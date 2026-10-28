# Fabric Notebook: 02 — Distributed Backtest & Champion Selection
# ================================================================
# Run this notebook after 01_data_prep.py to:
#   1. Read processed actuals from the Lakehouse Delta table.
#   2. Run walk-forward backtest across all configured models in parallel.
#   3. Select the champion model per LOB.
#   4. Write backtest results and leaderboard to the Lakehouse.
#
# Prerequisites: 01_data_prep.py must have run successfully.

# %% [markdown]
# ## 0 — Configuration

# %%
import os
import sys
import yaml

PLATFORM_ROOT = "/lakehouse/default/Files/forecasting-product"
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

WORKSPACE       = os.environ.get("FABRIC_WORKSPACE", "my-workspace")
LAKEHOUSE       = os.environ.get("FABRIC_LAKEHOUSE", "my-lakehouse")
LOB             = os.environ.get("FORECAST_LOB", "rossmann")
CONFIG_PATH     = os.environ.get("FORECAST_CONFIG", "configs/platform_config.yaml")
ENVIRONMENT     = os.environ.get("FABRIC_ENVIRONMENT", "development")
FABRIC_CFG_PATH = os.environ.get("FABRIC_CONFIG", "configs/fabric_config.yaml")

with open(FABRIC_CFG_PATH) as _f:
    fabric_yaml = yaml.safe_load(_f)

print(f"Workspace  : {WORKSPACE}")
print(f"LOB        : {LOB}")
print(f"Config     : {CONFIG_PATH}")

# %% [markdown]
# ## 1 — Spark Session

# %%
from src.spark.session import get_or_create_spark

spark = get_or_create_spark(app_name=f"ForecastingPlatform-Backtest-{LOB}")
spark.sparkContext.setLogLevel("WARN")
print(f"Spark version: {spark.version}")

# %% [markdown]
# ## 2 — Load platform config

# %%
from src.config.loader import load_config

config = load_config(CONFIG_PATH)
config.lob = LOB
print(f"Models to evaluate : {config.forecast.forecasters}")
print(f"Backtest folds     : {config.backtest.n_folds}")
print(f"Validation weeks   : {config.backtest.val_weeks}")

# %% [markdown]
# ## 3 — Read actuals from Lakehouse

# %%
from src.fabric.config import FabricConfig
from src.fabric.lakehouse import FabricLakehouse

fabric_cfg = FabricConfig(workspace=WORKSPACE, lakehouse=LAKEHOUSE, environment=ENVIRONMENT)
lh = FabricLakehouse(spark, fabric_cfg)

actuals_sdf = lh.read_table("actuals")
print(f"Actuals loaded: {actuals_sdf.count():,} rows")
actuals_sdf.printSchema()

# %% [markdown]
# ## 4 — Build canonical series panel
#
# Column mapping is read from fabric_config.yaml → series_builder.
# No LOB-specific column names appear in this cell.

# %%
from src.spark.series_builder import SparkSeriesBuilder

builder = SparkSeriesBuilder.from_config(fabric_yaml["series_builder"])
actuals_series_sdf = builder.build(actuals_sdf)

actuals_series_sdf.show(5)

# %% [markdown]
# ## 5 — Run distributed backtest

# %%
from src.spark.pipeline import (
    SparkForecastPipeline,
    backtest_metrics_for_legacy_storage,
    leaderboard_for_legacy_storage,
)

pipeline = SparkForecastPipeline(spark, config)

backtest_results_sdf = pipeline.run_backtest(
    actuals_sdf=actuals_series_sdf,
    model_names=config.forecast.forecasters,
)
backtest_results_sdf.cache()
print(f"Backtest result rows: {backtest_results_sdf.count():,}")
backtest_results_sdf.show(20)

# %% [markdown]
# ## 6 — Select champion model

# %%
from pyspark.sql import functions as F

leaderboard_sdf = pipeline.select_champion(
    backtest_results_sdf,
    primary_metric=config.backtest.primary_metric,
    secondary_metric=config.backtest.secondary_metric,
)
leaderboard_sdf.show(truncate=False)

champion_model = leaderboard_sdf.filter(F.col("rank") == 1).select("model").first()[0]
print(f"\nChampion model: {champion_model}")

# %% [markdown]
# ## 7 — Write results to Lakehouse

# %%
from datetime import date
from src.fabric.delta_writer import DeltaWriter

writer = DeltaWriter(spark, fabric_cfg)
run_date = date.today().isoformat()

backtest_results_sdf_out = (
    backtest_metrics_for_legacy_storage(backtest_results_sdf, config.forecast.series_id_column)
    .withColumn("lob", F.lit(LOB))
    .withColumn("run_date", F.lit(run_date))
)
writer.append(
    backtest_results_sdf_out,
    table_name="backtest_results",
    partition_by=["lob", "run_date"],
)
print("Backtest results written.")

leaderboard_out = (
    leaderboard_for_legacy_storage(leaderboard_sdf)
    .withColumn("lob", F.lit(LOB))
    .withColumn("run_date", F.lit(run_date))
    .withColumn("champion_model", F.lit(champion_model))
)
writer.upsert(
    leaderboard_out,
    table_name="leaderboard",
    merge_keys=["lob", "run_date", "model"],
)
print("Leaderboard written.")

# %% [markdown]
# ## 8 — Summary

# %%
print("=" * 60)
print(f"LOB            : {LOB}")
print(f"Run date       : {run_date}")
print(f"Champion model : {champion_model}")
leaderboard_sdf.show(truncate=False)
print("Backtest notebook complete.")
