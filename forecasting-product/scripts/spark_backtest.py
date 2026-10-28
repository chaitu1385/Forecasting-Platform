"""
spark_backtest.py — CLI entry point for distributed backtest on Spark.

Submits a Spark job that runs walk-forward cross-validation across all
configured models and series, then writes results to the Lakehouse.

Usage (local / dev)
-------------------
python scripts/spark_backtest.py \
    --config   configs/platform_config.yaml \
    --data-dir data/ \
    --lob      rossmann \
    --models   naive_seasonal lgbm_direct xgboost_direct

Usage (Fabric / spark-submit)
-----------------------------
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    scripts/spark_backtest.py \
    --config   configs/platform_config.yaml \
    --lob      surface \
    --workspace my-fabric-ws \
    --lakehouse my-lakehouse \
    --models   naive_seasonal lgbm_direct xgboost_direct
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# ── ensure platform src is importable ────────────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("spark_backtest")


def parse_args():
    p = argparse.ArgumentParser(description="Distributed backtest on Spark")
    p.add_argument("--config",    default="configs/platform_config.yaml",
                   help="Path to platform_config.yaml")
    p.add_argument("--lob",       default="rossmann",
                   help="Line-of-business identifier")
    p.add_argument("--data-dir",  default="data/",
                   help="Local data directory (CSV files, dev mode)")
    p.add_argument("--workspace", default="",
                   help="Fabric workspace name (overrides env FABRIC_WORKSPACE)")
    p.add_argument("--lakehouse", default="",
                   help="Fabric lakehouse name (overrides env FABRIC_LAKEHOUSE)")
    p.add_argument("--models",    nargs="*",
                   help="Model names to backtest (default: from config)")
    p.add_argument("--write-lakehouse", action="store_true",
                   help="Write results to Fabric Lakehouse (requires --workspace/--lakehouse)")
    p.add_argument("--fabric-config", default="configs/fabric_config.yaml",
                   help="Path to fabric_config.yaml (drives series_builder mapping)")
    p.add_argument("--output-dir", default="data/backtest_results/",
                   help="Local output directory for backtest results (non-Fabric mode)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Spark session ──────────────────────────────────────────────────────
    from src.spark.session import get_or_create_spark
    spark = get_or_create_spark(app_name=f"BacktestPipeline-{args.lob}")
    logger.info("SparkSession ready (version=%s)", spark.version)

    # ── 2. Platform config + fabric config ───────────────────────────────────
    import yaml
    from src.config.loader import load_config
    from src.spark.series_builder import SparkSeriesBuilder

    config = load_config(args.config)
    config.lob = args.lob
    model_names = args.models or config.forecast.forecasters
    logger.info("Models: %s | Folds: %d", model_names, config.backtest.n_folds)

    with open(args.fabric_config) as _f:
        fabric_yaml = yaml.safe_load(_f)

    # ── 3. Load actuals ───────────────────────────────────────────────────────
    if args.write_lakehouse and (args.workspace or args.lakehouse):
        import os
        ws = args.workspace or os.environ.get("FABRIC_WORKSPACE", "")
        lh_name = args.lakehouse or os.environ.get("FABRIC_LAKEHOUSE", "")
        from src.spark.utils import abfss_uri
        from src.spark.loader import SparkDataLoader
        base_path = abfss_uri(ws, lh_name)
        loader = SparkDataLoader(spark, base_path)
        actuals_raw = loader.read_actuals(format="delta")
    else:
        from src.spark.loader import SparkDataLoader
        loader = SparkDataLoader(spark, args.data_dir)
        train_sdf, _, store_sdf = loader.read_rossmann_all()
        actuals_raw = train_sdf.join(store_sdf, on="Store", how="left")

    from pyspark.sql import functions as F

    # Build canonical series panel via config — no hard-coded column names
    builder = SparkSeriesBuilder.from_config(fabric_yaml["series_builder"])
    actuals_sdf = builder.build(actuals_raw)
    logger.info(
        "Series rows: %d | Series: %d",
        actuals_sdf.count(),
        actuals_sdf.select("series_id").distinct().count(),
    )

    # ── 5. Run backtest ───────────────────────────────────────────────────────
    from src.spark.pipeline import (
        SparkForecastPipeline,
        backtest_metrics_for_legacy_storage,
        leaderboard_for_legacy_storage,
    )
    pipeline = SparkForecastPipeline(spark, config)

    backtest_results_sdf = pipeline.run_backtest(actuals_sdf, model_names=model_names)
    backtest_results_sdf.cache()
    logger.info("Backtest result rows: %d", backtest_results_sdf.count())

    # ── 6. Champion selection ─────────────────────────────────────────────────
    leaderboard_sdf = pipeline.select_champion(
        backtest_results_sdf,
        primary_metric=config.backtest.primary_metric,
        secondary_metric=config.backtest.secondary_metric,
    )
    leaderboard_sdf.show(truncate=False)

    champion = leaderboard_sdf.filter(F.col("rank") == 1).select("model").collect()[0][0]
    logger.info("Champion model: %s", champion)

    # ── 7. Write results ──────────────────────────────────────────────────────
    run_date = date.today().isoformat()

    if args.write_lakehouse:
        from src.fabric.config import FabricConfig
        from src.fabric.delta_writer import DeltaWriter
        import os
        ws = args.workspace or os.environ.get("FABRIC_WORKSPACE", "")
        lh_name = args.lakehouse or os.environ.get("FABRIC_LAKEHOUSE", "")
        fabric_cfg = FabricConfig(workspace=ws, lakehouse=lh_name)
        writer = DeltaWriter(spark, fabric_cfg)

        out_sdf = (
            backtest_metrics_for_legacy_storage(backtest_results_sdf, config.forecast.series_id_column)
            .withColumn("lob", F.lit(args.lob))
            .withColumn("run_date", F.lit(run_date))
        )
        writer.append(out_sdf, "backtest_results", partition_by=["lob", "run_date"])

        lb_out = (
            leaderboard_for_legacy_storage(leaderboard_sdf)
            .withColumn("lob", F.lit(args.lob))
            .withColumn("run_date", F.lit(run_date))
            .withColumn("champion_model", F.lit(champion))
        )
        writer.upsert(lb_out, "leaderboard", merge_keys=["lob", "run_date", "model"])
        logger.info("Results written to Lakehouse.")
    else:
        output_path = Path(args.output_dir) / args.lob / run_date
        output_path.mkdir(parents=True, exist_ok=True)
        backtest_results_sdf.toPandas().to_parquet(output_path / "backtest_results.parquet", index=False)
        leaderboard_sdf.toPandas().to_parquet(output_path / "leaderboard.parquet", index=False)
        logger.info("Results written locally to %s", output_path)

    logger.info("spark_backtest.py complete. Champion: %s", champion)
    spark.stop()


if __name__ == "__main__":
    main()
