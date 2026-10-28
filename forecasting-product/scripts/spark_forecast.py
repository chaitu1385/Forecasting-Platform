"""
spark_forecast.py — CLI entry point for distributed production forecast on Spark.

Reads actuals from the Lakehouse (or local CSV), fits the champion model on
all series in parallel, and writes forecasts back to the Lakehouse.

Usage (local / dev)
-------------------
python scripts/spark_forecast.py \
    --config       configs/platform_config.yaml \
    --data-dir     data/ \
    --lob          rossmann \
    --model        naive_seasonal

Usage (Fabric / spark-submit)
-----------------------------
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    scripts/spark_forecast.py \
    --config       configs/platform_config.yaml \
    --lob          surface \
    --workspace    my-fabric-ws \
    --lakehouse    my-lakehouse \
    --write-lakehouse \
    --model        lgbm_direct
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("spark_forecast")


def parse_args():
    p = argparse.ArgumentParser(description="Distributed production forecast on Spark")
    p.add_argument("--config",    default="configs/platform_config.yaml",
                   help="Path to platform_config.yaml")
    p.add_argument("--lob",       default="rossmann",
                   help="Line-of-business identifier")
    p.add_argument("--data-dir",  default="data/",
                   help="Local data directory (dev mode)")
    p.add_argument("--workspace", default="",
                   help="Fabric workspace name")
    p.add_argument("--lakehouse", default="",
                   help="Fabric lakehouse name")
    p.add_argument("--model",     default="",
                   help="Champion model name (empty = read from leaderboard table)")
    p.add_argument("--horizon",   type=int, default=0,
                   help="Forecast horizon in weeks (0 = from config)")
    p.add_argument("--write-lakehouse", action="store_true",
                   help="Write forecasts to Fabric Lakehouse")
    p.add_argument("--write-mode", default="upsert",
                   choices=["upsert", "overwrite_partition", "append"],
                   help="Write strategy for the forecasts Delta table")
    p.add_argument("--fabric-config", default="configs/fabric_config.yaml",
                   help="Path to fabric_config.yaml (drives series_builder mapping)")
    p.add_argument("--output-dir", default="data/forecasts/",
                   help="Local output directory (non-Fabric mode)")
    return p.parse_args()


def _resolve_champion(args, spark, lh):
    """
    Return the champion model name as a Python string.

    Aggregation and ranking stay on Spark executors; only the single
    winning model name (one string) is collected to the driver via
    ``.first()``.  Raises clearly if no backtest results exist yet.
    """
    from pyspark.sql import functions as F

    if args.model:
        logger.info("Using CLI-supplied champion model: %s", args.model)
        return args.model

    row = (
        lh.read_table("leaderboard")
        .filter(F.col("lob") == args.lob)
        .orderBy(F.col("run_date").desc(), F.col("rank").asc())
        .select("champion_model")
        .limit(1)
        .first()
    )
    if row is None:
        raise RuntimeError(
            f"No champion model found for lob='{args.lob}' in leaderboard table.  "
            "Run spark_backtest.py first, or supply --model."
        )
    return row[0]


def main():
    args = parse_args()

    # ── 1. Spark session ──────────────────────────────────────────────────────
    from src.spark.session import get_or_create_spark
    spark = get_or_create_spark(app_name=f"ForecastPipeline-{args.lob}")
    logger.info("SparkSession ready (version=%s)", spark.version)

    # ── 2. Platform config + fabric config ───────────────────────────────────
    import yaml
    from src.config.loader import load_config
    from src.spark.series_builder import SparkSeriesBuilder

    config = load_config(args.config)
    config.lob = args.lob
    horizon = args.horizon or config.forecast.horizon_weeks
    logger.info("Horizon: %d weeks", horizon)

    with open(args.fabric_config) as _f:
        fabric_yaml = yaml.safe_load(_f)

    # ── 3. Fabric / Lakehouse client (optional) ───────────────────────────────
    import os
    lh = None
    if args.write_lakehouse:
        from src.fabric.config import FabricConfig
        from src.fabric.lakehouse import FabricLakehouse
        ws = args.workspace or os.environ.get("FABRIC_WORKSPACE", "")
        lh_name = args.lakehouse or os.environ.get("FABRIC_LAKEHOUSE", "")
        fabric_cfg = FabricConfig(workspace=ws, lakehouse=lh_name)
        lh = FabricLakehouse(spark, fabric_cfg)

    # ── 4. Resolve champion model ─────────────────────────────────────────────
    champion_model = _resolve_champion(args, spark, lh) if lh else (args.model or config.forecast.forecasters[0])
    logger.info("Champion model: %s", champion_model)

    # ── 5. Load actuals ───────────────────────────────────────────────────────
    from pyspark.sql import functions as F

    if lh is not None:
        actuals_raw = lh.read_table("actuals")
    else:
        from src.spark.loader import SparkDataLoader
        loader = SparkDataLoader(spark, args.data_dir)
        train_sdf, _, store_sdf = loader.read_rossmann_all()
        actuals_raw = train_sdf.join(store_sdf, on="Store", how="left")

    # Build canonical series panel via config — no hard-coded column names
    builder = SparkSeriesBuilder.from_config(fabric_yaml["series_builder"])
    actuals_sdf = builder.build(actuals_raw)
    logger.info(
        "Actuals: %d rows, %d series",
        actuals_sdf.count(),
        actuals_sdf.select("series_id").distinct().count(),
    )

    # ── 6. Run distributed forecast ───────────────────────────────────────────
    from src.spark.pipeline import SparkForecastPipeline
    pipeline = SparkForecastPipeline(spark, config)

    forecasts_sdf = pipeline.run_forecast(
        actuals_sdf=actuals_sdf,
        champion_model=champion_model,
        horizon=horizon,
    )
    forecasts_sdf.cache()
    logger.info("Forecast rows: %d", forecasts_sdf.count())

    # ── 7. Write forecasts ────────────────────────────────────────────────────
    forecast_origin = date.today().isoformat()

    if lh is not None:
        from src.fabric.delta_writer import DeltaWriter
        writer = DeltaWriter(spark, fabric_cfg)
        writer.write_forecasts(
            df=forecasts_sdf,
            lob=args.lob,
            forecast_origin=forecast_origin,
            mode=args.write_mode,
        )
        # Optimize for BI queries
        lh.optimize("forecasts", z_order_by=["series_id", "week"])
        logger.info("Forecasts written to Lakehouse and optimized.")
    else:
        import polars as pl

        from src.pipeline.run_store import atomic_frame_append, checked_child, validate_lob

        validate_lob(args.lob)
        output_path = checked_child(Path(args.output_dir), args.lob)
        filename = atomic_frame_append(
            pl.from_pandas(forecasts_sdf.toPandas()), output_path,
            prefix=f"forecast_{args.lob}_{forecast_origin}",
        )
        logger.info("Legacy Spark adapter forecast saved: run_id=%s file=%s; no local run-bundle guarantee",
                    pipeline.run_id, filename.name)

    logger.info("spark_forecast.py complete.")
    spark.stop()


if __name__ == "__main__":
    main()
