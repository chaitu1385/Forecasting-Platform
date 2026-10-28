#!/usr/bin/env python3
"""
Run production forecast pipeline from the command line.

Usage:
    python scripts/run_forecast.py --config configs/platform_config.yaml --data data/actuals.parquet --champion lgbm_direct
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import load_config, load_config_with_overrides
from src.config.schema import OutputConfig
from src.pipeline.forecast import ForecastPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run forecast pipeline")
    parser.add_argument("--config", required=True, help="Platform config YAML")
    parser.add_argument("--lob-override", default=None, help="LOB override YAML")
    parser.add_argument("--data", required=True, help="Actuals (Parquet/CSV)")
    parser.add_argument("--champion", default=None, help="Override the first configured model")
    parser.add_argument("--product-master", default=None)
    parser.add_argument("--mapping-table", default=None)
    parser.add_argument("--runs-dir", default=None, help="Explicit shared immutable run authority")
    args = parser.parse_args()

    if args.lob_override:
        config = load_config_with_overrides(args.config, args.lob_override)
    else:
        config = load_config(args.config)
    defaults = OutputConfig()
    if args.runs_dir is not None:
        config.output.runs_path = args.runs_dir
    elif config.output.runs_path is None and (
        Path(config.output.forecast_path) == Path(defaults.forecast_path)
        and Path(config.output.metrics_path) == Path(defaults.metrics_path)
    ):
        config.output.runs_path = "data/runs"

    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        actuals = pl.read_parquet(str(data_path))
    else:
        actuals = pl.read_csv(str(data_path), try_parse_dates=True)

    product_master = None
    if args.product_master:
        product_master = pl.read_csv(args.product_master)

    mapping_table = None
    if args.mapping_table:
        mapping_table = pl.read_csv(args.mapping_table)

    pipeline = ForecastPipeline(config, source_kind="cli")
    forecast = pipeline.run(
        actuals=actuals,
        champion_model=args.champion if args.champion is not None else config.forecast.forecasters[0],
        product_master=product_master,
        mapping_table=mapping_table,
    )

    print(f"\nForecast generated: {len(forecast)} rows")
    print(f"Committed run_id: {pipeline.last_run.run_id}")
    print(forecast.head(20))


if __name__ == "__main__":
    main()
