#!/usr/bin/env python3
"""
Run backtesting pipeline from the command line.

Usage:
    python scripts/run_backtest.py --config configs/platform_config.yaml --data data/actuals.parquet
    python scripts/run_backtest.py --config configs/platform_config.yaml --lob-override configs/lob/surface.yaml --data data/actuals.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import load_config, load_config_with_overrides
from src.config.schema import OutputConfig
from src.pipeline.backtest import BacktestPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtesting pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to base platform config YAML",
    )
    parser.add_argument(
        "--lob-override",
        default=None,
        help="Path to LOB-specific config override YAML",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to actuals data (Parquet or CSV)",
    )
    parser.add_argument(
        "--product-master",
        default=None,
        help="Path to product master (for transitions)",
    )
    parser.add_argument(
        "--mapping-table",
        default=None,
        help="Path to SKU mapping table (for transitions)",
    )
    parser.add_argument("--runs-dir", default=None, help="Explicit shared immutable run authority")
    args = parser.parse_args()

    # Load config
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

    logger.info("Config loaded: LOB=%s", config.lob)

    # Load data
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        actuals = pl.read_parquet(str(data_path))
    else:
        actuals = pl.read_csv(str(data_path), try_parse_dates=True)

    logger.info("Actuals loaded: %d rows", len(actuals))

    # Optional: product master and mapping table
    product_master = None
    if args.product_master:
        product_master = pl.read_csv(args.product_master)

    mapping_table = None
    if args.mapping_table:
        mapping_table = pl.read_csv(args.mapping_table)

    # Run pipeline
    pipeline = BacktestPipeline(config, source_kind="cli")
    results = pipeline.run(
        actuals=actuals,
        product_master=product_master,
        mapping_table=mapping_table,
    )

    # Print summary
    print(f"Committed run_id: {results['run_id']}")
    if not results["champions"].is_empty():
        print("\n" + "=" * 60)
        print("CHAMPION MODEL(S)")
        print("=" * 60)
        print(results["champions"])

    if not results["leaderboard"].is_empty():
        print("\n" + "=" * 60)
        print("MODEL LEADERBOARD")
        print("=" * 60)
        print(results["leaderboard"])


if __name__ == "__main__":
    main()
