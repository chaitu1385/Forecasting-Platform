"""
Product transition engine.

Handles three scenarios based on launch_date relative to forecast_origin:

  Scenario A — Already launched (launch_date ≤ forecast_origin):
      Stitch old SKU history onto new SKU as one continuous series.

  Scenario B — Launches within horizon (0 < gap ≤ transition_window):
      Ramp-down old SKU, ramp-up new SKU over the transition window.

  Scenario C — Launches beyond horizon:
      Forecast old SKU only.  New SKU flagged as "pending transition".

The engine reads mappings from the SKU mapping module and applies
planner overrides when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import polars as pl

from ..config.schema import TransitionConfig
from ..utils.calendar import advance_date

VALID_RAMP_SHAPES = frozenset({"linear", "scurve", "step"})


class TransitionScenario(str, Enum):
    """Which transition scenario applies to a given (old, new) SKU pair."""
    A_LAUNCHED = "A"          # new SKU already launched
    B_IN_HORIZON = "B"        # new SKU launches within forecast horizon
    C_BEYOND_HORIZON = "C"    # new SKU launches after horizon end
    MANUAL = "manual"         # planner override


@dataclass
class TransitionPlan:
    """A concrete plan for how to transition from old → new SKU."""
    old_sku: str
    new_sku: str
    scenario: TransitionScenario
    proportion: float                  # demand proportion allocated to new SKU
    ramp_start: Optional[date] = None  # when ramp-down begins
    ramp_end: Optional[date] = None    # when ramp-up completes
    ramp_shape: str = "linear"
    notes: str = ""


class TransitionEngine:
    """
    Determines transition scenario and builds stitched time series.

    Usage
    -----
    >>> engine = TransitionEngine(config)
    >>> plans = engine.compute_plans(mapping_table, product_master, forecast_origin)
    >>> stitched = engine.stitch_series(actuals, plans)
    """

    def __init__(self, config: TransitionConfig, frequency: str = "W"):
        self.window_periods = config.transition_window_periods
        self.frequency = frequency
        self.ramp_shape = config.ramp_shape
        if self.ramp_shape not in VALID_RAMP_SHAPES:
            raise ValueError(
                f"Unknown ramp_shape {self.ramp_shape!r}. "
                f"Supported: {sorted(VALID_RAMP_SHAPES)}"
            )

    def compute_plans(
        self,
        mapping_table: pl.DataFrame,
        product_master: pl.DataFrame,
        forecast_origin: date,
        horizon_weeks: int = 39,
        overrides: Optional[pl.DataFrame] = None,
    ) -> List[TransitionPlan]:
        """
        For each (old_sku, new_sku) mapping, determine the transition
        scenario and build a plan.

        Parameters
        ----------
        mapping_table:
            Output from SKU mapping pipeline.  Must have columns:
            old_sku, new_sku, proportion.
        product_master:
            Must have: sku_id, launch_date.
        forecast_origin:
            The date from which the forecast starts.
        horizon_weeks:
            Number of weeks in the forecast horizon.
        overrides:
            Planner overrides with columns: old_sku, new_sku, scenario,
            proportion, ramp_shape.  Overrides win unconditionally.
        """
        # Build launch-date lookup
        launch_dates: Dict[str, date] = {}
        for row in product_master.iter_rows(named=True):
            ld = row.get("launch_date")
            if ld is not None:
                launch_dates[row["sku_id"]] = ld

        # Build override lookup
        override_map: Dict[Tuple[str, str], dict] = {}
        if overrides is not None and not overrides.is_empty():
            for row in overrides.iter_rows(named=True):
                key = (row["old_sku"], row["new_sku"])
                override_map[key] = row

        horizon_end = advance_date(forecast_origin, self.frequency, horizon_weeks)
        plans: List[TransitionPlan] = []

        for row in mapping_table.iter_rows(named=True):
            old_sku = row["old_sku"]
            new_sku = row["new_sku"]
            proportion = row.get("proportion", 1.0)

            # Check for override
            override = override_map.get((old_sku, new_sku))
            if override:
                plans.append(TransitionPlan(
                    old_sku=old_sku,
                    new_sku=new_sku,
                    scenario=TransitionScenario.MANUAL,
                    proportion=override.get("proportion", proportion),
                    ramp_shape=override.get("ramp_shape", self.ramp_shape),
                    notes=f"Planner override: {override.get('notes', '')}",
                ))
                continue

            new_launch = launch_dates.get(new_sku)

            if new_launch is None or new_launch <= forecast_origin:
                # Scenario A — already launched
                plans.append(TransitionPlan(
                    old_sku=old_sku,
                    new_sku=new_sku,
                    scenario=TransitionScenario.A_LAUNCHED,
                    proportion=proportion,
                    notes="New SKU already launched. Pure history stitch.",
                ))
            elif new_launch <= horizon_end:
                # Scenario B — launches within horizon
                ramp_start = advance_date(
                    new_launch, self.frequency, -(self.window_periods // 2)
                )
                ramp_end = advance_date(
                    new_launch, self.frequency, self.window_periods // 2
                )
                plans.append(TransitionPlan(
                    old_sku=old_sku,
                    new_sku=new_sku,
                    scenario=TransitionScenario.B_IN_HORIZON,
                    proportion=proportion,
                    ramp_start=ramp_start,
                    ramp_end=ramp_end,
                    ramp_shape=self.ramp_shape,
                    notes=(
                        f"Transition window: {ramp_start} → {ramp_end}. "
                        f"Ramp shape: {self.ramp_shape}."
                    ),
                ))
            else:
                # Scenario C — beyond horizon
                plans.append(TransitionPlan(
                    old_sku=old_sku,
                    new_sku=new_sku,
                    scenario=TransitionScenario.C_BEYOND_HORIZON,
                    proportion=proportion,
                    notes="New SKU launches beyond forecast horizon. Old SKU forecast only.",
                ))

        return plans

    def stitch_series(
        self,
        actuals: pl.DataFrame,
        plans: List[TransitionPlan],
        time_column: str = "week",
        series_id_column: str = "series_id",
        value_column: str = "quantity",
    ) -> pl.DataFrame:
        """
        Apply transition plans to create stitched time series.

        For Scenario A: concatenate old SKU history before new SKU history.
        For Scenario B: apply ramp weights during the transition window.
        For Scenario C: keep old SKU series as-is.
        """
        results: List[pl.DataFrame] = []

        # SKUs already handled (stitched into another series)
        stitched_old_skus = set()
        stitched_new_skus = set()

        for plan in plans:
            if plan.scenario == TransitionScenario.C_BEYOND_HORIZON:
                # No stitching — old SKU continues
                continue

            old_data = actuals.filter(pl.col(series_id_column) == plan.old_sku)
            new_data = actuals.filter(pl.col(series_id_column) == plan.new_sku)

            if plan.scenario in (
                TransitionScenario.A_LAUNCHED,
                TransitionScenario.MANUAL,
            ):
                # Pure stitch: rename old SKU data to new SKU ID,
                # concatenate with new SKU actuals
                old_renamed = old_data.with_columns(
                    pl.lit(plan.new_sku).alias(series_id_column)
                )
                if not new_data.is_empty():
                    # Only use old data for dates before new SKU has actuals
                    new_min_date = new_data[time_column].min()
                    old_renamed = old_renamed.filter(
                        pl.col(time_column) < new_min_date
                    )

                stitched = pl.concat(
                    [old_renamed, new_data],
                    how="vertical_relaxed",
                ).sort(time_column)
                results.append(stitched)
                stitched_old_skus.add(plan.old_sku)
                stitched_new_skus.add(plan.new_sku)

            elif plan.scenario == TransitionScenario.B_IN_HORIZON:
                # Ramp: apply weights during transition window
                old_weighted = self._apply_ramp_weights(
                    old_data, plan, time_column, value_column,
                    direction="down",
                )
                new_weighted = self._apply_ramp_weights(
                    new_data, plan, time_column, value_column,
                    direction="up",
                )
                # Combine under new SKU ID
                old_weighted = old_weighted.with_columns(
                    pl.lit(plan.new_sku).alias(series_id_column)
                )
                stitched = pl.concat(
                    [old_weighted, new_weighted],
                    how="vertical_relaxed",
                ).sort(time_column)
                results.append(stitched)
                stitched_old_skus.add(plan.old_sku)
                stitched_new_skus.add(plan.new_sku)

        # Include unaffected series (not part of any transition)
        unaffected = actuals.filter(
            ~pl.col(series_id_column).is_in(
                list(stitched_old_skus | stitched_new_skus)
            )
        )
        results.append(unaffected)

        if results:
            return pl.concat(results, how="vertical_relaxed").sort(
                [series_id_column, time_column]
            )
        return actuals

    def _apply_ramp_weights(
        self,
        df: pl.DataFrame,
        plan: TransitionPlan,
        time_column: str,
        value_column: str,
        direction: str,  # "up" or "down"
    ) -> pl.DataFrame:
        """Apply ramp-up or ramp-down weights to a series."""
        if plan.ramp_start is None or plan.ramp_end is None:
            return df

        ramp_start = plan.ramp_start
        ramp_end = plan.ramp_end
        total_days = (ramp_end - ramp_start).days
        if total_days <= 0:
            return df

        def weight_expr() -> pl.Expr:
            days_in = (pl.col(time_column).cast(pl.Date) - pl.lit(ramp_start)).dt.total_days()
            progress = (days_in / total_days).clip(0.0, 1.0)

            if plan.ramp_shape == "scurve":
                # S-curve: smooth sigmoid-like transition
                raw = progress
                # Approximation: 3t² - 2t³ (Hermite smoothstep)
                w = 3.0 * raw ** 2 - 2.0 * raw ** 3
            elif plan.ramp_shape == "step":
                w = (progress >= 0.5).cast(pl.Float64)
            else:
                # Linear (default)
                w = progress

            if direction == "down":
                w = 1.0 - w
            return w

        before_ramp = df.filter(pl.col(time_column).cast(pl.Date) < ramp_start)
        during_ramp = df.filter(
            (pl.col(time_column).cast(pl.Date) >= ramp_start)
            & (pl.col(time_column).cast(pl.Date) <= ramp_end)
        )
        after_ramp = df.filter(pl.col(time_column).cast(pl.Date) > ramp_end)

        if not during_ramp.is_empty():
            during_ramp = during_ramp.with_columns(
                (pl.col(value_column) * weight_expr()).alias(value_column)
            )

        # For ramp-down: after ramp the old SKU is zero
        # For ramp-up: before ramp the new SKU is zero
        if direction == "down":
            after_ramp = after_ramp.with_columns(
                pl.lit(0.0).alias(value_column)
            )
        else:
            before_ramp = before_ramp.with_columns(
                pl.lit(0.0).alias(value_column)
            )

        parts = [p for p in [before_ramp, during_ramp, after_ramp] if not p.is_empty()]
        if parts:
            return pl.concat(parts, how="vertical_relaxed")
        return df
