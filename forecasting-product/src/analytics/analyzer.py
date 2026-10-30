"""
DataAnalyzer — automated data profiling and configuration recommendation.

Orchestrates schema detection, hierarchy detection, forecastability assessment,
data quality profiling, hypothesis generation, and PlatformConfig recommendation.
Designed to work standalone (pure statistics) with optional LLM enhancement.

Usage
-----
>>> from src.analytics.analyzer import DataAnalyzer
>>> analyzer = DataAnalyzer(lob_name="rossmann")
>>> report = analyzer.analyze(df)
>>> print(report.recommended_config)
>>> print(report.hypotheses)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

import polars as pl

from ..config.schema import (
    BacktestConfig,
    CalibrationConfig,
    CleansingConfig,
    DataQualityConfig,
    DataQualityReportConfig,
    ForecastConfig,
    HierarchyConfig,
    OutputConfig,
    PlatformConfig,
    ReconciliationConfig,
    StructuralBreakConfig,
)
from .forecastability import ForecastabilityAnalyzer, ForecastabilityReport

logger = logging.getLogger(__name__)


class MissingNumericTargetError(ValueError):
    """The analyzed input has no numeric demand column distinct from time."""


class MissingTimeColumnError(ValueError):
    """The analyzed input has no recognized time/date column."""


# --------------------------------------------------------------------------- #
#  Result dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class SchemaDetection:
    """Result of automatic column role detection."""

    time_column: str
    target_column: str
    id_columns: List[str]
    dimension_columns: List[str]
    numeric_columns: List[str]
    n_rows: int
    n_series: int
    date_range: Tuple[str, str]
    frequency_guess: str
    confidence: float


@dataclass
class HierarchyDetection:
    """Result of hierarchy structure detection."""

    hierarchies: List[HierarchyConfig] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis output from DataAnalyzer."""

    schema: SchemaDetection
    hierarchy: HierarchyDetection
    forecastability: ForecastabilityReport
    recommended_config: PlatformConfig
    config_reasoning: List[str]
    hypotheses: List[str]
    warnings: List[str] = field(default_factory=list)
    regressor_columns: List[str] = field(default_factory=list)
    dimension_sources: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Heuristic constants
# --------------------------------------------------------------------------- #

_TIME_COLUMN_PATTERNS = {"week", "date", "ds", "time", "timestamp", "period", "day"}
_TARGET_COLUMN_PATTERNS = {"quantity", "sales", "demand", "revenue", "volume", "units",
                           "target", "value", "amount", "qty", "count"}
_HIERARCHY_NAME_HINTS = {
    "region": "geography",
    "country": "geography",
    "state": "geography",
    "city": "geography",
    "store": "geography",
    "location": "geography",
    "category": "product",
    "subcategory": "product",
    "product": "product",
    "brand": "product",
    "sku": "product",
    "item": "product",
    "family": "product",
    "department": "product",
    "channel": "channel",
    "segment": "channel",
}


# --------------------------------------------------------------------------- #
#  DataAnalyzer
# --------------------------------------------------------------------------- #

class DataAnalyzer:
    """Automated data analysis and PlatformConfig recommendation.

    Parameters
    ----------
    lob_name : str
        Name to assign to the generated config.
    season_length : int
        Assumed seasonal period for forecastability signals.
    """

    def __init__(self, lob_name: str = "analyzed", season_length: int = 52):
        self.lob_name = lob_name
        self.season_length = season_length

    def analyze(self, df: pl.DataFrame) -> AnalysisReport:
        """Run the full analysis pipeline.

        Parameters
        ----------
        df : pl.DataFrame
            Raw input data.  Must contain at least a time column, a numeric
            target column, and one or more identifier columns.

        Returns
        -------
        AnalysisReport
        """
        warnings: List[str] = []

        # 1. Schema detection
        schema = self.detect_schema(df)

        # 2. Hierarchy detection
        hierarchy = self.detect_hierarchy(df, schema)

        # 3. Forecastability assessment
        # Use detected frequency to set the correct season_length
        try:
            from ..config.schema import get_frequency_profile
            _detected_sl = get_frequency_profile(schema.frequency_guess)["season_length"]
        except (ValueError, KeyError):
            _detected_sl = self.season_length
        fa = ForecastabilityAnalyzer(season_length=_detected_sl)
        # Build series_id from id_columns if needed
        analysis_df, sid_col = self._prepare_analysis_df(df, schema)
        forecastability = fa.analyze(analysis_df, schema.target_column,
                                     schema.time_column, sid_col)
        warnings.extend(forecastability.warnings)

        # 4. Hypotheses
        hypotheses = self.generate_hypotheses(schema, forecastability)

        # 5. Config recommendation
        config, config_reasoning = self.recommend_config(
            schema, hierarchy, forecastability
        )

        return AnalysisReport(
            schema=schema,
            hierarchy=hierarchy,
            forecastability=forecastability,
            recommended_config=config,
            config_reasoning=config_reasoning,
            hypotheses=hypotheses,
            warnings=warnings,
        )

    # ----- Schema Detection ------------------------------------------------ #

    def detect_schema(self, df: pl.DataFrame) -> SchemaDetection:
        """Auto-detect column roles from data types and naming conventions."""
        cols = df.columns
        dtypes = {c: df[c].dtype for c in cols}

        # Time column: prefer date/datetime types, then string columns matching patterns
        time_col = self._find_time_column(df, cols, dtypes)

        # Target column: numeric, highest non-null count, prefer named matches
        target_col = self._find_target_column(df, cols, dtypes, time_col)

        # Dimension columns: string/categorical with reasonable cardinality
        dimension_cols = []
        for c in cols:
            if c in (time_col, target_col):
                continue
            if dtypes[c] in (pl.Utf8, pl.Categorical, pl.String):
                n_unique = df[c].n_unique()
                n_rows = len(df)
                # Not too many unique values (would be an ID, not a dimension)
                # and not just one value (constant, useless)
                if 1 < n_unique <= max(1000, n_rows // 10):
                    dimension_cols.append(c)

        # ID columns: dimensions that together form the series key
        id_columns = self._infer_id_columns(df, dimension_cols, time_col)

        # Numeric columns (potential regressors)
        numeric_cols = [
            c for c in cols
            if c not in (time_col, target_col) and c not in dimension_cols
            and dtypes[c] in (pl.Float32, pl.Float64, pl.Int8, pl.Int16,
                              pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                              pl.UInt32, pl.UInt64)
        ]

        # Frequency guess
        freq, n_series = self._guess_frequency(df, time_col, id_columns)

        # Date range
        date_min = str(df[time_col].min()) if len(df) > 0 else ""
        date_max = str(df[time_col].max()) if len(df) > 0 else ""

        # Confidence: high if we found clear matches
        confidence = 1.0
        if time_col == cols[0]:
            confidence -= 0.2  # fell back to first column
        if target_col == cols[-1]:
            confidence -= 0.1

        return SchemaDetection(
            time_column=time_col,
            target_column=target_col,
            id_columns=id_columns,
            dimension_columns=dimension_cols,
            numeric_columns=numeric_cols,
            n_rows=len(df),
            n_series=n_series,
            date_range=(date_min, date_max),
            frequency_guess=freq,
            confidence=max(0.0, confidence),
        )

    def _find_time_column(self, df: pl.DataFrame, cols: list, dtypes: dict) -> str:
        """Find the time/date column."""
        # First pass: actual date/datetime types
        for c in cols:
            if dtypes[c] in (pl.Date, pl.Datetime):
                return c
        # Second pass: name matching
        for c in cols:
            if c.lower() in _TIME_COLUMN_PATTERNS:
                return c
        # Third pass: try parsing string columns
        for c in cols:
            if dtypes[c] in (pl.Utf8, pl.String):
                try:
                    parsed = df[c].str.to_date(strict=False)
                    if parsed.null_count() < len(df) * 0.5:
                        return c
                except pl.exceptions.ComputeError:
                    continue
        raise MissingTimeColumnError(
            "Could not detect a time/date column. Include dates in a supported format."
        )

    def _find_target_column(self, df: pl.DataFrame, cols: list, dtypes: dict, time_col: str) -> str:
        """Find the numeric target column."""
        numeric_cols = [c for c in cols if c != time_col and dtypes[c].is_numeric()]

        if not numeric_cols:
            raise MissingNumericTargetError(
                "Could not detect a numeric target column. Include a numeric demand column."
            )

        # Prefer name matches
        for c in numeric_cols:
            if c.lower() in _TARGET_COLUMN_PATTERNS:
                return c

        # Pick the one with highest non-null count (most populated)
        best = max(numeric_cols, key=lambda c: len(df) - df[c].null_count())
        return best

    def _infer_id_columns(self, df, dimension_cols, time_col) -> List[str]:
        """Infer which dimension columns together form the series key.

        The series key is the minimal set of columns where
        (key, time) → unique rows.
        """
        if not dimension_cols:
            return []

        n_rows = len(df)

        # Try single columns first
        for c in dimension_cols:
            n_combos = df.select(c, time_col).n_unique()
            if n_combos >= n_rows * 0.95:
                return [c]

        # Try pairs
        if len(dimension_cols) >= 2:
            best_pair = None
            best_coverage = 0
            for i, c1 in enumerate(dimension_cols):
                for c2 in dimension_cols[i + 1:]:
                    n_combos = df.select(c1, c2, time_col).n_unique()
                    if n_combos > best_coverage:
                        best_coverage = n_combos
                        best_pair = [c1, c2]
                    if n_combos >= n_rows * 0.95:
                        return [c1, c2]
            if best_pair:
                return best_pair

        # Fall back to all dimensions
        return dimension_cols

    def _guess_frequency(self, df, time_col, id_columns) -> Tuple[str, int]:
        """Guess the data frequency and count series."""
        # Count series
        if id_columns:
            n_series = df.select(id_columns).n_unique()
        else:
            n_series = 1

        # Compute median gap in days between consecutive timestamps
        if id_columns:
            sample_id = df[id_columns[0]].unique().to_list()[0]
            sample = df.filter(pl.col(id_columns[0]) == sample_id).sort(time_col)
        else:
            sample = df.sort(time_col)

        if sample.height < 2:
            return "W", n_series

        try:
            dates = sample[time_col].to_list()
            gaps = []
            for i in range(1, min(len(dates), 20)):
                d1, d0 = dates[i], dates[i - 1]
                if isinstance(d0, date) and isinstance(d1, date):
                    gaps.append((d1 - d0).days)
            if gaps:
                median_gap = sorted(gaps)[len(gaps) // 2]
                if median_gap <= 2:
                    return "D", n_series
                elif median_gap <= 10:
                    return "W", n_series
                elif median_gap <= 35:
                    return "M", n_series
                else:
                    return "Q", n_series
        except Exception:
            logger.debug("Frequency detection failed, defaulting to weekly", exc_info=True)

        return "W", n_series

    # ----- Hierarchy Detection --------------------------------------------- #

    def detect_hierarchy(
        self, df: pl.DataFrame, schema: SchemaDetection
    ) -> HierarchyDetection:
        """Detect parent-child relationships among dimension columns.

        For each pair (A, B), check if B is functionally dependent on A:
        every unique value of B maps to exactly one value of A.
        If so, A is a candidate parent of B.
        """
        dim_cols = schema.dimension_columns
        if len(dim_cols) < 2:
            if dim_cols:
                return HierarchyDetection(
                    hierarchies=[HierarchyConfig(
                        name=self._guess_hierarchy_name(dim_cols[0]),
                        levels=[dim_cols[0]],
                        id_column=dim_cols[0],
                    )],
                    reasoning=[f"Single dimension '{dim_cols[0]}' → flat hierarchy"],
                )
            return HierarchyDetection(warnings=["No dimension columns detected"])

        # Build parent → child edges
        edges: Dict[str, List[str]] = {c: [] for c in dim_cols}
        reasoning: List[str] = []

        for child in dim_cols:
            for parent in dim_cols:
                if parent == child:
                    continue
                # Check: does each value of child map to exactly one parent value?
                mapping = df.select(child, parent).unique()
                child_counts = mapping.group_by(child).agg(
                    pl.col(parent).n_unique().alias("n_parents")
                )
                if child_counts["n_parents"].max() == 1:
                    # child → parent is functional dependency
                    # So parent is higher in hierarchy (parent of child)
                    edges[parent].append(child)
                    reasoning.append(
                        f"'{child}' is functionally dependent on '{parent}' "
                        f"(each {child} maps to exactly one {parent})"
                    )

        # Build hierarchies from edges: find roots (columns that are children of no one)
        all_children = {c for children in edges.values() for c in children}
        roots = [c for c in dim_cols if c not in all_children]

        hierarchies: List[HierarchyConfig] = []
        used = set()

        for root in roots:
            # DFS to build level order
            levels = self._build_level_order(root, edges, used)
            if levels:
                hier_name = self._guess_hierarchy_name(root)
                hierarchies.append(HierarchyConfig(
                    name=hier_name,
                    levels=levels,
                    id_column=levels[-1],  # leaf level
                    fixed=df[root].n_unique() <= 3,
                ))

        # Remaining columns not in any hierarchy
        for c in dim_cols:
            if c not in used:
                hier_name = self._guess_hierarchy_name(c)
                hierarchies.append(HierarchyConfig(
                    name=hier_name,
                    levels=[c],
                    id_column=c,
                    fixed=df[c].n_unique() <= 3,
                ))
                used.add(c)

        warnings = []
        if not hierarchies:
            warnings.append("Could not detect hierarchy structure")

        return HierarchyDetection(
            hierarchies=hierarchies,
            reasoning=reasoning,
            warnings=warnings,
        )

    def _build_level_order(
        self, root: str, edges: Dict[str, List[str]], used: set
    ) -> List[str]:
        """DFS from root to build ordered list of levels."""
        if root in used:
            return []
        levels = [root]
        used.add(root)
        # Pick the child with highest cardinality at each step
        current = root
        while edges.get(current):
            children = [c for c in edges[current] if c not in used]
            if not children:
                break
            # Pick child that isn't already used
            current = children[0]
            levels.append(current)
            used.add(current)
        return levels

    @staticmethod
    def _guess_hierarchy_name(col_name: str) -> str:
        """Guess a hierarchy dimension name from a column name."""
        lower = col_name.lower().replace("_", "")
        for pattern, name in _HIERARCHY_NAME_HINTS.items():
            if pattern in lower:
                return name
        return col_name

    # ----- Prepare analysis DataFrame -------------------------------------- #

    def _prepare_analysis_df(
        self, df: pl.DataFrame, schema: SchemaDetection
    ) -> Tuple[pl.DataFrame, str]:
        """Build a DataFrame with a single series_id column for forecastability."""
        if not schema.id_columns:
            # Single series
            result = df.with_columns(pl.lit("all").alias("_series_id"))
            return result, "_series_id"

        if len(schema.id_columns) == 1:
            return df, schema.id_columns[0]

        # Concatenate multiple ID columns into composite key
        result = df.with_columns(
            pl.concat_str(schema.id_columns, separator="__").alias("_series_id")
        )
        return result, "_series_id"

    # ----- Hypothesis Generation ------------------------------------------- #

    def generate_hypotheses(
        self,
        schema: SchemaDetection,
        forecastability: ForecastabilityReport,
    ) -> List[str]:
        """Generate rule-based hypotheses from analysis results."""
        hyps: List[str] = []
        n = forecastability.n_series

        # Forecastability distribution
        high_pct = forecastability.score_distribution.get("high", 0) / max(n, 1) * 100
        low_pct = forecastability.score_distribution.get("low", 0) / max(n, 1) * 100

        hyps.append(
            f"Overall forecastability is {'high' if forecastability.overall_score >= 0.6 else 'medium' if forecastability.overall_score >= 0.3 else 'low'} "
            f"(score: {forecastability.overall_score:.2f}). "
            f"{high_pct:.0f}% of series are highly forecastable, {low_pct:.0f}% are difficult."
        )

        # Seasonality
        if forecastability.per_series is not None:
            ps = forecastability.per_series
            strong_seasonal = ps.filter(pl.col("seasonal_strength") > 0.5).height
            if strong_seasonal > 0:
                pct = strong_seasonal / n * 100
                hyps.append(
                    f"{pct:.0f}% of series ({strong_seasonal}/{n}) show strong seasonality "
                    f"(seasonal_strength > 0.5) — statistical models with seasonal components "
                    f"(AutoARIMA, AutoTheta, MSTL) should perform well."
                )

            # Trend
            trending = ps.filter(pl.col("trend_strength") > 0.5).height
            if trending > 0:
                pct = trending / n * 100
                hyps.append(
                    f"{pct:.0f}% of series ({trending}/{n}) exhibit detectable trend — "
                    f"trend-aware models recommended."
                )

        # Demand classification
        sparse_count = sum(
            forecastability.demand_class_distribution.get(c, 0)
            for c in ("intermittent", "lumpy")
        )
        if sparse_count > 0:
            pct = sparse_count / max(n, 1) * 100
            hyps.append(
                f"{pct:.0f}% of series ({sparse_count}/{n}) have sparse/intermittent demand — "
                f"consider intermittent-specific models (Croston SBA, TSB)."
            )

        # Data length
        try:
            d0 = date.fromisoformat(schema.date_range[0][:10])
            d1 = date.fromisoformat(schema.date_range[1][:10])
            weeks = (d1 - d0).days // 7
            if weeks < 52:
                hyps.append(
                    f"Only {weeks} weeks of history — insufficient for most ML models. "
                    f"Statistical models with few parameters are safest."
                )
            elif weeks < 104:
                hyps.append(
                    f"{weeks} weeks of history — sufficient for statistical models, "
                    f"marginal for ML (LightGBM, XGBoost). Neural models need more data."
                )
            else:
                hyps.append(
                    f"{weeks} weeks of history — sufficient for the full model zoo "
                    f"including ML and neural models."
                )
        except (ValueError, IndexError):
            pass

        # Series count
        if n > 50:
            hyps.append(
                f"{n} series — enough for global/cross-learning ML models "
                f"(LightGBM, XGBoost, neural) to learn shared patterns."
            )
        elif n > 10:
            hyps.append(
                f"{n} series — moderate scale. Per-series statistical models "
                f"are safe; ML models may benefit from cross-learning."
            )
        else:
            hyps.append(
                f"Only {n} series — per-series statistical models recommended. "
                f"Global ML models may overfit with this few series."
            )

        return hyps

    # ----- Config Recommendation ------------------------------------------- #

    def recommend_config(
        self,
        schema: SchemaDetection,
        hierarchy: HierarchyDetection,
        forecastability: ForecastabilityReport,
    ) -> Tuple[PlatformConfig, List[str]]:
        """Build a PlatformConfig tailored to the data characteristics."""
        reasoning: List[str] = []

        # Data length in periods (frequency-aware)
        freq = schema.frequency_guess
        try:
            from ..config.schema import get_frequency_profile
            profile = get_frequency_profile(freq)
        except ValueError:
            from ..config.schema import FREQUENCY_PROFILES
            profile = FREQUENCY_PROFILES["W"]
            freq = "W"

        try:
            d0 = date.fromisoformat(schema.date_range[0][:10])
            d1 = date.fromisoformat(schema.date_range[1][:10])
            from ..utils.calendar import PeriodCalendar
            data_periods = max(1, PeriodCalendar.from_dates([d0, d1], freq).index(d1))
        except (ValueError, IndexError):
            data_periods = profile["min_series_length"]
        data_weeks = data_periods  # backward-compat alias for threshold logic

        # ----- Forecast config ----- #
        default_horizon = profile["default_horizon"]
        horizon = min(default_horizon, max(profile["default_val_periods"], data_periods // 4))
        reasoning.append(f"Horizon set to {horizon} periods (freq={freq!r}, data_length / 4, capped at {default_horizon})")

        # Model selection based on forecastability and data characteristics
        forecasters = ["naive_seasonal"]
        reasoning.append("naive_seasonal included as baseline")

        n_series = schema.n_series
        overall_score = forecastability.overall_score

        # Statistical models — always include if enough data
        if data_weeks >= 26:
            forecasters.extend(["auto_ets", "auto_arima"])
            reasoning.append("auto_ets, auto_arima added (≥26 weeks of data)")

        # Seasonal-aware models
        strong_seasonal_pct = 0
        if forecastability.per_series is not None:
            ps = forecastability.per_series
            strong_seasonal_pct = ps.filter(pl.col("seasonal_strength") > 0.5).height / max(n_series, 1)

        if strong_seasonal_pct > 0.3:
            forecasters.extend(["auto_theta", "mstl"])
            reasoning.append(f"auto_theta, mstl added ({strong_seasonal_pct:.0%} of series are strongly seasonal)")

        # ML models
        if data_weeks >= 78 and n_series >= 5:
            forecasters.extend(["lgbm_direct", "xgboost_direct"])
            reasoning.append(f"lgbm_direct, xgboost_direct added (≥78 weeks, {n_series} series)")

        # Neural models — recommend when sufficient data and series
        if data_weeks >= 156 and n_series >= 30:
            forecasters.extend(["nhits", "nbeats"])
            reasoning.append(
                f"nhits, nbeats added (≥156 weeks, {n_series} series). "
                "NOTE: Neural models evaluated at CPU defaults; "
                "production performance requires GPU training with "
                "max_steps≥2000"
            )

        # Intermittent models
        sparse_count = sum(
            forecastability.demand_class_distribution.get(c, 0)
            for c in ("intermittent", "lumpy")
        )
        intermittent_forecasters: List[str] = []
        if sparse_count > 0:
            sparse_pct = sparse_count / max(n_series, 1)
            if sparse_pct > 0.1:
                intermittent_forecasters = ["croston_sba", "tsb"]
                reasoning.append(
                    f"Intermittent models (croston_sba, tsb) added — "
                    f"{sparse_pct:.0%} of series are sparse"
                )

        # External regressors
        from ..config.schema import ExternalRegressorConfig
        ext_reg = ExternalRegressorConfig(
            enabled=len(schema.numeric_columns) > 0,
            feature_columns=schema.numeric_columns,
        )
        if ext_reg.enabled:
            reasoning.append(f"External regressors enabled: {schema.numeric_columns}")

        forecast_config = ForecastConfig(
            horizon_weeks=horizon,
            frequency=schema.frequency_guess,
            target_column=schema.target_column,
            time_column=schema.time_column,
            series_id_column=schema.id_columns[0] if schema.id_columns else "series_id",
            forecasters=forecasters,
            quantiles=[0.1, 0.5, 0.9],
            intermittent_forecasters=intermittent_forecasters,
            sparse_detection=bool(intermittent_forecasters),
            external_regressors=ext_reg,
            calibration=CalibrationConfig(enabled=True, coverage_targets={"80": 0.80}),
        )

        # ----- Backtest config ----- #
        max_folds = max(1, data_weeks // (2 * horizon))
        n_folds = min(5, max(2, max_folds))
        champion_gran = "series" if n_series < 20 else "lob"
        reasoning.append(
            f"Backtest: {n_folds} folds, val_weeks={horizon}, "
            f"champion_granularity='{champion_gran}'"
        )

        backtest_config = BacktestConfig(
            n_folds=n_folds,
            val_weeks=horizon,
            champion_granularity=champion_gran,
        )

        # ----- Data quality config ----- #
        reasoning.append(
            "Demand cleansing and structural-break fitting remain disabled: "
            "backtesting requires fold-local preprocessing. "
            "Review these options separately for production-only forecasts."
        )
        if overall_score < 0.5:
            reasoning.append(
                "Forecastability < 0.5 suggests noisy data; review data quality "
                "without fitting whole-history cleansing before backtesting."
            )

        dq_config = DataQualityConfig(
            min_series_length_weeks=max(26, horizon * 2),
            cleansing=CleansingConfig(enabled=False),
            structural_breaks=StructuralBreakConfig(enabled=False),
            report=DataQualityReportConfig(enabled=True, sparse_classification=True),
        )

        # ----- Reconciliation config ----- #
        multi_level = any(len(h.levels) > 1 for h in hierarchy.hierarchies)
        reasoning.append(
            "Reconciliation: 'bottom_up' (supported leaf-level pipeline output; "
            f"{'detected multi-level hierarchy retained' if multi_level else 'flat hierarchy'})"
        )

        recon_config = ReconciliationConfig(method="bottom_up")

        # ----- Output config ----- #
        output_grain = {}
        for h in hierarchy.hierarchies:
            if h.levels:
                output_grain[h.name] = h.levels[-1]

        output_config = OutputConfig(grain=output_grain)

        # ----- Assemble ----- #
        config = PlatformConfig(
            lob=self.lob_name,
            description=f"Auto-generated config for {self.lob_name} ({n_series} series, {data_weeks} weeks)",
            hierarchies=hierarchy.hierarchies,
            reconciliation=recon_config,
            forecast=forecast_config,
            backtest=backtest_config,
            data_quality=dq_config,
            output=output_config,
            metrics=["wmape", "normalized_bias", "mase"],
        )

        return config, reasoning
