"""
External feature lifecycle management for mlforecast-based models.

Handles:
  1. Detection of external feature columns during fit.
  2. Merging features into the training DataFrame.
  3. Generating future feature values for prediction (forward-fill fallback
     for known-ahead features only; contemporaneous features require explicit
     future values or are dropped).
  4. Accepting user-provided future features via ``set_future_features``.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd  # noqa: F401 — required for .to_pandas() and type hints
import polars as pl

from ..config.schema import get_frequency_profile
from ..utils.calendar import PeriodCalendar

logger = logging.getLogger(__name__)

# pandas is used at the boundary with mlforecast (which requires pandas DataFrames).
# All internal processing uses Polars; .to_pandas() is called only when returning
# DataFrames to the external library.
class MLForecastFeatureManager:
    """
    Manages external features for mlforecast models.

    Separates feature handling from model fitting/prediction logic
    so that ``_DirectMLBase`` doesn't need inline feature plumbing.
    """

    def __init__(self, feature_types: Optional[Dict[str, str]] = None, frequency: str = "W") -> None:
        get_frequency_profile(frequency)
        self.frequency = frequency
        self._calendar: Optional[PeriodCalendar] = None
        self._feature_cols: List[str] = []
        self._train: Optional[pl.DataFrame] = None
        self._future_features: Optional[pl.DataFrame] = None
        # Maps column name → "known_ahead" | "contemporaneous"
        self._feature_types: Dict[str, str] = feature_types or {}

    @property
    def feature_cols(self) -> List[str]:
        """External feature column names detected during fit."""
        return list(self._feature_cols)

    @property
    def has_features(self) -> bool:
        return len(self._feature_cols) > 0

    def detect_features(
        self, df: pl.DataFrame, id_col: str, time_col: str, target_col: str
    ) -> List[str]:
        """Detect external feature columns (anything beyond id, time, target).

        Only numeric columns are kept as external features; string/categorical
        columns (e.g. hierarchy identifiers) are silently excluded because
        tree-based learners cannot consume them without explicit encoding.
        """
        core_cols = {id_col, time_col, target_col}
        all_extra = [c for c in df.columns if c not in core_cols]

        # Only keep numeric columns — string/categorical columns would cause
        # errors in LightGBM/XGBoost and produce NaN lags in mlforecast.
        numeric_extra = [
            c for c in all_extra if df[c].dtype.is_numeric()
        ]
        dropped = set(all_extra) - set(numeric_extra)
        if dropped:
            logger.debug(
                "Excluding non-numeric columns from ML features: %s",
                sorted(dropped),
            )

        # Strip contemporaneous features that have no future values —
        # training on features unavailable at prediction time hurts accuracy.
        if self._future_features is None:
            usable = []
            for col in numeric_extra:
                if self._get_feature_type(col) == "contemporaneous":
                    logger.warning(
                        "Dropping contemporaneous feature '%s' from training: "
                        "no future values provided. The model will not use "
                        "this feature. Use set_future_features() or mark as "
                        "'known_ahead' in config if this feature is plannable.",
                        col,
                    )
                else:
                    usable.append(col)
            self._feature_cols = usable
        else:
            self._feature_cols = numeric_extra

        return self._feature_cols

    def prepare_fit(
        self,
        df: pl.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> "pd.DataFrame":
        """
        Build the pandas training DataFrame for mlforecast.

        Returns the full DataFrame (core + features) renamed to
        mlforecast conventions (unique_id, ds, y).

        Contemporaneous features without future values are excluded from
        training to prevent the model from learning on features that
        won't be available at prediction time.
        """
        self._calendar = PeriodCalendar.from_dates(df[time_col].unique().to_list(), self.frequency)
        if df.select([id_col, time_col]).is_duplicated().any():
            raise ValueError("Duplicate series/date keys in feature training data")
        self.detect_features(df, id_col, time_col, target_col)

        # Core columns (keep in Polars)
        core_pl = df.select([id_col, time_col, target_col]).rename(
            {id_col: "unique_id", time_col: "ds", target_col: "y"}
        )
        # Store Polars version for internal use
        self._train = core_pl

        if not self._feature_cols:
            # External library (mlforecast) requires pandas DataFrame
            pdf = core_pl.to_pandas()
            pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
            return pdf

        # Merge external features using Polars
        exog_pl = df.select([id_col, time_col] + self._feature_cols).rename(
            {id_col: "unique_id", time_col: "ds"}
        )
        merged_pl = core_pl.join(exog_pl, on=["unique_id", "ds"], how="left")
        merged_pl = merged_pl.with_columns(
            [pl.col(c).fill_null(0) for c in self._feature_cols]
        )

        # Store training data with features for prepare_predict
        self._train = merged_pl

        # External library (mlforecast) requires pandas DataFrame
        pdf = merged_pl.to_pandas()
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
        return pdf

    def _get_feature_type(self, col: str) -> str:
        """Return the temporal type for a feature column."""
        return self._feature_types.get(col, "known_ahead")

    def _eligible_predict_cols(self) -> List[str]:
        """
        Return feature columns eligible for prediction.

        Contemporaneous features (e.g. actual promo ratio) are dropped
        unless explicit future values were provided via set_future_features.
        """
        if self._future_features is not None:
            # User provided explicit future values — all features are usable
            return list(self._feature_cols)

        eligible = []
        for col in self._feature_cols:
            if self._get_feature_type(col) == "contemporaneous":
                logger.warning(
                    "Dropping contemporaneous feature '%s' at prediction time: "
                    "no future values provided. Use set_future_features() or "
                    "mark as 'known_ahead' in config if this feature is "
                    "plannable.",
                    col,
                )
            else:
                eligible.append(col)
        return eligible

    def prepare_predict(self, horizon: int) -> Optional["pd.DataFrame"]:
        """
        Build future feature DataFrame for mlforecast predict.

        Returns None if no external features are present.
        Uses user-provided future features if available, otherwise
        forward-fills the last known values per series for known-ahead
        features only. Contemporaneous features without explicit future
        values are dropped with a warning.
        """
        if not self._feature_cols:
            return None

        # User-provided future features take priority
        if self._future_features is not None:
            # External library (mlforecast) requires pandas DataFrame
            pdf = self._future_features.to_pandas()
            pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
            return pdf

        # Determine which features can be forward-filled
        eligible_cols = self._eligible_predict_cols()
        if not eligible_cols:
            return None

        # Forward-fill fallback (known-ahead features only) using Polars
        if self._train is None:
            return None

        # Get the last row per unique_id with eligible feature values
        last_per_series = (
            self._train
            .sort("ds")
            .group_by("unique_id")
            .last()
        )

        # Build future rows for each series and horizon step
        rows = []
        for last in last_per_series.iter_rows(named=True):
            for future_ds in self._calendar.future_dates(last["ds"], horizon):
                rows.append({
                    "unique_id": last["unique_id"], "ds": future_ds,
                    **{col: float(last[col]) if last.get(col) is not None else 0.0 for col in eligible_cols},
                })
        future_pl = pl.DataFrame(rows)

        # External library (mlforecast) requires pandas DataFrame
        pdf = future_pl.to_pandas()
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
        return pdf

    def set_future_features(
        self,
        future_features: pl.DataFrame,
        id_col: str = "series_id",
        time_col: str = "week",
    ) -> None:
        """Set user-provided external feature values for the forecast horizon."""
        if future_features is not None and not future_features.is_empty():
            self._future_features = future_features.rename(
                {id_col: "unique_id", time_col: "ds"}
            )

    @property
    def train_pdf(self) -> Optional["pd.DataFrame"]:
        """Stored training pandas DataFrame (for quantile model retraining)."""
        if self._train is None:
            return None
        # External library (mlforecast) requires pandas DataFrame
        pdf = self._train.to_pandas()
        pdf["ds"] = pdf["ds"].astype("datetime64[ns]")
        return pdf
