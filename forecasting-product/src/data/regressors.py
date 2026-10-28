"""
External regressor data loading, holiday generation, and validation.

Provides utilities for loading promotional calendars, generating holiday
features, and validating that external regressors align with actuals data.
"""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl

from ..utils.calendar import PeriodCalendar, advance_date

logger = logging.getLogger(__name__)

# Holiday generation — optional dependency
try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False


def load_external_features(path: str) -> pl.DataFrame:
    """
    Load external feature data from Parquet or CSV.

    Expected columns: at minimum a time column (e.g. 'week') and one or more
    feature columns.  May optionally include a series_id column for
    series-specific features (e.g. per-SKU promotions).

    Parameters
    ----------
    path:
        Path to a Parquet or CSV file.

    Returns
    -------
    DataFrame with external feature columns.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External features file not found: {path}")

    if p.suffix == ".parquet":
        return pl.read_parquet(path)
    elif p.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv")


def generate_holiday_calendar(
    country: str,
    start_date: date,
    end_date: date,
    time_column: str = "week",
    frequency: str = "W",
) -> pl.DataFrame:
    """
    Generate a holiday flag calendar for a given country and frequency.

    Requires the `holidays` package (pip install holidays).
    Each period gets a count of holidays falling within that period.

    Parameters
    ----------
    country:
        ISO 3166-1 alpha-2 country code (e.g. 'US', 'DE', 'GB').
    start_date:
        Start of the date range.
    end_date:
        End of the date range.
    time_column:
        Name for the time column in the output.
    frequency:
        Data frequency — ``"D"``, ``"W"``, ``"M"``, ``"Q"``.

    Returns
    -------
    DataFrame with columns [time_column, 'holiday_flag', 'holiday_count'].
    """
    if not _HAS_HOLIDAYS:
        logger.warning(
            "holidays package not installed. "
            "Install with: pip install holidays. Returning empty calendar."
        )
        return pl.DataFrame(schema={
            time_column: pl.Date,
            "holiday_flag": pl.Int8,
            "holiday_count": pl.Int32,
        })

    calendar = PeriodCalendar.from_dates([start_date], frequency)
    years = list(range(advance_date(start_date, frequency, -1).year, advance_date(end_date, frequency, 1).year + 1))
    country_holidays = _holidays_lib.country_holidays(country, years=years)

    # Generate date range at the configured frequency
    weeks = []
    index = 0
    while calendar.at(index) <= end_date:
        weeks.append(calendar.at(index))
        index += 1

    holiday_counts = []
    for week_start in weeks:
        if frequency in ("M", "Q") and start_date.day != 1:
            window_start = calendar.shift(week_start, -1) + timedelta(days=1)
            window_end = week_start + timedelta(days=1)
        else:
            window_start = week_start
            window_end = calendar.shift(week_start, 1)
        count = sum(
            window_start + timedelta(days=offset) in country_holidays
            for offset in range((window_end - window_start).days)
        )
        holiday_counts.append(count)

    return pl.DataFrame({
        time_column: weeks,
        "holiday_flag": [1 if c > 0 else 0 for c in holiday_counts],
        "holiday_count": holiday_counts,
    })


def validate_regressors(
    external_features: pl.DataFrame,
    actuals: pl.DataFrame,
    feature_columns: List[str],
    time_column: str = "week",
    id_column: Optional[str] = None,
    horizon_weeks: int = 0,
    feature_types: Optional[dict] = None,
    frequency: str = "W",
) -> List[str]:
    """
    Validate external regressor data against actuals.

    Checks:
    1. All feature columns exist in external_features.
    2. Time column alignment (same grain as actuals).
    3. No nulls in feature columns during training period.
    4. If horizon_weeks > 0, future feature values must be present.
    5. Series ID alignment (if series-level features).
    6. Temporal causality — contemporaneous features without future values
       cannot be used for forecasting.

    Parameters
    ----------
    external_features:
        The external feature DataFrame.
    actuals:
        The actuals DataFrame (for alignment checks).
    feature_columns:
        List of expected feature column names.
    time_column:
        Time column name.
    id_column:
        Optional series ID column (for series-level features).
    horizon_weeks:
        If > 0, validates that future feature values exist for this many weeks.
    feature_types:
        Optional dict mapping column name to ``"known_ahead"`` or
        ``"contemporaneous"``.  Unlisted columns default to ``"known_ahead"``.

    Returns
    -------
    List of warning/error messages. Empty list means all checks passed.
    """
    issues: List[str] = []
    feature_types = feature_types or {}

    # Check 1: Feature columns exist
    missing_cols = [c for c in feature_columns if c not in external_features.columns]
    if missing_cols:
        issues.append(f"Missing feature columns in external data: {missing_cols}")
        return issues  # Can't proceed without columns

    # Check 2: Time column exists
    if time_column not in external_features.columns:
        issues.append(f"Time column '{time_column}' not found in external features")
        return issues

    # Check 3: Nulls in feature columns
    for col in feature_columns:
        null_count = external_features[col].null_count()
        if null_count > 0:
            issues.append(
                f"Feature '{col}' has {null_count} null values. "
                f"Consider filling with forward-fill or zeros."
            )

    # Check 4: Time range coverage
    if time_column in actuals.columns:
        actuals_max = actuals[time_column].max()
        features_max = external_features[time_column].max()
        actuals_min = actuals[time_column].min()
        features_min = external_features[time_column].min()

        if features_min is not None and actuals_min is not None:
            if features_min > actuals_min:
                issues.append(
                    f"External features start ({features_min}) is after "
                    f"actuals start ({actuals_min}). Early periods will have null features."
                )

        if horizon_weeks > 0 and actuals_max is not None and features_max is not None:
            calendar = PeriodCalendar.from_dates(actuals[time_column].unique().to_list(), frequency)
            required_end = calendar.shift(actuals_max, horizon_weeks)
            if features_max < required_end:
                issues.append(
                    f"External features end ({features_max}) does not cover "
                    f"the forecast horizon (need through {required_end}). "
                    f"Prediction will fail without future feature values."
                )

    # Check 5: Series ID alignment (if series-level features)
    if id_column and id_column in external_features.columns and id_column in actuals.columns:
        feat_ids = set(external_features[id_column].unique().to_list())
        actual_ids = set(actuals[id_column].unique().to_list())
        missing_ids = actual_ids - feat_ids
        if missing_ids:
            issues.append(
                f"{len(missing_ids)} series in actuals have no external features. "
                f"These series will get null feature values (filled with 0)."
            )

    # Check 6: Temporal causality — contemporaneous features need future values
    if horizon_weeks > 0 and time_column in actuals.columns:
        actuals_max = actuals[time_column].max()
        features_max = external_features[time_column].max()
        for col in feature_columns:
            ftype = feature_types.get(col, "known_ahead")
            if ftype == "contemporaneous":
                if (
                    actuals_max is not None
                    and features_max is not None
                    and features_max <= actuals_max
                ):
                    issues.append(
                        f"Feature '{col}' is marked as contemporaneous but has "
                        f"no future values beyond the training period "
                        f"(features end: {features_max}, actuals end: {actuals_max}). "
                        f"This feature will be dropped at prediction time. "
                        f"Either provide future values or remove it from the model."
                    )

    return issues
