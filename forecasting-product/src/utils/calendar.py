"""Anchored period arithmetic for explicitly configured D/W/M/Q series.

Months and quarters are calendar periods, never approximate durations.
Monthly/quarterly model grids support period-start and period-end labels,
including fiscal quarters. Weekly grids retain their observed weekday.
"""

from calendar import month_abbr, monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from numbers import Integral
from typing import Iterable, List

from ..config.schema import get_frequency_profile


def _integer_periods(periods: int) -> int:
    if isinstance(periods, bool) or not isinstance(periods, Integral):
        raise ValueError("Periods must be an integer")
    return int(periods)


def advance_date(value: date, frequency: str, periods: int = 1) -> date:
    """Offset a date, preserving month-end or clamping its day if necessary.

    For a series, use ``PeriodCalendar`` instead: its original anchor survives
    February and validation prevents drifting/irregular input from being guessed.
    This scalar helper also supports arbitrary event dates (e.g. SKU launches).
    """
    get_frequency_profile(frequency)
    periods = _integer_periods(periods)
    if frequency in ("D", "W"):
        return value + timedelta(days=periods * (7 if frequency == "W" else 1))
    month_index = value.year * 12 + value.month - 1 + periods * (3 if frequency == "Q" else 1)
    year, month = divmod(month_index, 12)
    month += 1
    last_day = monthrange(year, month)[1]
    day = last_day if value.day == monthrange(value.year, value.month)[1] else min(value.day, last_day)
    return value.replace(year=year, month=month, day=day)


@dataclass(frozen=True)
class PeriodCalendar:
    """One validated, explicitly configured calendar and its stable anchor."""

    frequency: str
    origin: date

    @classmethod
    def from_dates(cls, dates: Iterable[date], frequency: str) -> "PeriodCalendar":
        """Validate date labels against a frequency; gaps are allowed, drift is not."""
        get_frequency_profile(frequency)
        values = list(dates)
        if not values or any(not isinstance(value, date) for value in values):
            raise ValueError("A calendar requires non-null date labels")
        values = sorted(set(values))
        if frequency in ("M", "Q"):
            starts = all(value.day == 1 for value in values)
            ends = all(value.day == monthrange(value.year, value.month)[1] for value in values)
            if not (starts or ends):
                raise ValueError(
                    f"Frequency {frequency!r} requires a consistent calendar "
                    "period-start or period-end anchor; irregular dates cannot be inferred"
                )
        calendar = cls(frequency, values[0])
        for value in values:
            calendar.index(value)
        return calendar

    def at(self, index: int) -> date:
        """Return the date at a signed integer period index from the anchor."""
        return advance_date(self.origin, self.frequency, index)

    def index(self, value: date) -> int:
        """Return an exact period index, rejecting off-grid dates."""
        if self.frequency in ("D", "W"):
            days = (value - self.origin).days
            index = days // (7 if self.frequency == "W" else 1)
        else:
            months = (value.year - self.origin.year) * 12 + value.month - self.origin.month
            index = months // (3 if self.frequency == "Q" else 1)
        if self.at(index) != value:
            raise ValueError(f"{value!r} is off the {self.frequency!r} calendar anchored at {self.origin!r}")
        return index

    def shift(self, value: date, periods: int) -> date:
        """Move by periods without losing the original calendar anchor."""
        return self.at(self.index(value) + _integer_periods(periods))

    def future_dates(self, last_date: date, horizon: int) -> List[date]:
        """The next ``horizon`` period labels after ``last_date``."""
        horizon = _integer_periods(horizon)
        if horizon < 1:
            raise ValueError("Forecast horizon must be at least one period")
        start = self.index(last_date)
        return [self.at(start + step) for step in range(1, horizon + 1)]

    def date_range(self, start: date, end: date) -> List[date]:
        """All calendar labels in an inclusive, aligned range."""
        return [self.at(index) for index in range(self.index(start), self.index(end) + 1)]

    @property
    def pandas_frequency(self) -> str:
        """Equivalent anchored offset for Nixtla/pandas boundaries."""
        if self.frequency == "D":
            return "D"
        if self.frequency == "W":
            return "W-" + ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")[self.origin.weekday()]
        suffix = "S" if self.origin.day == 1 else "E"
        if self.frequency == "M":
            return "M" + suffix
        return "Q" + suffix + "-" + month_abbr[self.origin.month].upper()
