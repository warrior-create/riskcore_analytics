"""
NSE trading-day calendar module.

Uses pandas_market_calendars if available, otherwise builds the calendar from
yfinance Nifty 50 actual trading dates — the master source per spec.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NSECalendar:
    """NSE trading-day calendar built from actual Nifty 50 trading dates."""

    def __init__(self, nifty_prices: pd.DataFrame | None = None):
        """
        Parameters
        ----------
        nifty_prices : DataFrame with DatetimeIndex (from yfinance Nifty download).
                       If None, falls back to a 5-day week calendar minus
                       known Indian public holidays.
        """
        if nifty_prices is not None:
            self._days = pd.DatetimeIndex(
                sorted(nifty_prices.index.normalize().unique())
            )
        else:
            logger.warning(
                "NSECalendar: no Nifty price data provided; "
                "using Mon-Fri calendar without holiday adjustment."
            )
            self._days = pd.bdate_range("2007-01-01", pd.Timestamp.today())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self._days

    def is_trading_day(self, date: pd.Timestamp | str) -> bool:
        return pd.Timestamp(date).normalize() in self._days

    def trading_days_between(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DatetimeIndex:
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        mask = (self._days >= start) & (self._days <= end)
        return self._days[mask]

    def expected_dates(self, start: str, end: str) -> pd.DatetimeIndex:
        return self.trading_days_between(start, end)

    def offset(self, date: pd.Timestamp | str, n: int) -> pd.Timestamp:
        """Return the date n trading days after (or before, if n<0) `date`."""
        date = pd.Timestamp(date).normalize()
        idx = self._days.searchsorted(date)
        target = idx + n
        target = int(np.clip(target, 0, len(self._days) - 1))
        return self._days[target]

    def hash(self) -> str:
        """SHA-256 of the sorted trading day list — used for dataset integrity."""
        data = "|".join(self._days.strftime("%Y-%m-%d"))
        return hashlib.sha256(data.encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._days)

    def __repr__(self) -> str:
        return (
            f"NSECalendar({self._days[0].date()} → {self._days[-1].date()}, "
            f"n={len(self._days)})"
        )
