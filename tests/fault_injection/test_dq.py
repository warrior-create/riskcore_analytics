"""
Fault injection tests for data quality module.

Requirements per spec:
  - Missing dates must be caught
  - 40% spike must be flagged as corporate-action jump
  - Frozen/stale prices (zero change for N days) must be detected
"""

import pytest
import sqlite3
import pandas as pd
import numpy as np
from datetime import date, timedelta

from marketrisk.data.quality import DataQualityChecker
from marketrisk.data.calendar import NSECalendar


def make_calendar(start="2023-01-02", end="2023-12-29"):
    dates = pd.bdate_range(start, end)
    return NSECalendar(pd.DataFrame(index=dates))


def make_prices(dates, seed=42):
    rng = np.random.default_rng(seed)
    tickers = ["HDFCBANK.NS", "RELIANCE.NS"]
    data = {
        t: pd.Series(
            1000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
            index=dates,
        )
        for t in tickers
    }
    return pd.DataFrame(data)


DEFAULT_CFG = {
    "data_quality": {
        "zscore_threshold": 5.0,
        "mad_threshold": 10.0,
        "stale_price_days": 3,
        "max_return_threshold": 0.40,
    }
}


@pytest.fixture
def conn():
    from marketrisk.data.schema import init_db
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    c = init_db(path)
    yield c
    c.close()
    os.unlink(path)


class TestFaultInjection:

    def test_missing_date_detected(self, conn):
        """Removing a business day from prices triggers missing_trading_date issue."""
        cal = make_calendar()
        dates = pd.bdate_range("2023-01-02", "2023-12-29")
        prices = make_prices(dates)

        # Inject: drop 10 random dates from prices (but they remain in calendar)
        drop_dates = pd.bdate_range("2023-06-01", "2023-06-15")
        prices_missing = prices.drop(index=drop_dates, errors="ignore")

        checker = DataQualityChecker(conn, cal, DEFAULT_CFG)
        issues = checker.check_missing_dates(prices_missing, "2023-01-02", "2023-12-29")

        missing_issues = [i for i in issues if i["check_name"] == "missing_trading_date"]
        assert len(missing_issues) > 0, "Expected missing date issues to be detected"

    def test_stale_price_detected(self, conn):
        """Frozen price for 3+ consecutive days triggers stale_price issue."""
        cal = make_calendar()
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        prices = make_prices(dates)

        # Inject: freeze HDFCBANK at a constant price for 5 consecutive days
        freeze_dates = dates[100:105]
        prices.loc[freeze_dates, "HDFCBANK.NS"] = prices.loc[dates[99], "HDFCBANK.NS"]

        checker = DataQualityChecker(conn, cal, DEFAULT_CFG)
        issues = checker.check_stale_prices(prices)

        stale = [i for i in issues if i["check_name"] == "stale_price" and i["ticker"] == "HDFCBANK.NS"]
        assert len(stale) > 0, "Expected stale price to be detected"

    def test_40pct_spike_detected(self, conn):
        """A 40% single-day price jump triggers corporate_action_jump."""
        cal = make_calendar()
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        prices = make_prices(dates)

        # Inject: 50% spike on one day
        spike_date = dates[50]
        prices.loc[spike_date, "RELIANCE.NS"] *= 1.50

        checker = DataQualityChecker(conn, cal, DEFAULT_CFG)
        issues = checker.check_corporate_action_jumps(prices)

        jumps = [
            i for i in issues
            if i["check_name"] == "corporate_action_jump"
            and i["ticker"] == "RELIANCE.NS"
        ]
        assert len(jumps) > 0, "Expected 40%+ spike to be flagged"

    def test_return_outlier_detected(self, conn):
        """A 10-sigma return is flagged by z-score detector."""
        cal = make_calendar()
        dates = pd.bdate_range("2023-01-02", "2023-06-30")
        prices = make_prices(dates)

        # Inject: manually create a 10-sigma log return
        std = float(np.log(prices["HDFCBANK.NS"] / prices["HDFCBANK.NS"].shift(1)).std())
        prices.iloc[60, 0] *= np.exp(10 * std)

        checker = DataQualityChecker(conn, cal, DEFAULT_CFG)
        issues = checker.check_return_outliers(prices)

        outliers = [i for i in issues if i["check_name"] == "return_outlier"]
        assert len(outliers) > 0, "Expected 10-sigma outlier to be detected"

    def test_no_false_positives_on_clean_data(self, conn):
        """Clean data should produce zero DQ issues."""
        cal = make_calendar()
        dates = pd.bdate_range("2023-01-02", "2023-03-31")
        prices = make_prices(dates, seed=99)

        checker = DataQualityChecker(conn, cal, DEFAULT_CFG)

        # Only check stale and outliers (missing dates may fire due to calendar mismatch)
        stale = checker.check_stale_prices(prices)
        jumps = checker.check_corporate_action_jumps(prices)

        assert len(stale) == 0, f"Unexpected stale issues on clean data: {stale}"
        assert len(jumps) == 0, f"Unexpected jumps on clean data: {jumps}"
