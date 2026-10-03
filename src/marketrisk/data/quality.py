"""
Data Quality module for MarketRisk-Lab.

Checks
------
1. Missing trading dates (against NSE calendar)
2. Stale prices (zero change for N consecutive days)
3. Z-score / MAD outliers in returns
4. Corporate-action jumps (>40% 1-day move)
5. Cross-source reconciliation placeholder

All failures are written to the dq_log table.
Fault-injection tests in tests/fault_injection/test_dq.py
exercise every check.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from marketrisk.data.calendar import NSECalendar

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Run all DQ checks and write failures to dq_log."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        calendar: NSECalendar,
        cfg: dict,
    ):
        self.conn = conn
        self.cal = calendar
        self.cfg = cfg
        self.dq_cfg = cfg.get("data_quality", {})
        self.zscore_thr = self.dq_cfg.get("zscore_threshold", 5.0)
        self.mad_thr = self.dq_cfg.get("mad_threshold", 10.0)
        self.stale_days = self.dq_cfg.get("stale_price_days", 3)
        self.max_return = self.dq_cfg.get("max_return_threshold", 0.40)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_all(
        self, prices: pd.DataFrame, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """
        Run all checks on a wide prices DataFrame (dates × tickers).

        Returns a DataFrame of issues (mirrors dq_log table).
        """
        issues: list[dict] = []
        issues.extend(self.check_missing_dates(prices, start, end))
        issues.extend(self.check_stale_prices(prices))
        issues.extend(self.check_return_outliers(prices))
        issues.extend(self.check_corporate_action_jumps(prices))

        if issues:
            self._write_to_db(issues)
            logger.warning("DQ: %d issues found and logged", len(issues))
        else:
            logger.info("DQ: all checks passed")

        return pd.DataFrame(issues)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_missing_dates(
        self,
        prices: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict]:
        """Compare observed dates against the NSE calendar."""
        issues: list[dict] = []
        if prices.empty:
            return issues

        observed = set(prices.index.normalize())
        start = start or str(prices.index.min().date())
        end = end or str(prices.index.max().date())
        expected = set(self.cal.expected_dates(start, end))

        missing = sorted(expected - observed)
        for dt in missing:
            issues.append(
                _issue(
                    "missing_trading_date",
                    None,
                    str(dt.date()),
                    "WARNING",
                    f"Date {dt.date()} is a trading day but absent from prices table",
                )
            )
        return issues

    def check_stale_prices(self, prices: pd.DataFrame) -> list[dict]:
        """Flag tickers where close price is unchanged for >= stale_days consecutive days."""
        issues: list[dict] = []
        for ticker in prices.columns:
            series = prices[ticker].dropna()
            if len(series) < self.stale_days + 1:
                continue
            diff = series.diff().abs()
            # Rolling window: if sum of abs changes == 0 for stale_days
            rolling_sum = diff.rolling(self.stale_days).sum()
            stale_dates = rolling_sum[rolling_sum == 0].index
            for dt in stale_dates:
                issues.append(
                    _issue(
                        "stale_price",
                        ticker,
                        str(dt.date()),
                        "WARNING",
                        f"{ticker}: price unchanged for {self.stale_days} consecutive days ending {dt.date()}",
                    )
                )
        return issues

    def check_return_outliers(self, prices: pd.DataFrame) -> list[dict]:
        """Z-score and MAD outlier detection on log returns."""
        issues: list[dict] = []
        log_rets = np.log(prices / prices.shift(1)).dropna(how="all")

        for ticker in log_rets.columns:
            r = log_rets[ticker].dropna()
            if len(r) < 30:
                continue

            # Z-score
            mu, sigma = r.mean(), r.std()
            if sigma == 0:
                continue
            z = (r - mu) / sigma
            outliers_z = r[z.abs() > self.zscore_thr]

            # MAD
            median = r.median()
            mad = (r - median).abs().median()
            if mad == 0:
                continue
            modified_z = 0.6745 * (r - median) / mad
            outliers_mad = r[modified_z.abs() > self.mad_thr]

            flagged = outliers_z.index.union(outliers_mad.index)
            for dt in flagged:
                issues.append(
                    _issue(
                        "return_outlier",
                        ticker,
                        str(dt.date()),
                        "WARNING",
                        (
                            f"{ticker}: log-return {r.loc[dt]:.4f} on {dt.date()} "
                            f"(z={z.loc[dt]:.2f}) flagged as outlier"
                        ),
                    )
                )
        return issues

    def check_corporate_action_jumps(self, prices: pd.DataFrame) -> list[dict]:
        """Flag absolute 1-day price moves > max_return_threshold."""
        issues: list[dict] = []
        rets = prices.pct_change().dropna(how="all")
        for ticker in rets.columns:
            r = rets[ticker].dropna()
            big = r[r.abs() > self.max_return]
            for dt in big.index:
                issues.append(
                    _issue(
                        "corporate_action_jump",
                        ticker,
                        str(dt.date()),
                        "ERROR",
                        (
                            f"{ticker}: {r.loc[dt]*100:.1f}% 1-day move on {dt.date()} — "
                            "check for corporate action (split/bonus/dividend)"
                        ),
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_db(self, issues: list[dict]) -> None:
        rows = [
            (
                issue["check_name"],
                issue.get("ticker"),
                issue.get("date_affected"),
                issue["severity"],
                issue["message"],
            )
            for issue in issues
        ]
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO dq_log (check_name, ticker, date_affected, severity, message)
               VALUES (?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()

    def get_open_issues(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM dq_log WHERE resolved=0 ORDER BY logged_at DESC", self.conn
        )

    def resolve_issue(self, issue_id: int) -> None:
        self.conn.execute(
            "UPDATE dq_log SET resolved=1 WHERE id=?", (issue_id,)
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _issue(
    check_name: str,
    ticker: Optional[str],
    date_affected: Optional[str],
    severity: str,
    message: str,
) -> dict:
    return {
        "check_name": check_name,
        "ticker": ticker,
        "date_affected": date_affected,
        "severity": severity,
        "message": message,
        "logged_at": datetime.utcnow().isoformat(),
    }
