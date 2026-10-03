"""
Legacy EUC mock tool: simulates the Excel/VBA parametric VaR calculator.

This module provides the Python equivalent of the legacy VBA tool that
computes VaR and exceptions using:
  - Fixed 250-day lookback
  - Equal-weight portfolio
  - Normal distribution (parametric)
  - No EWMA — simple historical variance

Used for EUC reconciliation. The VBA original is in euc/vba/.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class LegacyVBACalculator:
    """
    Replicates the legacy Excel/VBA parametric VaR tool.

    The VBA tool used:
    - lookback = 250 business days (hard-coded)
    - Equal weights across all equity positions
    - Simple (not EWMA) covariance matrix
    - Normal distribution, one-tailed

    All parameters are hard-coded to match the VBA tool exactly.
    """

    LOOKBACK_DAYS = 250
    CONFIDENCE = 0.99
    DISTRIBUTION = "normal"

    def compute_var(
        self,
        prices: pd.DataFrame,
        as_of_date: str | None = None,
    ) -> dict:
        """
        Compute parametric VaR matching the VBA tool's output.

        Parameters
        ----------
        prices : wide DataFrame of adj closes (dates × tickers)
        as_of_date : compute using data up to this date

        Returns
        -------
        dict with var, es, vol, weights, n_obs
        """
        if as_of_date:
            prices = prices.loc[:as_of_date]

        tail = prices.iloc[-self.LOOKBACK_DAYS:]
        if len(tail) < 30:
            raise ValueError(f"Insufficient data: {len(tail)} days")

        # Log returns
        log_rets = np.log(tail / tail.shift(1)).dropna()

        N = log_rets.shape[1]
        weights = np.ones(N) / N   # equal weight

        # Simple covariance (NOT EWMA — matches VBA tool)
        Sigma = np.cov(log_rets.T)
        port_var_daily = weights @ Sigma @ weights
        port_vol = np.sqrt(port_var_daily)

        z = stats.norm.ppf(self.CONFIDENCE)
        var_1d = port_vol * z
        es_1d = port_vol * stats.norm.pdf(z) / (1 - self.CONFIDENCE)

        return {
            "model": "legacy_vba_parametric",
            "confidence": self.CONFIDENCE,
            "var_1d": float(var_1d),
            "es_1d": float(es_1d),
            "port_vol": float(port_vol),
            "n_obs": len(log_rets),
            "weights": weights.tolist(),
            "lookback_days": self.LOOKBACK_DAYS,
        }

    def compute_exceptions(
        self,
        prices: pd.DataFrame,
        pnl_series: pd.Series,
        window: int = 250,
    ) -> pd.DataFrame:
        """Compute rolling exceptions matching VBA tool output."""
        records = []
        for i in range(self.LOOKBACK_DAYS, len(prices)):
            date = prices.index[i]
            hist_prices = prices.iloc[i - self.LOOKBACK_DAYS:i]
            try:
                result = self.compute_var(hist_prices)
                var_est = result["var_1d"]
            except Exception:
                continue

            actual_loss = -pnl_series.get(date, 0.0)
            records.append({
                "date": date,
                "var_estimate": var_est,
                "actual_loss": actual_loss,
                "exception": actual_loss > var_est,
            })

        return pd.DataFrame(records)
