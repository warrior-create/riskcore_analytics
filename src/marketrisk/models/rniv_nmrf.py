"""
RNIV (Risk Not In VaR) and NMRF (Non-Modellable Risk Factor) module.

Purpose
-------
Demonstrates how short-history risk factors (e.g. LICI.NS IPO) are
handled under FRTB-style governance:

1. Modellability Test: factor is modellable if it has ≥24 observations
   in the last year (BCBS rule). LICI with <1 year of data fails this.

2. Proxy Assignment: LICI → Nifty (or sector peer). The proxy mapping
   is documented in the model inventory with stated basis risk.

3. Stress Scenario Charge: for NMRF factors, charge = stressed return
   at 99.9% confidence × position × stress_multiplier.

Documented limitation: this is a simplified illustrative implementation.
Full FRTB NMRF requires regulatory-approved proxy rules and SES aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class NMRFResult:
    ticker: str
    is_modellable: bool
    n_observations: int
    required_observations: int
    proxy_ticker: str | None
    stress_charge: float
    confidence: float
    narrative: str


def modellability_test(
    returns: pd.Series,
    lookback_years: float = 1.0,
    min_obs: int = 24,
) -> tuple[bool, int]:
    """
    Test whether a risk factor is modellable.

    BCBS rule (simplified): ≥ 24 real price observations in the last year.
    Each observation must represent a genuine trade/quote, not an estimate.

    Returns (is_modellable, n_obs)
    """
    n_trading_days = int(lookback_years * 252)
    recent = returns.iloc[-n_trading_days:].dropna()
    # Count non-zero observations (proxy for actual quotes)
    n_obs = int((recent != 0).sum())
    is_modellable = n_obs >= min_obs
    return is_modellable, n_obs


def nmrf_stress_charge(
    returns: pd.Series,
    position_size: float,
    confidence: float = 0.999,
    stress_multiplier: float = 1.5,
) -> float:
    """
    NMRF stress scenario charge.

    Charge = |position| × |stressed_return| × stress_multiplier
    where stressed_return = empirical quantile at (1 - confidence) from the left tail.

    If insufficient data, uses a ±60% stress (illustrative).
    """
    r = returns.dropna()
    if len(r) < 10:
        stressed_return = 0.60   # fallback stress
        logger.warning("Insufficient data for NMRF stress; using 60%% floor")
    else:
        # One-sided: worst-case loss
        stressed_return = float(-np.percentile(r, (1 - confidence) * 100))

    charge = abs(position_size) * stressed_return * stress_multiplier
    return float(charge)


def proxy_assignment(
    ticker: str,
    available_returns: pd.DataFrame,
    proxy_map: dict[str, str] | None = None,
) -> str | None:
    """
    Assign a proxy risk factor for an NMRF ticker.

    Default proxy map from config. Returns proxy ticker or None.
    """
    if proxy_map and ticker in proxy_map:
        return proxy_map[ticker]
    # Fallback: find highest correlation
    if ticker not in available_returns.columns:
        return None
    corr = available_returns.corrwith(available_returns[ticker]).drop(ticker)
    if corr.empty:
        return None
    return str(corr.idxmax())


def run_nmrf_assessment(
    ipo_ticker: str,
    ipo_returns: pd.Series,
    proxy_ticker: str,
    proxy_returns: pd.Series,
    position_size: float,
    cfg: dict,
) -> NMRFResult:
    """
    Full NMRF assessment for a single short-history factor.
    """
    nmrf_cfg = cfg.get("rniv_nmrf", {})
    min_obs = nmrf_cfg.get("min_observations_modellable", 24)
    lookback = nmrf_cfg.get("lookback_years", 1)
    multiplier = nmrf_cfg.get("stress_multiplier", 1.5)

    is_modellable, n_obs = modellability_test(ipo_returns, lookback, min_obs)

    # Use proxy returns if not modellable
    returns_for_stress = proxy_returns if not is_modellable else ipo_returns
    charge = nmrf_stress_charge(returns_for_stress, position_size, 0.999, multiplier)

    narrative = (
        f"{ipo_ticker} has {n_obs} real observations (need {min_obs}) in the past "
        f"{lookback:.0f} year(s). Status: {'MODELLABLE' if is_modellable else 'NON-MODELLABLE (NMRF)'}. "
        f"Proxy assigned: {proxy_ticker}. Stress charge: {charge:,.0f} INR."
    )

    logger.info(narrative)

    return NMRFResult(
        ticker=ipo_ticker,
        is_modellable=is_modellable,
        n_observations=n_obs,
        required_observations=min_obs,
        proxy_ticker=proxy_ticker if not is_modellable else None,
        stress_charge=charge,
        confidence=0.999,
        narrative=narrative,
    )
