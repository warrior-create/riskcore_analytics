"""
Stressed ES and Liquidity-Horizon scaling per simplified FRTB methodology.

Stressed ES
-----------
Formula (FRTB, simplified):
    ES_stressed = ES(t, full) × ES(s, reduced) / ES(t, reduced)

where:
  t = current calibration period
  s = stress period (worst 12-month window, selected automatically)
  full set = all risk factors
  reduced set = subset of factors (equities + FX here, as per config)

Liquidity-Horizon Scaling
--------------------------
Cascade formula (BCBS d457, simplified form):
    VaR(LH) = VaR(1d) × √LH

We apply by risk-factor class using the configured liquidity horizons.

References
----------
BCBS (2019). Minimum capital requirements for market risk (d457).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from marketrisk.models.var_models import HistoricalVaR, compute_es

logger = logging.getLogger(__name__)


def find_stress_window(
    losses: pd.Series,
    window_days: int = 252,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """
    Find the 12-month (252-trading-day) window with the highest average loss.

    Returns
    -------
    (start, end, stressed_es)
    """
    rolling_mean = losses.rolling(window_days).mean()
    worst_end = rolling_mean.idxmax()
    worst_start = losses.index[max(0, losses.index.get_loc(worst_end) - window_days + 1)]
    stressed_es = float(rolling_mean.loc[worst_end])
    return worst_start, worst_end, stressed_es


def stressed_es(
    current_returns: pd.DataFrame,
    historical_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.975,
    window_days: int = 252,
    reduced_set_cols: Optional[list[str]] = None,
) -> dict:
    """
    Compute stressed ES using the FRTB scaling formula.

    Parameters
    ----------
    current_returns    : recent return DataFrame (T_current × N)
    historical_returns : full history DataFrame (T_full × N)
    weights            : portfolio weights
    confidence         : ES confidence level
    window_days        : stress window length (trading days)
    reduced_set_cols   : columns representing the reduced risk-factor set

    Returns
    -------
    dict with keys: ES_current_full, ES_stress_reduced, ES_current_reduced,
                    ES_stressed (the final number), stress_start, stress_end
    """
    def _get_es(rets: pd.DataFrame, w: np.ndarray) -> float:
        r = rets.dropna().values
        port_loss = -(r @ w)
        _, es = compute_es(port_loss, confidence)
        return es

    # Current ES on full factor set
    ES_t_full = _get_es(current_returns, weights)

    # Reduced set
    if reduced_set_cols is None:
        reduced_set_cols = list(current_returns.columns)

    # Map reduced columns to weights
    all_cols = list(current_returns.columns)
    reduced_idx = [all_cols.index(c) for c in reduced_set_cols if c in all_cols]
    w_reduced = np.zeros(len(all_cols))
    for idx in reduced_idx:
        w_reduced[idx] = weights[idx]
    if w_reduced.sum() > 0:
        w_reduced /= w_reduced.sum()

    ES_t_reduced = _get_es(current_returns[reduced_set_cols] if reduced_set_cols else current_returns,
                           w_reduced[reduced_idx] if reduced_idx else weights)

    # Find stress window
    hist_r = historical_returns.dropna()
    port_loss_hist = -(hist_r.values @ weights[:hist_r.shape[1]])
    loss_series = pd.Series(port_loss_hist, index=hist_r.index)
    stress_start, stress_end, _ = find_stress_window(loss_series, window_days)

    stress_rets = hist_r.loc[stress_start:stress_end]
    if len(stress_rets) < 20:
        logger.warning("Stress window too short (%d days), using full history", len(stress_rets))
        stress_rets = hist_r

    ES_s_reduced = _get_es(
        stress_rets[reduced_set_cols] if reduced_set_cols else stress_rets,
        w_reduced[reduced_idx] if reduced_idx else weights,
    )

    # Scaling
    if ES_t_reduced > 0:
        scaling = ES_s_reduced / ES_t_reduced
    else:
        scaling = 1.0

    ES_stressed_final = ES_t_full * scaling

    logger.info(
        "Stressed ES: ES_t_full=%.4f, scaling=%.2f → ES_stressed=%.4f",
        ES_t_full, scaling, ES_stressed_final,
    )

    return {
        "ES_current_full": ES_t_full,
        "ES_current_reduced": ES_t_reduced,
        "ES_stress_reduced": ES_s_reduced,
        "scaling": scaling,
        "ES_stressed": ES_stressed_final,
        "stress_start": str(stress_start.date()),
        "stress_end": str(stress_end.date()),
    }


# ---------------------------------------------------------------------------
# Liquidity Horizon Scaling
# ---------------------------------------------------------------------------

LIQUIDITY_HORIZONS = {
    "equity": 10,
    "fx": 10,
    "rates": 20,
    "credit": 40,
    "commodity": 20,
}


def scale_to_liquidity_horizon(
    var_1d: float,
    risk_class: str = "equity",
    lh_override: Optional[int] = None,
) -> float:
    """
    Scale 1-day VaR to the risk-class liquidity horizon.

    Simplified cascade: VaR(LH) = VaR(1d) × √LH

    Full FRTB cascade formula:
        VaR(LH) = √[Σ_j VaR_j² × (LH_j - LH_{j-1})]
    This simplified form is documented as an assumption.
    """
    lh = lh_override or LIQUIDITY_HORIZONS.get(risk_class, 10)
    return var_1d * np.sqrt(lh)


def portfolio_liquidity_adjusted_var(
    component_vars: dict[str, float],
    risk_class_map: dict[str, str],
    lh_config: Optional[dict] = None,
) -> dict[str, float]:
    """
    Compute liquidity-adjusted VaR for each component and total.

    Parameters
    ----------
    component_vars : {factor_name: var_1d}
    risk_class_map : {factor_name: risk_class}
    lh_config      : liquidity horizon overrides

    Returns
    -------
    {factor_name: var_lh, "total": aggregate_var_lh}
    """
    result = {}
    total_sq = 0.0
    for factor, var_1d in component_vars.items():
        rc = risk_class_map.get(factor, "equity")
        lh = (lh_config or {}).get(rc, LIQUIDITY_HORIZONS.get(rc, 10))
        var_lh = var_1d * np.sqrt(lh)
        result[factor] = var_lh
        total_sq += var_lh ** 2   # simple sum of squares (no correlation credit)

    result["total"] = float(np.sqrt(total_sq))
    return result
