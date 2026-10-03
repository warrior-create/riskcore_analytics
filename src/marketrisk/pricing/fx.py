"""
FX forward pricing using Covered Interest Rate Parity (CIP).

Theory
------
F_USDINR(T) = S_USDINR × exp((r_INR - r_USD) × T)

where:
  S      = spot USDINR
  r_INR  = INR risk-free rate (from synthetic yield curve)
  r_USD  = USD LIBOR/SOFR (from FRED)
  T      = tenor in years

Mark-to-market P&L (daily):
  PnL = N_USD × (F_new - F_old)

Assumptions (documented in MDD):
  - CIP holds continuously; bid-ask and basis swap ignored.
  - r_INR from synthetic 3-point curve, linearly interpolated.
  - Rolling monthly: at each month-end we re-enter at the prevailing F.
"""

from __future__ import annotations

import numpy as np


def fx_forward_rate(
    spot: float,
    r_inr: float,
    r_usd: float,
    tenor_years: float,
) -> float:
    """
    Covered Interest Parity forward rate.

    Parameters
    ----------
    spot        : USDINR spot rate
    r_inr       : INR risk-free rate (decimal, annualised)
    r_usd       : USD risk-free rate (decimal, annualised)
    tenor_years : forward tenor

    Returns
    -------
    USDINR forward rate
    """
    return spot * np.exp((r_inr - r_usd) * tenor_years)


def fx_forward_pnl(
    notional_usd: float,
    forward_old: float,
    forward_new: float,
    direction: int = 1,   # 1=long USD, -1=short USD
) -> float:
    """
    Daily mark-to-market P&L on an FX forward position.

    Returns P&L in INR.
    """
    return direction * notional_usd * (forward_new - forward_old)


def fx_forward_dv01(
    notional_usd: float,
    spot: float,
    r_inr: float,
    r_usd: float,
    tenor_years: float,
    bump_bps: float = 1.0,
) -> float:
    """
    DV01 of the FX forward w.r.t. INR rate (INR per 1bp shift).

    Finite difference: bump r_INR by bump_bps.
    """
    F_base = fx_forward_rate(spot, r_inr, r_usd, tenor_years)
    F_bumped = fx_forward_rate(spot, r_inr + bump_bps / 10000, r_usd, tenor_years)
    return notional_usd * (F_bumped - F_base)


def fx_delta(notional_usd: float, spot: float) -> float:
    """Spot delta: sensitivity of INR MtM to 1 INR move in spot."""
    return notional_usd  # linear: dV/dS = N_USD (for forward long USD)
