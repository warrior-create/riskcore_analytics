"""
Black-76 options pricing and Greeks for European options on Nifty 50.

Assumptions
-----------
- Nifty options are European (no early exercise) → Black-76 is exact.
- Implied vol = India VIX / 100 × skew_slope per moneyness bucket (flat per day).
- Forward = Spot × exp((r_inr - q) × T), where q = dividend yield (constant).
- No vol-surface smile interpolation in v1 (documented limitation).

References
----------
Black (1976), "The Pricing of Commodity Contracts", JFE.
Hull, Options, Futures and Other Derivatives, 10th ed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Documented constant: Nifty 50 historical dividend yield ≈ 1.2% p.a.
NIFTY_DIV_YIELD = 0.012

# Skew slope: linear approximation, calibrated illustratively
# Put wing: sigma(K) = sigma_atm + SKEW_SLOPE × (1 - K/F)
# This is a stated simplification — see limitations in MDD
SKEW_SLOPE = 0.10


def black76_price(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-76 option price.

    Parameters
    ----------
    F : forward price
    K : strike
    T : time to expiry (years)
    r : risk-free rate (INR)
    sigma : implied volatility (annualised)
    option_type : 'call' or 'put'

    Returns
    -------
    option price (same currency as F and K)
    """
    if T <= 0:
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return intrinsic * np.exp(-r * T)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    discount = np.exp(-r * T)
    if option_type == "call":
        return discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Black-76 Greeks.

    Returns
    -------
    dict with keys: price, delta, gamma, vega, theta, rho
    """
    if T <= 0:
        intrinsic = max(F - K, 0) if option_type == "call" else max(K - F, 0)
        return {
            "price": intrinsic * np.exp(-r * max(T, 0)),
            "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
        }

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    nd1 = norm.pdf(d1)

    price = black76_price(F, K, T, r, sigma, option_type)

    # Delta w.r.t. forward (not spot)
    if option_type == "call":
        delta = discount * norm.cdf(d1)
        rho_ = -T * price   # rho w.r.t. r in Black-76
    else:
        delta = -discount * norm.cdf(-d1)
        rho_ = -T * price

    gamma = discount * nd1 / (F * sigma * np.sqrt(T))
    vega = F * discount * nd1 * np.sqrt(T)              # per unit vol (not per 1%)
    theta = (
        -F * sigma * discount * nd1 / (2 * np.sqrt(T))
        - r * price
    )

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,   # normalise: per 1 vol point (like Hull)
        "theta": theta / 252, # per calendar day
        "rho": rho_,
    }


def implied_vol_with_skew(
    sigma_atm: float,
    F: float,
    K: float,
    skew_slope: float = SKEW_SLOPE,
) -> float:
    """
    Simple linear skew model: sigma(K) = sigma_atm + slope × (1 - K/F).

    Limitation: flat vol surface within a day; no term structure.
    Documented in MDD as known simplification.
    """
    moneyness = K / F   # < 1 → OTM put, > 1 → OTM call
    return float(np.clip(sigma_atm + skew_slope * (1.0 - moneyness), 0.01, 5.0))


def nifty_forward(
    spot: float,
    r_inr: float,
    T: float,
    div_yield: float = NIFTY_DIV_YIELD,
) -> float:
    """Cost-of-carry forward: F = S × exp((r - q) × T)."""
    return spot * np.exp((r_inr - div_yield) * T)


def put_call_parity_check(
    call_price: float,
    put_price: float,
    F: float,
    K: float,
    r: float,
    T: float,
    tol: float = 0.01,
) -> bool:
    """
    Verify put-call parity: C - P = disc × (F - K).
    Returns True if satisfied within tol (in price units).
    """
    lhs = call_price - put_price
    rhs = np.exp(-r * T) * (F - K)
    return abs(lhs - rhs) < tol


def full_revaluation_pnl(
    portfolio_options: list[dict],
    spot_old: float,
    spot_new: float,
    vix_old: float,
    vix_new: float,
    r: float,
    T_old: float,
    dt_years: float = 1 / 252,
) -> float:
    """
    Full revaluation P&L for a list of options.

    Each option dict: {type, K, contracts, sigma_skew_slope}
    where sigma is sourced from VIX / 100.
    """
    total = 0.0
    for opt in portfolio_options:
        K = opt["K"]
        otype = opt["option_type"]
        contracts = opt["contracts"]
        lot_size = opt.get("lot_size", 50)   # Nifty lot = 50

        T_new = max(T_old - dt_years, 1e-6)
        sigma_old = implied_vol_with_skew(vix_old / 100, nifty_forward(spot_old, r, T_old), K)
        sigma_new = implied_vol_with_skew(vix_new / 100, nifty_forward(spot_new, r, T_new), K)

        F_old = nifty_forward(spot_old, r, T_old)
        F_new = nifty_forward(spot_new, r, T_new)

        p_old = black76_price(F_old, K, T_old, r, sigma_old, otype)
        p_new = black76_price(F_new, K, T_new, r, sigma_new, otype)

        total += contracts * lot_size * (p_new - p_old)
    return total
