"""
Bond pricing: DCF, duration, convexity, DV01.

Instruments
-----------
Government of India G-sec with semi-annual coupons.
Constant maturity bond (rolled to maintain tenor at each period-end).

Assumptions (documented in MDD)
---------------------------------
- Flat yield curve per day (YTM pricing); no term-structure fitted.
- Semi-annual coupon, ACT/365 day count.
- No credit spread over risk-free curve (G-sec ≈ risk-free for INR).
- Convexity adjustment included; higher-order terms ignored.

References
----------
Fabozzi, Fixed Income Mathematics (4th ed.).
"""

from __future__ import annotations

import numpy as np


def bond_price(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    DCF bond price.

    Parameters
    ----------
    face         : face/par value
    coupon_rate  : annual coupon rate (decimal)
    ytm          : yield to maturity (decimal, annualised)
    maturity_years: time to maturity in years
    frequency    : coupon payments per year (2 = semi-annual)

    Returns
    -------
    Clean bond price (full price ≈ clean price at coupon date)
    """
    n = int(round(maturity_years * frequency))
    if n == 0:
        return face
    coupon = face * coupon_rate / frequency
    y = ytm / frequency
    t = np.arange(1, n + 1)
    pv_coupons = np.sum(coupon / (1 + y) ** t)
    pv_face = face / (1 + y) ** n
    return pv_coupons + pv_face


def modified_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Modified duration (years)."""
    n = int(round(maturity_years * frequency))
    if n == 0:
        return 0.0
    coupon = face * coupon_rate / frequency
    y = ytm / frequency
    t = np.arange(1, n + 1)
    pv_coupons = coupon / (1 + y) ** t
    pv_face = face / (1 + y) ** n
    pv_all = np.append(pv_coupons, pv_face)
    times = np.append(t / frequency, n / frequency)
    price = bond_price(face, coupon_rate, ytm, maturity_years, frequency)
    mac_dur = np.sum(times * pv_all) / price
    mod_dur = mac_dur / (1 + y)
    return mod_dur


def convexity(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Bond convexity (dimensionless)."""
    n = int(round(maturity_years * frequency))
    if n == 0:
        return 0.0
    coupon = face * coupon_rate / frequency
    y = ytm / frequency
    t = np.arange(1, n + 1)
    price = bond_price(face, coupon_rate, ytm, maturity_years, frequency)
    conv = 0.0
    for i, ti in enumerate(t):
        cf = coupon if i < n - 1 else coupon + face
        conv += cf * ti * (ti + 1) / (1 + y) ** (ti + 2)
    conv /= price * frequency**2
    return conv


def dv01(
    face: float,
    coupon_rate: float,
    ytm: float,
    maturity_years: float,
    frequency: int = 2,
    bump_bps: float = 1.0,
) -> float:
    """
    DV01: price change per 1bp parallel shift in yield.
    Negative for long bond (price falls when yield rises).
    """
    p_base = bond_price(face, coupon_rate, ytm, maturity_years, frequency)
    p_bumped = bond_price(face, coupon_rate, ytm + bump_bps / 10000, maturity_years, frequency)
    return p_bumped - p_base


def bond_pnl(
    face: float,
    coupon_rate: float,
    ytm_old: float,
    ytm_new: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Daily MtM P&L from yield change (full revaluation)."""
    p_old = bond_price(face, coupon_rate, ytm_old, maturity_years, frequency)
    p_new = bond_price(face, coupon_rate, ytm_new, maturity_years, frequency)
    return p_new - p_old


def bond_pnl_delta_gamma(
    face: float,
    coupon_rate: float,
    ytm: float,
    delta_ytm: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """
    Delta-gamma approximation of bond P&L.

    P&L ≈ -Duration × P × Δy + 0.5 × Convexity × P × Δy²
    """
    p = bond_price(face, coupon_rate, ytm, maturity_years, frequency)
    d = modified_duration(face, coupon_rate, ytm, maturity_years, frequency)
    c = convexity(face, coupon_rate, ytm, maturity_years, frequency)
    return -d * p * delta_ytm + 0.5 * c * p * delta_ytm**2
