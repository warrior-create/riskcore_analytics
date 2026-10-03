"""
Interest Rate Swap (IRS) pricing and curve bootstrapping.

Implements standard OIS-style IRS pricing and simple bootstrapping.
"""

from __future__ import annotations

import numpy as np


class YieldCurve:
    """Simple linearly interpolated yield curve."""
    def __init__(self, tenors: list[float], rates: list[float]):
        self.tenors = np.array(tenors)
        self.rates = np.array(rates)
        
    def get_rate(self, t: float) -> float:
        """Interpolate zero rate for maturity t."""
        return float(np.interp(t, self.tenors, self.rates))
        
    def get_discount_factor(self, t: float) -> float:
        """Calculate discount factor for maturity t."""
        r = self.get_rate(t)
        return float(np.exp(-r * t))


def price_irs(
    notional: float,
    fixed_rate: float,
    tenor_years: float,
    payment_freq: int,
    yield_curve: YieldCurve,
    is_payer: bool = True,
) -> dict:
    """
    Price a plain vanilla Interest Rate Swap (IRS).
    
    Parameters
    ----------
    notional     : Swap notional
    fixed_rate   : Fixed rate paid/received
    tenor_years  : Swap maturity in years
    payment_freq : Number of payments per year (e.g., 2 for semi-annual)
    yield_curve  : YieldCurve object for discounting
    is_payer     : True if paying fixed, False if receiving fixed
    
    Returns
    -------
    dict with swap NPV, fixed leg PV, float leg PV, and DV01.
    """
    dt = 1.0 / payment_freq
    n_payments = int(tenor_years * payment_freq)
    
    fixed_leg_pv = 0.0
    float_leg_pv = 0.0
    
    # We assume the floating leg prices at par if we are at a reset date.
    # Therefore, PV(Float Leg) = Notional * (1 - DF(T))
    
    df_T = yield_curve.get_discount_factor(tenor_years)
    float_leg_pv = notional * (1.0 - df_T)
    
    # Calculate Fixed Leg PV
    pv01 = 0.0
    for i in range(1, n_payments + 1):
        t = i * dt
        df = yield_curve.get_discount_factor(t)
        pv01 += df * dt
        
    fixed_leg_pv = notional * fixed_rate * pv01
    
    if is_payer:
        npv = float_leg_pv - fixed_leg_pv
        dv01 = -pv01 * notional / 10000  # PV01 is per 1 unit of rate, DV01 is per 1 bps
    else:
        npv = fixed_leg_pv - float_leg_pv
        dv01 = pv01 * notional / 10000
        
    return {
        "npv": float(npv),
        "fixed_leg_pv": float(fixed_leg_pv),
        "float_leg_pv": float(float_leg_pv),
        "dv01": float(dv01),
        "pv01": float(pv01 * notional)
    }
