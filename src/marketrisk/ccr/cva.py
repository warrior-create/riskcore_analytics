"""
Credit Value Adjustment (CVA) calculation.

Basic CVA integration using an Expected Exposure (EE) profile,
a constant hazard rate (implied from CDS or assumed PD),
and a fixed Recovery Rate (RR).
"""

from __future__ import annotations

import numpy as np


def calculate_cva(
    time_grid: list[float],
    ee_profile: list[float],
    hazard_rate: float,
    recovery_rate: float,
    r_df: float,
) -> float:
    """
    Calculate unilateral CVA for a given Expected Exposure profile.
    
    CVA = (1 - R) * sum_i [ DF(t_i) * EE(t_i) * PD(t_{i-1}, t_i) ]
    
    Parameters
    ----------
    time_grid     : List of time points in years (must start at 0).
    ee_profile    : List of Expected Exposure values at each time point.
    hazard_rate   : Constant hazard rate lambda.
    recovery_rate : Recovery rate (e.g., 0.40).
    r_df          : Risk-free rate for discounting.
    
    Returns
    -------
    CVA amount.
    """
    t = np.array(time_grid)
    ee = np.array(ee_profile)
    
    if len(t) < 2:
        return 0.0
        
    cva = 0.0
    lgd = 1.0 - recovery_rate
    
    for i in range(1, len(t)):
        t_prev = t[i-1]
        t_curr = t[i]
        
        # Mid-point approximation for EE and DF
        t_mid = 0.5 * (t_prev + t_curr)
        ee_mid = 0.5 * (ee[i-1] + ee[i])
        df = np.exp(-r_df * t_mid)
        
        # Default probability in the interval (t_prev, t_curr)
        # PD = exp(-lambda * t_prev) - exp(-lambda * t_curr)
        pd_interval = np.exp(-hazard_rate * t_prev) - np.exp(-hazard_rate * t_curr)
        
        cva += df * ee_mid * pd_interval
        
    return lgd * cva
