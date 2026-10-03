"""
Counterparty Credit Risk (CCR) Exposure modelling.

Monte Carlo simulation for FX forward exposure using Geometric Brownian Motion (GBM).
Calculates Expected Exposure (EE), Expected Positive Exposure (EPE), 
and Potential Future Exposure (PFE).
"""

from __future__ import annotations

import numpy as np


def simulate_fx_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_scenarios: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate FX spot paths using Geometric Brownian Motion.
    
    Parameters
    ----------
    S0          : Initial spot price
    mu          : Drift (r_dom - r_for)
    sigma       : Volatility
    T           : Total time in years
    n_steps     : Number of time steps
    n_scenarios : Number of simulated paths
    
    Returns
    -------
    Array of shape (n_steps + 1, n_scenarios) with simulated spot paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_scenarios))
    paths[0] = S0
    
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_scenarios)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
    return paths


def fx_forward_exposure(
    notional: float,
    K: float,
    S0: float,
    r_dom: float,
    r_for: float,
    sigma: float,
    T: float,
    n_steps: int = 100,
    n_scenarios: int = 5000,
    percentile_pfe: float = 95.0,
) -> dict:
    """
    Calculate CCR exposure metrics for a long FX forward (receive foreign, pay domestic).
    Using simple payoff approximation: max(V(t), 0)
    where V(t) is the MtM of the forward at time t.
    
    Returns
    -------
    dict with time grid, EE profile, PFE profile, and scalar EPE.
    """
    # 1. Simulate Spot paths
    mu = r_dom - r_for
    spot_paths = simulate_fx_gbm(S0, mu, sigma, T, n_steps, n_scenarios)
    
    time_grid = np.linspace(0, T, n_steps + 1)
    
    # 2. Calculate MtM at each time step
    # V(t) = Notional * (S_t * exp(-r_for * (T-t)) - K * exp(-r_dom * (T-t)))
    mtm_paths = np.zeros_like(spot_paths)
    for i, t in enumerate(time_grid):
        time_to_mat = T - t
        discount_for = np.exp(-r_for * time_to_mat)
        discount_dom = np.exp(-r_dom * time_to_mat)
        mtm_paths[i] = notional * (spot_paths[i] * discount_for - K * discount_dom)
        
    # 3. Calculate Exposure (E_t = max(V_t, 0))
    exposure_paths = np.maximum(mtm_paths, 0)
    
    # 4. Calculate metrics
    # Expected Exposure (EE)
    ee_profile = np.mean(exposure_paths, axis=1)
    
    # Potential Future Exposure (PFE)
    pfe_profile = np.percentile(exposure_paths, percentile_pfe, axis=1)
    
    # Expected Positive Exposure (EPE) - time average of EE
    epe = np.mean(ee_profile)
    
    return {
        "time_grid": time_grid.tolist(),
        "ee_profile": ee_profile.tolist(),
        "pfe_profile": pfe_profile.tolist(),
        "epe": float(epe),
        "max_pfe": float(np.max(pfe_profile)),
        "percentile_pfe": percentile_pfe
    }
