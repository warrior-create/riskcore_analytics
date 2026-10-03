"""
Hull-White 1-Factor model for Counterparty Credit Risk (EE/EPE/PFE).
Simulates interest rate paths to generate realistic exposure profiles for Swaps.
"""
import numpy as np

class HullWhite1F:
    def __init__(self, a=0.05, sigma=0.01):
        """
        a: mean reversion speed
        sigma: volatility
        """
        self.a = a
        self.sigma = sigma

    def simulate_paths(self, r0, T, dt, n_paths, seed=42):
        """
        Simulate short rate paths using exact discretisation.
        dr_t = (theta_t - a * r_t)dt + sigma * dW_t
        Simplified here using constant theta=0.03 for illustration.
        """
        np.random.seed(seed)
        n_steps = int(T / dt)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0
        
        theta = 0.03
        
        # Exact solution for constant theta
        # r(t+dt) = r(t)*exp(-a*dt) + theta*(1-exp(-a*dt)) + sigma * sqrt((1-exp(-2*a*dt))/(2*a)) * Z
        decay = np.exp(-self.a * dt)
        drift = theta * (1 - decay)
        std = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * decay + drift + std * Z
            
        return paths

    def calculate_swap_exposure(self, rate_paths, notional, strike, dt):
        """
        Calculate EE, EPE, PFE for a simplified payer swap.
        Assume payoff is max(0, r - strike) * notional * dt at each step.
        """
        n_paths, n_steps = rate_paths.shape
        
        # Exposure matrix (max(V, 0))
        exposures = np.maximum(0, (rate_paths - strike) * notional)
        
        # Expected Exposure (EE)
        ee = np.mean(exposures, axis=0)
        
        # Expected Positive Exposure (EPE) - time average of EE
        epe = np.mean(ee)
        
        # Potential Future Exposure (PFE) at 99%
        pfe = np.percentile(exposures, 99, axis=0)
        
        return {
            "ee": ee,
            "epe": epe,
            "pfe": pfe,
            "time_grid": np.arange(n_steps) * dt
        }
