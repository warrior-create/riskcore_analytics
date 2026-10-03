"""
Nelson-Siegel-Svensson (NSS) Yield Curve Bootstrapping.
"""
import numpy as np
from scipy.optimize import minimize

class NSSYieldCurve:
    def __init__(self):
        # Parameters: beta0, beta1, beta2, beta3, tau1, tau2
        self.params = np.array([0.03, -0.01, 0.01, 0.01, 1.0, 2.0])

    def nss_curve(self, tau, params):
        """Calculate NSS yield for maturity tau."""
        b0, b1, b2, b3, t1, t2 = params
        
        # Handle tau = 0
        tau = np.maximum(tau, 1e-6)
        
        term1 = (1 - np.exp(-tau / t1)) / (tau / t1)
        term2 = term1 - np.exp(-tau / t1)
        term3 = ((1 - np.exp(-tau / t2)) / (tau / t2)) - np.exp(-tau / t2)
        
        return b0 + b1 * term1 + b2 * term2 + b3 * term3

    def fit(self, maturities, yields):
        """Fit NSS parameters to market yields using least squares."""
        maturities = np.array(maturities)
        yields = np.array(yields)
        
        def objective(params):
            predicted = self.nss_curve(maturities, params)
            return np.sum((predicted - yields) ** 2)
            
        bounds = (
            (0.0, 0.20),      # beta0: long term yield
            (-0.20, 0.20),    # beta1: short term slope
            (-0.20, 0.20),    # beta2: medium term curvature 1
            (-0.20, 0.20),    # beta3: medium term curvature 2
            (0.1, 5.0),       # tau1: decay rate 1
            (1.0, 20.0)       # tau2: decay rate 2
        )
        
        result = minimize(
            objective, 
            self.params, 
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            self.params = result.x
        return result.success

    def get_yield(self, maturity):
        """Get the bootstrapped yield for a specific maturity."""
        return self.nss_curve(maturity, self.params)

    def get_discount_factor(self, maturity):
        """Get the discount factor for a specific maturity."""
        y = self.get_yield(maturity)
        return np.exp(-y * maturity)
