"""
FRTB P&L Attribution Test (PLAT).
Implements the exact BCBS thresholds for Spearman Correlation and Kolmogorov-Smirnov (KS) test
to compare Risk-Theoretical P&L (RTPL) against Hypothetical P&L (HPL).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp

class PLATest:
    def __init__(self):
        """
        BCBS FRTB Thresholds:
        Green: Spearman >= 0.80 and KS <= 0.09
        Amber: (0.70 <= Spearman < 0.80) or (0.09 < KS <= 0.12)
        Red: Spearman < 0.70 or KS > 0.12
        """
        self.spearman_green = 0.80
        self.spearman_amber = 0.70
        self.ks_green = 0.09
        self.ks_amber = 0.12

    def run(self, hpl: pd.Series, rtpl: pd.Series):
        """
        Run the P&L Attribution test on a time series of HPL and RTPL.
        Returns the metrics and the traffic light status (Green, Amber, Red).
        """
        # Align series and drop NaNs
        df = pd.concat([hpl, rtpl], axis=1).dropna()
        if len(df) < 30:
            return {"error": "Not enough data points for reliable PLAT (need >= 30)."}

        hpl_vals = df.iloc[:, 0].values
        rtpl_vals = df.iloc[:, 1].values

        # 1. Spearman Correlation
        spearman_corr, spearman_p = spearmanr(hpl_vals, rtpl_vals)

        # 2. Kolmogorov-Smirnov Test (on the distributions)
        ks_stat, ks_p = ks_2samp(hpl_vals, rtpl_vals)

        # Determine zone
        if spearman_corr >= self.spearman_green and ks_stat <= self.ks_green:
            zone = "Green"
        elif spearman_corr < self.spearman_amber or ks_stat > self.ks_amber:
            zone = "Red"
        else:
            zone = "Amber"

        return {
            "spearman_correlation": float(spearman_corr),
            "ks_statistic": float(ks_stat),
            "zone": zone,
            "observations": len(df)
        }
