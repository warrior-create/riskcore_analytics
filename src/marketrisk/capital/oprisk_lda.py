"""
Operational Risk Economic Capital using Loss Distribution Approach (LDA).

Model: Poisson-Lognormal LDA
Frequency of loss events: Poisson distribution
Severity of loss events: Lognormal distribution
Economic Capital is calculated at the 99.9% confidence level over 1 year.
"""

from __future__ import annotations

import numpy as np


def oprisk_lda_monte_carlo(
    lambda_freq: float,
    mu_sev: float,
    sigma_sev: float,
    n_scenarios: int = 100000,
    confidence: float = 0.999,
    seed: int = 42,
) -> dict:
    """
    Compute Operational Risk Economic Capital via Monte Carlo LDA.

    Parameters
    ----------
    lambda_freq : Expected number of loss events per year (Poisson lambda).
    mu_sev      : Mean of the underlying normal distribution for severity.
    sigma_sev   : Standard deviation of the underlying normal distribution for severity.
    n_scenarios : Number of Monte Carlo years to simulate.
    confidence  : Confidence level for the Economic Capital.
    seed        : Random seed for reproducibility.

    Returns
    -------
    dict with Expected Loss, Unexpected Loss (EC), and VaR at the given confidence.
    """
    rng = np.random.default_rng(seed)

    # 1. Simulate the number of loss events for each scenario (year)
    N_events = rng.poisson(lambda_freq, n_scenarios)

    # 2. Simulate the total loss for each scenario
    total_losses = np.zeros(n_scenarios)

    # To avoid looping over scenarios, we can simulate all severities at once
    # Total number of events across all scenarios
    total_num_events = np.sum(N_events)
    
    if total_num_events > 0:
        # Simulate all severities
        severities = rng.lognormal(mu_sev, sigma_sev, total_num_events)
        
        # Group severities back into scenarios
        # We can use np.split or bincount. Bincount is very efficient.
        # Create an array of scenario indices for each event
        scenario_indices = np.repeat(np.arange(n_scenarios), N_events)
        total_losses = np.bincount(scenario_indices, weights=severities, minlength=n_scenarios)

    # Calculate metrics
    expected_loss = np.mean(total_losses)
    var = np.quantile(total_losses, confidence)
    unexpected_loss = var - expected_loss  # Economic Capital

    return {
        "model": "Poisson-Lognormal LDA",
        "expected_loss": float(expected_loss),
        "oprisk_var": float(var),
        "oprisk_ec": float(unexpected_loss),
        "confidence": confidence,
        "n_scenarios": n_scenarios,
        "note": "Illustrative inputs. Not regulatory-approved.",
    }
