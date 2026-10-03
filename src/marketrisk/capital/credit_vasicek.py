"""
Vasicek single-factor credit model for economic capital.

Credit EC = Vasicek asymptotic single-factor model at 99.9% confidence.
Illustrative inputs — actual credit data not available.

References: Vasicek (1991, 2002), Basel II IRB formula.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def vasicek_loss_quantile(
    pd_: float,
    lgd: float,
    rho: float,
    confidence: float = 0.999,
    ead: float = 1.0,
) -> float:
    """
    Vasicek asymptotic single-factor model (Basel IRB formula).

    Parameters
    ----------
    pd_        : probability of default
    lgd        : loss given default (decimal)
    rho        : asset correlation
    confidence : confidence level (99.9% for regulatory capital)
    ead        : exposure at default (face value)

    Returns
    -------
    Credit loss quantile (= credit economic capital per unit EAD)
    """
    if pd_ <= 0 or pd_ >= 1:
        raise ValueError(f"PD must be in (0,1), got {pd_}")
    if lgd <= 0 or lgd >= 1:
        raise ValueError(f"LGD must be in (0,1), got {lgd}")
    if rho <= 0 or rho >= 1:
        raise ValueError(f"rho must be in (0,1), got {rho}")

    n_pd = norm.ppf(pd_)
    n_conf = norm.ppf(confidence)

    conditional_pd = norm.cdf(
        (n_pd + np.sqrt(rho) * n_conf) / np.sqrt(1 - rho)
    )
    credit_ec = ead * lgd * conditional_pd
    return float(credit_ec)


def portfolio_credit_ec(
    exposures: list[dict],
    rho: float = 0.20,
    confidence: float = 0.999,
) -> dict:
    """
    Compute credit EC for a portfolio of exposures.

    Each exposure dict: {name, ead, pd, lgd}
    """
    total_ec = 0.0
    details = []
    for exp in exposures:
        ec = vasicek_loss_quantile(
            pd_=exp["pd"],
            lgd=exp["lgd"],
            rho=rho,
            confidence=confidence,
            ead=exp["ead"],
        )
        total_ec += ec
        details.append({**exp, "credit_ec": ec})

    return {
        "total_credit_ec": total_ec,
        "rho": rho,
        "confidence": confidence,
        "details": details,
        "note": "Illustrative inputs. Not regulatory-approved.",
    }
