"""
SIMM-lite: simplified Initial Margin calculation.

Covers delta margin for IR, FX and Equity buckets using
ISDA SIMM-like methodology (published risk weights and correlations).

Documented simplifications:
  - Single bucket per asset class (no sub-bucket splits)
  - No curvature margin
  - No vega margin for options
  - Risk weights are indicative (as of 2023 ISDA parameters)

For the full regulatory SIMM, use ISDA's published model.
"""

from __future__ import annotations

import numpy as np

# Indicative risk weights (% of notional)
RISK_WEIGHTS = {
    "equity": 0.30,   # 30% — illustrative Tier 1 equity weight
    "fx": 0.075,      # 7.5% — major EM currency
    "ir": 0.005,      # 0.5% — medium tenor (5Y)
}

# Intra-asset class correlations
CORRELATIONS = {
    ("equity", "fx"): 0.25,
    ("equity", "ir"): 0.15,
    ("fx", "ir"): 0.20,
}


def delta_margin(
    sensitivities: dict[str, float],
    asset_class: str,
) -> float:
    """
    Delta IM for a single asset class.

    Parameters
    ----------
    sensitivities : {risk_factor: delta_sensitivity}
    asset_class   : 'equity' | 'fx' | 'ir'

    Returns
    -------
    Delta initial margin (same units as sensitivities)
    """
    rw = RISK_WEIGHTS.get(asset_class, 0.15)
    ws = np.array(list(sensitivities.values())) * rw
    # Intra-bucket correlation assumed 0.99 (same bucket for simplicity)
    rho = 0.99
    n = len(ws)
    cov = np.eye(n) * (1 - rho) + rho * np.ones((n, n))
    im = float(np.sqrt(ws @ cov @ ws))
    return im


def portfolio_simm_im(
    equity_sensitivities: dict[str, float],
    fx_sensitivities: dict[str, float],
    ir_sensitivities: dict[str, float],
) -> dict:
    """
    Simplified SIMM total IM aggregating across asset classes.

    Returns
    -------
    dict with IM by asset class and total
    """
    im_eq = delta_margin(equity_sensitivities, "equity") if equity_sensitivities else 0.0
    im_fx = delta_margin(fx_sensitivities, "fx") if fx_sensitivities else 0.0
    im_ir = delta_margin(ir_sensitivities, "ir") if ir_sensitivities else 0.0

    ims = np.array([im_eq, im_fx, im_ir])
    classes = ["equity", "fx", "ir"]
    n = len(ims)
    corr_matrix = np.eye(n)
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i != j:
                corr_matrix[i, j] = CORRELATIONS.get((c1, c2), CORRELATIONS.get((c2, c1), 0.0))

    total_im = float(np.sqrt(ims @ corr_matrix @ ims))

    return {
        "equity_im": im_eq,
        "fx_im": im_fx,
        "ir_im": im_ir,
        "total_im": total_im,
        "note": "SIMM-lite: simplified implementation. See docs for limitations.",
    }
