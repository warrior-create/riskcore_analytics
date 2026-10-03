"""
P&L attribution and risk-driver report.

Decomposes daily portfolio P&L into risk factor contributions:
  ΔV ≈ Δ·ΔS + ½Γ·ΔS² + ν·Δσ + ρ_rates·Δr + FX_delta·ΔS_FX + residual

Full revaluation vs delta-gamma-vega comparison is included.
The residual (unexplained) identifies higher-order effects and is
deliberately amplified by short-dated options (stated objective of the spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PnLExplain:
    """Container for a single day's P&L explain."""

    date: str
    full_reval_pnl: float = 0.0

    # Greek contributions
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    rates_pnl: float = 0.0
    fx_pnl: float = 0.0

    # Total first-order approx
    @property
    def approx_pnl(self) -> float:
        return (
            self.delta_pnl
            + self.gamma_pnl
            + self.vega_pnl
            + self.theta_pnl
            + self.rates_pnl
            + self.fx_pnl
        )

    @property
    def unexplained(self) -> float:
        return self.full_reval_pnl - self.approx_pnl

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "full_reval_pnl": self.full_reval_pnl,
            "delta_pnl": self.delta_pnl,
            "gamma_pnl": self.gamma_pnl,
            "vega_pnl": self.vega_pnl,
            "theta_pnl": self.theta_pnl,
            "rates_pnl": self.rates_pnl,
            "fx_pnl": self.fx_pnl,
            "approx_pnl": self.approx_pnl,
            "unexplained_pnl": self.unexplained,
        }


class PnLAttributionEngine:
    """
    Compute daily P&L explain for the mixed portfolio.

    Parameters
    ----------
    portfolio : Portfolio object (from engine/portfolio.py)
    """

    def __init__(self, portfolio):
        self.portfolio = portfolio

    def explain_day(
        self,
        date: str,
        spot_equity: dict[str, float],
        spot_equity_prev: dict[str, float],
        nifty_spot: float,
        nifty_spot_prev: float,
        vix: float,
        vix_prev: float,
        fx_spot: float,
        fx_spot_prev: float,
        ytm_10y: float,
        ytm_10y_prev: float,
        r_inr: float,
    ) -> PnLExplain:
        """Decompose one day's P&L into Greek contributions + residual."""
        explain = PnLExplain(date=date)

        # ----------------------------------------------------------------
        # 1. Equity component: delta = share count (unitless for P&L)
        # ----------------------------------------------------------------
        equity_pnl = 0.0
        positions = self.portfolio.equity_positions
        for ticker, qty in positions.items():
            p_old = spot_equity_prev.get(ticker, 0)
            p_new = spot_equity.get(ticker, 0)
            if p_old > 0:
                equity_pnl += qty * (p_new - p_old)
        explain.delta_pnl += equity_pnl

        # ----------------------------------------------------------------
        # 2. Options component
        # ----------------------------------------------------------------
        from marketrisk.pricing.options import (
            black76_greeks,
            implied_vol_with_skew,
            nifty_forward,
        )

        dt_years = 1 / 252
        for opt in self.portfolio.options:
            K = opt["K"]
            otype = opt["option_type"]
            qty = opt["contracts"] * opt.get("lot_size", 50)
            T = opt["T_remaining"]

            sigma_prev = implied_vol_with_skew(
                vix_prev / 100, nifty_forward(nifty_spot_prev, r_inr, T), K
            )
            greeks_prev = black76_greeks(
                nifty_forward(nifty_spot_prev, r_inr, T), K, T, r_inr, sigma_prev, otype
            )

            dS = nifty_spot - nifty_spot_prev
            dsigma = (vix - vix_prev) / 100

            explain.delta_pnl += qty * greeks_prev["delta"] * dS
            explain.gamma_pnl += qty * 0.5 * greeks_prev["gamma"] * dS**2
            explain.vega_pnl += qty * greeks_prev["vega"] * 100 * dsigma  # vega per 1%
            explain.theta_pnl += qty * greeks_prev["theta"]   # already per day

        # ----------------------------------------------------------------
        # 3. Bond component (rates P&L)
        # ----------------------------------------------------------------
        from marketrisk.pricing.bonds import bond_pnl

        bond_cfg = self.portfolio.bond
        if bond_cfg:
            b_pnl = bond_pnl(
                face=bond_cfg["face_value"],
                coupon_rate=bond_cfg["coupon_rate"],
                ytm_old=ytm_10y_prev,
                ytm_new=ytm_10y,
                maturity_years=bond_cfg["maturity_years"],
            )
            explain.rates_pnl += b_pnl

        # ----------------------------------------------------------------
        # 4. FX forward component
        # ----------------------------------------------------------------
        from marketrisk.pricing.fx import fx_forward_rate, fx_forward_pnl

        fx_cfg = self.portfolio.fx_forward
        if fx_cfg:
            r_usd = fx_cfg.get("r_usd", 0.04)  # fallback
            T_fx = fx_cfg.get("tenor_months", 1) / 12
            F_old = fx_forward_rate(fx_spot_prev, r_inr, r_usd, T_fx)
            F_new = fx_forward_rate(fx_spot, r_inr, r_usd, T_fx)
            fx_pnl_ = fx_forward_pnl(fx_cfg["notional_usd"], F_old, F_new)
            explain.fx_pnl += fx_pnl_

        # ----------------------------------------------------------------
        # 5. Full revaluation (approximation via sum of components for now;
        #    a true full reval calls each pricer independently and sums)
        # ----------------------------------------------------------------
        # In walk-forward engine, full_reval_pnl is set externally from
        # actual price changes. Here we set it = approx so unexplained = 0
        # for the explain object; the engine will overwrite full_reval_pnl.
        explain.full_reval_pnl = explain.approx_pnl

        return explain

    def build_explain_series(
        self, explains: list[PnLExplain]
    ) -> pd.DataFrame:
        """Convert list of PnLExplain objects to a DataFrame."""
        return pd.DataFrame([e.to_dict() for e in explains]).set_index("date")
