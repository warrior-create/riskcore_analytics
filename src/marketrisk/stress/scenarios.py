"""
Stress scenarios module.

Implements historical and hypothetical stress tests, plus reverse stress.

Historical scenarios
--------------------
2008 GFC, 2013 Taper Tantrum, 2016 Demonetisation,
2018 IL&FS, 2020 COVID, 2022 Rate Hikes

Hypothetical scenarios
----------------------
- Nifty -20%
- USDINR +10%
- Rates +200bps
- Vol spike (VIX ×3)

Reverse stress
--------------
Find the minimum simultaneous set of factor shocks that produces
a portfolio loss exceeding a target (e.g. 3% of MV).
Solved via scipy.optimize.minimize.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from marketrisk.pricing.bonds import bond_pnl
from marketrisk.pricing.fx import fx_forward_rate, fx_forward_pnl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

HISTORICAL_SCENARIOS = {
    "2008_GFC": {
        "nifty_shock": -0.55,
        "usdinr_shock": +0.20,
        "rate_shock_bps": +150,
        "vix_multiplier": 4.0,
        "description": "2008 Global Financial Crisis (Sep–Nov 2008 peak)",
    },
    "2013_Taper": {
        "nifty_shock": -0.18,
        "usdinr_shock": +0.12,
        "rate_shock_bps": +80,
        "vix_multiplier": 2.5,
        "description": "2013 Taper Tantrum (May–Aug 2013)",
    },
    "2016_Demonetisation": {
        "nifty_shock": -0.10,
        "usdinr_shock": +0.02,
        "rate_shock_bps": -25,
        "vix_multiplier": 1.8,
        "description": "2016 Indian Demonetisation (Nov 2016)",
    },
    "2018_ILFS": {
        "nifty_shock": -0.12,
        "usdinr_shock": +0.08,
        "rate_shock_bps": +50,
        "vix_multiplier": 2.0,
        "description": "2018 IL&FS Credit Crisis (Sep–Dec 2018)",
    },
    "2020_COVID": {
        "nifty_shock": -0.38,
        "usdinr_shock": +0.07,
        "rate_shock_bps": -75,
        "vix_multiplier": 6.0,
        "description": "2020 COVID-19 Crash (Feb–Mar 2020)",
    },
    "2022_RateHikes": {
        "nifty_shock": -0.16,
        "usdinr_shock": +0.05,
        "rate_shock_bps": +250,
        "vix_multiplier": 2.0,
        "description": "2022 Global Rate Hike Cycle",
    },
}

HYPOTHETICAL_SCENARIOS = {
    "Nifty_-20pct": {
        "nifty_shock": -0.20,
        "usdinr_shock": 0.0,
        "rate_shock_bps": 0,
        "vix_multiplier": 3.0,
        "description": "Hypothetical: Nifty -20% shock",
    },
    "USDINR_+10pct": {
        "nifty_shock": -0.05,
        "usdinr_shock": +0.10,
        "rate_shock_bps": 0,
        "vix_multiplier": 1.5,
        "description": "Hypothetical: USDINR +10% depreciation",
    },
    "Rates_+200bps": {
        "nifty_shock": -0.08,
        "usdinr_shock": +0.03,
        "rate_shock_bps": 200,
        "vix_multiplier": 1.5,
        "description": "Hypothetical: Parallel +200bps rate shock",
    },
    "Vol_Spike": {
        "nifty_shock": -0.10,
        "usdinr_shock": +0.04,
        "rate_shock_bps": 50,
        "vix_multiplier": 4.0,
        "description": "Hypothetical: VIX ×4 vol spike",
    },
}


@dataclass
class StressResult:
    scenario_name: str
    description: str
    equity_pnl: float
    bond_pnl: float
    fx_pnl: float
    options_pnl: float
    total_pnl: float
    shocks: dict = field(default_factory=dict)


class StressEngine:
    """
    Compute stress P&L for historical and hypothetical scenarios.
    """

    def __init__(self, portfolio_cfg: dict, market_state: dict):
        """
        Parameters
        ----------
        portfolio_cfg : portfolio section of settings.yaml
        market_state  : {nifty_spot, usdinr_spot, vix, ytm_10y, r_inr, r_usd}
        """
        self.port = portfolio_cfg
        self.state = market_state

    def run_all(self) -> pd.DataFrame:
        """Run all historical and hypothetical scenarios."""
        results = []
        all_scenarios = {**HISTORICAL_SCENARIOS, **HYPOTHETICAL_SCENARIOS}
        for name, sc in all_scenarios.items():
            r = self.compute_scenario(name, sc)
            results.append(r)
        return pd.DataFrame([vars(r) for r in results])

    def compute_scenario(self, name: str, scenario: dict) -> StressResult:
        """Compute stressed P&L for one scenario."""
        ms = self.state
        nifty_new = ms["nifty_spot"] * (1 + scenario["nifty_shock"])
        usdinr_new = ms["usdinr_spot"] * (1 + scenario["usdinr_shock"])
        ytm_new = ms["ytm_10y"] + scenario["rate_shock_bps"] / 10000
        vix_new = ms["vix"] * scenario["vix_multiplier"]

        # 1. Equity P&L (linear)
        equity_mv = sum(
            self.port["equities"].get(t, 0) * ms.get("equity_prices", {}).get(t, 1000)
            for t in self.port["equities"]
        )
        eq_pnl = equity_mv * scenario["nifty_shock"]

        # 2. Bond P&L
        bond_cfg = self.port.get("bond", {})
        b_pnl = 0.0
        if bond_cfg:
            b_pnl = bond_pnl(
                face=bond_cfg["face_value"],
                coupon_rate=bond_cfg["coupon_rate"],
                ytm_old=ms["ytm_10y"],
                ytm_new=ytm_new,
                maturity_years=bond_cfg["maturity_years"],
            )

        # 3. FX forward P&L
        fx_cfg = self.port.get("fx_forward", {})
        fx_pnl_ = 0.0
        if fx_cfg:
            r_inr = ms["r_inr"]
            r_usd = ms["r_usd"]
            T_fx = fx_cfg["tenor_months"] / 12
            F_old = fx_forward_rate(ms["usdinr_spot"], r_inr, r_usd, T_fx)
            F_new = fx_forward_rate(usdinr_new, r_inr, r_usd, T_fx)
            fx_pnl_ = fx_forward_pnl(fx_cfg["notional_usd"], F_old, F_new)

        # 4. Options P&L (simplified: delta-gamma approximation)
        from marketrisk.pricing.options import black76_greeks, nifty_forward, implied_vol_with_skew
        opt_pnl = 0.0
        for opt in self.port.get("options", {}).values():
            otype = opt["type"]
            T = opt["maturity_days"] / 252
            contracts = opt["contracts"]
            lot_size = 50
            K = ms["nifty_spot"] * opt["moneyness"]

            sigma_old = implied_vol_with_skew(ms["vix"] / 100, nifty_forward(ms["nifty_spot"], ms["r_inr"], T), K)
            g = black76_greeks(nifty_forward(ms["nifty_spot"], ms["r_inr"], T), K, T, ms["r_inr"], sigma_old, otype)

            dS = nifty_new - ms["nifty_spot"]
            dsigma = (vix_new - ms["vix"]) / 100

            opt_pnl += contracts * lot_size * (
                g["delta"] * dS + 0.5 * g["gamma"] * dS**2 + g["vega"] * 100 * dsigma
            )

        total = eq_pnl + b_pnl + fx_pnl_ + opt_pnl

        return StressResult(
            scenario_name=name,
            description=scenario["description"],
            equity_pnl=eq_pnl,
            bond_pnl=b_pnl,
            fx_pnl=fx_pnl_,
            options_pnl=opt_pnl,
            total_pnl=total,
            shocks=scenario,
        )

    def reverse_stress(
        self,
        target_loss: float,
        method: str = "L-BFGS-B",
    ) -> dict:
        """
        Find the minimum-severity shock vector that causes a loss ≥ target_loss.

        Minimises: ||shocks||₂ subject to loss(shocks) ≥ target_loss.
        """
        def loss_fn(x):
            sc = {
                "nifty_shock": x[0],
                "usdinr_shock": x[1],
                "rate_shock_bps": x[2] * 100,
                "vix_multiplier": max(1.0, 1 + x[3]),
                "description": "Reverse stress",
            }
            result = self.compute_scenario("reverse", sc)
            return -result.total_pnl   # minimise negative loss = maximise loss

        def penalty(x):
            # Penalise if loss < target
            sc = {
                "nifty_shock": x[0],
                "usdinr_shock": x[1],
                "rate_shock_bps": x[2] * 100,
                "vix_multiplier": max(1.0, 1 + x[3]),
                "description": "Reverse stress",
            }
            result = self.compute_scenario("reverse", sc)
            shortfall = max(0, target_loss + result.total_pnl)
            return shortfall

        x0 = np.array([-0.05, 0.02, 0.5, 1.0])
        bounds = [(-0.80, 0.20), (-0.10, 0.30), (0, 5.0), (0, 8.0)]

        from scipy.optimize import minimize as sp_minimize
        res = sp_minimize(
            lambda x: np.sum(x**2) + 1000 * penalty(x),
            x0,
            method=method,
            bounds=bounds,
        )

        x_opt = res.x
        return {
            "nifty_shock": float(x_opt[0]),
            "usdinr_shock": float(x_opt[1]),
            "rate_shock_bps": float(x_opt[2] * 100),
            "vix_multiplier": float(max(1.0, 1 + x_opt[3])),
            "target_loss": target_loss,
            "achieved_loss": float(-res.fun) if res.success else None,
            "success": bool(res.success),
        }
