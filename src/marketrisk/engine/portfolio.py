"""
Portfolio container for MarketRisk-Lab.

Holds positions and provides accessors used by the pricing,
risk model and backtest modules.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Stateless portfolio container built from config/settings.yaml.

    The Portfolio is a read-only snapshot at a given date.
    The WalkForwardEngine updates positions daily (roll rules).
    """

    def __init__(self, cfg: dict):
        port_cfg = cfg["portfolio"]
        self.equity_positions: dict[str, int] = port_cfg["equities"]
        self.options: list[dict] = self._build_options(port_cfg)
        self.bond: Optional[dict] = port_cfg.get("bond")
        self.fx_forward: Optional[dict] = port_cfg.get("fx_forward")
        self.irs: Optional[dict] = port_cfg.get("irs")

    # ------------------------------------------------------------------
    # Static option specs
    # ------------------------------------------------------------------

    def _build_options(self, port_cfg: dict) -> list[dict]:
        """Build option position list from config."""
        result = []
        for name, spec in port_cfg.get("options", {}).items():
            result.append(
                {
                    "name": name,
                    "underlying": spec["underlying"],
                    "option_type": spec["type"],
                    "moneyness": spec["moneyness"],
                    "maturity_days": spec["maturity_days"],
                    "contracts": spec["contracts"],
                    "lot_size": 50,   # Nifty standard lot
                    # K and T_remaining are set dynamically at each valuation date
                    "K": None,
                    "T_remaining": spec["maturity_days"] / 252,
                }
            )
        return result

    def set_option_strikes(self, nifty_spot: float) -> None:
        """Set strikes relative to current Nifty spot and moneyness."""
        for opt in self.options:
            moneyness = opt.get("moneyness", 1.0)
            opt["K"] = round(nifty_spot * moneyness / 50) * 50   # nearest 50-pt strike

    def equity_tickers(self) -> list[str]:
        return list(self.equity_positions.keys())

    def total_equity_market_value(self, prices: dict[str, float]) -> float:
        return sum(
            qty * prices.get(ticker, 0)
            for ticker, qty in self.equity_positions.items()
        )

    def __repr__(self) -> str:
        return (
            f"Portfolio("
            f"equities={len(self.equity_positions)}, "
            f"options={len(self.options)}, "
            f"bond={self.bond is not None}, "
            f"fx_fwd={self.fx_forward is not None})"
        )
