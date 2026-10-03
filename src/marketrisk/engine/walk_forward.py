"""
Walk-Forward Backtesting Engine.

Protocol
--------
- VaR for day t is estimated using ONLY data available up to t-1 (strict).
- Warm-up: first 500 trading days used for initial calibration.
- Both rolling (fixed 500-day) and expanding windows tested.
- Multiple models run in parallel per step.
- Results stored in var_results and backtest_results DB tables.

The engine runs clean P&L (full revaluation) by default;
dirty P&L can be computed separately.

Performance
-----------
~15 years × 250 trading days = 3,750 steps.
Each step runs 4 models × 2 confidence levels → ~30,000 VaR estimates.
Typical runtime: 5–15 minutes on a modern laptop.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from marketrisk.models.var_models import (
    ParametricVaR,
    HistoricalVaR,
    MonteCarloVaR,
    FilteredHistoricalSimVaR,
    TCoMonteCarlo,
    VaRResult,
)

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """
    Walk-forward VaR/ES engine.

    Parameters
    ----------
    returns       : T × N DataFrame of portfolio risk factor log-returns
    portfolio_weights : N-vector of weights (equal if None)
    cfg           : master config dict
    conn          : SQLite connection for storing results
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray],
        cfg: dict,
        conn: sqlite3.Connection,
    ):
        self.returns = returns
        self.weights = portfolio_weights
        if self.weights is None:
            N = returns.shape[1]
            self.weights = np.ones(N) / N

        self.cfg = cfg
        self.conn = conn
        self.warmup = cfg["data"]["backtest_warmup_days"]
        self.confidences = cfg["models"]["confidence_levels"]
        self.rolling_window = cfg["models"]["hs"]["lookback_days"]

        # Instantiate models
        self._models = {
            "parametric": ParametricVaR(cfg["models"]["parametric"]["ewma_lambda"]),
            "hs": HistoricalVaR(),
            "fhs_garch": FilteredHistoricalSimVaR(vol_model="GARCH"),
            "fhs_ewma": FilteredHistoricalSimVaR(vol_model="EWMA"),
            "mc_normal": MonteCarloVaR(n_scenarios=cfg["models"]["mc"]["n_scenarios"]),
            "mc_t_copula": TCoMonteCarlo(n_scenarios=cfg["models"]["mc"]["n_scenarios"]),
        }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        window_type: str = "rolling",   # "rolling" | "expanding"
        models: Optional[list[str]] = None,
        pnl_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Execute walk-forward loop.

        Parameters
        ----------
        window_type : calibration window type
        models      : list of model keys to run (all if None)
        pnl_series  : actual daily P&L series (for backtesting)

        Returns
        -------
        DataFrame of results indexed by date
        """
        models = models or list(self._models.keys())
        T = len(self.returns)

        if T <= self.warmup:
            raise ValueError(
                f"Not enough data: {T} days, warmup={self.warmup}. "
                "Need at least warmup+1 days."
            )

        all_results: list[dict] = []

        logger.info(
            "Walk-forward: %d test days, window=%s, models=%s",
            T - self.warmup,
            window_type,
            models,
        )

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Walk-forward ({window_type})…",
                total=T - self.warmup,
            )

            for t in range(self.warmup, T):
                test_date = self.returns.index[t]

                # Calibration window
                if window_type == "rolling":
                    cal_data = self.returns.iloc[max(0, t - self.rolling_window):t]
                else:
                    cal_data = self.returns.iloc[:t]

                if len(cal_data) < 30:
                    progress.advance(task)
                    continue

                # Actual loss for this date (for breach detection)
                actual_loss = None
                if pnl_series is not None and test_date in pnl_series.index:
                    actual_loss = -float(pnl_series.loc[test_date])  # positive = loss

                # Run each model
                for model_name in models:
                    model = self._models.get(model_name)
                    if model is None:
                        continue

                    for conf in self.confidences:
                        try:
                            result = model.compute(
                                cal_data, confidence=conf, weights=self.weights
                            )
                            row = {
                                "date": test_date,
                                "model": model_name,
                                "confidence": conf,
                                "var_1d": result.var_1d,
                                "es_1d": result.es_1d,
                                "actual_loss": actual_loss,
                                "exception": (
                                    actual_loss is not None
                                    and actual_loss > result.var_1d
                                ),
                                "window_type": window_type,
                            }
                            all_results.append(row)
                        except Exception as exc:
                            logger.warning(
                                "Model %s failed on %s: %s", model_name, test_date, exc
                            )

                progress.advance(task)

        df = pd.DataFrame(all_results)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            self._save_var_results(df)

        logger.info("Walk-forward complete: %d results", len(df))
        return df

    # ------------------------------------------------------------------
    # Incremental VaR (live trade support)
    # ------------------------------------------------------------------

    def incremental_var(
        self,
        trade: dict,
        model_name: str = "hs",
        confidence: float = 0.99,
    ) -> dict:
        """
        Compute incremental VaR/ES for a proposed trade.

        Parameters
        ----------
        trade : {ticker: str, quantity: float, price: float}

        Returns
        -------
        dict with base_var, new_var, incremental_var, incremental_es,
             is_nmrf_flagged
        """
        ticker = trade.get("ticker", "")
        qty = float(trade.get("quantity", 0))

        cal_data = self.returns.iloc[-self.rolling_window:]
        model = self._models.get(model_name, HistoricalVaR())

        # Base VaR
        base = model.compute(cal_data, confidence, self.weights)

        # New weights including trade
        N = len(self.weights)
        new_weights = self.weights.copy()
        if ticker in cal_data.columns:
            idx = list(cal_data.columns).index(ticker)
            new_weights[idx] += qty / (abs(qty) + 1e-8) * 0.01   # 1% notional bump
            new_weights /= new_weights.sum()

        new_result = model.compute(cal_data, confidence, new_weights)
        incr_var = new_result.var_1d - base.var_1d
        incr_es = new_result.es_1d - base.es_1d

        # NMRF flag: short history in returns
        n_obs = cal_data[ticker].notna().sum() if ticker in cal_data.columns else 0
        is_nmrf = n_obs < 24

        return {
            "base_var": base.var_1d,
            "new_var": new_result.var_1d,
            "incremental_var": incr_var,
            "incremental_es": incr_es,
            "is_nmrf_flagged": is_nmrf,
            "n_obs": n_obs,
        }

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _save_var_results(self, df: pd.DataFrame) -> None:
        rows = []
        for _, row in df.iterrows():
            rows.append((
                str(row["date"].date()),
                row["model"],
                float(row["confidence"]),
                1,                          # holding_period = 1 day
                float(row["var_1d"]),
                float(row["es_1d"]) if pd.notna(row.get("es_1d")) else None,
                None,                       # stressed_es computed separately
            ))
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO var_results
               (date, model, confidence, holding_period, var_1d, es_1d, stressed_es)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        logger.info("Saved %d var_results rows", len(rows))
