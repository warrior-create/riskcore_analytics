"""
Monitoring framework: Key Risk Indicators, KRI thresholds, escalation matrix.

KRIs computed
-------------
- rolling_breach_rate      : 60-day rolling exception rate
- rolling_kupiec_pvalue    : 60-day rolling Kupiec p-value
- model_stability_ewma     : EWMA of |VaR(t) - VaR(t-1)| / VaR(t-1)
- var_pnl_ratio            : rolling VaR / average |P&L|

Escalation matrix
-----------------
Status  Condition                           Action
GREEN   All KRIs within green thresholds   Normal monitoring
AMBER   Any KRI in amber zone              Notify Risk Manager; review within 5 days
RED     Any KRI in red zone               Immediate escalation; model review; possible capital add-on

Alerts are written to data/alerts.jsonl.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from marketrisk.backtest.statistical_tests import kupiec_pof

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KRI definitions
# ---------------------------------------------------------------------------

KRI_THRESHOLDS = {
    "rolling_breach_rate": {
        "amber": 0.035,   # > 3.5% breach rate over rolling 60 days
        "red":   0.05,    # > 5% breach rate
    },
    "rolling_kupiec_pvalue": {
        "amber": 0.10,    # Kupiec p-value < 0.10 (weaker evidence)
        "red":   0.05,    # Kupiec p-value < 0.05 (statistically significant overcount)
    },
    "model_stability": {
        "amber": 0.15,    # > 15% EWMA daily VaR change
        "red":   0.30,    # > 30% EWMA daily VaR change
    },
    "var_pnl_ratio": {
        "amber": 3.0,     # VaR > 3× average |P&L|
        "red":   5.0,     # VaR > 5× average |P&L|
    },
}


class MonitoringFramework:
    """
    Compute KRIs and generate alerts.

    Parameters
    ----------
    backtest_df : WalkForwardEngine output
    pnl_series  : actual daily P&L
    cfg         : master config dict
    """

    def __init__(
        self,
        backtest_df: pd.DataFrame,
        pnl_series: pd.Series,
        cfg: dict,
        alert_log_path: Optional[str] = None,
    ):
        self.bt = backtest_df.copy()
        self.pnl = pnl_series.copy()
        self.cfg = cfg
        self.alert_path = Path(
            alert_log_path or cfg.get("reporting", {}).get("alert_log", "data/alerts.jsonl")
        )
        self.alert_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, model: str = "hs", confidence: float = 0.99) -> pd.DataFrame:
        """Compute all KRIs and return a DataFrame, writing alerts."""
        subset = (
            self.bt[(self.bt["model"] == model) & (self.bt["confidence"] == confidence)]
            .sort_values("date")
            .set_index("date")
        )
        subset.index = pd.to_datetime(subset.index)

        if subset.empty:
            logger.warning("No data for model=%s, conf=%s", model, confidence)
            return pd.DataFrame()

        pnl_aligned = self.pnl.reindex(subset.index)
        actual_loss = -pnl_aligned

        kri_records = []
        window = 60

        for i in range(window, len(subset) + 1):
            win_idx = subset.index[max(0, i - window):i]
            win_var = subset["var_1d"].iloc[max(0, i - window):i]
            win_loss = actual_loss.reindex(win_idx)
            win_exc = win_loss > win_var

            date = subset.index[i - 1]

            # 1. Rolling breach rate
            breach_rate = float(win_exc.sum() / len(win_exc)) if len(win_exc) > 0 else 0.0

            # 2. Rolling Kupiec p-value
            kup = kupiec_pof(win_exc, confidence)
            kupiec_pv = kup.pvalue

            # 3. Model stability (EWMA of |ΔVaR/VaR|)
            var_series = win_var
            var_changes = var_series.pct_change().abs().ewm(span=20).mean()
            stability = float(var_changes.iloc[-1]) if not var_changes.empty else 0.0

            # 4. VaR / |P&L| ratio
            avg_abs_pnl = win_loss.abs().mean()
            var_pnl_ratio = (
                float(win_var.iloc[-1] / avg_abs_pnl) if avg_abs_pnl > 0 else 0.0
            )

            kris = {
                "rolling_breach_rate": breach_rate,
                "rolling_kupiec_pvalue": kupiec_pv,
                "model_stability": stability,
                "var_pnl_ratio": var_pnl_ratio,
            }

            # Determine status
            overall_status = "green"
            for kri_name, val in kris.items():
                thr = KRI_THRESHOLDS.get(kri_name, {})
                red_thr = thr.get("red")
                amber_thr = thr.get("amber")

                # For kupiec pvalue: lower = worse
                if kri_name == "rolling_kupiec_pvalue":
                    kri_status = (
                        "red" if val < (red_thr or 0.05)
                        else "amber" if val < (amber_thr or 0.10)
                        else "green"
                    )
                else:
                    kri_status = (
                        "red" if red_thr and val > red_thr
                        else "amber" if amber_thr and val > amber_thr
                        else "green"
                    )

                if kri_status == "red":
                    overall_status = "red"
                elif kri_status == "amber" and overall_status == "green":
                    overall_status = "amber"

            rec = {
                "date": str(date.date()),
                "model": model,
                "confidence": confidence,
                "status": overall_status,
                **kris,
            }
            kri_records.append(rec)

            # Write alert if amber or red
            if overall_status in ("amber", "red"):
                self._write_alert(date, overall_status, kris, model, confidence)

        df = pd.DataFrame(kri_records)
        return df

    def _write_alert(
        self, date, status: str, kris: dict, model: str, confidence: float
    ) -> None:
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "date": str(date.date()),
            "status": status,
            "model": model,
            "confidence": confidence,
            "kris": kris,
            "action": (
                "Immediate escalation: convene model review committee; consider capital add-on"
                if status == "red"
                else "Notify Risk Manager; schedule model review within 5 business days"
            ),
        }
        with open(self.alert_path, "a") as f:
            f.write(json.dumps(alert) + "\n")


# ---------------------------------------------------------------------------
# FRTB P&L Attribution Test
# ---------------------------------------------------------------------------

def pnl_attribution_test(
    risk_theoretical_pnl: pd.Series,
    hypothetical_pnl: pd.Series,
    cfg: dict,
) -> dict:
    """
    FRTB P&L Attribution Test (PLAT, BCBS d457).

    Metrics
    -------
    1. Spearman rank correlation between RTPL and HPL.
    2. Kolmogorov-Smirnov distance between their distributions.

    Thresholds (from d457)
    ----------------------
    Spearman ρ ≥ 0.80 → green; 0.70–0.80 → amber; < 0.70 → red
    KS statistic ≤ 0.09 → green; 0.09–0.12 → amber; > 0.12 → red
    """
    from scipy import stats

    df = pd.DataFrame({"rtpl": risk_theoretical_pnl, "hpl": hypothetical_pnl}).dropna()

    if len(df) < 30:
        return {"error": "Insufficient data for PLAT"}

    # Spearman
    rho, rho_pv = stats.spearmanr(df["rtpl"], df["hpl"])

    # KS
    ks_stat, ks_pv = stats.ks_2samp(df["rtpl"], df["hpl"])

    plat_cfg = cfg.get("backtest", {})
    rho_green = plat_cfg.get("plat_spearman_green", 0.80)
    rho_amber = plat_cfg.get("plat_spearman_amber", 0.70)
    ks_green = plat_cfg.get("plat_ks_green", 0.09)
    ks_amber = plat_cfg.get("plat_ks_amber", 0.12)

    rho_status = (
        "green" if rho >= rho_green
        else "amber" if rho >= rho_amber
        else "red"
    )
    ks_status = (
        "green" if ks_stat <= ks_green
        else "amber" if ks_stat <= ks_amber
        else "red"
    )
    overall = "red" if "red" in (rho_status, ks_status) else (
        "amber" if "amber" in (rho_status, ks_status) else "green"
    )

    return {
        "n_obs": len(df),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(rho_pv),
        "spearman_status": rho_status,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pv),
        "ks_status": ks_status,
        "overall_status": overall,
    }
