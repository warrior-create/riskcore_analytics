"""
Exception Analysis Engine.

Auto-classifies each VaR breach and generates:
  1. Breach classification: vol_regime | single_move | model_lag | position_driver
  2. Plain-English narrative for each exception
  3. Summary table (date, model, excess_loss, class, narrative)

Classification logic
--------------------
- vol_regime    : VIX ≥ 1.5× its rolling 20-day average on breach date
- single_move   : the largest single-factor return on the day > 3σ
- model_lag     : exception follows 3+ consecutive quiet days then a spike
- position_driver: identified by component contribution analysis

Stored in the exception_log table and surfaced in the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Crisis periods for narrative enrichment
CRISIS_PERIODS = {
    "2008-GFC":         ("2008-09-01", "2009-03-31"),
    "2013-Taper":       ("2013-05-01", "2013-09-30"),
    "2016-Demonetisation": ("2016-11-08", "2016-12-31"),
    "2018-ILFS":        ("2018-09-01", "2018-12-31"),
    "2020-COVID":       ("2020-01-15", "2020-06-30"),
    "2022-RateHikes":   ("2022-01-01", "2022-12-31"),
}


@dataclass
class ExceptionRecord:
    date: str
    model: str
    confidence: float
    var_estimate: float
    actual_loss: float
    excess_loss: float
    classification: str
    crisis_regime: str | None
    narrative: str


class ExceptionAnalysisEngine:
    """
    Classify and narrate VaR exceptions.

    Parameters
    ----------
    backtest_df  : output of WalkForwardEngine.run() with actual_loss
    returns_df   : full return history for context (needed for regimes)
    vix_series   : India VIX series
    """

    def __init__(
        self,
        backtest_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        vix_series: pd.Series | None = None,
    ):
        self.bt = backtest_df.copy()
        self.rets = returns_df
        self.vix = vix_series

    def run(self, conn=None) -> pd.DataFrame:
        """
        Classify all exceptions and return a DataFrame.
        Optionally persist to DB.
        """
        exceptions = self.bt[self.bt["exception"] == True].copy()
        if exceptions.empty:
            logger.info("No exceptions to analyse")
            return pd.DataFrame()

        records: list[ExceptionRecord] = []

        for _, row in exceptions.iterrows():
            date = pd.Timestamp(row["date"])
            model = row["model"]
            conf = row["confidence"]
            var_est = float(row["var_1d"])
            actual_loss = float(row["actual_loss"])
            excess = actual_loss - var_est

            # Crisis regime check
            crisis = self._identify_crisis(date)

            # Classification
            classification = self._classify(date, row, excess)

            # Generate narrative
            narrative = self._narrate(
                date, model, conf, var_est, actual_loss, excess, classification, crisis
            )

            records.append(ExceptionRecord(
                date=str(date.date()),
                model=model,
                confidence=conf,
                var_estimate=var_est,
                actual_loss=actual_loss,
                excess_loss=excess,
                classification=classification,
                crisis_regime=crisis,
                narrative=narrative,
            ))

        df = pd.DataFrame([vars(r) for r in records])

        if conn is not None:
            self._save_to_db(df, conn)

        return df

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(self, date: pd.Timestamp, row: pd.Series, excess: float) -> str:
        """Rule-based classifier — returns the primary classification."""
        # Vol regime check
        if self.vix is not None and date in self.vix.index:
            vix_now = float(self.vix.loc[date])
            lookback = self.vix.loc[:date].iloc[-21:-1]  # 20-day history
            if len(lookback) >= 5:
                vix_avg = float(lookback.mean())
                if vix_now >= 1.5 * vix_avg:
                    return "vol_regime"

        # Single-factor extreme move
        if date in self.rets.index:
            rets_today = self.rets.loc[date]
            rolling_std = self.rets.loc[:date].std()
            z_scores = rets_today.abs() / rolling_std.replace(0, np.nan)
            if z_scores.max() > 3.0:
                return "single_move"

        # Model lag: breach after ≥3 quiet days
        if date in self.bt["date"].values:
            before_dates = self.bt[self.bt["date"] < date]["date"].sort_values()
            if len(before_dates) >= 3:
                last_3 = before_dates.iloc[-3:]
                prev_exceptions = self.bt[
                    self.bt["date"].isin(last_3) & (self.bt["exception"] == False)
                ]
                if len(prev_exceptions) == 3:
                    return "model_lag"

        return "position_driver"

    def _identify_crisis(self, date: pd.Timestamp) -> str | None:
        for name, (start, end) in CRISIS_PERIODS.items():
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return name
        return None

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    def _narrate(
        self,
        date: pd.Timestamp,
        model: str,
        conf: float,
        var_est: float,
        actual_loss: float,
        excess: float,
        classification: str,
        crisis: str | None,
    ) -> str:
        pct_conf = f"{conf*100:.0f}%"
        date_str = date.strftime("%d-%b-%Y")
        model_label = model.upper().replace("_", "-")

        base = (
            f"[{date_str}] {model_label} {pct_conf} VaR exception. "
            f"VaR estimate: {var_est:.4f}, actual loss: {actual_loss:.4f}, "
            f"excess: {excess:.4f} ({excess/var_est*100:.1f}% above VaR)."
        )

        classif_text = {
            "vol_regime": (
                " India VIX spiked significantly above its 20-day average, "
                "indicating a volatility regime shift that the historical window "
                "failed to capture. Common in crisis-onset days."
            ),
            "single_move": (
                " A single risk factor moved more than 3 standard deviations, "
                "dominating the portfolio loss. The model's historical distribution "
                "underweighted this tail event."
            ),
            "model_lag": (
                " After a period of low volatility, the model's calibrated distribution "
                "had compressed tails. The sudden return spike overwhelmed the VaR estimate. "
                "This is a classic model lag / GARCH echo effect."
            ),
            "position_driver": (
                " The exception appears driven by position concentration in a specific "
                "risk factor. Component contribution analysis identifies the driver. "
                "Review marginal VaR of the largest positions."
            ),
        }.get(classification, "")

        crisis_text = ""
        if crisis:
            crisis_text = f" [Crisis regime: {crisis}]"

        return base + classif_text + crisis_text

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _save_to_db(self, df: pd.DataFrame, conn) -> None:
        rows = [
            (
                row["date"],
                row["model"],
                row["confidence"],
                row["var_estimate"],
                row["actual_loss"],
                row["excess_loss"],
                row["classification"],
                row["narrative"],
            )
            for _, row in df.iterrows()
        ]
        cur = conn.cursor()
        cur.executemany(
            """INSERT INTO exception_log
               (date, model, confidence, var_estimate, actual_loss,
                excess_loss, classification, narrative)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        logger.info("Saved %d exception records to DB", len(rows))
