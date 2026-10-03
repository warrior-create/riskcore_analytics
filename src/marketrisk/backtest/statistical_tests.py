"""
Backtesting module: statistical tests on VaR/ES.

Tests implemented
-----------------
1. Basel Traffic Light (250-day, 99% VaR): green/amber/red classification.
2. Kupiec POF (Proportion of Failures) test.
3. Christoffersen Independence and Conditional Coverage tests.
4. Acerbi-Székely Z1 and Z2 tests for ES.

All tests return (statistic, p-value, decision) tuples.

References
----------
Kupiec (1995). Techniques for Verifying the Accuracy of Risk Models.
Christoffersen (1998). Evaluating Interval Forecasts.
Acerbi & Székely (2014). Back-testing Expected Shortfall.
BCBS (2022). Basel traffic light thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from .pnl_attribution import PLATest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TrafficLightResult:
    n_obs: int
    n_exceptions: int
    exception_rate: float
    color: str    # green | amber | red
    confidence: float


@dataclass
class KupiecResult:
    statistic: float
    pvalue: float
    reject_null: bool   # null: exception rate = 1-confidence
    n_obs: int
    n_exceptions: int


@dataclass
class ChristoffersenResult:
    # Independence test
    indep_statistic: float
    indep_pvalue: float
    # Conditional coverage (joint)
    cc_statistic: float
    cc_pvalue: float
    n_00: int
    n_01: int
    n_10: int
    n_11: int


@dataclass
class AcerbiResult:
    z1: float
    z2: float
    z1_pvalue: float
    z2_pvalue: float
    n_simulations: int


# ---------------------------------------------------------------------------
# 1. Basel Traffic Light
# ---------------------------------------------------------------------------

def traffic_light(
    exceptions: pd.Series,
    confidence: float = 0.99,
    window: int = 250,
    green_threshold: int = 4,
    amber_threshold: int = 9,
) -> TrafficLightResult:
    """
    Apply Basel traffic-light test to the last `window` trading days.

    Parameters
    ----------
    exceptions : boolean Series (True = exception / breach)
    confidence : VaR confidence level
    window     : rolling window (Basel: 250)

    Returns
    -------
    TrafficLightResult
    """
    recent = exceptions.iloc[-window:] if len(exceptions) >= window else exceptions
    n_obs = len(recent)
    n_exc = int(recent.sum())
    exc_rate = n_exc / n_obs if n_obs > 0 else 0.0

    if n_exc <= green_threshold:
        color = "green"
    elif n_exc <= amber_threshold:
        color = "amber"
    else:
        color = "red"

    return TrafficLightResult(
        n_obs=n_obs,
        n_exceptions=n_exc,
        exception_rate=exc_rate,
        color=color,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 2. Kupiec POF
# ---------------------------------------------------------------------------

def kupiec_pof(
    exceptions: pd.Series,
    confidence: float = 0.99,
) -> KupiecResult:
    """
    Kupiec (1995) Proportion of Failures likelihood ratio test.

    H0: exception rate = 1 - confidence
    Test statistic: LR_POF ~ chi2(1) under H0
    """
    T = len(exceptions)
    x = int(exceptions.sum())
    p = 1.0 - confidence   # expected failure rate

    if T == 0 or x == 0:
        # Handle edge cases
        return KupiecResult(0.0, 1.0, False, T, x)

    p_hat = x / T

    # Log-likelihood ratio
    if p_hat == 0:
        lr = 2 * T * np.log(1 - p) - 2 * T * np.log(1 - p_hat + 1e-10)
    elif p_hat == 1:
        lr = 2 * T * np.log(p) - 2 * T * np.log(p_hat - 1e-10)
    else:
        lr = 2 * (
            x * np.log(p_hat / p) + (T - x) * np.log((1 - p_hat) / (1 - p))
        )

    pvalue = float(1 - stats.chi2.cdf(lr, df=1))
    reject = pvalue < 0.05

    return KupiecResult(
        statistic=float(lr),
        pvalue=pvalue,
        reject_null=reject,
        n_obs=T,
        n_exceptions=x,
    )


# ---------------------------------------------------------------------------
# 3. Christoffersen
# ---------------------------------------------------------------------------

def christoffersen(
    exceptions: pd.Series,
    confidence: float = 0.99,
) -> ChristoffersenResult:
    """
    Christoffersen (1998) independence and conditional coverage tests.

    Independence: are exceptions clustered (bad) or independent?
    CC: joint test of unconditional coverage + independence.
    """
    exc = exceptions.astype(int).values

    # Transition counts
    n_00 = int(np.sum((exc[:-1] == 0) & (exc[1:] == 0)))
    n_01 = int(np.sum((exc[:-1] == 0) & (exc[1:] == 1)))
    n_10 = int(np.sum((exc[:-1] == 1) & (exc[1:] == 0)))
    n_11 = int(np.sum((exc[:-1] == 1) & (exc[1:] == 1)))

    # Transition probabilities
    pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
    pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0
    pi = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    # Independence LR
    def _log_l(pi_a, pi_b, n_a0, n_a1, n_b0, n_b1):
        ll = 0.0
        for p_, n0, n1 in [(pi_a, n_a0, n_a1), (pi_b, n_b0, n_b1)]:
            if n0 > 0 and (1 - p_) > 0:
                ll += n0 * np.log(1 - p_ + 1e-10)
            if n1 > 0 and p_ > 0:
                ll += n1 * np.log(p_ + 1e-10)
        return ll

    ll_indep = _log_l(pi, pi, n_00 + n_10, n_01 + n_11, 0, 0)
    ll_markov = _log_l(pi_01, pi_11, n_00, n_01, n_10, n_11)

    # Recompute ll_indep correctly
    n_tot = n_00 + n_01 + n_10 + n_11
    n_exc = n_01 + n_11
    ll_indep = (
        (n_tot - n_exc) * np.log(1 - pi + 1e-10)
        + n_exc * np.log(pi + 1e-10)
    )

    indep_lr = 2 * (ll_markov - ll_indep)
    indep_lr = max(0.0, indep_lr)
    indep_pv = float(1 - stats.chi2.cdf(indep_lr, df=1))

    # Unconditional coverage (Kupiec)
    T = n_tot + 1   # approximate
    p = 1 - confidence
    kupiec_lr = 0.0
    if pi > 0 and pi < 1 and p > 0 and p < 1:
        kupiec_lr = 2 * (
            n_exc * np.log(pi / p) + (n_tot - n_exc) * np.log((1 - pi) / (1 - p))
        )

    cc_lr = max(0.0, kupiec_lr + indep_lr)
    cc_pv = float(1 - stats.chi2.cdf(cc_lr, df=2))

    return ChristoffersenResult(
        indep_statistic=float(indep_lr),
        indep_pvalue=indep_pv,
        cc_statistic=float(cc_lr),
        cc_pvalue=cc_pv,
        n_00=n_00,
        n_01=n_01,
        n_10=n_10,
        n_11=n_11,
    )


# ---------------------------------------------------------------------------
# 4. Acerbi-Székely ES tests
# ---------------------------------------------------------------------------

def acerbi_szekely(
    losses: pd.Series,
    es_estimates: pd.Series,
    var_estimates: pd.Series,
    confidence: float = 0.975,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> AcerbiResult:
    """
    Acerbi & Székely (2014) Z1 and Z2 tests for ES.

    Z1: ratio of exceedance losses to ES estimate
        Under H0 (model correct), E[Z1] = 0.
    Z2: uses all observations, not just exceedances.

    P-values computed by simulation under the null.

    Parameters
    ----------
    losses       : actual loss series (positive = loss)
    es_estimates : model ES estimates (one-to-one with losses)
    var_estimates: model VaR estimates
    confidence   : confidence level
    n_simulations: bootstrap simulations for p-values

    Returns
    -------
    AcerbiResult
    """
    rng = np.random.default_rng(random_seed)
    T = len(losses)
    alpha = 1 - confidence

    # Align series
    df = pd.DataFrame({
        "loss": losses.values,
        "es": es_estimates.values,
        "var": var_estimates.values,
    })
    df = df.dropna()
    T = len(df)

    l = df["loss"].values
    es = df["es"].values
    var_ = df["var"].values

    # Z1: exceedance-weighted
    exc_mask = l > var_
    if exc_mask.sum() == 0:
        z1 = 0.0
    else:
        z1 = float(
            np.mean((l[exc_mask] / es[exc_mask])) - 1
        )

    # Z2: all observations
    z2 = float(
        np.mean(l / es * (l > var_) / alpha) - 1
    )

    # P-values by simulation under null (model is correct)
    z1_null = np.zeros(n_simulations)
    z2_null = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Draw from normal with estimated vol (simplified null)
        null_losses = rng.normal(loc=0, scale=es.mean() * 0.5, size=T)
        null_var = es.mean() * 0.8   # approximate
        exc_null = null_losses > null_var
        if exc_null.sum() > 0:
            z1_null[i] = float(np.mean(null_losses[exc_null] / es.mean())) - 1
        z2_null[i] = float(
            np.mean(null_losses / es.mean() * (null_losses > null_var) / alpha)
        ) - 1

    z1_pv = float(np.mean(z1_null >= z1))
    z2_pv = float(np.mean(z2_null >= z2))

    return AcerbiResult(
        z1=z1,
        z2=z2,
        z1_pvalue=z1_pv,
        z2_pvalue=z2_pv,
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# Aggregate backtest runner
# ---------------------------------------------------------------------------

def run_backtest_battery(
    results_df: pd.DataFrame,
    pnl_series: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """
    Run all tests for each (model, confidence) combination.

    Parameters
    ----------
    results_df : output of WalkForwardEngine.run()
    pnl_series : actual daily P&L (clean)
    cfg        : master config dict

    Returns
    -------
    Summary DataFrame with one row per (model, confidence)
    """
    bt_cfg = cfg.get("backtest", {})
    window = bt_cfg.get("traffic_light_window", 250)
    green_thr = bt_cfg.get("traffic_light_thresholds", {}).get("green", 4)
    amber_thr = bt_cfg.get("traffic_light_thresholds", {}).get("amber", 9)

    summaries = []

    for (model, conf), grp in results_df.groupby(["model", "confidence"]):
        grp = grp.sort_values("date").set_index("date")
        grp = grp.join(pnl_series.rename("actual_pnl"), how="left")

        if grp["actual_pnl"].isna().all():
            continue

        actual_loss = -grp["actual_pnl"]
        exceptions = actual_loss > grp["var_1d"]

        # Tests
        tl = traffic_light(exceptions, conf, window, green_thr, amber_thr)
        kup = kupiec_pof(exceptions, conf)
        chr_ = christoffersen(exceptions, conf)

        # Acerbi (only for ES conf = 0.975)
        acerbi = None
        if abs(conf - 0.975) < 0.001 and grp["es_1d"].notna().any():
            acerbi = acerbi_szekely(
                actual_loss, grp["es_1d"].fillna(grp["var_1d"]),
                grp["var_1d"],
                confidence=conf,
                n_simulations=bt_cfg.get("acerbi_simulations", 500),
            )

        # Add PLAT test
        actual_pnl = grp["actual_pnl"].dropna()
        plat = None
        if len(actual_pnl) > 30:
            plat_engine = PLATest()
            # For demonstration, synthetic RTPL is actual_pnl + some noise
            # In a real system, RTPL would be passed in or computed from Greeks
            rtpl = actual_pnl * np.random.normal(1, 0.05, len(actual_pnl))
            plat = plat_engine.run(actual_pnl, rtpl)

        row = {
            "model": model,
            "confidence": conf,
            "test_start": str(grp.index.min().date()),
            "test_end": str(grp.index.max().date()),
            "n_obs": tl.n_obs,
            "n_exceptions": tl.n_exceptions,
            "exception_rate": tl.exception_rate,
            "traffic_light": tl.color,
            "kupiec_stat": kup.statistic,
            "kupiec_pvalue": kup.pvalue,
            "chr_indep_stat": chr_.indep_statistic,
            "chr_indep_pvalue": chr_.indep_pvalue,
            "chr_cc_stat": chr_.cc_statistic,
            "chr_cc_pvalue": chr_.cc_pvalue,
            "acerbi_z1": acerbi.z1 if acerbi else None,
            "acerbi_z2": acerbi.z2 if acerbi else None,
            "acerbi_z1_pvalue": acerbi.z1_pvalue if acerbi else None,
            "acerbi_z2_pvalue": acerbi.z2_pvalue if acerbi else None,
            "plat_zone": plat["zone"] if plat and "zone" in plat else None,
            "plat_spearman": plat["spearman_correlation"] if plat and "spearman_correlation" in plat else None,
            "plat_ks": plat["ks_statistic"] if plat and "ks_statistic" in plat else None,
        }
        summaries.append(row)

    return pd.DataFrame(summaries)
