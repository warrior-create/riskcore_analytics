"""
VaR and ES model library for MarketRisk-Lab.

Models implemented
------------------
v1 – Parametric (delta-normal, EWMA covariance)
v1 – Historical Simulation (basic)
v1 – Monte Carlo (Gaussian)
v2 – Historical Simulation with full revaluation support
v3 – Filtered Historical Simulation (EWMA and GARCH(1,1))
v4 – Monte Carlo with Student-t marginals + t-copula

All models expose a common interface:
    compute(returns, confidence, **kwargs) → VaRResult

Units: returns are log-returns (dimensionless), VaR/ES are in return space
       (positive = loss). Multiply by portfolio value to get INR amounts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Fixed random seed per spec
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VaRResult:
    model: str
    confidence: float
    var_1d: float            # positive number = loss
    es_1d: float             # Expected Shortfall (CVaR)
    n_obs: int = 0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v1: Parametric (Delta-Normal) VaR
# ---------------------------------------------------------------------------

class ParametricVaR:
    """
    Delta-normal (parametric) VaR using EWMA covariance.

    Limitations (v1 baseline):
    - Linear delta approximation: fails for short-dated options (documented).
    - Normal distribution assumption: fat tails not captured.
    - EWMA lambda calibrated to 0.94 (RiskMetrics standard).
    """

    VERSION = "v1"

    def __init__(self, ewma_lambda: float = 0.94):
        self.lam = ewma_lambda

    def compute(
        self,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        """
        Parameters
        ----------
        returns    : T × N DataFrame of log-returns
        confidence : VaR confidence level
        weights    : portfolio weights (equal-weighted if None)

        Returns
        -------
        VaRResult
        """
        r = returns.dropna().values
        T, N = r.shape

        if weights is None:
            weights = np.ones(N) / N

        # EWMA covariance
        Sigma = self._ewma_cov(r)

        # Portfolio variance
        port_var = weights @ Sigma @ weights
        port_vol = np.sqrt(port_var)

        # VaR and ES under normality
        z = stats.norm.ppf(confidence)
        var_1d = port_vol * z

        # ES = sigma × phi(z) / (1 - alpha)
        es_1d = port_vol * stats.norm.pdf(z) / (1 - confidence)

        return VaRResult(
            model="parametric",
            confidence=confidence,
            var_1d=float(var_1d),
            es_1d=float(es_1d),
            n_obs=T,
            meta={"ewma_lambda": self.lam, "port_vol": float(port_vol)},
        )

    def _ewma_cov(self, r: np.ndarray) -> np.ndarray:
        """Exponentially weighted covariance matrix."""
        T, N = r.shape
        Sigma = np.cov(r[-min(60, T):].T)   # initialise with recent cov
        for t in range(1, T):
            rt = r[t - 1:t].T
            Sigma = self.lam * Sigma + (1 - self.lam) * (rt @ rt.T)
        return Sigma


# ---------------------------------------------------------------------------
# v1: Historical Simulation (basic)
# ---------------------------------------------------------------------------

class HistoricalVaR:
    """
    Basic historical simulation VaR/ES.

    v1 version: uses equally-weighted P&L history.
    Note: P&L is approximated as w·r (no full revaluation).
    Full revaluation version is v2 (FHSFullReval) in engine.
    """

    VERSION = "v1"

    def compute(
        self,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        r = returns.dropna().values
        T, N = r.shape

        if weights is None:
            weights = np.ones(N) / N

        port_rets = r @ weights  # portfolio return for each scenario
        loss = -port_rets        # sign flip: positive = loss

        var_1d = float(np.percentile(loss, confidence * 100))
        tail = loss[loss >= var_1d]
        es_1d = float(tail.mean()) if len(tail) > 0 else var_1d

        return VaRResult(
            model="hs",
            confidence=confidence,
            var_1d=var_1d,
            es_1d=es_1d,
            n_obs=T,
        )


# ---------------------------------------------------------------------------
# v1: Monte Carlo (Gaussian)
# ---------------------------------------------------------------------------

class MonteCarloVaR:
    """Monte Carlo VaR/ES assuming multivariate normal log-returns."""

    VERSION = "v1"

    def __init__(self, n_scenarios: int = 50_000, random_seed: int = 42):
        self.n = n_scenarios
        self.rng = np.random.default_rng(random_seed)

    def compute(
        self,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        r = returns.dropna().values
        T, N = r.shape

        if weights is None:
            weights = np.ones(N) / N

        mu = r.mean(axis=0)
        Sigma = np.cov(r.T)

        # Draw scenarios
        scenarios = self.rng.multivariate_normal(mu, Sigma, size=self.n)
        port_rets = scenarios @ weights
        loss = -port_rets

        var_1d = float(np.percentile(loss, confidence * 100))
        es_1d = float(loss[loss >= var_1d].mean()) if any(loss >= var_1d) else var_1d

        return VaRResult(
            model="mc_normal",
            confidence=confidence,
            var_1d=var_1d,
            es_1d=es_1d,
            n_obs=T,
            meta={"n_scenarios": self.n},
        )


# ---------------------------------------------------------------------------
# v3: Filtered Historical Simulation (EWMA or GARCH)
# ---------------------------------------------------------------------------

class FilteredHistoricalSimVaR:
    """
    Filtered Historical Simulation (FHS) – v3.

    Algorithm
    ---------
    1. Fit GARCH(1,1) (or EWMA) to each return series.
    2. Standardise residuals: z_t = r_t / sigma_t.
    3. Draw from the empirical distribution of z to obtain
       conditionally heteroscedastic scenarios.
    4. Scale by the current (day T+1) conditional volatility.

    References: Barone-Adesi, Giannopoulos & Vosper (1999).
    """

    VERSION = "v3"

    def __init__(
        self,
        vol_model: str = "GARCH",   # "GARCH" or "EWMA"
        ewma_lambda: float = 0.94,
        n_scenarios: int = 10_000,
        random_seed: int = 42,
    ):
        self.vol_model = vol_model
        self.lam = ewma_lambda
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(random_seed)

    def compute(
        self,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        r = returns.dropna()
        T, N = r.shape

        if weights is None:
            weights = np.ones(N) / N

        # Fit vol model and get standardised residuals + forecast vol
        std_resids = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
        sigma_t1 = np.zeros(N)   # forecast vol for t+1

        for i, col in enumerate(r.columns):
            sr, sv = self._fit_vol_model(r[col].values)
            std_resids[col] = sr
            sigma_t1[i] = sv

        # Draw scenarios from standardised residual empirical distribution
        z_matrix = std_resids.values  # T × N
        idx = self.rng.integers(0, T, size=self.n_scenarios)
        z_draw = z_matrix[idx]  # n_scenarios × N

        # Scale by current volatility → scenarios in return space
        scenarios = z_draw * sigma_t1[np.newaxis, :]
        port_rets = scenarios @ weights
        loss = -port_rets

        var_1d = float(np.percentile(loss, confidence * 100))
        tail_mask = loss >= var_1d
        es_1d = float(loss[tail_mask].mean()) if tail_mask.any() else var_1d

        return VaRResult(
            model=f"fhs_{self.vol_model.lower()}",
            confidence=confidence,
            var_1d=var_1d,
            es_1d=es_1d,
            n_obs=T,
            meta={"vol_model": self.vol_model, "n_scenarios": self.n_scenarios},
        )

    def _fit_vol_model(self, r: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns standardised residuals and forecast sigma for t+1.
        """
        if self.vol_model == "EWMA":
            return self._ewma(r)
        else:
            return self._garch11(r)

    def _ewma(self, r: np.ndarray) -> tuple[np.ndarray, float]:
        T = len(r)
        sigma2 = np.full(T, r.var())
        for t in range(1, T):
            sigma2[t] = self.lam * sigma2[t - 1] + (1 - self.lam) * r[t - 1] ** 2
        sigma2 = np.maximum(sigma2, 1e-10)
        sigma = np.sqrt(sigma2)
        std_r = r / sigma
        sigma_t1 = np.sqrt(self.lam * sigma2[-1] + (1 - self.lam) * r[-1] ** 2)
        return std_r, float(sigma_t1)

    def _garch11(self, r: np.ndarray) -> tuple[np.ndarray, float]:
        """GARCH(1,1) via arch library or manual MLE fallback."""
        try:
            from arch import arch_model
            am = arch_model(r * 100, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            cond_vol = res.conditional_volatility / 100
            std_r = r / np.maximum(cond_vol, 1e-10)
            # Forecast one step ahead
            fcast = res.forecast(horizon=1, reindex=False)
            sigma_t1 = float(np.sqrt(fcast.variance.values[-1, 0])) / 100
            return std_r, sigma_t1
        except Exception as exc:
            logger.warning("GARCH fit failed (%s), falling back to EWMA", exc)
            return self._ewma(r)


# ---------------------------------------------------------------------------
# v4: Monte Carlo with Student-t marginals + t-copula
# ---------------------------------------------------------------------------

class TCoMonteCarlo:
    """
    Monte Carlo VaR/ES with Student-t marginals and t-copula.

    Method
    ------
    1. Fit univariate t-distribution (mu, sigma, df) to each return series via MLE.
    2. Transform to uniform margins via t-CDF.
    3. Fit a multivariate t-copula (df_c fitted by MLE via profile likelihood).
    4. Simulate from the copula → back-transform via t-quantile functions.
    5. Aggregate to portfolio P&L → compute VaR and ES.

    Documented limitation: copula df fitted jointly (not per-pair), which is
    a simplification of the full t-copula MLE.
    """

    VERSION = "v4"

    def __init__(self, n_scenarios: int = 50_000, random_seed: int = 42,
                 t_df_min: float = 2, t_df_max: float = 30):
        self.n = n_scenarios
        self.rng = np.random.default_rng(random_seed)
        self.df_min = t_df_min
        self.df_max = t_df_max

    def compute(
        self,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        r = returns.dropna().values
        T, N = r.shape

        if weights is None:
            weights = np.ones(N) / N

        # Step 1: Fit t-marginals
        params = []
        uniform_obs = np.zeros_like(r)
        for i in range(N):
            mu, sigma, df = self._fit_t(r[:, i])
            params.append((mu, sigma, df))
            # CDF transform to uniforms
            uniform_obs[:, i] = stats.t.cdf((r[:, i] - mu) / sigma, df=df)

        # Clip to avoid ±inf in Gaussian copula inversion
        u = np.clip(uniform_obs, 1e-6, 1 - 1e-6)

        # Step 2: Fit t-copula df via Gaussian correlation on normal scores
        z = stats.norm.ppf(u)
        Rho = np.corrcoef(z.T)

        # Step 3: Fit copula df by profile likelihood
        df_cop = self._fit_copula_df(u, Rho)

        # Step 4: Simulate from t-copula
        # Sample from multivariate t → transform → apply marginal quantiles
        L = np.linalg.cholesky(Rho)
        chi2_samp = self.rng.chisquare(df_cop, size=self.n)
        norm_samp = self.rng.standard_normal(size=(self.n, N))
        t_samp = norm_samp @ L.T / np.sqrt(chi2_samp / df_cop)[:, np.newaxis]

        # Copula uniform draws
        u_sim = stats.t.cdf(t_samp, df=df_cop)
        u_sim = np.clip(u_sim, 1e-6, 1 - 1e-6)

        # Back-transform using marginal t-quantiles
        x_sim = np.zeros_like(u_sim)
        for i, (mu, sigma, df) in enumerate(params):
            x_sim[:, i] = mu + sigma * stats.t.ppf(u_sim[:, i], df=df)

        # Step 5: Portfolio P&L
        port_rets = x_sim @ weights
        loss = -port_rets

        var_1d = float(np.percentile(loss, confidence * 100))
        tail = loss[loss >= var_1d]
        es_1d = float(tail.mean()) if len(tail) > 0 else var_1d

        return VaRResult(
            model="mc_t_copula",
            confidence=confidence,
            var_1d=var_1d,
            es_1d=es_1d,
            n_obs=T,
            meta={"copula_df": df_cop, "marginal_dfs": [p[2] for p in params]},
        )

    def _fit_t(self, r: np.ndarray) -> tuple[float, float, float]:
        """Fit Student-t via scipy MLE. Returns (mu, sigma, df)."""
        df, mu, sigma = stats.t.fit(r)
        df = float(np.clip(df, self.df_min, self.df_max))
        return float(mu), float(sigma), df

    def _fit_copula_df(self, u: np.ndarray, Rho: np.ndarray) -> float:
        """
        Profile likelihood for t-copula df.
        Simplified: maximise likelihood of t-copula over df ∈ [2, 30].
        """
        best_ll = -np.inf
        best_df = 4.0
        for df in np.arange(self.df_min, min(self.df_max + 1, 21), 1.0):
            try:
                t_scores = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=df)
                ll = stats.multivariate_normal.logpdf(t_scores, cov=Rho).sum()
                if ll > best_ll:
                    best_ll = ll
                    best_df = df
            except Exception:
                continue
        return float(best_df)


# ---------------------------------------------------------------------------
# Common: ES at arbitrary confidence level
# ---------------------------------------------------------------------------

def compute_es(
    losses: np.ndarray, confidence: float = 0.975
) -> tuple[float, float]:
    """
    Compute VaR and ES from a loss array.

    Parameters
    ----------
    losses     : array of losses (positive = loss), can be simulated or historical
    confidence : confidence level

    Returns
    -------
    (var, es)
    """
    var = float(np.percentile(losses, confidence * 100))
    tail = losses[losses >= var]
    es = float(tail.mean()) if len(tail) > 0 else var
    return var, es


# ---------------------------------------------------------------------------
# Marginal and Component Contributions
# ---------------------------------------------------------------------------

def component_es_contributions(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.975,
) -> dict[str, float]:
    """
    Component ES contributions using the average-P&L-in-tail method.
    Contributions are exactly additive: sum(contributions) = ES.

    Method: ES_i = -E[r_i | portfolio_return ≤ -VaR]
    """
    r = returns.dropna().values
    port_rets = r @ weights
    loss = -port_rets

    var = np.percentile(loss, confidence * 100)
    tail_mask = loss >= var

    if not tail_mask.any():
        return {col: 0.0 for col in returns.columns}

    tail_r = r[tail_mask]   # scenarios in tail
    # Component contribution = -mean of each asset's return in tail × weight
    contrib = {}
    for i, col in enumerate(returns.columns):
        contrib[col] = float(-tail_r[:, i].mean() * weights[i])
    return contrib


def marginal_var_contributions(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    epsilon: float = 0.01,
) -> dict[str, float]:
    """
    Marginal VaR contributions via finite differences (neighbourhood averaging).

    Noisy in historical simulation — use component ES for stability.
    """
    r = returns.dropna().values
    port_rets = r @ weights
    loss = -port_rets
    var_base = float(np.percentile(loss, confidence * 100))

    contribs = {}
    for i, col in enumerate(returns.columns):
        w_up = weights.copy()
        w_up[i] += epsilon
        w_up /= w_up.sum()
        port_up = r @ w_up
        var_up = float(np.percentile(-port_up, confidence * 100))
        contribs[col] = (var_up - var_base) / epsilon

    return contribs
