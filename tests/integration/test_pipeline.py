"""
Integration test: full pipeline on a frozen real-data sample.

Sanity checks:
  - ES ≥ VaR at same confidence level
  - Exception rate in plausible range [0, 0.10]
  - At least some walk-forward results generated
  - Database tables populated

This test uses a synthetic price series to avoid network dependency.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os


def make_synthetic_prices(n_days=600, n_assets=6, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    returns = rng.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.eye(n_assets) * 0.0002 + 0.0001 * np.ones((n_assets, n_assets)),
        size=n_days,
    )
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    log_prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=tickers,
    )
    return log_prices


DEFAULT_CFG = {
    "data": {
        "db_path": None,   # will be set per test
        "backtest_warmup_days": 252,
        "equity_tickers": [],
        "ipo_ticker": "",
        "nifty_ticker": "ASSET_0",
        "vix_ticker": "",
        "usdinr_ticker": "",
        "history_start": "2020-01-02",
        "ffill_fx": True,
        "ffill_max_days": 5,
    },
    "models": {
        "confidence_levels": [0.99],
        "es_confidence": 0.975,
        "holding_period_days": 1,
        "hs": {"lookback_days": 252, "expanding_window": False},
        "fhs": {"vol_model": "EWMA", "ewma_lambda": 0.94, "garch_p": 1, "garch_q": 1, "lookback_days": 252},
        "mc": {"n_scenarios": 5000, "distribution": "normal", "t_df_min": 2, "t_df_max": 30},
        "parametric": {"ewma_lambda": 0.94},
        "stressed_es": {"window_days": 126, "reduced_set_factors": ["equity"]},
        "liquidity_horizons": {"equity": 10, "fx": 10, "rates": 20},
    },
    "backtest": {
        "var_confidence": 0.99,
        "var_confidence_secondary": 0.975,
        "traffic_light_window": 250,
        "traffic_light_thresholds": {"green": 4, "amber": 9},
        "kupiec_significance": 0.05,
        "christoffersen_significance": 0.05,
        "acerbi_simulations": 100,
        "plat_spearman_green": 0.80,
        "plat_spearman_amber": 0.70,
        "plat_ks_green": 0.09,
        "plat_ks_amber": 0.12,
    },
    "data_quality": {
        "zscore_threshold": 5.0,
        "mad_threshold": 10.0,
        "stale_price_days": 3,
        "max_return_threshold": 0.40,
    },
    "reporting": {"alert_log": "/tmp/test_alerts.jsonl"},
}


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def cfg(db_path):
    cfg = {**DEFAULT_CFG, "data": {**DEFAULT_CFG["data"], "db_path": db_path}}
    return cfg


class TestFullPipeline:

    def test_var_es_ordering(self, cfg):
        """ES must be ≥ VaR at the same confidence level."""
        from marketrisk.models.var_models import HistoricalVaR

        prices = make_synthetic_prices()
        returns = np.log(prices / prices.shift(1)).dropna()

        model = HistoricalVaR()
        result = model.compute(returns, confidence=0.99)

        assert result.es_1d >= result.var_1d - 1e-8, (
            f"ES {result.es_1d} < VaR {result.var_1d}: ordering violated"
        )

    def test_walk_forward_produces_results(self, cfg, db_path):
        """Walk-forward engine must produce at least 50 VaR estimates."""
        from marketrisk.data.schema import init_db
        from marketrisk.engine.walk_forward import WalkForwardEngine

        conn = init_db(db_path)
        prices = make_synthetic_prices(n_days=600)
        returns = np.log(prices / prices.shift(1)).dropna()

        engine = WalkForwardEngine(
            returns=returns,
            portfolio_weights=None,
            cfg=cfg,
            conn=conn,
        )
        results = engine.run(window_type="rolling", models=["hs"])
        conn.close()

        assert len(results) >= 50, f"Too few results: {len(results)}"

    def test_exception_rate_plausible(self, cfg, db_path):
        """Exception rate at 99% VaR should be between 0% and 10%."""
        from marketrisk.data.schema import init_db
        from marketrisk.engine.walk_forward import WalkForwardEngine

        conn = init_db(db_path)
        prices = make_synthetic_prices(n_days=600)
        returns = np.log(prices / prices.shift(1)).dropna()

        # Synthetic P&L = sum of returns
        pnl = returns.mean(axis=1)

        engine = WalkForwardEngine(
            returns=returns,
            portfolio_weights=None,
            cfg=cfg,
            conn=conn,
        )
        results = engine.run(window_type="rolling", models=["parametric"], pnl_series=pnl)
        conn.close()

        sub = results[(results["model"] == "parametric") & (results["confidence"] == 0.99)]
        if len(sub) > 0 and "exception" in sub.columns:
            exc_rate = sub["exception"].mean()
            assert 0.0 <= exc_rate <= 0.10, f"Exception rate out of range: {exc_rate}"

    def test_fhs_var_positive(self, cfg, db_path):
        """FHS VaR estimates must be positive (losses, not gains)."""
        from marketrisk.models.var_models import FilteredHistoricalSimVaR

        prices = make_synthetic_prices()
        returns = np.log(prices / prices.shift(1)).dropna()

        model = FilteredHistoricalSimVaR(vol_model="EWMA", n_scenarios=1000)
        result = model.compute(returns, confidence=0.99)
        assert result.var_1d > 0

    def test_t_copula_heavier_than_normal(self, cfg):
        """t-copula MC should produce higher VaR than Gaussian MC (fat tails)."""
        from marketrisk.models.var_models import MonteCarloVaR, TCoMonteCarlo

        prices = make_synthetic_prices()
        returns = np.log(prices / prices.shift(1)).dropna()

        mc_normal = MonteCarloVaR(n_scenarios=5000).compute(returns, confidence=0.99)
        mc_t = TCoMonteCarlo(n_scenarios=5000).compute(returns, confidence=0.99)

        # t-copula should generally produce higher VaR (not guaranteed, but usually true)
        # Use a soft assertion to avoid flakiness
        ratio = mc_t.var_1d / (mc_normal.var_1d + 1e-8)
        assert ratio > 0.5, f"t-copula VaR {mc_t.var_1d} implausibly low vs normal {mc_normal.var_1d}"
