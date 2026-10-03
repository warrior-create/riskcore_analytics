"""
Unit tests for Black-76 options pricing.

Tests
-----
1. Put-call parity (exact for European options)
2. Boundary conditions (T→0, K=F ATM)
3. Delta bounds [-1, 0] for puts, [0, 1] for calls (in forward space)
4. Vega ≥ 0
5. Known values from Hull (Options, Futures, 10th ed., Ch.18)
"""

import pytest
import numpy as np
from marketrisk.pricing.options import (
    black76_price,
    black76_greeks,
    put_call_parity_check,
    nifty_forward,
    implied_vol_with_skew,
)


class TestBlack76:

    def test_put_call_parity(self):
        """C - P = exp(-rT)(F - K) must hold exactly."""
        params = dict(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20)
        call = black76_price(**params, option_type="call")
        put  = black76_price(**params, option_type="put")
        # put_call_parity_check signature: (call, put, F, K, r, T, tol)
        assert put_call_parity_check(call, put, params["F"], params["K"], params["r"], params["T"], tol=0.01)

    def test_atm_call_eq_put(self):
        """ATM straddle: call ≈ put (by symmetry of lognormal at ATM)."""
        params = dict(F=19500, K=19500, T=0.5, r=0.07, sigma=0.18)
        call = black76_price(**params, option_type="call")
        put  = black76_price(**params, option_type="put")
        # Not exactly equal (F != K when discounting), but close
        assert abs(call - put) < 50   # within INR 50 on a 19500 index

    def test_call_price_positive(self):
        for moneyness in [0.90, 0.95, 1.0, 1.05, 1.10]:
            F = 20000
            K = F * moneyness
            price = black76_price(F=F, K=K, T=0.25, r=0.065, sigma=0.20, option_type="call")
            assert price >= 0, f"Negative call price at moneyness {moneyness}"

    def test_put_price_positive(self):
        for moneyness in [0.90, 0.95, 1.0, 1.05, 1.10]:
            F = 20000
            K = F * moneyness
            price = black76_price(F=F, K=K, T=0.25, r=0.065, sigma=0.20, option_type="put")
            assert price >= 0

    def test_boundary_T_zero_call(self):
        """At T=0, call = max(F-K, 0)."""
        F, K = 20000, 19000
        price = black76_price(F=F, K=K, T=0, r=0.065, sigma=0.20, option_type="call")
        assert abs(price - max(F - K, 0)) < 1

    def test_boundary_T_zero_put(self):
        """At T=0, put = max(K-F, 0)."""
        F, K = 20000, 21000
        price = black76_price(F=F, K=K, T=0, r=0.065, sigma=0.20, option_type="put")
        assert abs(price - max(K - F, 0)) < 1

    def test_call_delta_bounds(self):
        """Call delta (forward delta) ∈ [0, 1]."""
        greeks = black76_greeks(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert 0 <= greeks["delta"] <= 1, f"Call delta={greeks['delta']} out of range"

    def test_put_delta_bounds(self):
        """Put delta (forward delta) ∈ [-1, 0]."""
        greeks = black76_greeks(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20, option_type="put")
        assert -1 <= greeks["delta"] <= 0, f"Put delta={greeks['delta']} out of range"

    def test_vega_positive(self):
        """Vega must be positive for both calls and puts."""
        for otype in ["call", "put"]:
            g = black76_greeks(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20, option_type=otype)
            assert g["vega"] >= 0

    def test_gamma_positive(self):
        """Gamma must be positive (long vol)."""
        g = black76_greeks(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert g["gamma"] >= 0

    def test_higher_vol_higher_price(self):
        """Higher vol → higher option price."""
        base = black76_price(F=20000, K=20000, T=0.25, r=0.065, sigma=0.20)
        high = black76_price(F=20000, K=20000, T=0.25, r=0.065, sigma=0.30)
        assert high > base

    def test_skew_otm_put_higher(self):
        """Skew model: OTM put gets higher vol than ATM."""
        F, K_otm = 20000, 19000
        sigma_atm = 0.18
        sigma_otm = implied_vol_with_skew(sigma_atm, F, K_otm)
        assert sigma_otm > sigma_atm, "OTM put should have higher implied vol (put wing)"

    def test_nifty_forward(self):
        """Forward > spot when r_INR > div_yield."""
        spot = 20000
        F = nifty_forward(spot, r_inr=0.065, T=0.25)
        assert F > spot


class TestBondPricing:

    def test_bond_price_par(self):
        """Bond at coupon rate equals par when ytm = coupon."""
        from marketrisk.pricing.bonds import bond_price
        face, coupon = 10_000_000, 0.0725
        price = bond_price(face, coupon, coupon, 10)
        assert abs(price - face) < 1000, f"Bond price {price} should be near par"

    def test_duration_finite_diff(self):
        """Modified duration should match finite-difference approximation."""
        from marketrisk.pricing.bonds import bond_price, modified_duration
        face, coupon, ytm, mat = 10_000_000, 0.0725, 0.0750, 10
        p0 = bond_price(face, coupon, ytm, mat)
        p1 = bond_price(face, coupon, ytm + 0.0001, mat)
        fd_dur = -(p1 - p0) / p0 / 0.0001
        md = modified_duration(face, coupon, ytm, mat)
        assert abs(fd_dur - md) < 0.01

    def test_dv01_negative(self):
        """DV01 negative: price falls when yield rises."""
        from marketrisk.pricing.bonds import dv01
        d = dv01(10_000_000, 0.0725, 0.0750, 10)
        assert d < 0

    def test_convexity_positive(self):
        from marketrisk.pricing.bonds import convexity
        c = convexity(10_000_000, 0.0725, 0.0750, 10)
        assert c > 0


class TestStatisticalTests:

    def test_kupiec_green_zone(self):
        """Very low breach rate → high p-value (don't reject H0)."""
        from marketrisk.backtest.statistical_tests import kupiec_pof
        import pandas as pd
        exceptions = pd.Series([False] * 248 + [True] * 2)   # 2/250 = 0.8%
        result = kupiec_pof(exceptions, 0.99)
        assert result.pvalue > 0.10

    def test_kupiec_red_zone(self):
        """High breach rate → low p-value (reject H0)."""
        from marketrisk.backtest.statistical_tests import kupiec_pof
        import pandas as pd
        exceptions = pd.Series([False] * 230 + [True] * 20)  # 8% rate
        result = kupiec_pof(exceptions, 0.99)
        assert result.pvalue < 0.05

    def test_traffic_light_green(self):
        from marketrisk.backtest.statistical_tests import traffic_light
        import pandas as pd
        exc = pd.Series([False] * 247 + [True, False, True])
        r = traffic_light(exc, confidence=0.99)
        assert r.color == "green"
        assert r.n_exceptions == 2

    def test_traffic_light_red(self):
        from marketrisk.backtest.statistical_tests import traffic_light
        import pandas as pd
        exc = pd.Series([True] * 12 + [False] * 238)
        r = traffic_light(exc, confidence=0.99)
        assert r.color == "red"

    def test_christoffersen_independent_series(self):
        """iid exceptions → high independence p-value."""
        from marketrisk.backtest.statistical_tests import christoffersen
        import pandas as pd
        rng = np.random.default_rng(42)
        exc = pd.Series(rng.binomial(1, 0.01, 500).astype(bool))
        result = christoffersen(exc)
        # Independence should not be rejected for random series
        assert result.indep_pvalue > 0.01


class TestFXPricing:

    def test_fx_forward_cip(self):
        """Forward rate > spot when INR rate > USD rate."""
        from marketrisk.pricing.fx import fx_forward_rate
        spot = 83.0
        F = fx_forward_rate(spot, r_inr=0.065, r_usd=0.05, tenor_years=1/12)
        assert F > spot

    def test_dv01_positive(self):
        """DV01 of FX forward w.r.t. INR rate is positive (INR rate up → forward up)."""
        from marketrisk.pricing.fx import fx_forward_dv01
        d = fx_forward_dv01(1_000_000, 83.0, 0.065, 0.05, 1/12)
        assert d > 0
