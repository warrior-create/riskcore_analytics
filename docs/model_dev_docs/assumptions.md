# Key Modelling Assumptions

**Document ID:** MDD-ASSUMPTIONS-v1.0
**Status:** Approved
**Owner:** Risk Methodologies Group

---

## 1. INR Yield Curve

**Assumption:** The INR risk-free yield curve is approximated by a 3-point synthetic curve
constructed as: `r_INR(τ) = r_USD_FRED(τ) + spread(τ)`

| Tenor | FRED Series | Spread (bps) | INR Rate (approx) |
|-------|------------|-------------|-------------------|
| 2Y | DGS2 | +300 | ~7.0% |
| 5Y | DGS5 | +320 | ~7.25% |
| 10Y | DGS10 | +350 | ~7.5% |

**Rationale:** Daily FBIL/CCIL INR G-sec data is not publicly available in machine-readable format.
The spread is calibrated to approximate RBI-observed 10Y G-sec yields (~7.25-7.5% during 2023-24).

**Limitations:** Does not capture term structure dynamics, INR-specific credit/liquidity premium variations,
or RBI policy interventions. Bond pricing sensitivity to yield changes is correct directionally but
absolute levels are approximated.

**Action:** Replace with FBIL daily data upon subscription or RBI website automated parsing.

---

## 2. Options Volatility Surface

**Assumption:** Implied vol for Nifty options = `σ(K) = VIX/100 + SKEW_SLOPE × (1 - K/F)`

- VIX/100 is used as the ATM implied vol (India VIX ≈ 30-day ATM IV, in % terms).
- Linear skew slope = 0.10 (constant, not re-calibrated to live market).
- No term structure: vol is flat across all maturities within a day.

**Limitations:** 
- VIX ≈ ATM IV is a reasonable approximation but includes a small VRP premium.
- True smile surface has non-linear wings; put wings can be steeper than 0.10.
- Options P&L attribution errors increase for deep OTM strikes.

**Action:** Calibrate skew slope to actual NSE option chain data when available.

---

## 3. Nifty Options

**Assumption:** Nifty options are priced as European using Black-76.

**Rationale:** NSE Nifty options are European-style (cash-settled at expiry). Black-76 is the
exact analytical solution. No early-exercise premium adjustment is needed.

---

## 4. USDINR Forward

**Assumption:** CIP (covered interest rate parity) holds continuously.

**Reality:** CIP basis (deviation from parity) exists due to USD funding costs, but is typically
< 10bps for 1-month tenor. This is within the model's VaR uncertainty band.

---

## 5. Dividend Yield

**Assumption:** Nifty 50 dividend yield = 1.2% p.a. (constant).

**Historical average:** 1.0–1.4% over 2010–2023 per Bloomberg consensus. 1.2% is the midpoint.

---

## 6. Portfolio Weights

**Assumption:** Portfolio weights are static (defined by share counts in config).

**Reality:** A real portfolio rebalances daily. For backtesting purposes, constant weights
are a simplification that understates turnover and transaction costs.

---

## 7. FRTB Compliance

**Assumption:** The stressed ES, liquidity-horizon scaling, and RNIV/NMRF
implementations are simplified approximations of FRTB (d457).

**Documented simplifications:**
- Reduced risk-factor set = {equity, FX} only (not the full FRTB regulatory list)
- Liquidity-horizon cascade: simplified to `VaR(LH) = VaR(1d) × √LH`
- NMRF proxy: single proxy per factor (not multi-factor proxy regression)
- Not regulatory-approved; for educational purposes only

---

## 8. Calendar

**Assumption:** NSE trading calendar is the master. FX and US data are forward-filled
on NSE trading days (max 5 consecutive missing days before flagging as data issue).

**Rationale:** Per project specification. Forward-fill is standard practice for
non-trading-day data alignment.
