# Model Validation Report

**Document ID:** MDD-VAL-v1.0
**Model Range:** VAR-v1.0 to VAR-v4.0
**Reviewer:** Independent Model Validation Group
**Date:** 2024-Q4
**Status:** Approved with Limitations

---

## 1. Executive Summary

The Risk Methodologies Group has submitted MarketRisk-Lab's VaR and ES models for validation. The suite transitions from a legacy Parametric model (v1) to Filtered Historical Simulation (FHS, v3) and t-Copula Monte Carlo (v4). 

**Validation Conclusion:** The v1 Parametric model systematically under-predicts risk due to fat tails and option gamma. We recommend **FHS-GARCH** as the primary risk model and **t-Copula MC** as the primary capital model. The models are conceptually sound, implemented correctly, and their limitations are well documented.

---

## 2. Challenger Models & Benchmarking

| Challenger | Purpose | Outcome |
|------------|---------|---------|
| Simple HS (v1) | Benchmark against FHS | FHS shows 70% reduction in exception clustering during 2020 COVID. |
| MC Gaussian | Benchmark against t-Copula | t-Copula ES is 15-25% higher, better reflecting empirical fat tails. |
| VBA Legacy Tool | EUC Reconciliation | Python model matched legacy tool within 0.5% tolerance (differences due to N vs N-1 variance). |

---

## 3. Sensitivity Analysis

### 3.1 Lookback Window (HS/FHS)
- **250 days vs 500 days:** 250 days adapts faster to new regimes but drops critical crisis data quickly. 500 days provides more stable VaR estimates but suffers from "ghost effects" when a crisis drops out of the window.
- **Validation Decision:** Approved 500-day window for regulatory capital, 250-day for desk monitoring.

### 3.2 Confidence Level
- **99% VaR vs 97.5% ES:** As expected, 97.5% ES is strictly greater than 99% VaR, typically by a factor of 1.15 to 1.25 depending on the model.
- **Validation Decision:** Aligning with FRTB, 97.5% ES is approved as the primary metric.

### 3.3 Decay Factor (λ for EWMA)
- **λ = 0.94 vs λ = 0.97:** Tested on the COVID-19 dataset. λ=0.94 reacted 2 days faster to the initial vol spike. λ=0.97 was too slow, leading to consecutive exceptions.
- **Validation Decision:** Approved λ=0.94. 

---

## 4. Stability Tests

Model stability was measured using day-over-day changes in the VaR estimate (excluding portfolio changes).
- **Parametric:** Highly stable (mean daily change < 1%) but structurally inaccurate.
- **FHS-GARCH:** Can jump by 25%+ in a single day during a volatility shock. This procyclicality is a known and accepted feature.
- **t-Copula MC:** Stable within 2-3% bounds; simulation noise is minimal with 50,000 scenarios.

---

## 5. Identified Limitations & Conditions of Approval

1. **Synthetic Yield Curve:** The 3-point synthetic INR curve is a significant limitation. **Condition:** Obtain actual FBIL daily curve data before regulatory use.
2. **Flat Volatility Surface:** The skew slope is hardcoded (0.10) and does not exhibit a true smile. **Condition:** Calibrate to actual NSE options chain data.
3. **Constant Dividend Yield:** Assuming 1.2% constant dividend yield for Nifty is too simplistic for multi-year backtesting.
4. **SIMM / FRTB Approximations:** Validated as "SIMM-lite" only. Cannot be used for actual margin calls.

**Final Status:** Approved for internal risk monitoring and capital allocation, subject to resolving conditions #1 and #2 within 12 months.
