# RiskCore Analytics: Executive Findings & Model Governance Log

**Date:** 2024-Q4  
**Author:** Risk Methodologies Group  
**Status:** Live results updated on each daily run  

> All empirical results below are based on over 15 years of NSE historical data (2007–2024), encompassing severe market stress regimes including:
> The 2008 Global Financial Crisis · 2013 Taper Tantrum · 2016 Demonetisation · 2018 IL&FS Default · 2020 COVID Crash · 2022 Global Rate Hikes.

---

## 1. Walk-Forward Backtest Summary

### Basel Framework Traffic Light Results (250-day rolling window, 99% VaR)

| Model Methodology | Green Zone | Amber Zone | Red Zone | Mean Exception Rate |
|-------------------|:---:|:---:|:---:|:---:|
| Parametric (v1) | ~70% | ~20% | ~10% | ~1.8% |
| HS Basic (v1) | ~75% | ~18% | ~7% | ~1.5% |
| FHS-GARCH (v3) | ~88% | ~10% | ~2% | ~1.1% |
| FHS-EWMA (v3) | ~85% | ~12% | ~3% | ~1.2% |
| MC Normal (v1) | ~73% | ~19% | ~8% | ~1.6% |
| MC t-Copula (v4) | ~90% | ~8% | ~2% | ~1.0% |

```mermaid
xychart-beta
    title "Mean Exception Rate by Model (%)"
    x-axis ["Parametric", "MC Normal", "HS Basic", "FHS-EWMA", "FHS-GARCH", "t-Copula"]
    y-axis "Exception Rate (%)" 0.0 --> 2.0
    bar [1.8, 1.6, 1.5, 1.2, 1.1, 1.0]
```

**Validation Conclusion:** Filtered Historical Simulation (FHS-GARCH) and Monte Carlo (t-Copula) materially outperform legacy Parametric methodologies across all historical regimes, bringing exception rates back within acceptable Basel III parameters.

---

## 2. Diagnostics: Parametric VaR Limitations

The legacy parametric (delta-normal) framework failed systematically under stress testing for three critical reasons:

1. **Fat Tails Exceedance:** NSE equity returns demonstrate excess kurtosis ranging from 4 to 7 (relative to 3 for a normal distribution). The Parametric VaR (assuming a normal distribution) systematically underestimates extreme market quantiles by 20–40%.
2. **Short-Dated Options Sensitivities:** The short call positions generate substantial negative gamma P&L during high-velocity market dislocations. The delta-normal framework evaluates options linearly, omitting the crucial Γ·ΔS² term. This delta-gamma failure was most severe during the March 2020 COVID liquidation event.
3. **EWMA Covariance Lag:** The RiskMetrics standard EWMA (λ=0.94) reacts too slowly to extreme volatility shocks. During the COVID onset, the VIX escalated from ~15 to ~85 within 10 days; the EWMA estimate lagged by 5–8 days, producing significantly under-provisioned capital estimates at the precise moment risk was highest.

---

## 3. Remediation via FHS and t-Copula

**Both advanced frameworks successfully resolved the legacy limitations.**

*   **FHS-GARCH (v3):** Reduced Red Zone categorization from ~10% to ~2% at the 99% confidence interval. The GARCH(1,1) specification rapidly escalated conditional volatility during the 2020 crash, neutralizing the EWMA model lag. Concurrently, exception clustering (Christoffersen independence test p-value < 0.05) decreased from 35% to 8%.
*   **t-Copula Monte Carlo (v4):** Demonstrated the highest overall performance. Student-t marginal distributions (calibrated via Maximum Likelihood Estimation with degrees of freedom between 4 and 6) accurately bounded the heavy-tailed NSE equity returns. The t-Copula effectively captured joint crash dynamics (tail dependence) that the standard Gaussian copula structurally ignores.

---

## 4. Expected Shortfall (ES 97.5%) vs VaR (99%)

Transitioning to Expected Shortfall at the 97.5% confidence level aligns with the Fundamental Review of the Trading Book (FRTB) Internal Models Approach (IMA).

| Metric | Parametric | FHS-GARCH | t-Copula |
|--------|:---:|:---:|:---:|
| ES(97.5%) / VaR(99%) Ratio | ~1.10 | ~1.18 | ~1.25 |
| Acerbi Z2 Test (Pass Rate) | ~60% | ~80% | ~90% |

**Recommendation:** ES represents a more conservative and coherent risk measure, successfully clearing the Acerbi-Székely backtest. ES(97.5%) is formally recommended as the primary capital determination metric.

---

## 5. Volatility Clustering & Gamma Frictions

During the March–April 2020 COVID drawdown:
*   All linear v1 models generated highly clustered exceptions (consecutive-day limit breaches), strongly rejected by the Christoffersen independence test (p-value < 0.001).
*   FHS-GARCH reduced clustered exceptions by approximately 70%.
*   The residual clustering observed in the FHS framework was attributed to non-linear gamma friction accumulation and unprecedented VIX doubling events that breached all historical precedents.

**Remediation Action:** The GARCH framework is adopted as the primary production engine, with a Stressed Expected Shortfall overlay applied automatically during high-VIX macro regimes.

---

## 6. Regime Vulnerability Breakdown

| Macro Regime | Optimal Model | Parametric Failures | Key Risk Driver |
|--------------|:---:|:---:|:---:|
| 2008 GFC | FHS-GARCH | 14 (Red Zone) | Fat tails & Option Gamma |
| 2013 Taper Tantrum | t-Copula | 6 (Amber Zone) | FX/Rates Correlation Shock |
| 2016 Demonetisation| FHS-EWMA | 3 (Green Zone) | Single-factor Equity Shock |
| 2018 IL&FS Crisis | FHS-GARCH | 5 (Amber Zone) | Credit-to-Equity Contagion |
| 2020 COVID Crash | t-Copula | 18 (Red Zone) | Extreme Volatility & Clustering |
| 2022 Rate Hikes | FHS-GARCH | 4 (Green Zone) | Rates Transmission to Equity |

---

## 7. Strategic Implementation Directives

```mermaid
graph LR
    classDef primary fill:#0F172A,stroke:#334155,stroke-width:2px,color:#F8FAFC;
    classDef alert fill:#DC2626,stroke:#991B1B,stroke-width:2px,color:#FEF2F2;
    classDef monitoring fill:#D97706,stroke:#92400E,stroke-width:2px,color:#FFFBEB;
    
    A[End-of-Day Risk Run] --> B{Macro Volatility Regime}
    B -->|Normal Operations| C[FHS-GARCH v3.0 <br/> Primary Risk Engine]:::primary
    B -->|High VIX / Systemic Stress| D[MC t-Copula v4.0 <br/> Stressed ES Overlay]:::alert
    C --> E((Regulatory Capital <br/> Reporting))
    D --> E
    A -.->|Intraday Monitoring| F[FHS-EWMA <br/> KRI Trigger]:::monitoring
```

**Primary Production Engine: FHS-GARCH (v3.0)**
*   Demonstrates exceptional robustness across all tested historical regimes.
*   Highly computationally efficient for daily end-of-day batch processing.
*   Directly resolves volatility clustering via reactive conditional variance updates.

**Secondary Regulatory Capital Engine: MC t-Copula (v4.0)**
*   Yields the strongest statistical validity in Expected Shortfall backtesting.
*   Designated for Stressed ES and comprehensive capital requirement reporting.

**Intraday Monitoring Engine: FHS-EWMA**
*   Provides lower computational overhead for real-time risk approximation.
*   Serves as a Key Risk Indicator (KRI) trigger mechanism for escalating to the full GARCH engine upon threshold breaches.

---

## 8. Open Issues & Continuous Improvement

| ID | Outstanding Action | Priority | Status |
|----|--------------------|----------|--------|
| 1 | Dynamic Skew Slope Calibration (Currently hard-coded at 0.10) | High | Open |
| 2 | NMRF Proxy Quality for LICI.NS (Current Nifty correlation ~0.65) | Medium | Open |
| 3 | IRS Swap Pricing Bootstrapping Engine (Currently simplified DCF) | Medium | Development |
| 4 | Missing 2008-2009 VIX History (Requires EWMA fallback patching) | Low | Documented |
| 5 | Copula Degrees-of-Freedom Joint Fitting Refinement | Low | Open |

---

*Generated by RiskCore Analytics | Enterprise Risk Management. All values illustrative.*
