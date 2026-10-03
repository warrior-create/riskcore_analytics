# Model Inventory — MarketRisk-Lab

**Maintainer:** Risk Methodologies Group
**Review Cycle:** Annual (or on model change)
**Last Updated:** 2024-Q4

---

## Tiering Framework

| Tier | Description | Review Frequency |
|------|-------------|-----------------|
| 1 – High | Material capital impact; regulatory reporting | Quarterly |
| 2 – Medium | Significant business decisions | Semi-annual |
| 3 – Low | Informational / indicative | Annual |

---

## Registered Models

| Model ID | Name | Version | Tier | Status | Owner | Approved | Next Review |
|----------|------|---------|------|--------|-------|---------|------------|
| VAR-PARAM-v1.0 | Parametric Delta-Normal VaR | 1.0 | 2 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| VAR-HS-v1.0 | Historical Simulation VaR (basic) | 1.0 | 2 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| VAR-HS-v2.0 | Historical Simulation VaR (full reval) | 2.0 | 1 | Production | Risk Methodologies | 2023-06-01 | 2025-06-01 |
| VAR-FHS-v3.0 | Filtered HS – GARCH(1,1) | 3.0 | 1 | Production | Risk Methodologies | 2023-09-01 | 2025-09-01 |
| VAR-FHS-EWMA-v3.0 | Filtered HS – EWMA | 3.0 | 1 | Production | Risk Methodologies | 2023-09-01 | 2025-09-01 |
| VAR-MC-NORM-v1.0 | Monte Carlo Gaussian VaR | 1.0 | 2 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| VAR-MC-TCOPULA-v4.0 | Monte Carlo t-Copula VaR | 4.0 | 1 | Production | Risk Methodologies | 2024-03-01 | 2026-03-01 |
| SES-v1.0 | Stressed ES (FRTB-simplified) | 1.0 | 1 | Production | Risk Methodologies | 2024-01-01 | 2026-01-01 |
| BLK76-v1.0 | Black-76 Options Pricer | 1.0 | 1 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| BOND-DCF-v1.0 | Bond DCF Pricer (G-sec) | 1.0 | 2 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| FX-CIP-v1.0 | FX Forward (CIP) Pricer | 1.0 | 2 | Production | Risk Methodologies | 2023-01-15 | 2025-01-15 |
| NMRF-v1.0 | NMRF / RNIV Assessment | 1.0 | 2 | Production | Risk Methodologies | 2024-01-01 | 2026-01-01 |

---

## Model Descriptions

### VAR-PARAM-v1.0 — Parametric Delta-Normal VaR
- **Purpose:** Baseline 1-day VaR at 99% and 95% for portfolio P&L distribution
- **Method:** Delta-normal, EWMA covariance (λ=0.94), normal distribution assumption
- **Key Assumptions:** Returns are normally distributed; portfolio is linear in risk factors
- **Key Limitations:**
  - Normal distribution underestimates fat tails (empirically shown in 2008 GFC and COVID)
  - Delta approximation invalid for short-dated options (gamma, vega effects ignored)
  - EWMA λ=0.94 calibrated on RiskMetrics data; not re-estimated
- **Output:** VaR, ES, EWMA covariance matrix

### VAR-FHS-v3.0 — Filtered Historical Simulation (GARCH)
- **Purpose:** Conditionally heteroscedastic VaR capturing volatility clustering
- **Method:** GARCH(1,1) standardised residuals, empirical distribution of residuals, current vol scaling
- **Key Improvements over v1/v2:** Captures vol regime shifts; reduces exception clustering
- **Key Limitations:**
  - GARCH(1,1) may not capture long-memory; EGARCH not implemented
  - Assumes residuals are iid across assets (independence in standardised space)
- **Output:** VaR, ES, conditional vol forecasts

### VAR-MC-TCOPULA-v4.0 — Monte Carlo t-Copula
- **Purpose:** Full tail-dependence model for joint extreme events
- **Method:** Student-t marginals (MLE), t-copula with profile-likelihood df estimation, 50K scenarios
- **Key Improvements:** Captures tail dependence; joint crash scenarios properly priced
- **Key Limitations:**
  - Copula df fitted jointly (not per-pair); simplified
  - Static copula: does not adapt to regime changes
  - Computationally expensive (50K × N assets scenarios per day)
- **Output:** VaR, ES, marginal distributions

---

## Version History

| Date | Model ID | Change | Approver |
|------|----------|--------|---------|
| 2023-01-15 | VAR-PARAM-v1.0 | Initial deployment (baseline) | Head of Risk |
| 2023-06-01 | VAR-HS-v2.0 | Added full revaluation; retired v1 HS | Head of Risk |
| 2023-09-01 | VAR-FHS-v3.0 | Added GARCH/EWMA filtering; approved after validation | Model Validation |
| 2024-01-01 | SES-v1.0 | Stressed ES with FRTB scaling | CRO |
| 2024-03-01 | VAR-MC-TCOPULA-v4.0 | t-copula MC; recommended primary model | Head of Risk |
