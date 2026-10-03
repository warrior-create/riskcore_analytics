# Model Change Impact Analysis

**Document ID:** MDD-CIA-v1.0
**Change:** Migration from Legacy Parametric VaR (v1) to FHS-GARCH (v3) and t-Copula (v4)
**Date:** 2024-Q4

---

## 1. Description of Change

The legacy Excel/VBA tool calculated 99% 1-day VaR using a 250-day simple covariance matrix and a Delta-Normal assumption. 
The new target state introduces:
1. **FHS-GARCH (v3)** as the primary monitoring metric.
2. **t-Copula Monte Carlo (v4)** as the primary capital metric.
3. Transition from 99% VaR to 97.5% Expected Shortfall (ES).

---

## 2. Impact on Risk Metrics

Based on the 15-year walk-forward backtest on the identical test portfolio:

| Metric | Legacy (v1 Parametric) | Target State (v4 t-Copula ES) | Change |
|--------|------------------------|-------------------------------|--------|
| Baseline VaR (Normal Regime) | ~ 1.25% | ~ 1.55% | **+ 24%** |
| Stressed VaR (Crisis Regime) | ~ 3.10% | ~ 4.80% | **+ 55%** |
| 99% VaR Exceptions (15 yr) | ~ 150 | ~ 45 | **- 70%** |
| Capital Requirement (Multiplier) | Baseline x 3.0 | Baseline x 1.5 (due to better backtest) | **Net Neutral to -10%** |

---

## 3. Impact on Operations and IT

- **Compute Time:** Increased from < 1 second (Parametric) to ~ 15 seconds per run (GARCH/t-Copula). Still well within the overnight batch window.
- **Data Storage:** Now storing full scenarios and marginal distributions in SQLite. Storage footprint increased from ~10MB to ~150MB.
- **Reporting:** Replaced manual Excel macros with an automated Python CLI (`marketrisk run-daily`) and a Streamlit interactive dashboard.

---

## 4. Conclusion

The model change resolves the critical under-estimation of tail risk present in the legacy tool. While raw VaR and ES numbers are higher, the reduction in regulatory multipliers (due to fewer backtesting exceptions) makes the overall capital impact roughly neutral. The operational automation significantly reduces key-person risk and EUC spreadsheet risk.
