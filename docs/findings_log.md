# Model Validation Findings Log

| Finding ID | Date Raised | Severity | Description | Owner Response / Remediation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VAL-2024-001** | 2024-10-01 | High | Parametric VaR underestimates tail risk systematically, as seen in >10% Red Windows. | Acknowledged. We have introduced FHS-GARCH and t-Copula models which successfully pull the exceptions back into the Green Zone. We recommend decommissioning Parametric VaR. | Closed |
| **VAL-2024-002** | 2024-10-01 | Medium | Option synthetic history assumes constant flat skew over time. | Acknowledged. We have documented this limitation. Next version will integrate implied volatility surface snapshots from NSE. | Open |
| **VAL-2024-003** | 2024-10-02 | Medium | Stressed ES relies on historical window matching, which may not capture unobserved forward shocks. | Accepted. Added Hypothetical Scenarios (e.g. Nifty -20%) and Reverse Stress Testing via optimization to complement SES. | Closed |
| **VAL-2024-004** | 2024-10-02 | Low | Yield curve bootstraps using simplistic 3-point assumption, leading to minor basis risk in bond DCF. | Fixed. We have upgraded the pricing module to use Nelson-Siegel-Svensson calibration over all active maturities. | Closed |
| **VAL-2024-005** | 2024-10-03 | Low | FRTB P&L attribution uses Spearman/KS but did not strictly enforce the Basel threshold limits. | Fixed. We implemented `PLATest` to strictly enforce BCBS thresholds (Green if Spearman>=0.8, KS<=0.09). | Closed |
