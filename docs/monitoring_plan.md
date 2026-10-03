# Model Monitoring Plan

**Models Covered:** All Tier 1 and Tier 2 models in the MarketRisk-Lab inventory
**Review Frequency:** Daily (automated), Quarterly (manual)
**Owner:** Risk Methodologies Group

---

## 1. Daily Automated Monitoring

Automated via `marketrisk run-daily --date YYYY-MM-DD` (CLI) and GitLab CI scheduled pipeline.

| Check | Frequency | Tool | Alert Threshold |
|-------|-----------|------|----------------|
| VaR computation for all models | Daily | CLI | Failure → RED alert |
| Exception flagging (vs actual P&L) | Daily | monitoring.py | Any exception logged |
| Rolling 60-day breach rate | Daily | MonitoringFramework | > 3.5% AMBER, > 5% RED |
| Rolling Kupiec p-value | Daily | statistical_tests.py | < 0.10 AMBER, < 0.05 RED |
| Model stability (EWMA VaR change) | Daily | monitoring.py | > 15% AMBER, > 30% RED |
| VaR/|P&L| ratio | Daily | monitoring.py | > 3× AMBER, > 5× RED |
| Data quality | Daily | quality.py | ERROR/CRITICAL → escalate |

---

## 2. Quarterly Manual Review

| Activity | Responsibility | Output |
|----------|---------------|--------|
| Backtest report (all models, all regimes) | Risk Quant | Report to Model Committee |
| Parameter re-calibration review (λ, GARCH) | Risk Quant | Updated config or MRC approval |
| Stress scenario update (new events) | Risk Manager | Revised scenario set |
| Model limitation review | Risk Quant | MDD update |
| RNIV/NMRF reassessment | Risk Quant | Updated modellability status |
| Traffic light trend analysis | Risk Analyst | Dashboard update |

---

## 3. Escalation Matrix

| Status | Trigger | Notification | Action | Deadline |
|--------|---------|-------------|--------|---------|
| GREEN | All KRIs within thresholds | None | None | — |
| AMBER | Any KRI in amber zone | Risk Manager email | Schedule review; no capital action | 5 business days |
| RED | Any KRI in red zone | CRO + Head of Risk immediately | Convene Model Review; possible capital add-on; consider model suspension | Same day |

---

## 4. Alert Log

All alerts are written to `data/alerts.jsonl` (JSONL format).
Fields: `timestamp`, `date`, `status`, `model`, `kris`, `action`.

---

## 5. Model Change Trigger Conditions

A model change review is triggered when:
- Kupiec p-value < 0.05 for 3 consecutive months
- Traffic light RED for > 10 business days in a quarter
- New crisis event not covered by the stress scenario set
- Regulatory requirement changes (FRTB, BCBS updates)
- Data source disruption > 5 consecutive trading days

---

## 6. Governance Contacts

| Role | Responsibility |
|------|---------------|
| Risk Methodologies Group | Model development and maintenance |
| Model Validation (independent) | Annual validation, challenge findings |
| Chief Risk Officer | Final approval for Tier 1 model changes |
| Model Risk Committee | Quarterly review, escalation decisions |
