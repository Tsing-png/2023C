# Q2 Final Result Analysis

## Main vs Baseline

| Metric | M1 (Newsvendor+Elasticity) | M2 (ARIMA+Deterministic) | Delta | Interpretation |
|---|---|---|---|---|
| 7-day expected profit | **2,587 yuan** | 3,817 yuan | −1,230 (−32%) | M1 correctly accounts for demand uncertainty |
| Avg markup rate | **95%** (endogenous) | 20% (fixed) | +75pp | M1 finds much higher optimal markups |
| Wholesale uncertainty | Prediction intervals + robust check | Point estimate only | — | M1 adds robustness dimension |
| Demand model R² | **0.51–0.85** (double-log) | 0.01–0.12 (linear) | — | Linear model fails — double-log essential |

## Key Findings

1. **M1 profit reflects real perishable economics (2,587 yuan vs M2 3,817 yuan).** The newsvendor model penalizes over-ordering (full cost loss on unsold vegetables) whereas M2's deterministic NLP treats demand as known. For perishable vegetables with full overage loss, conservative ordering is rational. M2's higher profit is misleading — it assumes perfect demand foresight, unrealistic for凌晨补货 decisions.

2. **Elasticity drives markup.** Optimal markup rates range from 8% to the upper bound of 150%. 花叶类、辣椒类、花菜类 all hit the 150% cap. This suggests that within the observed price range, demand is sufficiently inelastic to support aggressive markups. The 150% bound was imposed as a sanity constraint; true unconstrained optima may exceed this.

3. **Robustness.** Profit varies from −31% (elasticity+20%) to +55% (elasticity−20% & wholesale+10%). The demand sigma sensitivity is extreme: halving uncertainty → +135% profit; doubling → loss. This confirms that **demand forecasting accuracy is the single most impactful lever**.

4. **Wholesale price +10% increases profit.** Counterintuitive but correct: when wholesale rises, the endogenous markup model passes cost through to price. Since demand is inelastic in many categories, total revenue increases more than cost.

## Optimal Decisions (July 1–7, 2023)

| Category | Avg Daily Order (kg) | Avg Price (yuan/kg) | Avg Markup | 7-day Profit (yuan) |
|---|---|---|---|---|
| 花叶类 | 110.8 | 10.59 | 150% | 1,385.64 |
| 辣椒类 | 58.2 | 16.32 | 150% | 1,009.27 |
| 花菜类 | 16.4 | 23.44 | 150% | 148.34 |
| 辣椒类 | 6.8 | 7.00 | 46% | 33.77 |
| 水生根茎类 | 3.5 | 19.92 | 46% | 2.80 |
| 食用菌 | 5.1 | 6.66 | 29% | 7.31 |
| **Total** | **200.8** | — | — | **2,587.13** |

## Source Paths
- Decisions: `results/Q2/experiments/round1/tables/q2_optimal_decisions.csv`
- Metrics: `results/Q2/experiments/round1/metrics/q2_metrics.json`
- Robustness: `robustness/Q2/q2_robustness_summary.json`

## Limitations
1. Elasticity estimated from observational data — no exogenous price variation (no A/B test); estimates may be attenuated
2. Demand distribution assumed Normal for newsvendor — true distribution may be right-skewed (occasional spikes)
3. No competitor effects — prices assumed independent of competing supermarkets
4. Profit estimates are expected values; actual realized profit will vary with demand realization
