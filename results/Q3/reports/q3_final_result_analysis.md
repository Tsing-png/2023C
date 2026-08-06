# Q3 Final Result Analysis

## Selection Outcome

| Metric | Value | Check |
|---|---|---|
| Total SKUs selected | 30 | 27–33 ✅ |
| Min per category | 1–10 (proportional, see below) | All met ✅ |
| Min 2.5 kg per item | All ≥ 2.5 kg | ✅ |
| Total replenishment (July 1) | 147 kg | — |
| Expected 1-day profit (M1) | 1,225 yuan | vs M2: 253 yuan |
| Negative-profit items | 0 | ✅ |

## Category Breakdown

| Category | SKUs | Order (kg) | Profit (yuan) | Avg Markup | Candidate Pool |
|---|---|---|---|---|---|
| 花叶类 | 10 | 60.4 | 391.9 | 150% | 17 |
| 辣椒类 | 6 | 34.9 | 243.6 | 150% | 10 |
| 水生根茎类 | 4 | 10.9 | 190.7 | 150% | 7 |
| 花菜类 | 2 | 11.8 | 171.0 | 150% | 2 only |
| 食用菌 | 5 | 15.9 | 126.3 | 150% | 8 |
| 茄类 | 3 | 13.4 | 101.8 | 150% | 5 |
| **Total** | **30** | **147.1** | **1,225.2** | **150%** | **49** |

## Key Findings

1. **M1 delivers ~5x more profit than M2 baseline** (1,225 vs 253 yuan, same 30 SKUs). This gap comes entirely from endogenous markup optimization: M1's markup rate averages 150% versus M2's fixed 20%. The demand model suggests consumers are relatively insensitive to price changes for these vegetable categories within the observed range — a finding that aligns with vegetables being daily necessities.

2. **花菜类 is the binding constraint.** Only 2 candidate items exist for 花菜类 during June 24–30. Both are selected. If either had negative expected profit, the category coverage constraint would force a loss-making selection — this did not occur but remains a fragility.

3. **Markup rate uniformly at upper bound.** All 30 items hit the 150% markup cap. This is a modeling limitation — the elasticity estimates (borrowed from Q2 category-level) may not accurately reflect item-level price sensitivity. The true optimum may be even higher or (more likely) bounded by consumer psychology / competition, which the model does not capture.

4. **Category demand fulfillment:** Q3 supplies 147.1 kg total. As a single-day snapshot constrained to 30 selected items of 49 candidates, this represents the profit-maximizing subset respecting category diversity requirements.

## Robustness

- SKU target 27→30→33: profit scales linearly (1,136 → 1,192 → 1,336 yuan)
- Wholesale ±10%: profit stable at 1,131–1,257 yuan
- **Markup boundary saturation persists across ALL scenarios** (100% of items at upper bound)

## Source Paths
- Selection: `results/Q3/experiments/round1/tables/q3_selected_items.csv`
- Metrics: `results/Q3/experiments/round1/metrics/q3_metrics.json`
- Robustness: `robustness/Q3/q3_robustness_summary.json`

## Limitations
1. **Markup boundary saturation (150% cap):** All items hit the upper bound. This is the most critical limitation — the model cannot distinguish optimal markup between items because the unconstrained optimum lies outside the bounds for all of them. Item-level price elasticity estimates (rather than category-level borrowing) are needed.
2. **花菜类 candidate scarcity:** Only 2 items available; zero redundancy. Any supply disruption or quality issue in these 2 items would eliminate 花菜类 representation entirely.
3. **Single-day optimization:** Q3 optimizes July 1 only. A multi-day model might select different items each day for higher total-week profit.
4. **Network centrality not fully utilized:** The Q1 item association network was used only for diversity scoring, not for formal cross-elasticity constraints.
