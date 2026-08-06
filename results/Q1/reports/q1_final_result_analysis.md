# Q1 Final Result Analysis

## Main vs Baseline

| Metric | M1 (Main: Detrended) | M2 (Baseline: Raw) | Delta | Interpretation |
|---|---|---|---|---|
| Spearman mean |r| | 0.266 | 0.393 | +0.127 (inflation) | Raw overestimates by ~48% due to shared time trends |
| Significant pairs (p<0.05) | 15/15 | 15/15 | — | All pairs significant in both |
| Granger edges (p<0.1) | 20/30 | N/A | — | M1 adds causal direction (M2 incapable) |
| Category clustering sil (k=3) | 0.306 | 0.306 | — | K-means identical (same features) |
| Item clustering sil (k=3) | HC 0.169 / KM 0.181 | KM 0.181 | — | Weak separation — limitation |

## Key Findings

1. **Detrending matters.** Common time trends inflate raw Spearman by 0.128 on average. 花叶类-花菜类 raw r=0.633 drops substantially after detrending. The residual correlations are more trustworthy indicators of genuine demand coupling.

2. **Bidirectional causality.** 20/30 directed category pairs show Granger causality (p<0.1). Categories do not move independently — they form an interconnected market system. 花叶类 Granger-causes 5/5 other categories; 食用菌 causes 5/5.

3. **Extreme sales concentration.** Item Gini = 0.793. Top 20% of items capture 83.9% of sales. This has direct implications for Q3 SKU selection: the long tail of low-volume items contributes little to revenue but may be needed for category diversity.

4. **Weak item clustering.** DTW-Ward HC silhouette = 0.17 — below the 0.25 threshold for "acceptable" separation. This indicates that item sales shapes are not strongly differentiated; items within a category tend to follow similar patterns. This is a model limitation.

## Source Paths
- Metrics: `results/Q1/experiments/round1/metrics/q1_metrics.json`
- Tables: `results/Q1/experiments/round1/tables/`
- Figures: `results/Q1/experiments/round1/figures/`
- Robustness: `robustness/Q1/q1_robustness_summary.json`

## Limitations
1. Silhouette scores for item clustering (0.17) are below the acceptable threshold — item-level typological structure is weak
2. STL decomposition used instead of Prophet (not installed) — yearly/weekly separation less clean
3. Granger tests assume stationarity; some categories required first-differencing which changes interpretation
4. 130/246 items excluded from item-level analysis (<90 sale days)
