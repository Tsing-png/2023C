# Q1 Code Plan

## Target and Round

- **Language**: Python 3
- **Round**: round1 (initial implementation)
- **Approved decision**: `q1_method_choice` → M1 (main), M2 (baseline)
- **Seed**: 2026

## Input Fields and Units

| Source | Fields | Type | Notes |
|---|---|---|---|
| `sales_normal_only.csv` | 销售日期, 单品编码, 销量(kg), 销售单价(元/kg), 是否折扣销售 | datetime, int64, float64, float64, str | 878,042 rows |
| `product_info_with_loss.csv` | 单品编码, 单品名称, 品类编码, 品类名称, cat_loss_rate, item_loss_rate | int64, str, int64, str, float64, float64 | 251 items, 6 categories |
| `wholesale_prices.csv` | 日期, 单品编码, 批发价格(元/kg) | datetime, int64, float64 | 55,982 rows |

## M1 — Main: Prophet Detrend + Spearman Residuals + DTW Shape Clustering + Granger Causal Network

### Step 1: Category-level daily aggregation

- Aggregate `sales_normal_only` to category-daily: sum `销量(kg)` per (date, category)
- Pivot to 6-column daily time series (1085 days × 6 categories)

### Step 2: Prophet detrending (per category)

- For each category's daily sales series, fit Prophet model: `Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)`
- Extract: trend, yearly seasonality, weekly seasonality
- Compute: **residual = actual - trend - yearly - weekly** (removes common temporal confounds)
- Save: trend components, seasonal components, residuals

### Step 3: Spearman correlation on residuals

- Compute Spearman ρ between all category-pair residuals (6×6 matrix)
- Compute Spearman ρ on **original** sales for comparison (M1 vs M2 contrast)
- Mark correlations with p < 0.05 as significant
- Generate: residual correlation heatmap + original correlation heatmap (side-by-side Figure Type 1)

### Step 4: Item-level DTW shape clustering

- Filter: only items with ≥ 90 unique sale dates (116 items — stable subset)
- Aggregate to weekly kg per item → normalize each item's time series to z-scores
- Compute pairwise correlation distance matrix (116 × 116)
- Hierarchical clustering (Ward linkage) on distance matrix → k=3,4,5
- K-means++ on the same data for cross-validation → k=3 clusters
- Silhouette score comparison (HC vs K-means)
- Save: cluster labels, silhouette scores, dendrogram plot

### Step 5: Granger causality test

- For each directed category pair (c1 → c2, c1 ≠ c2), test at lags 1,2,3
- Use daily sales (after ADF stationarity check; first-difference if needed)
- Record min p-value across lags for each pair → build 6×6 directed causality matrix
- Filter: only edges with p < 0.1 (Granger-significant)
- Generate: directed causal network diagram (Figure Type 2, candidate for paper)

### Step 6: Single-item sales distribution statistics

- For all 246 items: compute mean daily kg, std daily kg, CV, zero-sale-day ratio, seasonal peak quarter
- For 116 stable items: compute pairwise Spearman within each category (intra-category correlation heatmaps)
- Supply-source analysis: group items by base name (strip `(数字)` suffix) → compute within-group correlation vs cross-group correlation

### Outputs

| Type | Path | Description |
|---|---|---|
| Figure Type 1 | `results/Q1/experiments/round1/figures/q1_cat_seasonal_decompose.png` | 6-panel Prophet decomposition per category |
| Figure Type 1 | `results/Q1/experiments/round1/figures/q1_corr_comparison.png` | Side-by-side: raw Spearman vs residual Spearman |
| Figure Type 2 | `results/Q1/experiments/round1/figures/q1_granger_network.png` | Directed Granger causality network |
| Figure Type 1 | `results/Q1/experiments/round1/figures/q1_item_clusters.png` | DTW + K-means item clustering result |
| Figure Type 1 | `results/Q1/experiments/round1/figures/q1_category_sales_dist.png` | 6-panel daily sales distribution + monthly boxplots |
| Table | `results/Q1/experiments/round1/tables/q1_correlation_matrices.csv` | Both Spearman matrices (raw + residual) |
| Table | `results/Q1/experiments/round1/tables/q1_granger_pvalues.csv` | 6×6 Granger min-p matrix |
| Table | `results/Q1/experiments/round1/tables/q1_item_cluster_labels.csv` | 116 items with cluster assignments |
| Metrics | `results/Q1/experiments/round1/metrics/q1_metrics.json` | Silhouette scores, significant corr count, Granger edge count |
| Summary | `results/Q1/experiments/round1/run_summary.json` | Per contract |

## M2 — Baseline: Spearman Raw + K-means++

### Step 1: Raw Spearman correlation (category)

- Compute Spearman ρ directly on original daily category sales (1085 days × 6 categories)
- No detrending
- Significance test (Bonferroni corrected)

### Step 2: K-means++ clustering (category)

- Features per category: mean daily kg, std daily kg, CV, peak-to-mean ratio, zero-day ratio
- Standardize → K-means++ with k=2,3
- Report silhouette scores

### Step 3: K-means++ clustering (item)

- On the 116 stable items (weekly-normalized), run K-means++ with k=3,4,5
- Report silhouette scores
- Label interpretation: describe each cluster by category composition + sales magnitude

## Comparable Metrics (M1 vs M2)

| Metric | M1 (Main) | M2 (Baseline) | Comparison |
|---|---|---|---|
| Category correlation matrix | Residual Spearman | Raw Spearman | M1 should show **lower** correlations if time trend is a confound |
| # significant pairs (p<0.05) | From residuals | From raw data | M1 should be more conservative |
| Clustering silhouette | HC + K-means | K-means only | Direct silhouette comparison |
| Causal direction (added value) | Granger network | N/A | M1 adds directed causation that M2 cannot produce |
| Item cluster coherence | DTW-Ward HC | K-means on static features | Qualitative comparison of cluster interpretability |

## Risk Probe Conditions to Monitor

| Condition | Monitoring Logic |
|---|---|
| Stationarity for Granger (ADF) | If any category is non-stationary at p<0.05, first-difference before Granger test |
| DTW distance threshold | If silhouette < 0.15 for all k, flag DTW clustering as insufficiently separated |
| Prophet fit quality | If any category's Prophet R² < 0.3, note "weak trend decomposition — residual correlation may still contain trend" |
| Low-persistence items | Explicitly label 130 excluded items in report; compute their category distribution |

## Fallback Trigger

N/A — Q1 has no conditional fallback.

## Dependencies

- `pip install prophet` if not present (cmdstanpy backend)
- `pip install dtaidistance` (optional — fast DTW; fallback to scipy `pdist(metric='correlation')`)

## Environment

```json
{
  "python": "3.14.5",
  "pandas": "3.0.3",
  "numpy": "2.5.1",
  "scipy": "1.18.0",
  "sklearn": "1.9.0",
  "matplotlib": "3.11.1",
  "statsmodels": "present",
  "prophet": "to be installed",
  "seed": 2026
}
```
