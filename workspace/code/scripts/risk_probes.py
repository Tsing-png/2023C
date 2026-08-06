import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")

# Load cleaned data
df1 = pd.read_csv(os.path.join(CLEAN, "product_info_with_loss.csv"))
df2 = pd.read_csv(os.path.join(CLEAN, "sales_normal_only.csv"))
df3 = pd.read_csv(os.path.join(CLEAN, "wholesale_prices.csv"))

# Map columns
item_col = df1.columns[0]
item_name_col = df1.columns[1]
cat_code_col = df1.columns[2]
cat_name_col = df1.columns[3]

sc_date = df2.columns[0]
sc_time = df2.columns[1]
sc_item = df2.columns[2]
sc_qty = df2.columns[3]
sc_price = df2.columns[4]
sc_discount = df2.columns[5]

wp_date = df3.columns[0]
wp_item = df3.columns[1]
wp_price = df3.columns[2]

# Convert dates
df2['date'] = pd.to_datetime(df2[sc_date]).dt.date
df3['date'] = pd.to_datetime(df3[wp_date]).dt.date

PROBE_DIR = os.path.join(BASE, "methods")
os.makedirs(os.path.join(PROBE_DIR, "Q1", "probes"), exist_ok=True)
os.makedirs(os.path.join(PROBE_DIR, "Q2", "probes"), exist_ok=True)
os.makedirs(os.path.join(PROBE_DIR, "Q3", "probes"), exist_ok=True)

# ============================================================
# Q1 PROBE: Correlation Analysis + Clustering
# ============================================================
print("=" * 60)
print("Q1 RISK PROBE")
print("=" * 60)

# Build category-level daily sales
cat_map = dict(zip(df1[item_col], df1[cat_name_col]))
df2_cat = df2.copy()
df2_cat['category'] = df2_cat[sc_item].map(cat_map)
cat_daily = df2_cat.groupby(['date', 'category'])[sc_qty].sum().reset_index()
cat_pivot = cat_daily.pivot(index='date', columns='category', values=sc_qty).fillna(0)

print(f"Category daily data: {cat_pivot.shape[0]} days x {cat_pivot.shape[1]} categories")

# --- Spearman correlation ---
spearman_corr = cat_pivot.corr(method='spearman')
print("\nSpearman correlation matrix:")
print(spearman_corr.round(3).to_string())

# Test significance
print("\nSpearman p-values:")
for i, c1 in enumerate(cat_pivot.columns):
    for c2 in cat_pivot.columns[i+1:]:
        r, p = stats.spearmanr(cat_pivot[c1], cat_pivot[c2])
        if p < 0.05:
            print(f"  {c1} vs {c2}: r={r:.3f}, p={p:.4f} *")

# --- DTW distance for item sales shape ---
# Aggregate to weekly for shape comparison (less noise)
df2['week'] = pd.to_datetime(df2[sc_date]).dt.isocalendar().week
df2['year'] = pd.to_datetime(df2[sc_date]).dt.isocalendar().year
item_weekly = df2.groupby(['year', 'week', sc_item])[sc_qty].sum().reset_index()

# Pivot: items x weeks
item_pivot_weekly = item_weekly.pivot_table(
    index=sc_item, columns=['year', 'week'], values=sc_qty, fill_value=0
)
print(f"\nItem-weekly pivot: {item_pivot_weekly.shape[0]} items x {item_pivot_weekly.shape[1]} weeks")

# Normalize each item's time series
item_norm = item_pivot_weekly.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1)
item_norm = item_norm.fillna(0)

# Fast DTW approximation: use correlation distance + euclidean on normalized
from scipy.spatial.distance import pdist, squareform
item_dist = pdist(item_norm.values, metric='correlation')
# Handle NaN in distance matrix
item_dist = np.nan_to_num(item_dist, nan=1.0)
print(f"Item distance matrix: {item_dist.shape[0]} pairwise distances")

# --- K-means++ clustering (literature standard) ---
# Cluster categories by sales characteristics
cat_features = pd.DataFrame({
    'mean_daily': cat_pivot.mean(),
    'std_daily': cat_pivot.std(),
    'cv': cat_pivot.std() / cat_pivot.mean(),
    'peak_to_mean': cat_pivot.max() / cat_pivot.mean(),
    'zero_days': (cat_pivot == 0).sum() / len(cat_pivot)
})
scaler = StandardScaler()
cat_features_scaled = scaler.fit_transform(cat_features)

# K-means with k=2,3
for k in [2, 3]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(cat_features_scaled)
    sil = silhouette_score(cat_features_scaled, labels)
    print(f"K-means k={k}: silhouette={sil:.3f}, labels={dict(zip(cat_pivot.columns, labels))}")

# --- Item clustering (on 120 items with >=90 sales days) ---
item_sales_days = df2.groupby(sc_item)['date'].nunique()
stable_items = item_sales_days[item_sales_days >= 90].index
print(f"\nStable items (>=90 days): {len(stable_items)}")

stable_item_data = item_norm.loc[item_norm.index.isin(stable_items)]
if len(stable_item_data) > 0:
    sdist = pdist(stable_item_data.values, metric='correlation')
    sdist = np.nan_to_num(sdist, nan=1.0)
    # Hierarchical clustering
    Z = linkage(sdist, method='ward')
    for k in [3, 4, 5]:
        labels = fcluster(Z, k, criterion='maxclust')
        sil = silhouette_score(stable_item_data.values, labels) if len(set(labels)) > 1 else 0
        print(f"Item HC k={k}: silhouette={sil:.3f}")

# --- Granger causality test ---
from statsmodels.tsa.stattools import grangercausalitytests
print("\nGranger causality tests (max lag=3):")
gc_results = {}
for c1 in cat_pivot.columns:
    for c2 in cat_pivot.columns:
        if c1 == c2:
            continue
        data = cat_pivot[[c2, c1]].dropna()
        try:
            gc = grangercausalitytests(data, maxlag=3, verbose=False)
            p_vals = [gc[lag][0]['ssr_chi2test'][1] for lag in [1, 2, 3]]
            min_p = min(p_vals)
            if min_p < 0.1:
                gc_results[f"{c1} -> {c2}"] = min_p
                print(f"  {c1} -> {c2}: min_p={min_p:.4f} (lag={p_vals.index(min_p)+1})")
        except:
            pass

# --- Output degeneracy check ---
print("\n--- Degeneracy Check ---")
# Category correlation should show variation
corr_vals = spearman_corr.values.flatten()
corr_vals = corr_vals[corr_vals < 1.0]  # exclude diagonal
print(f"Spearman corr range: [{corr_vals.min():.3f}, {corr_vals.max():.3f}]")
print(f"Spearman corr unique values: {len(np.unique(corr_vals.round(4)))}")
# Check if all zero (degenerate)
if np.allclose(corr_vals, 0, atol=0.05):
    print("WARNING: Near-zero correlations - output may be degenerate")
    degen_status = "CONDITIONAL"
else:
    print("OK: Correlations show variation")
    degen_status = "PASS"

# --- Perturbation sensitivity ---
# Bootstrap: resample days and recompute Spearman
np.random.seed(42)
orig_corr = spearman_corr.values
corr_diffs = []
for _ in range(100):
    idx = np.random.choice(len(cat_pivot), size=int(len(cat_pivot)*0.9), replace=True)
    boot_corr = cat_pivot.iloc[idx].corr(method='spearman').values
    corr_diffs.append(np.abs(boot_corr - orig_corr).mean())
perturb_mad = np.mean(corr_diffs)
print(f"Bootstrap perturbation MAD: {perturb_mad:.4f}")
print(f"Perturbation status: {'PASS' if perturb_mad < 0.05 else 'CONDITIONAL'}")

# ============================================================
# Q2 PROBE: Price-elasticity + Newsvendor
# ============================================================
print("\n" + "=" * 60)
print("Q2 RISK PROBE")
print("=" * 60)

# Build category-daily with pricing data
# Merge wholesale prices to sales
cat_daily_price = df2_cat.groupby(['date', 'category']).agg(
    total_qty=(sc_qty, 'sum'),
    avg_sales_price=(sc_price, 'mean'),
    discount_ratio=(sc_discount, lambda x: (x == '是').mean())
).reset_index()

# Get category average wholesale price per day
df3_cat = df3.copy()
df3_cat['category'] = df3_cat[wp_item].map(cat_map)
cat_wp_daily = df3_cat.groupby(['date', 'category'])[wp_price].mean().reset_index()

# Merge
cat_full = cat_daily_price.merge(cat_wp_daily, on=['date', 'category'], how='left')
# Forward fill missing wholesale (weekend gaps)
cat_full[wp_price] = cat_full.groupby('category')[wp_price].transform(lambda x: x.ffill())
cat_full = cat_full.dropna(subset=[wp_price])

# Add loss rates
cat_loss = df1.groupby(cat_name_col)['cat_loss_rate'].first().to_dict()
cat_full['loss_rate'] = cat_full['category'].map(cat_loss)

print(f"Category full dataset: {len(cat_full)} rows")

# --- Double-log demand model ---
import statsmodels.api as sm

# Prepare: log-log regression per category
# ln(Q) = a + sum(e_k * ln(P_k)) + e * ln(total_spend) + seasonal dummies
# Build price matrix for all categories per day
price_pivot = cat_full.pivot(index='date', columns='category', values='avg_sales_price').ffill()
qty_pivot = cat_full.pivot(index='date', columns='category', values='total_qty').fillna(0)
wp_pivot = cat_full.pivot(index='date', columns='category', values=wp_price).ffill()

# Total spend per day
total_spend = (qty_pivot * price_pivot).sum(axis=1)

# Run per-category regression
print("\nDouble-log demand model per category:")
elasticity_results = {}
for cat in cat_pivot.columns:
    y = np.log(qty_pivot[cat].clip(lower=1))
    X = pd.DataFrame()
    for cat2 in cat_pivot.columns:
        X[f'lnP_{cat2}'] = np.log(price_pivot[cat2].clip(lower=0.1))
    X['ln_spend'] = np.log(total_spend.clip(lower=1))
    X['dayofweek'] = pd.to_datetime(qty_pivot.index).dayofweek
    X['month'] = pd.to_datetime(qty_pivot.index).month
    X = sm.add_constant(X)

    valid = ~(y.isna() | X.isna().any(axis=1))
    try:
        model = sm.OLS(y[valid], X[valid].astype(float)).fit()
        own_elasticity = model.params.get(f'lnP_{cat}', np.nan)
        r2 = model.rsquared
        f_pval = model.f_pvalue
        # VIF check (simplified: max correlation among X)
        vif_max = X[valid].corr().abs().where(lambda x: x < 1).max().max()

        print(f"  {cat}: own_elasticity={own_elasticity:.3f}, R2={r2:.3f}, F_pval={f_pval:.4f}, max_corr={vif_max:.3f}")
        elasticity_results[cat] = {
            'own_elasticity': own_elasticity,
            'r2': r2,
            'f_pval': f_pval,
            'vif_max': vif_max
        }
    except Exception as e:
        print(f"  {cat}: FAILED - {e}")
        elasticity_results[cat] = {'error': str(e)}

# --- Newsvendor check ---
# Compute critical ratio: underage cost / (underage + overage)
# underage = lost profit (price - cost), overage = cost (no salvage)
sample_cat = '花叶类'
sample_data = cat_full[cat_full['category'] == sample_cat].iloc[-30:].copy()
if len(sample_data) > 0:
    avg_price = sample_data['avg_sales_price'].mean()
    avg_cost = sample_data[wp_price].mean()
    avg_loss = sample_data['loss_rate'].mean() / 100
    effective_cost = avg_cost / (1 - avg_loss)
    cu = avg_price - effective_cost  # underage cost
    co = effective_cost               # overage cost (no salvage on day-old vegetables)
    critical_ratio = cu / (cu + co) if (cu + co) > 0 else 0.5
    print(f"\n{sample_cat} sample: price={avg_price:.2f}, cost={avg_cost:.2f}, eff_cost={effective_cost:.2f}")
    print(f"  Underage={cu:.2f}, Overage={co:.2f}, Critical Ratio={critical_ratio:.3f}")
    print(f"  CR > 0.5 means stock more than mean demand")
    print(f"  CR < 0.5 means stock less than mean demand")

# --- Output degeneracy for optimization ---
print("\n--- Optimization Degeneracy Check ---")
# If elasticity ~ 0 or R2 < 0.1, the model degenerates
elasticity_issues = []
for cat, res in elasticity_results.items():
    if 'own_elasticity' in res:
        if abs(res['own_elasticity']) < 0.01:
            elasticity_issues.append(f"{cat}: near-zero elasticity ({res['own_elasticity']:.4f})")
        if res['r2'] < 0.1:
            elasticity_issues.append(f"{cat}: very low R2 ({res['r2']:.3f})")

if elasticity_issues:
    print("ISSUES:")
    for issue in elasticity_issues:
        print(f"  {issue}")
else:
    print("OK: All categories have meaningful elasticity estimates")

# ============================================================
# Q3 PROBE: SKU Selection Feasibility
# ============================================================
print("\n" + "=" * 60)
print("Q3 RISK PROBE")
print("=" * 60)

# Get June 24-30 tradeable items
june_start = pd.Timestamp('2023-06-24')
june_end = pd.Timestamp('2023-06-30')
june_mask = (pd.to_datetime(df2[sc_date]) >= june_start) & (pd.to_datetime(df2[sc_date]) <= june_end)
june_items = df2.loc[june_mask, sc_item].unique()
print(f"Items sold June 24-30: {len(june_items)}")

# Their category distribution
june_cats = df1[df1[item_col].isin(june_items)].groupby(cat_name_col).size()
print("\nCategory distribution of tradeable items:")
for cat, cnt in june_cats.items():
    print(f"  {cat}: {cnt}")

# Check: can we satisfy "at least k per category" with 27-33 total?
# With 6 categories, k=2 per category = 12 minimum, k=3 = 18 minimum, k=4 = 24 minimum
# 33 max / 6 = 5.5 per category max
for k in [2, 3, 4]:
    feasible = all(cnt >= k for cnt in june_cats)
    min_total = sum(max(k, 1) for cnt in june_cats)
    print(f"k={k}: feasible={feasible}, min_total_with_k={min_total}, fits_in_27_33={27 <= min_total <= 33}")

# Per-item sales during June 24-30
june_item_sales = df2.loc[june_mask].groupby(sc_item).agg(
    total_kg=(sc_qty, 'sum'),
    avg_price=(sc_price, 'mean'),
    days_sold=(sc_date, 'nunique')
)
# Merge category info
june_item_sales = june_item_sales.merge(
    df1[[item_col, item_name_col, cat_name_col, 'item_loss_rate']],
    left_index=True, right_on=item_col, how='left'
)
june_item_sales = june_item_sales.set_index(item_col)

# Get wholesale prices for June 24-30
june_wp = df3[(pd.to_datetime(df3[wp_date]) >= june_start) & (pd.to_datetime(df3[wp_date]) <= june_end)]
june_wp_avg = june_wp.groupby(wp_item)[wp_price].mean()
june_item_sales['avg_wholesale'] = june_item_sales.index.map(june_wp_avg)

# Compute profit margin estimate
june_item_sales['margin_est'] = (
    june_item_sales['avg_price'] -
    june_item_sales['avg_wholesale'] / (1 - june_item_sales['item_loss_rate']/100)
)
june_item_sales['profit_est'] = june_item_sales['margin_est'] * june_item_sales['total_kg']

print(f"\nJune 24-30 item stats:")
print(f"  Items with wholesale data: {june_item_sales['avg_wholesale'].notna().sum()}")
print(f"  Avg daily kg per item: {june_item_sales['total_kg'].mean()/7:.2f}")
print(f"  Items with avg daily > 2.5kg: {(june_item_sales['total_kg']/7 >= 2.5).sum()}")
print(f"  Items with positive estimated margin: {(june_item_sales['margin_est'] > 0).sum()}")
print(f"  Total items with all data (margin calculable): {june_item_sales['margin_est'].notna().sum()}")

# Feasibility check: can we select 27-33 items with each category >= 2?
print("\n--- Feasibility Check ---")
print(f"6 categories x min 2 = 12 minimum items (well within 27-33 range)")
print(f"Max items: 49 (all tradeable items)")
print(f"Feasible range: 12-49, target: 27-33")
print(f"At least 2 per category: {all(cnt >= 2 for cnt in june_cats)}")

# Check bottleneck category
min_cat = june_cats.idxmin()
min_cnt = june_cats.min()
print(f"Bottleneck category: {min_cat} ({min_cnt} items)")

print("\nDone. Probes complete.")
