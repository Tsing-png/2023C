# Q1 Baseline: Spearman Raw + K-means++ (literature standard)
# Approved decision: q1_method_choice → M2 (baseline)
# Seed: 2026

import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")
OUT = os.path.join(BASE, "results", "Q1", "experiments", "round1")
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

SEED = 2026
np.random.seed(SEED)

# ============================================================
# DATA LOADING
# ============================================================
df1 = pd.read_csv(os.path.join(CLEAN, "product_info_with_loss.csv"))
df2 = pd.read_csv(os.path.join(CLEAN, "sales_normal_only.csv"))

item_col, item_name_col, cat_code_col, cat_name_col = df1.columns[:4]
sc_date, sc_time, sc_item, sc_qty, sc_price, sc_discount = df2.columns[:6]

cat_map = dict(zip(df1[item_col], df1[cat_name_col]))
item_map = dict(zip(df1[item_col], df1[item_name_col]))

print("=" * 60)
print("Q1 BASELINE (M2): Raw Spearman + K-means++ Clustering")
print("=" * 60)

# ============================================================
# Category-Daily Aggregation
# ============================================================
df2['date'] = pd.to_datetime(df2[sc_date]).dt.date
df2['category'] = df2[sc_item].map(cat_map)
cat_daily = df2.groupby(['date', 'category'])[sc_qty].sum().reset_index()
cat_pivot = cat_daily.pivot(index='date', columns='category', values=sc_qty).fillna(0)
all_categories = list(cat_pivot.columns)
print(f"Category daily: {cat_pivot.shape[0]} days x {cat_pivot.shape[1]} categories")

# ============================================================
# STEP 1: Raw Spearman Correlation
# ============================================================
corr_raw = cat_pivot.corr(method='spearman')
print("\nRaw Spearman correlation (p-values with Bonferroni):")
bonferroni_threshold = 0.05 / 15  # 15 pairwise tests
sig_pairs = []
for i, c1 in enumerate(all_categories):
    for c2 in all_categories[i+1:]:
        r, p = stats.spearmanr(cat_pivot[c1], cat_pivot[c2])
        if p < bonferroni_threshold:
            sig_pairs.append((c1, c2, r, p))
            print(f"  {c1} <-> {c2}: r={r:.3f}, p={p:.6f} ***")
        elif p < 0.05:
            sig_pairs.append((c1, c2, r, p))
            print(f"  {c1} <-> {c2}: r={r:.3f}, p={p:.4f} *")

# ============================================================
# STEP 2: K-means++ Clustering (Category)
# ============================================================
cat_features = pd.DataFrame({
    'mean_daily_kg': cat_pivot.mean(),
    'std_daily_kg': cat_pivot.std(),
    'cv': cat_pivot.std() / cat_pivot.mean(),
    'peak_to_mean': cat_pivot.max() / cat_pivot.mean(),
    'zero_day_pct': (cat_pivot == 0).sum() / len(cat_pivot)
})
scaler = StandardScaler()
cat_feat_scaled = scaler.fit_transform(cat_features)

cat_cluster_results = {}
for k in [2, 3]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(cat_feat_scaled)
    sil = silhouette_score(cat_feat_scaled, labels)
    cat_cluster_results[k] = {
        'labels': dict(zip(all_categories, labels.astype(int).tolist())),
        'silhouette': float(sil)
    }
    print(f"\nCategory K-means k={k}: silhouette={sil:.3f}")
    for cat, lbl in zip(all_categories, labels):
        print(f"  {cat} → Cluster {lbl}")

# ============================================================
# STEP 3: K-means++ Clustering (Item)
# ============================================================
item_sale_days = df2.groupby(sc_item)['date'].nunique()
stable_items = item_sale_days[item_sale_days >= 90].index

# Weekly aggregation + normalization
df2['week'] = pd.to_datetime(df2[sc_date]).dt.isocalendar().week.astype(int)
df2['year'] = pd.to_datetime(df2[sc_date]).dt.isocalendar().year.astype(int)
item_weekly = df2.groupby(['year', 'week', sc_item])[sc_qty].sum().reset_index()
item_pivot_w = item_weekly.pivot_table(index=sc_item, columns=['year', 'week'], values=sc_qty, fill_value=0)
stable_pivot = item_pivot_w.loc[item_pivot_w.index.isin(stable_items)]
stable_norm = stable_pivot.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1).fillna(0)

item_cluster_results = {}
for k in [3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(stable_norm.values)
    sil = silhouette_score(stable_norm.values, labels)
    item_cluster_results[k] = {
        'silhouette': float(sil),
        'n_items': len(stable_norm),
        'cluster_sizes': dict(zip(*np.unique(labels, return_counts=True)))
    }
    # Category composition per cluster
    print(f"\nItem K-means k={k}: silhouette={sil:.3f}")
    for cl in sorted(set(labels)):
        cl_items = stable_norm.index[labels == cl]
        cl_cats = pd.Series([cat_map.get(i, 'unknown') for i in cl_items]).value_counts()
        print(f"  Cluster {cl} ({len(cl_items)} items): {cl_cats.to_dict()}")

# ============================================================
# OUTPUT: Figures
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# FIGURE B1: Category Spearman heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_raw.values, cmap='RdBu_r', vmin=-0.3, vmax=0.7, aspect='auto')
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(all_categories, fontsize=10)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f'{corr_raw.values[i, j]:.3f}', ha='center', va='center', fontsize=9)
ax.set_title('Q1 Baseline: Raw Spearman Correlation Matrix', fontsize=13)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q1_baseline_spearman.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: q1_baseline_spearman.png")

# FIGURE B2: Category cluster scatter (first 2 PCA components)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
cat_pca = pca.fit_transform(cat_feat_scaled)
k3_lbls = np.array(list(cat_cluster_results[3]['labels'].values()))
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(cat_pca[:, 0], cat_pca[:, 1], c=k3_lbls, s=200, cmap='Set2', edgecolors='k')
for i, cat in enumerate(all_categories):
    ax.annotate(cat, (cat_pca[i, 0], cat_pca[i, 1]), fontsize=10, xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Q1 Baseline: Category K-means Clusters (k=3, PCA projection)', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q1_baseline_category_clusters.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q1_baseline_category_clusters.png")

# ============================================================
# OUTPUT: Metrics
# ============================================================
metrics = {
    "spearman_mean_abs": float(np.abs(corr_raw.values[corr_raw.values < 1]).mean()),
    "n_sig_pairs_raw": len(sig_pairs),
    "n_sig_bonferroni": sum(1 for _, _, _, p in sig_pairs if p < bonferroni_threshold),
    "category_kmeans_silhouette": {str(k): v['silhouette'] for k, v in cat_cluster_results.items()},
    "item_kmeans_silhouette": {str(k): v['silhouette'] for k, v in item_cluster_results.items()},
    "n_stable_items": int(len(stable_norm))
}

with open(os.path.join(OUT, 'metrics', 'q1_baseline_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("Q1 BASELINE (M2) COMPLETE")
print(f"  Spearman mean |r| = {metrics['spearman_mean_abs']:.3f}")
print(f"  Significant pairs: {metrics['n_sig_pairs_raw']}")
print(f"  Category silhouette (k=3): {metrics['category_kmeans_silhouette']['3']:.3f}")
print(f"  Item silhouette (k=3): {metrics['item_kmeans_silhouette']['3']:.3f}")
print("=" * 60)
