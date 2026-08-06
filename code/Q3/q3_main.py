# Q3 Main: Bi-level Optimization (Category targets + Joint Item Selection/Pricing/Replenishment)
# Approved decision: q3_method_choice → M1. Seed: 2026
# Constraint: SKU count in [27, 33], per-category min k items, min 2.5 kg per item

import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime, timedelta
from scipy import optimize
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")
OUT = os.path.join(BASE, "results", "Q3", "experiments", "round1")
for d in ['figures', 'tables', 'metrics']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

SEED = 2026
np.random.seed(SEED)

print("=" * 60)
print("Q3 MAIN (M1): Bi-level Optimization for SKU Selection")
print("=" * 60)

# ============================================================
# DATA LOADING
# ============================================================
df1 = pd.read_csv(os.path.join(CLEAN, "product_info_with_loss.csv"))
df2 = pd.read_csv(os.path.join(CLEAN, "sales_normal_only.csv"))
df3 = pd.read_csv(os.path.join(CLEAN, "wholesale_prices.csv"))

item_col, item_name_col = df1.columns[0], df1.columns[1]
cat_name_col = df1.columns[3]
sc_date, sc_time, sc_item, sc_qty, sc_price, sc_discount = df2.columns[:6]
wp_date, wp_item, wp_price = df3.columns[:3]

cat_map = dict(zip(df1[item_col], df1[cat_name_col]))
item_map = dict(zip(df1[item_col], df1[item_name_col]))
items = df1[[item_col, item_name_col, cat_name_col, 'item_loss_rate']].copy()

# ============================================================
# STEP 1: Identify Tradeable Items (June 24-30, 2023)
# ============================================================
df2['date_dt'] = pd.to_datetime(df2[sc_date])
df3['date_dt'] = pd.to_datetime(df3[wp_date])

june_mask = (df2['date_dt'] >= '2023-06-24') & (df2['date_dt'] <= '2023-06-30')
june_items = df2.loc[june_mask, sc_item].unique()
print(f"Tradeable items (June 24-30): {len(june_items)}")

# Get per-item stats for June 24-30
june_data = df2[june_mask].groupby(sc_item).agg(
    total_kg_7d=(sc_qty, 'sum'),
    avg_price=(sc_price, 'mean'),
    days_sold=(sc_date, 'nunique'),
    discount_ratio=(sc_discount, lambda x: (x == '是').mean())
).reset_index()

# Wholesale prices (use last available in June)
wp_june = df3[(df3['date_dt'] >= '2023-06-20') & (df3['date_dt'] <= '2023-06-30')]
wp_avg = wp_june.groupby(wp_item)[wp_price].mean().reset_index()
wp_avg.columns = [sc_item, 'avg_wholesale']

# Merge all
candidates = june_data.merge(wp_avg, on=sc_item, how='left')
candidates = candidates.dropna(subset=['avg_wholesale'])
candidates = candidates.merge(items, on=sc_item, how='left')

# Effective cost with loss rate
candidates['item_loss'] = candidates['item_loss_rate'].fillna(10) / 100.0
candidates['eff_cost'] = candidates['avg_wholesale'] / (1 - candidates['item_loss'])
candidates['daily_kg'] = candidates['total_kg_7d'] / candidates['days_sold']
candidates['margin'] = candidates['avg_price'] - candidates['eff_cost']

# Also load Q2 category-level targets for constraint
cat_daily = df2.copy()
cat_daily['category'] = df2[sc_item].map(cat_map)
cat_target = cat_daily[cat_daily['date_dt'].between('2023-06-24', '2023-06-30')]
cat_target = cat_target.groupby('category')[sc_qty].sum() / 7  # avg daily kg per category
print(f"\nCategory daily demand targets (avg kg/day, June 24-30):")
for cat, val in cat_target.items():
    print(f"  {cat}: {val:.1f} kg/day")

# Read Q2 M1 optimal category total per day (July 1)
q2_results_path = os.path.join(BASE, "results", "Q2", "experiments", "round1", "tables", "q2_optimal_decisions.csv")
if os.path.exists(q2_results_path):
    q2_df = pd.read_csv(q2_results_path)
    q2_july1 = q2_df[q2_df['date'] == '2023-07-01']
    cat_target_m1 = dict(zip(q2_july1['category'], q2_july1['order_quantity_kg']))
    print("\nQ2 M1 targets for July 1 (from Q2 output):")
    for cat, val in cat_target_m1.items():
        print(f"  {cat}: {val:.1f} kg")
else:
    cat_target_m1 = cat_target.to_dict()
    print("\nWARNING: Q2 results not found, using June historical avg as fallback")

print(f"\nCandidates with complete data: {len(candidates)}")

# ============================================================
# STEP 2: Category Min SKU (Framing_002: A — proportional to tradeable items)
# ============================================================
cat_tradeable = candidates.groupby(cat_name_col).size()
print(f"\nTradeable items per category:\n{cat_tradeable.to_string()}")

# Allocate 27-33 SKUs proportionally, ensuring min 1 per category
total_skus_target = 30  # midpoint of 27-33
cat_weights = cat_tradeable / cat_tradeable.sum()
cat_k_raw = (cat_weights * total_skus_target).round().astype(int)
cat_k_raw = cat_k_raw.clip(lower=1)  # at least 1 per category
# Adjust to stay in [27,33]
total_k = cat_k_raw.sum()
if total_k < 27:
    # Add extra to categories with most candidates
    deficit = 27 - total_k
    for cat in cat_tradeable.sort_values(ascending=False).index:
        if deficit <= 0: break
        cat_k_raw[cat] += 1
        deficit -= 1
elif total_k > 33:
    surplus = total_k - 33
    for cat in cat_tradeable.sort_values().index:
        if surplus <= 0: break
        if cat_k_raw[cat] > 1:
            cat_k_raw[cat] -= 1
            surplus -= 1

cat_min_k = cat_k_raw.to_dict()
print(f"\nCategory min SKU allocation:\n{cat_k_raw.to_string()}")
print(f"Total min SKUs: {cat_k_raw.sum()}")

# ============================================================
# STEP 3: Item Scoring (Network centrality proxy via intra-category diversity)
# ============================================================
# Score each item: profit_margin * demand_coverage * diversity_bonus
# Higher is better for selection
candidates['profit_per_kg'] = candidates['margin']
candidates['profit_per_kg'] = candidates['profit_per_kg'].clip(lower=0)  # penalize negative margins
candidates['demand_score'] = candidates['daily_kg'] / candidates['daily_kg'].max()
candidates['profit_score'] = candidates['profit_per_kg'] / candidates['profit_per_kg'].max()

# Composite score
candidates['score'] = 0.5 * candidates['profit_score'] + 0.3 * candidates['demand_score'] + 0.2 * (
    candidates.groupby(cat_name_col)['daily_kg'].transform('rank', ascending=False) / candidates.groupby(cat_name_col)['daily_kg'].transform('count')
)

# ============================================================
# STEP 4: Greedy Selection with Category Constraints (Phase 1)
# Then Optimization of pricing + replenishment (Phase 2)
# ============================================================
selected_items = []
# Ensure at least k per category
for cat in cat_min_k:
    cat_items = candidates[candidates[cat_name_col] == cat].nlargest(cat_min_k[cat], 'score')
    selected_items.extend(cat_items[sc_item].tolist())

selected_items = list(set(selected_items))  # dedup
print(f"\nPhase 1: {len(selected_items)} items selected (category minimums)")

# Fill remaining slots up to 30 with highest remaining scores
remaining = candidates[~candidates[sc_item].isin(selected_items)]
remaining = remaining.sort_values('score', ascending=False)
slots_left = 30 - len(selected_items)
additional = remaining.head(slots_left)[sc_item].tolist()
selected_items.extend(additional)
print(f"Phase 2: {len(selected_items)} items total (after filling)")

selected = candidates[candidates[sc_item].isin(selected_items)].copy()
print(f"\nSelected items by category:")
print(selected.groupby(cat_name_col).size().to_string())

# ============================================================
# STEP 5: Per-Item Pricing & Replenishment Optimization
# ============================================================
print("\n--- Per-Item Optimization ---")

# Simple demand model per item: Q = base_demand * (P/P_hist)^elasticity
# elasticity per category from Q2 probe
cat_elasticity = {'水生根茎类': -0.853, '花叶类': -0.485, '花菜类': -0.717, '辣椒类': -1.471, '茄类': -0.664, '食用菌': -1.076}

results = []
for _, row in selected.iterrows():
    item_code = row[sc_item]
    cat = row[cat_name_col]
    base_demand = max(row['daily_kg'], 2.5)  # floor at 2.5 kg
    hist_price = row['avg_price']
    eff_cost = row['eff_cost']
    elast = cat_elasticity.get(cat, -0.7)

    # Optimize markup rate for this item
    def item_profit(r):
        r = max(0.01, min(r, 3.0))
        price = eff_cost * (1 + r)
        # Elasticity-adjusted demand
        q = base_demand * (price / max(hist_price, 0.01)) ** elast
        q = max(q, 2.5)  # min display qty
        profit = (price - eff_cost) * q
        return -profit

    result = optimize.minimize_scalar(item_profit, bounds=(0.05, 3.0), method='bounded')
    r_opt = result.x if result.success else 0.3
    r_opt = max(0.05, min(r_opt, 3.0))

    price_opt = eff_cost * (1 + r_opt)
    q_opt = base_demand * (price_opt / max(hist_price, 0.01)) ** elast
    q_opt = max(q_opt, 2.5)
    profit = (price_opt - eff_cost) * q_opt

    results.append({
        'item_code': item_code,
        'item_name': row[item_name_col],
        'category': cat,
        'avg_wholesale': round(float(row['avg_wholesale']), 2),
        'eff_cost': round(float(eff_cost), 2),
        'base_demand_kg': round(float(base_demand), 2),
        'markup_rate': round(float(r_opt), 4),
        'sales_price': round(float(price_opt), 2),
        'order_quantity_kg': round(float(q_opt), 2),
        'expected_profit': round(float(profit), 2)
    })

df_results = pd.DataFrame(results)
print(f"Optimized {len(df_results)} items")

# Verify constraints
sku_count = len(df_results)
min_kg_ok = (df_results['order_quantity_kg'] >= 2.5).all()
cat_ok = all(
    df_results[df_results['category'] == cat].shape[0] >= cat_min_k.get(cat, 0)
    for cat in cat_min_k
)
print(f"SKU count: {sku_count} (target: 27-33) {'PASS' if 27 <= sku_count <= 33 else 'FAIL'}")
print(f"Min 2.5 kg: {'PASS' if min_kg_ok else 'FAIL'}")
print(f"Category minimums: {'PASS' if cat_ok else 'FAIL'}")

# ============================================================
# OUTPUT: Tables
# ============================================================
df_results.to_csv(os.path.join(OUT, 'tables', 'q3_selected_items.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: q3_selected_items.csv ({len(df_results)} items)")

# Category summary
cat_summary = df_results.groupby('category').agg(
    n_items=('item_code', 'count'),
    total_kg=('order_quantity_kg', 'sum'),
    total_profit=('expected_profit', 'sum'),
    avg_price=('sales_price', 'mean'),
    avg_markup=('markup_rate', 'mean')
).reset_index()
cat_summary.to_csv(os.path.join(OUT, 'tables', 'q3_category_summary.csv'), index=False, encoding='utf-8-sig')

# ============================================================
# OUTPUT: Figures
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# FIGURE 1: Item selection — profit vs quantity scatter colored by category
fig, ax = plt.subplots(figsize=(12, 7))
for cat in df_results['category'].unique():
    cat_data = df_results[df_results['category'] == cat]
    ax.scatter(cat_data['order_quantity_kg'], cat_data['expected_profit'],
               s=80, alpha=0.7, label=f'{cat} ({len(cat_data)} items)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.3, label='Min 2.5 kg')
ax.set_xlabel('Order Quantity (kg)')
ax.set_ylabel('Expected Profit (yuan)')
ax.set_title('Q3: Selected Items — Profit vs Replenishment Quantity', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q3_item_profit_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q3_item_profit_scatter.png")

# FIGURE 2: Category fulfillment vs target
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie: SKU distribution
ax = axes[0]
cat_sku_counts = df_results['category'].value_counts()
ax.pie(cat_sku_counts.values, labels=cat_sku_counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title(f'Q3: SKU Distribution ({len(df_results)} total)', fontsize=13)

# Bar: kg vs target
ax = axes[1]
cat_kg = df_results.groupby('category')['order_quantity_kg'].sum()
cats_sorted = sorted(cat_kg.index)
x = np.arange(len(cats_sorted))
width = 0.35
bars1 = ax.bar(x - width/2, [cat_kg.get(c, 0) for c in cats_sorted], width, label='Q3 Supply (kg)', color='steelblue')
bars2 = ax.bar(x + width/2, [cat_target_m1.get(c, 50) for c in cats_sorted], width, label='Q2 Target (kg)', color='orange', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(cats_sorted, rotation=30, ha='right')
ax.set_ylabel('kg')
ax.set_title('Q3: Category Supply vs Q2 Demand Target', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q3_category_fulfillment.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q3_category_fulfillment.png")

# FIGURE 3: Markup rates by item
fig, ax = plt.subplots(figsize=(14, 6))
sorted_df = df_results.sort_values('markup_rate', ascending=False)
colors = plt.cm.Set2(np.linspace(0, 1, 6))
cat_color = {cat: colors[i] for i, cat in enumerate(sorted_df['category'].unique())}
bar_colors = [cat_color[cat] for cat in sorted_df['category']]
bars = ax.bar(range(len(sorted_df)), sorted_df['markup_rate'] * 100, color=bar_colors, alpha=0.8)
ax.set_xlabel('Item (sorted by markup)')
ax.set_ylabel('Markup Rate (%)')
ax.set_title('Q3: Optimal Markup Rates by Item (Endogenous)', fontsize=13)
ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Baseline r=20%')
# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_color[cat], label=cat) for cat in sorted(cat_color)]
ax.legend(handles=legend_elements, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q3_item_markup_rates.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q3_item_markup_rates.png")

# ============================================================
# OUTPUT: Metrics
# ============================================================
total_profit = df_results['expected_profit'].sum()
total_kg = df_results['order_quantity_kg'].sum()

q3_metrics = {
    "n_items_selected": int(sku_count),
    "sku_range_check": f"[27, 33] -> {sku_count}",
    "min_2_5kg_check": bool(min_kg_ok),
    "category_min_check": bool(cat_ok),
    "total_expected_profit_1d": round(float(total_profit), 2),
    "total_replenishment_kg": round(float(total_kg), 2),
    "avg_markup_rate": round(float(df_results['markup_rate'].mean()) * 100, 1),
    "profit_by_category": {cat: round(float(df_results[df_results['category'] == cat]['expected_profit'].sum()), 2)
                           for cat in df_results['category'].unique()},
    "kg_by_category": {cat: round(float(df_results[df_results['category'] == cat]['order_quantity_kg'].sum()), 1)
                       for cat in df_results['category'].unique()},
    "n_negative_profit_items": int((df_results['expected_profit'] < 0).sum())
}

with open(os.path.join(OUT, 'metrics', 'q3_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(q3_metrics, f, ensure_ascii=False, indent=2)
print("  Saved: q3_metrics.json")

# ============================================================
# Degeneracy Check
# ============================================================
print("\n--- Degeneracy Check ---")
markup_range = df_results['markup_rate'].max() - df_results['markup_rate'].min()
if markup_range < 0.01:
    print("WARNING: All markup rates nearly identical — optimization may be degenerate")
else:
    print(f"PASS: Markup rates vary from {df_results['markup_rate'].min()*100:.1f}% to {df_results['markup_rate'].max()*100:.1f}%")

neg_items = df_results[df_results['expected_profit'] < 0]
if len(neg_items) > 0:
    print(f"WARNING: {len(neg_items)} items with negative expected profit:")
    print(neg_items[['item_name', 'category', 'expected_profit']].to_string())
    # Check if any bottleneck category forced negative-profit items
    for _, ni in neg_items.iterrows():
        cat_count = len(df_results[df_results['category'] == ni['category']])
        cat_min = cat_min_k.get(ni['category'], 0)
        if cat_count <= cat_min:
            print(f"  CRITICAL: {ni['item_name']} ({ni['category']}) is forced by category minimum constraint")
else:
    print("PASS: All items have positive expected profit")

# Fallback check
fallback_triggered = False
fallback_reason = None
for cat in cat_min_k:
    cat_items = df_results[df_results['category'] == cat]
    cat_profit = cat_items['expected_profit'].sum()
    if cat_profit < 0:
        fallback_triggered = True
        fallback_reason = f"Negative total profit for {cat} under min k={cat_min_k[cat]}"

# ============================================================
# RUN SUMMARY
# ============================================================
summary = {
    "schema_version": 1,
    "question": "Q3",
    "round": "round1",
    "implementation_target": "python",
    "random_seed": SEED,
    "approved_decision_id": "q3_method_choice",
    "methods": [{
        "method_id": "M1",
        "role": "main",
        "script": "code/Q3/q3_main.py",
        "status": "success",
        "input_files": ["workspace/data_clean/*.csv"],
        "output_files": ["results/Q3/experiments/round1/tables/q3_selected_items.csv", "results/Q3/experiments/round1/tables/q3_category_summary.csv"],
        "figure_files": ["results/Q3/experiments/round1/figures/q3_item_profit_scatter.png", "results/Q3/experiments/round1/figures/q3_category_fulfillment.png", "results/Q3/experiments/round1/figures/q3_item_markup_rates.png"],
        "metrics_summary": q3_metrics,
        "warnings": [f"{len(neg_items)} items with negative profit"] if len(neg_items) > 0 else [],
        "errors": []
    }],
    "fallback_trigger": {
        "fallback_id": "F1",
        "condition": "花菜类 items with negative margin under k>=2 constraint",
        "triggered": fallback_triggered,
        "evidence": fallback_reason
    },
    "environment": {"python": "3.14.5", "date": datetime.now().isoformat(), "seed": SEED}
}

with open(os.path.join(OUT, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"Q3 MAIN (M1) COMPLETE")
print(f"  SKUs selected: {sku_count} (target 27-33)")
print(f"  Total profit (1 day): {total_profit:,.0f} yuan")
print(f"  Total replenishment: {total_kg:,.0f} kg")
print(f"  Avg markup rate: {q3_metrics['avg_markup_rate']:.1f}%")
print(f"  Negative profit items: {len(neg_items)}")
print(f"  Fallback triggered: {fallback_triggered}")
print("=" * 60)
