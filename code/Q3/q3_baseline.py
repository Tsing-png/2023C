# Q3 Baseline: Greedy Selection + Independent Pricing (Literature Standard)
# Seed: 2026

import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime
from scipy import optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")
OUT = os.path.join(BASE, "results", "Q3", "experiments", "round1")
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

SEED = 2026
np.random.seed(SEED)

print("=" * 60)
print("Q3 BASELINE (M2): Greedy Selection + Independent Pricing")
print("=" * 60)

# ============================================================
# DATA LOADING
# ============================================================
df1 = pd.read_csv(os.path.join(CLEAN, "product_info_with_loss.csv"))
df2 = pd.read_csv(os.path.join(CLEAN, "sales_normal_only.csv"))
df3 = pd.read_csv(os.path.join(CLEAN, "wholesale_prices.csv"))

item_col, item_name_col, cat_name_col = df1.columns[0], df1.columns[1], df1.columns[3]
sc_date, sc_item, sc_qty, sc_price, sc_discount = df2.columns[[0, 2, 3, 4, 5]]
wp_date, wp_item, wp_price = df3.columns[:3]

cat_map = dict(zip(df1[item_col], df1[cat_name_col]))

# ============================================================
# Identify Tradeable Items (June 24-30, 2023)
# ============================================================
df2['date_dt'] = pd.to_datetime(df2[sc_date])
df3['date_dt'] = pd.to_datetime(df3[wp_date])

june_mask = (df2['date_dt'] >= '2023-06-24') & (df2['date_dt'] <= '2023-06-30')

june_item_stats = df2[june_mask].groupby(sc_item).agg(
    total_kg_7d=(sc_qty, 'sum'),
    avg_price=(sc_price, 'mean'),
    days_sold=(sc_date, 'nunique')
).reset_index()

wp_avg = df3[(df3['date_dt'] >= '2023-06-20') & (df3['date_dt'] <= '2023-06-30')]
wp_avg = wp_avg.groupby(wp_item)[wp_price].mean().reset_index()
wp_avg.columns = [sc_item, 'avg_wholesale']

candidates = june_item_stats.merge(wp_avg, on=sc_item, how='left').dropna(subset=['avg_wholesale'])
candidates = candidates.merge(df1[[item_col, item_name_col, cat_name_col, 'item_loss_rate']], on=item_col, how='left')
candidates['eff_cost'] = candidates['avg_wholesale'] / (1 - candidates['item_loss_rate'].fillna(10)/100)
candidates['daily_kg'] = candidates['total_kg_7d'] / candidates['days_sold']
candidates['profit_per_kg'] = candidates['avg_price'] - candidates['eff_cost']
candidates['profit_per_day'] = candidates['profit_per_kg'] * candidates['daily_kg']

# ============================================================
# STEP 1: Greedy Selection (by profit_per_day, with category balance)
# ============================================================
# Sort by profit per day, then greedily pick ensuring >=2 per category
selected = []
per_cat_count = {cat: 0 for cat in candidates[cat_name_col].unique()}
MIN_PER_CAT = 2
TARGET_COUNT = 30

# Phase 1: ensure 2 per category
for cat in per_cat_count:
    cat_items = candidates[candidates[cat_name_col] == cat].nlargest(MIN_PER_CAT, 'profit_per_day')
    for _, item in cat_items.iterrows():
        selected.append(item[item_col])
        per_cat_count[cat] += 1

# Phase 2: fill remaining with highest profit
remaining = candidates[~candidates[item_col].isin(selected)]
remaining = remaining.sort_values('profit_per_day', ascending=False)
for _, item in remaining.iterrows():
    if len(selected) >= TARGET_COUNT:
        break
    selected.append(item[item_col])
    per_cat_count[item[cat_name_col]] += 1

selected_df = candidates[candidates[item_col].isin(selected)].copy()
print(f"Greedy selected: {len(selected_df)} items")
for cat, cnt in per_cat_count.items():
    print(f"  {cat}: {cnt}")

# ============================================================
# STEP 2: Independent Pricing (fixed markup r=0.20 for all)
# ============================================================
FIXED_MARKUP = 0.20
results = []
for _, row in selected_df.iterrows():
    eff_cost = row['eff_cost']
    price = eff_cost * (1 + FIXED_MARKUP)
    qty = max(row['daily_kg'], 2.5)
    profit = (price - eff_cost) * qty
    results.append({
        'item_code': row[item_col],
        'item_name': row[item_name_col],
        'category': row[cat_name_col],
        'eff_cost': round(float(eff_cost), 2),
        'sales_price': round(float(price), 2),
        'order_quantity_kg': round(float(qty), 2),
        'expected_profit': round(float(profit), 2)
    })

df_results = pd.DataFrame(results)

# ============================================================
# OUTPUT
# ============================================================
df_results.to_csv(os.path.join(OUT, 'tables', 'q3_baseline_selection.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: q3_baseline_selection.csv")

total_profit = df_results['expected_profit'].sum()
profit_by_cat = df_results.groupby('category')['expected_profit'].sum().to_dict()

metrics = {
    "n_items_selected": int(len(df_results)),
    "fixed_markup": FIXED_MARKUP,
    "total_profit_1d": round(float(total_profit), 2),
    "profit_by_category": {k: round(float(v), 2) for k, v in profit_by_cat.items()}
}

with open(os.path.join(OUT, 'metrics', 'q3_baseline_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("  Saved: q3_baseline_metrics.json")

print("\n" + "=" * 60)
print(f"Q3 BASELINE (M2) COMPLETE")
print(f"  SKUs: {len(df_results)}, Profit: {total_profit:,.0f} yuan (r=0.20)")
print(f"  Profit by category: {metrics['profit_by_category']}")
print("=" * 60)
