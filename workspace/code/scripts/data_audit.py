import pandas as pd
import numpy as np
import os, json, hashlib, re
from datetime import datetime

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
RAW = os.path.join(BASE, "workspace", "problem")
CLEAN = os.path.join(BASE, "workspace", "data_clean")
os.makedirs(CLEAN, exist_ok=True)

# ============================================================
# ATTACHMENT 1: Product Info
# ============================================================
print("=" * 60)
print("ATTACHMENT 1: Product Info")
print("=" * 60)
df1 = pd.read_excel(os.path.join(RAW, "附件1.xlsx"))
item_code_col, item_name_col, cat_code_col, cat_name_col = df1.columns
print(f"Shape: {df1.shape}")
print(f"Dtypes:\n{df1.dtypes}")
print(f"Nulls:\n{df1.isnull().sum()}")
print(f"Category distribution:")
for cat, cnt in df1.groupby(cat_name_col).size().sort_values(ascending=False).items():
    code = df1[df1[cat_name_col]==cat][cat_code_col].iloc[0]
    print(f"  {cat} ({code}): {cnt} items")

# Supplier number in name
has_num = df1[item_name_col].apply(lambda n: bool(re.search(r'\(\d+\)', str(n)))).sum()
print(f"Items with supplier number: {has_num}/{len(df1)}")

# Check: item_code <-> cat_code consistency
ic_cc_pairs = df1.groupby(item_code_col)[cat_code_col].nunique()
if (ic_cc_pairs > 1).any():
    print("WARNING: Some item codes map to multiple category codes!")

# Write clean copy (附件1 is clean structurally)
df1.to_csv(os.path.join(CLEAN, "product_info.csv"), index=False, encoding='utf-8-sig')
print("Cleaned: product_info.csv")

# ============================================================
# ATTACHMENT 2: Sales Transactions
# ============================================================
print("\n" + "=" * 60)
print("ATTACHMENT 2: Sales Transactions")
print("=" * 60)
df2 = pd.read_excel(os.path.join(RAW, "附件2.xlsx"))
sc_date, sc_time, sc_item, sc_qty, sc_price, sc_type, sc_discount = df2.columns
n_total = len(df2)
print(f"Shape: {df2.shape}")
print(f"Date range: {df2[sc_date].min()} ~ {df2[sc_date].max()}")
print(f"Nulls:\n{df2.isnull().sum()}")

# Parse scan time
df2['scan_hour'] = df2[sc_time].astype(str).str[:2].str.extract(r'(\d+)').astype(float)
print(f"Scan hour range: {df2['scan_hour'].min():.0f} ~ {df2['scan_hour'].max():.0f}")

# Sales type and discount
print(f"Sales type: {df2[sc_type].value_counts().to_dict()}")
print(f"Discount: {df2[sc_discount].value_counts().to_dict()}")

# Returns
returns = df2[df2[sc_type] == '退货']
print(f"Returns: {len(returns)} ({len(returns)/n_total*100:.3f}%)")

# Discount transactions
discounts = df2[df2[sc_discount] == '是']
print(f"Discount rows: {len(discounts)} ({len(discounts)/n_total*100:.2f}%)")

# Negative quantities
neg_qty = df2[df2[sc_qty] < 0]
print(f"Negative qty rows: {len(neg_qty)}")

# Zero prices
zero_p = df2[df2[sc_price] <= 0]
print(f"Zero/negative price rows: {len(zero_p)}")

# Quantity stats
print(f"Qty stats (kg): mean={df2[sc_qty].mean():.3f}, std={df2[sc_qty].std():.3f}, "
      f"min={df2[sc_qty].min():.4f}, Q99={df2[sc_qty].quantile(0.99):.2f}, max={df2[sc_qty].max():.2f}")

# Price stats
print(f"Price stats (yuan/kg): mean={df2[sc_price].mean():.2f}, std={df2[sc_price].std():.2f}, "
      f"min={df2[sc_price].min():.2f}, Q1={df2[sc_price].quantile(0.25):.2f}, "
      f"median={df2[sc_price].quantile(0.5):.2f}, Q3={df2[sc_price].quantile(0.75):.2f}, "
      f"max={df2[sc_price].max():.2f}")

# Daily aggregation
df2['date_only'] = df2[sc_date].dt.date
daily_kg = df2.groupby('date_only')[sc_qty].sum()
print(f"Daily volume (kg): mean={daily_kg.mean():.1f}, std={daily_kg.std():.1f}, "
      f"min={daily_kg.min():.1f}, max={daily_kg.max():.1f}")

# Item sales persistence
item_date_cnt = df2.groupby(sc_item)['date_only'].nunique()
print(f"Item sales days: Q1={item_date_cnt.quantile(0.25):.0f}, median={item_date_cnt.quantile(0.5):.0f}, "
      f"Q3={item_date_cnt.quantile(0.75):.0f}, max={item_date_cnt.max()}")
print(f"Items sold <10 days: {(item_date_cnt < 10).sum()}, <30 days: {(item_date_cnt < 30).sum()}")

# Per-item daily aggregation (for downstream modeling)
df2_daily_item = df2.groupby(['date_only', sc_item]).agg(
    total_qty=(sc_qty, 'sum'),
    avg_price=(sc_price, 'mean'),
    discount_ratio=(sc_discount, lambda x: (x == '是').mean()),
    return_qty=(sc_qty, lambda x: x[df2.loc[x.index, sc_type] == '退货'].sum() if (df2.loc[x.index, sc_type] == '退货').any() else 0)
).reset_index()

# Separate returns and non-returns for clean sales
mask_normal = df2[sc_type] == '销售'
df2_normal = df2[mask_normal].copy()
print(f"Normal sales rows: {len(df2_normal)}")

# Mark returns removed
df2_clean = df2_normal.drop(columns=[sc_type]).copy()
df2_clean.to_csv(os.path.join(CLEAN, "sales_normal_only.csv"), index=False, encoding='utf-8-sig')
print("Cleaned: sales_normal_only.csv (returns excluded)")

# ============================================================
# ATTACHMENT 3: Wholesale Prices
# ============================================================
print("\n" + "=" * 60)
print("ATTACHMENT 3: Wholesale Prices")
print("=" * 60)
df3 = pd.read_excel(os.path.join(RAW, "附件3.xlsx"))
wp_date, wp_item, wp_price = df3.columns
print(f"Shape: {df3.shape}")
print(f"Date range: {df3[wp_date].min()} ~ {df3[wp_date].max()}")
print(f"Unique items: {df3[wp_item].nunique()}")
print(f"Nulls:\n{df3.isnull().sum()}")

# Price stats
print(f"Wholesale price: mean={df3[wp_price].mean():.2f}, std={df3[wp_price].std():.2f}, "
      f"min={df3[wp_price].min():.2f}, Q1={df3[wp_price].quantile(0.25):.2f}, "
      f"median={df3[wp_price].quantile(0.5):.2f}, Q3={df3[wp_price].quantile(0.75):.2f}, "
      f"max={df3[wp_price].max():.2f}")

# Extreme prices (>100 yuan/kg)
extreme = df3[df3[wp_price] > 100]
print(f"Wholesale price >100 yuan/kg: {len(extreme)} rows")
if len(extreme) > 0:
    extreme_items = extreme[wp_item].unique()
    extreme_names = df1[df1[item_code_col].isin(extreme_items)][[item_code_col, item_name_col]]
    print("Extreme price items:")
    print(extreme_names.to_string())

# Item coverage
item_wp_cnt = df3.groupby(wp_item)[wp_date].nunique()
print(f"Item wholesale records: Q1={item_wp_cnt.quantile(0.25):.0f}, median={item_wp_cnt.quantile(0.5):.0f}, "
      f"Q3={item_wp_cnt.quantile(0.75):.0f}, min={item_wp_cnt.min()}, max={item_wp_cnt.max()}")
print(f"Items with <30 records: {(item_wp_cnt < 30).sum()}")

# Check: does EVERY item-date have ONE wholesale record?
dupes = df3.groupby([wp_date, wp_item]).size()
multi = dupes[dupes > 1]
if len(multi) > 0:
    print(f"WARNING: {len(multi)} item-date combos with multiple wholesale prices!")
    print(multi.head())

# Write clean
df3.to_csv(os.path.join(CLEAN, "wholesale_prices.csv"), index=False, encoding='utf-8-sig')
print("Cleaned: wholesale_prices.csv")

# ============================================================
# ATTACHMENT 4: Loss Rates
# ============================================================
print("\n" + "=" * 60)
print("ATTACHMENT 4: Loss Rates")
print("=" * 60)
df4_cat = pd.read_excel(os.path.join(RAW, "附件4.xlsx"), sheet_name=0)
df4_item = pd.read_excel(os.path.join(RAW, "附件4.xlsx"), sheet_name=1)
print("Category loss rates:")
print(df4_cat.to_string())
print(f"\nItem loss rate stats: min={df4_item.iloc[:,-1].min():.2f}%, "
      f"max={df4_item.iloc[:,-1].max():.2f}%, mean={df4_item.iloc[:,-1].mean():.2f}%")

# Merge category loss to product info
df1_clean = pd.read_csv(os.path.join(CLEAN, "product_info.csv"))
df1_clean = df1_clean.merge(
    df4_cat.rename(columns={df4_cat.columns[0]: cat_code_col, df4_cat.columns[2]: 'cat_loss_rate'}),
    on=cat_code_col, how='left'
)
df1_clean = df1_clean.merge(
    df4_item.rename(columns={df4_item.columns[0]: item_code_col, df4_item.columns[2]: 'item_loss_rate'}),
    on=item_code_col, how='left'
)
df1_clean.to_csv(os.path.join(CLEAN, "product_info_with_loss.csv"), index=False, encoding='utf-8-sig')
print("Cleaned: product_info_with_loss.csv (category + item loss rates merged)")

# ============================================================
# CROSS-ATTACHMENT CONSISTENCY
# ============================================================
print("\n" + "=" * 60)
print("CROSS-ATTACHMENT CONSISTENCY")
print("=" * 60)

items_1 = set(df1[item_code_col])
items_2 = set(df2[sc_item].unique())
items_3 = set(df3[wp_item].unique())
items_4 = set(df4_item.iloc[:,0])

print(f"Items in A1: {len(items_1)}")
print(f"Items in A2: {len(items_2)}")
print(f"Items in A3: {len(items_3)}")
print(f"Items in A4: {len(items_4)}")
print(f"Items in ALL 4: {len(items_1 & items_2 & items_3 & items_4)}")
print(f"Items in A1 ONLY: {len(items_1 - items_2 - items_3 - items_4)}")
print(f"Items in A2 but not A1: {len(items_2 - items_1)}")
print(f"Items in A3 but not A1: {len(items_3 - items_1)}")
print(f"Items in A4 but not A1: {len(items_4 - items_1)}")
print(f"Items with sales but NO loss rate: {len(items_2 - items_4)}")

# ============================================================
# READINESS ASSESSMENT
# ============================================================
print("\n" + "=" * 60)
print("READINESS PER QUESTION")
print("=" * 60)

# Q1 needs: product info + sales data
# Q1 is ready if: items with meaningful sales exist
items_with_enough_sales = item_date_cnt[item_date_cnt >= 30].index
print(f"Q1: {len(items_with_enough_sales)} items with >=30 days of sales data")
print("   -> READY (with warning on low-persistence items)")

# Q2 needs: category-level sales + wholesale prices + loss rates
# Rolling up to category level avoids per-item sparsity
print("Q2: Category-level aggregation feasible for all 6 categories")
print("   -> READY")

# Q3 needs: items tradeable during June 24-30, 2023
june_24_30_start = pd.Timestamp('2023-06-24')
june_24_30_end = pd.Timestamp('2023-06-30')
mask_june = (df2[sc_date] >= june_24_30_start) & (df2[sc_date] <= june_24_30_end)
june_items = df2.loc[mask_june, sc_item].unique()
print(f"Q3: {len(june_items)} items sold during June 24-30, 2023")
# Check which have wholesale prices on those dates
june_wp = df3[(df3[wp_date] >= june_24_30_start) & (df3[wp_date] <= june_24_30_end)]
print(f"   Items with wholesale in same period: {june_wp[wp_item].nunique()}")
print("   -> READY")

# Q4 depends on Q1-Q3 execution (experience-based)
print("Q4: ~ PENDING (depends on Q1-Q3 findings)")

# ============================================================
# COMPUTE PROFILE STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("PROFILE STATISTICS")
print("=" * 60)

# Class imbalance (category-level)
cat_sales_total = df2.merge(df1[[item_code_col, cat_name_col]], on=item_code_col, how='inner')
cat_sales_kg = cat_sales_total.groupby(cat_name_col)[sc_qty].sum().sort_values(ascending=False)
print("Category sales volume (kg):")
for cat, kg in cat_sales_kg.items():
    print(f"  {cat}: {kg:,.0f} kg ({kg/cat_sales_kg.sum()*100:.1f}%)")

# Imbalance ratio (max/min)
max_cat = cat_sales_kg.max()
min_cat = cat_sales_kg.min()
print(f"Category imbalance ratio (max/min): {max_cat/min_cat:.1f}")

# Item-level sales concentration (Gini)
from collections import Counter
item_sales_kg = df2.groupby(sc_item)[sc_qty].sum().sort_values()
total = item_sales_kg.sum()
cumsum = item_sales_kg.cumsum()
n_items = len(item_sales_kg)
gini = 2 * sum(i * v for i, v in enumerate(item_sales_kg.values, 1)) / (n_items * total) - (n_items + 1) / n_items
print(f"Item sales Gini coefficient: {gini:.3f}")
print(f"Top 10% items account for: {item_sales_kg.nlargest(int(n_items*0.1)).sum()/total*100:.1f}% of sales")
print(f"Top 20% items account for: {item_sales_kg.nlargest(int(n_items*0.2)).sum()/total*100:.1f}% of sales")

# Time gaps
all_dates = sorted(df2['date_only'].unique())
date_diffs = [(all_dates[i+1] - all_dates[i]).days for i in range(len(all_dates)-1)]
gaps = [d for d in date_diffs if d > 1]
print(f"Date gaps >1 day: {len(gaps)} (max gap: {max(gaps) if gaps else 0} days)")

# Seasonality check
df2['month'] = df2[sc_date].dt.month
monthly_sales = df2.groupby('month')[sc_qty].sum()
print("Monthly sales (kg):")
for m, kg in monthly_sales.items():
    print(f"  Month {m:2d}: {kg:,.0f} kg")

print("\nDone.")
