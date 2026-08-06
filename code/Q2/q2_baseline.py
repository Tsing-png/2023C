# Q2 Baseline: ARIMA + Linear Regression + Deterministic NLP (fixed markup r=0.20)
# Seed: 2026

import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime, timedelta
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")
OUT = os.path.join(BASE, "results", "Q2", "experiments", "round1")
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

SEED = 2026
np.random.seed(SEED)

# ============================================================
# DATA LOADING
# ============================================================
df1 = pd.read_csv(os.path.join(CLEAN, "product_info_with_loss.csv"))
df2 = pd.read_csv(os.path.join(CLEAN, "sales_normal_only.csv"))
df3 = pd.read_csv(os.path.join(CLEAN, "wholesale_prices.csv"))

item_col, item_name_col, cat_code_col, cat_name_col = df1.columns[:4]
sc_date, sc_time, sc_item, sc_qty, sc_price, sc_discount = df2.columns[:6]
wp_date, wp_item, wp_price = df3.columns[:3]

cat_map = dict(zip(df1[item_col], df1[cat_name_col]))
cat_loss = df1.groupby(cat_name_col)['cat_loss_rate'].first().to_dict()
all_categories = sorted(cat_loss.keys())

print("=" * 60)
print("Q2 BASELINE (M2): ARIMA + Linear Regression + Deterministic NLP")
print("=" * 60)

# ============================================================
# Category-Daily Aggregation
# ============================================================
df2['date'] = pd.to_datetime(df2[sc_date]).dt.date
df3['date'] = pd.to_datetime(df3[wp_date]).dt.date
df2['category'] = df2[sc_item].map(cat_map)

cat_daily = df2.groupby(['date', 'category']).agg(
    total_qty=(sc_qty, 'sum'),
    avg_sales_price=(sc_price, 'mean')
).reset_index()

df3['category'] = df3[wp_item].map(cat_map)
cat_wp = df3.groupby(['date', 'category'])[wp_price].mean().reset_index()
cat_wp.columns = ['date', 'category', 'avg_wholesale']

cat_daily = cat_daily.merge(cat_wp, on=['date', 'category'], how='left')
cat_daily['avg_wholesale'] = cat_daily.groupby('category')['avg_wholesale'].ffill()
cat_daily = cat_daily.dropna(subset=['avg_wholesale'])
cat_daily['date_dt'] = pd.to_datetime(cat_daily['date'])

# ============================================================
# STEP 1: ARIMA forecast (sales & wholesale, 7 days ahead)
# ============================================================
from statsmodels.tsa.arima.model import ARIMA

FORECAST_DAYS = 7
forecast_start = pd.Timestamp('2023-07-01')
forecast_dates = [forecast_start + timedelta(days=i) for i in range(FORECAST_DAYS)]

daily_pivot = cat_daily.pivot(index='date_dt', columns='category', values='total_qty').fillna(0)
wp_pivot_daily = cat_daily.pivot(index='date_dt', columns='category', values='avg_wholesale').ffill()

arima_forecast = {}
for cat in all_categories:
    # Sales forecast
    hist_qty = daily_pivot[cat].values[-180:]
    try:
        model = ARIMA(hist_qty, order=(2, 1, 2))
        fitted = model.fit()
        fcst = fitted.forecast(steps=FORECAST_DAYS)
    except:
        fcst = np.full(FORECAST_DAYS, hist_qty[-30:].mean())

    # Wholesale forecast
    hist_wp = wp_pivot_daily[cat].dropna().values[-180:]
    try:
        wp_model = ARIMA(hist_wp, order=(1, 1, 1))
        wp_fitted = wp_model.fit()
        wp_fcst = wp_fitted.forecast(steps=FORECAST_DAYS)
    except:
        wp_fcst = np.full(FORECAST_DAYS, hist_wp[-30:].mean())

    arima_forecast[cat] = {'qty': np.maximum(fcst, 0.1), 'wholesale': np.maximum(wp_fcst, 0.5), 'qty_std': np.std(hist_qty)}
    print(f"  {cat}: ARIMA qty[0]={fcst[0]:.1f}, wp[0]={wp_fcst[0]:.2f}")

# ============================================================
# STEP 2: Linear Regression (Qty ~ Price) + Fixed Markup r=0.20
# ============================================================
import statsmodels.api as sm

FIXED_MARKUP = 0.20
linear_models = {}
for cat in all_categories:
    cat_data = cat_daily[cat_daily['category'] == cat].copy()
    if len(cat_data) < 30:
        linear_models[cat] = {'intercept': 100, 'slope': -5}
        continue

    X = sm.add_constant(cat_data['avg_sales_price'].values)
    y = cat_data['total_qty'].values
    try:
        model = sm.OLS(y, X).fit()
        linear_models[cat] = {'intercept': float(model.params[0]), 'slope': float(model.params[1]), 'r2': float(model.rsquared)}
        print(f"  {cat}: Q = {model.params[0]:.1f} - {abs(model.params[1]):.2f}*P, R2={model.rsquared:.3f}")
    except:
        linear_models[cat] = {'intercept': 100, 'slope': -5, 'r2': 0.3}

# ============================================================
# STEP 3: Deterministic NLP Optimization
# ============================================================
results = []
loss_r_map = {cat: cat_loss[cat] / 100.0 for cat in all_categories}

for cat in all_categories:
    fcst_qty = arima_forecast[cat]['qty']
    fcst_wp = arima_forecast[cat]['wholesale']
    lm = linear_models[cat]
    loss_r = loss_r_map[cat]

    for day_idx in range(FORECAST_DAYS):
        wp_day = fcst_wp[day_idx]
        eff_cost = wp_day / (1 - loss_r)
        sales_price = eff_cost * (1 + FIXED_MARKUP)

        # Demand at this price
        demand = lm['intercept'] + lm['slope'] * sales_price
        demand = max(demand, 0.1)

        # Optimal replenishment = demand (deterministic)
        Q = demand

        # Profit
        revenue = sales_price * Q
        cost_total = eff_cost * Q
        profit = revenue - cost_total

        results.append({
            'date': forecast_dates[day_idx].strftime('%Y-%m-%d'),
            'category': cat,
            'forecast_demand_kg': round(float(demand), 2),
            'wholesale_price': round(float(wp_day), 2),
            'eff_cost': round(float(eff_cost), 2),
            'markup_rate': FIXED_MARKUP,
            'sales_price': round(float(sales_price), 2),
            'order_quantity_kg': round(float(Q), 2),
            'expected_profit': round(float(profit), 2)
        })

df_results = pd.DataFrame(results)

# ============================================================
# OUTPUT
# ============================================================
df_results.to_csv(os.path.join(OUT, 'tables', 'q2_baseline_decisions.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: q2_baseline_decisions.csv")

total_profit = df_results['expected_profit'].sum()
profit_by_cat = df_results.groupby('category')['expected_profit'].sum().to_dict()

# Metrics
metrics = {
    "total_profit_7d": round(float(total_profit), 2),
    "fixed_markup_rate": FIXED_MARKUP,
    "profit_by_category": {k: round(float(v), 2) for k, v in profit_by_cat.items()},
    "linear_model_R2": {cat: round(linear_models[cat].get('r2', 0), 3) for cat in all_categories}
}

with open(os.path.join(OUT, 'metrics', 'q2_baseline_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Figure
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, cat in enumerate(all_categories):
    ax = axes[i // 3, i % 3]
    cd = df_results[df_results['category'] == cat]
    ax.bar(range(1, 8), cd['order_quantity_kg'].values, alpha=0.6, color='orange', label='Replenishment')
    ax2 = ax.twinx()
    ax2.plot(range(1, 8), cd['sales_price'].values, 'bo-', label='Price')
    ax.set_title(f'{cat} (Baseline: r=0.20)', fontsize=11)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([f'7/{d}' for d in range(1, 8)])
fig.suptitle('Q2 Baseline: ARIMA + Deterministic NLP (Fixed Markup r=0.20)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q2_baseline_plan.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q2_baseline_plan.png")

print("\n" + "=" * 60)
print(f"Q2 BASELINE (M2) COMPLETE")
print(f"  Total 7-day profit: {total_profit:,.0f} yuan (fixed r=0.20)")
print(f"  Profit by category: {metrics['profit_by_category']}")
print("=" * 60)
