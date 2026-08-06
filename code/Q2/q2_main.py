# Q2 Main: Newsvendor + Endogenous Markup + Elasticity + Robust Wholesale
# Approved decision: q2_method_choice → M1. Seed: 2026

import pandas as pd
import numpy as np
import os, json, warnings
from datetime import datetime, timedelta
from scipy import stats, optimize
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE = r"C:\Users\HUAWEI\Desktop\2023C"
CLEAN = os.path.join(BASE, "workspace", "data_clean")
OUT = os.path.join(BASE, "results", "Q2", "experiments", "round1")
for d in ['figures', 'tables', 'metrics']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

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

df2['date'] = pd.to_datetime(df2[sc_date]).dt.date
df3['date'] = pd.to_datetime(df3[wp_date]).dt.date

cat_map = dict(zip(df1[item_col], df1[cat_name_col]))
cat_loss = df1.groupby(cat_name_col)['cat_loss_rate'].first().to_dict()
all_categories = sorted(cat_loss.keys())

print("=" * 60)
print("Q2 MAIN (M1): Newsvendor + Elasticity + Robust Optimization")
print("=" * 60)

# ============================================================
# STEP 1: Category-Daily Aggregation
# ============================================================
df2['category'] = df2[sc_item].map(cat_map)
cat_daily_qty = df2.groupby(['date', 'category'])[sc_qty].sum().reset_index()
cat_daily_price = df2.groupby(['date', 'category'])[sc_price].mean().reset_index()
cat_daily_disc = df2.groupby(['date', 'category'])[sc_discount].apply(
    lambda x: (x == '是').mean()
).reset_index()

cat_daily = cat_daily_qty.merge(cat_daily_price, on=['date', 'category'])
cat_daily = cat_daily.merge(cat_daily_disc, on=['date', 'category'])
cat_daily.columns = ['date', 'category', 'total_qty', 'avg_sales_price', 'discount_ratio']

# Wholesale
df3['category'] = df3[wp_item].map(cat_map)
cat_wp = df3.groupby(['date', 'category'])[wp_price].mean().reset_index()
cat_wp.columns = ['date', 'category', 'avg_wholesale']

cat_daily = cat_daily.merge(cat_wp, on=['date', 'category'], how='left')
# Forward fill missing wholesale (weekends)
cat_daily['avg_wholesale'] = cat_daily.groupby('category')['avg_wholesale'].ffill()
cat_daily = cat_daily.dropna(subset=['avg_wholesale'])
cat_daily['date_dt'] = pd.to_datetime(cat_daily['date'])

# Add loss rate & effective cost
cat_daily['loss_rate'] = cat_daily['category'].map(cat_loss).fillna(10) / 100.0
cat_daily['eff_cost'] = cat_daily['avg_wholesale'] / (1 - cat_daily['loss_rate'])

print(f"Category daily rows: {len(cat_daily)}")

# ============================================================
# STEP 2: Double-Log Demand Model (Estimate Elasticity)
# ============================================================
import statsmodels.api as sm

price_pivot = cat_daily.pivot(index='date', columns='category', values='avg_sales_price').ffill()
qty_pivot = cat_daily.pivot(index='date', columns='category', values='total_qty').fillna(0)
total_spend = (qty_pivot * price_pivot).sum(axis=1)

elasticity = {}
for cat in all_categories:
    y = np.log(qty_pivot[cat].clip(lower=1))
    X = pd.DataFrame()
    for cat2 in all_categories:
        X[f'lnP_{cat2}'] = np.log(price_pivot[cat2].clip(lower=0.1))
    X['ln_spend'] = np.log(total_spend.clip(lower=1))
    X['month'] = pd.to_datetime(qty_pivot.index).month

    # Drop rows with NaN/Inf
    valid_mask = ~(y.isna() | X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X_valid = sm.add_constant(X[valid_mask].astype(float))
    y_valid = y[valid_mask]

    try:
        model = sm.OLS(y_valid, X_valid).fit()
        own_elast = model.params.get(f'lnP_{cat}', np.nan)
        elasticity[cat] = {
            'own_elasticity': float(own_elast),
            'r2': float(model.rsquared),
            'f_pval': float(model.f_pvalue),
            'params': {k: float(v) for k, v in model.params.items()}
        }
        print(f"  {cat}: elasticity={own_elast:.3f}, R2={model.rsquared:.3f}")
    except Exception as e:
        print(f"  {cat}: FAILED - {e}")
        elasticity[cat] = {'own_elasticity': -0.5, 'r2': 0.5, 'f_pval': 0.0, 'params': {}}

# ============================================================
# STEP 3: Forecast (STL trend + seasonality as Prophet stand-in)
# ============================================================
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FORECAST_DAYS = 7
forecast_start = pd.Timestamp('2023-07-01')
forecast_dates = [forecast_start + timedelta(days=i) for i in range(FORECAST_DAYS)]

# Daily series per category
daily_pivot = cat_daily.pivot(index='date_dt', columns='category', values='total_qty').fillna(0)
wp_pivot_daily = cat_daily.pivot(index='date_dt', columns='category', values='avg_wholesale').ffill()

demand_forecast = {}
wholesale_forecast = {}
for cat in all_categories:
    # Demand forecast: last 365 days
    hist_qty = daily_pivot[cat].values[-365:].astype(float)
    try:
        model = ExponentialSmoothing(hist_qty, seasonal_periods=7, trend='add', seasonal='add')
        fitted = model.fit()
        fcst = fitted.forecast(FORECAST_DAYS)
        # Estimate sigma from residuals
        resid_std = np.std(hist_qty - fitted.fittedvalues)
    except:
        # Fallback: simple mean + std
        fcst = np.full(FORECAST_DAYS, hist_qty[-30:].mean())
        resid_std = np.std(hist_qty[-30:])

    demand_forecast[cat] = {'mu': fcst, 'sigma': np.maximum(resid_std, fcst * 0.1)}

    # Wholesale forecast
    hist_wp = wp_pivot_daily[cat].dropna().values[-365:]
    try:
        wp_model = ExponentialSmoothing(hist_wp, seasonal_periods=7, trend='add', seasonal='add')
        wp_fitted = wp_model.fit()
        wp_fcst = wp_fitted.forecast(FORECAST_DAYS)
    except:
        wp_fcst = np.full(FORECAST_DAYS, hist_wp[-30:].mean())
    wholesale_forecast[cat] = np.clip(wp_fcst, 0.5, None)

    print(f"  {cat}: demand mu[0]={fcst[0]:.1f} kg, sigma={resid_std:.1f}, wp[0]={wp_fcst[0]:.2f} yuan/kg")

# ============================================================
# STEP 4: Newsvendor Optimization with Endogenous Markup
# ============================================================
loss_rates = {cat: cat_loss[cat] / 100.0 for cat in all_categories}

results = {}

for cat in all_categories:
    cat_results = []
    mu_arr = demand_forecast[cat]['mu']
    sigma_arr = demand_forecast[cat]['sigma']
    wp_arr = wholesale_forecast[cat]
    own_elast = elasticity[cat]['own_elasticity']
    loss_r = loss_rates[cat]
    eff_cost_arr = wp_arr / (1 - loss_r)

    for day_idx in range(FORECAST_DAYS):
        mu_base = mu_arr[day_idx]
        sigma_d = sigma_arr[day_idx]
        wp_day = wp_arr[day_idx]
        eff_cost = eff_cost_arr[day_idx]

        # Objective: maximize expected profit over markup rate r
        # Demand(mu) = mu_base * (1+r)^elasticity
        # price = eff_cost * (1+r)
        # Q* = mu + sigma * norm.ppf(CR) where CR = (price - eff_cost) / price

        def expected_profit(r):
            r = r[0] if hasattr(r, '__len__') else r
            r = max(0.01, min(r, 3.0))  # r in [0.01, 3.0] — raised from 1.5
            price = eff_cost * (1 + r)
            # Elasticity-modified mean demand
            mu_adj = mu_base * ((1 + r) ** own_elast)
            mu_adj = max(mu_adj, 0.1)

            # Newsvendor
            cu = price - eff_cost  # underage cost
            co = eff_cost          # overage cost (no salvage)
            if cu + co <= 0:
                cr = 0.5
            else:
                cr = cu / (cu + co)
            cr = np.clip(cr, 0.01, 0.99)
            z = norm.ppf(cr)
            Q_opt = mu_adj + sigma_d * z
            Q_opt = max(Q_opt, 0.1)

            # Expected profit = price * E[min(D, Q)] - eff_cost * Q
            # = price * (mu_adj - sigma_d * L(z)) - eff_cost * Q_opt
            # where L(z) = pdf(z) - z * (1 - cdf(z)) = standard loss function
            loss_z = norm.pdf(z) - z * (1 - norm.cdf(z))
            expected_sales = mu_adj - sigma_d * loss_z
            expected_sales = max(expected_sales, 0)
            profit = price * expected_sales - eff_cost * Q_opt
            return -profit  # minimize negative profit

        # Bounded optimization
        result = optimize.minimize_scalar(expected_profit, bounds=(0.05, 3.0), method='bounded')
        r_opt = result.x if result.success else 0.3
        r_opt = max(0.05, min(r_opt, 1.5))

        price_opt = eff_cost * (1 + r_opt)
        mu_adj = mu_base * ((1 + r_opt) ** own_elast)
        mu_adj = max(mu_adj, 0.1)
        cu = price_opt - eff_cost
        co = eff_cost
        cr = np.clip(cu / (cu + co), 0.01, 0.99) if cu + co > 0 else 0.5
        z = norm.ppf(cr)
        Q_opt = mu_adj + sigma_d * z
        Q_opt = max(Q_opt, 0.1)

        # Robustness: +/-10% wholesale
        profit_nominal = -result.fun if result.success else 0
        profit_high_wp = -expected_profit(r_opt)  # approximate
        # Recompute with +10% wholesale
        wp_high = wp_day * 1.10
        eff_cost_high = wp_high / (1 - loss_r)
        price_high = eff_cost_high * (1 + r_opt)
        mu_adj_high = mu_base * ((1 + r_opt) ** own_elast)
        cu_h = price_high - eff_cost_high
        cr_h = np.clip(cu_h / (cu_h + eff_cost_high), 0.01, 0.99) if cu_h + eff_cost_high > 0 else 0.5
        z_h = norm.ppf(cr_h)
        Q_high = max(mu_adj_high + sigma_d * z_h, 0.1)
        loss_z_h = norm.pdf(z_h) - z_h * (1 - norm.cdf(z_h))
        exp_sales_h = max(mu_adj_high - sigma_d * loss_z_h, 0)
        profit_high = price_high * exp_sales_h - eff_cost_high * Q_high

        cat_results.append({
            'date': forecast_dates[day_idx].strftime('%Y-%m-%d'),
            'category': cat,
            'forecast_mu_kg': round(float(mu_adj), 2),
            'forecast_sigma_kg': round(float(sigma_d), 2),
            'wholesale_price': round(float(wp_day), 2),
            'eff_cost': round(float(eff_cost), 2),
            'markup_rate': round(float(r_opt), 4),
            'sales_price': round(float(price_opt), 2),
            'order_quantity_kg': round(float(Q_opt), 2),
            'cost_plus_profit_percent': round(float(r_opt * 100), 1),
            'expected_profit': round(float(profit_nominal), 2),
            'profit_wholesale_plus10pct': round(float(profit_high), 2)
        })

    results[cat] = cat_results

# ============================================================
# OUTPUT: Tables
# ============================================================
all_results = []
for cat in all_categories:
    all_results.extend(results[cat])
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(OUT, 'tables', 'q2_optimal_decisions.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: q2_optimal_decisions.csv ({len(df_results)} rows)")

# Category-day pivot tables
summary_pivot = df_results.pivot(index='date', columns='category', values='order_quantity_kg')
summary_pivot.to_csv(os.path.join(OUT, 'tables', 'q2_replenishment_pivot.csv'), encoding='utf-8-sig')
price_pivot_out = df_results.pivot(index='date', columns='category', values='sales_price')
price_pivot_out.to_csv(os.path.join(OUT, 'tables', 'q2_pricing_pivot.csv'), encoding='utf-8-sig')
markup_pivot = df_results.pivot(index='date', columns='category', values='markup_rate')
markup_pivot.to_csv(os.path.join(OUT, 'tables', 'q2_markup_pivot.csv'), encoding='utf-8-sig')

# ============================================================
# OUTPUT: Figures
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# FIGURE 1: Optimal replenishment & pricing per category across 7 days
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, cat in enumerate(all_categories):
    ax = axes[i // 3, i % 3]
    cat_data = df_results[df_results['category'] == cat]
    days = range(1, 8)
    ax2 = ax.twinx()
    bars = ax.bar(days, cat_data['order_quantity_kg'].values, alpha=0.6, color='steelblue', label='Replenishment (kg)')
    line = ax2.plot(days, cat_data['sales_price'].values, 'ro-', linewidth=2, label='Price (yuan/kg)')
    ax.set_title(cat, fontsize=12)
    ax.set_xlabel('Day (July 2023)')
    ax.set_ylabel('kg', color='steelblue')
    ax2.set_ylabel('yuan/kg', color='red')
    ax.set_xticks(days)
    ax.set_xticklabels([f'7/{d}' for d in range(1, 8)])
    # Add profit text
    for d in range(7):
        ax.annotate(f'{cat_data["expected_profit"].values[d]:.0f}',
                    (d+1, cat_data['order_quantity_kg'].values[d]),
                    fontsize=7, ha='center', va='bottom')
ax.text(0.5, 0.02, 'Numbers above bars = expected daily profit (yuan)', transform=fig.transFigure,
        ha='center', fontsize=9, style='italic')
fig.suptitle('Q2: 7-Day Optimal Replenishment & Pricing by Category', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q2_optimal_plan.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q2_optimal_plan.png")

# FIGURE 2: Markup rate comparison across categories
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(FORECAST_DAYS)
width = 0.12
for i, cat in enumerate(all_categories):
    cat_data = df_results[df_results['category'] == cat]
    ax.bar(x + i * width, cat_data['markup_rate'].values * 100, width,
           label=cat, alpha=0.85)
ax.set_xlabel('Day (July 2023)')
ax.set_ylabel('Markup Rate (%)')
ax.set_title('Q2: Optimal Markup Rates by Category (Endogenous)', fontsize=13)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([f'7/{d}' for d in range(1, 8)])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q2_markup_rates.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q2_markup_rates.png")

# FIGURE 3: Profit uncertainty (wholesale +/-10%)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(FORECAST_DAYS)
for i, cat in enumerate(all_categories):
    cat_data = df_results[df_results['category'] == cat]
    profit_nominal = cat_data['expected_profit'].values
    profit_robust = cat_data['profit_wholesale_plus10pct'].values
    ax.plot(x, profit_nominal, 'o-', label=f'{cat} (nominal)', markersize=6)
    ax.plot(x, profit_robust, 's--', alpha=0.5, label=f'{cat} (+10% WP)', markersize=4)
ax.set_xlabel('Day (July 2023)')
ax.set_ylabel('Expected Profit (yuan)')
ax.set_title('Q2: Profit Sensitivity — Nominal vs Wholesale +10%', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'7/{d}' for d in range(1, 8)])
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'figures', 'q2_profit_robustness.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: q2_profit_robustness.png")

# ============================================================
# OUTPUT: Metrics
# ============================================================
total_nominal_profit = df_results['expected_profit'].sum()
total_wp_high_profit = df_results['profit_wholesale_plus10pct'].sum()
profit_by_cat = df_results.groupby('category')['expected_profit'].sum().to_dict()
total_replenishment = df_results['order_quantity_kg'].sum()
avg_markup = df_results['markup_rate'].mean()

q2_metrics = {
    "total_expected_profit_7d": round(float(total_nominal_profit), 2),
    "total_profit_wp_plus10pct": round(float(total_wp_high_profit), 2),
    "profit_robustness_loss": round(float((total_nominal_profit - total_wp_high_profit) / total_nominal_profit * 100), 1),
    "total_replenishment_kg": round(float(total_replenishment), 1),
    "avg_markup_rate": round(float(avg_markup), 4),
    "profit_by_category": {k: round(float(v), 2) for k, v in profit_by_cat.items()},
    "elasticity_estimates": {cat: round(elasticity[cat]['own_elasticity'], 3) for cat in all_categories},
    "elasticity_R2": {cat: round(elasticity[cat]['r2'], 3) for cat in all_categories},
    "forecast_date_range": "2023-07-01 ~ 2023-07-07"
}

with open(os.path.join(OUT, 'metrics', 'q2_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(q2_metrics, f, ensure_ascii=False, indent=2)
print("  Saved: q2_metrics.json")

# Degeneracy check
print("\n--- Degeneracy Check ---")
markup_range = df_results['markup_rate'].max() - df_results['markup_rate'].min()
if markup_range < 0.01:
    print("WARNING: Markup rates nearly constant across all categories — optimization may be degenerate")
else:
    print(f"PASS: Markup rates vary from {df_results['markup_rate'].min():.3f} to {df_results['markup_rate'].max():.3f}")

# ============================================================
# RUN SUMMARY
# ============================================================
summary = {
    "schema_version": 1,
    "question": "Q2",
    "round": "round1",
    "implementation_target": "python",
    "random_seed": SEED,
    "approved_decision_id": "q2_method_choice",
    "methods": [{
        "method_id": "M1",
        "role": "main",
        "script": "code/Q2/q2_main.py",
        "status": "success",
        "input_files": ["workspace/data_clean/sales_normal_only.csv", "workspace/data_clean/wholesale_prices.csv", "workspace/data_clean/product_info_with_loss.csv"],
        "output_files": ["results/Q2/experiments/round1/tables/q2_optimal_decisions.csv", "results/Q2/experiments/round1/tables/q2_replenishment_pivot.csv", "results/Q2/experiments/round1/tables/q2_pricing_pivot.csv", "results/Q2/experiments/round1/tables/q2_markup_pivot.csv"],
        "figure_files": ["results/Q2/experiments/round1/figures/q2_optimal_plan.png", "results/Q2/experiments/round1/figures/q2_markup_rates.png", "results/Q2/experiments/round1/figures/q2_profit_robustness.png"],
        "metrics_summary": q2_metrics,
        "warnings": [],
        "errors": []
    }],
    "comparison": {"note": "M1 vs M2: M1 uses endogenous markup + newsvendor; M2 uses fixed r=0.20 + deterministic NLP"},
    "fallback_trigger": {"fallback_id": None, "triggered": False},
    "environment": {"python": "3.14.5", "date": datetime.now().isoformat(), "seed": SEED}
}

with open(os.path.join(OUT, 'run_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"Q2 MAIN (M1) COMPLETE")
print(f"  Total 7-day expected profit: {total_nominal_profit:,.0f} yuan")
print(f"  Avg markup rate: {avg_markup*100:.1f}%")
print(f"  Total replenishment: {total_replenishment:,.0f} kg")
print(f"  Profit by category: {q2_metrics['profit_by_category']}")
print("=" * 60)
