# Q2 Code Plan

## Target and Round
- **Language**: Python 3, **Round**: round1, **Seed**: 2026
- **Approved decision**: `q2_method_choice` → M1 (main), M2 (baseline)
- **Framing**: 加成率内生 (framing_001:B), 输出含不确定性区间 (framing_004:C)

## Input Fields
| Source | Key Fields |
|---|---|
| `sales_normal_only.csv` | 销售日期, 单品编码, 销量(kg), 销售单价(元/kg), 是否折扣销售 |
| `wholesale_prices.csv` | 日期, 单品编码, 批发价格(元/kg) |
| `product_info_with_loss.csv` | 品类名称, cat_loss_rate |

## M1 — Main: Newsvendor + Endogenous Markup + Robust Wholesale

### Step 1: Category-daily feature engineering
- Aggregate to category-daily: total_qty, avg_sales_price, avg_wholesale_price, discount_ratio
- Merge loss rates → effective_cost = wholesale / (1 - loss_rate)

### Step 2: Double-log demand model (price elasticity)
- Per category: ln(Q) = α + β·ln(P_own) + Σ γ_k·ln(P_other_k) + δ·ln(total_spend) + month dummies
- Extract own-price elasticity β from regression
- Build elasticity-modified demand forecast: Q = exp(α + β·ln(1+r) + ...)

### Step 3: Prophet demand & wholesale forecast (7 days ahead)
- Fit Prophet on category daily sales (last 365 days)
- Fit Prophet on category avg wholesale (last 365 days)
- Generate 7-day forecasts with prediction intervals (yhat, yhat_lower, yhat_upper)

### Step 4: Newsvendor optimization per category per day
- Decision variable: markup rate r (and thus price P = cost × (1+r)) and order quantity Q
- Demand D ~ Normal(μ, σ) where μ = Prophet forecast × elasticity_modifier, σ = Prophet uncertainty
- Profit = P × min(Q, D) - cost × Q
- Critical ratio: (P - cost) / P  →  Q* = μ + σ × Φ⁻¹(CR)
- For each category-day, solve: maximize expected profit over (r, Q)

### Step 5: Robust wholesale handling
- 10% perturbation on wholesale → recompute optimal → report profit range

## M2 — Baseline: ARIMA + Linear Regression + Deterministic NLP
- ARIMA forecast sales & wholesale (7 days)
- Linear regression: Q = a + b × price (per category)
- Fixed markup r = 0.20
- Deterministic NLP: max total_profit subject to Q = f(price)

## Comparable Metrics
| Metric | M1 | M2 |
|---|---|---|
| Expected 7-day profit | ✓ | ✓ |
| Optimal markup rates (range) | Endogenous | Fixed 0.20 |
| Wholesale uncertainty handling | Prediction interval | Point only |
| Demand distribution | Normal(μ,σ) | Point estimate |

## Risk Probe Conditions
- Elasticity significance: all β < -0.4 and p < 0.05 → OK
- CR合理性: 花叶类CR=0.182 → 保守补货 is correct behavior
- Prophet fit: R² per category check
