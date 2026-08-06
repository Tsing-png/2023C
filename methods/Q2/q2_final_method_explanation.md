# Q2 Final Method Explanation

## Goal and Scope

以品类为单位，建模日销量与成本加成定价的关系。输出 2023-07-01 至 2023-07-07 每日 6 品类的补货总量 (kg) 和定价策略 (元/kg)，最大化 7 天总收益。

## Method Selection

**选定方法**: M1 (双对数需求弹性 + 时间序列预测 + 报童随机优化 + 内生加成率)

**决策来源**: `q2_method_choice` (2026-08-06, decided_by: human)

**选择理由**: 该方法是唯一同时解决三个核心问题的方案: (1) 价格弹性从双对数模型估计并内生于优化目标, 而非固定加成率; (2) 报童模型自然处理需求不确定性, 区分超订损失(蔬菜残值为零)和欠订损失(机会成本); (3) 批发价预测使用区间估计替代点估计, 在鲁棒性分析中验证。

## Assumptions

| 假设 | 类型 | 内容 |
|---|---|---|
| A1 | 必要 | 历史销量-价格关系在未来一周仍然成立 (弹性时不变) |
| A2 | 必要 | 蔬菜当日未售出隔日残值为零 (超订成本 = 批发价 / (1-损耗率)) |
| A3 | 简化 | 日需求服从正态分布 N(μ, σ²)，其中 μ 由预测模型给出, σ 由历史残差估计 |
| A4 | 简化 | 品类间需求独立 (Spearman 残差相关 ≤0.36 — 弱至中等，独立近似合理) |
| A5 | 简化 | 批发价格在决策时刻未知，使用基于历史数据的 Prophet/ETS 预测 |

## Mathematical Formulation

### Step 1: 双对数需求模型 (价格弹性估计)

对每个品类 c，使用历史日数据进行 OLS 回归:

$$\ln Q_{c,t} = \alpha_c + \beta_{c} \ln P_{c,t} + \sum_{k \neq c} \gamma_{k} \ln P_{k,t} + \delta \ln(\text{Spend}_t) + \text{month}_t + \varepsilon_t$$

其中 β_c 为品类 c 的自价格弹性, 估计范围为 [-0.485, -1.471], R² 为 0.51-0.85。

### Step 2: 时间序列预测

对每个品类 c 的未来 7 天需求基础值和批发价进行指数平滑预测 (ETS), 得到基线需求 μ_0 和需求标准差 σ:

$$\mu_{0,c,d} = \text{ETS}(y_{c, 1:T}), \quad \sigma_{c,d} = \text{std}(y_{c,t} - \hat{y}_{c,t})$$

$$\text{wp}_{c,d} = \text{ETS}(w_{c, 1:T})$$

### Step 3: 弹性修正需求

加成率 r 内生改变需求:

$$\mu_{c,d}(r) = \mu_{0,c,d} \cdot (1 + r)^{\beta_c}$$

### Step 4: 报童模型优化

每个品类 c 每天 d，选择最优加成率 r*:

$$r^* = \arg\max_r \quad P(r) \cdot \mathbb{E}[\min(D, Q^*)] - C_{\text{eff}} \cdot Q^*$$

其中:
- 售价: P(r) = C_eff · (1 + r)
- 有效成本: C_eff = wp / (1 - loss_rate)
- 关键比: CR = (P - C_eff) / P = r / (1 + r)
- 最优补货量: Q* = μ(r) + σ · Φ⁻¹(CR)
- 期望销量: E[min(D, Q*)] = μ(r) - σ · L(z), 其中 z = Φ⁻¹(CR), L(z) = φ(z) - z·(1-Φ(z))

### Step 5: 鲁棒性检查

对批发价施加 ±10% 扰动, 重新计算最优利润, 报告利润区间。

## Key Findings

1. 7 天总预期利润: **2,587 元** (vs M2 固定加成率 20% 下的 3,817 元 — M2 忽略不确定性，利润虚高)
2. 最优内生加成率: 从 29% (食用菌) 到 150% (3 品类触顶上界)
3. 价格弹性最低的品类 (花叶类 β=-0.485) 同时具有最低的关键比 CR=0.18 — 保守补货是生鲜高损耗下的理性行为
4. 需求方差 σ 是利润的最敏感参数: σ 减半 → 利润 +135%, σ 加倍 → 亏损

## Baseline (M2)

ARIMA 点预测 + 线性回归 Q=β₀+β₁P (R²=0.01-0.12) + 固定加成率 20% + 确定性 NLP。M2 利润更高 (3,817 元) 但完全忽略需求不确定性。

## Limitations

1. 弹性估计从观测数据获得 — 无外生价格变异 (无 A/B 实验)，估计可能衰减
2. 需求分布假设为正态 — 实际可能有偏分布 (偶尔的销量暴增)
3. 无竞争效应 — 定价独立于其他超市
4. 30% 的品类加成率触碰 150% 上界 — 模型局限性
5. 弹性估计假设参数时不变 (3 年窗口)，未做滚动窗口验证

## Source Artifacts

- Decisions: `results/Q2/experiments/round1/tables/q2_optimal_decisions.csv`
- Metrics: `results/Q2/experiments/round1/metrics/q2_metrics.json`
- Robustness: `robustness/Q2/q2_robustness_summary.json`
- Code: `code/Q2/q2_main.py`
