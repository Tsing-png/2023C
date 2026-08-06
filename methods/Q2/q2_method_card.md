# Q2 Method Card

## Goal and success criteria

以品类为单位，建模日销量与成本加成定价的关系，输出 2023-07-01 ~ 2023-07-07 每日 6 品类的补货总量 (kg) 和定价策略 (元/kg)，最大化 7 天总收益。成本加成率 r 为决策变量（含不确定性区间）。

## Human constraints

- Output form: 数值解 + 不确定性区间 （带置信区间的建议）
- Priority: 收益最优 + 预测不确定性量化
- Unacceptable failure: 预测误差导致补货建议实际上亏损（而非利润最大）
- Experiment budget: 一次完整的品类级预测+优化求解+回测验证

## Shortlist

| ID | Role | Mathematical idea | Why eligible | Main risk | Implementation cost |
|---|---|---|---|---|---|
| **M1** | main_candidate | **Prophet 区间预测 + 双对数需求弹性 + 报童随机优化（加成率内生，鲁棒批发价）** | 唯一融合三点：①价格弹性从双对数模型估计并内生于优化目标；②报童模型自然处理需求不确定性（欠订/超订成本不对称）；③批发价预测用 Prophet 区间替代点预测，求解鲁棒对等 | 双对数模型的弹性估计受限于观测数据（只有发生过的价格，没有反事实）；Prophet 在 3 年数据上需调参 | 中（Prophet + statsmodels + scipy 优化） |
| **M2** | usable_baseline | **ARIMA 点预测 + 线性回归销量-定价 + 确定性非线性规划（固定加成率）** | 与 6/10 文献的方法一致，可直接对比；ARIMA 成熟稳定，非线性规划求解可靠 | 预测与优化分离；忽略需求不确定性（点预测输入确定性优化）；固定加成率放弃定价优化自由度 | 低（statsmodels ARIMA + scipy minimize） |

## Baseline validity

- Real task completed: 是——输出 6 品类 × 7 天的补货总量和定价
- Comparable output/metric: 是——直接对比优化后的总收益和实际历史收益基准
- Classification: usable_baseline ✓

## Risk-probe summary

| ID | Executability | Data/assumptions | Degeneracy | Sensitivity | Scale | Verdict |
|---|---|---|---|---|---|---|
| M1 | PASS | PASS (all 6 categories have significant own-price elasticity: -0.48 to -1.47, R² 0.51-0.85, max_vif 0.49 well below 10) | PASS (elasticity estimates all < -0.4 and statistically significant; no zero-elasticity degeneracy) | CONDITIONAL (花叶类 critical ratio=0.182 — 建议显著少订于均值需求，若需求分布右偏可能欠订) | PASS (<3s full run) | **PASS** |
| M2 | PASS | PASS | PASS | PASS | PASS (<2s) | **PASS** |

## Fallback trigger

N/A — M1 和 M2 的弹性估计均为显著，无需激活备选。

## Compact history

- 2026-08-06: Initial card. Framing_001: 加成率内生(B). M1 (Newsvendor+elasticity) proposed; M2 (ARIMA+deterministic NLP) as baseline. Both probe-verified.
