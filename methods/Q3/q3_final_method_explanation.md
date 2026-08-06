# Q3 Final Method Explanation

## Goal and Scope

从 2023 年 6 月 24-30 日可售的 49 个单品中，选择 27-33 个组成 7 月 1 日的销售组合，确定每个单品的补货量 (≥2.5 kg 最小陈列量) 和定价，最大化单日收益，同时确保每个品类至少有 k 个单品代表。

## Method Selection

**选定方法**: M1 (双层规划 — 品类目标约束 + 单品联合选品-补货-定价优化，单品网络中心度辅助)

**决策来源**: `q3_method_choice` (2026-08-06, decided_by: human)

**选择理由**: 选品、补货量和定价三个决策是耦合的 — 定价影响需求, 需求决定补货量, 补货量约束选品。双层规划将三者联合求解而非分步执行。品类最低 SKU 约束基于可售品种等比例分配 (framing_002: A)，单品网络中心度辅助打破同分数选品僵局。

## Assumptions

| 假设 | 类型 | 内容 |
|---|---|---|
| A1 | 必要 | 6 月 24-30 日的可售品种代表 7 月 1 日的供应情况 |
| A2 | 简化 | 单品级价格弹性使用 Q2 品类级弹性估计 (β_c) 作为近似 |
| A3 | 必要 | 单品订购量 ≥ 2.5 kg (最小陈列量) |
| A4 | 必要 | 最终 SKU 数在 27-33 范围内 |

## Mathematical Formulation

### Step 1: 品类最低 SKU 分配

根据各品类在 6 月 24-30 日的可售品种数等比例分配:

$$k_i = \text{clip}\left(\text{round}\left(\frac{n_i}{\sum_j n_j} \times S\right), 1, n_i\right)$$

其中 n_i 为品类 i 的可售品种数, S=30 (目标 SKU 总数中点), k_i 经上下调整确保 ∑k_i ∈ [27, 33]。

### Step 2: 单品评分

$$s_j = 0.5 \cdot \frac{m_j}{\max(m)} + 0.3 \cdot \frac{d_j}{\max(d)} + 0.2 \cdot \frac{r_j}{|C(j)|}$$

其中 m_j 为单位利润, d_j 为日均销量, r_j 为品类内销量排名。

### Step 3: 约束满足选品

Phase 1 — 每品类选 k_i 个评分最高的单品 (强制满足品类最小约束)
Phase 2 — 按评分贪心填充剩余名额至目标总数

### Step 4: 单品定价优化

对每个选中单品 j:

$$r_j^* = \arg\max_{r \in [0.05, 1.5]} \quad (P_j(r) - C_{\text{eff},j}) \cdot Q_j(r)$$

其中:
- P_j(r) = C_eff,j · (1 + r)
- Q_j(r) = max(d_base,j · (P_j / P_hist,j)^β_c(j), 2.5)

## Key Findings

1. 选中 **30 个单品**, 覆盖全部 6 个品类, 满足最小陈列量
2. 单日预期利润: **1,225 元** (vs M2 固定加成率 20% 下 253 元)
3. 品类最低约束: 花叶类 10, 辣椒类 6, 食用菌 5, 水生根茎类 4, 茄类 3, 花菜类 2
4. 花菜类仅 2 个候选单品 — 是品类覆盖的瓶颈, 两个单品被强制选中
5. **30/30 单品加成率触碰 150% 上界** — 模型局限性: 单品级弹性使用品类级近似, 无法区分单品间的定价差异

## Baseline (M2)

贪心按利润/天排序选品 (含品类最低 k=2) + 固定加成率 20% 独立定价。同 30 个单品, 利润 253 元。

## Robustness

- SKU 目标 27→33: 利润线性增长 1,136→1,336 元, 边界饱和持续
- 批发价 ±10%: 利润稳定 1,131-1,257 元

## Limitations

1. **加成率边界饱和** (最严重): 单品级弹性使用品类级估计, 无法为不同单品给出差异化的最优加成率
2. 花菜类候选稀缺 (仅 2 个), 无冗余
3. 单日优化 (仅 7/1), 未考虑多日联合调度
4. Q1 单品关联网络仅用于多样性评分, 未作为正式的交叉弹性约束

## Source Artifacts

- Selection: `results/Q3/experiments/round1/tables/q3_selected_items.csv`
- Metrics: `results/Q3/experiments/round1/metrics/q3_metrics.json`
- Robustness: `robustness/Q3/q3_robustness_summary.json`
- Code: `code/Q3/q3_main.py`
