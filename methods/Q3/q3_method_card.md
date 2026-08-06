# Q3 Method Card

## Goal and success criteria

从 6/24-6/30 的实际在售 49 个单品中，选择 27-33 个组成 7 月 1 日的销售组合，确定每个单品补货量 (≥2.5 kg) + 定价 (元/kg)，最大化单日收益，同时确保每个品类至少有 k 个单品（k 根据品类在 6/24-6/30 的可售品种数等比例确定）。

## Human constraints

- Output form: 数值解 + 不确定性区间
- Priority: 品类需求覆盖（每品类至少 k 个单品）→ 收益最大化
- Unacceptable failure: 某品类完全无代表单品（品类断层）
- Experiment budget: 一次完整选品+补货+定价优化求解

## Shortlist

| ID | Role | Mathematical idea | Why eligible | Main risk | Implementation cost |
|---|---|---|---|---|---|
| **M1** | main_candidate | **双层规划：上层品类目标约束（Q2 输出）+ 下层单品联合优化（离散选品 + 连续补货量 + 定价，单品网络中心度辅助）** | 选品和定价联合求解（非分步）；品类最低 SKU 约束 + 需求目标值（由 Q2 报童解给出）作为硬/软约束；单品网络中心度（Q1 关联网络）用于打破平局和促进多样性 | 49 个候选 → 选 27-33 是一个中等规模的组合优化，混合整数非线性可能需要启发式 | 中（scipy + MILP 求解器或 GA） |
| **M2** | usable_baseline | **贪心选品 + 单独定价优化**（文献标准做法：苏茜、聂宇旋） | 按利润/销量贪心排序单品 → 逐个选入直至满足品类最小约束和总量上限 → 对选中单品独立优化定价（品类需求目标作为软约束） | 贪心选品不能保证最优组合；选品和定价分开做丢失联合优化机会 | 低（纯排序 + scipy optimize） |

## Baseline validity

- Real task completed: 是——输出 27-33 个单品补货量和定价，满足最小陈列量 2.5kg
- Comparable output/metric: 是——直接对比总收益 + 品类覆盖度
- Classification: usable_baseline ✓

## Risk-probe summary

| ID | Executability | Data/assumptions | Degeneracy | Sensitivity | Scale | Verdict |
|---|---|---|---|---|---|---|
| M1 | PASS | CONDITIONAL (花菜类仅 2 个候选单品——k≥2 时该品类无冗余，所有单品必选；若 k=1 则每个品类均可满足) | PASS (49 候选 × 27-33 选择 → 组合数 C(49,30)=~1.9e13 但双层规划避免穷举) | CONDITIONAL (花菜类是瓶颈——若其 2 个单品中任一边际利润为负，可能被迫亏本选入以维持品类覆盖) | PASS (<5s) | **PASS** |
| M2 | PASS | PASS | PASS | PASS | PASS (<1s) | **PASS** |

## Fallback trigger

- **触发条件**：花菜类 2 个候选单品中至少 1 个边际利润为负，且总收益因此低于历史基准日收益 80%
- **激活后操作**：降低花菜类 k 值从基于可售品种比例计算的值到 k=1（仍符合 framing_002 的品类最低 SKU 约束精神）

## Compact history

- 2026-08-06: Initial card. Framing_002: 品类最低 SKU 数约束(A). M1 (双层规划) proposed; M2 (贪心) as baseline. Bottleneck identified: 花菜类仅 2 candidates.
