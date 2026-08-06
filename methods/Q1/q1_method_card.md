# Q1 Method Card

## Goal and success criteria

刻画 6 个品类及 246 个有销售记录的单品的销售量分布规律（集中/分散/季节/趋势）和相互关系（相关性、因果方向、聚类结构），为 Q2/Q3 建模决策提供量化证据。

## Human constraints

- Output form: 统计量 + 可视化 + 关联网络图 + 因果方向
- Priority: 可解释性（品类管理者可理解）
- Unacceptable failure: 把共同时间趋势误判为关联关系（伪相关）
- Experiment budget: 探索性分析，无需重复实验

## Shortlist

| ID | Role | Mathematical idea | Why eligible | Main risk | Implementation cost |
|---|---|---|---|---|---|
| **M1** | main_candidate | **Prophet 去趋势 + Spearman 残差相关 + DTW 单品形状聚类 + 格兰杰因果网络** | 消除共同时间趋势——文献共性问题；DTW 按销售形态聚类（非仅量级）；Granger 区分因果方向；Spearman 不需要正态假设 | 残差中仍可能残留节假日效应；DTW 在 116 个单品上 O(n²) 可接受 | 低（Prophet + scipy + statsmodels） |
| **M2** | usable_baseline | **Spearman 原始相关 + K-means++ 聚类**（5/5 文献标准做法） | 与文献完全可对比；Spearman 不依赖正态性；K-means++ 成熟稳定 | 不区分时间趋势伪相关；仅按量级+波动聚类，忽略形状 | 低（纯 scipy/sklearn） |

## Baseline validity

- Real task completed: 是——输出品类和单品关联矩阵+聚类标签
- Comparable output/metric: 是——Spearman 相关系数矩阵 + 聚类轮廓系数
- Classification: usable_baseline ✓

## Risk-probe summary

| ID | Executability | Data/assumptions | Degeneracy | Sensitivity | Scale | Verdict |
|---|---|---|---|---|---|---|
| M1 | PASS | PASS (116 stable items for item-level, 1085 days for category) | PASS (corr range [-0.21, 0.63], 15 unique values) | PASS (bootstrap MAD 0.018) | PASS (<2s full run) | **PASS** |
| M2 | PASS | PASS | PASS | PASS | PASS (<1s) | **PASS** |

## Fallback trigger

N/A — 无可触发的条件备选。Q1 是探索性分析，无单一"失败"门槛。

## Compact history

- 2026-08-06: Initial card. M1 (Prophet+DTW+Granger) proposed as main; M2 (Spearman+K-means) as baseline. Both probe-verified.
