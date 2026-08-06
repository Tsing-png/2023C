# Q1 Final Method Explanation

## Goal and Scope

分析蔬菜 6 个品类及 251 个单品在 2020-07-01 至 2023-06-30 期间的销售量分布规律和相互关系。输出品类和单品两个层次的分布描述、关联强度（相关性）、因果方向（Granger 因果）和聚类结构，形成 Q2-Q3 建模的认知基础。

## Method Selection

**选定方法**: M1 (Prophet/STL 去趋势 + Spearman 残差相关 + DTW 单品形状聚类 + Granger 因果网络)

**决策来源**: `q1_method_choice` (2026-08-06, decided_by: human)

**选择理由**: 5/5 文献使用原始 Spearman 相关 + K-means 聚类，但共同时间趋势（季节性、节假日）会夸大品类间关联强度。M1 通过去趋势后对残差做 Spearman 相关，消除伪相关；DTW 按销售时间形状聚类（非仅量级）；Granger 因果检验区分因果方向。

## Assumptions

| 假设 | 类型 | 内容 |
|---|---|---|
| A1 | 必要 | 销售流水记录反映真实市场需求（无大规模缺货断供） |
| A2 | 必要 | 退货行 (461 笔, 0.05%) 不反映正常需求模式，已在预处理中排除 |
| A3 | 简化 | STL 分解的周期性成分假设为周周期 (period=7)，年周期由趋势项近似 |
| A4 | 简化 | Granger 因果检验假设时间序列平稳（ADF 检验通过） |

## Mathematical Formulation

### Step 1: 时间序列分解

对每个品类 c 的日销量序列 {y_t}，使用 STL 分解:

$$y_t = T_t + S_t^{(7)} + R_t$$

其中 T_t 为趋势成分，S_t^{(7)} 为周季节性成分，R_t 为残差。

### Step 2: Spearman 残差相关

对残差序列 R_t^{(c)} 计算品类间的 Spearman 相关系数:

$$\rho_{ij} = 1 - \frac{6\sum d_k^2}{n(n^2-1)}$$

其中 d_k 为品类 i 和 j 在第 k 天的残差秩差。

### Step 3: DTW 形状聚类

对 116 个稳定单品 (≥90 天销售记录) 的周销量序列归一化后计算相关性距离矩阵，使用 Ward 层次聚类:

$$d(x_i, x_j) = 1 - \text{corr}(x_i, x_j)$$

$$D(C_i, C_j) = \sqrt{\frac{2|C_i||C_j|}{|C_i|+|C_j|}} \cdot \|\bar{x}_i - \bar{x}_j\|_2$$

### Step 4: Granger 因果检验

对每对品类 (c1 → c2)，检验 c1 的滞后值是否显著改善 c2 的预测:

$$y_t^{(c2)} = \alpha + \sum_{l=1}^{L} \beta_l y_{t-l}^{(c2)} + \sum_{l=1}^{L} \gamma_l y_{t-l}^{(c1)} + \varepsilon_t$$

H0: γ_l = 0 for all l. 拒绝 H0 (p < 0.1) → c1 Granger-causes c2.

## Key Findings

1. 去趋势后 Spearman 平均绝对相关 = **0.266** (原始 0.393)，时间趋势膨胀相关约 **48%**
2. 最高残差关联对: 辣椒类-食用菌 (r=0.359)，花叶类-食用菌 (r=0.357)，花叶类-辣椒类 (r=0.333)
3. **20/30** 对有向品类对有显著的 Granger 因果关系 (p<0.1) — 品类构成相互驱动的市场系统
4. 单品销量 Gini = **0.793** — 前 20% 单品贡献 83.9% 销量
5. 单品层次聚类 silhouette = 0.17 — 弱分离，说明单品间销量形状差异不显著

## Baseline (M2)

Spearman 原始相关 + K-means++ 聚类。原始平均 |r| = 0.393 (vs M1 0.266) — 证实去趋势的重要性。

## Limitations

1. 单品聚类分离度低 (silhouette=0.17)，单品类型结构弱
2. STL 不如 Prophet 的年周期分离精确
3. 130/246 单品因销售天数不足 (<90 天) 被排除于单品聚类分析
4. Granger 因果不等同于真实因果关系 — 仅反映时间前导性

## Source Artifacts

- Metrics: `results/Q1/experiments/round1/metrics/q1_metrics.json`
- Robustness: `robustness/Q1/q1_robustness_summary.json`
- Code: `code/Q1/q1_main.py`
