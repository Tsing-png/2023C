# 外卖配送优化 — Lean 模式全完成状态

## 项目结构

```
2023C/
├── AGENTS.md                         # 工作流政策 + 论文写作规范(反AI味)
├── CLAUDE.md                         # Claude 操作规则
├── .claude/settings.json             # 权限规则 (raw data冻结, git push门控)
│
├── literature/                       # 17篇PDF文献 + 综述
│   ├── literature_synthesis.md       # 方法矩阵 + 9创新点
│   ├── 问题一/ 问题二/ 问题三/ 问题四/
│   └── _extracts.json
│
├── planning/                         # 状态管理
│   ├── session_config.json           # learning + lean
│   ├── framing_decisions.jsonl       # 全局框架决策 (4项)
│   ├── parse/problem_parse.json      # 赛题解析
│   ├── classification/problem_classification.json
│   └── manifests/Q1.json Q2.json Q3.json Q4.json
│
├── methods/                          # 方法卡片 + 决策记录 + 风险探针
│   ├── Q1/ (q1_method_card.md, q1_decisions.jsonl, probes/)
│   ├── Q2/ (q2_method_card.md, q2_decisions.jsonl, probes/)
│   └── Q3/ (q3_method_card.md, q3_decisions.jsonl, probes/)
│
├── code/                             # Python实现
│   ├── Q1/ (q1_main.py, q1_baseline.py, q1_code_plan.md, reviews/)
│   ├── Q2/ (q2_main.py, q2_baseline.py, q2_code_plan.md, reviews/)
│   └── Q3/ (q3_main.py, q3_baseline.py, reviews/)
│
├── workspace/
│   ├── problem/                      # 原始附件 (1-4.xlsx) + problem.md
│   ├── data_clean/                   # 清洗后CSV (4个)
│   ├── data/data_profile.json        # 数据画像
│   └── code/scripts/                 # 审计/探针/鲁棒性脚本
│
├── results/                          # 实验结果
│   ├── Q1/experiments/round1/ (6figs + 6tables + metrics + run_summary)
│   ├── Q2/experiments/round1/ (4figs + 5tables + metrics + run_summary)
│   ├── Q3/experiments/round1/ (3figs + 3tables + metrics + run_summary)
│   ├── Q4/reports/q4_data_recommendations.md
│   └── reports/scoped_consistency.json
│
└── robustness/                       # 鲁棒性检查
    ├── Q1/q1_robustness_summary.json
    ├── Q2/q2_robustness_summary.json
    └── Q3/q3_robustness_summary.json
```

## 最终数值摘要

| 指标 | 值 | 来源 |
|---|---|---|
| Q1 品类关联 (去趋势后) | mean \|r\|=0.266, 花叶-花菜 r=0.633 | `q1_metrics.json` |
| Q1 单品集中度 | Gini=0.793, Top20%→83.9% | `q1_metrics.json` |
| Q1 Granger 因果边 | 20/30 显著 (p<0.1) | `q1_granger_pvalues.csv` |
| Q1 单品聚类 sil | DTW-HC=0.17, K-means=0.18 | `q1_metrics.json` |
| Q2 7天预期利润 (M1) | **2,587 元** | `q2_optimal_decisions.csv` |
| Q2 平均内生加成率 | 95% | `q2_metrics.json` |
| Q2 需求模型 R² | 0.51–0.85 (double-log) | `q2_metrics.json` |
| Q2 利润对 σ 敏感 | 减半→+135%, 加倍→亏损 | `q2_robustness_summary.json` |
| Q3 单品数/利润 (M1) | 30 SKU, **1,225 元/天** | `q3_selected_items.csv` |
| Q3 vs M2 | M1 利润 = 5× M2 (253元) | `q3_metrics.json` |
| Q3 加成率边界饱和 | 30/30 触碰 150% 上界 | `q3_selected_items.csv` |
| Q3 花菜类瓶颈 | 仅2候选, 无冗余 | `q3_selected_items.csv` |

## 三项模型局限性

1. **单品聚类 silhouette 偏低 (0.17)** — DTW-Ward 层次聚类在 116 个稳定单品上分离度不足, 说明销量形状差异不显著, 单品层面缺乏自然聚类结构
2. **花菜类候选稀缺 (2个单品)** — Q3 中花菜类仅有 2 个可售单品, 是品类覆盖约束的瓶颈, 两单品被强制选中, 无冗余空间
3. **加成率边界饱和 (100%)** — Q2 3/6 品类和 Q3 全部 30 个单品的最优加成率触碰 150% 上界, 品类级弹性估计应用于单品导致模型无法区分各单品的最优定价

## Lean 模式全完成清单

| Gate | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| G1 (解析+分类+数据+框架) | ✅ | ✅ | ✅ | ✅ |
| G2 (方法卡片+风险探针) | ✅ | ✅ | ✅ | N/A |
| G2.5 (人类方法选择) | ✅ | ✅ | ✅ | N/A |
| G3 (代码+基线+审查) | ✅ | ✅ | ✅ | N/A |
| G4 lean (结果判断+鲁棒性+结果报告) | ✅ | ✅ | ✅ | ✅ |
| G5 (论文写作) | ⏳ | ⏳ | ⏳ | ⏳ |
| G6 (最终审计) | ⏳ | ⏳ | ⏳ | ⏳ |
