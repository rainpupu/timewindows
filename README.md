<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README_EN.md">🇺🇸 English</a>
</p>

<div align="center">
  <h1>🧠 TIMEWINDOW</h1>
  <p><b>Early-career Observation Window Analysis for Academic Development Prediction</b></p>
  <p><i>When can we make a reliable prediction?</i></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Academic_Development_Prediction-blue.svg" alt="Task">
  <img src="https://img.shields.io/badge/Model-GCN%20%2B%20LSTM%20%2B%20MLP-brightgreen.svg" alt="Model">
  <img src="https://img.shields.io/badge/Data-DBLP-orange.svg" alt="Data">
  <img src="https://img.shields.io/badge/Window-1--10_years-purple.svg" alt="Window">
</p>

---

## 🔍 背景（Background）

在青年人才评估、科研资助配置与学术政策设计中，如何基于科研人员早期阶段的学术行为，对其长期发展潜力进行预测，已成为一个关键问题。

现有研究通常聚焦于 **feature engineering** 与 **model design**，但往往默认采用固定的 early-career observation window，而较少追问：

> **究竟需要观察多久，才能在“预测可靠性”与“决策及时性”之间取得平衡？**

---

## ❓ 研究问题（Research Questions）

- **RQ1**：预测性能如何随 observation window \(W\) 变化？是否存在拐点或饱和区间？  
- **RQ2**：哪些特征在不同窗口下保持稳定预测能力？  
- **RQ3**：不同子领域（AI / SE / Theory）是否存在不同最优窗口？

---

## ✨ 核心思路（Core Idea）
> ```md
> <p align="center">
>   <img src="./images/framework.png" alt="Framework" width="760"/>
> </p>
> ```
---

## ⚙️ 实验流程（Pipeline）

> **Data → Feature → Graph → Sequence → Evaluation**

### 1️⃣ 数据切片与 cohort 构建（Data Construction）

基于 DBLP 构建计算机科学领域作者级 longitudinal dataset，并按年份对论文记录进行切片与清洗。

**相关代码与功能：**

- `mini_cut_by_year.py`：按 publication year 对原始 DBLP 数据进行年度切片
- `startyear.py`：统计作者首次发表年份，识别作者入行时间
- `change_format_to_json.py`：将中间结果由 `parquet` 转换为 `jsonl`，便于后续逐年处理
- `group_authors_by_start_year.py`：按 start year 对作者 cohort 分组保存
- `cut1998_2022.py`：截取 1998–2022 年论文记录，并去除 preprint
- `cut_slices.py`：进一步生成年度 paper slices，并统计每年的保留论文数

> **说明：** 最终选取 **1998–2009** 年入行学者作为研究对象。2009 年入行学者的第 11 年对应 2019 年，能够保证 h@11 标签完整；同时 2020 年后 DBLP 部分记录缺失较明显，纳入更晚 cohort 会影响 temporal consistency。

### 2️⃣ 作者级特征构建（Feature Engineering）

在每位学者的 early-career 阶段，逐年提取生产力、合作结构与研究质量三类特征。

**相关代码与功能：**

- `complete_info_and_features.py`：补全作者逐年论文信息，为特征构建准备 author-year 级基础记录
- `final_info.py`：计算并保存作者逐年特征，输出 `author_year_features_full.jsonl`
- 外部数据：`CCF 推荐会议期刊目录 2019.xlsx`、`中科院分区2019.xlsx`
  - 用于识别 top-tier / mid-tier venue 并统计相应一作产出

**Feature Set：**

| 类别 | 特征名 | 含义 |
|------|--------|------|
| 生产力 | `n_papers` | 年发文数 |
| 生产力 | `n_papers_cum` | 累计发文 |
| 生产力 | `venues_dedup_year` | venue 数 |
| 合作结构 | `avg_team_size` | 团队规模 |
| 合作结构 | `unique_coauthors_year` | 合作者数 |
| 合作结构 | `new_coauthors_year` | 新合作者 |
| 合作结构 | `cum_unique_coauthors` | 累计合作者 |
| 合作结构 | `repeat_collab_ratio` | 重复合作比例 |
| 合作行为 | `first_author_share` | 一作占比 |
| 合作行为 | `single_author_share` | 单作者占比 |
| 质量 | `top_first` | 顶级一作 |
| 质量 | `mid_first` | 中级一作 |

> **预处理：** 对 count-based features 进行 `log transform`，并进一步采用 `z-score normalization`，以减弱长尾分布与尺度差异对训练的影响。

### 3️⃣ 长期影响标签构建（Label Construction）

围绕入行第 11 年，构造作者级长期影响标签，包括 citation series 与 h-index。

**相关代码与功能：**

- `cohort_career11_from_slices_progress_series_peryear.py`：统计作者截至 career year 11 的逐年累计引文序列
- `career11_hindex_from_slices_peryear.py`：计算作者在 career year 11 的 h-index，并生成 cohort summary

**输出内容：**

- `authors_career11_citations_series.jsonl`
- `authors_career11_hindex.jsonl`
- `career11_hindex_summary.json`

### 4️⃣ 合著网络构建（Graph Modeling）

构建年度 co-authorship graph snapshots，并提取作者在合作网络中的结构表示。

**相关代码与功能：**

- `build_coauthor_map.py`：构建 1998–2018 年合著图快照
- `GCN.py`：基于图快照训练 GCN，输出年度结构嵌入 `embeddings_gcn/...`

**图构建说明：**

- 节点：入行学者及其一跳、二跳邻居
- 边：同年合作关系
- 权重：共同发表论文数
- 强边筛选：合作阈值与 top-N 约束

> **作用：** GCN 用于编码研究者在年度合作网络中的结构位置，以补充行为特征难以表达的 relational information。

### 5️⃣ 时序建模与多窗口训练（Sequence Modeling）

将 graph embeddings 与 author-year features 拼接为多变量时间序列，在不同 \(W\) 下分别训练模型。

**相关代码与功能：**

- `LATM.py`：在不同 observation window 下训练 LSTM 回归模型，并保存最优参数
- 训练输出：`timeWindow/runs/lstm_windows/best_W{}.pt`

**模型输入：**

- 图向量（GCN embeddings）
- 时序行为特征（author-year features）

**模型目标：**

- 预测入行第 11 年的 h-index（h@11）

**训练策略：**

- 对每个 \(W\) 单独训练模型
- 每个窗口重复训练 6 次，选取最优结果保存

### 6️⃣ 预测评估与可视化（Evaluation & Visualization）

从回归、排序与顶尖学者识别三个层面评估不同 observation window 的预测表现。

**相关代码与功能：**

- `forecast_h_index(1).py`：保存 h-index 预测结果并计算 `RMSE`、`Spearman`
- `compare_top1_and_10.py`：计算 Top-1% / Top-10% 命中率

**评估指标：**

- `RMSE`：数值误差，衡量预测值与真实值的平均偏差
- `Spearman`：排序一致性，衡量模型是否正确区分“谁未来更高”
- `Hit@1% / Hit@10%`：衡量顶尖学者的识别能力

> **评估策略：** Top scholar evaluation 采用“等人数评估”方式：对每年先确定真实 Top 人数，再从预测排序中取相同人数进行匹配，从而避免并列排名对 recall 的不公平压制。

---

## 📊 结果（Results）

<p align="center">
  <img src="./images/resultes.png" alt="Observation window results" width="900"/>
</p>

**整体趋势。** 随着 observation window 从 1 年逐步扩展到 10 年，模型的预测性能整体持续提升，但增长速度并不均匀：前 4 年提升最为显著，之后增益逐渐减弱并趋于平台。

**总领域结果。** 从整体样本来看，Spearman 随窗口增大快速上升，并在 4 年左右出现明显“拐点”；结合 Hit@1% 与 Hit@10% 的变化趋势可见，对于整体排序一致性以及 Top 10% 学者识别，4 年已经能够提供较高的信息密度，而对于 Top 1% 顶尖学者的识别，5 年窗口会更稳健。

**子领域差异。** 在 AI、Theory 和 SE 三个子领域中，性能曲线整体都呈现“先快升、后趋稳”的共同模式，但最优窗口并不完全一致： **AI = 4、Theory = 5、SE = 6**；而在 Hit@K 任务上，Top 10% 的最优窗口更集中在 5 年左右，Top 1% 的识别则体现出更明显的学科差异。

这些结果共同说明：**最优 observation window 并非越长越好，而是存在一个信息收益显著放缓的拐点；同时，这一拐点在总体上较为稳定，但在不同子领域中会表现出一定程度的差异。**

---

## 🚀 项目价值（Takeaway）

- **对青年人才评估机制的启发**  
  明确“何时评估”这一长期依赖经验的问题

- **对科研资助决策的支持**  
  平衡“信息充分性”与“决策时机”

- **对学术发展预测问题的重新定义**  
  从“能否预测”转向“何时预测”

- **对跨学科问题的参考意义**  
  `window-as-variable` 视角可迁移

---

## ⭐ 核心结论

<p align="center">
  <b>Optimal Window ≈ 4–5 years</b>
</p>

<p align="center">
  <i>Most predictive information emerges early.</i>
</p>
