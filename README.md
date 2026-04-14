

# 🧠 TIMEWINDOW  
### Early-career Observation Window Analysis for Academic Development Prediction

<p align="center">
  <i>When can we make a reliable prediction?</i>
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

> 将 observation window 从“固定设定”转化为“可分析变量”

- 统一建模：**GCN + LSTM + MLP**
- 同一模型，不同窗口训练
- 

---

## 🏷 任务定义（Task Definition）

- **主任务：h@11（Regression）**  
  → 入行第 11 年的 h-index  

- **辅助任务：Top-1% / Top-10%（Classification）**  
  → 是否进入顶尖学者区间  

---

## ⚙️ 实验流程（Pipeline）

> **Data → Feature → Graph → Sequence → Evaluation**

---

### 1️⃣ 数据切片与 cohort 构建（Data Construction）

基于 DBLP 构建计算机科学领域作者级 longitudinal dataset，并按年份对论文记录进行切片与清洗。

<sub>

**相关代码与功能：**

- `mini_cut_by_year.py`：按 publication year 对原始 DBLP 数据进行年度切片  
- `startyear.py`：统计作者首次发表年份，识别作者入行时间  
- `change_format_to_json.py`：将中间结果由 `parquet` 转换为 `jsonl`，便于后续逐年处理  
- `group_authors_by_start_year.py`：按 start year 对作者 cohort 分组保存  
- `cut1998_2022.py`：截取 1998–2022 年论文记录，并去除 preprint  
- `cut_slices.py`：进一步生成年度 paper slices，并统计每年的保留论文数  

**设计说明：**

最终选取 **1998–2009** 年入行学者作为研究对象。原因在于：  
2009 年入行学者的第 11 年对应 2019 年，能够保证 h@11 标签完整；同时 2020 年后 DBLP 部分记录缺失较明显，纳入更晚 cohort 会影响 temporal consistency。

</sub>

---

### 2️⃣ 作者级特征构建（Feature Engineering）

在每位学者的 early-career 阶段，逐年提取生产力、合作结构与研究质量三类特征。

<sub>

**相关代码与功能：**

- `complete_info_and_features.py`：补全作者逐年论文信息，为特征构建准备 author-year 级基础记录  
- `final_info.py`：计算并保存作者逐年特征，输出 `author_year_features_full.jsonl`  
- 外部数据：`CCF 推荐会议期刊目录 2019.xlsx`、`中科院分区2019.xlsx`  
  用于识别 top-tier / mid-tier venue 并统计相应一作产出

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


**预处理说明：**

对 count-based features 进行 log transform，并进一步采用 z-score normalization，以减弱长尾分布与尺度差异对训练的影响。

</sub>

---

### 3️⃣ 长期影响标签构建（Label Construction）

围绕入行第 11 年，构造作者级长期影响标签，包括 citation series 与 h-index。

<sub>

**相关代码与功能：**

- `cohort_career11_from_slices_progress_series_peryear.py`：统计作者截至 career year 11 的逐年累计引文序列  
- `career11_hindex_from_slices_peryear.py`：计算作者在 career year 11 的 h-index，并生成 cohort summary  

**输出内容：**

- `authors_career11_citations_series.jsonl`
- `authors_career11_hindex.jsonl`
- `career11_hindex_summary.json`

</sub>

---

### 4️⃣ 合著网络构建（Graph Modeling）

构建年度 co-authorship graph snapshots，并提取作者在合作网络中的结构表示。

<sub>

**相关代码与功能：**

- `build_coauthor_map.py`：构建 1998–2018 年合著图快照  
- `GCN.py`：基于图快照训练 GCN，输出年度结构嵌入 `embeddings_gcn/...`

**图构建说明：**

- 节点：入行学者及其一跳、二跳邻居  
- 边：同年合作关系  
- 权重：共同发表论文数  
- 强边筛选：合作阈值与 top-N 约束

GCN 部分主要用于编码研究者在年度合作网络中的结构位置，以补充行为特征难以表达的 relational information。

</sub>

---

### 5️⃣ 时序建模与多窗口训练（Sequence Modeling）

将 graph embeddings 与 author-year features 拼接为多变量时间序列，在不同 \(W\) 下分别训练模型。

<sub>

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
- 每个窗口重复训练6次，选取最优结果保存  

</sub>

---

### 6️⃣ 预测评估与可视化（Evaluation & Visualization）

从回归、排序与顶尖学者识别三个层面评估不同 observation window 的预测表现。

<sub>

**相关代码与功能：**

- `forecast_h_index(1).py`：保存 h-index 预测结果并计算 `RMSE`、`Spearman`  
- `compare_top1_and_10.py`：计算 Top-1% / Top-10% 命中率  


**评估指标：**

- `RMSE`：数值误差，衡量预测值与真实值的平均偏差  
- `Spearman`：排序一致性，衡量模型是否正确区分“谁未来更高”  
- `Hit@1% / Hit@10%`：衡量顶尖学者的识别能力  

**评估策略说明：**

Top scholar evaluation 采用“等人数评估”方式：  
对每年先确定真实 Top 人数，再从预测排序中取相同人数进行匹配，从而避免并列排名对 recall 的不公平压制。

</sub>

---

## 📊 结果（Results）

### 🔹 预测性能

<p align="center">
  <img src="/imgs/plot_val_Spearman_W0-10.png" width="65%">
</p>

- W=1→4：快速提升  
- W≥5：趋于饱和  

---

### 🔹 顶尖学者识别

<p align="center">
  <img src="/imgs/plot_top1_precision.png" width="60%">
  <img src="/imgs/plot_top10_precision.png" width="60%">
</p>

- 趋势与 Spearman 一致  
- 后期进入平台区间  

---

### 🔹 子领域差异

<p align="center">
  <img src="/imgs/subfield_results.png" width="45%">
</p>

<sub>
AI < Theory < SE，但整体集中在 4–6 年
</sub>

---

## 📌 机制洞察（Insights）

- **早期（W≤3）**：生产力 + 扩张  
- **中期（W≈4–6）**：多维平衡  
- **后期（W≥7）**：累计结构  

---

## 🚀 项目价值（Takeaway）

- **对青年人才评估机制的启发**  
  明确“何时评估”这一长期依赖经验的问题  

- **对科研资助决策的支持**  
  平衡“信息充分性”与“决策时机”  

- **对学术发展预测问题的重新定义**  
  从“能否预测”转向“何时预测”  

- **对跨学科问题的参考意义**  
  window-as-variable 视角可迁移  

---

## ⭐ 核心结论

<p align="center">
  <b>Optimal Window ≈ 4–5 years</b>
</p>

<p align="center">
  <i>Most predictive information emerges early.</i>
</p>

---







