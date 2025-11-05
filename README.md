# 项目名称
TIMEWINDOW
# 研究问题
**RQ1：预测性能随 W 如何变化，是否存在拐点/饱和点？**

**RQ2：哪些特征在不同 W 下保持稳定预测力？**

**RQ3：不同子领域（AI/Systems/Theory…）的最佳 W 是否不同？**

# 核心创新
**核心创新：**

**COW 指标**（成本感知最优窗口）：COW(W)=S(W)−λ⋅W/Wmax，兼顾性能与等待成本

**稳定特征评估框架**：SHAP 重要性随 W的变异系数/排名稳定性

**MIG 边际信息增益**：MIG(W)=Perf(W)−Perf(W−1)，量化“多等一年是否值得”

# 实验步骤

## 1. 对DBLP数据集按年份切片

**程序：**`mini_cut_by_year.py` 

**结果：**
>  dblp_out/shards_by_year_mini/..



## 2.统计每一个作者的入行年份并保留1998-2012入行作者名单
**程序：** `startyear.py`
**结果**：
> dblp_out/author_start_years_all.parquet
 dblp_out/author_start_years_1998_2012.parquet
dblp_out/author_ids_1998_2012.parquet


## 3.将parquet格式转换为jsonl格式
**程序：** `change_format_to_json.py`
**结果**：
> dblp_out/exports/author_start_years_1998_2012.jsonl


## 4.将1998-2009入行学者按年份切片，分别保存对应信息
**程序：**`group_authors_by_start_year.py`
**结果**：
> dblp_out/exports/by_start_year/..

## 5.截断1998-2022年论文并保存对应论文的全部信息（去除预印本）
**程序：**`cut1998_2022.py`
**结果**：
> dblp_out/slice_1998_2022_nopreprint/..

## 6.把1998-2022年论文按时间切片并保存对应论文的全部信息，并统计了每年的存储论文数（去除预印本）
**程序：**`cut_slices.py`
**结果**：
> dblp_out/year_slices/..

 - [ ] 进而决定使用**1998-2009**年入行学者（2020年起DBLP数据大量缺失，而2009年入行学者的第11年数据刚好截止至2019年）**!!**



## 7.补全入行学者对应每年发表的论文信息（feature部分不作使用）
**程序:**`complete_info_and_features.py`
**结果**：
> full_info_of_author_new/{YYYY}_from_slice/author_papers.jsonl

## 8.在CCF和中国科学院官网下载分区信息

**功能**：用于统计学者每年顶刊（CCF-A,中科院1区）和中顶刊（CCF-B.中科院2区）的一作数

**结果**：
> CCF 推荐会议期刊目录 2019.xlsx
中科院分区2019.xlsx

## 9.补全所有1998-2009年入行学者的全部论文信息并计算相关特征指数
### 程序
`final_info.py`
### 结果
> full_info_of_author_new/{YYYY}_from_slice/author_year_features_full.jsonl
### 具体特征
| name | 含义 |
|--|--|
| author_id | 作者id |
| start_year | 入行年份 |
| year | 当前年份 |
|n_papers|当年发表论文数|
| n_papers_cum | 从入行年起累计到当年的论文总数 |
| first_author_share | 当年“一作论文数 / 当年论文总数”的占比 |
| avg_team_size | 当年每篇论文的作者数的平均值 |
| single_author_share | 当年“单作者论文数 / 当年论文总数”的占比 |
| venues_dedup_year | 当年去重后的发表 venue 数 |
| unique_coauthors_year | 当年合作过的唯一合作者人数 |
| new_coauthors_year | 当年第一次出现的新合作者人数。 |
| cum_unique_coauthors | 从入行起累计的唯一合作者人数 |
| repeat_collab_ratio | 当年“与既有合作者的合作次数 / 当年全部合作次数”的比例 |
| top_first | 当年顶会/顶刊的一作篇数 |
| mid_first | 当年“中顶”层级的一作篇数 |

## 10.统计所有学者截止入行第11年每年累计引文数
### 程序
`cohort_career11_from_slices_progress_series_peryear.py`
### 结果
> full_info_of_author_new/{YYYY}_from_slice/authors_career11_citations_series.jsonl

## 11.统计所有学者入行第11年的h-index作为最终的回归指标
### 程序
`career11_hindex_from_slices_peryear.py`
### 结果
> full_info_of_author_new/{YYYY}_from_slice/authors_career11_hindex.jsonl
full_info_of_authors_new/{YYYY}_from_slice/career11_hindex_summary.json

## 12.构建合著网络
### 程序
`career11_hindex_from_slices_peryear.py`


### 内容
1998-2018年合著图快照，包括所有入行学者，一跳合作学者，二跳强边（合作阈值大于等于2，top20）

### 结果
> graphs\snapshots\..

## 13.GCN输出图向量
### 程序
`GCN.py`
### 结果
> embeddings_gcn\..

## 14.LSTM训练并回归
### 程序
`LATM.py`

### 内容
在LSTM中输入图向量+时序特征序列，分不同时间窗口训练，每个窗口训练6次，找到最优模型参数保存

### 结果
> timeWindow\runs\lstm_windows\best_W{}.pt
> statics_basic_info.py

## 15.保存预测的h-index，并计算模型准确度
### 程序
`forecast_h_index(1).py`

### 内容
准确度包括：RMSE、Spearman

### 结果
> runs\lstm_windows\pre_h_index\..
runs\lstm_windows\metrics_eval_0-10.jsonl

![输入图片说明](/imgs/RMSE.png)
![输入图片说明](/imgs/plot_val_Spearman_W0-10.png)
![输入图片说明](/imgs/plot_test_Spearman_W0-10.png)

**Spearmen：**只看排序是否一致的相关系数（-1～1）。越接近 1，说明模型把“谁更高谁更低”排得越准；对单调变换（如 log/exp）不敏感。
**RMSE:**预测值和真值的数值差的均方根（单位=目标本身的单位）。这里就是“平均偏差多少个 h 指数点”。越小越好，对离群误差更敏感。
## 16.h-index的命中率
### 程序
`compare_top1_and_10.py`

### 内容
计算预测的top1%和top10%在不同窗口W下的命中率

**等人数评估**：每年先数出真实 Top 的人数 ，再从预测排序里取**同样的人数 ** 个作“预测 Top”。这样 precision=recall=命中率，更公平，也能显著抬高现在被“并列”压住的召回。

### 结果
> runs\lstm_windows\match_pred_vs_real\..
其中summary为统计的最终结果，其他为具体人员名单

![输入图片说明](/imgs/plot_top1_precision.png)
![输入图片说明](/imgs/plot_top10_precision.png)

<font color="red">**结论： 存在W=6/7的饱和点**</font>






















