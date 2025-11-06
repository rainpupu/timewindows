
# RQ2: Stable Early-Feature Groups via Conditional Perturbation Importance (CPI)

**Goal.** Identify which **early-career feature groups** remain consistently predictive across different windows \(W\) (e.g., first \(W{+}1\) career years), while **respecting correlations** among features.

This document summarizes the method, equations, and reporting recipe for **group-level CPI** as implemented in your codebase (LSTM forecaster + conditional kNN replacement).

---

## 1. Why Conditional, Group-Level Importance?

- **Correlated features** break conventional permutation importance: shuffling one feature may create *unrealistic* samples and spuriously inflate/deflate importance. Conditional permutation avoids this by **holding the joint structure fixed** and only removing the *independent* information of the target variables.
- **Groups not singletons.** Early-career metrics are often collinear or semantically linked (e.g., `n_paper` and `n_paper_num`). Grouping avoids attribution ping‑pong among highly related variables and answers “**which kind of information** does the model rely on?”

**Key references (see full list at the end):** Strobl *et al.* (2008) conditional permutation for RF; Hooker & Mentch (2019) on extrapolation pitfalls when permuting; Gregorutti *et al.* (2015) grouped importance; Aas–Jullum–Løland (2021) conditional SHAP under dependence; Altmann *et al.* (2010) repeated-permutation inference.

---

## 2. Group-Level CPI: Definition

Let \(f\) be the trained predictor, \(\mathcal L\) a loss (MSE/MAE/\(1{-}\)Spearman), and \(G\) a **feature group**. We define CPI as the mean **loss increase** after conditionally replacing only the columns in \(G\).

**Importance**
$$
\mathrm{imp}(G)
= \frac{1}{B}\sum_{b=1}^{B}
\Big[
\mathcal{L}\!\big(y,\ f(X^{(b)}_{\setminus G})\big)
- \mathcal{L}\!\big(y,\ f(X)\big)
\Big].
$$

**Standard error**
$$
\mathrm{se}(G)
= \frac{\operatorname{std}\!\left(\Delta_1,\ldots,\Delta_B\right)}{\sqrt{B}},
\quad
\Delta_b
= \mathcal{L}\!\big(y,\ f(X^{(b)}_{\setminus G})\big)
- \mathcal{L}\!\big(y,\ f(X)\big).
$$

**Normalized share (for visualization)**
$$
\mathrm{imp\_norm}(G)
=
\frac{\max\!\big(0,\ \mathrm{imp}(G)\big)}
{\sum\limits_{H}\max\!\big(0,\ \mathrm{imp}(H)\big)}.
$$

### Constructing the conditional perturbation

Let the sequence length of sample \(i\) be \(L_i\). For iteration \(b\), choose a **conditional neighbor** \(j=i_b\) in the *condition set* (all **group‑external** last‑year features) using kNN. Replace the whole time‑series of group‑columns \(G\) with the neighbor’s values:

$$
X^{(b)}_{\setminus G}[i, t, c]
=
\begin{cases}
X[j,\ \min(t,\ L_j-1),\ c], & c\in \mathcal{C}_G,\ \ 0\le t < L_i, \\\\
X[i,\ t,\ c], & \text{otherwise}.
\end{cases}
$$

Notes:
- Only the **yearly feature segment** is perturbed; **embeddings** (e.g., GCN vectors) stay fixed.
- kNN is built on the **last‑year condition subspace** to preserve realistic dependency structure and keep computations tractable.
- Use \(B=10\text{–}50\) for stable estimates; report `imp` + `se` + `imp_norm`.

---

## 3. Per‑W Grouping and Cross‑W Analysis (Lineage)

- For each \(W\), build groups via hierarchical clustering on the **absolute last‑year correlation** matrix, distance \(d=1-|\rho|\), threshold \(\tau\) (e.g., \(\tau=0.70\)).
- Groups will **evolve** as \(W\) grows. To analyze stability across \(W\), track **lineages** between successive partitions using **Jaccard‑based maximum matching**:
  - **continue** if Jaccard\(\ge \theta_c\) (e.g., 0.5);
  - **split** / **merge** for multi‑match cases with Jaccard\(\ge \theta_s\) (e.g., 0.3);
  - **birth/death** otherwise.
- Plot per‑lineage curves of `imp_norm` vs. \(W\); add global partition indices (entropy \(H(W)\), effective number of groups \(EN_g\)).

---

## 4. Reporting Checklist

1. `groups_map.csv` per \(W\): group → member features.  
2. `imp_W_groups.csv` per \(W\): `group, imp, se, B, imp_norm`.  
3. Stability view: lineage table and curves; optionally category‑level aggregation (产出/合作/中心性/引用等).  
4. Sensitivity: vary \(B\), \(K\), and correlation threshold \(\tau\).

---

## 5. References

- **Conditional permutation importance (CPI).**  
  *Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T.* (2008). Conditional variable importance for random forests. **BMC Bioinformatics**, 9, 307.  
  URL: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-307

- **Why unconditional permutation fails under dependence.**  
  *Hooker, G., Mentch, L., & Zhou, S.* (2019). Unrestricted Permutation forces Extrapolation: Variable Importance Requires at least One More Model. **arXiv:1905.03151**.  
  URL: https://arxiv.org/abs/1905.03151

- **Grouped variable importance (random forests).**  
  *Gregorutti, B., Michel, B., & Saint‑Pierre, P.* (2015). Grouped variable importance with random forests and application to multiple functional data analysis. **Computational Statistics & Data Analysis**, 90, 15–35.  
  URL: https://www.sciencedirect.com/science/article/abs/pii/S0167947315000997

- **Conditional SHAP for dependent features.**  
  *Aas, K., Jullum, M., & Løland, A.* (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. **Artificial Intelligence**, 298, 103502.  
  Open PDF: https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf

- **Inference via repeated permutation (PIMP).**  
  *Altmann, A., Toloşi, L., Sander, O., & Lengauer, T.* (2010). Permutation importance: a corrected feature importance measure. **Bioinformatics**, 26(10), 1340–1347.  
  URL: https://academic.oup.com/bioinformatics/article/26/10/1340/193348

- **CPI revisited and clarified.**  
  *Gromping, U.* (2020). Conditional permutation importance revisited. **BMC Bioinformatics**, 21, 332.  
  URL: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03622-2

---

## 6. Minimal “How‑to” (fits your code)

1. Build validation tensors and model for each \(W\): `build_data_for_W`, `build_model_for_W`, load `best_W{W}.pt`.
2. Cluster last‑year features into groups (`d=1-|ρ|`, threshold \(\tau\)).
3. For each group \(G\): build kNN on **group‑external** last‑year features; repeat \(B\) times conditional replacement of full sequences in \(G\); record \(\mathrm{imp}(G), \mathrm{se}(G)\).
4. Save `imp_W_groups.csv` and `groups_map.csv`.  
5. (Optional) Do lineage matching across \(W\) for stability curves.

