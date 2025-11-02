# build_coauthor_snapshots_fixed.py
# 作用：基于 DBLP 按年作者-论文明细 + 1998–2009 入行学者清单，
#      生成 1998–2018 的【累计合著图快照】（TA∪一跳；累计共著次数为权重）。
# 产出：out/graphs/snapshots/edges_YYYY.parquet, nodes_YYYY.parquet

import os
from collections import Counter
# build_coauthor_snapshots_multi_authorfiles.py
# 作用：
#   - 从 full_info_of_authors_new\XXXX_from_slice\*.jsonl（1998–2009）读取目标作者 + 入行年
#   - 从 DBLP 按年 JSONL 切片读取论文作者列表
#   - 生成 1998–2018 的【累计合著图快照】（仅保留“至少一端为目标作者”的边）
#
# 输出：
#   out/graphs/snapshots/edges_YYYY.parquet  （src, dst, weight, year_abs）
#   out/graphs/snapshots/nodes_YYYY.parquet  （node_id, is_target, year_abs）

import os, re, json
from collections import Counter
from itertools import combinations
from typing import Dict, Set, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================
# 配置区（按需修改为你的真实路径）
# ==============================
# 1) 目标作者多文件目录（内部结构示例：full_info_of_authors_new/1998_from_slice/*.jsonl, 1999_from_slice/*.jsonl, ...）
AUTHORS_MULTI_DIR = r"full_info_of_authors_new"

# 2) DBLP 按年 JSONL 切片目录（文件名需能提取 4 位年份，如 DBLP_1998.jsonl）
DBLP_YEAR_DIR = r"dblp_out\year_slices\by_year"

# 3) 输出目录
OUTPUT_DIR = r"graphs"

# 4) 目标入行年范围、快照范围与可读取年份范围
TARGET_START_MIN = 1998
TARGET_START_MAX = 2009
INPUT_YEAR_START = 1998   # DBLP 切片可读的最小年份
INPUT_YEAR_END   = 2019   # DBLP 切片可读的最大年份
SNAPSHOT_START   = 1998   # 输出快照的最小绝对年
SNAPSHOT_END     = 2018   # 输出快照的最大绝对年（因 2009+10=2019，一般写到 2018）

# 5) 窗口与裁剪（控制规模）
WINDOW_YEARS     = 0      # 0=累计到当年；>0=滚动窗口（例如 3 表示仅累计最近 3 年）
MIN_WEIGHT       = 1      # 累计共著次数的最小阈值（强边门槛）
TOPK_PER_NODE    = 30     # 每个节点保留权重 Top-k 邻边；<=0 表示不启用

# 6) 是否额外纳入 N1<->N1 的强边（先保持 False，跑通后再考虑）
INCLUDE_N1N1_STRONG = True
N1N1_MIN_WEIGHT     = 2
N1N1_TOPK_PER_NODE  = 20
# ==============================


# ---------- 基本工具 ----------
def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")

def to_year(y) -> int:
    if y is None: return None
    try:
        return int(y)
    except Exception:
        s = str(y).strip()
        m = re.fullmatch(r"-?\d+", s)
        return int(s) if m else None

def year_from_text(txt: str) -> int:
    m = re.search(r"(19|20)\d{2}", txt)
    return int(m.group(0)) if m else None

def authors_list_in_dblp(rec: dict) -> List[dict]:
    """尽量兼容 DBLP 的作者字段结构。"""
    a = rec.get("authors") or rec.get("v12_authors") or []
    if isinstance(a, dict) and "author" in a:
        a = a.get("author") or []
    if isinstance(a, dict):
        a = [a]
    return a if isinstance(a, list) else []

def author_id_from_obj(aobj: dict) -> str:
    if not isinstance(aobj, dict):
        return None
    cand = aobj.get("id") or aobj.get("author_id")
    if cand is None:
        ids = aobj.get("ids")
        if isinstance(ids, list) and ids:
            cand = ids[0]
    return str(cand) if cand is not None else None

# ---------- 加载目标作者（多文件） ----------
def iter_author_jsonl_files(base_dir: str) -> Iterable[Tuple[int, str]]:
    """遍历 base_dir 下所有 .jsonl 文件，返回 (cohort_year, path)。
       允许目录名或文件名含 4 位年份（1998..2009）；若路径里没年份，则 cohort_year=None。"""
    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.lower().endswith(".jsonl"):
                continue
            path = os.path.join(root, fn)
            # 优先从目录/文件名提取年份
            text = os.path.join(root, fn).replace("\\","/")
            year = year_from_text(text)
            if year is not None and not (TARGET_START_MIN <= year <= TARGET_START_MAX):
                # 文件名上的年份不在目标范围则跳过（若你希望“文件名不准也尝试读”，把这里的 continue 去掉）
                continue
            yield (year, path)

def load_targets_from_multi_dir(base_dir: str) -> Tuple[Set[str], Dict[str,int]]:
    targets: Set[str] = set()
    start_map: Dict[str,int] = {}
    files = list(iter_author_jsonl_files(base_dir))
    if not files:
        raise RuntimeError(f"在 {base_dir} 下未找到符合条件的 .jsonl 文件")

    for cohort_year, fpath in tqdm(files, desc="load target authors (multi-jsonl)"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                aid = str(j.get("author_id") or "")
                if not aid:
                    continue
                sy = j.get("start_year")
                if sy is None:
                    # 兜底：若没写 start_year，尝试由 year/career_year 推回（如果行内有）
                    y  = to_year(j.get("year"))
                    cy = to_year(j.get("career_year"))
                    if y is not None and cy is not None:
                        sy = y - cy + 1
                sy = to_year(sy) if sy is not None else cohort_year
                if sy is None:
                    continue
                if TARGET_START_MIN <= sy <= TARGET_START_MAX:
                    targets.add(aid)
                    if (aid not in start_map) or (sy < start_map[aid]):
                        start_map[aid] = sy
    return targets, start_map

# ---------- 遍历 DBLP 年度切片 ----------
def iter_dblp_year_files(dir_path: str, y_start: int, y_end: int) -> Iterable[Tuple[int,str]]:
    files = []
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(".jsonl"):
            continue
        y = year_from_text(fn)
        if y is None:
            continue
        if y_start <= y <= y_end:
            files.append((y, os.path.join(dir_path, fn)))
    files.sort(key=lambda x: x[0])
    for y, p in files:
        yield y, p

def add_pairs_TA_anchored(aids: List[str], targets: Set[str], year_counter: Counter):
    """只计至少一端为目标作者（TA）的配对（TA<->TA 与 TA<->N1；不计 N1<->N1）。"""
    if not aids: return
    s = set(aids)
    tas = s & targets
    if not tas: return
    others = s - tas
    # TA <-> others
    for ta in tas:
        for o in others:
            u, v = (ta, o) if ta < o else (o, ta)
            year_counter[(u, v)] += 1
    # TA <-> TA
    if len(tas) >= 2:
        for (u, v) in combinations(sorted(tas), 2):
            year_counter[(u, v)] += 1

def prune_edges_topk(edges_df: pd.DataFrame, topk=30, min_weight=1) -> pd.DataFrame:
    """按每个端点取 Top-k 强边，并集；再应用最小权重过滤。"""
    if edges_df is None or edges_df.empty:
        # 直接返回一个空的规范列框架，避免后续 KeyError
        return pd.DataFrame(columns=["src","dst","weight"])

    # 先做强边过滤
    df = edges_df.loc[edges_df["weight"] >= min_weight, ["src","dst","weight"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["src","dst","weight"])

    # 为每个端点各取 Top-k
    buckets = []
    for col in ["src","dst"]:
        tmp = df[["src","dst","weight"]].copy()
        if col == "src":
            tmp = tmp.rename(columns={"src":"node","dst":"nbr"})
        else:
            tmp = tmp.rename(columns={"dst":"node","src":"nbr"})
        tmp = (tmp.sort_values(["node","weight"], ascending=[True, False])
                  .groupby("node", as_index=False)
                  .head(topk))
        buckets.append(tmp)

    keep = pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame(columns=["node","nbr","weight"])
    if keep.empty:
        return pd.DataFrame(columns=["src","dst","weight"])

    # 规范无向边顺序为 (u,v)，并只保留 u,v 两列用于与原 df 合并取权重
    keep["u"] = np.where(keep["node"] < keep["nbr"], keep["node"], keep["nbr"])
    keep["v"] = np.where(keep["node"] < keep["nbr"], keep["nbr"], keep["node"])
    keep_uv = keep[["u","v"]].drop_duplicates()

    # 在原 df 上也做 (u,v) 规范化，以便合并取唯一的 weight
    df["u"] = np.where(df["src"] < df["dst"], df["src"], df["dst"])
    df["v"] = np.where(df["src"] < df["dst"], df["dst"], df["src"])
    df_uv = df[["u","v","weight"]].drop_duplicates()

    out = keep_uv.merge(df_uv, on=["u","v"], how="left")
    out = out.rename(columns={"u":"src","v":"dst"})
    # 若合并后仍有缺失（极少数边在过滤链路中丢失），用 0 填充并再过滤一次
    out["weight"] = out["weight"].fillna(0).astype(int)
    out = out.loc[out["weight"] >= min_weight, ["src","dst","weight"]].drop_duplicates()
    return out


# ---------- 主流程 ----------
def main():
    # 1) 读取目标作者集合（1998–2009 入行）
    targets, start_map = load_targets_from_multi_dir(AUTHORS_MULTI_DIR)
    print(f"[INFO] 目标作者数（{TARGET_START_MIN}–{TARGET_START_MAX} 入行）：{len(targets):,}")

    out_dir = os.path.join(OUTPUT_DIR, "snapshots")
    os.makedirs(out_dir, exist_ok=True)

    yearly_pairs: Dict[int, Counter] = {}  # 窗口内的年计数
    cum_pairs = Counter()

    # 2) 遍历 DBLP 年度切片，累加“至少一端是 TA”的配对
    print(f"[INFO] 扫描 DBLP 年份：{INPUT_YEAR_START}..{INPUT_YEAR_END}；输出快照：{SNAPSHOT_START}..{SNAPSHOT_END}")
    for y, fpath in tqdm(list(iter_dblp_year_files(DBLP_YEAR_DIR, INPUT_YEAR_START, INPUT_YEAR_END)), desc="scan years"):
        year_counter = Counter()
        with open(fpath, "r", encoding="utf-8") as fr:
            for line in fr:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                ry = to_year(rec.get("year"))
                if ry is None or ry != y:
                    continue
                aobjs = authors_list_in_dblp(rec)
                if not aobjs:
                    continue
                aids = [author_id_from_obj(a) for a in aobjs]
                aids = [a for a in aids if a]
                if not aids:
                    continue
                add_pairs_TA_anchored(aids, targets, year_counter)

        yearly_pairs[y] = year_counter
        # 全量累计或滚动累计
        cum_pairs.update(year_counter)
        if WINDOW_YEARS > 0:
            drop_year = y - WINDOW_YEARS
            if drop_year in yearly_pairs:
                for pair, c in yearly_pairs[drop_year].items():
                    if c > 0:
                        cum_pairs[pair] -= c
                        if cum_pairs[pair] <= 0:
                            del cum_pairs[pair]
                del yearly_pairs[drop_year]

        # 3) 写出当年快照
        if SNAPSHOT_START <= y <= SNAPSHOT_END:
            if cum_pairs:
                edges = pd.DataFrame([(u,v,w) for (u,v), w in cum_pairs.items() if w > 0],
                                     columns=["src","dst","weight"])
            else:
                edges = pd.DataFrame(columns=["src","dst","weight"])

            # 规模裁剪
            edges = prune_edges_topk(edges, topk=TOPK_PER_NODE, min_weight=MIN_WEIGHT)

            # 节点表（当前快照的端点集合）
            if not edges.empty:
                nodes = pd.DataFrame({"node_id": pd.unique(pd.concat([edges["src"], edges["dst"]], ignore_index=True))})
            else:
                nodes = pd.DataFrame({"node_id": pd.Series(dtype=str)})
            nodes["is_target"] = nodes["node_id"].isin(targets).astype("int8")
            edges["year_abs"] = y
            nodes["year_abs"] = y

            # （可选）加入 N1<->N1 强边：建议后续再做增量脚本，这里先不展开
            # if INCLUDE_N1N1_STRONG: ...

            # 写盘
            write_parquet(edges, os.path.join(out_dir, f"edges_{y}.parquet"))
            write_parquet(nodes, os.path.join(out_dir, f"nodes_{y}.parquet"))

            nV = len(nodes); nE = len(edges)
            avg_deg = (2*nE/nV) if nV>0 else 0.0
            print(f"[SNAP {y}] |V|={nV:,} |E|={nE:,} avg_deg={avg_deg:.2f} "
                  f"(min_w>={MIN_WEIGHT}, topk={TOPK_PER_NODE}, window={WINDOW_YEARS or 'cum'})")

    print("✅ 完成：累计合著图快照已写出 →", out_dir)

if __name__ == "__main__":
    main()
