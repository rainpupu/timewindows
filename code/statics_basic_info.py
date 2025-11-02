#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖：pandas numpy tqdm（pip install pandas numpy tqdm）
建议：Python 3.9+，SSD 磁盘

目录约定：
 full_info_of_author/
   1998_from_slice/author_papers.jsonl
   1999_from_slice/author_papers.jsonl
   ...
   2012_from_slice/author_papers.jsonl

切片（可选，用于精确团队规模/合作者结构）：
 dblp_out/slice_1998_2022_nopreprint/DBLP_1998_2022_no_preprint.jsonl
"""

import os, re, glob, json, math, sqlite3
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# ===================== 配置区（根据需要修改） =====================

ROOT_DIR   = r"full_info_of_author"   # 你的 author_papers.jsonl 根目录
YEAR_MIN   = 1998
YEAR_MAX   = 2009

OUT_DIR    = r"metrics_1998_2009"               # 统计输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 是否使用 DBLP 切片做“精确团队规模/合作者结构”（包含论文里的所有作者，满足你的要求）
USE_SLICE  = True
SLICE_PATH = r"dblp_out\slice_1998_2022_nopreprint\DBLP_1998_2022_no_preprint.jsonl"

# 是否计算“合作者结构”（新增/累计唯一合作者、回头合作者比例）
# 注：开启需要一定内存与时间；若机器吃紧可设为 False
CALC_COLLAB = True

# 生存曲线的分母来源：
# False = 用“样本内有论文作者人数”（满足“没有论文的不算”的要求）【默认】
# True  = 用外部各届清单（提供 start_year_*.jsonl 目录）
USE_EXTERNAL_COHORT_DENOM = False
START_YEAR_DIR = r"cohorts\starts"  # 当 USE_EXTERNAL_COHORT_DENOM=True 时有效

# pandas 分块大小（按行），内存吃紧可适当调小
CHUNK = 400_000

# ================================================================


def pct(n, d):
    return (n / d) if d else 0.0

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_author_files(root, year_min, year_max):
    patt = os.path.join(root, "*_from_slice", "author_papers.jsonl")
    files = sorted(glob.glob(patt))
    out = []
    for fp in files:
        dname = os.path.basename(os.path.dirname(fp))
        m = re.search(r"(\d{4})", dname)
        if not m:
            continue
        y = int(m.group(1))
        if year_min <= y <= year_max:
            out.append(fp)
    return out

def build_sqlite_from_slice(slice_path, sample_pids, sqlite_path, need_authors=True, batch_commit=10000):
    """把切片里用到的 paper_id 建索引：pid -> team_size, authors(JSON)"""
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE t(pid TEXT PRIMARY KEY, team_size INTEGER, authors TEXT)")
    conn.commit()

    keep = set(sample_pids)
    wrote = 0
    with open(slice_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="build sqlite (from slice)", unit="line"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except:
                continue
            pid = rec.get("id") or rec.get("v12_id") or rec.get("paper_id")
            if pid not in keep:
                continue
            auths = rec.get("authors") or rec.get("v12_authors") or []
            if isinstance(auths, dict):
                auths = [auths]
            alist = []
            for a in auths:
                if not isinstance(a, dict):
                    continue
                aid = a.get("id") or a.get("author_id")
                if aid is None:
                    ids = a.get("ids")
                    if isinstance(ids, list) and ids:
                        aid = ids[0]
                if aid is not None:
                    alist.append(str(aid))
            team_size = len(alist)
            js = json.dumps(alist, ensure_ascii=False) if need_authors else None
            cur.execute("INSERT OR REPLACE INTO t(pid, team_size, authors) VALUES (?, ?, ?)",
                        (pid, team_size, js))
            wrote += 1
            if wrote % batch_commit == 0:
                conn.commit()
    conn.commit()
    cur.close()
    conn.close()
    return sqlite_path

def fetch_team_and_authors(conn, pid_list, need_authors=False, chunk=800):
    """批量查询 pid → team_size（和 authors，可选）"""
    team_map = {}
    authors_map = {} if need_authors else None
    cur = conn.cursor()
    for i in range(0, len(pid_list), chunk):
        sub = pid_list[i:i+chunk]
        q = "SELECT pid, team_size" + (", authors" if need_authors else "") + \
            " FROM t WHERE pid IN (%s)" % ",".join(["?"]*len(sub))
        cur.execute(q, sub)
        rows = cur.fetchall()
        for row in rows:
            if need_authors:
                pid, ts, js = row
                team_map[pid] = int(ts)
                try:
                    authors_map[pid] = json.loads(js) if js else []
                except:
                    authors_map[pid] = []
            else:
                pid, ts = row
                team_map[pid] = int(ts)
    cur.close()
    return (team_map, authors_map)

def main():
    # ---------- 收集输入 ----------
    files = find_author_files(ROOT_DIR, YEAR_MIN, YEAR_MAX)
    if not files:
        raise SystemExit(f"未在 {ROOT_DIR} 下找到 {YEAR_MIN}-{YEAR_MAX} 的 author_papers.jsonl")
    print(f"[INFO] 发现 {len(files)} 个 cohort 文件：")
    for fp in files:
        print(" -", fp)

    # ---------- 第一阶段：样本聚合（只含“有论文的作者”） ----------
    total_rows = 0

    # unique paper 维度
    paper_seen = set()
    papers_by_year = Counter()
    paper_doi_have = Counter()
    paper_missing_title = 0
    paper_missing_venue = 0
    paper_missing_year  = 0
    paper_row_counts = Counter()
    sample_pids = set()

    # 作者维度
    authors_with_any = set()
    # 用于“每年有论文的新入行作者数”：记录每个作者的 start_year（全局唯一）
    author_start_map = {}

    # 首篇核查
    author_min_year = {}
    author_start_year_seen = {}

    # career_year / 角色
    ay_counts = Counter()
    first_by_k = Counter()
    last_by_k  = Counter()
    rows_by_k  = Counter()

    usecols = ["author_id","start_year","career_year","paper_id","title","venue_raw",
               "year","doi","is_first_author","is_last_author"]

    for fp in files:
        print(f"[LOAD] {fp}")
        for chunk in pd.read_json(fp, lines=True, chunksize=CHUNK, dtype=False):
            # 只保留必要列（read_json 没有 usecols 参数，这里再筛）
            df = chunk[[c for c in usecols if c in chunk.columns]].copy()

            # 规范类型
            if "year" in df:        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            if "start_year" in df:  df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").astype("Int64")
            if "career_year" in df: df["career_year"] = pd.to_numeric(df["career_year"], errors="coerce").astype("Int64")
            if "is_first_author" in df: df["is_first_author"] = df["is_first_author"].fillna(False).astype(bool)
            if "is_last_author"  in df: df["is_last_author"]  = df["is_last_author"].fillna(False).astype(bool)

            total_rows += len(df)

            # ---- unique paper 维度（首次遇到时统计缺失/年份/DOI）----
            if "paper_id" not in df:
                continue
            new_mask = ~df["paper_id"].isin(paper_seen)
            new_papers = df.loc[new_mask, ["paper_id","year","title","venue_raw","doi"]]
            for pid, y, ti, ve, doi in new_papers.itertuples(index=False):
                paper_seen.add(pid)
                sample_pids.add(pid)
                # 年份
                if pd.notna(y):
                    y = int(y)
                    papers_by_year[y] += 1
                    if isinstance(doi, str) and doi.strip():
                        paper_doi_have[y] += 1
                else:
                    paper_missing_year += 1
                # 缺失字段
                if not isinstance(ti, str) or not ti.strip():
                    paper_missing_title += 1
                if not isinstance(ve, str) or not ve.strip():
                    paper_missing_venue += 1

            # 行级重复（author×paper）
            vc = df["paper_id"].value_counts()
            for pid, cnt in vc.items():
                paper_row_counts[pid] += int(cnt)

            # 作者集合（样本内有论文作者）
            if "author_id" in df:
                authors_with_any.update(df["author_id"].astype(str).unique().tolist())

            # 记录每位作者的 start_year（用于“每年有论文的新入行作者数”）
            if "author_id" in df and "start_year" in df:
                dsy = df.dropna(subset=["author_id","start_year"]).groupby("author_id")["start_year"].min()
                for aid, sy in dsy.items():
                    aid = str(aid); sy = int(sy)
                    prev = author_start_map.get(aid)
                    if prev is None or sy < prev:
                        author_start_map[aid] = sy

            # 作者首篇核查：样本窗口内的最早年份 vs start_year
            if "author_id" in df and "year" in df:
                dmin = df.groupby("author_id")["year"].min()
                for aid, y0 in dmin.items():
                    if pd.isna(y0):
                        continue
                    aid = str(aid); y0 = int(y0)
                    old = author_min_year.get(aid)
                    if old is None or y0 < old:
                        author_min_year[aid] = y0
            if "author_id" in df and "start_year" in df:
                dmode = df.groupby("author_id")["start_year"].agg(lambda s: int(pd.Series(s.dropna()).mode().iloc[0]) if s.dropna().size else None)
                for aid, sy in dmode.items():
                    if sy is None:
                        continue
                    aid = str(aid); sy = int(sy)
                    old = author_start_year_seen.get(aid)
                    author_start_year_seen[aid] = min(old, sy) if old is not None else sy

            # 生涯年计数与角色
            if all(c in df for c in ["author_id","start_year","career_year"]):
                g = df.groupby(["author_id","start_year","career_year"]).size()
                for (aid, sy, k), c in g.items():
                    if pd.isna(sy) or pd.isna(k):
                        continue
                    ay_counts[(str(aid), int(sy), int(k))] += int(c)
            if "career_year" in df and "paper_id" in df:
                gk = df.groupby("career_year").agg(
                    rows=("paper_id","size"),
                    first=("is_first_author","sum") if "is_first_author" in df else ("paper_id","size"),
                    last=("is_last_author","sum") if "is_last_author" in df else ("paper_id","size")
                )
                # 若缺 is_first/last，则 first/last 会等于行数（比例=1.0），通常不会发生
                for k, row in gk.iterrows():
                    if pd.isna(k):
                        continue
                    k = int(k)
                    rows_by_k[k]  += int(row["rows"])
                    # 兼容：sum 结果若不是布尔求和，最多回退 0
                    first_by_k[k] += int(row.get("first", 0))
                    last_by_k[k]  += int(row.get("last", 0))

    print("\n[阶段1] 样本聚合完成。")

    # ---------- 质量/覆盖 ----------
    unique_papers = len(paper_seen)
    dup_rows = sum(cnt-1 for cnt in paper_row_counts.values() if cnt>0)
    dup_ratio_rows = pct(dup_rows, total_rows)
    dup_papers = sum(1 for pid,cnt in paper_row_counts.items() if cnt>1)
    dup_ratio_papers = pct(dup_papers, unique_papers)

    miss_title_ratio = pct(paper_missing_title, unique_papers)
    miss_venue_ratio = pct(paper_missing_venue, unique_papers)
    miss_year_ratio  = pct(paper_missing_year , unique_papers)

    doi_total_have = sum(paper_doi_have.values())
    doi_total_ratio = pct(doi_total_have, unique_papers)

    per_year = [{"year": y, "papers": int(n)} for y,n in sorted(papers_by_year.items())]
    doi_by_year = [{"year": y, "papers": int(papers_by_year[y]),
                    "doi_have": int(paper_doi_have[y]),
                    "doi_ratio": pct(paper_doi_have[y], papers_by_year[y])}
                   for y in sorted(papers_by_year.keys())]

    # 作者首篇核查
    chk_total=0; chk_match=0; mismatch=[]
    for aid, y0 in author_min_year.items():
        sy = author_start_year_seen.get(aid)
        if sy is None:
            continue
        chk_total += 1
        if y0 == sy:
            chk_match += 1
        else:
            if len(mismatch) < 50:
                mismatch.append({"author_id": aid, "start_year": int(sy), "min_year_in_sample": int(y0)})

    quality = {
      "规模与范围": {
        "样本内有论文作者总数": len(authors_with_any),
        "论文总数_unique": unique_papers,
        "年份范围": [min(papers_by_year) if papers_by_year else None,
                   max(papers_by_year) if papers_by_year else None],
        "每年论文数": per_year,
        "每年有论文的新入行作者数_start_year分布": [
            {"start_year": y, "authors": int(n)}
            for y, n in sorted(Counter(author_start_map.values()).items())
        ]
      },
      "缺失与一致性": {
        "title缺失比例": miss_title_ratio,
        "venue缺失比例": miss_venue_ratio,
        "year缺失比例":  miss_year_ratio,
        "重复paper_id占比_按行": dup_ratio_rows,
        "重复paper占比_按唯一paper": dup_ratio_papers,
        "DOI覆盖_整体": doi_total_ratio,
        "DOI覆盖_分年": doi_by_year
      },
      "结构合理性": {
        "作者首篇是否等于start_year_核查": {
          "checked_authors": chk_total,
          "match": chk_match,
          "match_ratio": pct(chk_match, chk_total),
          "examples_mismatch": mismatch
        },
        "团队规模与单作者比例_说明": "未启用 USE_SLICE：下界近似（仅样本内）。如需精确请设 USE_SLICE=True。"
      }
    }

    # ---------- 队列/生存/产出 ----------
    rows_ay = [ (aid,int(sy),int(k),int(c)) for (aid,sy,k),c in ay_counts.items() ]
    df_ay = pd.DataFrame(rows_ay, columns=["author_id","start_year","career_year","paper_count"]) if rows_ay else pd.DataFrame(columns=["author_id","start_year","career_year","paper_count"])

    # 分母：默认 = 样本内“有论文作者”人数（满足“没有论文的不算”）
    use_external_denom = USE_EXTERNAL_COHORT_DENOM and os.path.isdir(START_YEAR_DIR)
    cohort_denominator = {}

    if use_external_denom:
        for y in range(YEAR_MIN, YEAR_MAX + 1):
            p = os.path.join(START_YEAR_DIR, f"start_year_{y}.jsonl")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    c = sum(1 for line in f if line.strip())
                cohort_denominator[y] = c

        if not cohort_denominator:
            print("[WARN] 未从 START_YEAR_DIR 取得分母，改用样本内“有论文作者”人数。")
            use_external_denom = False

    if not use_external_denom:
        # 由样本内“有论文作者”的 start_year 构建分母
        cnt = Counter(int(sy) for sy in author_start_map.values() if sy is not None)
        cohort_denominator = {int(y): int(n) for y, n in cnt.items()}

    # 队列规模与占比（分母口径见上）
    total_authors_sample = sum(cohort_denominator.values()) or 1
    cohort_size_rows = []
    for y in sorted(cohort_denominator.keys()):
        n = cohort_denominator[y]
        cohort_size_rows.append({"start_year": y, "authors_total": n, "ratio": pct(n, total_authors_sample)})
    pd.DataFrame(cohort_size_rows).to_csv(os.path.join(OUT_DIR, "cohort_size.csv"), index=False)

    # 生存曲线 k=1..11
    surv = []
    for y, dn in sorted(cohort_denominator.items()):
        if df_ay.empty:
            continue
        dfc = df_ay[df_ay["start_year"]==y]
        for k in range(1,12):
            active = int(dfc[dfc["career_year"]==k]["author_id"].nunique())
            surv.append({"start_year": y, "career_year": k,
                         "active_authors": active, "denominator": int(dn),
                         "survival_ratio": pct(active, dn)})
    pd.DataFrame(surv).to_csv(os.path.join(OUT_DIR, "survival_curve.csv"), index=False)

    # 入行后前 W 年至少 1 篇
    W_list = [0,1,2,3,4,5,6,7,8,9,10]
    earlys=[]
    for y, dn in sorted(cohort_denominator.items()):
        if df_ay.empty:
            continue
        dfc = df_ay[df_ay["start_year"]==y]
        for W in W_list:
            act = dfc[dfc["career_year"]<=W].groupby("author_id")["paper_count"].sum()
            active = int((act>0).sum())
            earlys.append({"start_year": y, "W": W, "active_authors": active, "denominator": int(dn), "ratio": pct(active,int(dn))})
    pd.DataFrame(earlys).to_csv(os.path.join(OUT_DIR, "early_activity_W.csv"), index=False)

    # 产出强度
    prods=[]
    if not df_ay.empty:
        for y in sorted(df_ay["start_year"].dropna().unique()):
            dfc = df_ay[df_ay["start_year"]==y]
            for k in range(1,12):
                s = dfc[dfc["career_year"]==k]["paper_count"]
                if s.empty:
                    prods.append({"start_year": y, "career_year": k, "mean":0,"median":0,"p10":0,"p90":0})
                else:
                    prods.append({"start_year": y, "career_year": k,
                                  "mean": float(s.mean()),
                                  "median": float(s.median()),
                                  "p10": float(np.percentile(s,10)),
                                  "p90": float(np.percentile(s,90))})
    pd.DataFrame(prods).to_csv(os.path.join(OUT_DIR, "productivity_stats.csv"), index=False)

    # 高产比例（11 年累计）
    his=[]
    if not df_ay.empty:
        for y in sorted(df_ay["start_year"].dropna().unique()):
            tot = df_ay[df_ay["start_year"]==y].groupby("author_id")["paper_count"].sum()
            if tot.empty:
                continue
            q90 = float(np.percentile(tot, 90))
            q99 = float(np.percentile(tot, 99))
            n = tot.shape[0]
            n90 = int((tot >= q90).sum())
            n99 = int((tot >= q99).sum())
            his.append({"start_year": y, "authors": int(n),
                        "p90_threshold": q90, "top10_count": n90, "top10_ratio": pct(n90, n),
                        "p99_threshold": q99, "top1_count": n99, "top1_ratio": pct(n99, n)})
    pd.DataFrame(his).to_csv(os.path.join(OUT_DIR, "high_producers.csv"), index=False)

    # career_year 分布（整体）
    pd.DataFrame(rows_ay, columns=["author_id","start_year","career_year","paper_count"]).to_csv(
        os.path.join(OUT_DIR, "author_year_counts.csv"), index=False)
    summ=[]
    if not df_ay.empty:
        for k, s in df_ay.groupby("career_year")["paper_count"]:
            if s.empty:
                continue
            summ.append({"career_year": int(k),
                         "mean": float(s.mean()),
                         "median": float(s.median()),
                         "p10": float(np.percentile(s,10)),
                         "p90": float(np.percentile(s,90))})
    pd.DataFrame(summ).to_csv(os.path.join(OUT_DIR, "author_year_summary.csv"), index=False)

    # 角色（基础）
    role_rows=[]
    for k in sorted(rows_by_k.keys()):
        rows_k = rows_by_k[k]
        role_rows.append({"career_year": k, "rows": int(rows_k),
                          "first_ratio": pct(first_by_k[k], rows_k),
                          "last_ratio":  pct(last_by_k[k], rows_k)})
    pd.DataFrame(role_rows).to_csv(os.path.join(OUT_DIR, "role_team_stats.csv"), index=False)

    # ---------- 结构合理性：团队规模/单作者比例（若开启 USE_SLICE） ----------
    if USE_SLICE and os.path.exists(SLICE_PATH):
        print(f"[SLICE] 使用切片精确计算团队规模/单作者比例/合作结构：{SLICE_PATH}")
        sqlite_path = os.path.join(OUT_DIR, "pid_coauthors.sqlite")
        # 建索引（包含 authors 列：供合作结构使用）
        build_sqlite_from_slice(SLICE_PATH, list(sample_pids), sqlite_path, need_authors=True)
        conn = sqlite3.connect(sqlite_path)
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")

        # —— (1) 团队规模均值/单作者比例（按 career_year）
        rows_by_k_exact = Counter()
        team_sum_by_k_exact = Counter()
        single_by_k_exact  = Counter()

        # 也计算“每篇论文作者数分布”的总体（均值/中位/分位），用于质量报告
        team_sizes_all = []

        for fp in files:
            for chunk in pd.read_json(fp, lines=True, chunksize=CHUNK, dtype=False):
                if not all(c in chunk.columns for c in ["paper_id","career_year"]):
                    continue
                sub = chunk[["paper_id","career_year"]].dropna()
                sub["career_year"] = pd.to_numeric(sub["career_year"], errors="coerce").astype("Int64")
                pids = sub["paper_id"].astype(str).unique().tolist()
                team_map, _ = fetch_team_and_authors(conn, pids, need_authors=False)
                # 回填团队规模
                sub["team_size"] = sub["paper_id"].map(lambda x: team_map.get(str(x), np.nan))
                sub = sub.dropna(subset=["team_size"])
                sub["team_size"] = sub["team_size"].astype(int)
                team_sizes_all.extend(sub["team_size"].tolist())

                g = sub.groupby("career_year").agg(rows=("paper_id","size"),
                                                   team_sum=("team_size","sum"),
                                                   singles=("team_size", lambda s: int((s==1).sum())))
                for k, r in g.iterrows():
                    k = int(k)
                    rows_by_k_exact[k]     += int(r["rows"])
                    team_sum_by_k_exact[k] += int(r["team_sum"])
                    single_by_k_exact[k]   += int(r["singles"])

        # 回写 role_team_stats.csv 增补精确团队信息
        df_role = pd.read_csv(os.path.join(OUT_DIR, "role_team_stats.csv"))
        mean_team = []
        single_ratio = []
        for _, row in df_role.iterrows():
            k = int(row["career_year"])
            if rows_by_k_exact[k] > 0:
                mean_team.append(team_sum_by_k_exact[k] / rows_by_k_exact[k])
                single_ratio.append(pct(single_by_k_exact[k], rows_by_k_exact[k]))
            else:
                mean_team.append(None)
                single_ratio.append(None)
        df_role["mean_team_size"]    = mean_team
        df_role["single_author_ratio"]= single_ratio
        df_role.to_csv(os.path.join(OUT_DIR, "role_team_stats.csv"), index=False)

        # 更新质量报告中的团队规模分布（总体）
        if team_sizes_all:
            team_sizes_all = np.array(team_sizes_all, dtype=np.int64)
            team_dist = {
                "mean": float(team_sizes_all.mean()),
                "median": float(np.median(team_sizes_all)),
                "p10": float(np.percentile(team_sizes_all, 10)),
                "p90": float(np.percentile(team_sizes_all, 90)),
                "single_author_ratio": float((team_sizes_all==1).sum() / len(team_sizes_all))
            }
            quality["结构合理性"]["每篇论文作者数分布_总体"] = team_dist
            quality["结构合理性"]["团队规模与单作者比例_说明"] = "已使用切片精确计算（包含论文中的全部作者）。"
        else:
            quality["结构合理性"]["每篇论文作者数分布_总体"] = {}

        # —— (2) 合作结构（新增/累计唯一合作者、回头合作者比例）
        note_path = os.path.join(OUT_DIR, "collaboration_note.txt")
        if CALC_COLLAB:
            # 为降低内存：按块取需要的 paper 的 authors 列，逐行累积到 author→year 的集合
            # 说明：这里把“论文里的所有作者”都纳入（不限制 1998–2012 入行）
            author_year_co = defaultdict(lambda: defaultdict(set))  # aid -> {k: set(coauthors)}
            for fp in files:
                for chunk in pd.read_json(fp, lines=True, chunksize=min(CHUNK, 200_000), dtype=False):
                    if not all(c in chunk.columns for c in ["author_id","career_year","paper_id"]):
                        continue
                    sub = chunk[["author_id","career_year","paper_id"]].dropna()
                    sub["career_year"] = pd.to_numeric(sub["career_year"], errors="coerce").astype("Int64")
                    sub = sub.dropna(subset=["career_year"])
                    pids = sub["paper_id"].astype(str).unique().tolist()
                    _, auth_map = fetch_team_and_authors(conn, pids, need_authors=True)
                    for aid, k, pid in sub.itertuples(index=False):
                        aid = str(aid); k = int(k); pid = str(pid)
                        alist = auth_map.get(pid) or []
                        co = set(alist)
                        if aid in co:
                            co.remove(aid)
                        if co:
                            author_year_co[aid][k].update(co)

            # 聚合：按 career_year 计算新增/累计唯一合作者，回头合作比例
            recs=[]
            seen_cache = {}  # aid -> set()（累计唯一）
            for aid, per_k in author_year_co.items():
                seen = seen_cache.get(aid)
                if seen is None:
                    seen=set()
                    seen_cache[aid]=seen
                for k in sorted(per_k.keys()):
                    cur = per_k[k]
                    new = cur - seen
                    rep = len(cur & seen)
                    allp= len(cur)
                    seen.update(cur)
                    recs.append((k, len(new), len(seen), rep, allp))
            if recs:
                dfc = pd.DataFrame(recs, columns=["career_year","new_co","cum_unique_co","repeat_pairs","all_pairs"])
                agg = dfc.groupby("career_year").agg(
                    authors=("new_co","size"),
                    new_co_mean=("new_co","mean"),
                    new_co_median=("new_co","median"),
                    cum_co_mean=("cum_unique_co","mean"),
                    cum_co_median=("cum_unique_co","median"),
                    repeat_pairs_sum=("repeat_pairs","sum"),
                    all_pairs_sum=("all_pairs","sum")
                )
                agg["repeat_ratio"] = agg["repeat_pairs_sum"] / agg["all_pairs_sum"].replace({0:np.nan})
                out = agg.reset_index()[["career_year","authors","new_co_mean","new_co_median",
                                         "cum_co_mean","cum_co_median","repeat_ratio"]]
                out.to_csv(os.path.join(OUT_DIR, "collaboration_summary.csv"), index=False)
                with open(note_path, "w", encoding="utf-8") as f:
                    f.write("已使用切片：合作统计包含论文中的全部作者（包括非 1998–2012 入行者）。\n")
            else:
                with open(note_path, "w", encoding="utf-8") as f:
                    f.write("未命中任何合作者数据，或样本过小。\n")
        else:
            with open(os.path.join(OUT_DIR, "collaboration_note.txt"), "w", encoding="utf-8") as f:
                f.write("CALC_COLLAB=False：跳过合作者结构统计。\n")

        conn.close()
    else:
        # 未使用切片时的说明
        with open(os.path.join(OUT_DIR, "collaboration_note.txt"), "w", encoding="utf-8") as f:
            f.write("未启用 USE_SLICE 或找不到切片：团队/合作仅为样本内近似；合作结构跳过。\n")

    # 最终落地质量报告
    write_json(os.path.join(OUT_DIR, "quality_overview.json"), quality)

    # README
    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
"""本目录为统计结果导出：
- quality_overview.json：数据质量/覆盖/一致性/团队规模说明（若启用切片则精确）
- cohort_size.csv：各届样本内“有论文作者”人数与占比（默认分母）；如需全体入行分母，请改用外部清单
- survival_curve.csv：生存曲线（k=1..11）
- early_activity_W.csv：前W年（1/2/3/5/7/10）至少一篇
- productivity_stats.csv：各届在第k年的产出强度（均值/中位/分位）
- high_producers.csv：各届11年累计的高产阈值与占比（前10%、前1%）
- author_year_counts.csv：author×career_year 的篇数明细
- author_year_summary.csv：按 career_year 的整体分布（均值/中位/分位）
- role_team_stats.csv：第k年的首/末作者占比；若启用切片，含精确“人均作者数/单作者比例”
- collaboration_summary.csv（启用切片且 CALC_COLLAB=True）：新增/累计合作者、回头合作比例
- collaboration_note.txt：关于是否使用切片/是否计算合作结构的说明
"""
        )

    print(f"\n[OK] 全部统计完成，输出目录：{OUT_DIR}")
    print("建议先查看：quality_overview.json、cohort_size.csv、survival_curve.csv、productivity_stats.csv")

if __name__ == "__main__":
    if USE_SLICE and not os.path.exists(SLICE_PATH):
        print(f"[WARN] 未找到切片文件：{SLICE_PATH}，将 USE_SLICE=False 继续。")
        USE_SLICE = False
    main()

