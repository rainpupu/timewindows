#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, re, sys
from collections import defaultdict
from decimal import Decimal
from tqdm.auto import tqdm

# ============ 可选：用于写 parquet ============
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = pq = None
# ============================================

try:
    import orjson
    USE_ORJSON = True
except Exception:
    USE_ORJSON = False

# ======== 配置（按需修改）========
SLICE_JSONL = os.path.join("dblp_out", "slice_1998_2022_nopreprint", "DBLP_1998_2022_no_preprint.jsonl")
START_FILE  = os.path.join("dblp_out", "exports", "by_start_year", "start_year_1998.jsonl")  # 入行=1998 的作者清单

OUT_DIR   = os.path.join("full_info_of_authors_new", "1998_from_slice")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_ROWS  = os.path.join(OUT_DIR, "author_papers.jsonl")
OUT_NOHIT = os.path.join(OUT_DIR, "authors_no_papers.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "stats.json")

# 年份与 cohort 口径
COHORT_START_YEAR = 1998
YEAR_START, YEAR_END = 1998, 2007  # 绝对年窗口（包含端点）

# 输出年度特征
OUT_FEATURES_PARQUET = os.path.join(OUT_DIR, "author_year_features.parquet")
OUT_FEATURES_CSV     = os.path.join(OUT_DIR, "author_year_features.csv")
OUT_SAMPLE_JSONL     = os.path.join(OUT_DIR, "author_year_features_sample.jsonl")
SAMPLE_AUTHORS       = 150
# =================================

# ===== 序列化工具 =====
def _default(o):
    if isinstance(o, Decimal):
        try:
            return float(o)
        except Exception:
            return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError

def dumps(obj) -> str:
    if USE_ORJSON:
        return orjson.dumps(obj, default=_default, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    else:
        return json.dumps(obj, ensure_ascii=False)

# ===== 基础工具 =====
def to_int_year(y):
    if y is None or isinstance(y, bool): return None
    if isinstance(y, Decimal):
        try: return int(y)
        except Exception: return None
    try:
        return int(y)
    except Exception:
        try:
            s = str(y).strip()
            return int(s) if s and re.fullmatch(r"-?\d+", s) else None
        except Exception:
            return None

def authors_list(rec):
    """
    兼容多种结构：
    - {"authors": [...]}
    - {"authors": {"author": [...]}}
    - {"v12_authors": [...]}
    - 单个 dict 时转为单元素列表
    """
    a = rec.get("authors") or rec.get("v12_authors") or []
    if isinstance(a, dict) and "author" in a:
        a = a.get("author") or []
    if isinstance(a, dict):
        a = [a]
    return a if isinstance(a, list) else []

def author_id_from(author_obj):
    """优先取作者id；若无则 None（不回退 name 以免与 start 名单不一致）"""
    if not isinstance(author_obj, dict): return None
    cand = author_obj.get("id") or author_obj.get("author_id")
    if cand is None:
        ids = author_obj.get("ids")
        if isinstance(ids, list) and ids:
            cand = ids[0]
    return str(cand) if cand is not None else None

def author_name_from(author_obj):
    if not isinstance(author_obj, dict): return None
    return author_obj.get("name")

def author_org_from(author_obj):
    if not isinstance(author_obj, dict): return None
    return author_obj.get("org") or author_obj.get("affiliation") or None

def venue_raw_from(rec):
    """
    提取刊源名称：
    - 优先 venue.raw / venue.name
    - 其次 journal / booktitle
    - 兜底 'venue.raw'（扁平字段名）
    """
    v = rec.get("venue")
    if isinstance(v, dict):
        cand = v.get("raw") or v.get("name") or v.get("raw_venue")
        if cand: return cand
    if isinstance(v, str):
        return v
    return rec.get("venue.raw") or rec.get("journal") or rec.get("booktitle")

def norm_venue(s: str) -> str:
    if not s: return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def paper_id_from(rec):
    return rec.get("id") or rec.get("v12_id") or rec.get("paper_id")

# ===== 载入“入行年份映射” =====
def load_start_authors(path):
    start_map = {}  # author_id(str) -> start_year(int)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                aid = str(j["author_id"])
                sy = to_int_year(j.get("start_year"))
                if aid and sy:
                    start_map[aid] = sy
            except Exception:
                continue
    return start_map

def main():
    start_map = load_start_authors(START_FILE)
    # 只保留 cohort=1998 的作者作为统计主体
    target_authors = {aid for aid, sy in start_map.items() if sy == COHORT_START_YEAR}

    total_authors = len(target_authors)
    hit_authors = set()

    counters = defaultdict(int)

    # ========= 1) 行级输出（author_papers.jsonl） =========
    fw_rows = open(OUT_ROWS, "w", encoding="utf-8")

    # ========= 2) 年度特征累加器（按 author × year）=========
    # 计数/求和
    n_papers = defaultdict(int)
    first_sum = defaultdict(int)       # 一作个数
    first_den = defaultdict(int)       # 一作分母（有论文的篇数；我们的一作永远是0/1）
    team_sum  = defaultdict(int)       # team_size 总和
    team_den  = defaultdict(int)       # team_size 分母（篇数）
    single_sum = defaultdict(int)      # 单作者论文个数
    venues_set = defaultdict(set)      # 当年去重 venue
    coauthors_year = defaultdict(set)  # 当年唯一合作者（基于完整作者表）

    # 首次合作最早年份（窗口内）：(author, coauthor) -> first_year
    first_collab = {}                  # 用 dict 保存 min(year)

    # ========== 扫描 DBLP 切片 ==========
    with open(SLICE_JSONL, "r", encoding="utf-8") as fr:
        pbar = tqdm(fr, desc="scan slice (ndjson)", unit="line")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counters["skip_bad_json"] += 1
                continue

            year = to_int_year(rec.get("year"))
            if year is None or year < YEAR_START or year > YEAR_END:
                continue

            auths = authors_list(rec)
            if not auths:
                counters["skip_no_authors"] += 1
                continue

            paper_id = paper_id_from(rec)
            if not paper_id:
                counters["skip_no_paper_id"] += 1
                continue

            venue_raw = venue_raw_from(rec)
            venue_key = norm_venue(venue_raw)
            author_count = len(auths)

            # 预取整篇论文作者的 (id, name)，便于自环过滤
            paper_ids  = [author_id_from(a) for a in auths]
            paper_names= [author_name_from(a) for a in auths]

            # 遍历论文作者；对 cohort 作者进行统计与写出
            for idx, aobj in enumerate(auths):
                aid = author_id_from(aobj)
                if not aid or aid not in target_authors:
                    continue

                sy = start_map.get(aid)
                if sy != COHORT_START_YEAR:
                    continue

                # 行级输出：一作判定（首位或单作者）
                is_first = 1 if (idx == 0 or author_count == 1) else 0

                row = {
                    "author_id": aid,
                    "start_year": sy,
                    "career_year": year - sy + 1,
                    "paper_id": paper_id,
                    "title": rec.get("title"),
                    "doc_type": rec.get("doc_type") or rec.get("docType") or rec.get("type"),
                    "venue_raw": venue_raw,
                    "author_org": author_org_from(aobj),
                    "year": year,
                    "n_citation": rec.get("n_citation") or rec.get("nCitation"),
                    "doi": rec.get("doi"),
                    "keywords": rec.get("keywords") or [],
                    "fos.name": rec.get("fos.name") or [],
                    "fos.w": rec.get("fos.w") or [],
                    "references": rec.get("references") or [],
                    "is_first_author": is_first,
                }
                fw_rows.write(dumps(row) + "\n")

                hit_authors.add(aid)

                # 年度特征累加
                key = (aid, year)
                n_papers[key] += 1
                first_sum[key] += is_first
                first_den[key] += 1

                team_sum[key]  += author_count
                team_den[key]  += 1
                single_sum[key]+= 1 if author_count == 1 else 0
                if venue_key:
                    venues_set[key].add(venue_key)

                # 合作：当年唯一合作者 & 首次合作年
                # 用 (co_id or co_name) 作为 coauthor 键，排除自环（id 或 name 同人）
                my_name = author_name_from(aobj)
                for j, (cid, cname) in enumerate(zip(paper_ids, paper_names)):
                    if j == idx:
                        continue
                    co_key = cid or cname
                    if not co_key:
                        continue
                    if co_key == aid or (my_name and co_key == my_name):
                        continue
                    coauthors_year[key].add(co_key)

                    pair = (aid, co_key)
                    fy = first_collab.get(pair)
                    first_collab[pair] = year if fy is None else (year if year < fy else fy)

        pbar.close()

    fw_rows.close()

    # 没命中的作者（cohort 但窗口内0论文）
    nohit_authors = [aid for aid in target_authors if aid not in hit_authors]
    with open(OUT_NOHIT, "w", encoding="utf-8") as f:
        for aid in nohit_authors:
            f.write(dumps({"author_id": aid, "start_year": COHORT_START_YEAR}) + "\n")

    # ======== 汇总：构造每人每年的特征行 ========
    years = list(range(YEAR_START, YEAR_END + 1))

    # 预先把“首次合作年”按作者聚合成 histogram，便于 O(年数) 得到 new/cum
    first_by_author = defaultdict(lambda: defaultdict(int))  # aid -> {year: count_new}
    for (aid, co_key), fy in first_collab.items():
        if fy is not None and YEAR_START <= fy <= YEAR_END:
            first_by_author[aid][fy] += 1

    out_rows = []
    for aid in tqdm(sorted(target_authors), desc="assemble features (author × year)"):
        cum_papers = 0
        # 为累计唯一合作者准备 running sum
        new_hist = first_by_author.get(aid, {})
        cum_unique = 0
        for y in years:
            key = (aid, y)
            npap = n_papers.get(key, 0)
            cum_papers += npap

            # 一作占比
            fs, fd = first_sum.get(key, 0), first_den.get(key, 0)
            first_share = (fs / fd) if fd > 0 else None

            # 平均团队规模 & 单作者占比
            ts, td = team_sum.get(key, 0), team_den.get(key, 0)
            avg_team = (ts / td) if td > 0 else None
            single_share = (single_sum.get(key, 0) / npap) if npap > 0 else None

            # 渠道
            venues_year = len(venues_set.get(key, set()))

            # 合作
            uniq_co = len(coauthors_year.get(key, set()))
            new_co  = new_hist.get(y, 0)
            cum_unique += new_co
            repeat_ratio = ((uniq_co - new_co) / uniq_co) if uniq_co > 0 else None

            out_rows.append({
                "author_id": aid,
                "year": y,
                "n_papers": npap,
                "n_papers_cum": cum_papers,
                "first_author_share": first_share,
                "avg_team_size": avg_team,
                "single_author_share": single_share,
                "venues_dedup_year": venues_year,
                "unique_coauthors_year": uniq_co,
                "new_coauthors_year": new_co,
                "cum_unique_coauthors": cum_unique,
                "repeat_collab_ratio": repeat_ratio,
            })

    # ======== 写出年度特征 ========
    wrote_parquet = False
    if pd is not None:
        df = pd.DataFrame(out_rows)
        if pq is not None:
            try:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, OUT_FEATURES_PARQUET)
                wrote_parquet = True
                print(f"[OK] 写出 Parquet: {OUT_FEATURES_PARQUET}")
            except Exception as e:
                print(f"[WARN] 写 Parquet 失败，将写 CSV。原因: {e}", file=sys.stderr)
        if not wrote_parquet:
            df.to_csv(OUT_FEATURES_CSV, index=False, encoding="utf-8")
            print(f"[OK] 写出 CSV 兜底: {OUT_FEATURES_CSV}")
    else:
        # 没有 pandas：写 JSONL 作为兜底
        with open(OUT_FEATURES_CSV, "w", encoding="utf-8") as f:
            f.write("author_id,year,n_papers,n_papers_cum,first_author_share,avg_team_size,single_author_share,venues_dedup_year,unique_coauthors_year,new_coauthors_year,cum_unique_coauthors,repeat_collab_ratio\n")
            for r in out_rows:
                # 简单 CSV 序列化（无逗号字段）
                f.write(",".join(str(r[k]) if r[k] is not None else "" for k in [
                    "author_id","year","n_papers","n_papers_cum","first_author_share","avg_team_size",
                    "single_author_share","venues_dedup_year","unique_coauthors_year","new_coauthors_year",
                    "cum_unique_coauthors","repeat_collab_ratio"
                ]) + "\n")
        print(f"[OK] 写出 CSV 兜底: {OUT_FEATURES_CSV}")

    # ======== 抽样 JSONL 预览 ========
    # 取出现过或未出现的 cohort 作者各自的记录（优先已出现）
    sample_ids = list(hit_authors)[:SAMPLE_AUTHORS]
    if len(sample_ids) < SAMPLE_AUTHORS:
        # 补足
        rest = [aid for aid in target_authors if aid not in hit_authors]
        sample_ids += rest[:(SAMPLE_AUTHORS - len(sample_ids))]

    with open(OUT_SAMPLE_JSONL, "w", encoding="utf-8") as fo:
        for aid in sample_ids:
            for y in range(YEAR_START, YEAR_END + 1):
                # 从 out_rows 里捞（也可直接重算一遍）
                # 为简单起见，直接过滤字典（规模可接受）
                recs = [r for r in out_rows if r["author_id"] == aid and r["year"] == y]
                if recs:
                    fo.write(dumps(recs[0]) + "\n")

    # ======== 统计摘要 ========
    stats = {
        "cohort_start_year": COHORT_START_YEAR,
        "authors_total": total_authors,
        "authors_with_papers_in_window": len(hit_authors),
        "authors_without_papers_in_window": len(nohit_authors),
        "window_years": [YEAR_START, YEAR_END],
        "skipped": {
            "bad_json": int(counters["skip_bad_json"]),
            "no_authors": int(counters["skip_no_authors"]),
            "no_paper_id": int(counters["skip_no_paper_id"]),
        },
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write(dumps(stats))

    print("\n[SUMMARY]")
    print(f"Cohort authors: {stats['authors_total']}  | with papers: {stats['authors_with_papers_in_window']} | without: {stats['authors_without_papers_in_window']}")
    print(f"Features: {OUT_FEATURES_PARQUET if wrote_parquet else OUT_FEATURES_CSV}")
    print(f"Rows (author × year): {len(out_rows)}")
    print(f"Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
