# import os, re, sys, json, codecs
# from pathlib import Path
# from collections import defaultdict
# from tqdm.auto import tqdm
#
# # ================= 配置（按需修改） =================
# AUTHOR_JSONL   = r"full_info_of_authors_new\1999_from_slice/author_papers.jsonl"  # 你已对齐过 author×paper 的主数据（1998 cohort 的 1998-2007 全部论文）
#
# # 优先使用：按年小切片目录（包含 1998.jsonl、1999.jsonl … 或者文件名里含 1998/1999…）
# YEAR_SLICES_DIR = r"dblp_out/year_slices/by_year"   # 留空或不存在则自动回退到大切片
# # 备用：大切片（整段 1998-2022）
# BIG_SLICE_JSONL = r"dblp_out/slice_1998_2022_nopreprint/DBLP_1998_2022_no_preprint.jsonl"
#
# # CCF / CAS（2019 版；严格列名）
# CCF_XLSX     = r"CCF 推荐会议期刊目录 2019.xlsx"      # 列：number, name_abbr, name, level, doc_type
# CAS_XLSX     = r"中科院分区2019.xlsx"                  # 列：number, name, level（1-42→1区，43-135→2区）
#
# YEAR_START, YEAR_END = 1999, 2008
# OUT_JSONL     = r"full_info_of_authors_new\1998_from_slice/author_year_features_full.jsonl"
# # ====================================================
#
# # -------- JSON 读写（鲁棒） --------
# try:
#     import orjson
#     def jdumps(o): return orjson.dumps(o).decode("utf-8")
#     def jloads(s): return orjson.loads(s)
# except Exception:
#     def jdumps(o): return json.dumps(o, ensure_ascii=False)
#     def jloads(s): return json.loads(s)
#
# def load_jsonl(path: str):
#     out, bad = [], 0
#     with open(path, "rb") as f:
#         for ln, raw in enumerate(f, 1):
#             line = raw.replace(codecs.BOM_UTF8, b"").replace(b"\x00", b"").strip()
#             if not line: continue
#             if line.endswith(b","): line = line[:-1]
#             try:
#                 obj = jloads(line.decode("utf-8-sig"))
#             except Exception:
#                 try: obj = jloads(line.decode("utf-8"))
#                 except Exception:
#                     bad += 1
#                     if bad <= 5: print(f"[WARN] 跳过坏行 {ln}", file=sys.stderr)
#                     continue
#             if isinstance(obj, dict):
#                 out.append(obj)
#     if not out:
#         raise ValueError(f"{path} 为空或无有效 JSON 对象")
#     if bad: print(f"[WARN] {path}: 跳过坏行 {bad} 条")
#     return out
#
# # -------- 规范化 & doc_type --------
# def norm_basic(s: str) -> str:
#     if s is None: return ""
#     s = str(s).strip().lower()
#     return re.sub(r"[\s\.\-:;,_/\\()\[\]{}&+|]+", "", s)
#
# def as_abbr(s: str) -> str:
#     if s is None: return ""
#     return re.sub(r"[^A-Za-z]+", "", str(s)).upper()
#
# def canon_doc_type(s: str) -> str:
#     if not s: return "other"
#     v = str(s).strip().lower()
#     if "journal" in v or v == "期刊": return "journal"
#     if "conference" in v or "conf" in v or v == "会议": return "conference"
#     return v
#
# def to_int_year(y):
#     try: return int(y)
#     except Exception:
#         try:
#             s = str(y).strip()
#             return int(s) if s and re.fullmatch(r"-?\d+", s) else None
#         except Exception:
#             return None
#
# # -------- 读取 CCF / CAS（固定列） --------
# def load_ccf_2019(path: str):
#     try:
#         import pandas as pd
#     except Exception:
#         print("[WARN] 未安装 pandas，跳过 CCF 匹配。")
#         return {}, {}
#     p = Path(path)
#     if not p.exists():
#         print(f"[WARN] 未找到 CCF 文件：{p}，跳过。"); return {}, {}
#     df = pd.read_excel(p, engine="openpyxl", dtype=str)
#     df.columns = [str(c).strip().lower() for c in df.columns]
#     must = {"number","name_abbr","name","level","doc_type"}
#     if not must.issubset(df.columns):
#         print(f"[WARN] CCF 表缺列（需要 {sorted(must)}），跳过。"); return {}, {}
#     df = df[["name_abbr","name","level","doc_type"]].copy()
#     df["level"]    = df["level"].str.upper().str.strip()
#     df["doc_type"] = df["doc_type"].apply(canon_doc_type)
#     df = df[df["level"].isin(["A","B"])]
#     df["name_basic"] = df["name"].apply(norm_basic)
#     df["abbr"]       = df["name_abbr"].apply(lambda x: (str(x).strip().upper() if x is not None else ""))
#     by_name, by_abbr = {}, {}
#     for _, r in df.iterrows():
#         if r["name_basic"]: by_name[r["name_basic"]] = (r["level"], r["doc_type"])
#     for _, r in df.iterrows():
#         if r["abbr"]: by_abbr[r["abbr"]] = (r["level"], r["doc_type"])
#     print(f"[INFO] CCF: name 映射 {len(by_name)} 条；abbr 映射 {len(by_abbr)} 条。")
#     return by_name, by_abbr
#
# def load_cas_2019(path: str):
#     try:
#         import pandas as pd
#     except Exception:
#         print("[WARN] 未安装 pandas，跳过 CAS 匹配。")
#         return {}
#     p = Path(path)
#     if not p.exists():
#         print(f"[WARN] 未找到 CAS 文件：{p}，跳过。"); return {}
#     df = pd.read_excel(p, engine="openpyxl", dtype=str)
#     df.columns = [str(c).strip().lower() for c in df.columns]
#     must = {"number","name","level"}
#     if not must.issubset(df.columns):
#         print(f"[WARN] CAS 表缺列（需要 {sorted(must)}），跳过。"); return {}
#     def to_zone(row):
#         txt = (row["level"] or "").strip()
#         m = re.search(r"([1-4])", txt)
#         if m:
#             z = int(m.group(1))
#             if z in (1,2): return z
#         try:
#             num = int(str(row["number"]).strip())
#             if 1 <= num <= 42: return 1
#             if 43 <= num <= 135: return 2
#         except Exception: pass
#         return None
#     df["zone"] = df.apply(to_zone, axis=1)
#     df = df[df["zone"].isin([1,2])]
#     df["name_basic"] = df["name"].apply(norm_basic)
#     by_name = {r["name_basic"]: int(r["zone"]) for _, r in df.iterrows() if r["name_basic"]}
#     print(f"[INFO] CAS: name 映射 {len(by_name)} 条（仅 1/2 区）。")
#     return by_name
#
# def classify_tier(venue_raw: str, doc_type: str,
#                   cas_by_name: dict, ccf_by_name: dict, ccf_by_abbr: dict):
#     v_basic = norm_basic(venue_raw)
#     v_abbr  = as_abbr(venue_raw)
#     dcat    = canon_doc_type(doc_type)
#     lvl = None
#     hit = ccf_by_name.get(v_basic)
#     if hit and hit[1] == dcat:
#         lvl = hit[0]
#     else:
#         hit2 = ccf_by_abbr.get(v_abbr)
#         if hit2 and hit2[1] == dcat:
#             lvl = hit2[0]
#     zone = cas_by_name.get(v_basic) if dcat == "journal" else None
#     is_top = (lvl == "A") or (zone == 1)
#     is_mid = (lvl == "B") or (zone == 2)
#     if is_top: return "top"
#     if is_mid: return "mid"
#     return "other"
#
# # -------- 工具：找“按年小切片”的文件 --------
# def find_year_files(year_dir: str, year: int):
#     """返回该年份的所有候选文件路径（递归搜 *.jsonl/*.ndjson/*.json，文件名含 4 位年份）"""
#     year_str = str(year)
#     out = []
#     for root, _, files in os.walk(year_dir):
#         for name in files:
#             low = name.lower()
#             if not (low.endswith(".jsonl") or low.endswith(".ndjson") or low.endswith(".json")):
#                 continue
#             if year_str in name:  # 文件名里包含年份
#                 out.append(os.path.join(root, name))
#     return sorted(out)
#
# # -------- 用“按年小切片”反查作者列表（只扫需要的年份与所需 pid） --------
# def build_paper_authors_from_year_slices(year_dir: str, needed_pids_by_year: dict[int, set[str]]):
#     pid2_team = {}
#     pid2_authkeys = {}
#     aid2_names = defaultdict(set)
#
#     for y, pidset in sorted(needed_pids_by_year.items()):
#         if not pidset: continue
#         files = find_year_files(year_dir, y)
#         if not files:
#             print(f"[WARN] 年 {y} 未找到按年切片文件（{year_dir}）")
#             continue
#
#         remaining = set(pidset)  # 本年还需要命中的 paper_id
#         for path in files:
#             with open(path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     if not remaining: break
#                     line = line.strip()
#                     if not line: continue
#                     try:
#                         rec = jloads(line)
#                     except Exception:
#                         continue
#                     pid = rec.get("id") or rec.get("v12_id") or rec.get("paper_id")
#                     if pid not in remaining:
#                         continue
#                     auths = rec.get("authors") or rec.get("v12_authors") or []
#                     if isinstance(auths, dict) and "author" in auths:
#                         auths = auths.get("author") or []
#                     if isinstance(auths, dict): auths = [auths]
#                     if not isinstance(auths, list): auths = []
#
#                     keys = []
#                     for a in auths:
#                         aid = (a.get("id") or a.get("author_id") or
#                                (a.get("ids")[0] if isinstance(a.get("ids"), list) and a.get("ids") else None))
#                         nm  = a.get("name")
#                         key = str(aid) if aid is not None else (nm if nm else None)
#                         if key:
#                             keys.append(key)
#                         if aid and nm:
#                             aid2_names[str(aid)].add(nm)
#
#                     pid2_team[pid] = len(keys) if keys else 1
#                     pid2_authkeys[pid] = keys if keys else []
#                     remaining.remove(pid)
#
#             if not remaining:
#                 break  # 本年所有需要的 paper 已命中 → 换下一年
#
#         if remaining:
#             # 有些 paper 在该年文件中没找到（可能切片路径不完整/文件名不规则）
#             print(f"[WARN] 年 {y} 仍有 {len(remaining)} 篇 paper 未命中（例如 {next(iter(remaining))}）")
#
#     return pid2_team, pid2_authkeys, aid2_names
#
# # -------- 回退：用“大切片”反查（不推荐、但可用） --------
# def build_paper_authors_from_big_slice(slice_jsonl: str, needed_pids: set[str]):
#     pid2_team = {}
#     pid2_authkeys = {}
#     aid2_names = defaultdict(set)
#     remaining = set(needed_pids)
#
#     with open(slice_jsonl, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="scan big slice for needed papers", unit="line"):
#             if not remaining: break
#             line = line.strip()
#             if not line: continue
#             try:
#                 rec = jloads(line)
#             except Exception:
#                 continue
#             pid = rec.get("id") or rec.get("v12_id") or rec.get("paper_id")
#             if pid not in remaining:
#                 continue
#             auths = rec.get("authors") or rec.get("v12_authors") or []
#             if isinstance(auths, dict) and "author" in auths:
#                 auths = auths.get("author") or []
#             if isinstance(auths, dict): auths = [auths]
#             if not isinstance(auths, list): auths = []
#
#             keys = []
#             for a in auths:
#                 aid = (a.get("id") or a.get("author_id") or
#                        (a.get("ids")[0] if isinstance(a.get("ids"), list) and a.get("ids") else None))
#                 nm  = a.get("name")
#                 key = str(aid) if aid is not None else (nm if nm else None)
#                 if key:
#                     keys.append(key)
#                 if aid and nm:
#                     aid2_names[str(aid)].add(nm)
#
#             pid2_team[pid] = len(keys) if keys else 1
#             pid2_authkeys[pid] = keys if keys else []
#             remaining.remove(pid)
#
#     if remaining:
#         print(f"[WARN] 大切片模式：仍有 {len(remaining)} 篇 paper 未命中（例如 {next(iter(remaining))}）")
#     return pid2_team, pid2_authkeys, aid2_names
#
# # ================= 主流程 =================
# def main():
#     # 1) 读 author_papers.jsonl（只保留年份窗口内）
#     rows = load_jsonl(AUTHOR_JSONL)
#     base = []
#     for r in rows:
#         y = to_int_year(r.get("year"))
#         if y is None or y < YEAR_START or y > YEAR_END:
#             continue
#         base.append({
#             "author_id": str(r.get("author_id")) if r.get("author_id") is not None else None,
#             "paper_id":  str(r.get("paper_id"))  if r.get("paper_id")  is not None else None,
#             "year":      y,
#             "is_first_author": 1 if str(r.get("is_first_author")).strip().lower() in {"1","true","yes"} else 0,
#             "venue_raw": r.get("venue_raw"),
#             "doc_type":  r.get("doc_type"),
#         })
#     if not base:
#         sys.exit("[ERR] author_papers.jsonl 在该年份窗口内无记录。")
#
#     # 作者全集、（年→所需 paper 集合）
#     authors = sorted({b["author_id"] for b in base if b["author_id"]})
#     need_pids_by_year = defaultdict(set)
#     for b in base:
#         if b["paper_id"]:
#             need_pids_by_year[b["year"]].add(b["paper_id"])
#     all_needed_pids = {pid for s in need_pids_by_year.values() for pid in s}
#
#     # 2) 反查作者列表（优先按年小切片）
#     pid2_team, pid2_authkeys, aid2_names = {}, {}, defaultdict(set)
#     year_dir = Path(YEAR_SLICES_DIR)
#     if year_dir.exists() and year_dir.is_dir():
#         print("[INFO] 使用按年小切片目录：", YEAR_SLICES_DIR)
#         pid2_team, pid2_authkeys, aid2_names = build_paper_authors_from_year_slices(
#             YEAR_SLICES_DIR, need_pids_by_year
#         )
#     elif Path(BIG_SLICE_JSONL).exists():
#         print("[INFO] 回退到大切片：", BIG_SLICE_JSONL)
#         pid2_team, pid2_authkeys, aid2_names = build_paper_authors_from_big_slice(
#             BIG_SLICE_JSONL, all_needed_pids
#         )
#     else:
#         print("[WARN] 未提供可用的切片源，team_size/合作者将基于缺省：team_size=1、无合作者。")
#
#     # 3) 读取 CCF / CAS
#     ccf_by_name, ccf_by_abbr = load_ccf_2019(CCF_XLSX)
#     cas_by_name = load_cas_2019(CAS_XLSX)
#
#     # 4) 年度累加器
#     n_papers       = defaultdict(int)
#     first_sum      = defaultdict(int)
#     team_sum       = defaultdict(int)
#     team_den       = defaultdict(int)
#     single_sum     = defaultdict(int)
#     venues_set     = defaultdict(set)
#     coauthors_year = defaultdict(set)
#     first_collab   = {}                  # (aid, co_key) -> first_year
#     top_first_cnt  = defaultdict(int)
#     mid_first_cnt  = defaultdict(int)
#
#     # 5) 聚合（逐条 base）
#     for b in tqdm(base, desc="aggregate per author-year", unit="row"):
#         aid, pid, y = b["author_id"], b["paper_id"], b["year"]
#         key = (aid, y)
#         n_papers[key]  += 1
#         first_sum[key] += b["is_first_author"]
#
#         venues_set[key].add(norm_basic(b["venue_raw"]))
#
#         team = pid2_team.get(pid, 1)
#         team_sum[key]  += team
#         team_den[key]  += 1
#         single_sum[key]+= 1 if team == 1 else 0
#
#         # 合作：同篇其他作者键，去自环（按 id 或 name）
#         authkeys = pid2_authkeys.get(pid, [aid])  # 无法反查则视为单作者
#         my_names = aid2_names.get(aid, set())
#         for ck in authkeys:
#             if ck == aid:  # 同 id
#                 continue
#             if ck in my_names:  # 对方键正好是我的名字
#                 continue
#             coauthors_year[key].add(ck)
#             pair = (aid, ck)
#             fy = first_collab.get(pair)
#             first_collab[pair] = y if fy is None else (y if y < fy else fy)
#
#         # 顶/中顶：仅一作才计数
#         if b["is_first_author"] == 1 and (ccf_by_name or ccf_by_abbr or cas_by_name):
#             tier = classify_tier(b["venue_raw"] or "", b["doc_type"] or "", cas_by_name, ccf_by_name, ccf_by_abbr)
#             if tier == "top": top_first_cnt[key] += 1
#             elif tier == "mid": mid_first_cnt[key] += 1
#
#     # 6) 生成作者×年份全框架并写 JSONL
#     years = list(range(YEAR_START, YEAR_END + 1))
#     first_by_author = defaultdict(lambda: defaultdict(int))
#     for (aid, ck), fy in first_collab.items():
#         if fy is not None and YEAR_START <= fy <= YEAR_END:
#             first_by_author[aid][fy] += 1
#
#     Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
#     out_rows = 0
#     with open(OUT_JSONL, "w", encoding="utf-8") as fo:
#         for aid in tqdm(authors, desc="write (author × year)"):
#             cum_papers = 0
#             cum_unique = 0
#             new_hist   = first_by_author.get(aid, {})
#             for y in years:
#                 key = (aid, y)
#                 npap = n_papers.get(key, 0)
#                 cum_papers += npap
#
#                 first_share = (first_sum.get(key, 0) / npap) if npap > 0 else None
#                 avg_team    = (team_sum.get(key, 0) / team_den.get(key, 1)) if team_den.get(key, 0) > 0 else None
#                 single_share= (single_sum.get(key, 0) / npap) if npap > 0 else None
#
#                 venues_year = len(venues_set.get(key, set()))
#                 uniq_co     = len(coauthors_year.get(key, set()))
#                 new_co      = new_hist.get(y, 0)
#                 cum_unique += new_co
#                 repeat_ratio= ((uniq_co - new_co) / uniq_co) if uniq_co > 0 else None
#
#                 top_first = None if npap == 0 else int(top_first_cnt.get(key, 0))
#                 mid_first = None if npap == 0 else int(mid_first_cnt.get(key, 0))
#
#                 rec = {
#                     "author_id": aid,
#                     "year": y,
#                     "n_papers": npap,
#                     "n_papers_cum": cum_papers,
#                     "first_author_share": first_share,
#                     "avg_team_size": avg_team,
#                     "single_author_share": single_share,
#                     "venues_dedup_year": venues_year,
#                     "unique_coauthors_year": uniq_co,
#                     "new_coauthors_year": new_co,
#                     "cum_unique_coauthors": cum_unique,
#                     "repeat_collab_ratio": repeat_ratio,
#                     "top_first": top_first,
#                     "mid_first": mid_first,
#                 }
#                 fo.write(jdumps(rec) + "\n")
#                 out_rows += 1
#
#     print(f"[OK] 写出：{OUT_JSONL}  行数={out_rows}  作者数={len(authors)}  年份={YEAR_START}-{YEAR_END}")
#
# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批处理 1999–2009 全部 cohort：
- 输入：full_info_of_authors_new/{YYYY}_from_slice/author_papers.jsonl
- 反查：优先使用按年小切片 YEAR_SLICES_DIR（只打开需要年份、只命中需要的 paper_id 即停）
       若缺失则回退 BIG_SLICE_JSONL
- 匹配：CCF 2019（A/B，先 name 后 abbr，且 doc_type 一致），CAS 2019（期刊 1/2 区）
- 输出：每个 cohort 目录下写 author_year_features_full.jsonl（只 JSONL，不写 parquet/csv）
- 无发文年份：top_first/mid_first = null

按需修改“配置区”的路径。
"""

import os, re, sys, json, codecs
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

# ================= 配置区 =================
COHORT_ROOT       = r"full_info_of_authors_new"  # 根目录
COHORT_YEARS      = list(range(1999, 2010))      # 1999..2009

AUTHOR_FILE_NAME  = "author_papers.jsonl"
OUTPUT_FILE_NAME  = "author_year_features_full.jsonl"

# 按年小切片目录（文件名包含年份，后缀 .jsonl/.ndjson/.json）
YEAR_SLICES_DIR   = r"dblp_out/year_slices/by_year"
# 备用：大切片文件（整段 1998-2022）
BIG_SLICE_JSONL   = r"dblp_out/slice_1998_2022_nopreprint/DBLP_1998_2022_no_preprint.jsonl"

# CCF / CAS（2019 固定列）
CCF_XLSX          = r"CCF 推荐会议期刊目录 2019.xlsx"      # 列：number, name_abbr, name, level, doc_type
CAS_XLSX          = r"中科院分区2019.xlsx"                  # 列：number, name, level（1-42→1区，43-135→2区）

# 如需同时写一个“全体汇总”的 JSONL，打开此开关
WRITE_COMBINED    = False
COMBINED_JSONL    = r"./author_year_features_full_all_cohorts.jsonl"
# ========================================

# -------- JSON 读写（鲁棒） --------
try:
    import orjson
    def jdumps(o): return orjson.dumps(o).decode("utf-8")
    def jloads(s): return orjson.loads(s)
except Exception:
    def jdumps(o): return json.dumps(o, ensure_ascii=False)
    def jloads(s): return json.loads(s)

def load_jsonl(path: str):
    out, bad = [], 0
    with open(path, "rb") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.replace(codecs.BOM_UTF8, b"").replace(b"\x00", b"").strip()
            if not line: continue
            if line.endswith(b","): line = line[:-1]
            try:
                obj = jloads(line.decode("utf-8-sig"))
            except Exception:
                try: obj = jloads(line.decode("utf-8"))
                except Exception:
                    bad += 1
                    if bad <= 5: print(f"[WARN] {path} 跳过坏行 {ln}", file=sys.stderr)
                    continue
            if isinstance(obj, dict):
                out.append(obj)
    if not out:
        raise ValueError(f"{path} 为空或无有效 JSON 对象")
    if bad: print(f"[WARN] {path}: 跳过坏行 {bad} 条")
    return out

# -------- 规范化 & doc_type --------
def norm_basic(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    return re.sub(r"[\s\.\-:;,_/\\()\[\]{}&+|]+", "", s)

def as_abbr(s: str) -> str:
    if s is None: return ""
    return re.sub(r"[^A-Za-z]+", "", str(s)).upper()

def canon_doc_type(s: str) -> str:
    if not s: return "other"
    v = str(s).strip().lower()
    if "journal" in v or v == "期刊": return "journal"
    if "conference" in v or "conf" in v or v == "会议": return "conference"
    return v

def to_int_year(y):
    try: return int(y)
    except Exception:
        try:
            s = str(y).strip()
            return int(s) if s and re.fullmatch(r"-?\d+", s) else None
        except Exception:
            return None

# -------- 读取 CCF / CAS（固定列） --------
def load_ccf_2019(path: str):
    try:
        import pandas as pd
    except Exception:
        print("[WARN] 未安装 pandas，跳过 CCF 匹配。")
        return {}, {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] 未找到 CCF 文件：{p}，跳过。"); return {}, {}
    df = pd.read_excel(p, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]
    must = {"number","name_abbr","name","level","doc_type"}
    if not must.issubset(df.columns):
        print(f"[WARN] CCF 表缺列（需要 {sorted(must)}），跳过。"); return {}, {}
    df = df[["name_abbr","name","level","doc_type"]].copy()
    df["level"]    = df["level"].str.upper().str.strip()
    df["doc_type"] = df["doc_type"].apply(canon_doc_type)
    df = df[df["level"].isin(["A","B"])]
    df["name_basic"] = df["name"].apply(norm_basic)
    df["abbr"]       = df["name_abbr"].apply(lambda x: (str(x).strip().upper() if x is not None else ""))
    by_name, by_abbr = {}, {}
    for _, r in df.iterrows():
        if r["name_basic"]: by_name[r["name_basic"]] = (r["level"], r["doc_type"])
    for _, r in df.iterrows():
        if r["abbr"]: by_abbr[r["abbr"]] = (r["level"], r["doc_type"])
    print(f"[INFO] CCF: name 映射 {len(by_name)} 条；abbr 映射 {len(by_abbr)} 条。")
    return by_name, by_abbr

def load_cas_2019(path: str):
    try:
        import pandas as pd
    except Exception:
        print("[WARN] 未安装 pandas，跳过 CAS 匹配。")
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] 未找到 CAS 文件：{p}，跳过。"); return {}
    df = pd.read_excel(p, engine="openpyxl", dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]
    must = {"number","name","level"}
    if not must.issubset(df.columns):
        print(f"[WARN] CAS 表缺列（需要 {sorted(must)}），跳过。"); return {}
    def to_zone(row):
        txt = (row["level"] or "").strip()
        m = re.search(r"([1-4])", txt)
        if m:
            z = int(m.group(1))
            if z in (1,2): return z
        try:
            num = int(str(row["number"]).strip())
            if 1 <= num <= 42: return 1
            if 43 <= num <= 135: return 2
        except Exception: pass
        return None
    df["zone"] = df.apply(to_zone, axis=1)
    df = df[df["zone"].isin([1,2])]
    df["name_basic"] = df["name"].apply(norm_basic)
    by_name = {r["name_basic"]: int(r["zone"]) for _, r in df.iterrows() if r["name_basic"]}
    print(f"[INFO] CAS: name 映射 {len(by_name)} 条（仅 1/2 区）。")
    return by_name

def classify_tier(venue_raw: str, doc_type: str,
                  cas_by_name: dict, ccf_by_name: dict, ccf_by_abbr: dict):
    v_basic = norm_basic(venue_raw)
    v_abbr  = as_abbr(venue_raw)
    dcat    = canon_doc_type(doc_type)
    lvl = None
    hit = ccf_by_name.get(v_basic)
    if hit and hit[1] == dcat:
        lvl = hit[0]
    else:
        hit2 = ccf_by_abbr.get(v_abbr)
        if hit2 and hit2[1] == dcat:
            lvl = hit2[0]
    zone = cas_by_name.get(v_basic) if dcat == "journal" else None
    is_top = (lvl == "A") or (zone == 1)
    is_mid = (lvl == "B") or (zone == 2)
    if is_top: return "top"
    if is_mid: return "mid"
    return "other"

# -------- 找按年小切片文件 --------
def find_year_files(year_dir: str, year: int):
    year_str = str(year)
    out = []
    for root, _, files in os.walk(year_dir):
        for name in files:
            low = name.lower()
            if not (low.endswith(".jsonl") or low.endswith(".ndjson") or low.endswith(".json")):
                continue
            if year_str in name:
                out.append(os.path.join(root, name))
    return sorted(out)

# -------- 用按年小切片反查作者列表 --------
def build_paper_authors_from_year_slices(year_dir: str, needed_pids_by_year: dict[int, set[str]]):
    pid2_team = {}
    pid2_authkeys = {}
    aid2_names = defaultdict(set)

    for y, pidset in sorted(needed_pids_by_year.items()):
        if not pidset: continue
        files = find_year_files(year_dir, y)
        if not files:
            print(f"[WARN] 年 {y} 未找到按年切片文件（{year_dir}）")
            continue
        remaining = set(pidset)
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not remaining: break
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = jloads(line)
                    except Exception:
                        continue
                    pid = rec.get("id") or rec.get("v12_id") or rec.get("paper_id")
                    if pid not in remaining:
                        continue
                    auths = rec.get("authors") or rec.get("v12_authors") or []
                    if isinstance(auths, dict) and "author" in auths:
                        auths = auths.get("author") or []
                    if isinstance(auths, dict): auths = [auths]
                    if not isinstance(auths, list): auths = []

                    keys = []
                    for a in auths:
                        aid = (a.get("id") or a.get("author_id") or
                               (a.get("ids")[0] if isinstance(a.get("ids"), list) and a.get("ids") else None))
                        nm  = a.get("name")
                        key = str(aid) if aid is not None else (nm if nm else None)
                        if key:
                            keys.append(key)
                        if aid and nm:
                            aid2_names[str(aid)].add(nm)

                    pid2_team[pid] = len(keys) if keys else 1
                    pid2_authkeys[pid] = keys if keys else []
                    remaining.remove(pid)
            if not remaining:
                break
        if remaining:
            print(f"[WARN] 年 {y} 有 {len(remaining)} 篇 paper 未命中（例如 {next(iter(remaining))}）")
    return pid2_team, pid2_authkeys, aid2_names

# -------- 回退：大切片反查 --------
def build_paper_authors_from_big_slice(slice_jsonl: str, needed_pids: set[str]):
    pid2_team = {}
    pid2_authkeys = {}
    aid2_names = defaultdict(set)
    remaining = set(needed_pids)

    with open(slice_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="scan big slice for needed papers", unit="line"):
            if not remaining: break
            line = line.strip()
            if not line: continue
            try:
                rec = jloads(line)
            except Exception:
                continue
            pid = rec.get("id") or rec.get("v12_id") or rec.get("paper_id")
            if pid not in remaining:
                continue
            auths = rec.get("authors") or rec.get("v12_authors") or []
            if isinstance(auths, dict) and "author" in auths:
                auths = auths.get("author") or []
            if isinstance(auths, dict): auths = [auths]
            if not isinstance(auths, list): auths = []

            keys = []
            for a in auths:
                aid = (a.get("id") or a.get("author_id") or
                       (a.get("ids")[0] if isinstance(a.get("ids"), list) and a.get("ids") else None))
                nm  = a.get("name")
                key = str(aid) if aid is not None else (nm if nm else None)
                if key:
                    keys.append(key)
                if aid and nm:
                    aid2_names[str(aid)].add(nm)

            pid2_team[pid] = len(keys) if keys else 1
            pid2_authkeys[pid] = keys if keys else []
            remaining.remove(pid)

    if remaining:
        print(f"[WARN] 大切片模式：仍有 {len(remaining)} 篇 paper 未命中（例如 {next(iter(remaining))}）")
    return pid2_team, pid2_authkeys, aid2_names

# -------- 单个 cohort 处理 --------
def process_one_cohort(start_year: int,
                       ccf_by_name, ccf_by_abbr, cas_by_name,
                       year_slices_dir: str, big_slice_jsonl: str,
                       combined_writer=None):
    cohort_dir = Path(COHORT_ROOT) / f"{start_year}_from_slice"
    ap_path = cohort_dir / AUTHOR_FILE_NAME
    out_path = cohort_dir / OUTPUT_FILE_NAME

    if not ap_path.exists():
        print(f"[WARN] 跳过 {start_year}：未找到 {ap_path}")
        return 0

    print(f"\n=== Cohort {start_year} ===")
    rows = load_jsonl(str(ap_path))

    # 过滤到入行年..入行年+9
    y0, y1 = start_year, start_year + 9
    base = []
    for r in rows:
        y = to_int_year(r.get("year"))
        if y is None or y < y0 or y > y1:
            continue
        base.append({
            "author_id": str(r.get("author_id")) if r.get("author_id") is not None else None,
            "paper_id":  str(r.get("paper_id"))  if r.get("paper_id")  is not None else None,
            "year":      y,
            "is_first_author": 1 if str(r.get("is_first_author")).strip().lower() in {"1","true","yes"} else 0,
            "venue_raw": r.get("venue_raw"),
            "doc_type":  r.get("doc_type"),
        })
    if not base:
        print(f"[WARN] cohort {start_year} 在 {y0}-{y1} 没有记录。")
        return 0

    authors = sorted({b["author_id"] for b in base if b["author_id"]})
    need_pids_by_year = defaultdict(set)
    for b in base:
        if b["paper_id"]:
            need_pids_by_year[b["year"]].add(b["paper_id"])
    all_needed_pids = {pid for s in need_pids_by_year.values() for pid in s}

    # 反查作者列表：优先按年小切片
    pid2_team, pid2_authkeys, aid2_names = {}, {}, defaultdict(set)
    year_dir = Path(year_slices_dir)
    if year_dir.exists() and year_dir.is_dir():
        print("[INFO] 使用按年小切片目录：", year_slices_dir)
        pid2_team, pid2_authkeys, aid2_names = build_paper_authors_from_year_slices(year_slices_dir, need_pids_by_year)
    elif big_slice_jsonl and Path(big_slice_jsonl).exists():
        print("[INFO] 回退到大切片：", big_slice_jsonl)
        pid2_team, pid2_authkeys, aid2_names = build_paper_authors_from_big_slice(big_slice_jsonl, all_needed_pids)
    else:
        print("[WARN] 未提供可用切片，team_size/合作将按缺省（team=1，无合作者）。")

    # 年度累加器
    n_papers       = defaultdict(int)
    first_sum      = defaultdict(int)
    team_sum       = defaultdict(int)
    team_den       = defaultdict(int)
    single_sum     = defaultdict(int)
    venues_set     = defaultdict(set)
    coauthors_year = defaultdict(set)
    first_collab   = {}                  # (aid, co_key) -> first_year
    top_first_cnt  = defaultdict(int)
    mid_first_cnt  = defaultdict(int)

    # 聚合
    for b in tqdm(base, desc=f"aggregate {start_year}"):
        aid, pid, y = b["author_id"], b["paper_id"], b["year"]
        key = (aid, y)
        n_papers[key]  += 1
        first_sum[key] += b["is_first_author"]
        venues_set[key].add(norm_basic(b["venue_raw"]))

        team = pid2_team.get(pid, 1)
        team_sum[key]  += team
        team_den[key]  += 1
        single_sum[key]+= 1 if team == 1 else 0

        authkeys = pid2_authkeys.get(pid, [aid])  # 无法反查则视为单作者
        my_names = aid2_names.get(aid, set())
        for ck in authkeys:
            if ck == aid:     # 同 id
                continue
            if ck in my_names:  # 我自己的名字
                continue
            coauthors_year[key].add(ck)
            pair = (aid, ck)
            fy = first_collab.get(pair)
            first_collab[pair] = y if fy is None else (y if y < fy else fy)

        if b["is_first_author"] == 1 and (ccf_by_name or ccf_by_abbr or cas_by_name):
            tier = classify_tier(b["venue_raw"] or "", b["doc_type"] or "", cas_by_name, ccf_by_name, ccf_by_abbr)
            if tier == "top": top_first_cnt[key] += 1
            elif tier == "mid": mid_first_cnt[key] += 1

    # 写 JSONL（每个作者补齐 10 年）
    out_rows = 0
    years = list(range(y0, y1 + 1))
    first_by_author = defaultdict(lambda: defaultdict(int))
    for (aid, ck), fy in first_collab.items():
        if fy is not None and y0 <= fy <= y1:
            first_by_author[aid][fy] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for aid in tqdm(authors, desc=f"write {start_year}"):
            cum_papers = 0
            cum_unique = 0
            new_hist   = first_by_author.get(aid, {})
            for y in years:
                key = (aid, y)
                npap = n_papers.get(key, 0)
                cum_papers += npap

                first_share = (first_sum.get(key, 0) / npap) if npap > 0 else None
                avg_team    = (team_sum.get(key, 0) / team_den.get(key, 1)) if team_den.get(key, 0) > 0 else None
                single_share= (single_sum.get(key, 0) / npap) if npap > 0 else None

                venues_year = len(venues_set.get(key, set()))
                uniq_co     = len(coauthors_year.get(key, set()))
                new_co      = new_hist.get(y, 0)
                cum_unique += new_co
                repeat_ratio= ((uniq_co - new_co) / uniq_co) if uniq_co > 0 else None

                top_first = None if npap == 0 else int(top_first_cnt.get(key, 0))
                mid_first = None if npap == 0 else int(mid_first_cnt.get(key, 0))

                rec = {
                    "author_id": aid,
                    "start_year": start_year,
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
                    "top_first": top_first,
                    "mid_first": mid_first,
                }
                line = jdumps(rec)
                fo.write(line + "\n")
                if combined_writer:
                    combined_writer.write(line + "\n")
                out_rows += 1

    print(f"[OK] {start_year}: 写出 {out_rows} 行 → {out_path}")
    return out_rows

# ================= 主程序 =================
def main():
    # 预载 CCF/CAS 映射（一次加载，多 cohort 复用）
    ccf_by_name, ccf_by_abbr = load_ccf_2019(CCF_XLSX)
    cas_by_name = load_cas_2019(CAS_XLSX)

    combined_fp = open(COMBINED_JSONL, "w", encoding="utf-8") if WRITE_COMBINED else None

    total_rows = 0
    for sy in COHORT_YEARS:
        total_rows += process_one_cohort(
            start_year=sy,
            ccf_by_name=ccf_by_name, ccf_by_abbr=ccf_by_abbr, cas_by_name=cas_by_name,
            year_slices_dir=YEAR_SLICES_DIR, big_slice_jsonl=BIG_SLICE_JSONL,
            combined_writer=combined_fp
        )

    if combined_fp:
        combined_fp.close()
        print(f"[OK] 合并写出：{COMBINED_JSONL}")

    print(f"\n====== 全部完成 ======")
    print(f"cohorts: {COHORT_YEARS[0]}–{COHORT_YEARS[-1]}  总输出行数: {total_rows}")

if __name__ == "__main__":
    main()
