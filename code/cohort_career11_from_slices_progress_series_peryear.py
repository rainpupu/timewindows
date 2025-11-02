# cohort_career10_from_slices_progress_series_peryear.py
# -*- coding: utf-8 -*-
"""
统计 1998–2009 入行学者，在“入行当年为第1年、……、第10年”为止的【逐年累计被引总数】（10 个截点），
并把结果分别输出到各自的入行年份文件夹：full_info_of_authors_new\\{year}_from_slice\\authors_career10_citations_series.jsonl

输入：
- Cohort 多文件（year=1998..2009）：full_info_of_authors_new\\{year}_from_slice\\author_papers.jsonl
  仅用于确定每位学者的 start_year 与其论文 ID 集。
- 年切片目录（包含 1998.jsonl, 1999.jsonl, …, 2022.jsonl 等；每个文件是一年的所有论文，需含 references）

输出：
- 对每个入行年 y ∈ [1998..2009]，在 full_info_of_authors_new\\{y}_from_slice\\authors_career10_citations_series.jsonl
  写出该年的所有 Cohort 学者的 10 长度累计序列：
  {
    "author_id": "...",
    "start_year": 2003,
    "career10_cutoff_year": 2012,            # = start_year + 9
    "cum_citations_by_career_year": [c1..c10]# 截止第1..第10年（含端点）的累计被引
  }

实现要点：
- 只扫描对窗口有贡献的年份切片：[min(start_year), max(start_year)+9]。
- 并行：每个年份文件一个任务；当年内对同一被引论文的重复引用去重。
- 为每篇 Cohort 论文构建“按年前缀和”，快速查询任意截止年累计值。
"""

from __future__ import annotations
import os, re, json, sys, bisect
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# =================== CONFIG ===================
COHORT_PATH_TEMPLATE   = r"full_info_of_authors_new\{year}_from_slice\author_papers.jsonl"
OUTPUT_PATH_TEMPLATE   = r"full_info_of_authors_new\{year}_from_slice\authors_career11_citations_series.jsonl"
COHORT_START_MIN       = 1998
COHORT_START_MAX       = 2009

YEAR_SLICES_DIR        = r"dblp_out\year_slices\by_year"
WORKERS                = 0    # 0/<=0 自动取 CPU 核心数
# ==============================================

# ---------- JSON ----------
try:
    import orjson
    def jloads(s: str): return orjson.loads(s)
    def jdumps(o) -> str: return orjson.dumps(o).decode("utf-8")
except Exception:
    def jloads(s: str): return json.loads(s)
    def jdumps(o) -> str: return json.dumps(o, ensure_ascii=False)

# ---------- 简易进度条 ----------
class PBar:
    def __init__(self, total: int, prefix: str, width: int = 30):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.width = width
        self.cur = 0
        self.last = -1
    def step(self, n: int = 1):
        self.cur += n
        if self.cur > self.total: self.cur = self.total
        pct = int(self.cur * 100 / self.total)
        if pct == self.last: return
        self.last = pct
        filled = int(self.width * pct // 100)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(f"\r{self.prefix} [{bar}] {pct:3d}% ({self.cur:,}/{self.total:,})")
        sys.stdout.flush()
    def done(self):
        self.step(0)
        sys.stdout.write("\n"); sys.stdout.flush()

# ---------- 读取 Cohort（按入行年 1998..2009 多文件） ----------
def load_cohort_multi(path_template: str, y0: int, y1: int):
    author_papers_by_year: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    author_start: dict[str, int] = {}
    all_cohort_pids: set[str] = set()

    files = []
    for y in range(y0, y1 + 1):
        p = path_template.format(year=y)
        if os.path.exists(p):
            files.append((y, p))
        else:
            print(f"  [WARN] 未找到 Cohort 文件：{p}")
    if not files:
        raise SystemExit("未找到任何 Cohort 文件。")

    pbar = PBar(len(files), prefix="读取 Cohort 年度文件")
    for y, path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = jloads(line)
                except Exception:
                    continue
                aid = rec.get("author_id")
                sy  = rec.get("start_year")
                pid = rec.get("paper_id")
                if aid is None or sy is None or pid is None:
                    continue
                try:
                    syi = int(sy)
                except Exception:
                    continue
                # 按入行年归组
                author_papers_by_year[syi][str(aid)].add(str(pid))
                # 记录首次 start_year
                if str(aid) not in author_start:
                    author_start[str(aid)] = syi
                all_cohort_pids.add(str(pid))
        pbar.step(1)
    pbar.done()

    if not author_start:
        raise SystemExit("Cohort 学者为空，请检查输入。")

    min_start = min(author_start.values())
    max_cutoff = max(s+10 for s in author_start.values())  # 第11年截止 = start+10
    return author_papers_by_year, author_start, all_cohort_pids, min_start, max_cutoff

# ---------- 罗列年份文件 ----------
_year_pat = re.compile(r"(19|20)\d{2}")
def list_year_files(year_dir: str):
    files = []
    for name in os.listdir(year_dir):
        if not name.lower().endswith((".jsonl", ".ndjson", ".json")): continue
        m = _year_pat.search(name)
        if not m: continue
        y = int(m.group(0))
        files.append((y, os.path.join(year_dir, name)))
    files.sort(key=lambda x: x[0])
    return files

# ---------- worker：统计单年对 Cohort 论文的引用 ----------
def _count_one_year(args):
    year, path, cohort_ids = args
    cohort_ids = frozenset(cohort_ids)
    local = defaultdict(int)  # cited_pid -> count（该年的总次数）
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = jloads(line)
            except Exception:
                continue
            refs = rec.get("references")
            if not isinstance(refs, list) or not refs:
                continue
            # 单篇内部去重
            for rid in set(refs):
                rid = str(rid)
                if rid in cohort_ids:
                    local[rid] += 1
    return year, dict(local)

# ---------- 为每篇 Cohort 论文构建“按年前缀和” ----------
def build_pid_prefix(cited_counts_by_year: dict[str, dict[int,int]]):
    years_arr = {}
    prefix_arr = {}
    for pid, ym in cited_counts_by_year.items():
        ys = sorted([y for y in ym.keys() if isinstance(y, int)])
        if not ys:
            continue
        years_arr[pid] = ys
        pref = []
        s = 0
        for y in ys:
            s += int(ym.get(y, 0))
            pref.append(s)
        prefix_arr[pid] = pref
    return years_arr, prefix_arr

def query_pid_prefix(pid_years, pid_prefix, pid: str, cutoff: int) -> int:
    ys = pid_years.get(pid)
    if not ys:
        return 0
    import bisect
    idx = bisect.bisect_right(ys, cutoff) - 1
    if idx < 0:
        return 0
    return pid_prefix[pid][idx]

# ---------- 主流程 ----------
def main():
    print("[1/7] 读取 Cohort（1998..2009 多文件）并分年归组 ...")
    (author_papers_by_year, author_start, cohort_pids,
     min_start, max_cutoff) = load_cohort_multi(COHORT_PATH_TEMPLATE, COHORT_START_MIN, COHORT_START_MAX)
    total_authors = sum(len(d) for d in author_papers_by_year.values())
    print(f"    Cohort 学者总数: {total_authors:,} | Cohort 论文: {len(cohort_pids):,}")
    print(f"    引用扫描年份：{min_start} — {max_cutoff}（第10年截止 = start+9）")

    print("[2/7] 列出年份切片文件 ...")
    all_year_files = list_year_files(YEAR_SLICES_DIR)
    if not all_year_files:
        raise SystemExit(f"未在 {YEAR_SLICES_DIR} 找到年份切片文件。")
    year_files = [(y, p) for (y, p) in all_year_files if (min_start <= y <= max_cutoff)]
    if not year_files:
        raise SystemExit("没有可用年份文件（不在 Cohort 窗口内）。")
    print(f"    将扫描年份：{year_files[0][0]}–{year_files[-1][0]}（共 {len(year_files)} 个文件）")

    print("[3/7] 并行统计每年的“对 Cohort 论文的引用次数” ...")
    if WORKERS is None or WORKERS <= 0:
        try:
            workers = os.cpu_count() or 4
        except Exception:
            workers = 4
    else:
        workers = WORKERS

    cited_counts_by_year = defaultdict(dict)  # pid -> {year -> count}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_count_one_year, (y, path, cohort_pids)) for (y, path) in year_files]
        pbar = PBar(len(futs), prefix="处理年份文件")
        for fu in as_completed(futs):
            year, ymap = fu.result()
            for pid, c in ymap.items():
                cited_counts_by_year[pid][year] = cited_counts_by_year[pid].get(year, 0) + int(c)
            pbar.step(1)
        pbar.done()

    print("[4/7] 为 Cohort 论文构建“按年前缀和” ...")
    pid_years, pid_prefix = build_pid_prefix(cited_counts_by_year)
    print(f"    有被引记录的 Cohort 论文：{len(pid_years):,} 篇")

    print("[5/7] 准备按入行年分别输出 ...")
    # 预创建输出文件夹并打开写句柄
    writers = {}
    counts_by_year = {}
    for y in range(COHORT_START_MIN, COHORT_START_MAX + 1):
        out_path = OUTPUT_PATH_TEMPLATE.format(year=y)
        out_dir  = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        writers[y] = open(out_path, "w", encoding="utf-8")
        counts_by_year[y] = 0

    print("[6/7] 逐作者写出 11 个累计截点（第1..第11年截止），并按入行年分文件 ...")
    total_authors = sum(len(d) for d in author_papers_by_year.values())
    pbar2 = PBar(total_authors, prefix="写出作者序列")

    for y in range(COHORT_START_MIN, COHORT_START_MAX + 1):
        group = author_papers_by_year.get(y, {})
        if not group:
            continue
        fout = writers[y]
        for aid, pids in group.items():
            S = y
            series = []
            for k in range(0, 11):
                cutoff = S + k   # 第k年截止
                total = 0
                for pid in pids:
                    total += query_pid_prefix(pid_years, pid_prefix, pid, cutoff)
                series.append(int(total))
            row = {
                "author_id": aid,
                "start_year": S,
                "career11_cutoff_year": S + 10,
                "cum_citations_by_career_year": series
            }
            fout.write(jdumps(row) + "\n")
            counts_by_year[y] += 1
            pbar2.step(1)

    pbar2.done()
    for w in writers.values():
        try: w.close()
        except Exception: pass

    print("[7/7] 完成")
    for y in range(COHORT_START_MIN, COHORT_START_MAX + 1):
        out_path = OUTPUT_PATH_TEMPLATE.format(year=y)
        print(f"  {y}: {counts_by_year[y]:,} 行 -> {out_path}")

if __name__ == "__main__":
    main()
