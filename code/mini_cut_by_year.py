# make_shards_mini.py  (V14 版)
# 顶层为 JSON 数组：逐条流式读取 → 过滤预印本(arXiv/CoRR) → 按 year 写入 {year, author_ids}
# 预印本判定(V14数据)：
#   A) venue.raw 含 "corr" 或 "arxiv"
#   B) 任一 url 含 "arxiv.org"
#   C) doi 以 "10.48550/arXiv" 开头
# 以上任一命中即视为预印本

import os, re
from decimal import Decimal
from tqdm.auto import tqdm

# ijson：优先 C 后端
try:
    import ijson.backends.yajl2_c as ijson
except Exception:
    try:
        import ijson.backends.yajl2 as ijson
    except Exception:
        import ijson

# orjson：高速写；不可用则回退到内置 json
try:
    import orjson
    def dumps(obj): return orjson.dumps(obj)
except Exception:
    import json
    def dumps(obj): return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

INPUT_FILE = "dblp.json"
OUT_DIR    = "dblp_out"
SHARD_DIR  = os.path.join(OUT_DIR, "shards_by_year_mini")
os.makedirs(SHARD_DIR, exist_ok=True)

# ---- helpers ----
def to_int_year(y):
    if isinstance(y, Decimal):
        try: return int(y)
        except Exception: return None
    try:
        return int(y)
    except Exception:
        m = re.match(r"(\d{4})", str(y or ""))
        return int(m.group(1)) if m else None

def _lower(x):
    try: return str(x).strip().lower()
    except Exception: return ""

def _as_list(x):
    if x is None: return []
    return x if isinstance(x, (list, tuple)) else [x]

def authors_iter(rec):
    """
    V14里常见：
      - authors: [ {name, org, id, ...}, ...]
      - v12_authors: [ {name, org, id, ...}, ...]
      - 某些导出也可能有 'author' 键
    """
    for key in ("authors", "v12_authors", "author"):
        v = rec.get(key)
        if v:
            for a in _as_list(v):
                if isinstance(a, dict):
                    yield a

def pick_author_id(a):
    # V14作者ID通常是 a["id"] 或 v12_authors.id；也兼容 orcid/pid
    return (a.get("id")
            or a.get("pid")
            or a.get("orcid")
            or a.get("v12_authors.id"))

def is_preprint_v14(rec):
    # A) venue.raw 含 corr/arxiv
    vraw = _lower(rec.get("venue.raw"))
    if ("corr" in vraw) or ("arxiv" in vraw):
        return True

    # B) url 列表（或单值）含 arxiv.org
    for u in _as_list(rec.get("url")):
        if "arxiv.org" in _lower(u):
            return True

    # C) doi 以 10.48550/arXiv 开头
    doi = _lower(rec.get("doi"))
    if doi.startswith("10.48550/arxiv"):
        return True

    # （如你还有其它字段能识别预印本，可在这里追加）
    return False

# ---- main ----
print("模式：顶层数组 → 过滤预印本(V14) → 按年份分片（仅 year + author_ids）")
handles = {}
kept = dropped_preprint = dropped_no_year = dropped_no_authors = 0

with open(INPUT_FILE, "rb") as f:
    for rec in tqdm(ijson.items(f, "item"), desc="scan array"):
        # 1) 过滤预印本
        if is_preprint_v14(rec):
            dropped_preprint += 1
            continue

        # 2) 年份
        y = to_int_year(rec.get("year"))
        if y is None:
            dropped_no_year += 1
            continue

        # 3) 提取作者ID
        ids = []
        for a in authors_iter(rec):
            aid = pick_author_id(a)
            if aid:
                ids.append(str(aid))
        if not ids:
            dropped_no_authors += 1
            continue

        # 4) 写入对应年的 JSONL
        if y not in handles:
            handles[y] = open(os.path.join(SHARD_DIR, f"{y}.jsonl"), "ab")
        handles[y].write(dumps({"year": y, "author_ids": ids}))
        handles[y].write(b"\n")
        kept += 1

for fp in handles.values():
    fp.close()

print(f"✅ mini 分片完成：{kept} 条；目录：{SHARD_DIR}")
print(f"   丢弃统计 → 预印本: {dropped_preprint}，无 year: {dropped_no_year}，无作者ID: {dropped_no_authors}")



