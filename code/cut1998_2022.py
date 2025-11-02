#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
from collections import defaultdict
from decimal import Decimal
from tqdm.auto import tqdm

# 依赖：ijson, orjson, tqdm
try:
    import ijson.backends.yajl2_c as ijson  # 更快更省内存
except Exception:
    import ijson
import orjson

INPUT_DBLP = "DBLP.json"  # 支持顶层数组 JSON 或 NDJSON/JSONL
YEAR_MIN, YEAR_MAX = 2023, 2024  # <== 改到 2022

OUT_DIR = os.path.join("dblp_out", "slice_2023_2024_nopreprint")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "DBLP_2023_2024_no_preprint.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "stats.json")

# -------------------- 工具函数 --------------------

def _default(o):
    # 统一把 Decimal -> float；如需严格无损可改为 str(o)
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8", "ignore")
        except Exception:
            return str(o)
    # 交还给 orjson 处理其它类型
    raise TypeError

def dumps(obj) -> str:
    return orjson.dumps(
        obj,
        default=_default,                 # 你已有的 Decimal 等处理器
        option=orjson.OPT_NON_STR_KEYS    # 关键：允许 int 等作为字典键
    ).decode("utf-8")

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
            return int(s) if s and s.isdigit() else None
        except Exception:
            return None

def venue_raw_from(rec):
    v = rec.get("venue")
    if isinstance(v, dict):
        return v.get("raw") or v.get("name") or v.get("raw_venue")
    if isinstance(v, str):
        return v
    return rec.get("venue.raw") or rec.get("journal") or rec.get("booktitle")

def ensure_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

_re_key_corr   = re.compile(r"^journals/corr/", re.I)
_re_corr_arxiv = re.compile(r"(?:^|[^a-z])(corr|arxiv)(?:[^a-z]|$)", re.I)
_re_doi_arxiv  = re.compile(r"^10\.48550/arxiv", re.I)

def is_preprint(rec) -> bool:
    # 1) key
    k = rec.get("key")
    if isinstance(k, str) and _re_key_corr.match(k):
        return True
    # 2) venue.raw 等字段命中 corr/arxiv
    vr = venue_raw_from(rec)
    if vr and _re_corr_arxiv.search(str(vr)):
        return True
    # 3) 任意 URL 含 arxiv.org
    for u in ensure_list(rec.get("url") or rec.get("urls") or rec.get("ee")):
        try:
            if "arxiv.org" in str(u).lower():
                return True
        except Exception:
            continue
    # 4) DOI 以 10.48550/arXiv 开头
    doi = rec.get("doi")
    if isinstance(doi, str) and _re_doi_arxiv.match(doi):
        return True
    return False

def detect_container_type(path) -> str:
    """返回 'array' 或 'ndjson'"""
    with open(path, "rb") as f:
        head = f.read(4096)
    s = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return "array" if s[:1] == b"[" else "ndjson"

# -------------------- 主逻辑 --------------------

def main():
    stats = defaultdict(int)
    kept_by_year = defaultdict(int)

    container = detect_container_type(INPUT_DBLP)

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        if container == "array":
            with open(INPUT_DBLP, "rb") as fb:
                it = ijson.items(fb, "item")
                pbar = tqdm(it, desc="scan DBLP.json (array items)", unit="obj")
                for rec in pbar:
                    stats["scanned"] += 1

                    y = to_int_year(rec.get("year"))
                    if y is None:
                        stats["skip_no_year"] += 1
                        continue
                    if y < YEAR_MIN or y > YEAR_MAX:
                        stats["skip_year_out_of_range"] += 1
                        continue
                    if is_preprint(rec):
                        stats["skip_preprint"] += 1
                        continue

                    fout.write(dumps(rec) + "\n")
                    stats["kept"] += 1
                    kept_by_year[y] += 1
        else:
            with open(INPUT_DBLP, "r", encoding="utf-8") as f:
                pbar = tqdm(f, desc="scan DBLP.json (ndjson lines)", unit="line")
                for line in pbar:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        stats["skip_bad_json"] += 1
                        continue

                    stats["scanned"] += 1

                    y = to_int_year(rec.get("year"))
                    if y is None:
                        stats["skip_no_year"] += 1
                        continue
                    if y < YEAR_MIN or y > YEAR_MAX:
                        stats["skip_year_out_of_range"] += 1
                        continue
                    if is_preprint(rec):
                        stats["skip_preprint"] += 1
                        continue

                    fout.write(dumps(rec) + "\n")
                    stats["kept"] += 1
                    kept_by_year[y] += 1

    out_stats = {
        "year_range": [YEAR_MIN, YEAR_MAX],
        "scanned": int(stats["scanned"]),
        "kept": int(stats["kept"]),
        "skipped": {
            "preprint": int(stats["skip_preprint"]),
            "no_year": int(stats["skip_no_year"]),
            "year_out_of_range": int(stats["skip_year_out_of_range"]),
            "bad_json": int(stats["skip_bad_json"]),
        },
        "kept_by_year": dict(sorted(kept_by_year.items()))
    }
    with open(OUT_STATS, "w", encoding="utf-8") as fs:
        fs.write(dumps(out_stats))

    print("\n[SUMMARY]")
    print(f"Container: {container}, Year Range: {YEAR_MIN}-{YEAR_MAX}")
    print(f"Scanned: {out_stats['scanned']}, Kept (non-preprint): {out_stats['kept']}")
    print(f"Skipped -> preprint: {out_stats['skipped']['preprint']}, no_year: {out_stats['skipped']['no_year']}, out_of_range: {out_stats['skipped']['year_out_of_range']}, bad_json: {out_stats['skipped']['bad_json']}")
    print(f"Output: {OUT_JSONL}")
    print(f"Stats : {OUT_STATS}")

if __name__ == "__main__":
    main()
