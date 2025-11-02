import os, json, gzip
from collections import Counter, defaultdict
from decimal import Decimal
from tqdm.auto import tqdm

# ==================== 配置区（按需修改） ====================
INPUT_PATH = r"dblp_out\slice_1998_2022_nopreprint\DBLP_1998_2022_no_preprint.jsonl"
OUT_DIR    = r"dblp_out\year_slices"     # 输出目录
YEAR_MIN, YEAR_MAX = 1998, 2022
GZIP_OUTPUT = False                      # True 则输出 .jsonl.gz
# ===========================================================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def detect_container_type(path) -> str:
    """返回 'ndjson' 或 'array'"""
    with open(path, "rb") as f:
        head = f.read(4096)
    s = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return "array" if s[:1] == b"[" else "ndjson"

def to_int_year(y):
    if y is None or isinstance(y, bool): return None
    if isinstance(y, Decimal):
        try: return int(y)
        except Exception: return None
    try:
        return int(y)
    except Exception:
        try:
            return int(float(str(y).strip()))
        except Exception:
            return None

# 为顶层数组场景准备一个稳健的 dumps（NDJSON 输入我们直接复用原始行，不重新序列化）
try:
    import orjson
    def dumps_obj(obj) -> str:
        def _default(o):
            if isinstance(o, Decimal):  # ijson 可能给 Decimal
                return float(o)
            if isinstance(o, (set, tuple)): return list(o)
            if isinstance(o, bytes): return o.decode("utf-8", "ignore")
            raise TypeError
        return orjson.dumps(obj, default=_default, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
except Exception:
    def dumps_obj(obj) -> str:
        class _Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, Decimal):
                    return float(o)
                return super().default(o)
        return json.dumps(obj, ensure_ascii=False, cls=_Enc)

def open_writer(base_dir, year, gzip_mode=False):
    ensure_dir(base_dir)
    if gzip_mode:
        fp = os.path.join(base_dir, f"{year}.jsonl.gz")
        return gzip.open(fp, "at", encoding="utf-8"), fp
    else:
        fp = os.path.join(base_dir, f"{year}.jsonl")
        return open(fp, "w", encoding="utf-8"), fp

def main():
    ensure_dir(OUT_DIR)
    out_year_dir = os.path.join(OUT_DIR, "by_year")
    ensure_dir(out_year_dir)

    container = detect_container_type(INPUT_PATH)
    print(f"[INFO] 输入文件：{INPUT_PATH}")
    print(f"[INFO] 检测到格式：{container}")

    # 懒加载每年的 writer
    writers = {}     # year -> file handle
    writer_paths = {}# year -> path
    counts = Counter()

    stats = defaultdict(int)  # scanned/skip_no_year/skip_out_of_range/skip_bad_json

    def write_record(year, text_line: str):
        if year not in writers:
            fh, path = open_writer(out_year_dir, year, gzip_mode=GZIP_OUTPUT)
            writers[year] = fh
            writer_paths[year] = path
        writers[year].write(text_line)
        if not text_line.endswith("\n"):
            writers[year].write("\n")

    try:
        if container == "ndjson":
            total_bytes = os.path.getsize(INPUT_PATH)
            with open(INPUT_PATH, "rb") as f, \
                 tqdm(total=total_bytes, unit="B", unit_scale=True, desc="scan NDJSON") as pbar:
                for raw in f:
                    pbar.update(len(raw))
                    line = raw.decode("utf-8", errors="ignore").strip()
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
                        stats["skip_out_of_range"] += 1
                        continue

                    # 按原始行写入，避免重复序列化误差
                    write_record(y, line)
                    counts[y] += 1

        else:
            # 顶层数组：用 ijson 流式逐对象
            try:
                import ijson.backends.yajl2_c as ijson
            except Exception:
                import ijson
            with open(INPUT_PATH, "rb") as fb:
                it = ijson.items(fb, "item")
                for rec in tqdm(it, desc="scan JSON array", unit="obj"):
                    stats["scanned"] += 1
                    y = to_int_year(rec.get("year"))
                    if y is None:
                        stats["skip_no_year"] += 1
                        continue
                    if y < YEAR_MIN or y > YEAR_MAX:
                        stats["skip_out_of_range"] += 1
                        continue

                    # 这里需要重新序列化
                    write_record(y, dumps_obj(rec))
                    counts[y] += 1

    finally:
        # 关闭所有 writer
        for fh in writers.values():
            try:
                fh.close()
            except Exception:
                pass

    # 写出统计
    # 1) 每年论文数 CSV
    csv_path = os.path.join(OUT_DIR, "counts_by_year.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("year,count\n")
        for y in range(YEAR_MIN, YEAR_MAX + 1):
            f.write(f"{y},{counts.get(y,0)}\n")

    # 2) stats.json
    out_stats = {
        "input": INPUT_PATH,
        "year_range": [YEAR_MIN, YEAR_MAX],
        "scanned": int(stats["scanned"]),
        "skipped": {
            "no_year": int(stats["skip_no_year"]),
            "out_of_range": int(stats["skip_out_of_range"]),
            "bad_json": int(stats["skip_bad_json"]),
        },
        "counts_by_year": {str(y): int(counts[y]) for y in sorted(counts.keys())},
        "outputs": {str(y): writer_paths.get(y) for y in sorted(writer_paths.keys())}
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(out_stats, f, ensure_ascii=False, indent=2)

    # 控制台打印结果
    print("\n[SUMMARY] 每年论文数（1998–2022）")
    for y in range(YEAR_MIN, YEAR_MAX + 1):
        print(f"{y}: {counts.get(y,0)}")
    print(f"\n输出目录：{OUT_DIR}")
    print(f"- 年份切片：{out_year_dir}")
    print(f"- 每年计数：{csv_path}")

if __name__ == "__main__":
    main()
