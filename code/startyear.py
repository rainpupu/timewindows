# parallel_min_year_mini.py 并行计算每位学者的入行年份，并统计1998-2012年的学者
import os, glob, orjson, pandas as pd
from multiprocessing import Pool, cpu_count, get_context
from tqdm.auto import tqdm

OUT_DIR = "dblp_out"
SHARD_DIR = os.path.join(OUT_DIR, "shards_by_year_mini")
NUM_WORKERS = min(24, cpu_count())

def process_one(path):
    import collections
    min_year = collections.defaultdict(lambda: 10**9)
    y = int(os.path.splitext(os.path.basename(path))[0])
    with open(path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            for aid in obj.get("author_ids", []):
                if y < min_year[aid]:
                    min_year[aid] = y
    # 返回为 DataFrame，主进程统一归并
    return pd.DataFrame({"author_id": list(min_year.keys()),
                         "start_year": list(min_year.values())})

if __name__ == "__main__":
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.jsonl")))
    assert shards, "未找到 mini 年分片，请先运行 make_shards_mini.py"
    print(f"并行处理 {len(shards)} 个年份分片，进程数 {NUM_WORKERS}")
    dfs = []
    with get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for df in tqdm(pool.imap_unordered(process_one, shards), total=len(shards), desc="reduce"):
            dfs.append(df)
    # 全局最小
    df_all = pd.concat(dfs, ignore_index=True).groupby("author_id", as_index=False)["start_year"].min()
    df_all.to_parquet(os.path.join(OUT_DIR, "author_start_years_all.parquet"), compression="snappy")
    sub = df_all[(df_all.start_year>=1998)&(df_all.start_year<=2012)]
    sub.to_parquet(os.path.join(OUT_DIR, "author_start_years_1998_2012.parquet"), compression="snappy")
    sub[["author_id"]].to_parquet(os.path.join(OUT_DIR, "author_ids_1998_2012.parquet"), compression="snappy")
    print(f"✅ 全作者：{len(df_all)}；1998–2012 入行作者：{len(sub)}")
