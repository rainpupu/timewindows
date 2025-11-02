# -*- coding: utf-8 -*-
"""
group_authors_by_start_year.py
用途：
- 读取你导出的 start_year 映射（支持 .jsonl 或 .json），字段需包含 author_id, start_year
- 将作者按 start_year 分组，输出：
  1) by_start_year/start_year_YYYY.jsonl   （每行: {"author_id": "...", "start_year": YYYY}）
  2) by_start_year/start_year_YYYY.csv     （两列 author_id,start_year）
  3) by_start_year/group_counts.csv        （每年作者数统计）
  4) by_start_year/start_year_groups.json  （{YYYY: [author_id,...], ...} 大字典）
  5) by_start_year/conflicts.csv           （同一 author_id 出现多个 start_year 的冲突明细，如有）

直接在 PyCharm “Run” 即可，无需命令行参数。
"""

import os
import json
from collections import defaultdict, Counter

import pandas as pd

# ======== 你可以按需修改这两项 ========
IN_PATH = os.path.abspath(os.path.join("dblp_out", "exports", "author_start_years_1998_2012.jsonl"))
OUT_DIR = os.path.abspath(os.path.join("dblp_out", "exports", "by_start_year"))
# 如果只想保留 1998–2012，可启用过滤：
FILTER_MIN_YEAR = 1998
FILTER_MAX_YEAR = 2012
ENABLE_YEAR_FILTER = True
# ====================================


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_json_records(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到输入文件：{path}")

    ext = os.path.splitext(path)[1].lower()
    records = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"第 {i} 行 JSON 解析失败：{e}")
                records.append(obj)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                # 兼容 {records: [...]} 或 {year: [...]} 这类结构
                if "records" in obj and isinstance(obj["records"], list):
                    records = obj["records"]
                else:
                    # 尝试把 {year: [author_id,...]} 还原为记录
                    for k, v in obj.items():
                        try:
                            y = int(k)
                        except Exception:
                            continue
                        if isinstance(v, list):
                            for aid in v:
                                records.append({"author_id": str(aid), "start_year": y})
            elif isinstance(obj, list):
                records = obj
            else:
                raise ValueError("JSON 文件既不是数组也不是可识别的字典结构。")
    else:
        raise ValueError("仅支持 .jsonl 或 .json 输入。")

    if not records:
        raise RuntimeError("输入为空，没有可分组的记录。")
    return records


def normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    # 统一列名
    df.columns = [c.lower() for c in df.columns]
    if "start_year" not in df.columns and "year" in df.columns:
        df = df.rename(columns={"year": "start_year"})

    # 只保留两列
    need = ["author_id", "start_year"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列：{missing}；现有列：{list(df.columns)}")

    # 规范类型
    df = df[need].copy()
    df["author_id"] = df["author_id"].astype(str).str.strip()
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").astype("Int64")

    # 丢弃空值
    before = len(df)
    df = df.dropna(subset=["author_id", "start_year"])
    after = len(df)
    print(f"[CLEAN] 丢弃空 author_id/start_year 行：{before - after:,}；保留：{after:,}")

    # 可选按年份过滤
    if ENABLE_YEAR_FILTER:
        df = df[(df["start_year"] >= FILTER_MIN_YEAR) & (df["start_year"] <= FILTER_MAX_YEAR)]
        print(f"[FILTER] 保留 {FILTER_MIN_YEAR}–{FILTER_MAX_YEAR}：{len(df):,} 行")

    return df


def detect_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    # 同一 author_id 是否对应多个 start_year
    counts = df.groupby("author_id")["start_year"].nunique(dropna=True)
    conflict_ids = counts[counts > 1].index.tolist()
    if not conflict_ids:
        print("[CHECK] 未发现 author_id↔start_year 冲突。")
        return pd.DataFrame()

    conflicts = df[df["author_id"].isin(conflict_ids)].sort_values(["author_id", "start_year"])
    print(f"[CHECK] 发现冲突 author_id：{len(conflict_ids):,} 个；冲突明细行数：{len(conflicts):,}")
    return conflicts


def write_year_groups(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    # 写每年的 jsonl/csv
    groups = defaultdict(list)
    for _, row in df.iterrows():
        y = int(row["start_year"])
        groups[y].append({"author_id": row["author_id"], "start_year": y})

    # 汇总计数
    counts = []
    for y in sorted(groups.keys()):
        y_records = groups[y]
        # jsonl
        jsonl_path = os.path.join(out_dir, f"start_year_{y}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in y_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # csv
        csv_path = os.path.join(out_dir, f"start_year_{y}.csv")
        pd.DataFrame(y_records)[["author_id", "start_year"]].to_csv(csv_path, index=False, encoding="utf-8-sig")

        counts.append({"start_year": y, "author_count": len(y_records)})

    # group_counts
    count_df = pd.DataFrame(counts).sort_values("start_year")
    count_path = os.path.join(out_dir, "group_counts.csv")
    count_df.to_csv(count_path, index=False, encoding="utf-8-sig")

    # 大字典 {year: [author_id,...]}
    mapping = {int(y): [rec["author_id"] for rec in recs] for y, recs in groups.items()}
    mapping_path = os.path.join(out_dir, "start_year_groups.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[WRITE] 分组完成：{len(groups)} 个年份")
    print(f"  - 每年 jsonl/csv 已写入：{out_dir}")
    print(f"  - 计数汇总：{count_path}")
    print(f"  - 大字典：{mapping_path}")


def main():
    print(f"[INFO] 读取：{IN_PATH}")
    records = load_json_records(IN_PATH)
    print(f"[INFO] 读取记录数：{len(records):,}")

    df = pd.DataFrame.from_records(records)
    df = normalize_and_validate(df)

    # 冲突检测（同一 author_id 多个 start_year）
    conflicts = detect_conflicts(df)
    if not conflicts.empty:
        conflict_path = os.path.join(OUT_DIR, "conflicts.csv")
        ensure_dir(OUT_DIR)
        conflicts.to_csv(conflict_path, index=False, encoding="utf-8-sig")
        print(f"[WRITE] 冲突明细：{conflict_path}")

    # 去重（若同一 author_id 在同一年出现多次，只保留一次）
    before = len(df)
    df = df.drop_duplicates(subset=["author_id", "start_year"])
    after = len(df)
    if after < before:
        print(f"[CLEAN] 去重重复(author_id,start_year) 组合：{before - after:,} 行")

    # 按年份写文件
    write_year_groups(df, OUT_DIR)

    # 控制台简单预览
    print("\n====== 预览（前 5 年份）======")
    preview = (df.groupby("start_year")
                 .size().rename("author_count")
                 .reset_index()
                 .sort_values("start_year")
                 .head(5))
    print(preview.to_string(index=False))

    print("\n✅ 完成。")


if __name__ == "__main__":
    main()
# [INFO] 读取：C:\杨雨霏大学\Co-author Network and Leaders\timeWindow\dblp_out\exports\author_start_years_1998_2012.jsonl
# [INFO] 读取记录数：3,122,900
# [CLEAN] 丢弃空 author_id/start_year 行：0；保留：3,122,900
# [FILTER] 保留 1998–2012：3,122,900 行
# [CHECK] 未发现 author_id↔start_year 冲突。
# [WRITE] 分组完成：15 个年份
#   - 每年 jsonl/csv 已写入：C:\杨雨霏大学\Co-author Network and Leaders\timeWindow\dblp_out\exports\by_start_year
#   - 计数汇总：C:\杨雨霏大学\Co-author Network and Leaders\timeWindow\dblp_out\exports\by_start_year\group_counts.csv
#   - 大字典：C:\杨雨霏大学\Co-author Network and Leaders\timeWindow\dblp_out\exports\by_start_year\start_year_groups.json
#
# ====== 预览（前 5 年份）======
#  start_year  author_count
#        1998         91400
#        1999         94812
#        2000        109464
#        2001        117655
#        2002        133693
#
# ✅ 完成。
