# -*- coding: utf-8 -*-
"""
export_start_year_map.py
用途：
- 读取 dblp_out/author_start_years_1998_2012.parquet
- 规范列名为 author_id, start_year（兼容 year→start_year）
- 导出：
  1) JSONL（每行一条，适合大文件）
  2) 预览 JSON（前 N 条，便捷查看）
  3) CSV（utf-8-sig，便于 Excel）
  4) 重复 author_id 明细（dups）
  5) start_year 分布

直接在 PyCharm 里 Run 即可，无需命令行参数。
"""

import os
import json
import pandas as pd

# ===== 你可以按需修改这两行 =====
IN_PATH  = os.path.abspath(os.path.join("dblp_out", "author_start_years_1998_2012.parquet"))
OUT_DIR  = os.path.abspath(os.path.join("dblp_out", "exports"))
PREVIEW_N = 200   # 预览 JSON 输出的前 N 条
# =================================

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def main():
    print(f"[INFO] 读取：{IN_PATH}")
    if not os.path.isfile(IN_PATH):
        print("❌ 找不到输入文件，请检查 IN_PATH。")
        return

    df = pd.read_parquet(IN_PATH)
    print(f"[INFO] 原始行数：{len(df):,}  原始列：{list(df.columns)}")

    # 统一列名小写
    df.columns = [c.lower() for c in df.columns]

    # 兼容 year → start_year
    if "start_year" not in df.columns and "year" in df.columns:
        df = df.rename(columns={"year": "start_year"})

    # 只保留这两列，多余列先不丢弃（为方便查看可选择保留或丢弃）
    keep_cols = []
    for c in ["author_id", "start_year"]:
        if c in df.columns:
            keep_cols.append(c)
    if not {"author_id", "start_year"} <= set(keep_cols):
        print(f"❌ 需要包含列 author_id 和 start_year/ year，当前列：{list(df.columns)}")
        return

    # 这里既导出“完整列版”，也导出“精简两列版”
    df_full = df.copy()
    df_two  = df[["author_id", "start_year"]].copy()

    # 规范类型
    df_two["author_id"] = df_two["author_id"].astype(str).str.strip()
    df_two["start_year"] = pd.to_numeric(df_two["start_year"], errors="coerce").astype("Int64")

    # 去掉空 author_id 或空 start_year
    before = len(df_two)
    df_two = df_two.dropna(subset=["author_id", "start_year"])
    after = len(df_two)
    print(f"[CLEAN] 丢弃空 author_id/start_year 行：{before - after:,}  保留：{after:,}")

    # 概览统计
    print(f"[STATS] 唯一 author_id 数：{df_two['author_id'].nunique():,}")
    print(f"[STATS] start_year 范围：{int(df_two['start_year'].min()) if len(df_two)>0 else 'N/A'}"
          f" — {int(df_two['start_year'].max()) if len(df_two)>0 else 'N/A'}")

    # 输出路径
    ensure_dir(OUT_DIR)
    base = os.path.splitext(os.path.basename(IN_PATH))[0]
    jsonl_path   = os.path.join(OUT_DIR, f"{base}.jsonl")              # 精简两列版 JSONL
    json_preview = os.path.join(OUT_DIR, f"{base}.preview.json")       # 预览 JSON（前 N 条）
    csv_path     = os.path.join(OUT_DIR, f"{base}.csv")                # 精简两列版 CSV
    csv_full     = os.path.join(OUT_DIR, f"{base}.full.csv")           # 完整列 CSV（便于排查）
    dups_path    = os.path.join(OUT_DIR, f"{base}.dup_author_id.csv")  # 重复 author_id 明细
    dist_path    = os.path.join(OUT_DIR, f"{base}.start_year_dist.csv")# start_year 分布

    # 1) JSONL（两列精简）
    print(f"[WRITE] JSONL：{jsonl_path}")
    df_two.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    # 2) 预览 JSON（前 N 条，两列精简）
    print(f"[WRITE] 预览 JSON（前 {PREVIEW_N} 条）：{json_preview}")
    preview_records = df_two.head(PREVIEW_N).to_dict(orient="records")
    with open(json_preview, "w", encoding="utf-8") as f:
        json.dump(preview_records, f, ensure_ascii=False, indent=2)

    # 3) CSV（两列精简，utf-8-sig 便于 Excel）
    print(f"[WRITE] CSV（两列精简）：{csv_path}")
    df_two.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 4) 完整列 CSV（方便肉眼排查）
    print(f"[WRITE] CSV（完整列）：{csv_full}")
    df_full.to_csv(csv_full, index=False, encoding="utf-8-sig")

    # 5) 重复 author_id 明细
    dups_mask = df_two.duplicated(subset=["author_id"], keep=False)
    if dups_mask.any():
        dups_df = df_two.loc[dups_mask].sort_values(["author_id", "start_year"])
        dups_df.to_csv(dups_path, index=False, encoding="utf-8-sig")
        print(f"[WRITE] 发现重复 author_id：{dups_df['author_id'].nunique():,} 个，已输出 {dups_path}")
    else:
        print("[OK] 没有重复 author_id。")

    # 6) start_year 分布
    dist = (df_two["start_year"]
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("start_year")
            .reset_index(name="count"))
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"[WRITE] start_year 分布：{dist_path}")

    print("\n✅ 完成。主要输出：")
    print(" -", jsonl_path)
    print(" -", json_preview)
    print(" -", csv_path)
    print(" -", csv_full)
    if os.path.exists(dups_path): print(" -", dups_path)
    print(" -", dist_path)

if __name__ == "__main__":
    main()
