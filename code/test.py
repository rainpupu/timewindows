# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os, re, json, glob
#
# ROOT = r"full_info_of_author"  # 如需改路径，在这里改
#
# def to_int(x):
#     if x is None or isinstance(x, bool):
#         return None
#     try:
#         return int(x)
#     except Exception:
#         try:
#             s = str(x).strip()
#             if not s:
#                 return None
#             # 兼容 "11.0" / Decimal 字符串
#             return int(float(s))
#         except Exception:
#             return None
#
# def main():
#     pattern = os.path.join(ROOT, "*_from_slice", "author_papers.jsonl")
#     files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"(\d{4})", os.path.dirname(p)).group(1)) if re.search(r"(\d{4})", os.path.dirname(p)) else 0)
#     if not files:
#         print("未找到任何 author_papers.jsonl")
#         return
#
#     results = {}  # year -> set(author_id)
#     for fp in files:
#         m = re.search(r"(\d{4})", os.path.basename(os.path.dirname(fp)))
#         if not m:
#             continue
#         year = int(m.group(1))
#         authors = set()
#         with open(fp, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     continue
#                 try:
#                     rec = json.loads(line)
#                 except Exception:
#                     continue
#                 k = to_int(rec.get("career_year"))
#                 if k != 11:
#                     continue
#                 aid = rec.get("author_id")
#                 if aid is None:
#                     continue
#                 authors.add(str(aid))
#         results[year] = authors
#
#     for y in sorted(results.keys()):
#         print(f"{y}: {len(results[y])}")
#
# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import os, re, json
# from decimal import Decimal
# from collections import Counter
# from tqdm.auto import tqdm
#
# # ================== 配置：修改这里 ==================
# START_FILE = r"dblp_out\exports\by_start_year\start_year_2012.jsonl"   # 2012届作者清单，每行：{"author_id": "...", "start_year": 2012}
# DBLP_SLICE = r"dblp_out\slice_1998_2022_nopreprint\DBLP_1998_2022_no_preprint.jsonl"
# # 如果你有“仅 2022 年”的切片，也可以直接填这个文件路径
# TARGET_YEAR = 2022
# PRINT_EXAMPLES = 10   # 打印前 N 个命中作者样例，便于核对（设为0可关闭）
# # ===================================================
#
# def to_int_year(y):
#     if y is None or isinstance(y, bool): return None
#     if isinstance(y, Decimal):
#         try: return int(y)
#         except Exception: return None
#     try:
#         return int(y)
#     except Exception:
#         try:
#             s = str(y).strip()
#             if not s: return None
#             return int(float(s))
#         except Exception:
#             return None
#
# def authors_list(rec):
#     a = rec.get("authors") or rec.get("v12_authors") or []
#     if isinstance(a, dict): a = [a]
#     return a if isinstance(a, list) else []
#
# def author_id_from(aobj):
#     if not isinstance(aobj, dict): return None
#     cand = aobj.get("id") or aobj.get("author_id")
#     if cand is None:
#         ids = aobj.get("ids")
#         if isinstance(ids, list) and ids:
#             cand = ids[0]
#     return str(cand) if cand is not None else None
#
# def detect_container_type(path):
#     # 判断切片是 NDJSON(一行一个JSON) 还是 顶层数组JSON
#     with open(path, "rb") as f:
#         head = f.read(4096)
#     s = head.lstrip(b"\xef\xbb\xbf \t\r\n")
#     return "array" if s[:1] == b"[" else "ndjson"
#
# def load_cohort_2012(path):
#     cohort = set()
#     total = 0
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             if not line.strip(): continue
#             try:
#                 j = json.loads(line)
#             except Exception:
#                 continue
#             aid = j.get("author_id")
#             sy  = j.get("start_year")
#             if aid is None:
#                 continue
#             total += 1
#             if to_int_year(sy) == 2012:
#                 cohort.add(str(aid))
#     return cohort, total
#
# def main():
#     # 1) 载入 2012 届作者集合
#     cohort_2012, total_in_file = load_cohort_2012(START_FILE)
#     print(f"[INFO] start_year_2012.jsonl 总行数(含非2012或坏行)：{total_in_file}")
#     print(f"[INFO] 识别为 2012 届作者数（unique）：{len(cohort_2012)}")
#
#     if not os.path.exists(DBLP_SLICE):
#         print(f"[ERROR] 未找到 DBLP 切片：{DBLP_SLICE}")
#         return
#
#     container = detect_container_type(DBLP_SLICE)
#     print(f"[INFO] 切片格式：{container}")
#
#     # 2) 扫描切片，统计 2022 年所有论文 & 2022 年独立作者数（全局） & 2012 届命中人数
#     active_2012 = set()
#     unique_authors_2022 = set()
#     papers_2022 = 0
#
#     if container == "ndjson":
#         # 按行读取，进度条按字节显示真实百分比
#         total_bytes = os.path.getsize(DBLP_SLICE)
#         with open(DBLP_SLICE, "rb") as f, \
#              tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"scan {os.path.basename(DBLP_SLICE)}") as pbar:
#             for raw in f:
#                 pbar.update(len(raw))
#                 line = raw.decode("utf-8", errors="ignore").strip()
#                 if not line:
#                     continue
#                 try:
#                     rec = json.loads(line)
#                 except Exception:
#                     continue
#
#                 y = to_int_year(rec.get("year"))
#                 if y != TARGET_YEAR:
#                     continue
#
#                 papers_2022 += 1
#                 auths = authors_list(rec)
#                 for a in auths:
#                     aid = author_id_from(a)
#                     if not aid:
#                         continue
#                     unique_authors_2022.add(aid)
#                     if aid in cohort_2012:
#                         active_2012.add(aid)
#     else:
#         # 顶层数组：使用 ijson 流式
#         try:
#             import ijson.backends.yajl2_c as ijson
#         except Exception:
#             import ijson
#         with open(DBLP_SLICE, "rb") as fb:
#             it = ijson.items(fb, "item")
#             for rec in tqdm(it, desc="scan array items", unit="obj"):
#                 y = to_int_year(rec.get("year"))
#                 if y != TARGET_YEAR:
#                     continue
#                 papers_2022 += 1
#                 auths = authors_list(rec)
#                 for a in auths:
#                     aid = author_id_from(a)
#                     if not aid:
#                         continue
#                     unique_authors_2022.add(aid)
#                     if aid in cohort_2012:
#                         active_2012.add(aid)
#
#     # 3) 打印结果与快速体检
#     print("\n======= RESULT (2012 cohort active in 2022) =======")
#     print(f"2012届作者总数（去重，来自 start_year_2012.jsonl）：{len(cohort_2012)}")
#     print(f"2022年论文记录数（来自切片，非预印本）：{papers_2022}")
#     print(f"2022年参与发文的作者数（全体，去重）：{len(unique_authors_2022)}")
#     print(f"2012届在2022年仍有发文的作者数（去重）：{len(active_2012)}")
#     if PRINT_EXAMPLES and active_2012:
#         print(f"样例（前{min(PRINT_EXAMPLES, len(active_2012))}个）:", ", ".join(list(sorted(active_2012))[:PRINT_EXAMPLES]))
#
#     # 4) 若结果异常小，提供几条诊断线索
#     if len(active_2012) < len(cohort_2012) * 0.05:  # <5% 看起来偏小
#         print("\n[WARN] 命中比例很低，可能原因与排查建议：")
#         print("  1) 切片是否只保留了“非预印本”？2022 年很多论文是 arXiv，若被剔除会显著降低命中；")
#         print("     可尝试用“未去预印本”的 2022 切片再跑一次做对比。")
#         print("  2) author_id 体系是否一致？start_year_2012.jsonl 与 切片中的 authors.id 必须同源（DBLP Author ID）。")
#         print("     若你的 start_year_2012.jsonl 来自其他库（如 MAG/AMiner），需要 ID 映射。")
#         print("  3) 切片是否真的包含 2022 年？上面已打印 2022 年论文条数与作者数，核对是否正常。")
#         print("  4) 2012 届文件中是否混入了非 2012 的行？脚本已经只取 start_year==2012，但可抽样检查源文件。")
#
# if __name__ == "__main__":
#     main()
import pandas as pd
df = pd.read_parquet(r"embeddings_gcn\author_rel\author_year_embeddings_gcn.parquet")
print(df.dtypes)
print(df.head(3))
