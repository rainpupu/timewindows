# 计算比较阈值上的差异
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import os, json, math, re
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Iterable
# from collections import defaultdict
# from tqdm.auto import tqdm
#
# # =================== 配置 ===================
#
# @dataclass
# class CFG:
#     ckpt_dir: str   = r"runs\lstm_windows"                 # 放 preds_W*.jsonl 的目录
#     cohort_root: str= r"full_info_of_authors_new"          # 放 career11_hindex_summary.json 的根目录
#     years: Tuple[int,int] = (1998, 2009)                   # 起止届（含）
#     W_SELECT: int  = 10                                    # 选择哪个窗口的预测
#     splits: Tuple[str,...] = ("train","valid","test")      # 参与计算的预测文件 split
#     round_mode: str = "floor"                              # 将连续预测变为整数: "round"|"floor"|"ceil"
#
# CFG = CFG()
#
# # =============== 工具函数 ===============
#
# def iter_jsonl(path: str) -> Iterable[dict]:
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             s = ln.strip()
#             if not s: continue
#             try:
#                 yield json.loads(s)
#             except:
#                 pass
#
# def intize(x: float, mode: str) -> int:
#     x = max(0.0, float(x))
#     if mode == "floor": return int(math.floor(x))
#     if mode == "ceil":  return int(math.ceil(x))
#     return int(round(x))  # default
#
# def load_real_summary_one(year: int) -> dict:
#     fp = os.path.join(CFG.cohort_root, f"{year}_from_slice", "career11_hindex_summary.json")
#     if not os.path.isfile(fp):
#         return {}
#     try:
#         with open(fp, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except:
#         return {}
#
# def kth_threshold_from_ints(int_values: List[int], pct: float) -> int:
#     """
#     给定整型 h 预测列表（每人一个），返回“前 pct%”的阈值。
#     做法：按降序排序，c = ceil(pct% * n)，阈值 = 排名第 c 的值（1-based）。
#     """
#     n = len(int_values)
#     if n == 0: return 0
#     c = max(1, math.ceil(pct/100.0 * n))
#     ints_sorted = sorted(int_values, reverse=True)
#     return int(ints_sorted[c-1])
#
# def top_ids_by_score(pairs: List[Tuple[str, float]], pct: float) -> List[str]:
#     """
#     pairs: (author_id, predicted_h_float)
#     返回前 pct% 的 author_id（按连续预测从高到低，ties 随序）
#     """
#     n = len(pairs)
#     if n == 0: return []
#     c = max(1, math.ceil(pct/100.0 * n))
#     pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
#     return [aid for aid, _ in pairs_sorted[:c]]
#
# # =============== 主逻辑 ===============
#
# def main():
#     y0, y1 = CFG.years
#     years = list(range(y0, y1+1))
#
#     # 1) 收集预测：preds_W{W}_{split}.jsonl
#     year_pred_float: Dict[int, List[Tuple[str, float]]] = {y: [] for y in years}
#     W = CFG.W_SELECT
#     missing_files = []
#
#     for split in CFG.splits:
#         fp = os.path.join(CFG.ckpt_dir, f"preds_W{W}_{split}.jsonl")
#         if not os.path.isfile(fp):
#             missing_files.append(fp)
#             continue
#         for r in tqdm(iter_jsonl(fp), desc=f"[read] W={W} {split}", unit="rec"):
#             y = int(r.get("start_year", -1))
#             if y < y0 or y > y1: continue
#             h_pred = r.get("h_index_career11_pred")
#             aid = r.get("author_id")
#             if aid is None or h_pred is None: continue
#             year_pred_float[y].append((str(aid), float(h_pred)))
#
#     if missing_files:
#         print("[WARN] 缺少以下预测文件（将忽略）:")
#         for p in missing_files: print("  -", p)
#
#     # 2) 读取真实阈值（逐年）
#     real_summ = {y: load_real_summary_one(y) for y in years}
#
#     # 3) 逐年计算阈值与 top 集合
#     out_summary = os.path.join(CFG.ckpt_dir, f"pred_thresholds_W{W}.jsonl")
#     out_topsets = os.path.join(CFG.ckpt_dir, f"pred_top_ids_W{W}_topsets.jsonl")
#
#     # 覆盖写
#     for fp in (out_summary, out_topsets):
#         try:
#             if os.path.exists(fp): os.remove(fp)
#         except:
#             pass
#
#     total_rows = 0
#     for y in years:
#         pairs = year_pred_float.get(y, [])
#         n = len(pairs)
#         if n == 0:
#             # 仍写一行 summary，方便对齐
#             rec = {
#                 "year": y,
#                 "n_pred": 0,
#                 "pred_top1_threshold": None,
#                 "pred_top10_threshold": None,
#                 "real_top1_threshold": real_summ.get(y, {}).get("top1_threshold"),
#                 "real_top10_threshold": real_summ.get(y, {}).get("top10_threshold"),
#                 "delta_top1": None,
#                 "delta_top10": None
#             }
#             with open(out_summary, "a", encoding="utf-8") as fw:
#                 fw.write(json.dumps(rec, ensure_ascii=False)+"\n")
#             continue
#
#         # 计算整数阈值
#         pred_ints = [intize(v, CFG.round_mode) for _, v in pairs]
#         t1  = kth_threshold_from_ints(pred_ints, 1.0)
#         t10 = kth_threshold_from_ints(pred_ints, 10.0)
#
#         # 计算 top 集合（按连续预测选前 k%，用于下游评估/标注）
#         top1_ids  = top_ids_by_score(pairs, 1.0)
#         top10_ids = top_ids_by_score(pairs, 10.0)
#
#         # 真实阈值
#         real_top1 = real_summ.get(y, {}).get("top1_threshold")
#         real_top10= real_summ.get(y, {}).get("top10_threshold")
#
#         rec = {
#             "year": y,
#             "n_pred": n,
#             "pred_top1_threshold": t1,
#             "pred_top10_threshold": t10,
#             "real_top1_threshold": real_top1,
#             "real_top10_threshold": real_top10,
#             "delta_top1": (None if real_top1 is None else (t1 - int(real_top1))),
#             "delta_top10": (None if real_top10 is None else (t10 - int(real_top10))),
#         }
#         with open(out_summary, "a", encoding="utf-8") as fw:
#             fw.write(json.dumps(rec, ensure_ascii=False)+"\n")
#
#         rec_ids = {
#             "year": y,
#             "n_pred": n,
#             "top1_count": max(1, math.ceil(0.01*n)),
#             "top10_count": max(1, math.ceil(0.10*n)),
#             "top1_ids": top1_ids,
#             "top10_ids": top10_ids
#         }
#         with open(out_topsets, "a", encoding="utf-8") as fw:
#             fw.write(json.dumps(rec_ids, ensure_ascii=False)+"\n")
#
#         total_rows += 1
#
#     print(f"[OK] 阈值对比写入: {out_summary}")
#     print(f"[OK] Top 集合写入: {out_topsets}")
#     print(f"[INFO] 年份覆盖: {total_rows} / {len(years)}，窗口 W={W}, splits={CFG.splits}, round_mode={CFG.round_mode}")
#
# if __name__ == "__main__":
#     main()

#比较top1%和10%具体人员正确预测个数
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
from collections import defaultdict
from tqdm.auto import tqdm

# =============== 配置 ===============
@dataclass
class CFG:
    ckpt_dir: str    = r"runs\lstm_windows"      # 预测文件目录：preds_W{W}_{train|valid|test}.jsonl
    cohort_root: str = r"full_info_of_authors_new"
    year_start: int  = 1998
    year_end: int    = 2009
    W_SELECTS: Tuple[int, ...] = (0,1,2,3,4,5,6,7,8,9,10)           # 要评估的窗口，可写 (7,8,9,10)
    splits: Tuple[str, ...]    = ("train","valid","test")  # 用哪些 split 的预测（一般全选）
    ks: Tuple[int, ...]        = (1, 10)         # 评估的百分位（Top-1%、Top-10%）
    out_prefix: str            = "equalcard_top_match"     # 输出文件名前缀

CFG = CFG()

# =============== I/O ===============
def iter_jsonl(fp: str) -> Iterable[dict]:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            try:
                yield json.loads(s)
            except:
                pass

def load_real_hindex(year: int) -> Dict[str, float]:
    """读取某届所有作者真实 h@10：authors_career11_hindex.jsonl -> {author_id: h}"""
    fp = os.path.join(CFG.cohort_root, f"{year}_from_slice", "authors_career11_hindex.jsonl")
    res={}
    if not os.path.isfile(fp): return res
    for r in iter_jsonl(fp):
        aid=r.get("author_id"); h=r.get("h_index_career11")
        if aid is None or h is None: continue
        try:
            res[str(aid)]=float(h)
        except: pass
    return res

def load_real_summary(year: int) -> dict:
    """读取某届阈值等统计：career11_hindex_summary.json"""
    fp = os.path.join(CFG.cohort_root, f"{year}_from_slice", "career11_hindex_summary.json")
    if not os.path.isfile(fp): return {}
    try:
        with open(fp,"r",encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def kth_threshold_from_real(h_map: Dict[str, float], pct: float) -> int:
    """当 summary 缺失时：按真实 h 降序第 ceil(pct*n) 个值作为整数阈值"""
    if not h_map: return 0
    vals=sorted([float(v) for v in h_map.values()], reverse=True)
    k=max(1, math.ceil(pct/100.0*len(vals)))
    return int(vals[k-1])

def real_top_set_by_threshold(h_map: Dict[str, float], thr: float) -> Set[str]:
    """真实 Top 集合：h >= thr"""
    return {aid for aid, hv in h_map.items() if hv is not None and float(hv) >= float(thr)}

def read_predictions_for_W(W:int) -> Dict[int, List[Tuple[str,float]]]:
    """
    汇总某个 W 的预测到: year -> [(author_id, pred_h_float)]
    假设不同 split 的年份不重叠（train/valid/test 按年切分），所以不会重复同届ID。
    """
    pred_by_year=defaultdict(list)
    missing=[]
    for split in CFG.splits:
        fp=os.path.join(CFG.ckpt_dir, f"preds_W{W}_{split}.jsonl")
        if not os.path.isfile(fp):
            missing.append(fp); continue
        for r in tqdm(iter_jsonl(fp), desc=f"[read preds] W={W} {split}", unit="rec"):
            y=r.get("start_year"); aid=r.get("author_id"); p=r.get("h_index_career11_pred")
            if y is None or aid is None or p is None: continue
            try:
                y=int(y); p=float(p)
            except:
                continue
            if y<CFG.year_start or y>CFG.year_end: continue
            pred_by_year[y].append((str(aid), p))
    if missing:
        print("[WARN] 缺少如下预测文件（忽略）:")
        for m in missing: print("  -", m)
    return pred_by_year

# =============== 排序/评估 ===============
def top_ids_by_pred(pairs: List[Tuple[str,float]], top_n: int) -> List[str]:
    """按预测分数/ID 稳定降序，取前 top_n 个 author_id"""
    if top_n<=0 or not pairs: return []
    pairs_sorted=sorted(pairs, key=lambda x:(x[1], x[0]), reverse=True)
    return [aid for aid,_ in pairs_sorted[:top_n]]

def evaluate_equal_cardinality(pairs: List[Tuple[str,float]],
                               real_top_set: Set[str]) -> Tuple[int,int,int,float,float,List[str]]:
    """
    等人数评估：预测侧取与真实Top同样的人数。
    返回: (n_pred_equal, n_real, n_hit, precision, recall, hit_ids)
    """
    n_real=len(real_top_set)
    n_pred_equal=n_real
    pred_equal_ids=set(top_ids_by_pred(pairs, n_pred_equal))
    hits=sorted(list(pred_equal_ids & real_top_set))
    n_hit=len(hits)
    prec = (n_hit/n_pred_equal) if n_pred_equal>0 else None
    rec  = (n_hit/n_real) if n_real>0 else None
    return n_pred_equal, n_real, n_hit, prec, rec, hits

def evaluate_fixed_percent(pairs: List[Tuple[str,float]],
                           real_top_set: Set[str], pct: float) -> Tuple[int,int,int,float,float,float]:
    """
    固定比例评估（对比用，不影响等人数主结论）
    返回: (n_pred_fixed, n_real, n_hit, precision, recall, cap)
    cap = n_pred_fixed / n_real 反映理论召回上限（并列会使 cap < 1）
    """
    n_all=len(pairs)
    n_real=len(real_top_set)
    n_pred_fixed=max(1, math.ceil(pct/100.0*n_all)) if n_all>0 else 0
    pred_fixed_ids=set(top_ids_by_pred(pairs, n_pred_fixed))
    n_hit=len(pred_fixed_ids & real_top_set)
    prec=(n_hit/n_pred_fixed) if n_pred_fixed>0 else None
    rec=(n_hit/n_real) if n_real>0 else None
    cap=(n_pred_fixed/n_real) if (n_real>0) else None
    return n_pred_fixed, n_real, n_hit, prec, rec, cap

# =============== 主流程 ===============
def main():
    years=list(range(CFG.year_start, CFG.year_end+1))

    for W in CFG.W_SELECTS:
        pred_by_year = read_predictions_for_W(W)

        out_detail=os.path.join(CFG.ckpt_dir, f"{CFG.out_prefix}_W{W}.jsonl")
        out_ids   =os.path.join(CFG.ckpt_dir, f"{CFG.out_prefix}_W{W}_ids.jsonl")
        # 覆盖旧文件
        for fp in (out_detail, out_ids):
            try:
                if os.path.exists(fp): os.remove(fp)
            except: pass

        # 汇总统计（按 k）
        totals = {k: {"n_pred_equal":0,"n_real":0,"n_hit_equal":0,
                      "n_pred_fixed":0,"n_hit_fixed":0} for k in CFG.ks}

        for y in years:
            pairs = pred_by_year.get(y, [])
            real_h = load_real_hindex(y)
            real_s = load_real_summary(y)

            rec_year = {"year": y, "W": W, "n_pred_total": len(pairs)}

            # 对每个 k（如 1%、10%）做两套评估
            for k in CFG.ks:
                # 真实阈值（优先 summary；没有则按真实排序算分位阈值）
                thr_key = f"top{k}_threshold"
                thr = real_s.get(thr_key)
                if thr is None:
                    thr = kth_threshold_from_real(real_h, float(k))
                real_top = real_top_set_by_threshold(real_h, thr)

                # 等人数评估（主结果）
                n_pred_equal, n_real, n_hit_equal, prec_equal, rec_equal, hit_ids = \
                    evaluate_equal_cardinality(pairs, real_top)

                # 固定比例评估（对比用）
                n_pred_fixed, n_real2, n_hit_fixed, prec_fixed, rec_fixed, cap = \
                    evaluate_fixed_percent(pairs, real_top, float(k))
                assert n_real2 == n_real

                # 写 detail（该年）
                rec_year.update({
                    f"top{k}_real_threshold": int(thr) if thr is not None else None,
                    f"top{k}_real_count": n_real,
                    f"top{k}_pred_equal_count": n_pred_equal,
                    f"top{k}_hit_equal_count": n_hit_equal,
                    f"top{k}_precision_equal": prec_equal,
                    f"top{k}_recall_equal": rec_equal,
                    # 对比：固定比例
                    f"top{k}_pred_fixed_count": n_pred_fixed,
                    f"top{k}_hit_fixed_count": n_hit_fixed,
                    f"top{k}_precision_fixed": prec_fixed,
                    f"top{k}_recall_fixed": rec_fixed,
                    f"top{k}_cap_fixed": cap,
                })
                # 写 ids（单独文件）
                with open(out_ids,"a",encoding="utf-8") as fw_ids:
                    fw_ids.write(json.dumps({
                        "year": y, "W": W, "k": k,
                        "hit_equal_ids": hit_ids  # 等人数下命中的ID
                    }, ensure_ascii=False)+"\n")

                # 累计
                totals[k]["n_pred_equal"] += n_pred_equal
                totals[k]["n_real"]       += n_real
                totals[k]["n_hit_equal"]  += n_hit_equal
                totals[k]["n_pred_fixed"] += n_pred_fixed
                totals[k]["n_hit_fixed"]  += n_hit_fixed

            # 年度行落盘
            with open(out_detail,"a",encoding="utf-8") as fw:
                fw.write(json.dumps(rec_year, ensure_ascii=False)+"\n")

        # 汇总行
        summary = {"W": W, "years": [CFG.year_start, CFG.year_end]}
        for k in CFG.ks:
            ne = totals[k]["n_pred_equal"]; nr = totals[k]["n_real"]; he = totals[k]["n_hit_equal"]
            nf = totals[k]["n_pred_fixed"]; hf = totals[k]["n_hit_fixed"]
            summary.update({
                f"top{k}_sum_real": nr,
                f"top{k}_sum_pred_equal": ne,
                f"top{k}_sum_hit_equal": he,
                f"top{k}_precision_equal": (he/ne if ne else None),
                f"top{k}_recall_equal":    (he/nr if nr else None),
                # 对比：固定比例
                f"top{k}_sum_pred_fixed": nf,
                f"top{k}_sum_hit_fixed":  hf,
                f"top{k}_precision_fixed": (hf/nf if nf else None),
                f"top{k}_recall_fixed":    (hf/nr if nr else None),
                f"top{k}_cap_fixed":       (nf/nr if nr else None)
            })
        # 汇总行也写入 detail 末尾
        with open(out_detail,"a",encoding="utf-8") as fw:
            fw.write(json.dumps({"SUMMARY": summary}, ensure_ascii=False)+"\n")

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[OK] 逐年详情 -> {out_detail}")
        print(f"[OK] 命中ID（等人数） -> {out_ids}")

if __name__ == "__main__":
    main()
