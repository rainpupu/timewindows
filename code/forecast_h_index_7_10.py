#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
export_hindex_preds_W7_10_jsonl.py
----------------------------------
不重训；从 best_W{W}.pt (W=7..10) 读取模型与训练时 cfg，复现数据拼接并导出 h@10 预测为 JSONL。

输入（按需改 CONFIG）：
- cohort_root: full_info_of_authors_new/{YYYY}_from_slice/
    - author_year_features_full.jsonl
    - authors_career11_hindex.jsonl
- embeddings_dir: embeddings_gcn/author_rel/*.parquet   (author_id + rel_year + e1..eK) 或 yearly_gcn/*.parquet
- ckpt_dir: best_W*.pt 所在目录（来自你的单窗口训练脚本 LSTM.py）

输出：
- {ckpt_dir}/preds_W{W}_train.jsonl
- {ckpt_dir}/preds_W{W}_valid.jsonl
- {ckpt_dir}/preds_W{W}_test.jsonl
- {ckpt_dir}/preds_schema.json   # 记录本脚本导出的字段名，以及 ckpt 内 val/test 指标的键名（用于核对参数命名一致性）
"""

import os, re, json, math
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Set, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 配置 =====================
@dataclass
class CONFIG:
    cohort_root: str   = r"full_info_of_authors_new"
    embeddings_dir: str= r"embeddings_gcn\author_rel"
    ckpt_dir: str      = r"runs\lstm_windows"       # 你的 best_W*.pt 放这里
    W_list:    tuple   = tuple(range(10, 11))        # 默认导出 W=7..10
    batch_size: int    = 256
    num_workers: int   = 0
    SAVE_ALL_AUTHORS: bool = False                  # False=仅导出有真值作者（与训练/评测一致）；True=含无真值作者

CFG = CONFIG()

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 通用 I/O =====================
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                yield json.loads(s)
            except:
                pass

def read_cohort_from_features(root: str) -> Dict[int, Set[str]]:
    cohort={}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"cohort_root 不存在：{root}")
    for name in os.listdir(root):
        m=re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        y=int(m.group(1))
        fp=os.path.join(root,name,"author_year_features_full.jsonl")
        if not os.path.isfile(fp): continue
        ids=set()
        for r in read_jsonl(fp):
            aid=r.get("author_id")
            if aid is not None: ids.add(str(aid))
        if ids: cohort[y]=ids
    return cohort

def build_start_year_map(cohort: Dict[int, Set[str]]) -> Dict[str,int]:
    return {aid:y for y,S in cohort.items() for aid in S}

def load_features_map(root: str) -> Dict[Tuple[str,int], Dict[str,Any]]:
    fmap={}
    for name in os.listdir(root):
        m=re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        fp=os.path.join(root,name,"author_year_features_full.jsonl")
        if not os.path.isfile(fp): continue
        for r in read_jsonl(fp):
            aid=r.get("author_id"); ya=r.get("year")
            if aid is None or ya is None: continue
            fmap[(str(aid), int(ya))]=r
    return fmap

def _try_float(x)->Optional[float]:
    try:
        v=float(x)
        if math.isfinite(v): return v
    except:
        pass
    return None

def extract_h_at10(d: Dict[str,Any]) -> Optional[float]:
    # 你的主字段名优先
    for k in ["h_index_career11","h_index_at_11","h_at_10","hindex_at_10","hindex11","h_at11",
              "career11_hindex","hindex_c11","hindex_at11","hindex"]:
        if k in d:
            v=_try_float(d[k])
            if v is not None: return v
    return None

def load_targets(root: str) -> Dict[str, Dict[str,Any]]:
    tmap={}
    for name in os.listdir(root):
        m=re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        y=int(m.group(1))
        fp=os.path.join(root,name,"authors_career11_hindex.jsonl")
        if not os.path.isfile(fp): continue
        for r in read_jsonl(fp):
            aid=r.get("author_id")
            if aid is None: continue
            aid=str(aid)
            h=extract_h_at10(r)
            rec={"author_id":aid,"start_year":y}
            if h is not None: rec["h_at_10"]=h
            if (aid not in tmap) or (y<tmap[aid]["start_year"]): tmap[aid]=rec
    return tmap

# ===================== Embeddings Loader =====================
def _detect_cols(cols: List[str]):
    id_col  = "author_id" if "author_id" in cols else ("node_id" if "node_id" in cols else None)
    key_col = "rel_year" if "rel_year" in cols else ("year_abs" if "year_abs" in cols else ("year" if "year" in cols else None))
    emb_cols= sorted([c for c in cols if len(c)>=2 and c[0].lower()=="e" and c[1:].isdigit()], key=lambda x:int(x[1:]))
    return id_col, key_col, emb_cols

def build_embedding_map_from_dir(path_dir: str,
                                 allowed_authors: Optional[Set[str]]=None,
                                 emb_dim_expected: Optional[int]=None):
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"embeddings_dir 不存在: {path_dir}")
    files=[os.path.join(path_dir,f) for f in os.listdir(path_dir) if f.lower().endswith(".parquet")]
    if not files: raise RuntimeError(f"未在 {path_dir} 找到 .parquet")
    emap={}; index_mode=None; emb_dim_infer=None
    for fp in sorted(files):
        df=pd.read_parquet(fp)
        cols=df.columns.tolist()
        id_col, key_col, emb_cols=_detect_cols(cols)
        if id_col is None or key_col is None or not emb_cols:
            raise ValueError(f"{fp} 缺 id/year/e* 列；实际：{cols}")
        mode = "rel" if key_col=="rel_year" else "abs"
        if index_mode is None: index_mode=mode
        elif index_mode!=mode:
            raise ValueError("同一目录混合了 rel_year 与 year/year_abs；请统一或分目录。")
        dim=len(emb_cols)
        if emb_dim_infer is None: emb_dim_infer=dim
        elif emb_dim_infer!=dim: raise ValueError(f"嵌入维度不一致：{emb_dim_infer} vs {dim} @ {fp}")
        use=df[[id_col,key_col]+emb_cols]
        if allowed_authors is not None:
            use=use[use[id_col].astype(str).isin(allowed_authors)]
        # 清洗年列为 int
        key_vals = pd.to_numeric(use[key_col], errors="coerce").round().astype("Int64")
        use = use.loc[key_vals.notna()].copy()
        use[key_col] = key_vals.loc[key_vals.notna()].astype(int)
        # 装入 map
        for row in use.itertuples(index=False, name=None):
            aid=str(row[0]); key=int(row[1]); vec=np.asarray(row[2:], dtype=np.float32)
            emap[(aid,key)]=vec
    if (emb_dim_expected is not None) and (emb_dim_infer!=emb_dim_expected):
        print(f"[WARN] cfg.emb_dim={emb_dim_expected} 与文件维度={emb_dim_infer} 不同，已以文件维度为准。")
    return emap, (index_mode or "rel"), emb_dim_infer

# ===================== Dataset / Model =====================
def log1p_safe(x):
    try: return float(math.log1p(max(0.0, float(x))))
    except: return 0.0

def to_float(x):
    try: return float(x)
    except: return 0.0

class AuthorSeqDataset(Dataset):
    def __init__(self, author_ids, start_year_map, fmap, emb_map, emb_index, targets, W, cfg_ns):
        self.ids=author_ids; self.sy=start_year_map; self.fmap=fmap
        self.emb=emb_map; self.eidx=emb_index; self.tars=targets; self.W=W; self.cfg=cfg_ns
        # —— 列名与训练保持一致（优先 ckpt['cfg']） —— #
        x_count = tuple(cfg_ns.get("x_count_cols", ("n_papers","n_papers_cum","venues_dedup_year",
                                                    "unique_coauthors_year","new_coauthors_year","cum_unique_coauthors",
                                                    "top_first","mid_first")))
        x_ratio = tuple(cfg_ns.get("x_ratio_cols", ("first_author_share","single_author_share","repeat_collab_ratio")))
        x_other = tuple(cfg_ns.get("x_other_cols", ("avg_team_size",)))
        self.add_missing = bool(cfg_ns.get("add_missing_flag", True))
        self.emb_dim = int(cfg_ns.get("emb_dim", 0))
        self.cols = (x_count, x_ratio, x_other)
        self.x_dim = len(x_count)+len(x_ratio)+len(x_other)+(1 if self.add_missing else 0)
        self.input_dim = self.x_dim + self.emb_dim

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        aid=self.ids[idx]; syear=int(self.sy[aid])
        y_reg=float(self.tars.get(aid,{}).get("h_at_10",-1.0))   # 真实 h@10
        steps=max(1, self.W if self.W>0 else 1)
        X=np.zeros((steps, self.input_dim), dtype=np.float32); seq_len=0
        x_count, x_ratio, x_other = self.cols
        for rel in range(steps):
            y_abs=syear+rel
            fr=self.fmap.get((aid, y_abs))
            feat=[]
            for c in x_count: feat.append(log1p_safe(fr.get(c) if fr else None))
            for c in x_ratio: feat.append(to_float(fr.get(c) if fr else None))
            for c in x_other: feat.append(to_float(fr.get(c) if fr else None))
            em = self.emb.get((aid, rel+1)) if self.eidx=="rel" else self.emb.get((aid, y_abs))
            miss=1.0 if (fr is None or em is None) else 0.0
            if self.add_missing: feat.append(miss)
            if em is None: em=np.zeros(self.emb_dim or 0, dtype=np.float32)
            vec=np.concatenate([np.asarray(feat,dtype=np.float32), em.astype(np.float32)], axis=0)
            X[rel,:]=vec
            if fr is not None: seq_len=rel+1
        seq_len=max(1, seq_len if self.W>0 else 1)
        return torch.from_numpy(X), torch.tensor(seq_len).long(), torch.tensor(y_reg).float(), {"author_id":aid,"start_year":syear}

def collate_fn(batch):
    Xs, Ls, Ys, Metas = zip(*batch)
    return torch.stack(Xs,0), torch.stack(Ls,0), torch.stack(Ys,0), list(Metas)

class LSTMFusion(nn.Module):
    def __init__(self,input_dim,cfg_ns):
        super().__init__()
        hidden = int(cfg_ns.get("lstm_hidden", 128))
        layers = int(cfg_ns.get("lstm_layers", 1))
        drop   = float(cfg_ns.get("dropout", 0.1)) if layers>1 else 0.0
        bidi   = bool(cfg_ns.get("bidirectional", False))
        self.lstm=nn.LSTM(input_dim,hidden,layers,batch_first=True,dropout=drop,bidirectional=bidi)
        hdim=hidden*(2 if bidi else 1)
        self.head=nn.Sequential(nn.Linear(hdim,64),nn.ReLU(),nn.Dropout(float(cfg_ns.get("dropout",0.1))),nn.Linear(64,1))
    def forward(self,X,L):
        if X.size(1)==1:
            _,(h,_) = self.lstm(X)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(X,L.cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)  # 预测的是 z^ = log1p(h)

# ===================== 预测 & 写盘 =====================
def infer_and_save_jsonl(model, dl, device, split_name, W, out_dir):
    """
    保存逐作者预测为 JSONL。
    字段名（保证与旧版一致）：
      author_id, start_year, split, W,
      h_index_career11_true, h_index_career11_pred, log_pred_z, residual
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, f"preds_W{W}_{split_name}.jsonl")
    n=0
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fw:
        for X,L,Y,M in dl:
            X=X.to(device); L=L.to(device)
            z = model(X,L).cpu().numpy()     # z^
            p = np.expm1(z)                  # -> h@10
            t = Y.numpy()
            for t_i, p_i, m_i, z_i in zip(t, p, M, z):
                rec = {
                    "author_id": m_i["author_id"],
                    "start_year": int(m_i["start_year"]),
                    "split": split_name,
                    "W": int(W),
                    "h_index_career11_true": (float(t_i) if t_i>=0 else None),
                    "h_index_career11_pred": float(p_i),
                    "log_pred_z": float(z_i),
                    "residual": (float(p_i)-float(t_i) if t_i>=0 else None),
                }
                fw.write(json.dumps(rec, ensure_ascii=False)+"\n")
                n+=1
    print(f"[save] {split_name} -> {out_path} (rows={n:,})")

# ===================== 主流程 =====================
def main():
    # 1) 收集 ckpt
    ckpts={}
    for W in CFG.W_list:
        p=os.path.join(CFG.ckpt_dir, f"best_W{W}.pt")
        if os.path.isfile(p):
            ckpts[W]=torch.load(p, map_location="cpu")
        else:
            print(f"[WARN] 缺少 {p}，跳过 W={W}")
    if not ckpts:
        raise RuntimeError(f"在 {CFG.ckpt_dir} 未发现 best_W7..10.pt")

    # 2) 复用 ckpt 的 cfg（保证列名/划分一致）
    any_cfg = list(ckpts.values())[0].get("cfg", {})
    train_years = tuple(any_cfg.get("train_years", tuple(range(1998,2005))))
    valid_years = tuple(any_cfg.get("valid_years", (2005,2006)))
    test_years  = tuple(any_cfg.get("test_years",  (2007,2008,2009)))

    # 3) cohort / features / targets
    cohort = read_cohort_from_features(CFG.cohort_root)
    start_year_map = build_start_year_map(cohort)
    fmap    = load_features_map(CFG.cohort_root)
    targets = load_targets(CFG.cohort_root)

    # 4) 分集作者（默认仅有真值作者，保证与训练/评测一致）
    def ids_of(years, must_have_target: bool):
        ids=set()
        for y in years:
            ids |= set(cohort.get(y, set()))
        if must_have_target:
            ids = {i for i in ids if (i in targets) and ("h_at_10" in targets[i])}
        return sorted(ids)

    ids_train = ids_of(train_years, must_have_target=not CFG.SAVE_ALL_AUTHORS)
    ids_valid = ids_of(valid_years, must_have_target=not CFG.SAVE_ALL_AUTHORS)
    ids_test  = ids_of(test_years,  must_have_target=not CFG.SAVE_ALL_AUTHORS)

    # 5) 嵌入（只加载预测涉及的作者）
    allowed = set(ids_train) | set(ids_valid) | set(ids_test)
    emb_map, emb_index, emb_dim = build_embedding_map_from_dir(
        CFG.embeddings_dir, allowed_authors=allowed, emb_dim_expected=any_cfg.get("emb_dim")
    )

    # 6) 导出每个 W 的预测
    #   顺便写一个 schema 文件，记录本脚本输出字段名 & ckpt metrics 的键名（用于参数名核对）
    schema_path = os.path.join(CFG.ckpt_dir, "preds_schema.json")
    schema = {
        "prediction_fields": [
            "author_id","start_year","split","W",
            "h_index_career11_true","h_index_career11_pred","log_pred_z","residual"
        ],
        "ckpt_metric_keys_example": list((list(ckpts.values())[0].get("val", {}) or {}).keys()) or
                                    ["RMSE","MAE","R2","NRMSE","Spearman","baseline_global_rmse","baseline_bycohort_rmse"],
        "emb_index_mode": emb_index,
        "emb_dim": int(emb_dim),
        "train_years": train_years,
        "valid_years": valid_years,
        "test_years":  test_years,
        "save_all_authors": CFG.SAVE_ALL_AUTHORS
    }
    with open(schema_path, "w", encoding="utf-8") as fs:
        json.dump(schema, fs, ensure_ascii=False, indent=2)
    print("[schema] ->", schema_path)

    for W, ck in sorted(ckpts.items()):
        print(f"\n========== EXPORT  W={W} ==========")
        cfg_ns  = ck.get("cfg", {})
        in_dim  = int(ck.get("input_dim", 0))

        # DataLoaders
        ds_tr=AuthorSeqDataset(ids_train, start_year_map, fmap, emb_map, emb_index, targets, W, cfg_ns)
        ds_va=AuthorSeqDataset(ids_valid, start_year_map, fmap, emb_map, emb_index, targets, W, cfg_ns)
        ds_te=AuthorSeqDataset(ids_test,  start_year_map, fmap, emb_map, emb_index, targets, W, cfg_ns)

        dl_tr=DataLoader(ds_tr, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)
        dl_va=DataLoader(ds_va, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)
        dl_te=DataLoader(ds_te, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)

        # 模型
        model=LSTMFusion(in_dim or ds_tr.input_dim, cfg_ns)
        model.load_state_dict(ck["state_dict"])
        model.to(DEVICE).eval()

        # 写盘（JSONL）
        infer_and_save_jsonl(model, dl_tr, DEVICE, "train", W, CFG.ckpt_dir)
        infer_and_save_jsonl(model, dl_va, DEVICE, "valid", W, CFG.ckpt_dir)
        infer_and_save_jsonl(model, dl_te, DEVICE, "test",  W, CFG.ckpt_dir)

    print("\n[OK] 全部完成。")

if __name__=="__main__":
    main()
