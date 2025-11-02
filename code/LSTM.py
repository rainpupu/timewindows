#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_lstm_single_window.py  —— 只训练【一个】时间窗口切片（W）
- 用于多台电脑/多进程并行：每台机器各跑一个 W（如 8、9、10），互不覆盖输出
- 不依赖 author_papers.jsonl；cohort/targets/features 均来自 full_info_of_authors_new\{YYYY}_from_slice\*
- 兼容两类 GCN 嵌入目录：
    1) embeddings_gcn/author_rel/author_year_embeddings_gcn.parquet   (author_id + rel_year + e1..eK)
    2) embeddings_gcn/yearly_gcn/emb_YYYY.parquet                      (node_id  + year_abs + e1..eK)
- 自动识别列名（author_id/node_id, rel_year/year_abs），仅加载 Train/Valid/Test 出现的作者，省内存
- 评测：RMSE, MAE, R², NRMSE, Spearman + 全局/分届均值 RMSE 基线
- 训练参数与原 windows.py 对齐：6 个 epoch、无早停（patience 设为极大值）
- 自动选择 GPU（如可用），也可通过 --device 指定 cpu/cuda
- 默认将输出写入 runs/lstm_windows/W{W}/，避免并行互相覆盖
用法示例：
  # 单机仅跑 W=8，强制用 GPU
  python train_lstm_single_window.py --W 8 --device cuda
  # 多机/多进程并行：分别在不同机器上执行（可改线程数与输出目录）
  python train_lstm_single_window.py --W 8 --threads 4 --save_dir runs\lstm_windows
  python train_lstm_single_window.py --W 9 --threads 4 --save_dir runs\lstm_windows
  python train_lstm_single_window.py --W 10 --threads 4 --save_dir runs\lstm_windows
"""

import os, re, json, math, random, sys, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config（与 windows.py 训练参数对齐）
# =========================
@dataclass
class Config:
    cohort_root: str = r"full_info_of_authors_new"
    embeddings_dir: str = r"embeddings_gcn\author_rel"   # 可改成 embeddings_gcn\yearly_gcn
    save_dir: str = r"runs\lstm_windows"                 # 最终会自动拼接子目录 W{W}

    # 注意：本脚本只接受单个 W；这里不使用 windows 列表
    emb_dim: Optional[int] = None  # None=自动识别

    # LSTM 结构
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.1

    # Train（与 windows.py 一致：6 epoch、无早停）
    batch_size: int = 96
    lr: float = 7e-4
    weight_decay: float = 1e-4
    max_epochs: int = 6
    patience: int = 999           # 等价禁用早停
    num_workers: int = 0
    seed: int = 2025

    # 年度特征列
    x_count_cols: Tuple[str, ...] = (
        "n_papers", "n_papers_cum",
        "venues_dedup_year",
        "unique_coauthors_year", "new_coauthors_year", "cum_unique_coauthors",
        "top_first", "mid_first",
    )
    x_ratio_cols: Tuple[str, ...] = ("first_author_share", "single_author_share", "repeat_collab_ratio")
    x_other_cols: Tuple[str, ...] = ("avg_team_size",)
    add_missing_flag: bool = True

    # 按届划分
    train_years: Tuple[int, ...] = tuple(range(1998, 2005))
    valid_years: Tuple[int, ...] = (2005, 2006)
    test_years:  Tuple[int, ...] = (2007, 2008, 2009)

cfg = Config()

# ---------------
# CLI
# ---------------
parser = argparse.ArgumentParser()
parser.add_argument("--W", type=int, required=True, help="单个时间窗口切片（0..10 之间的整数）")
parser.add_argument("--save_dir", type=str, default=None, help="根输出目录；程序会自动在其下创建 W{W} 子目录")
parser.add_argument("--threads", type=int, default=None, help="限制本进程的 PyTorch 计算线程数")
parser.add_argument("--device", type=str, default=None, choices=["cpu","cuda"], help="指定 cpu/cuda；不写则自动检测")
parser.add_argument("--cohort_root", type=str, default=None, help="覆盖 cohort 根目录")
parser.add_argument("--embeddings_dir", type=str, default=None, help="覆盖嵌入目录（author_rel 或 yearly_gcn）")
args = parser.parse_args()

# 应用覆盖
if args.save_dir:
    cfg.save_dir = args.save_dir
if args.cohort_root:
    cfg.cohort_root = args.cohort_root
if args.embeddings_dir:
    cfg.embeddings_dir = args.embeddings_dir

# 线程限制（便于多机/多进程并行时避免互抢）
if args.threads:
    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, args.threads // 2))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.threads))

# 设备选择
_auto_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.device == "cpu":
    device = torch.device("cpu")
elif args.device == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = _auto_device

# 基本校验
if not (0 <= args.W <= 10):
    raise SystemExit(f"--W 必须在 [0,10]，收到 {args.W}")

# 最终运行目录（防止并行覆盖）
RUN_DIR = os.path.join(cfg.save_dir, f"W{args.W}")
os.makedirs(RUN_DIR, exist_ok=True)

# Repro
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(cfg.seed)

# =========================
# I/O + Cohort/Targets
# =========================
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try: yield json.loads(s)
                except: pass

def read_cohort_from_targets(root: str) -> Dict[int, Set[str]]:
    cohort = {}
    if not os.path.isdir(root):
        return cohort
    for name in os.listdir(root):
        m = re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        y = int(m.group(1))
        fp = os.path.join(root, name, "authors_career11_hindex.jsonl")
        if not os.path.isfile(fp): continue
        S = set()
        for r in read_jsonl(fp):
            aid = r.get("author_id")
            if aid is not None: S.add(str(aid))
        if S: cohort[y] = S
    return cohort

def read_cohort_from_features(root: str) -> Dict[int, Set[str]]:
    cohort = {}
    if not os.path.isdir(root):
        return cohort
    for name in os.listdir(root):
        m = re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        y = int(m.group(1))
        fp = os.path.join(root, name, "author_year_features_full.jsonl")
        if not os.path.isfile(fp): continue
        S = set()
        for r in read_jsonl(fp):
            aid = r.get("author_id")
            if aid is not None: S.add(str(aid))
        if S: cohort[y] = S
    return cohort

def build_start_year_map_from_cohort(cohort: Dict[int, Set[str]]) -> Dict[str, int]:
    return {aid: y for y, S in cohort.items() for aid in S}

def load_features_map_from_cohort(root: str) -> Dict[Tuple[str,int], Dict[str,Any]]:
    fmap = {}
    if not os.path.isdir(root):
        return fmap
    for name in os.listdir(root):
        m = re.fullmatch(r"(\d{4})_from_slice", name)
        if not m: continue
        fp = os.path.join(root, name, "author_year_features_full.jsonl")
        if not os.path.isfile(fp): continue
        for r in read_jsonl(fp):
            try:
                aid = str(r["author_id"]); ya = int(r["year"])
                fmap[(aid, ya)] = r
            except: pass
    return fmap

def _try_float(x)->Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v): return v
    except: pass
    return None

def extract_h_at10(rec: Dict[str,Any]) -> Optional[float]:
    for k in ["h_index_career11","h_at_10","hindex_at_10","hindex11","h_at11",
              "career11_hindex","hindex_c11","h_index_at_11","hindex_at11","hindex"]:
        if k in rec:
            v = _try_float(rec[k])
            if v is not None: return v
    return None

def load_targets_from_cohort(root: str) -> Dict[str, Dict[str,Any]]:
    tmap = {}
    if not os.path.isdir(root):
        return tmap
    for name in os.listdir(root):
        m = re.fullmatch(r"(\d{4})_from_slice",name)
        if not m: continue
        y = int(m.group(1))
        fp = os.path.join(root,name,"authors_career11_hindex.jsonl")
        if not os.path.isfile(fp): continue
        for r in read_jsonl(fp):
            aid = r.get("author_id")
            if aid is None: continue
            aid = str(aid)
            h = extract_h_at10(r)
            rec = {"author_id": aid, "start_year": y}
            if h is not None: rec["h_at_10"] = h
            if (aid not in tmap) or (y < tmap[aid]["start_year"]): tmap[aid] = rec
    return tmap

# =========================
# Embeddings Loader (Parquet)
# =========================
def _detect_cols(cols: List[str]):
    id_col  = "author_id" if "author_id" in cols else ("node_id" if "node_id" in cols else None)
    key_col = "rel_year" if "rel_year" in cols else ("year_abs" if "year_abs" in cols else ("year" if "year" in cols else None))
    emb_cols= sorted([c for c in cols if len(c)>=2 and c[0]=="e" and c[1:].isdigit()], key=lambda x:int(x[1:]))
    return id_col, key_col, emb_cols

def build_embedding_map_from_dir(path_dir: str,
                                 allowed_authors: Optional[Set[str]]=None,
                                 emb_dim_expected: Optional[int]=None):
    if os.path.isfile(path_dir) and path_dir.lower().endswith(".parquet"):
        files=[path_dir]
    else:
        if not os.path.isdir(path_dir):
            raise FileNotFoundError(f"embeddings_dir 不存在: {path_dir}")
        files=[os.path.join(path_dir,f) for f in os.listdir(path_dir) if f.lower().endswith(".parquet")]
        rel_file=[f for f in files if os.path.basename(f).startswith("author_year_embeddings_gcn")]
        if rel_file:
            files=rel_file  # 只用总表
    if not files:
        raise RuntimeError(f"未在 {path_dir} 找到 .parquet")

    emap={}; index_mode=None; emb_dim_inferred=None
    for fp in files:
        df=pd.read_parquet(fp)
        cols=df.columns.tolist()
        id_col, key_col, emb_cols=_detect_cols(cols)
        if id_col is None or key_col is None or not emb_cols:
            raise ValueError(f"{fp} 缺列（需 id/year(e.g. rel_year/year_abs)/e1..eK），实际：{cols}")

        cur_mode = "rel" if key_col=="rel_year" else "abs"
        if index_mode is None: index_mode=cur_mode
        elif index_mode!=cur_mode:
            raise ValueError(f"同一加载批次出现混合索引模式：{index_mode} vs {cur_mode} @ {fp}")

        cur_dim=len(emb_cols)
        if emb_dim_inferred is None: emb_dim_inferred=cur_dim
        elif emb_dim_inferred!=cur_dim:
            raise ValueError(f"不同文件的嵌入维度不一致：{emb_dim_inferred} vs {cur_dim} @ {fp}")

        use = df[[id_col, key_col] + emb_cols]
        if allowed_authors is not None:
            use = use[use[id_col].astype(str).isin(allowed_authors)]
        for row in use.itertuples(index=False, name=None):
            aid=str(row[0]); key=int(row[1]); vec=np.asarray(row[2:], dtype=np.float16)
            emap[(aid, key)] = vec

    if (emb_dim_expected is not None) and (emb_dim_inferred != emb_dim_expected):
        print(f"[WARN] 嵌入维度={emb_dim_inferred} 与期望={emb_dim_expected} 不同，已以文件为准。")
    return emap, (index_mode or "rel"), emb_dim_inferred

# =========================
# Dataset / Model
# =========================
def log1p_safe(x):
    try: return float(math.log1p(max(0.0, float(x))))
    except: return 0.0

def to_float(x):
    try: return float(x)
    except: return 0.0

class AuthorSeqDataset(Dataset):
    def __init__(self, author_ids, start_year_map, fmap, emb_map, emb_index, targets, W, cfg:Config):
        self.ids=author_ids; self.sy=start_year_map; self.fmap=fmap
        self.emb=emb_map; self.eidx=emb_index; self.tars=targets; self.W=W; self.cfg=cfg
        self.x_dim = len(cfg.x_count_cols)+len(cfg.x_ratio_cols)+len(cfg.x_other_cols)+(1 if cfg.add_missing_flag else 0)
        self.input_dim=self.x_dim+(cfg.emb_dim or 0)
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        aid=self.ids[idx]; syear=int(self.sy[aid])
        y_reg=float(self.tars.get(aid,{}).get("h_at_10",-1.0))
        steps=max(1, self.W if self.W>0 else 1)
        X=np.zeros((steps, self.input_dim), dtype=np.float32); seq_len=0
        for rel in range(steps):
            y_abs=syear+rel
            fr=self.fmap.get((aid, y_abs))
            feat=[]
            for c in self.cfg.x_count_cols: feat.append(log1p_safe(fr.get(c) if fr else None))
            for c in self.cfg.x_ratio_cols: feat.append(to_float(fr.get(c) if fr else None))
            for c in self.cfg.x_other_cols: feat.append(to_float(fr.get(c) if fr else None))
            em = self.emb.get((aid, rel+1)) if self.eidx=="rel" else self.emb.get((aid, y_abs))
            miss=1.0 if (fr is None or em is None) else 0.0
            if self.cfg.add_missing_flag: feat.append(miss)
            if em is None: em=np.zeros(self.cfg.emb_dim or 0, dtype=np.float16)
            vec=np.concatenate([np.asarray(feat,dtype=np.float32), em.astype(np.float32)], axis=0)
            X[rel,:]=vec
            if fr is not None: seq_len=rel+1
        seq_len=max(1, seq_len if self.W>0 else 1)
        return torch.from_numpy(X), torch.tensor(seq_len).long(), torch.tensor(y_reg).float(), {"author_id":aid,"start_year":syear}

def collate_fn(batch):
    Xs, Ls, Ys, Metas = zip(*batch)
    return torch.stack(Xs,0), torch.stack(Ls,0), torch.stack(Ys,0), list(Metas)

class LSTMFusion(nn.Module):
    def __init__(self,input_dim,cfg:Config):
        super().__init__()
        self.lstm=nn.LSTM(input_dim,cfg.lstm_hidden,cfg.lstm_layers,batch_first=True,
                          dropout=(cfg.dropout if cfg.lstm_layers>1 else 0.0),
                          bidirectional=cfg.bidirectional)
        hdim=cfg.lstm_hidden*(2 if cfg.bidirectional else 1)
        self.head=nn.Sequential(nn.Linear(hdim,64),nn.ReLU(),nn.Dropout(cfg.dropout),nn.Linear(64,1))
    def forward(self,X,L):
        if X.size(1)==1:
            _,(h,_) = self.lstm(X)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(X,L.cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)

# =========================
# Metrics
# =========================
def _ranks_with_ties(x: np.ndarray) -> np.ndarray:
    order=np.argsort(x, kind="mergesort")
    ranks=np.empty_like(order,dtype=np.float64); ranks[order]=np.arange(1,len(x)+1,dtype=np.float64)
    i=0
    while i<len(x):
        j=i
        while j+1<len(x) and x[order[j+1]]==x[order[i]]: j+=1
        if j>i:
            avg=(ranks[order[i]]+ranks[order[j]])/2.0
            ranks[order[i:j+1]]=avg
        i=j+1
    return ranks

def spearmanr_np(y_true, y_pred):
    y=np.asarray(y_true); p=np.asarray(y_pred)
    m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    if y.size<2: return float("nan")
    ry=_ranks_with_ties(y); rp=_ranks_with_ties(p)
    y0=ry-ry.mean(); p0=rp-rp.mean()
    den=np.sqrt(np.sum(y0**2)*np.sum(p0**2))
    if den==0: return float("nan")
    return float(np.sum(y0*p0)/den)

def eval_full_metrics(model, dl, device):
    model.eval(); ys=[]; ps=[]; starts=[]
    with torch.no_grad():
        for X,L,Y,M in dl:
            X=X.to(device); L=L.to(device)
            z=model(X,L); pred=torch.expm1(z).detach().cpu().numpy(); true=Y.numpy()
            for t,p,m in zip(true,pred,M):
                if t>=0: ys.append(float(t)); ps.append(float(p)); starts.append(int(m["start_year"]))
    ys=np.asarray(ys); ps=np.asarray(ps); starts=np.asarray(starts)
    out={"n":int(ys.size)}
    if ys.size==0:
        for k in ["RMSE","MAE","R2","NRMSE","Spearman","baseline_global_rmse","baseline_bycohort_rmse"]: out[k]=float("nan")
        return out
    rmse=float(np.sqrt(np.mean((ps-ys)**2)))
    mae =float(np.mean(np.abs(ps-ys)))
    var =float(np.var(ys, ddof=0))
    r2  =float(1.0 - np.mean((ps-ys)**2)/(var if var>0 else np.inf))
    nrmse=float(rmse/(np.sqrt(var) if var>0 else np.nan))
    spr =spearmanr_np(ys,ps)
    # baselines
    y_mean=float(np.mean(ys))
    rmse_global=float(np.sqrt(np.mean((y_mean-ys)**2)))
    preds=np.zeros_like(ys);
    for y in np.unique(starts):
        m=(starts==y)
        if m.any(): preds[m]=np.mean(ys[m])
    rmse_bycohort=float(np.sqrt(np.mean((preds-ys)**2)))
    out.update(dict(RMSE=rmse,MAE=mae,R2=r2,NRMSE=nrmse,Spearman=spr,
                    baseline_global_rmse=rmse_global, baseline_bycohort_rmse=rmse_bycohort))
    return out

# =========================
# Train
# =========================
def train_one_W(W, train_ids, valid_ids, test_ids, start_year_map, fmap, emb_map, emb_index, targets, cfg, device):
    ds_tr=AuthorSeqDataset(train_ids,start_year_map,fmap,emb_map,emb_index,targets,W,cfg)
    ds_va=AuthorSeqDataset(valid_ids,start_year_map,fmap,emb_map,emb_index,targets,W,cfg)
    ds_te=AuthorSeqDataset(test_ids, start_year_map,fmap,emb_map,emb_index,targets,W,cfg)

    dl_tr=DataLoader(ds_tr,batch_size=cfg.batch_size,shuffle=True, num_workers=cfg.num_workers,collate_fn=collate_fn)
    dl_va=DataLoader(ds_va,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate_fn)
    dl_te=DataLoader(ds_te,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate_fn)

    model=LSTMFusion(ds_tr.input_dim,cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    mse=nn.MSELoss()

    best=float('inf'); bad=0; best_ep=-1
    os.makedirs(RUN_DIR, exist_ok=True)

    for ep in range(1, cfg.max_epochs+1):
        model.train(); tot=0.0
        for X,L,Y,_ in dl_tr:
            X=X.to(device); L=L.to(device); Y=Y.to(device)
            opt.zero_grad(); z=model(X,L)
            m=(Y>=0); loss=mse(z[m], torch.log1p(Y[m])) if m.any() else torch.tensor(0.0,device=device)
            loss.backward(); opt.step(); tot+=float(loss.item())
        tr_loss=tot/max(1,len(dl_tr))
        va=eval_full_metrics(model,dl_va,device); te=eval_full_metrics(model,dl_te,device)
        print(f"[W={W}] Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_RMSE={va['RMSE']:.4f} | test_RMSE={te['RMSE']:.4f} | val_Spearman={va['Spearman']:.4f}")
        if va["RMSE"]<best:
            best=va["RMSE"]; best_ep=ep; bad=0
            torch.save({"cfg":asdict(cfg),"W":W,"epoch":ep,"state_dict":model.state_dict(),"input_dim":ds_tr.input_dim,"val":va,"test":te},
                       os.path.join(RUN_DIR, f"best_W{W}.pt"))
        else:
            bad+=1
            if bad>=cfg.patience:
                print(f"[W={W}] Early stop at epoch {ep} (best@{best_ep})"); break

    # 最终一轮指标（记录在 CSV 中）
    final_va=eval_full_metrics(model,dl_va,device); final_te=eval_full_metrics(model,dl_te,device)
    out={"W":W,"best_val_rmse":best,"best_epoch":best_ep}
    for k,v in final_te.items(): out[f"test_{k}"]=v
    for k,v in final_va.items(): out[f"val_{k}"]=v
    return out

# =========================
# Main
# =========================
def main():
    print("==> Device:", device)
    print("==> Threads:", torch.get_num_threads())
    print("==> Cohort root:", cfg.cohort_root)
    print("==> Embeddings dir:", cfg.embeddings_dir)
    print("==> Run dir:", RUN_DIR)
    W = args.W

    # 1) Cohort（优先从目标）
    cohort = read_cohort_from_targets(cfg.cohort_root)
    if not cohort:
        print("[WARN] 目标缺失；退回用年度特征构建 cohort")
        cohort = read_cohort_from_features(cfg.cohort_root)
    if not cohort: raise RuntimeError("未找到任何届的作者（检查 cohort_root 路径）")

    start_year_map = build_start_year_map_from_cohort(cohort)
    print("Cohort sizes:", {y:len(S) for y,S in sorted(cohort.items())})

    # 2) 特征
    fmap = load_features_map_from_cohort(cfg.cohort_root)
    print("features rows:", len(fmap))

    # 3) 目标
    targets = load_targets_from_cohort(cfg.cohort_root)

    # 4) 划分 + 仅保留有目标的作者
    def filter_ids(years):
        return sorted({aid for y in years for aid in cohort.get(y,set()) if (aid in targets) and ("h_at_10" in targets[aid])})
    train_ids=filter_ids(cfg.train_years); valid_ids=filter_ids(cfg.valid_years); test_ids=filter_ids(cfg.test_years)
    if not train_ids or not valid_ids or not test_ids:
        print("[ERROR] Train/Valid/Test 中有空集，请检查目标覆盖率"); sys.exit(1)
    print(f"Authors with targets -> Train {len(train_ids):,} | Valid {len(valid_ids):,} | Test {len(test_ids):,}")

    # 5) 嵌入（仅加载需要的作者，自动识别 rel/year_abs）
    allowed=set(train_ids) | set(valid_ids) | set(test_ids)
    emb_map, emb_index, emb_dim = build_embedding_map_from_dir(cfg.embeddings_dir, allowed_authors=allowed, emb_dim_expected=cfg.emb_dim)
    cfg.emb_dim = emb_dim
    print(f"embeddings -> entries={len(emb_map):,}, mode={emb_index}, dim={cfg.emb_dim}")

    # 6) 仅训练单个窗口
    res = train_one_W(W, train_ids, valid_ids, test_ids, start_year_map, fmap, emb_map, emb_index, targets, cfg, device)

    # 7) 保存该切片的独立 CSV（防止并行覆盖）
    os.makedirs(RUN_DIR, exist_ok=True)
    out_csv = os.path.join(RUN_DIR, f"metrics_W{W}.csv")
    pd.DataFrame([res]).to_csv(out_csv, index=False, encoding="utf-8")
    print("Done. Metrics ->", out_csv)

if __name__=="__main__":
    main()
