import os, json, re, math, hashlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

import pyarrow as pa, pyarrow.parquet as pq


# =============== 配置 ===============
SNAPSHOT_DIR  = r"graphs/snapshots"      # 边/节点快照目录
AUTHORS_DIR   = r"full_info_of_authors_new"  # 1998_from_slice/*.jsonl ... 2009_from_slice/*.jsonl
EMB_OUT_DIR   = r"embeddings_gcn"

YEAR_START, YEAR_END = 1998, 2018
TARGET_START_MIN, TARGET_START_MAX = 1998, 2009

# 模型/训练
IN_DIM       = 1 + 1 + 1 + 8   # [logdeg, logwdeg, is_target, hash8]
HID_DIM      = 64
EMB_DIM      = 128
DROPOUT      = 0.2
LR           = 1e-3
EPOCHS_PER_YEAR = 2            # 每年训练迭代轮数（先小点试跑，可增大到 5）
NEG_SAMPLING_RATIO = 1.0       # 负样本与正样本比
BATCH_EDGES  = 200_000         # 边批大小（根据显存/内存调整）
NUM_NEIGHBORS = [15, 10]       # 邻居采样（两层 GCN）
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================


def write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def year_from_text(txt: str) -> int:
    m = re.search(r"(19|20)\d{2}", txt)
    return int(m.group(0)) if m else None


def load_targets_from_multi_dir(base_dir: str):
    targets, start_map = set(), {}
    files = []
    for root, _, fs in os.walk(base_dir):
        for fn in fs:
            if fn.lower().endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    if not files:
        raise RuntimeError(f"未在 {base_dir} 找到 .jsonl")

    for path in tqdm(files, desc="load target authors"):
        cohort = year_from_text(path.replace("\\","/"))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                try:
                    j = json.loads(s)
                except:
                    continue
                aid = str(j.get("author_id") or "")
                if not aid: continue
                sy  = j.get("start_year")
                if sy is None:
                    y  = j.get("year")
                    cy = j.get("career_year")
                    try:
                        sy = int(y) - int(cy) + 1 if (y is not None and cy is not None) else cohort
                    except:
                        sy = cohort
                try:
                    sy = int(sy) if sy is not None else None
                except:
                    sy = None
                if sy is None:
                    continue
                if TARGET_START_MIN <= sy <= TARGET_START_MAX:
                    targets.add(aid)
                    if (aid not in start_map) or (sy < start_map[aid]):
                        start_map[aid] = sy
    return targets, start_map


def load_snapshot(year: int):
    epath = os.path.join(SNAPSHOT_DIR, f"edges_{year}.parquet")
    npath = os.path.join(SNAPSHOT_DIR, f"nodes_{year}.parquet")
    edges = pd.read_parquet(epath) if os.path.exists(epath) else pd.DataFrame(columns=["src","dst","weight"])
    nodes = pd.read_parquet(npath) if os.path.exists(npath) else pd.DataFrame({"node_id": pd.Series(dtype=str), "is_target": pd.Series(dtype="int8")})
    edges["src"] = edges.get("src", pd.Series(dtype=str)).astype(str)
    edges["dst"] = edges.get("dst", pd.Series(dtype=str)).astype(str)
    if "weight" not in edges.columns: edges["weight"] = 1
    if "node_id" not in nodes.columns:
        nodes = pd.DataFrame({"node_id": pd.unique(pd.concat([edges["src"],edges["dst"]], ignore_index=True))})
        nodes["is_target"] = 0
    else:
        nodes["node_id"] = nodes["node_id"].astype(str)
        if "is_target" not in nodes.columns:
            nodes["is_target"] = 0
    return edges, nodes


def hash_noise_vec(s: str, dim=8):
    # 基于 author_id 的可复现“哈希噪声”向量，均匀[-0.5,0.5]
    h = hashlib.md5(s.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8)[:dim].astype(np.float32)
    arr = (arr / 255.0) - 0.5
    if len(arr) < dim:
        pad = np.zeros(dim - len(arr), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def build_pyg_data(edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
    # 节点映射
    node_ids = nodes_df["node_id"].astype(str).tolist()
    id2idx = {nid:i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    # 边
    if len(edges_df) > 0:
        src = edges_df["src"].map(id2idx).dropna().astype(int).to_numpy()
        dst = edges_df["dst"].map(id2idx).dropna().astype(int).to_numpy()
        w   = edges_df["weight"].astype(float).to_numpy()
        # 去除映射失败
        mask = (src >= 0) & (dst >= 0)
        src = src[mask]; dst = dst[mask]; w = w[mask]
        # 无向图，双向边
        edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        edge_weight = np.concatenate([w, w]).astype(np.float32)
    else:
        edge_index = np.zeros((2,0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    # 结构特征
    # 度 / 加权度
    deg = np.zeros(N, dtype=np.float32)
    wdeg= np.zeros(N, dtype=np.float32)
    if edge_index.shape[1] > 0:
        for u,v,ww in zip(*edge_index, edge_weight):
            deg[u] += 1.0
            wdeg[u]+= ww
    logdeg  = np.log1p(deg)
    logwdeg = np.log1p(wdeg)

    is_target = nodes_df["is_target"].astype(np.float32).to_numpy()
    # 哈希噪声
    noise = np.stack([hash_noise_vec(nid, dim=8) for nid in node_ids], axis=0).astype(np.float32)

    x = np.concatenate([
        logdeg.reshape(-1,1),
        logwdeg.reshape(-1,1),
        is_target.reshape(-1,1),
        noise
    ], axis=1).astype(np.float32)

    # 简单标准化（z-score）
    mean = x.mean(axis=0, keepdims=True)
    std  = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    data = Data(
        x = torch.from_numpy(x),
        edge_index = torch.from_numpy(edge_index).long(),
        edge_weight = torch.from_numpy(edge_weight)
    )
    data.num_nodes = N

    return data, node_ids


class GCNEncoder(nn.Module):
    def __init__(self, in_dim=IN_DIM, hid=HID_DIM, out_dim=EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid, add_self_loops=True, normalize=True)
        self.bn1   = BatchNorm(hid)
        self.conv2 = GCNConv(hid, out_dim, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def dot_decoder(z_src, z_dst):
    return (z_src * z_dst).sum(dim=-1)

def train_one_year(data: Data, model: GCNEncoder, optimizer: torch.optim.Optimizer):
    model.train()
    data = data.to(DEVICE)

    # 训练边（正样本 = 所有边的一半去重；负样本自动采样）
    # 用 LinkNeighborLoader 以边为单位抽批，并采样每层的邻居
    pos_edge_index = data.edge_index
    # 去重（无向边），只保留 i<j
    mask = pos_edge_index[0] < pos_edge_index[1]
    pos_edge_index = pos_edge_index[:, mask]

    loader = LinkNeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=pos_edge_index,
        edge_label=torch.ones(pos_edge_index.size(1)),
        batch_size=BATCH_EDGES,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        shuffle=True
    )

    total_loss, total_examples = 0.0, 0
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        optimizer.zero_grad()
        # batch 包含子图：batch.x, batch.edge_index, batch.edge_label_index, batch.edge_label
        z = model(batch.x, batch.edge_index, batch.edge_weight if "edge_weight" in batch else None)
        # 从 batch 中的 edge_label_index 取端点嵌入
        src, dst = batch.edge_label_index
        pred = dot_decoder(z[src], z[dst])
        loss = bce(pred, batch.edge_label.float())
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.edge_label.numel()
        total_examples += int(batch.edge_label.numel())

    return total_loss / max(1, total_examples)


@torch.no_grad()
def infer_all_embeddings(data: Data, model: GCNEncoder):
    model.eval()
    data = data.to(DEVICE)

    # 小图/中图：直接全图前向
    if data.edge_index.size(1) <= 2_000_000 and data.num_nodes <= 1_000_000:
        z = model(data.x, data.edge_index, data.edge_weight if "edge_weight" in data else None)
        z = F.normalize(z, p=2, dim=-1)
        return z.cpu()

    # 大图：分批节点推理（邻居采样）
    loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=100_000, input_nodes=None)
    outs = []
    for batch in loader:
        bz = model(batch.x, batch.edge_index, batch.edge_weight if "edge_weight" in batch else None)
        outs.append((batch.n_id.cpu(), bz.detach().cpu()))
    # 拼回原顺序
    z_all = torch.zeros((data.num_nodes, EMB_DIM), dtype=torch.float32)
    for n_id, bz in outs:
        z_all[n_id] = bz
    z_all = F.normalize(z_all, p=2, dim=-1)
    return z_all


def main():
    os.makedirs(EMB_OUT_DIR, exist_ok=True)
    yearly_dir = os.path.join(EMB_OUT_DIR, "yearly_gcn")
    rel_dir    = os.path.join(EMB_OUT_DIR, "author_rel")
    os.makedirs(yearly_dir, exist_ok=True); os.makedirs(rel_dir, exist_ok=True)

    targets, start_map = load_targets_from_multi_dir(AUTHORS_DIR)
    print(f"[INFO] 目标作者数：{len(targets):,}")

    model = GCNEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    all_rows = []

    for y in range(YEAR_START, YEAR_END + 1):
        print(f"\n===== Year {y} =====")
        edges, nodes = load_snapshot(y)
        if edges.empty and nodes.empty:
            # 写空并跳过
            write_parquet(pd.DataFrame(columns=["node_id","year_abs"]+[f"e{i+1}" for i in range(EMB_DIM)]),
                          os.path.join(yearly_dir, f"emb_{y}.parquet"))
            continue

        data, node_ids = build_pyg_data(edges, nodes)

        print(f"[SNAP {y}] |V|={data.num_nodes:,} |E|={data.edge_index.size(1)//2:,}  (无向边数)")
        if data.num_nodes == 0:
            write_parquet(pd.DataFrame(columns=["node_id","year_abs"]+[f"e{i+1}" for i in range(EMB_DIM)]),
                          os.path.join(yearly_dir, f"emb_{y}.parquet"))
            continue

        # 训练若干轮（权重沿用上一年，等价于跨年共享编码器）
        for ep in range(1, EPOCHS_PER_YEAR+1):
            loss = train_one_year(data, model, optimizer)
            print(f"  epoch {ep}/{EPOCHS_PER_YEAR}  loss={loss:.4f}")

        # 全节点推理，写出当年嵌入
        z = infer_all_embeddings(data, model).numpy()
        df_y = pd.DataFrame(z, columns=[f"e{i+1}" for i in range(EMB_DIM)])
        df_y.insert(0, "node_id", node_ids)
        df_y.insert(1, "year_abs", y)
        write_parquet(df_y, os.path.join(yearly_dir, f"emb_{y}.parquet"))

        # 收集目标作者行，后续做 rel_year 映射
        take = df_y[df_y["node_id"].isin(targets)].copy()
        all_rows.append(take)

    # 汇总 → 映射到生涯年（仅 1998–2009 入行）
    if all_rows:
        emb_all = pd.concat(all_rows, ignore_index=True)
        emb_all["start_year"] = emb_all["node_id"].map(start_map)
        emb_all = emb_all.dropna(subset=["start_year"]).copy()
        emb_all["start_year"] = emb_all["start_year"].astype(int)
        emb_all["rel_year"] = emb_all["year_abs"] - emb_all["start_year"] + 1
        emb_all = emb_all[(emb_all["rel_year"] >= 1) & (emb_all["rel_year"] <= 10)].copy()
        emb_all = emb_all.rename(columns={"node_id":"author_id"})
        emb_all = emb_all.sort_values(["author_id","rel_year"]).reset_index(drop=True)
    else:
        emb_all = pd.DataFrame(columns=["author_id","rel_year","year_abs"]+[f"e{i+1}" for i in range(EMB_DIM)])

    out_rel = os.path.join(rel_dir, "author_year_embeddings_gcn.parquet")
    write_parquet(emb_all, out_rel)
    print("\n✅ 完成：逐年嵌入写入 →", yearly_dir)
    print("✅ 完成：作者×生涯年嵌入写入 →", out_rel)
    print("提示：把该文件与你的 author_year_features.parquet 用 (author_id, rel_year) join 即可输入 LSTM。")


if __name__ == "__main__":
    main()
