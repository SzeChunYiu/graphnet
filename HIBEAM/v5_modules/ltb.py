
import numpy as np
import torch
import torch.nn as nn

LTB_EDGE_K = 8
LTB_EDGE_THR = 0.6
LTB_MIN_CLUSTER = 3
LTB_LR = 1e-3
LTB_EPOCHS = 4

class HitEmbedder(nn.Module):
    def __init__(self, in_dim=4, hid=64, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class EdgeAffinityHead(nn.Module):
    def __init__(self, hdim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hdim*2 + 8, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, hi: torch.Tensor, hj: torch.Tensor, dfeat: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([hi, hj, dfeat], dim=-1)).squeeze(-1)

def build_knn_edges(xyz: np.ndarray, t: np.ndarray | None, k: int, use_time: bool) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    if use_time and t is not None:
        t0 = t - np.median(t)
        iqr = np.quantile(t0,0.75)-np.quantile(t0,0.25)+1e-12
        Z = np.column_stack([xyz, t0/iqr])
    else:
        Z = xyz
    k = min(k, max(1, len(Z)-1))
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Z)
    _, idx = nbrs.kneighbors(Z)
    edges = set()
    for i,row in enumerate(idx):
        for j in row[1:]:
            a,b = (i,int(j)) if i<int(j) else (int(j),i)
            edges.add((a,b))
    return np.array(sorted(edges), dtype=np.int64)

def edge_features(xyz: np.ndarray, t: np.ndarray | None, edges: np.ndarray) -> np.ndarray:
    pi = edges[:,0]; pj = edges[:,1]
    d = xyz[pj] - xyz[pi]
    r = np.linalg.norm(d, axis=1, keepdims=True)
    dz = np.abs(d[:,2:3])
    feat = np.concatenate([d, np.abs(d), r, dz], axis=1).astype(np.float32)
    return feat

def union_find_from_edges(n: int, edges_keep: np.ndarray):
    parent = np.arange(n)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb = find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for a,b in edges_keep:
        union(int(a),int(b))
    comp = {}
    for i in range(n):
        r=find(i); comp.setdefault(r, []).append(i)
    return [np.array(v, dtype=np.int64) for v in comp.values()]

def read_event_track_ids_from_parquet(data_dir: str):
    import os, glob, pandas as pd
    mapping = {}
    pulses_dir = os.path.join(data_dir, "pulses")
    files = sorted(glob.glob(os.path.join(pulses_dir, "pulses_*.parquet")))
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["event_id","track_id"])
        except Exception:
            continue
        for eid, grp in df.groupby("event_id"):
            mapping[int(eid)] = grp["track_id"].to_numpy()
    return mapping

@torch.no_grad()
def extract_event_hits_from_batch(batch, i_evt: int, use_time: bool):
    node_batch = batch.batch.detach().cpu().numpy()
    x_np = batch.x.detach().cpu().numpy()
    has_t = use_time and (x_np.shape[1] >= 4)
    import numpy as np
    boundaries = np.where(np.diff(node_batch) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries]); ends = np.concatenate([boundaries, [len(node_batch)]])
    s,e = starts[i_evt], ends[i_evt]
    xyz = x_np[s:e, :3]
    tvec = x_np[s:e, 3] if has_t else None
    return xyz, tvec

def train_edge_head_supervised_or_pseudo(model, data_dir: str, graph_definition, use_time: bool,
                                         edge_k: int = LTB_EDGE_K, epochs: int = LTB_EPOCHS, lr: float = LTB_LR):
    if epochs <= 0:
        return None
    print("[V5/LTB] Stage-2 training start…")
    from graphnet.data.dataset import ParquetDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sklearn.cluster import DBSCAN
    from v5_modules.clustering import robust_build_scaled_features as _feat

    features = ["dom_x","dom_y","dom_z","dom_t"] if use_time else ["dom_x","dom_y","dom_z"]
    truth = ["position_x","position_y","position_z"]
    dataset = ParquetDataset(path=data_dir, pulsemaps=["pulses"], truth_table="truth",
                             features=features, truth=truth, graph_definition=graph_definition,
                             index_column="event_id")
    loader = PyGDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    in_dim = 4 if use_time else 3
    device = next(model.parameters()).device
    hit_emb = HitEmbedder(in_dim=in_dim, hid=64, out_dim=64).to(device)
    edge_head = EdgeAffinityHead(hdim=64).to(device)
    opt = torch.optim.Adam(list(hit_emb.parameters())+list(edge_head.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))

    tid_map = read_event_track_ids_from_parquet(data_dir)

    model.eval()
    for ep in range(epochs):
        tot, loss_acc = 0, 0.0
        for batch in loader:
            event_ids = batch.event_id.detach().cpu().numpy().tolist()
            xyz, tvec = extract_event_hits_from_batch(batch, 0, use_time)
            N = len(xyz)
            if N < 2:
                continue
            edges0 = build_knn_edges(xyz, tvec, k=edge_k, use_time=use_time)
            if len(edges0)==0:
                continue

            use_supervised = False
            y_np = None
            if len(event_ids)==1:
                eid = int(event_ids[0])
                if eid in tid_map:
                    tid = tid_map[eid]
                    if len(tid) == N:
                        y_np = ((tid[edges0[:,0]] == tid[edges0[:,1]]) & (tid[edges0[:,0]]!=-1))
                        use_supervised = True
            if not use_supervised:
                cfg = getattr(model, 'multiscale_cfg', [{'eps':0.5,'min_samples':3,'scale_xyz':(0.1,0.1,0.1)}])[0]
                Z0 = _feat(xyz, tvec, cfg, use_time=use_time)
                labels = DBSCAN(eps=max(2.0, float(cfg.get("eps",0.5))*2), min_samples=2).fit_predict(Z0)
                y_np = (labels[edges0[:,0]] == labels[edges0[:,1]]) & (labels[edges0[:,0]]!=-1)

            if y_np is None or y_np.sum()==0:
                continue

            if use_time and tvec is not None:
                Xhit = torch.from_numpy(np.column_stack([xyz, tvec])).to(device=device, dtype=torch.float32)
            else:
                Xhit = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)

            H = hit_emb(Xhit)
            dfeat = torch.from_numpy(edge_features(xyz, tvec, edges0)).to(device=device, dtype=torch.float32)
            import torch as _torch
            E_ij = _torch.from_numpy(edges0.astype(np.int64)).to(device)
            hi = H[E_ij[:,0]]; hj = H[E_ij[:,1]]
            logits = edge_head(hi, hj, dfeat)
            y = _torch.from_numpy(y_np.astype(np.float32)).to(device)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_acc += float(loss.item()); tot += 1
        if tot>0:
            print(f"[V5/LTB] epoch {ep+1}/{epochs}  loss={loss_acc/tot:.4f}")

    model.hit_embedder = hit_emb
    model.edge_head = edge_head
    model.use_time_for_edges = use_time
    print("[V5/LTB] Trained edge head ready.")
    return edge_head

@torch.no_grad()
def run_ltb_tracks(model, xyz: np.ndarray, tvec: np.ndarray | None, use_time: bool,
                   edge_k: int = LTB_EDGE_K, thr: float = LTB_EDGE_THR, min_cluster: int = LTB_MIN_CLUSTER):
    if not (hasattr(model, "edge_head") and hasattr(model, "hit_embedder")):
        return []
    if len(xyz) < 2:
        return []
    edges0 = build_knn_edges(xyz, tvec, k=edge_k, use_time=use_time)
    if len(edges0)==0:
        return []
    device = next(model.parameters()).device
    if use_time and tvec is not None:
        Xhit = torch.from_numpy(np.column_stack([xyz, tvec])).to(device=device, dtype=torch.float32)
    else:
        Xhit = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)
    H = model.hit_embedder(Xhit)
    dfeat = torch.from_numpy(edge_features(xyz, tvec, edges0)).to(device=device, dtype=torch.float32)
    import torch as _torch
    E_ij = _torch.from_numpy(edges0.astype(np.int64)).to(device)
    hi = H[E_ij[:,0]]; hj = H[E_ij[:,1]]
    p = _torch.sigmoid(model.edge_head(hi, hj, dfeat)).detach().cpu().numpy()
    keep = edges0[p >= thr]
    comps = union_find_from_edges(len(xyz), keep)
    clusters = []
    for comp in comps:
        if comp.size >= min_cluster:
            clusters.append({"scale_id": -2, "idxs": comp})
    return clusters
