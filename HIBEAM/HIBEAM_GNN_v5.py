# HIBEAM_GNN_v4_allinone.py
# -*- coding: utf-8 -*-
"""
HIBEAM_GNN_v4_allinone.py
=========================
單一檔案，**無需任何參數**即可執行以下流程：
1) 掃描訓練資料夾（v3 風格）：Large_data/training_data_smeared/**/res_*cm
2) 逐個 res_*cm 進行訓練（DynEdge backbone + v4 候選生成/物理）
3) 在對應的驗證資料夾上做推論，輸出：
   - event 級：predictions.parquet
   - candidate 級：candidates.parquet
   - track 級：tracks_hits.parquet（每行一條 track，x/y/z/t 為 list）
4) （可選）直接由輸出結果畫 XY / ZY：figs_v4/<res_tag>/...

如需關閉可視化或調整 epoch/batch，修改本檔頂部常數即可。
"""

from __future__ import annotations

# ---------------- 標準庫 ----------------
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any
import random

# ---------------- 數值 / 科學計算 ----------------
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# ---------------- PyTorch / Lightning / PyG ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from v5_modules.gnn_with_track_cls import VertexGNNWithTrackCls
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import scatter

# ---------------- GraphNeT（與 v3 相同風格） ----------------
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import ParquetDataset
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdge
from graphnet.training.loss_functions import LogCoshLoss

# ---------------- Detector（請確保你專案有此模組） ----------------
from hibeam_det import HIBEAM_Detector


# --------------------------------------------------
# Setting files origin and output directories (matched to v4_allinone style)

TRAIN_BASE = Path(f"data/training_data")
VALID_BASE = Path(f"data/validation_data")
RESULTS_BASE = Path("results/v5_results")
TRACKS_BASE = Path("results/v5_results_tracks")
FIGS_BASE = Path("results/v5_plots")
# --------------------------------------------------

# ==================================================
# 執行常數（可直接改這幾個就好）
# ==================================================
BATCH_SIZE = 64
MAX_EPOCHS = 25 #30
GPUS = [0]         # []=CPU, [0] 用第一張 GPU
PIPELINE_CONFIG = {
    "mode": "train",          # 或 "export"
    "use_time": True,          # 使用 dom_t 特徵
    "pulses_with_labels_path": None,
    "thresholds": {
        "quality": 0.10,
    },
}

def feature_names(use_time: bool, include_signal_bkg: bool) -> list[str]:
    feats = ["dom_x", "dom_y", "dom_z"]
    if use_time:
        feats.append("dom_t")
    if include_signal_bkg:
        feats.append("signal_bkg")
    return feats


def resolve_pipeline_mode(config: dict[str, Any] | None) -> str:
    """Normalize the requested pipeline mode into one of train/val/export."""

    mode_raw = "train" if config is None else config.get("mode", "train")
    mode = str(mode_raw).strip().lower()

    if mode in {"train", "training"}:
        return "train"
    if mode in {"val", "valid", "validation"}:
        return "val"
    if mode in {"test", "export", "predict", "inference"}:
        return "export"

    raise ValueError(f"Unsupported pipeline mode: {mode_raw!r}")


def _validate_feature_width(loader, expected_dim: int, name: str) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    actual = int(batch.x.shape[1])
    if actual != expected_dim:
        raise RuntimeError(
            f"[{name}] Feature dimension mismatch: loader produced {actual} features, expected {expected_dim}."
        )


def _load_signal_lookup(path: str | Path | None) -> dict[int, np.ndarray]:
    if path in (None, ""):
        return {}
    resolved = Path(path)
    if not resolved.exists():
        print(f"[predict] pulses_with_labels_path not found: {resolved}")
        return {}
    try:
        df = pd.read_parquet(resolved)
    except Exception as exc:
        print(f"[predict] Failed to read labels parquet {resolved}: {exc}")
        return {}
    if "signal_bkg" not in df.columns or "event_id" not in df.columns:
        print(f"[predict] File {resolved} missing required columns event_id/signal_bkg")
        return {}
    if "row_idx" not in df.columns:
        df = df.copy()
        df["row_idx"] = df.groupby("event_id").cumcount()
    df = df.sort_values(["event_id", "row_idx"])
    lookup: dict[int, np.ndarray] = {}
    for eid, group in df.groupby("event_id"):
        lookup[int(eid)] = group["signal_bkg"].to_numpy()
    return lookup
AUTO_VIZ = False   # 訓練+推論之後，是否自動畫圖（XY/ZY）
N_VIZ_EACH = 20    # 每個 res_tag 隨機可視化的事件數

# ROI 用於 signal 標記（保持 v3）
ROI_XY = (0.0, 0.0)
ROI_Z_MIN, ROI_Z_MAX = -0.04, 0.04
ROI_R_TRAIN = 0.2  # 20 cm

# Highland 需要 β·p 代理值（MeV），如有動量估計可替換
BETA_P_DEFAULT = 300.0

# 多尺度 DBSCAN 配置（保持 v3 風格參數）
MULTISCALE_CFG = [
    {'eps': 0.35, 'min_samples': 3, 'scale_xyz': (0.05, 0.05, 0.05), 't_scale': 5e-3},
    {'eps': 0.55, 'min_samples': 4, 'scale_xyz': (0.10, 0.10, 0.10), 't_scale': 1e-2},
    {'eps': 0.80, 'min_samples': 5, 'scale_xyz': (0.20, 0.20, 0.20), 't_scale': 2e-2},
]
# ==================================================


# ==================================================
# 物理工具：Highland 公式、多重散射、幾何輔助
# ==================================================
def highland_theta0(beta_p_MeV: float, x_over_X0: float) -> float:
    """Highland 公式：多重散射 RMS 角度（弧度），beta_p_MeV = β·p [MeV/c]。"""
    x = max(float(x_over_X0), 1e-9)
    bp = max(float(beta_p_MeV), 1e-6)
    return (13.6 / bp) * np.sqrt(x) * (1.0 + 0.038 * np.log(x))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _pca_dir_and_center(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = xyz.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(xyz - c, full_matrices=False)
    d = Vt[0]
    return _normalize(d), c.squeeze(0)


def _cyl_intersection_xy(x0: float, y0: float, dx: float, dy: float, R: float) -> Optional[Tuple[float, float, float]]:
    A = dx*dx + dy*dy
    B = 2.0*(x0*dx + y0*dy)
    C = x0*x0 + y0*y*y if False else x0*x0 + y0*y0 - R*R  # keep math clear
    C = x0*x0 + y0*y0 - R*R
    disc = B*B - 4*A*C
    if disc < 0 or abs(A) < 1e-12:
        return None
    t1 = (-B - np.sqrt(disc)) / (2*A)
    t2 = (-B + np.sqrt(disc)) / (2*A)
    t = None
    for cand in (t1, t2):
        if cand is not None and cand > 0 and (t is None or cand < t):
            t = cand
    if t is None:
        return None
    return x0 + t*dx, y0 + t*dy, t


def _cluster_shape_features(Xyz: np.ndarray, T: Optional[np.ndarray]) -> Dict[str, float]:
    n = Xyz.shape[0]
    center = Xyz.mean(axis=0, keepdims=True)
    X = Xyz - center
    cov = np.cov(X, rowvar=False) + 1e-9*np.eye(3)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(w)[::-1]
    w1, w2, w3 = w
    s = w.sum() + 1e-12
    e1 = w1 / s

    mins = Xyz.min(axis=0); maxs = Xyz.max(axis=0)
    dx, dy, dz = (maxs - mins)
    length, width, thickness = float(dx), float(dy), float(dz)

    volume = (dx*dy*dz) + 1e-6
    density = float(n) / float(volume)
    temporal_std = float(np.std(T)) if T is not None else 0.0

    return {
        "length": length, "width": width, "thickness": thickness,
        "linearity": float((w1 - w2) / (w1 + 1e-12)),
        "planarity": float((w2 - w3) / (w1 + 1e-12)),
        "sphericity": float(w3 / (w1 + 1e-12)),
        "var_exp_1": float(e1),
        "density": float(np.log1p(density)),
        "temporal_std": temporal_std,
    }


def _make_brokenline_candidate(origin: np.ndarray, direction: np.ndarray, z_target: float, dtheta: float) -> Tuple[float, float, float]:
    """在 xy 平面旋轉（小角近似），dz 保持；回推到 z=z_target。"""
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    x0, y0, z0 = float(origin[0]), float(origin[1]), float(origin[2])
    cd, sd = np.cos(dtheta), np.sin(dtheta)
    dx2 = cd*dx - sd*dy
    dy2 = sd*dx + cd*dy
    dz2 = dz
    t = (float(z_target) - z0) / (dz2 + 1e-12)
    vx = x0 + t*dx2
    vy = y0 + t*dy2
    vz = z_target
    return vx, vy, vz


# ==================================================
# v4 候選生成：多尺度 DBSCAN + 折線（Highland）+ 形狀/材料特徵
# ==================================================
def _extract_candidates_multiscale_v4(
    points: np.ndarray,
    z_target: float,
    use_time: bool,
    multiscale_cfg: List[dict],
    x_over_X0: float,
    beta_p_default: float = BETA_P_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    回傳：
      cands_raw: [C,5] = (vx,vy,vz,n_hits,scale_id)  （保持 v3 RAW 格式）
      extra    : [C,12] = 群形狀(9) + x/X0(1) + theta0(1) + r_surface(1)
    """
    if points.size == 0:
        return (
            np.array([[0.0, 0.0, float(z_target), 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 12), dtype=np.float32),
        )

    if use_time and points.shape[1] >= 4:
        x, y, z, t = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        P = np.column_stack([x, y, z])
        T = t
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        P = np.column_stack([x, y, z])
        T = None

    all_raw: List[np.ndarray] = []
    all_ex: List[np.ndarray] = []

    theta0 = float(highland_theta0(beta_p_default, x_over_X0))
    DTHETA_MULTS = [0.0, +1.0, -1.0, +2.0, -2.0]

    for sid, cfg in enumerate(multiscale_cfg):
        eps = float(cfg.get("eps", 0.5))
        min_samples = int(cfg.get("min_samples", 3))
        scale_xyz = cfg.get("scale_xyz", (0.1, 0.1, 0.1))
        t_scale = float(cfg.get("t_scale", 1.0))

        # 建縮放空間做 DBSCAN
        if use_time and T is not None:
            Zfeat = np.column_stack([x / (scale_xyz[0] + 1e-12),
                                     y / (scale_xyz[1] + 1e-12),
                                     z / (scale_xyz[2] + 1e-12),
                                     T / (t_scale + 1e-12)])
        else:
            Zfeat = np.column_stack([x / (scale_xyz[0] + 1e-12),
                                     y / (scale_xyz[1] + 1e-12),
                                     z / (scale_xyz[2] + 1e-12)])

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Zfeat)
        uniq = np.unique(labels)
        has_cluster = False

        for label in uniq:
            if label == -1:
                continue
            mask = labels == label
            cluster = P[mask, :]
            if cluster.shape[0] < 2:
                continue
            has_cluster = True
            # PCA 方向與中心
            direction, center = _pca_dir_and_center(cluster)
            if abs(direction[2]) < 1e-6:
                continue

            # 與桶體（外半徑假設 0.22 m，內半徑 0.21 m；只影響 r_surface 特徵，不影響幾何）
            # 若你要精確幾何，可把 hibeam 幾何引入，這裡簡化為量測 r_surface
            r_surface = float(np.hypot(center[0], center[1]))

            # 形狀特徵
            shp = _cluster_shape_features(cluster, T[mask] if (T is not None) else None)
            shp_vec = np.array([
                shp["length"], shp["width"], shp["thickness"],
                shp["linearity"], shp["planarity"], shp["sphericity"],
                shp["var_exp_1"], shp["density"], shp["temporal_std"],
            ], dtype=np.float32)

            for m in DTHETA_MULTS:
                dth = m * theta0
                vx, vy, vz = _make_brokenline_candidate(center, direction, z_target, dth)
                raw = np.array([vx, vy, vz, float(cluster.shape[0]), float(sid)], dtype=np.float32)
                ex = np.concatenate([shp_vec, np.array([x_over_X0, theta0, r_surface], dtype=np.float32)], axis=0)
                all_raw.append(raw)
                all_ex.append(ex)

        if not has_cluster:
            all_raw.append(np.array([0.0, 0.0, float(z_target), 0.0, float(sid)], dtype=np.float32))
            all_ex.append(np.zeros(12, dtype=np.float32))

    cands_raw = np.stack(all_raw, axis=0) if all_raw else np.array([[0.0, 0.0, float(z_target), 0.0, 0.0]], dtype=np.float32)
    extra = np.stack(all_ex, axis=0) if all_ex else np.zeros((1, 12), dtype=np.float32)
    return cands_raw, extra


# ==================================================
# 注意力池化（v3 單頭）
# ==================================================
class CandidateAttentionPool(nn.Module):
    """Single-head attention: query=event embedding, key/value=candidate embeddings."""
    def __init__(self, d_event: int, d_cand: int, d_att: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(d_event, d_att, bias=False)
        self.k_proj = nn.Linear(d_cand, d_att, bias=False)
        self.v_proj = nn.Linear(d_cand, d_att, bias=False)
        self.out = nn.Linear(d_att, d_cand, bias=False)

    def forward(self, event_emb: Tensor, cand_pad: Tensor, cand_mask: Tensor) -> tuple[Tensor, Tensor]:
        B, C, Dc = cand_pad.shape
        q = self.q_proj(event_emb).unsqueeze(1)                 # [B,1,Da]
        k = self.k_proj(cand_pad)                               # [B,C,Da]
        v = self.v_proj(cand_pad)                               # [B,C,Da]
        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1)  # [B,C]
        scores = scores / (k.shape[-1] ** 0.5)
        scores = scores.masked_fill(~cand_mask, float("-inf"))
        att = torch.softmax(scores, dim=-1)                     # [B,C]
        ctx = torch.matmul(att.unsqueeze(1), v).squeeze(1)      # [B,Da]
        pooled = self.out(ctx)                                  # [B,Dc]
        return pooled, att


# ==================================================
# 模型（LightningModule）— 類名/介面盡量保留 v3
# ==================================================
class MultiTrackModel(pl.LightningModule):
    """DynEdge backbone + attention over multi-scale candidates with v4 physics-aware features."""

    def __init__(
        self,
        graph_definition: KNNGraph,
        lr: float = 1e-3,
        z_target: float = 0.0,
        use_time: bool = True,
        multiscale_cfg: List[dict] | None = None,
        max_candidates: int = 64,
        quality_thresh: float = 0.10,
        x_over_X0: float = 0.01 / 0.0889,  # 厚 1cm、鋁 X0≈8.89cm
        feature_list: list[str] | None = None,
    ):
        super().__init__()
        try:
            self.save_hyperparameters({"lr": lr})
        except Exception:
            pass

        self.feature_list = list(feature_list) if feature_list else feature_names(use_time, include_signal_bkg=False)
        self.signal_feature_index = None
        if feature_list and "signal_bkg" in feature_list:
            self.signal_feature_index = feature_list.index("signal_bkg")
        self.coord_dims = 4 if use_time else 3

        # Backbone（與 v3 一致：DynEdge + 多種全域池化）
        self.backbone = DynEdge(
            nb_inputs=graph_definition.nb_outputs,
            global_pooling_schemes=["min", "max", "mean", "sum"],
        )
        Ge = self.backbone.nb_outputs

        # 候選嵌入（v4：加入 12 維附加特徵）
        self.n_scales = len(multiscale_cfg) if multiscale_cfg else 3
        self.cand_base_dim = 4      # (vx,vy,vz,n_hits)
        self.cand_extra_dim = 12    # 形狀/材料
        self.cand_in_dim = self.cand_base_dim + self.cand_extra_dim + self.n_scales
        self.Dc = 32
        self.cand_mlp = nn.Sequential(
            nn.Linear(self.cand_in_dim, 32), nn.ReLU(),
            nn.Linear(32, self.Dc), nn.ReLU(),
        )

        # Attention 池化（v3 單頭）
        self.cand_pool = CandidateAttentionPool(d_event=Ge, d_cand=self.Dc, d_att=128)

        # Event head（輸出 4 維：x,y,z + signal logit）
        self.event_head = nn.Sequential(
            nn.Linear(Ge + self.Dc, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

        # 候選輔助頭（quality / pointer）
        self.cand_quality_head = nn.Linear(self.Dc, 1)
        self.cand_pointer_head = nn.Linear(self.Dc, 1)

        # Losses
        self.pos_loss = LogCoshLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.qual_loss = nn.BCEWithLogitsLoss()

        # Config
        self.lr = lr
        self.z_target = z_target
        self.use_time = use_time
        self.max_candidates = max_candidates
        self.quality_thresh = quality_thresh
        self.multiscale_cfg = multiscale_cfg or MULTISCALE_CFG
        self.x_over_X0 = x_over_X0

    # Utilities ---------------------------------------------------------
    def _is_signal(self, vx: Tensor, vy: Tensor, vz: Tensor) -> Tensor:
        dx = vx - ROI_XY[0]
        dy = vy - ROI_XY[1]
        r = torch.sqrt(dx * dx + dy * dy)
        return ((r < ROI_R_TRAIN) & (vz.abs().le(max(abs(ROI_Z_MIN), abs(ROI_Z_MAX))))).float()

    def _get_event_signal_labels(self, batch: Batch) -> Tensor | None:
        if self.signal_feature_index is None:
            return None
        if self.signal_feature_index >= batch.x.shape[1]:
            return None
        node_signal = batch.x[:, self.signal_feature_index].detach()
        event_signal = scatter(
            node_signal,
            batch.batch,
            dim=0,
            dim_size=int(batch.num_graphs),
            reduce="max",
        )
        return event_signal.float()

    def _extract_per_event(self, event_x: Tensor) -> tuple[Tensor, Tensor, torch.Tensor]:
        """對單一事件的 node 特徵產生候選：返回 cand_emb, mask, cand_raw(cpu)."""
        device = event_x.device
        has_time = self.use_time and (event_x.shape[1] >= self.coord_dims)
        coord_cols = self.coord_dims
        pts = event_x[:, :coord_cols].detach().cpu().numpy()

        cands_raw, extra = _extract_candidates_multiscale_v4(
            points=pts,
            z_target=self.z_target,
            use_time=has_time,
            multiscale_cfg=self.multiscale_cfg,
            x_over_X0=self.x_over_X0,
            beta_p_default=BETA_P_DEFAULT,
        )  # [Ci,5], [Ci,12]

        scale_id = cands_raw[:, 4].astype(np.int64)
        one_hot = np.eye(self.n_scales, dtype=np.float32)[np.clip(scale_id, 0, self.n_scales-1)]
        base = cands_raw[:, :4].astype(np.float32)  # (vx,vy,vz,n_hits)
        feat = np.concatenate([base, extra.astype(np.float32), one_hot], axis=1)  # [Ci, 4+12+S]

        feat_t = torch.from_numpy(feat).to(device)
        cand_emb = self.cand_mlp(feat_t)  # [Ci,Dc]
        mask = torch.ones(cand_emb.shape[0], dtype=torch.bool, device=device)

        return cand_emb, mask, torch.from_numpy(cands_raw)

    def _candidates_forward(self, batch: Batch) -> tuple[Tensor, Tensor, List[torch.Tensor]]:
        device = batch.x.device
        event_id = batch.batch
        B = int(batch.num_graphs)
        cand_list: List[Tensor] = []
        mask_list: List[Tensor] = []
        raw_list: List[torch.Tensor] = []

        for i in range(B):
            mask = (event_id == i)
            emb_i, m_i, raw_i = self._extract_per_event(batch.x[mask])
            if emb_i.shape[0] == 0:
                emb_i = torch.zeros(1, self.Dc, device=device)
                m_i = torch.zeros(1, dtype=torch.bool, device=device)
                raw_i = torch.zeros(1, 5)
            cand_list.append(emb_i)
            mask_list.append(m_i)
            raw_list.append(raw_i)

        Cmax = min(self.max_candidates, max(ci.shape[0] for ci in cand_list))
        Cmax = max(Cmax, 1)
        cand_pad = torch.zeros(B, Cmax, self.Dc, device=device)
        cand_mask = torch.zeros(B, Cmax, dtype=torch.bool, device=device)

        for i, (ci, mi) in enumerate(zip(cand_list, mask_list)):
            c = min(Cmax, ci.shape[0])
            cand_pad[i, :c] = ci[:c]
            cand_mask[i, :c] = mi[:c]

        return cand_pad, cand_mask, raw_list

    # Lightning hooks ---------------------------------------------------
    def forward(self, data: Batch | List[Data] | Data) -> Tuple[Tensor, Tensor, dict]:
        if isinstance(data, (list, tuple)):
            batch = Batch.from_data_list(list(data))
        elif isinstance(data, Batch):
            batch = data
        else:
            batch = Batch.from_data_list([data])

        global_emb = self.backbone(batch)  # [B, Ge]
        cand_pad, cand_mask, cand_raw_list = self._candidates_forward(batch)

        pooled, att_w = self.cand_pool(global_emb, cand_pad, cand_mask)       # [B,Dc], [B,C]

        out = self.event_head(torch.cat([global_emb, pooled], dim=1))         # [B,4]
        vertex = out[:, :3]
        signal_logit = out[:, 3]

        qual_logit = self.cand_quality_head(cand_pad).squeeze(-1)             # [B,C]
        ptr_logit = self.cand_pointer_head(cand_pad).squeeze(-1)              # [B,C]

        aux = {
            "qual_logit": qual_logit,
            "ptr_logit": ptr_logit,
            "cand_mask": cand_mask,
            "cand_raw_list": cand_raw_list,
            "att_weights": att_w,
        }
        return vertex, signal_logit, aux

    def _build_candidate_targets(self, cand_raw_list: List[torch.Tensor], true_v: Tensor):
        quality_targets: List[Tensor] = []
        pointer_targets: List[int] = []

        for i in range(true_v.shape[0]):
            cand = cand_raw_list[i]  # CPU tensor [Ci,5]
            if cand.numel() == 0:
                cand = torch.tensor([[0., 0., self.z_target, 0., 0.]], dtype=torch.float32)
            pos = cand[:, :3]
            dist = torch.linalg.norm(pos - true_v[i].cpu().unsqueeze(0), dim=-1)  # [Ci]
            q = (dist <= self.quality_thresh).float()
            j = int(torch.argmin(dist))
            quality_targets.append(q)
            pointer_targets.append(j)

        return quality_targets, pointer_targets

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        vx, vy, vz = batch.position_x, batch.position_y, batch.position_z
        y_true = torch.stack([vx, vy, vz], dim=1)             # [B,3]

        vertex_pred, signal_logit, aux = self.forward(batch)
        loss_pos = self.pos_loss(vertex_pred, y_true)

        signal_targets = self._get_event_signal_labels(batch)
        if signal_targets is None:
            if not hasattr(self, "_missing_signal_warned"):
                print("[train] signal_bkg labels missing; skipping classification loss.")
                self._missing_signal_warned = True
            loss_sig = torch.tensor(0.0, device=vertex_pred.device)
        else:
            loss_sig = self.cls_loss(signal_logit, signal_targets)

        qual_logit = aux["qual_logit"]
        ptr_logit = aux["ptr_logit"]
        cand_mask = aux["cand_mask"]
        cand_raw_list = aux["cand_raw_list"]

        qual_t_list, ptr_idx_list = self._build_candidate_targets(cand_raw_list, y_true)

        B, C = cand_mask.shape
        qual_t_pad = torch.zeros(B, C, device=vertex_pred.device)
        ptr_idx = torch.zeros(B, dtype=torch.long, device=vertex_pred.device)
        for i in range(B):
            Ci = min(C, len(qual_t_list[i]))
            if Ci > 0:
                qual_t_pad[i, :Ci] = qual_t_list[i][:Ci].to(vertex_pred.device)
                ptr_idx[i] = min(Ci - 1, int(ptr_idx_list[i]))
            else:
                ptr_idx[i] = 0

        very_neg = torch.finfo(ptr_logit.dtype).min / 2
        ptr_logits_masked = ptr_logit.masked_fill(~cand_mask, very_neg)

        loss_qual = self.qual_loss(qual_logit[cand_mask], qual_t_pad[cand_mask])
        loss_ptr = F.cross_entropy(ptr_logits_masked, ptr_idx)

        att = aux["att_weights"]
        if att is not None:
            ptr_soft = F.one_hot(ptr_idx, num_classes=C).float()
            loss_align = F.mse_loss(att, ptr_soft)
        else:
            loss_align = torch.tensor(0.0, device=vertex_pred.device)

        loss = loss_pos + loss_sig + 0.5 * loss_qual + 0.5 * loss_ptr + 0.1 * loss_align

        self.log("train/pos", loss_pos, prog_bar=True)
        self.log("train/cls", loss_sig, prog_bar=True)
        self.log("train/qual", loss_qual, prog_bar=False)
        self.log("train/ptr", loss_ptr, prog_bar=False)
        self.log("train/align", loss_align, prog_bar=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        vx, vy, vz = batch.position_x, batch.position_y, batch.position_z
        y_true = torch.stack([vx, vy, vz], dim=1)

        with torch.no_grad():
            vertex_pred, signal_logit, aux = self.forward(batch)
            loss_pos = self.pos_loss(vertex_pred, y_true)
            signal_targets = self._get_event_signal_labels(batch)
            if signal_targets is None:
                loss_sig = torch.tensor(0.0, device=vertex_pred.device)
            else:
                loss_sig = self.cls_loss(signal_logit, signal_targets)

            qual_logit = aux["qual_logit"]
            ptr_logit = aux["ptr_logit"]
            cand_mask = aux["cand_mask"]
            cand_raw_list = aux["cand_raw_list"]

            qual_t_list, ptr_idx_list = self._build_candidate_targets(cand_raw_list, y_true)
            B, C = cand_mask.shape
            qual_t_pad = torch.zeros(B, C, device=vertex_pred.device)
            ptr_idx = torch.zeros(B, dtype=torch.long, device=vertex_pred.device)
            for i in range(B):
                Ci = min(C, len(qual_t_list[i]))
                if Ci > 0:
                    qual_t_pad[i, :Ci] = qual_t_list[i][:Ci].to(vertex_pred.device)
                    ptr_idx[i] = min(Ci - 1, int(ptr_idx_list[i]))
                else:
                    ptr_idx[i] = 0

            very_neg = torch.finfo(ptr_logit.dtype).min / 2
            ptr_logits_masked = ptr_logit.masked_fill(~cand_mask, very_neg)

            loss_qual = self.qual_loss(qual_logit[cand_mask], qual_t_pad[cand_mask])
            loss_ptr = F.cross_entropy(ptr_logits_masked, ptr_idx)

            att = aux["att_weights"]
            if att is not None:
                ptr_soft = F.one_hot(ptr_idx, num_classes=C).float()
                loss_align = F.mse_loss(att, ptr_soft)
            else:
                loss_align = torch.tensor(0.0, device=vertex_pred.device)

            loss = loss_pos + loss_sig + 0.5 * loss_qual + 0.5 * loss_ptr + 0.1 * loss_align

            self.log("val/pos", loss_pos, prog_bar=True)
            self.log("val/cls", loss_sig, prog_bar=True)
            self.log("val/qual", loss_qual, prog_bar=False)
            self.log("val/ptr", loss_ptr, prog_bar=False)
            self.log("val/align", loss_align, prog_bar=False)
            self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3))


# ==================================================
# 訓練 + 推論 + 匯出（單個 res_*cm 目錄）
# ==================================================
def train_and_eval_for_dir(
    data_dir: str,
    pred_dir: str,
    res_tag: str,
    mul_tag: str,
    max_epochs: int | None = None,
    folder: str | None = None,
    use_time: bool = True,
    config: dict[str, Any] | None = None,
) -> Path:
    if max_epochs is None:
        max_epochs = MAX_EPOCHS

    config = config or PIPELINE_CONFIG
    mode = resolve_pipeline_mode(config)
    use_time = bool(config.get("use_time", use_time))
    include_signal = mode in {"train", "val"}
    if mode == "export":
        raise RuntimeError("train_and_eval_for_dir cannot run in export mode; use inference_and_export_tracks instead.")
    quality_thresh_cfg = float(config.get("thresholds", {}).get("quality", 0.10))

    base = Path(folder) if folder else RESULTS_BASE
    out_root = base / mul_tag / res_tag
    out_root.mkdir(parents=True, exist_ok=True)

    detector = HIBEAM_Detector()
    graph_definition = KNNGraph(detector=detector)

    features = feature_names(use_time=use_time, include_signal_bkg=include_signal)
    truth = ["position_x", "position_y", "position_z"]

    dm = GraphNeTDataModule(
        dataset_reference=ParquetDataset,
        dataset_args={
            "path": data_dir,
            "pulsemaps": ["pulses"],
            "truth_table": "truth",
            "features": features,
            "truth": truth,
            "data_representation": graph_definition,
            "index_column": "event_id",
        },
        train_dataloader_kwargs={
            "batch_size": BATCH_SIZE,
            "num_workers": 2,
            "persistent_workers": True,
            "pin_memory": False,
            "shuffle": True,
        },
    )

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    _validate_feature_width(train_loader, len(features), "train")
    if val_loader is not None:
        _validate_feature_width(val_loader, len(features), "val")

    model = MultiTrackModel(graph_definition=graph_definition,
        lr=1e-3,
        z_target=0.0,
        use_time=use_time,
        multiscale_cfg=MULTISCALE_CFG,
        max_candidates=64,
        quality_thresh=quality_thresh_cfg,
        x_over_X0=0.01/0.0889,
        feature_list=features,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if GPUS else "cpu",
        devices=GPUS if GPUS else 1,
        max_epochs=max_epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        deterministic=False,
    )
    trainer.fit(model, train_loader, val_loader)

    # --------- Prediction（event/candidate/track 三表） ---------
    inference_and_export_tracks(
        model,
        pred_dir,
        out_dir=out_root/"event_level_information/",
        use_time=use_time,
        graph_definition=graph_definition,
        pulses_with_labels_path=config.get("pulses_with_labels_path"),
    )

    # 為了對齊 v3，我們另外生成一份 predictions.csv 與簡單 histogram
    from matplotlib import pyplot as plt
    # event 級預測
    preds_df = pd.read_parquet(out_root/"event_level_information/"/"predictions.parquet")
    # 真值
    truth_files = sorted(Path(pred_dir, 'truth').glob('truth_*.parquet'))
    if not truth_files:
        print(f"[warn] No truth files under {Path(pred_dir) / 'truth'}. Skipping residual histograms.")
        return out_root
    truth_df = pd.concat((pd.read_parquet(str(f)) for f in truth_files), ignore_index=True)
    merged = preds_df.merge(truth_df, on="event_id", how="inner")
    merged["dx"] = merged["position_x_pred"] - merged["position_x"]
    merged["dy"] = merged["position_y_pred"] - merged["position_y"]
    merged["dz"] = merged["position_z_pred"] - merged["position_z"]
    merged["dist_residual"] = np.sqrt(merged["dx"]**2 + merged["dy"]**2 + merged["dz"]**2)
    merged.to_csv(out_root / "predictions.csv", index=False)

    def _save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path) -> None:
        plt.figure(figsize=(6, 4))
        plt.hist(series, bins=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

    _save_hist(merged["dist_residual"], f"Distance residual — {res_tag}", "Residual [m]", out_root / f"hist_dist_residual__{res_tag}.png")
    _save_hist(merged["dx"], f"dx — {res_tag}", "dx [m]", out_root / f"hist_dx__{res_tag}.png")
    _save_hist(merged["dy"], f"dy — {res_tag}", "dy [m]", out_root / f"hist_dy__{res_tag}.png")
    _save_hist(merged["dz"], f"dz — {res_tag}", "dz [m]", out_root / f"hist_dz__{res_tag}.png")

    return out_root


# ==================================================
# 推論 + 匯出（三表）：使用訓練後的 model 直接作用於 pred_dir
# ==================================================
def _build_scaled_features(xyz: np.ndarray, t: np.ndarray | None, cfg: dict, use_time: bool) -> np.ndarray:
    sx, sy, sz = cfg.get("scale_xyz", (0.1,0.1,0.1))
    if use_time and t is not None:
        ts = cfg.get("t_scale", 1.0)
        Z = np.column_stack([xyz[:,0]/(sx+1e-12),
                             xyz[:,1]/(sy+1e-12),
                             xyz[:,2]/(sz+1e-12),
                             t/(ts+1e-12)])
    else:
        Z = np.column_stack([xyz[:,0]/(sx+1e-12),
                             xyz[:,1]/(sy+1e-12),
                             xyz[:,2]/(sz+1e-12)])
    return Z


def _inference_and_export_tracks_impl(
    model: MultiTrackModel,
    data_dir: str,
    out_dir: Path,
    use_time: bool,
    graph_definition: KNNGraph,
    pulses_with_labels_path: str | None = None,
):
    features = feature_names(use_time=use_time, include_signal_bkg=False)
    truth = ["position_x", "position_y", "position_z"]

    dataset = ParquetDataset(
        path=data_dir,
        pulsemaps=["pulses"],
        truth_table="truth",
        features=features,
        truth=truth,
        data_representation=graph_definition,
        index_column="event_id",
    )
    loader = PyGDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, persistent_workers=False)

    if getattr(dataset, "n_events", None) in (0, None) and len(dataset) == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([]).to_parquet(out_dir / "predictions.parquet", index=False)
        pd.DataFrame([]).to_parquet(out_dir / "candidates.parquet", index=False)
        pd.DataFrame([]).to_parquet(out_dir / "tracks_hits.parquet", index=False)
        print(f"[predict] Dataset empty. Wrote empty outputs to → {out_dir}")
        return

    ms_cfg = model.multiscale_cfg

    preds_rows: List[Dict[str, Any]] = []
    cands_rows: List[Dict[str, Any]] = []
    tracks_rows: List[Dict[str, Any]] = []
    tracks_class_rows: List[Dict[str, Any]] = []
    label_lookup = _load_signal_lookup(pulses_with_labels_path)
    warned_events: set[int] = set()
    track_truth_map: dict[tuple[int, int], float] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            vertex_pred, signal_logit, aux = model(batch)

            B = int(batch.num_graphs)
            event_ids = batch.event_id.detach().cpu().numpy().tolist()
            node_batch = batch.batch.detach().cpu().numpy()
            x_np = batch.x.detach().cpu().numpy()
            has_t = use_time and x_np.shape[1] >= 4

            signal_idx = getattr(model, "signal_feature_index", None)
            if signal_idx is not None and signal_idx < batch.x.shape[1]:
                node_truth_all = batch.x[:, signal_idx].detach().cpu().numpy()
            else:
                node_truth_all = None

            boundaries = np.where(np.diff(node_batch) != 0)[0] + 1
            starts = np.concatenate([[0], boundaries])
            ends = np.concatenate([boundaries, [len(node_batch)]])

            att = aux["att_weights"]
            qual_logit = aux["qual_logit"]
            ptr_logit = aux["ptr_logit"]
            cand_mask = aux["cand_mask"]
            cand_raw_list = aux["cand_raw_list"]

            very_neg = torch.finfo(ptr_logit.dtype).min / 2

            for i in range(B):
                eid = int(event_ids[i])
                vp = vertex_pred[i].detach().cpu().numpy()
                preds_rows.append({
                    "event_id": eid,
                    "position_x_pred": float(vp[0]),
                    "position_y_pred": float(vp[1]),
                    "position_z_pred": float(vp[2]),
                    "signal_score": float(torch.sigmoid(signal_logit[i]).item()),
                })

                valid = int(cand_mask[i].sum().item())
                att_i = att[i, :valid].detach().cpu().numpy() if valid > 0 else np.zeros((0,))
                qual_i = torch.sigmoid(qual_logit[i, :valid]).detach().cpu().numpy() if valid > 0 else np.zeros((0,))
                ptr_logits_masked = ptr_logit[i].masked_fill(~cand_mask[i], very_neg)
                ptr_i = int(torch.argmax(ptr_logits_masked).item()) if valid > 0 else 0

                raw = cand_raw_list[i].numpy() if cand_raw_list[i].numel() > 0 else np.zeros((0, 5), dtype=np.float32)
                for j in range(min(valid, raw.shape[0])):
                    vx, vy, vz, nh, sid = raw[j]
                    cands_rows.append({
                        "event_id": eid,
                        "cand_idx": j,
                        "vx": float(vx),
                        "vy": float(vy),
                        "vz": float(vz),
                        "n_hits": float(nh),
                        "scale_id": int(sid),
                        "att_weight": float(att_i[j]) if j < len(att_i) else float("nan"),
                        "qual_score": float(qual_i[j]) if j < len(qual_i) else float("nan"),
                        "is_pointer": int(j == ptr_i),
                    })

                s = starts[i]
                e = ends[i]
                xyz = x_np[s:e, :3]
                tvec = x_np[s:e, 3] if has_t else None

                labels_local: np.ndarray | None = None
                if node_truth_all is not None:
                    labels_local = node_truth_all[s:e]
                else:
                    lookup_vals = label_lookup.get(eid)
                    if lookup_vals is not None:
                        if len(lookup_vals) < (e - s) and eid not in warned_events:
                            print(
                                f"[predict] signal_bkg length mismatch for event {eid}: dataset {e - s}, labels {len(lookup_vals)}"
                            )
                            warned_events.add(eid)
                        labels_local = lookup_vals[: e - s]

                sid = int(raw[ptr_i, 4]) if raw.shape[0] > 0 else 0
                cfg = ms_cfg[sid]
                Zfeat = _build_scaled_features(xyz, tvec, cfg, use_time=has_t)
                labels = DBSCAN(eps=float(cfg.get("eps", 0.5)), min_samples=int(cfg.get("min_samples", 3))).fit_predict(Zfeat)

                uniq = np.unique(labels)
                for tid in uniq:
                    idxs = np.where(labels == tid)[0]
                    if idxs.size == 0:
                        continue
                    xs = xyz[idxs, 0].astype(float).tolist()
                    ys = xyz[idxs, 1].astype(float).tolist()
                    zs = xyz[idxs, 2].astype(float).tolist()
                    ts = (
                        tvec[idxs].astype(float).tolist()
                        if has_t and tvec is not None
                        else [float("nan")] * len(xs)
                    )

                    hit_labels: list[int] = []
                    track_fraction = float("nan")
                    max_idx = int(idxs.max()) if idxs.size > 0 else -1
                    if labels_local is not None and max_idx < len(labels_local):
                        local_hits = labels_local[idxs]
                        hit_labels = [int(round(float(v))) for v in local_hits.tolist()]
                        if len(local_hits) > 0:
                            track_fraction = float(np.mean(local_hits))
                    elif labels_local is not None and idxs.size > 0 and eid not in warned_events:
                        print(f"[predict] signal_bkg missing hits for event {eid}, track {int(tid)}")
                        warned_events.add(eid)

                    tracks_rows.append({
                        "event_id": eid,
                        "scale_id": sid,
                        "track_id": int(tid),
                        "x": xs,
                        "y": ys,
                        "z": zs,
                        "t": ts,
                        "hit_signal_bkg": hit_labels,
                        "track_signal_fraction": track_fraction,
                    })
                    if tid >= 0 and not np.isnan(track_fraction):
                        track_truth_map[(eid, int(tid))] = track_fraction

                left = np.where(labels < 0)[0]
                if left.size > 0:
                    xs = xyz[left, 0].astype(float).tolist()
                    ys = xyz[left, 1].astype(float).tolist()
                    zs = xyz[left, 2].astype(float).tolist()
                    ts = (
                        tvec[left].astype(float).tolist()
                        if has_t and tvec is not None
                        else [float("nan")] * len(xs)
                    )
                    hit_labels = []
                    track_fraction = float("nan")
                    max_idx_left = int(left.max()) if left.size > 0 else -1
                    if labels_local is not None and max_idx_left < len(labels_local):
                        local_hits = labels_local[left]
                        hit_labels = [int(round(float(v))) for v in local_hits.tolist()]
                        if len(local_hits) > 0:
                            track_fraction = float(np.mean(local_hits))
                    tracks_rows.append({
                        "event_id": eid,
                        "scale_id": -1,
                        "track_id": -1,
                        "x": xs,
                        "y": ys,
                        "z": zs,
                        "t": ts,
                        "hit_signal_bkg": hit_labels,
                        "track_signal_fraction": track_fraction,
                    })

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds_rows).to_parquet(out_dir / "predictions.parquet", index=False)
    pd.DataFrame(cands_rows).to_parquet(out_dir / "candidates.parquet", index=False)
    tracks_df = pd.DataFrame(tracks_rows)
    try:
        tracks_df.to_parquet(out_dir / "tracks_hits.parquet", index=False, engine="pyarrow")
    except Exception:
        tracks_df.to_parquet(out_dir / "tracks_hits.parquet", index=False)

    if tracks_class_rows:
        for row in tracks_class_rows:
            key = (row.get("event_id"), row.get("track_id"))
            if key in track_truth_map:
                row["track_signal_fraction"] = float(track_truth_map[key])
        pd.DataFrame(tracks_class_rows).to_parquet(out_dir / "tracks_class.parquet", index=False)

    print(f"[predict] Saved → {out_dir}")


def inference_and_export_tracks(
    model: MultiTrackModel,
    data_dir: str,
    out_dir: Path,
    use_time: bool,
    graph_definition: KNNGraph,
    pulses_with_labels_path: str | None = None,
):
    return _inference_and_export_tracks_impl(
        model=model,
        data_dir=data_dir,
        out_dir=out_dir,
        use_time=use_time,
        graph_definition=graph_definition,
        pulses_with_labels_path=pulses_with_labels_path,
    )

# ========================= PATCH: Robust clustering =========================
# 多尺度聯合集群 + 自適應 eps + 小樣本友好 + robust 時間處理
# 新增：平行近軌的「正交方向二分」(GMM/KMeans)，避免兩條軌被黐成一條。

from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def _build_scaled_features(xyz: np.ndarray, t: np.ndarray | None, cfg: dict, use_time: bool) -> np.ndarray:  # override
    sx, sy, sz = cfg.get("scale_xyz", (0.1, 0.1, 0.1))
    # 中位數置中，穩定距離
    xyzc = xyz - np.median(xyz, axis=0, keepdims=True)
    X = np.column_stack([
        xyzc[:, 0] / (sx + 1e-12),
        xyzc[:, 1] / (sy + 1e-12),
        xyzc[:, 2] / (sz + 1e-12),
    ])
    if use_time and t is not None:
        t0 = t - np.median(t)
        # robust 標準化（IQR）
        q25, q75 = np.quantile(t0, [0.25, 0.75])
        iqr_t = max(q75 - q25, 1e-12)
        t_scaled = t0 / iqr_t
        # 若時間幅度壓倒空間，忽略 t
        amp_xyz = np.median(np.linalg.norm(X, axis=1)) + 1e-9
        amp_t = np.median(np.abs(t_scaled)) + 1e-9
        if amp_t / amp_xyz > 50.0:
            return X
        # 否則納入但降權
        t_scaled = t_scaled / 5.0
        return np.column_stack([X, t_scaled])
    else:
        return X


def _adaptive_dbscan(Z: np.ndarray, base_eps: float, base_min_samples: int) -> np.ndarray:
    """用 kNN 中位數估起始 eps，再逐級放大；min_samples ≥ 2 對短軌友好。"""
    if len(Z) >= 5:
        k = min(4, len(Z) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        dists, _ = nbrs.kneighbors(Z)
        dk = dists[:, -1]
        eps_auto = float(np.median(dk) * 1.5)
        eps0 = max(float(base_eps), eps_auto)
    else:
        eps0 = float(base_eps)

    ms = max(2, int(base_min_samples))
    for mult in (1.0, 1.5, 2.0, 3.0):
        labels = DBSCAN(eps=eps0 * mult, min_samples=max(2, ms - 1)).fit_predict(Z)
        if np.any(labels >= 0):
            return labels
    return labels


# —— 取代原本的 _try_split_parallel：加上 k-lines 迭代重分配 + 可選時間輔助 ——
def _try_split_parallel(
    P: np.ndarray,
    t_sub: np.ndarray | None = None,
    min_size: int = 3,
    use_time_weight: float = 0.15,
    max_iter: int = 4,
    sep_thr: float = 1.7,   # 分離度閾值（均值距離 / pooled σ）
) -> list[np.ndarray]:
    """
    對單一 cluster 的點雲 P（N×3）：
    1) PCA 得主方向 u1 及正交方向 u2；
    2) 若沿 u2 分散夠，先用 GMM/KMeans 於 (u2[, t_norm*w]) 分兩組；
    3) 迭代：對每組做 SVD 擬合直線，按『到兩條線的正交距離』重分配點；
    4) 僅當(兩組大小≥min_size 且 分離度≥sep_thr) 才真正拆成兩組。
    回傳一個或兩個 index 子集（以 P 的局部索引計）。
    """
    N = P.shape[0]
    if N < 2 * min_size:
        return [np.arange(N)]

    # PCA 基向量
    C = P.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(P - C, full_matrices=False)
    u1, u2 = Vt[0], Vt[1]
    s1, s2 = S[0], S[1]

    # 正交方向方差太細就唔拆
    if s2 / (s1 + 1e-12) < 0.08:
        return [np.arange(N)]

    # 構造分拆特徵：u2 投影 +（可選）時間
    proj2 = (P - C) @ u2  # N×1
    if t_sub is not None:
        t0 = t_sub - np.median(t_sub)
        q25, q75 = np.quantile(t0, [0.25, 0.75])
        iqr = max(q75 - q25, 1e-12)
        t_feat = (t0 / iqr) * use_time_weight
        feat = np.column_stack([proj2, t_feat])
    else:
        feat = proj2.reshape(-1, 1)

    # 初始二分：GMM -> KMeans 備援
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gm.fit(feat)
        labels = gm.predict(feat)
        mu = gm.means_.ravel() if feat.shape[1] == 1 else gm.means_[:, 0]
        cov = gm.covariances_
        std = np.sqrt(cov.reshape(-1)) if feat.shape[1] == 1 else np.sqrt(cov[:, 0, 0])
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feat)
        labels = km.labels_
        mu = np.array([proj2[labels == 0].mean(), proj2[labels == 1].mean()])
        std = np.array([proj2[labels == 0].std() + 1e-9, proj2[labels == 1].std() + 1e-9])

    # 先驗分離度檢查
    sep = abs(mu[0] - mu[1]) / (np.mean(std) + 1e-9)
    if sep < 1.2:   # 太貼就唔拆
        return [np.arange(N)]

    # 迭代：兩條線模型 → 以線距重分配
    labels_prev = None
    for _ in range(max_iter):
        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        if len(idx0) < min_size or len(idx1) < min_size:
            return [np.arange(N)]

        # 用 SVD 擬合線（點 Ck、方向 dk）
        def fit_line(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Ck = Q.mean(axis=0, keepdims=True)
            _, _, Vt_k = np.linalg.svd(Q - Ck, full_matrices=False)
            dk = Vt_k[0]
            dk = dk / (np.linalg.norm(dk) + 1e-12)
            return Ck.squeeze(0), dk

        C0, d0 = fit_line(P[idx0])
        C1, d1 = fit_line(P[idx1])

        # 對每點計兩條線嘅正交距離，揀近嗰條
        def dist_to_line(pt: np.ndarray, Ck: np.ndarray, dk: np.ndarray) -> float:
            v = pt - Ck
            return np.linalg.norm(v - np.dot(v, dk) * dk)

        d0_all = np.array([dist_to_line(p, C0, d0) for p in P])
        d1_all = np.array([dist_to_line(p, C1, d1) for p in P])

        new_labels = (d1_all < d0_all).astype(int)

        if labels_prev is not None and np.array_equal(new_labels, labels_prev):
            break
        labels_prev = labels.copy()
        labels = new_labels

    # 最終檢查：兩組都夠大，且沿 u2 分離度足夠
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    if len(idx0) < min_size or len(idx1) < min_size:
        return [np.arange(N)]
    mu0, mu1 = proj2[idx0].mean(), proj2[idx1].mean()
    std0, std1 = proj2[idx0].std() + 1e-9, proj2[idx1].std() + 1e-9
    sep_final = abs(mu0 - mu1) / ((std0 + std1) * 0.5)
    if sep_final < sep_thr:
        return [np.arange(N)]

    return [idx0, idx1]



def _cluster_multiscale_union(xyz: np.ndarray, tvec: np.ndarray | None, ms_cfg: list[dict], use_time: bool) -> list[dict]:
    """各 scale 做 DBSCAN → 收集非 -1 群 → Jaccard>0.6 去重 → 對每個群做『平行二分』。"""
    # 1) 多尺度初始群
    clusters = []
    for sid, cfg in enumerate(ms_cfg):
        Z = _build_scaled_features(xyz, tvec, cfg, use_time=use_time)
        labels = _adaptive_dbscan(Z, cfg.get("eps", 0.5), cfg.get("min_samples", 3))
        for tid in np.unique(labels):
            if tid < 0:
                continue
            idxs = np.where(labels == tid)[0]
            if len(idxs) < 2:
                continue
            clusters.append({"scale_id": sid, "idxs": idxs})

    # 2) Jaccard 去重（保留較大者）
    merged = []
    used = [False] * len(clusters)
    for i in range(len(clusters)):
        if used[i]:
            continue
        A = set(clusters[i]["idxs"].tolist())
        group = [i]
        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            B = set(clusters[j]["idxs"].tolist())
            jacc = len(A & B) / max(1, len(A | B))
            if jacc > 0.6:
                used[j] = True
                group.append(j)
        used[i] = True
        best = max(group, key=lambda k: len(clusters[k]["idxs"]))
        merged.append(clusters[best])

    # 3) 對每個群做平行二分（帶 k-lines 重分配）
    final_clusters = []
    for c in merged:
        idxs = c["idxs"]
        P_sub = xyz[idxs, :]
        T_sub = tvec[idxs] if (tvec is not None) else None
        splits = _try_split_parallel(P_sub, t_sub=T_sub, min_size=3, use_time_weight=0.15, max_iter=4, sep_thr=1.7)
        if len(splits) == 1:
            final_clusters.append(c)
        else:
            for sp in splits:
                final_clusters.append({"scale_id": c["scale_id"], "idxs": idxs[sp]})
    return final_clusters

# ======================= END PATCH: Robust clustering =======================



# ==================================================
# 可視化：讀 parquet（XY / ZY）
# ==================================================
def plot_event_xy_zy(tracks_df: pd.DataFrame, preds_df: pd.DataFrame, event_id: int, outdir: Path):
    import matplotlib.pyplot as plt
    sub_tracks = tracks_df[tracks_df["event_id"] == event_id]
    if sub_tracks.empty:
        return
    pred_row = preds_df[preds_df["event_id"] == event_id]
    if pred_row.empty:
        return
    pred = pred_row.iloc[0]

    # XY
    plt.figure(figsize=(6, 6))
    for _, row in sub_tracks.iterrows():
        xs, ys = row["x"], row["y"]
        plt.plot(xs, ys, linewidth=1.2, alpha=0.85)
    plt.scatter([pred["position_x_pred"]], [pred["position_y_pred"]], marker="x", s=50)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Event {event_id} — XY")
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"event_{event_id}_XY.png", dpi=160)
    plt.close()

    # ZY
    plt.figure(figsize=(7, 5))
    for _, row in sub_tracks.iterrows():
        zs, ys = row["z"], row["y"]
        plt.plot(zs, ys, linewidth=1.2, alpha=0.85)
    plt.scatter([pred["position_z_pred"]], [pred["position_y_pred"]], marker="x", s=50)
    plt.xlabel("z [m]")
    plt.ylabel("y [m]")
    plt.title(f"Event {event_id} — ZY")
    plt.tight_layout()
    plt.savefig(outdir / f"event_{event_id}_ZY.png", dpi=160)
    plt.close()


def visualize_all(results_base: Path, n_each: int = 20, out_root: Path = FIGS_BASE):
    res_dirs = sorted([p for p in results_base.glob("*") if p.is_dir()])
    if not res_dirs:
        print("[viz] No results under:", results_base)
        return
    for rd in res_dirs:
        tracks = rd / "tracks_hits.parquet"
        preds = rd / "predictions.parquet"
        if not (tracks.exists() and preds.exists()):
            continue
        tracks_df = pd.read_parquet(tracks)
        preds_df = pd.read_parquet(preds)
        outdir = out_root / rd.name
        event_ids = sorted(tracks_df["event_id"].unique().tolist())
        sample_ids = random.sample(event_ids, min(n_each, len(event_ids)))
        for eid in sample_ids:
            plot_event_xy_zy(tracks_df, preds_df, eid, outdir)
        print(f"[viz] Saved {len(sample_ids)} events to {outdir}")


# ==================================================
# main（無參數，自動掃描與執行）
# ==================================================

def main():
    pl.seed_everything(42, workers=True)

    all_combos = sorted(TRAIN_BASE.glob("compton_*/*"))
    if not all_combos:
        print(f"[main] no combos under {TRAIN_BASE}/compton_*/particle_*")
        return

    for combo in all_combos:
        if not combo.is_dir():
            continue

        mul_tag = combo.parent.name   # e.g. "compton_0"
        res_tag = combo.name          # e.g. "particle_3"
        data_dir = str(combo)
        pred_dir = str((VALID_BASE / combo.relative_to(TRAIN_BASE)).resolve())

        print(f"=== Training for {mul_tag}/{res_tag} ==="f"Data: {data_dir}Validation: {pred_dir}")

        out_dir = train_and_eval_for_dir(
            data_dir=data_dir,
            pred_dir=pred_dir,
            res_tag=res_tag,
            mul_tag=mul_tag,
            max_epochs=MAX_EPOCHS,
            folder=str(RESULTS_BASE),
            use_time=PIPELINE_CONFIG.get("use_time", True),
            config=PIPELINE_CONFIG,
        )
        
        print(f"[main] Saved outputs to: {out_dir}")

if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
