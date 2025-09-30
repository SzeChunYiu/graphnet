#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIBEAM_GNN_two_stage.py  (per-combo version)
--------------------------------------------
Two-stage GNN pipeline with automatic looping over dataset combinations like:
  ./data/training_data/compton_1/particle_3/
  ./data/validation_data/compton_1/particle_3/

For EACH combo folder (same relative path under train/val), we:
  - Stage-1: train node classifier on TRAIN combo, export node probs p for VAL combo
  - Merge p into VAL pulsemaps
  - Stage-2: train vertex regressor on TRAIN combo (optionally also with p) and validate on VAL+ p

You can also run Stage-1 only OR Stage-2 only per-combo.
If no sub-combos are found, the script treats the given directory as a single dataset.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

# GraphNeT / PyG
from graphnet.data.dataset import ParquetDataset
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdge

try:
    from hibeam_det import HIBEAM_Detector
except Exception:
    HIBEAM_Detector = None

# ----------------------------
# Config / CLI
# ----------------------------
import argparse

def build_argparser():
    p = argparse.ArgumentParser(description="Two-stage GNN for HIBEAM (node classifier → vertex regressor), per-combo")
    p.add_argument("--train-dir", type=str, required=True, help="Path to TRAIN root (contains compton_*/*) or a single dataset")
    p.add_argument("--val-dir",   type=str, required=True, help="Path to VAL root (contains compton_*/*) or a single dataset")
    p.add_argument("--out-dir",   type=str, default="./two_stage_out", help="Output directory root")

    # Modes
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage1-train", action="store_true", help="Run Stage-1 only (per combo if combos exist)")
    g.add_argument("--stage2-train", action="store_true", help="Run Stage-2 only (per combo if combos exist)")
    g.add_argument("--run-two-stage", action="store_true", help="Run Stage-1 → merge → Stage-2 (per combo if combos exist)")

    # Model hyperparameters
    p.add_argument("--knn-k", type=int, default=int(os.getenv("KNN_K","8")), help="K for KNNGraph")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs1", type=int, default=30, help="Epochs for Stage-1")
    p.add_argument("--epochs2", type=int, default=50, help="Epochs for Stage-2")
    p.add_argument("--lr1", type=float, default=1e-3, help="LR for Stage-1")
    p.add_argument("--lr2", type=float, default=1e-3, help="LR for Stage-2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data columns
    p.add_argument("--feature-cols-s1", type=str, default='["dom_x","dom_y","dom_z","dom_t","signal_bkg"]',
                   help="JSON list of node columns to use for Stage-1")
    p.add_argument("--feature-cols-s2", type=str, default='["dom_x","dom_y","dom_z","p"]',
                   help="JSON list of node columns to use for Stage-2")
    p.add_argument("--truth-cols", type=str, default='["position_x","position_y","position_z"]',
                   help="JSON list of event-level truth columns (for Stage-2 MSE)")
    p.add_argument("--pulsemaps-name", type=str, default="pulses", help="Pulsemaps name in your parquet datasets")
    p.add_argument("--truth-table", type=str, default="truth", help="Truth table name in your parquet datasets")
    p.add_argument("--index-column", type=str, default="event_id", help="Index column for events")

    # IO
    p.add_argument("--prob-parquet-name", type=str, default="stage1_node_probs.parquet",
                   help="File name for exported p (placed under each combo's stage1 dir)")
    p.add_argument("--merged-val-subdir", type=str, default="val_with_p",
                   help="Subdir (under each combo output) to hold a VAL copy with p merged")
    p.add_argument("--merged-train-subdir", type=str, default="train_with_p",
                   help="Subdir (under each combo output) to hold a TRAIN copy with p merged")

    return p

# ----------------------------
# Utilities
# ----------------------------
def ensure_detector():
    if HIBEAM_Detector is None:
        raise RuntimeError("Could not import HIBEAM_Detector from hibeam_det. Please ensure it's on PYTHONPATH.")
    return HIBEAM_Detector()

def build_graph_definition(detector, knn_k: int):
    return KNNGraph(detector=detector, columns=[0,1,2], nb_nearest_neighbours=knn_k)

def make_dataset(path: str, features: List[str], truth_cols: List[str], graph_definition, index_column: str, pulsemaps_name: str, truth_table: str):
    ds = ParquetDataset(
        path=path,
        pulsemaps=[pulsemaps_name],
        truth_table=truth_table,
        features=features,
        truth=truth_cols,
        graph_definition=graph_definition,
        index_column=index_column,
    )
    return ds

def get_event_ids_from_batch(batch) -> torch.Tensor:
    for attr in ["event_id", "event_ids", "graph_id"]:
        if hasattr(batch, attr):
            return getattr(batch, attr)
    if hasattr(batch, "y"):
        n_graphs = batch.y.shape[0]
        return torch.arange(n_graphs, device=batch.y.device)
    raise AttributeError("Could not find event_id on batch.")

def get_graph_truth_xyz_from_batch(batch) -> torch.Tensor:
    y = getattr(batch, "y", None)
    if y is None:
        for name in ["truth", "labels", "targets"]:
            if hasattr(batch, name):
                y = getattr(batch, name)
                break
    if y is None:
        raise AttributeError("Could not find graph truth (y) on batch.")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] < 3:
        raise ValueError(f"Graph truth has shape {y.shape}, expected >=3 columns.")
    return y[:, :3]

def get_node_labels_from_batch(batch, idx_label: int) -> torch.Tensor:
    x = getattr(batch, "x", None)
    if x is None:
        raise AttributeError("Batch has no .x tensor for node features.")
    return x[:, idx_label].float()

def get_pulse_ids_from_batch(batch) -> Optional[torch.Tensor]:
    for name in ["pulse_id", "pulse_index", "hit_id", "dom_index"]:
        if hasattr(batch, name):
            return getattr(batch, name)
    return None

def collect_node_rows(batch, probs: np.ndarray) -> List[Tuple[int,int,float]]:
    if not hasattr(batch, "batch"):
        raise AttributeError("Batch lacks .batch (graph assignment).")
    gidx = batch.batch.cpu().numpy()  # [num_nodes]
    event_ids = get_event_ids_from_batch(batch).detach().cpu().numpy()
    pulse_ids = get_pulse_ids_from_batch(batch)
    if pulse_ids is None:
        pulse_ids_np = np.arange(len(probs), dtype=int)
    else:
        pulse_ids_np = pulse_ids.detach().cpu().numpy().astype(int)
    rows = [(int(event_ids[gidx[i]]), int(pulse_ids_np[i]), float(probs[i])) for i in range(len(probs))]
    return rows

def merge_node_probs_into_pulsemaps(source_dir: str, out_dir_with_p: str, pulsemaps_name: str, index_column: str, probs_parquet: str):
    out = Path(out_dir_with_p)
    out.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir)

    # Copy everything except pulsemaps
    for item in source_path.iterdir():
        if item.name == f"{pulsemaps_name}.parquet":
            continue
        target = out / item.name
        if item.is_dir():
            import shutil
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.write_bytes(item.read_bytes())

    pm_path = source_path / f"{pulsemaps_name}.parquet"
    if not pm_path.exists():
        raise FileNotFoundError(f"Pulsemaps parquet not found: {pm_path}")

    pulses = pd.read_parquet(pm_path)
    probs  = pd.read_parquet(probs_parquet)

    if "event_id" not in pulses.columns:
        if index_column in pulses.columns:
            pulses = pulses.rename(columns={index_column: "event_id"})
        else:
            raise KeyError("Pulsemaps does not contain event_id.")
    if "pulse_id" not in pulses.columns:
        if "index" in pulses.columns:
            pulses = pulses.rename(columns={"index": "pulse_id"})
        else:
            raise KeyError("Pulsemaps does not contain pulse_id.")

    merged = pulses.merge(probs, how="left", on=["event_id","pulse_id"])
    if "p" not in merged.columns:
        raise KeyError("Merge failed; 'p' column not present after join.")
    merged["p"] = merged["p"].fillna(0.0)

    out_pm = out / f"{pulsemaps_name}.parquet"
    merged.to_parquet(out_pm, index=False)
    print(f"[merge] wrote pulsemaps with p: {out_pm}")
    return str(out)


# ----------------------------
# Feature dimension helper
# ----------------------------
def get_nb_inputs_from_dataset(ds) -> int:
    # Peek one graph and read node feature dimension
    sample = ds[0]
    if not hasattr(sample, "x"):
        raise AttributeError("Dataset sample has no 'x' node feature tensor.")
    if sample.x.ndim != 2:
        raise ValueError(f"Unexpected node feature shape: {tuple(sample.x.shape)}")
    return int(sample.x.shape[1])
# ----------------------------
# Models
# ----------------------------
class NodeClassifier(nn.Module):
    def __init__(self, nb_inputs: int):
        super().__init__()
        self.backbone = DynEdge(nb_inputs=nb_inputs, global_pooling_schemes=[])
        self.head = nn.Sequential(nn.Linear(self.backbone.nb_outputs, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, data):
        node_emb = self.backbone(data)
        return self.head(node_emb).squeeze(-1)

class VertexRegressor(nn.Module):
    def __init__(self, nb_inputs: int):
        super().__init__()
        self.backbone = DynEdge(nb_inputs=nb_inputs, global_pooling_schemes=[])
        self.head = nn.Sequential(nn.Linear(self.backbone.nb_outputs, 128), nn.ReLU(), nn.Linear(128, 3))
    def forward(self, data):
        if not hasattr(data, "batch"):
            raise AttributeError("Batch missing .batch for pooling.")
        from torch_scatter import scatter_mean
        x = self.backbone(data)
        g = scatter_mean(x, data.batch, dim=0)
        return self.head(g)


# ----------------------------
# Device utils
# ----------------------------

def move_to_device(batch, device: str):
    """Send a PyG Data/Batch (or collection) to the target device."""

    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}

    to_fn = getattr(batch, "to", None)
    if callable(to_fn):
        return to_fn(device, non_blocking=True)
    return batch

# ----------------------------
# Training
# ----------------------------
def export_node_probabilities(model: nn.Module, loader, device: str, out_path: Path) -> str:
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            logits = model(batch)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            rows.extend(collect_node_rows(batch, p))

    p_df = pd.DataFrame(rows, columns=["event_id", "pulse_id", "p"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p_df.to_parquet(out_path, index=False)
    print(f"[Stage-1] wrote node probabilities → {out_path}")
    return str(out_path)


def train_stage1(train_dir: str, val_dir: str, out_dir: str, features_stage1: List[str],
                 pulsemaps_name: str, truth_table: str, index_column: str,
                 knn_k: int, device: str, batch_size: int, epochs: int, lr: float,
                 prob_parquet_name: str,
                 export_splits: Sequence[str] = ("val", "train")) -> Dict[str, str]:
    detector = ensure_detector()
    graph_def = build_graph_definition(detector, knn_k)

    ds_train = make_dataset(train_dir, features_stage1, ["position_x","position_y","position_z"],
                            graph_def, index_column, pulsemaps_name, truth_table)
    ds_val   = make_dataset(val_dir,   features_stage1, ["position_x","position_y","position_z"],
                            graph_def, index_column, pulsemaps_name, truth_table)

    from torch_geometric.loader import DataLoader as PyGDataLoader
    common_loader_kwargs = dict(batch_size=batch_size, num_workers=2, persistent_workers=True, pin_memory=False)
    train_loader = PyGDataLoader(ds_train, shuffle=True, **common_loader_kwargs)
    val_loader   = PyGDataLoader(ds_val,   shuffle=False, **common_loader_kwargs)
    train_eval_loader = PyGDataLoader(ds_train, shuffle=False, **common_loader_kwargs)

    nb_inputs = get_nb_inputs_from_dataset(ds_train)
    model = NodeClassifier(nb_inputs=nb_inputs).to(device)
    opt   = Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    if "signal_bkg" not in features_stage1:
        raise ValueError("Stage-1 features must include 'signal_bkg'.")
    idx_label = features_stage1.index("signal_bkg")

    best_val = float("inf")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "stage1_classifier.pt"

    def run_epoch(loader, train=True):
        model.train(train)
        tot, n = 0.0, 0
        for batch in loader:
            batch = move_to_device(batch, device)
            logits = model(batch)
            y = get_node_labels_from_batch(batch, idx_label)
            loss = lossf(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += float(loss.detach().cpu()); n += 1
        return tot/max(1,n)

    for ep in range(epochs):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        print(f"[Stage-1] {train_dir}  ep{ep:03d}  train={tr:.4f}  val={vl:.4f}")
        if vl < best_val - 1e-6:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)

    # Export p for VAL
    print(f"[Stage-1] exporting per-node probabilities")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    requested = set(export_splits)
    results: Dict[str, str] = {}
    if "val" in requested:
        out_parquet = Path(out_dir) / prob_parquet_name
        results["val"] = export_node_probabilities(model, val_loader, device, out_parquet)
    if "train" in requested:
        out_parquet_train = Path(out_dir) / f"train_{prob_parquet_name}"
        results["train"] = export_node_probabilities(model, train_eval_loader, device, out_parquet_train)

    return results

def train_stage2(train_dir: str, val_dir: str, out_dir: str, features_stage2: List[str], truth_cols: List[str],
                 pulsemaps_name: str, truth_table: str, index_column: str,
                 knn_k: int, device: str, batch_size: int, epochs: int, lr: float) -> None:
    detector = ensure_detector()
    graph_def = build_graph_definition(detector, knn_k)

    ds_train = make_dataset(train_dir, features_stage2, truth_cols, graph_def, index_column, pulsemaps_name, truth_table)
    ds_val   = make_dataset(val_dir,   features_stage2, truth_cols, graph_def, index_column, pulsemaps_name, truth_table)

    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=2, persistent_workers=True, pin_memory=False)
    val_loader   = PyGDataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=False)

    nb_inputs = get_nb_inputs_from_dataset(ds_train)
    model = VertexRegressor(nb_inputs=nb_inputs).to(device)
    opt   = Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best_val = float("inf")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "stage2_regressor.pt"

    def run_epoch(loader, train=True):
        model.train(train)
        tot, n = 0.0, 0
        for batch in loader:
            batch = move_to_device(batch, device)
            pred = model(batch)
            truth = get_graph_truth_xyz_from_batch(batch)
            loss = lossf(pred, truth)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += float(loss.detach().cpu()); n += 1
        return tot/max(1,n)

    for ep in range(epochs):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        print(f"[Stage-2] {train_dir}  ep{ep:03d}  train={tr:.5f}  val={vl:.5f}")
        if vl < best_val - 1e-7:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)
    print(f"[Stage-2] best val MSE: {best_val:.6f}; checkpoint saved to {ckpt_path}")

# ----------------------------
# Combo discovery
# ----------------------------
def list_combos(root: Path) -> List[Path]:
    """Return list of combo directories under `root` matching compton_*/* ."""
    combos = sorted(root.glob("compton_*/*"))
    return [c for c in combos if c.is_dir()]

def run_stage1_for_combo(args, rel_path: Path, export_splits: Optional[Sequence[str]] = None):
    train_combo = Path(args.train_dir) / rel_path
    val_combo   = Path(args.val_dir) / rel_path
    out_combo   = Path(args.out_dir) / rel_path / "stage1"
    out_combo.mkdir(parents=True, exist_ok=True)
    features_stage1 = json.loads(args.feature_cols_s1)
    splits = export_splits if export_splits is not None else ("val", "train")
    prob_paths = train_stage1(
        train_dir=str(train_combo),
        val_dir=str(val_combo),
        out_dir=str(out_combo),
        features_stage1=features_stage1,
        pulsemaps_name=args.pulsemaps_name,
        truth_table=args.truth_table,
        index_column=args.index_column,
        knn_k=args.knn_k,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs1,
        lr=args.lr1,
        prob_parquet_name=args.prob_parquet_name,
        export_splits=splits,
    )
    rendered = ", ".join(f"{split}={path}" for split, path in prob_paths.items())
    print(f"[done] Stage-1 {rel_path} → {rendered}")
    return prob_paths

def run_stage2_for_combo(args, rel_path: Path, train_with_p_dir: Optional[str] = None, val_with_p_dir: Optional[str] = None):
    train_combo = Path(train_with_p_dir) if train_with_p_dir is not None else Path(args.train_dir) / rel_path
    val_combo   = Path(val_with_p_dir) if val_with_p_dir is not None else Path(args.val_dir) / rel_path
    out_combo   = Path(args.out_dir) / rel_path / "stage2"
    out_combo.mkdir(parents=True, exist_ok=True)
    train_stage2(
        train_dir=str(train_combo),
        val_dir=str(val_combo),
        out_dir=str(out_combo),
        features_stage2=json.loads(args.feature_cols_s2),
        truth_cols=json.loads(args.truth_cols),
        pulsemaps_name=args.pulsemaps_name,
        truth_table=args.truth_table,
        index_column=args.index_column,
        knn_k=args.knn_k,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs2,
        lr=args.lr2,
    )

# ----------------------------
# Main
# ----------------------------
def main():
    args = build_argparser().parse_args()
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available in this environment.")


    train_root = Path(args.train_dir)
    val_root   = Path(args.val_dir)

    combos = list_combos(train_root)
    if combos:
        # Ensure matching structure exists in VAL
        for combo in combos:
            rel = combo.relative_to(train_root)
            if not (val_root / rel).exists():
                raise FileNotFoundError(f"Validation combo missing: {val_root/rel}")

        if args.stage1_train:
            for combo in combos:
                rel = combo.relative_to(train_root)
                run_stage1_for_combo(args, rel)

        elif args.stage2_train:
            for combo in combos:
                rel = combo.relative_to(train_root)
                # Expect user already merged p into these combos (both TRAIN/VAL as needed)
                run_stage2_for_combo(args, rel)

        elif args.run_two_stage:
            for combo in combos:
                rel = combo.relative_to(train_root)
                print(f"\n=== Combo: {rel} ===")
                prob_paths = run_stage1_for_combo(args, rel)
                val_probs = prob_paths.get("val")
                train_probs = prob_paths.get("train")
                if val_probs is None:
                    raise RuntimeError("Stage-1 did not produce validation probabilities; cannot continue two-stage pipeline.")
                if train_probs is None:
                    raise RuntimeError("Stage-1 did not produce training probabilities; cannot continue two-stage pipeline.")

                merged_val_dir = merge_node_probs_into_pulsemaps(
                    source_dir=str(val_root / rel),
                    out_dir_with_p=str(Path(args.out_dir) / rel / args.merged_val_subdir),
                    pulsemaps_name=args.pulsemaps_name,
                    index_column=args.index_column,
                    probs_parquet=val_probs,
                )
                merged_train_dir = merge_node_probs_into_pulsemaps(
                    source_dir=str(train_root / rel),
                    out_dir_with_p=str(Path(args.out_dir) / rel / args.merged_train_subdir),
                    pulsemaps_name=args.pulsemaps_name,
                    index_column=args.index_column,
                    probs_parquet=train_probs,
                )
                # Stage-2 for this combo (using merged TRAIN + VAL)
                run_stage2_for_combo(
                    args,
                    rel,
                    train_with_p_dir=merged_train_dir,
                    val_with_p_dir=merged_val_dir,
                )
        return

    # Fallback: single dataset mode (no combos under train_dir)
    if args.stage1_train:
        run_stage1_for_combo(args, Path("."))
    elif args.stage2_train:
        run_stage2_for_combo(args, Path("."))
    elif args.run_two_stage:
        print(f"\n=== Single dataset mode ===")
        prob_paths = run_stage1_for_combo(args, Path("."))
        val_probs = prob_paths.get("val")
        train_probs = prob_paths.get("train")
        if val_probs is None or train_probs is None:
            raise RuntimeError("Stage-1 did not produce both train and val probabilities in single dataset mode.")

        merged_val_dir = merge_node_probs_into_pulsemaps(
            source_dir=args.val_dir,
            out_dir_with_p=str(Path(args.out_dir) / args.merged_val_subdir),
            pulsemaps_name=args.pulsemaps_name,
            index_column=args.index_column,
            probs_parquet=val_probs,
        )
        merged_train_dir = merge_node_probs_into_pulsemaps(
            source_dir=args.train_dir,
            out_dir_with_p=str(Path(args.out_dir) / args.merged_train_subdir),
            pulsemaps_name=args.pulsemaps_name,
            index_column=args.index_column,
            probs_parquet=train_probs,
        )
        run_stage2_for_combo(
            args,
            Path("."),
            train_with_p_dir=merged_train_dir,
            val_with_p_dir=merged_val_dir,
        )

if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
