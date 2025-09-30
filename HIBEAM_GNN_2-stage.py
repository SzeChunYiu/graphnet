#!/usr/bin/env python3
"""Two-stage HIBEAM GraphNeT training pipeline.

This script implements a node classification (Stage-1) followed by a vertex
regression (Stage-2) Graph Neural Network workflow using GraphNeT components
and PyTorch/PyG. It automatically discovers dataset combinations under the
provided training/validation roots, trains one model per combination, and
handles probability merging between stages.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch_geometric.loader import DataLoader

from graphnet.data.dataset import ParquetDataset
from graphnet.exceptions.exceptions import ColumnMissingException

try:
    from graphnet.models.graphs import KNNGraph

    GRAPH_BUILDER_PARAM = "graph_definition"
except ImportError:  # pragma: no cover - compatibility with newer GraphNeT
    from graphnet.models.data_representation import KNNGraph

    GRAPH_BUILDER_PARAM = "data_representation"

from graphnet.models.gnn import DynEdge
from hibeam_det import HIBEAM_Detector


DEFAULT_STAGE1_FEATURES = ["dom_x", "dom_y", "dom_z", "dom_t", "signal_bkg"]
DEFAULT_STAGE2_FEATURES = ["dom_x", "dom_y", "dom_z", "p"]
DEFAULT_TRUTH_COLUMNS = ["position_x", "position_y", "position_z"]
PROBABILITY_COLUMN = "p"


@dataclass
class Paths:
    """Container describing per-combo paths."""

    train: Path
    validation: Path
    output: Path
    stage1_dir: Path
    stage2_dir: Path
    merged_val_dir: Path
    merged_train_dir: Optional[Path]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="HIBEAM two-stage GraphNeT training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--stage1-train", action="store_true", help="Train Stage-1 only")
    mode_group.add_argument("--stage2-train", action="store_true", help="Train Stage-2 only")
    mode_group.add_argument(
        "--run-two-stage",
        action="store_true",
        help="Run Stage-1 followed by Stage-2",
    )

    parser.add_argument("--train-dir", required=True, type=Path, help="Training data root or dataset")
    parser.add_argument("--val-dir", required=True, type=Path, help="Validation data root or dataset")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output root directory")

    parser.add_argument("--knn-k", type=int, default=8, help="Number of nearest neighbours in KNNGraph")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for both stages")
    parser.add_argument("--epochs1", type=int, default=30, help="Epochs for Stage-1 training")
    parser.add_argument("--epochs2", type=int, default=50, help="Epochs for Stage-2 training")
    parser.add_argument("--lr1", type=float, default=1e-3, help="Learning rate for Stage-1")
    parser.add_argument("--lr2", type=float, default=1e-3, help="Learning rate for Stage-2")
    parser.add_argument("--device", type=str, default=default_device, help="Device to use (e.g., cpu, cuda, cuda:0)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")

    parser.add_argument("--pulsemaps-name", type=str, default="pulses", help="Pulse map base name")
    parser.add_argument("--truth-table", type=str, default="truth", help="Truth parquet base name")
    parser.add_argument("--index-column", type=str, default="event_id", help="Event index column")

    parser.add_argument(
        "--feature-cols-s1",
        type=str,
        default=json.dumps(DEFAULT_STAGE1_FEATURES),
        help="JSON list of Stage-1 feature columns",
    )
    parser.add_argument(
        "--feature-cols-s2",
        type=str,
        default=json.dumps(DEFAULT_STAGE2_FEATURES),
        help="JSON list of Stage-2 feature columns",
    )
    parser.add_argument(
        "--truth-cols",
        type=str,
        default=json.dumps(DEFAULT_TRUTH_COLUMNS),
        help="JSON list of Stage-2 truth columns",
    )
    parser.add_argument(
        "--prob-parquet-name",
        type=str,
        default="stage1_node_probs.parquet",
        help="Filename for stored Stage-1 probabilities",
    )
    parser.add_argument(
        "--merged-val-subdir",
        type=str,
        default="val_with_p",
        help="Subdirectory name for validation data with probabilities",
    )
    parser.add_argument(
        "--merge-p-into-train",
        action="store_true",
        help="Also merge Stage-1 probabilities into training data",
    )

    return parser.parse_args()


def parse_json_list(raw: str, name: str) -> List[str]:
    """Parse a JSON list provided as a CLI argument."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"Failed to parse {name} as JSON list: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{name} must be a JSON-encoded list of strings")
    return value


def resolve_device(device_str: str) -> torch.device:
    """Resolve and validate device selection."""

    device_str = device_str.strip()
    if device_str.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested via --device but torch.cuda.is_available() is False."
            )
        return torch.device(device_str)
    return torch.device(device_str)


def is_dataset_dir(path: Path, pulsemaps_name: str, truth_table: str) -> bool:
    """Check whether a path looks like a dataset directory."""

    pulses_file = path / f"{pulsemaps_name}.parquet"
    pulses_dir = path / pulsemaps_name
    truth_file = path / f"{truth_table}.parquet"
    truth_dir = path / truth_table
    pulses_exists = pulses_file.exists() or pulses_dir.is_dir()
    truth_exists = truth_file.exists() or truth_dir.is_dir()
    return path.is_dir() and pulses_exists and truth_exists


def discover_combos(root: Path, pulsemaps_name: str, truth_table: str) -> List[Path]:
    """Discover dataset combinations under ``root``."""

    if is_dataset_dir(root, pulsemaps_name, truth_table):
        return [Path(".")]

    combos: List[Path] = []
    for pulses_file in sorted(root.rglob(f"{pulsemaps_name}.parquet")):
        dataset_dir = pulses_file.parent
        try:
            rel = dataset_dir.relative_to(root)
        except ValueError:
            continue
        if is_dataset_dir(dataset_dir, pulsemaps_name, truth_table):
            combos.append(rel)
    for pulses_dir in sorted(root.rglob(pulsemaps_name)):
        dataset_dir = pulses_dir.parent
        try:
            rel = dataset_dir.relative_to(root)
        except ValueError:
            continue
        if is_dataset_dir(dataset_dir, pulsemaps_name, truth_table):
            if rel not in combos:
                combos.append(rel)
    return sorted(combos)


def ensure_dataset(path: Path, pulsemaps_name: str, truth_table: str) -> None:
    """Validate dataset directory exists and contains required tables."""

    if not is_dataset_dir(path, pulsemaps_name, truth_table):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected pulsemaps '{pulsemaps_name}' and truth '{truth_table}'."
        )


def build_graph_definition(features: Sequence[str], knn_k: int) -> KNNGraph:
    """Construct a KNNGraph configured for the provided features."""

    detector = HIBEAM_Detector()
    xyz_indices = [i for i, name in enumerate(features) if name in ("dom_x", "dom_y", "dom_z")]
    if len(xyz_indices) < 3:
        xyz_indices = list(range(min(3, len(features))))
        print(
            "[WARN] Using default coordinate columns for KNN graph construction. "
            "Ensure dom_x/dom_y/dom_z are present in feature list."
        )
    return KNNGraph(
        detector=detector,
        nb_nearest_neighbours=knn_k,
        input_feature_names=list(features),
        columns=xyz_indices[:3] if xyz_indices else [0],
    )


def create_dataset(
    dataset_path: Path,
    features: Sequence[str],
    truth_columns: Sequence[str],
    index_column: str,
    pulsemaps_name: str,
    truth_table: str,
    graph_definition: KNNGraph,
) -> ParquetDataset:
    """Create a :class:`ParquetDataset` with the requested configuration."""

    dataset_kwargs = dict(
        path=str(dataset_path),
        pulsemaps=[pulsemaps_name],
        features=list(features),
        truth=list(truth_columns),
        index_column=index_column,
        truth_table=truth_table,
    )
    dataset_kwargs[GRAPH_BUILDER_PARAM] = graph_definition
    try:
        dataset = ParquetDataset(**dataset_kwargs)
    except ColumnMissingException as exc:
        raise RuntimeError(
            f"Dataset at {dataset_path} is missing required columns: {exc}"
        ) from exc
    return dataset


def create_dataloader(
    dataset: ParquetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Construct a PyG DataLoader with safe defaults."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
    )


def move_to_device(batch, device: torch.device):
    """Move a PyG batch to the desired device."""

    batch = batch.to(device)
    if hasattr(batch, "keys"):
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value):
                batch[key] = value.to(device)
    for attr in ("x", "edge_attr", "batch", "pos"):
        if hasattr(batch, attr):
            value = getattr(batch, attr)
            if torch.is_tensor(value):
                setattr(batch, attr, value.to(device))
    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        batch.edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    return batch


def assert_on_device(batch, device: torch.device) -> None:
    """Assert required tensors reside on the expected device."""

    if hasattr(batch, "x"):
        assert batch.x.device == device, "batch.x not on the requested device"
    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        assert batch.edge_index.device == device, "edge_index not on the requested device"
        assert batch.edge_index.dtype == torch.long, "edge_index must be torch.long"


class Stage1Classifier(nn.Module):
    """Node-level binary classifier based on DynEdge."""

    def __init__(self, input_dim: int, nb_neighbours: int) -> None:
        super().__init__()
        self.backbone = DynEdge(
            nb_inputs=input_dim,
            nb_neighbours=nb_neighbours,
            global_pooling_schemes=None,
        )
        self.head = nn.Linear(self.backbone.nb_outputs, 1)

    def forward(self, batch) -> Tensor:
        features = self.backbone(batch)
        logits = self.head(features).squeeze(-1)
        return logits


class Stage2Regressor(nn.Module):
    """Graph-level regressor predicting (x, y, z) vertex."""

    def __init__(self, input_dim: int, nb_neighbours: int) -> None:
        super().__init__()
        self.backbone = DynEdge(
            nb_inputs=input_dim,
            nb_neighbours=nb_neighbours,
            global_pooling_schemes=["mean"],
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.nb_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, batch) -> Tensor:
        pooled = self.backbone(batch)
        return self.head(pooled)


def extract_stage1_labels(batch, label_column: str) -> Tensor:
    """Extract node-level labels for Stage-1 training."""

    if hasattr(batch, label_column):
        labels = getattr(batch, label_column)
    else:
        raise AttributeError(
            f"Batch does not contain label column '{label_column}'."
        )
    if labels.ndim > 1:
        labels = labels.view(-1)
    return labels.float()


def extract_stage2_targets(batch, truth_columns: Sequence[str]) -> Tensor:
    """Extract graph-level regression targets."""

    targets: List[Tensor] = []
    for column in truth_columns:
        if not hasattr(batch, column):
            raise AttributeError(f"Batch missing required truth column '{column}'.")
        value = getattr(batch, column)
        if value.ndim == 0:
            value = value.view(1, 1)
        elif value.ndim == 1:
            value = value.view(-1, 1)
        targets.append(value.float())
    return torch.cat(targets, dim=1)


def train_stage1(
    model: Stage1Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    label_column: str,
) -> Tuple[Dict[str, List[float]], Dict[str, Tensor]]:
    """Train Stage-1 classifier and return history and best state dict."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = math.inf
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            assert_on_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            labels = extract_stage1_labels(batch, label_column)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.numel()
            train_loss += loss.item() * batch_size
            n_samples += batch_size
        avg_train = train_loss / max(n_samples, 1)

        val_loss = evaluate_stage1(model, val_loader, device, criterion, label_column)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        print(
            f"[Stage-1] Epoch {epoch:03d}/{epochs:03d} - Train Loss: {avg_train:.4f} - "
            f"Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return history, best_state


def evaluate_stage1(
    model: Stage1Classifier,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    label_column: str,
) -> float:
    """Evaluate Stage-1 model on validation data."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            assert_on_device(batch, device)
            logits = model(batch)
            labels = extract_stage1_labels(batch, label_column)
            loss = criterion(logits, labels)
            count = labels.numel()
            total_loss += loss.item() * count
            total_samples += count
    return total_loss / max(total_samples, 1)


def train_stage2(
    model: Stage2Regressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    truth_columns: Sequence[str],
) -> Tuple[Dict[str, List[float]], Dict[str, Tensor]]:
    """Train Stage-2 regressor and return history and best state dict."""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = math.inf
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_graphs = 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            assert_on_device(batch, device)
            optimizer.zero_grad()
            preds = model(batch)
            targets = extract_stage2_targets(batch, truth_columns)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            graphs = targets.shape[0]
            train_loss += loss.item() * graphs
            n_graphs += graphs
        avg_train = train_loss / max(n_graphs, 1)

        val_loss = evaluate_stage2(model, val_loader, device, criterion, truth_columns)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        print(
            f"[Stage-2] Epoch {epoch:03d}/{epochs:03d} - Train Loss: {avg_train:.4f} - "
            f"Val Loss: {val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return history, best_state


def evaluate_stage2(
    model: Stage2Regressor,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    truth_columns: Sequence[str],
) -> float:
    """Evaluate Stage-2 model."""

    model.eval()
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            assert_on_device(batch, device)
            preds = model(batch)
            targets = extract_stage2_targets(batch, truth_columns)
            loss = criterion(preds, targets)
            graphs = targets.shape[0]
            total_loss += loss.item() * graphs
            total_graphs += graphs
    return total_loss / max(total_graphs, 1)


def run_stage1_inference(
    model: Stage1Classifier,
    data_loader: DataLoader,
    device: torch.device,
    index_column: str,
    probability_column: str,
) -> pd.DataFrame:
    """Run Stage-1 inference and collect node probabilities."""

    model.eval()
    records: List[Dict[str, float]] = []
    event_counters: Dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for batch in data_loader:
            batch = move_to_device(batch, device)
            assert_on_device(batch, device)
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            node_to_event = batch.batch.detach().cpu().numpy()
            event_tensor = getattr(batch, index_column)
            event_ids = event_tensor.detach().cpu().numpy().reshape(-1)

            if hasattr(batch, "pulse_id"):
                pulse_attr = batch.pulse_id.detach().cpu().numpy()
            elif hasattr(batch, "pulse_index"):
                pulse_attr = batch.pulse_index.detach().cpu().numpy()
            else:
                pulse_attr = None

            for graph_idx, event_id in enumerate(event_ids):
                mask = node_to_event == graph_idx
                if not np.any(mask):
                    continue
                node_probs = probs[mask]
                if pulse_attr is not None:
                    pulse_ids = pulse_attr[mask].astype(np.int64)
                else:
                    start = event_counters[int(event_id)]
                    pulse_ids = np.arange(start, start + node_probs.size, dtype=np.int64)
                    event_counters[int(event_id)] = start + node_probs.size
                for pid, prob in zip(pulse_ids, node_probs):
                    records.append(
                        {
                            index_column: int(event_id),
                            "pulse_id": int(pid),
                            probability_column: float(prob),
                        }
                    )
    return pd.DataFrame.from_records(records)


def ensure_pulse_ids(df: pd.DataFrame, index_column: str) -> pd.DataFrame:
    """Ensure the pulse DataFrame contains a ``pulse_id`` column."""

    if "pulse_id" not in df.columns:
        df = df.copy()
        df["pulse_id"] = df.groupby(index_column).cumcount()
    return df


def pulses_have_column(dataset_path: Path, pulsemaps_name: str, column: str) -> bool:
    """Check if pulses table(s) contain a specific column."""

    file_path = dataset_path / f"{pulsemaps_name}.parquet"
    if file_path.exists():
        try:
            pd.read_parquet(file_path, columns=[column])
            return True
        except (KeyError, ValueError):
            return False

    dir_path = dataset_path / pulsemaps_name
    if dir_path.is_dir():
        for parquet_file in sorted(dir_path.glob("*.parquet")):
            try:
                pd.read_parquet(parquet_file, columns=[column])
                return True
            except (KeyError, ValueError):
                return False
    return False


def build_zero_probability_dataframe(
    dataset_path: Path,
    pulsemaps_name: str,
    index_column: str,
    probability_column: str,
) -> pd.DataFrame:
    """Construct a probability DataFrame filled with zeros for all pulses."""

    file_path = dataset_path / f"{pulsemaps_name}.parquet"
    dir_path = dataset_path / pulsemaps_name

    if file_path.exists():
        pulses_df = ensure_pulse_ids(pd.read_parquet(file_path), index_column)
    elif dir_path.is_dir():
        parts = [pd.read_parquet(parquet_file) for parquet_file in sorted(dir_path.glob("*.parquet"))]
        pulses_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        pulses_df = ensure_pulse_ids(pulses_df, index_column)
    else:
        raise FileNotFoundError(
            f"Could not locate pulses data in {dataset_path} when constructing zero probabilities."
        )

    if pulses_df.empty:
        return pd.DataFrame(columns=[index_column, "pulse_id", probability_column])

    zero_df = pulses_df[[index_column, "pulse_id"]].copy()
    zero_df[probability_column] = 0.0
    return zero_df


def merge_probabilities_into_dataset(
    src_dataset: Path,
    dest_dataset: Path,
    pulsemaps_name: str,
    truth_table: str,
    prob_df: pd.DataFrame,
    index_column: str,
    probability_column: str,
) -> None:
    """Copy dataset and merge probabilities into pulse map."""

    if dest_dataset.exists():
        shutil.rmtree(dest_dataset)
    shutil.copytree(src_dataset, dest_dataset)
    prob_df = prob_df.copy()
    if probability_column not in prob_df.columns:
        raise KeyError(f"Probability DataFrame missing column '{probability_column}'.")
    prob_df = prob_df[[index_column, "pulse_id", probability_column]]

    pulses_file = dest_dataset / f"{pulsemaps_name}.parquet"
    pulses_dir = dest_dataset / pulsemaps_name

    if pulses_file.exists():
        pulses_df = pd.read_parquet(pulses_file)
        pulses_df = ensure_pulse_ids(pulses_df, index_column)
        merged = pulses_df.merge(prob_df, on=[index_column, "pulse_id"], how="left")
        merged[probability_column] = merged[probability_column].fillna(0.0)
        merged.to_parquet(pulses_file, index=False)
    elif pulses_dir.is_dir():
        for parquet_file in sorted(pulses_dir.glob("*.parquet")):
            part_df = pd.read_parquet(parquet_file)
            part_df = ensure_pulse_ids(part_df, index_column)
            part_merged = part_df.merge(prob_df, on=[index_column, "pulse_id"], how="left")
            part_merged[probability_column] = part_merged[probability_column].fillna(0.0)
            part_merged.to_parquet(parquet_file, index=False)
    else:
        raise FileNotFoundError(
            f"No pulses data found in {dest_dataset} for '{pulsemaps_name}'."
        )

    truth_src = src_dataset / f"{truth_table}.parquet"
    if truth_src.exists():
        shutil.copy2(truth_src, dest_dataset / f"{truth_table}.parquet")


def configure_paths(
    combo: Path,
    train_root: Path,
    val_root: Path,
    out_root: Path,
    merged_val_subdir: str,
    merge_train: bool,
) -> Paths:
    """Construct path container for a dataset combination."""

    train_path = (train_root / combo).resolve() if combo != Path(".") else train_root.resolve()
    val_path = (val_root / combo).resolve() if combo != Path(".") else val_root.resolve()
    combo_out = (out_root / combo).resolve() if combo != Path(".") else out_root.resolve()
    stage1_dir = combo_out / "stage1"
    stage2_dir = combo_out / "stage2"
    merged_val_dir = combo_out / merged_val_subdir
    merged_train_dir = combo_out / "train_with_p" if merge_train else None
    return Paths(
        train=train_path,
        validation=val_path,
        output=combo_out,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        merged_val_dir=merged_val_dir,
        merged_train_dir=merged_train_dir,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    """Entry point for executing the requested pipeline mode."""

    features_s1 = parse_json_list(args.feature_cols_s1, "feature-cols-s1")
    features_s2 = parse_json_list(args.feature_cols_s2, "feature-cols-s2")
    truth_columns = parse_json_list(args.truth_cols, "truth-cols")

    if "signal_bkg" not in features_s1:
        raise ValueError("Stage-1 feature list must include 'signal_bkg'.")
    if PROBABILITY_COLUMN not in features_s2:
        raise ValueError(f"Stage-2 feature list must include '{PROBABILITY_COLUMN}'.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_root = args.train_dir.resolve()
    val_root = args.val_dir.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    combos = discover_combos(train_root, args.pulsemaps_name, args.truth_table)
    if not combos:
        raise RuntimeError(f"No datasets found under {train_root}.")

    single_val_dataset = is_dataset_dir(val_root, args.pulsemaps_name, args.truth_table)
    if single_val_dataset and len(combos) != 1:
        raise RuntimeError(
            "Validation directory points to a single dataset but multiple training combos were found."
        )

    for combo in combos:
        print(f"========== Processing combo: {combo.as_posix()} ==========")
        paths = configure_paths(
            combo,
            train_root,
            val_root,
            out_root,
            args.merged_val_subdir,
            args.merge_p_into_train,
        )
        ensure_dataset(paths.train, args.pulsemaps_name, args.truth_table)
        ensure_dataset(paths.validation, args.pulsemaps_name, args.truth_table)
        paths.output.mkdir(parents=True, exist_ok=True)
        if args.stage1_train or args.run_two_stage:
            run_stage1_for_combo(
                paths,
                features_s1,
                truth_columns,
                args,
                device,
            )
        if args.stage2_train or args.run_two_stage:
            run_stage2_for_combo(
                paths,
                features_s2,
                truth_columns,
                args,
                device,
            )


def run_stage1_for_combo(
    paths: Paths,
    features: Sequence[str],
    truth_columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train Stage-1 for a dataset combination and merge validation probabilities."""

    paths.stage1_dir.mkdir(parents=True, exist_ok=True)
    graph_definition = build_graph_definition(features, args.knn_k)

    train_dataset = create_dataset(
        paths.train,
        features,
        truth_columns,
        args.index_column,
        args.pulsemaps_name,
        args.truth_table,
        graph_definition,
    )
    val_dataset = create_dataset(
        paths.validation,
        features,
        truth_columns,
        args.index_column,
        args.pulsemaps_name,
        args.truth_table,
        build_graph_definition(features, args.knn_k),
    )

    train_loader = create_dataloader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = create_dataloader(val_dataset, args.batch_size, False, args.num_workers)

    model = Stage1Classifier(input_dim=len(features), nb_neighbours=args.knn_k).to(device)
    history, best_state = train_stage1(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs1,
        args.lr1,
        label_column="signal_bkg",
    )
    model.load_state_dict(best_state)
    checkpoint_path = paths.stage1_dir / "stage1_classifier.pt"
    torch.save(best_state, checkpoint_path)
    print(f"Saved Stage-1 model to {checkpoint_path}")

    val_infer_loader = create_dataloader(val_dataset, args.batch_size, False, args.num_workers)
    prob_df = run_stage1_inference(
        model,
        val_infer_loader,
        device,
        args.index_column,
        PROBABILITY_COLUMN,
    )
    prob_path = paths.stage1_dir / args.prob_parquet_name
    prob_df.to_parquet(prob_path, index=False)
    print(f"Stored validation node probabilities at {prob_path}")

    merge_probabilities_into_dataset(
        paths.validation,
        paths.merged_val_dir,
        args.pulsemaps_name,
        args.truth_table,
        prob_df,
        args.index_column,
        PROBABILITY_COLUMN,
    )
    print(f"Validation dataset with probabilities stored at {paths.merged_val_dir}")

    if args.merge_p_into_train:
        train_infer_loader = create_dataloader(train_dataset, args.batch_size, False, args.num_workers)
        train_prob_df = run_stage1_inference(
            model,
            train_infer_loader,
            device,
            args.index_column,
            PROBABILITY_COLUMN,
        )
        if paths.merged_train_dir is None:
            raise RuntimeError("Merged train directory expected but missing.")
        merge_probabilities_into_dataset(
            paths.train,
            paths.merged_train_dir,
            args.pulsemaps_name,
            args.truth_table,
            train_prob_df,
            args.index_column,
            PROBABILITY_COLUMN,
        )
        print(f"Training dataset with probabilities stored at {paths.merged_train_dir}")


def run_stage2_for_combo(
    paths: Paths,
    features: Sequence[str],
    truth_columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train Stage-2 regressor for a dataset combination."""

    stage2_train_path = (
        paths.merged_train_dir
        if args.merge_p_into_train
        and paths.merged_train_dir is not None
        and paths.merged_train_dir.exists()
        else paths.train
    )
    stage2_val_path = paths.merged_val_dir if paths.merged_val_dir.exists() else paths.validation

    paths.stage2_dir.mkdir(parents=True, exist_ok=True)

    if not pulses_have_column(stage2_train_path, args.pulsemaps_name, PROBABILITY_COLUMN):
        print(
            "[WARN] Training pulses lack probability column 'p'. Using zeros; consider --merge-p-into-train."
        )
        zero_prob_df = build_zero_probability_dataframe(
            stage2_train_path,
            args.pulsemaps_name,
            args.index_column,
            PROBABILITY_COLUMN,
        )
        zero_train_dir = paths.output / "train_with_zero_p"
        merge_probabilities_into_dataset(
            stage2_train_path,
            zero_train_dir,
            args.pulsemaps_name,
            args.truth_table,
            zero_prob_df,
            args.index_column,
            PROBABILITY_COLUMN,
        )
        stage2_train_path = zero_train_dir

    graph_definition_train = build_graph_definition(features, args.knn_k)
    graph_definition_val = build_graph_definition(features, args.knn_k)

    train_dataset = create_dataset(
        stage2_train_path,
        features,
        truth_columns,
        args.index_column,
        args.pulsemaps_name,
        args.truth_table,
        graph_definition_train,
    )
    val_dataset = create_dataset(
        stage2_val_path,
        features,
        truth_columns,
        args.index_column,
        args.pulsemaps_name,
        args.truth_table,
        graph_definition_val,
    )

    train_loader = create_dataloader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = create_dataloader(val_dataset, args.batch_size, False, args.num_workers)

    model = Stage2Regressor(input_dim=len(features), nb_neighbours=args.knn_k).to(device)
    history, best_state = train_stage2(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs2,
        args.lr2,
        truth_columns,
    )
    model.load_state_dict(best_state)
    checkpoint_path = paths.stage2_dir / "stage2_regressor.pt"
    torch.save(best_state, checkpoint_path)
    print(f"Saved Stage-2 model to {checkpoint_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
