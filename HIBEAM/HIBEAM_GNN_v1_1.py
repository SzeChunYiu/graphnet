from pathlib import Path
from glob import glob
import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam

from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import ParquetDataset
from graphnet.models.graphs import KNNGraph
from graphnet.models.gnn import DynEdge
from graphnet.models import StandardModel
from graphnet.models.task.reconstruction import PositionReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.training.callbacks import PiecewiseLinearLR

# Hyperparameters / runtime
# Graph construction (K for KNN). Change here or set env KNN_K.
KNN_K = int(os.getenv('KNN_K', '8'))
BATCH_SIZE = 64
MAX_EPOCHS = 30
GPUS       = [0]  # []=CPU, [0]=first GPU; use your environment as needed

# Detector and graph definition
from graphnet.models.detector.detector import Detector
from hibeam_det import HIBEAM_Detector

detector = HIBEAM_Detector()
graph_definition = KNNGraph(detector=detector,columns=[0, 1, 2],nb_nearest_neighbours=KNN_K)

# Utilities
from IPython.display import clear_output

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_hist(series, title, xlabel, outpath):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.hist(series, bins=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def train_and_eval_for_dir(DATA_DIR:str,PRED_DIR:str, res_tag:str,mul_tag:str, max_epochs=None,folder=None):

    global BATCH_SIZE, MAX_EPOCHS, GPUS
    if max_epochs is None:
        max_epochs = MAX_EPOCHS

    #res = particle mul = compton
    out_root = ensure_dir(Path("v1_results"+f"/{folder}")/mul_tag /res_tag)

    # 1) Data
    features = ["dom_x", "dom_y", "dom_z", "dom_t","signal_bkg"] #dom_t is disabled as changed in HIBEAM_det.py -> zeros, which should make it irrelavant!
    truth    = ["position_x", "position_y", "position_z"]
    dm = GraphNeTDataModule(
        dataset_reference=ParquetDataset,
        dataset_args={
            "path": DATA_DIR,
            "pulsemaps": ["pulses"],
            "truth_table": "truth",
            "features": features,
            "truth": truth,
            "graph_definition": graph_definition,
            "index_column": "event_id",
        },
        train_dataloader_kwargs={"batch_size": BATCH_SIZE, "num_workers": 2,"persistent_workers": True,"pin_memory": False},
    )
    train_loader = dm.train_dataloader
    val_loader   = dm.val_dataloader

    # 2) Model
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = PositionReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=["position_x", "position_y", "position_z"],
        loss_function=LogCoshLoss(),
    )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, len(train_loader)/2, len(train_loader)*20],
            "factors": [1e-2, 1, 1e-2],
        },
        scheduler_config={"interval": "step"},
    )

    # 3) Train
    
    clear_output(wait=True)
    
    model.fit(
        train_loader,
        val_loader,
        gpus=None if not GPUS else GPUS,
        distribution_strategy="auto",
        max_epochs=max_epochs,
        early_stopping_patience=5,
    )

    # 4) Predict (on same dir; switch to a validation dir if you have one)
    from torch_geometric.loader import DataLoader as PyGDataLoader
    clear_output(wait=True)
    
    dataset = ParquetDataset(
        path = str(PRED_DIR), #PRED_DIR
        pulsemaps=["pulses"],
        truth_table="truth",
        features=features,
        truth=truth,
        graph_definition=graph_definition,
        index_column="event_id",
    )
    inference_loader = PyGDataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False,persistent_workers=True,pin_memory=False)

    predictions = model.predict_as_dataframe(
        inference_loader,
        additional_attributes=["event_id"],
        gpus=GPUS,
    )
    (out_root / "predictions.parquet").write_bytes(predictions.to_parquet())  # ensure overwrite

    # 5) Merge and metrics
    clear_output(wait=True)
    truth_files = sorted(Path(PRED_DIR, "truth").glob("truth_*.parquet")) #PRED_DIR
    truth_df = pd.concat((pd.read_parquet(str(f)) for f in truth_files), ignore_index=True)

    merged = predictions.merge(truth_df, on="event_id", how="inner")
    merged["dx"] = merged["position_x_pred"] - merged["position_x"]
    merged["dy"] = merged["position_y_pred"] - merged["position_y"]
    merged["dz"] = merged["position_z_pred"] - merged["position_z"]
    merged["dist_residual"] = (merged["dx"]**2 + merged["dy"]**2 + merged["dz"]**2) ** 0.5
    
    merged.to_csv(out_root / "predictions.csv")
    
    
    metrics = {
        "mean_abs_dx": float(merged["dx"].abs().mean()),
        "mean_abs_dy": float(merged["dy"].abs().mean()),
        "mean_abs_dz": float(merged["dz"].abs().mean()),
        "mean_dist_residual": float(merged["dist_residual"].mean()),
        "n_events": int(len(merged)),
        "data_dir": str(PRED_DIR), #PRED_DIR
        "res_tag": res_tag,
        "mul_tag": mul_tag,
    }
    (out_root/"metrics.txt").write_text("\n".join(f"{k}: {v}" for k,v in metrics.items()))

    # 6) Plots — filenames include the resolution tag
    save_hist(merged["dist_residual"], f"Distance residual — {res_tag}", "Residual [m]", out_root/f"hist_dist_residual__{res_tag}.png")
    save_hist(merged["dx"]*100, f"dx — {res_tag}", "dx [cm]", out_root/f"hist_dx__{res_tag}.png")
    save_hist(merged["dy"]*100, f"dy — {res_tag}", "dy [cm]", out_root/f"hist_dy__{res_tag}.png")
    save_hist(merged["dz"]*100, f"dz — {res_tag}", "dz [cm]", out_root/f"hist_dz__{res_tag}.png")

    return out_root, metrics

# Discover smeared datasets and run
def main():

    training_set = "training_data"
    validation_set = "validation_data"
    additional_name = "normal"
    folder_name = f"{training_set}_{validation_set}_{additional_name}"

    BASE = Path(f"./data/{training_set}")
    all_metrics = []

    # loop over all compton/particle combinations
    for combo in sorted(BASE.glob("compton_*/*")):
        if not combo.is_dir():
            continue
        mul_tag = combo.parent.name   # e.g. "compton_0"
        res_tag = combo.name          # e.g. "particle_3"

        data_dir = str(combo)
        pred_dir = Path(str(combo).replace(training_set, validation_set))

        print(f"\n=== Training for {mul_tag}/{res_tag} ===\nData: {data_dir}\nValidation: {pred_dir}")
        out_dir, metrics = train_and_eval_for_dir(
            data_dir,
            pred_dir,
            res_tag,
            mul_tag,
            max_epochs=MAX_EPOCHS,
            folder=folder_name,
        )
        all_metrics.append(metrics)
        print(f"Saved outputs to {out_dir}")

    # collect metrics
    metrics_df = pd.DataFrame(all_metrics).sort_values(["mul_tag","res_tag"])
    ensure_dir(Path("v1_results"))
    metrics_df.to_csv(Path("v1_results")/f"{folder_name}/summary_metrics.csv", index=False)
    metrics_df



if __name__ == "__main__":
    # Ensure safe multiprocessing when using DataLoader workers / Lightning spawning.
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # It's fine if the start method was already set.
        pass
    main()
