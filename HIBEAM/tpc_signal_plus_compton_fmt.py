#!/usr/bin/env python3
"""
tpc_signal_plus_compton_fmt.py  (drop empty events + verified kinematics)
-----------------------------------------------------------------------
Writes per file index i:
  <out>/pulses/pulses_i.parquet
  <out>/truth/truth_i.parquet
  <out>/truth/truth_tracks/truth_tracks_i.parquet

Behavior:
- SIGNAL: all tracks per event share a common vertex sampled uniformly in a target cylinder
          (r ≤ 0.20 m, z ∈ [-0.04, 0.04]). Directions are drawn uniformly over solid angle
          within cones of half-angle 45° about +y or -y (chosen 50/50). Each signal propagates
          from the inner-wall entry point (r = R_INNER).
- COMPTON: born on inner/outer cylindrical barrel surfaces at random (phi, z), with directions
           drawn isotropically over 4π and accepted only if a tiny forward step goes deeper
           into the TPC volume (so they move inward for outer, outward for inner).

- DROP EMPTY EVENTS: If an event generates zero hits in total (no signal and no compton hits),
  that event is skipped and **NOT** written to truth. We keep generating events until we have
  EXACTLY `events_per_file` kept events (bounded retries to avoid infinite loops).

- Output directory: data_set/compton_data/compton_{n_signal}/particle_{n_compton}
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# -------------------------------
# Geometry defaults (meters)
# -------------------------------
R_INNER_DEFAULT = 0.22
R_OUTER_DEFAULT = 0.32
TPC_HALF_LENGTH_DEFAULT = 0.516 / 2.0
N_LAYERS_DEFAULT = 10

def default_layer_radii(r_in: float, r_out: float, n_layers: int) -> np.ndarray:
    return np.linspace(r_in, r_out, n_layers)

# -------------------------------
# I/O helpers with Parquet fallback
# -------------------------------
def safe_to_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
        return path
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[fallback] {path.name} -> {csv_path.name} ({e})")
        return csv_path

# -------------------------------
# Math utils
# -------------------------------
def segment_cylinder_crossings(p0: np.ndarray, p1: np.ndarray, r_layer: float, z_half: float) -> List[np.ndarray]:
    d = p1 - p0
    dx, dy = float(d[0]), float(d[1])
    A = dx*dx + dy*dy
    if A < 1e-16:
        return []
    B = 2.0*(p0[0]*dx + p0[1]*dy)
    C = p0[0]*p0[0] + p0[1]*p0[1] - r_layer*r_layer
    disc = B*B - 4.0*A*C
    if disc < 0.0:
        return []
    sqrt_disc = math.sqrt(max(0.0, disc))
    t1 = (-B - sqrt_disc)/(2.0*A)
    t2 = (-B + sqrt_disc)/(2.0*A)
    ts = []
    if 0.0 <= t1 <= 1.0: ts.append(t1)
    if 0.0 <= t2 <= 1.0 and not math.isclose(t2, t1): ts.append(t2)
    pts: List[np.ndarray] = []
    for t in ts:
        p = p0 + t*d
        if abs(p[2]) <= z_half + 1e-12:
            pts.append(p)
    return pts

def march_until_boundary(p0: np.ndarray, u: np.ndarray, step_m: float,
                         r_in: float, r_out: float, z_half: float,
                         max_steps: int = 10000) -> np.ndarray:
    pts = [p0.copy()]
    p = p0.copy()
    for _ in range(max_steps):
        p_next = p + step_m*u
        r_next = math.hypot(p_next[0], p_next[1])
        if (r_next < r_in) or (r_next > r_out) or (abs(p_next[2]) > z_half):
            break
        pts.append(p_next)
        p = p_next
    return np.asarray(pts)

def polyline_layer_crossings(poly: np.ndarray, layer_radii: Sequence[float], z_half: float) -> List[Tuple[np.ndarray, int, float]]:
    s_vert = [0.0]
    for i in range(1, len(poly)):
        s_vert.append(s_vert[-1] + float(np.linalg.norm(poly[i]-poly[i-1])))
    out: List[Tuple[np.ndarray,int,float]] = []
    for i in range(len(poly)-1):
        p0, p1 = poly[i], poly[i+1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-12: continue
        for li, rL in enumerate(layer_radii):
            for p in segment_cylinder_crossings(p0, p1, float(rL), float(z_half)):
                t = float(np.dot(p - p0, d) / (L*L)) if L > 0 else 0.0
                t = max(0.0, min(1.0, t))
                s = s_vert[i] + t*L
                out.append((p, li, s))
    out.sort(key=lambda x: x[2])
    return out

# -------------------------------
# Sampling utilities
# -------------------------------
def sample_target_vertex(r_target: float, zmin: float, zmax: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform in a cylinder of radius r_target, z∈[zmin,zmax]."""
    u = rng.uniform(0.0, 1.0)
    r = math.sqrt((r_target*r_target)*u)
    phi = rng.uniform(0.0, 2.0*math.pi)
    z = rng.uniform(zmin, zmax)
    return np.array([r*math.cos(phi), r*math.sin(phi), z], dtype=float)

def isotropic_direction(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3); v /= (np.linalg.norm(v)+1e-12); return v

def sample_point_on_surface(r_in: float, r_out: float, z_half: float, surface: str, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0*math.pi); z = rng.uniform(-z_half, z_half)
    r = r_out if surface == "outer" else r_in
    return np.array([r*math.cos(phi), r*math.sin(phi), z], dtype=float)

def initial_direction_for_surface(surface: str, p0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    4π isotropic; accept only if a small forward step from the surface point
    goes deeper into the TPC volume (inward for outer, outward for inner).
    """
    eps = 1e-4  # meters
    def inside_tpc(p: np.ndarray, r_in: float, r_out: float, z_half: float) -> bool:
        r = math.hypot(p[0], p[1])
        return (r_in <= r <= r_out) and (abs(p[2]) <= z_half)

    r_hat = np.array([p0[0], p0[1], 0.0]); r_hat /= (np.linalg.norm(r_hat) + 1e-12)
    want_sign = -1.0 if surface == "outer" else (+1.0 if surface == "inner" else None)
    if want_sign is None:
        raise ValueError("surface must be 'inner' or 'outer'")

    r_in  = R_INNER_DEFAULT
    r_out = R_OUTER_DEFAULT
    z_half = TPC_HALF_LENGTH_DEFAULT

    for _ in range(10000):
        u = rng.normal(size=3); u /= (np.linalg.norm(u) + 1e-12)
        if want_sign * np.dot(u, r_hat) <= 0:
            continue
        p_next = p0 + eps * u
        if inside_tpc(p_next, r_in, r_out, z_half):
            return u
    return (-r_hat) if surface == "outer" else (+r_hat)

def ray_cylinder_entry(p0: np.ndarray, u: np.ndarray, r_layer: float, z_half: float):
    ux, uy = float(u[0]), float(u[1])
    A = ux*ux + uy*uy
    if A < 1e-16:
        return None
    B = 2.0*(p0[0]*ux + p0[1]*uy)
    C = p0[0]*p0[0] + p0[1]*p0[1] - r_layer*r_layer
    disc = B*B - 4.0*A*C
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(max(0.0, disc))
    t_candidates = [(-B - sqrt_disc)/(2.0*A), (-B + sqrt_disc)/(2.0*A)]
    t_pos = [t for t in t_candidates if t > 0.0]
    if not t_pos:
        return None
    t_entry = min(t_pos)
    p_entry = p0 + t_entry*u
    if abs(p_entry[2]) <= z_half + 1e-12:
        return p_entry
    return None

# -------------------------------
# Track builders (emit pulses rows; truth built at event level and per track)
# -------------------------------
def build_track_pulses(poly: np.ndarray, layer_radii: np.ndarray, z_half: float,
                       step_m: float, beta: float, t0_ns: float,
                       rng: np.random.Generator) -> List[dict]:
    hits = polyline_layer_crossings(poly, layer_radii, z_half)
    if not hits:
        step = max(1, len(poly)//10)
        hits = [(poly[i], -1, float(i)*step_m) for i in range(0, len(poly), step)]
    C_M_PER_NS = 0.299792458
    v = beta*C_M_PER_NS
    rows = []
    for (_, li, s), (pt, _, _) in zip(hits, hits):
        rows.append({
            "dom_x": float(pt[0]), "dom_y": float(pt[1]), "dom_z": float(pt[2]),
            "dom_t": float(t0_ns + s/v),
        })
    return rows

def build_signal_tracks(n_tracks: int, layer_radii: np.ndarray, rng: np.random.Generator,
                        r_in: float, r_out: float, z_half: float,
                        step_m: float, beta: float,
                        common_vertex: np.ndarray):
    tracks = []
    truth_tracks = []
    for _ in range(n_tracks):
        p0 = common_vertex.copy()

        # Direction within ±45° about +y or -y (uniform over solid angle within cone)
        theta_max = np.deg2rad(45.0); cos_max = np.cos(theta_max)
        if rng.random() < 0.5:
            cos_theta = rng.uniform(cos_max, 1.0); theta = np.arccos(cos_theta); phi = rng.uniform(0.0, 2.0*np.pi)
            u = np.array([np.sin(theta)*np.cos(phi),  cos_theta, np.sin(theta)*np.sin(phi)])
        else:
            cos_theta = rng.uniform(cos_max, 1.0); theta = np.arccos(cos_theta); phi = rng.uniform(0.0, 2.0*np.pi)
            u = np.array([np.sin(theta)*np.cos(phi), -cos_theta, np.sin(theta)*np.sin(phi)])

        # Enter inner cylinder and propagate from there
        p_entry = ray_cylinder_entry(p0, u, r_in, z_half)
        if p_entry is None:
            continue

        poly = march_until_boundary(p_entry, u, step_m, r_in, r_out, z_half)
        if len(poly) < 2:
            continue

        t0 = float(rng.uniform(0.0, 50.0))
        rows = build_track_pulses(poly, layer_radii, z_half, step_m, beta, t0, rng)
        if rows:
            tracks.append(rows)
            truth_tracks.append({
                "particle": "signal",
                "start_x": float(p0[0]), "start_y": float(p0[1]), "start_z": float(p0[2]),
                "dir_x": float(u[0]), "dir_y": float(u[1]), "dir_z": float(u[2]),
                "n_hits": int(len(rows)),
            })
    return tracks, truth_tracks

def build_compton_tracks(n_tracks: int, surface_mode: str, p_inner: float,
                         layer_radii: np.ndarray, rng: np.random.Generator,
                         r_in: float, r_out: float, z_half: float,
                         step_m: float, beta: float):
    tracks = []
    truth_tracks = []
    for _ in range(n_tracks):
        surface = surface_mode
        if surface_mode == "both":
            surface = "inner" if rng.random() < p_inner else "outer"
        p0 = sample_point_on_surface(r_in, r_out, z_half, surface, rng)

        # 4π direction, accept only if stepping into TPC
        u  = initial_direction_for_surface(surface, p0, rng)

        poly = march_until_boundary(p0, u, step_m, r_in, r_out, z_half)
        if len(poly) < 2:
            continue
        t0 = float(rng.uniform(0.0, 50.0))
        rows = build_track_pulses(poly, layer_radii, z_half, step_m, beta, t0, rng)
        if rows:
            tracks.append(rows)
            truth_tracks.append({
                "particle": "compton_electron",
                "surface": surface,
                "start_x": float(p0[0]), "start_y": float(p0[1]), "start_z": float(p0[2]),
                "dir_x": float(u[0]), "dir_y": float(u[1]), "dir_z": float(u[2]),
                "n_hits": int(len(rows)),
            })
    return tracks, truth_tracks

# -------------------------------
# Dataset writer
# -------------------------------
def generate_dataset(out_dir: Path,
                     n_files: int, events_per_file: int,
                     n_signal_per_event: int,
                     n_compton_per_event: int, surface_mode: str, p_inner: float,
                     r_in: float, r_out: float, z_half: float, n_layers: int,
                     step_cm: float, beta: float, seed: int) -> None:
    pulses_dir = out_dir / "pulses"; pulses_dir.mkdir(parents=True, exist_ok=True)
    truth_dir  = out_dir / "truth";  truth_dir.mkdir(parents=True, exist_ok=True)
    truth_tracks_dir = truth_dir / "truth_tracks"; truth_tracks_dir.mkdir(parents=True, exist_ok=True)

    layer_radii = default_layer_radii(r_in, r_out, n_layers)
    rng = np.random.default_rng(seed)
    step_m = step_cm / 100.0

    global_event_id = 0  # monotonically increasing across all files
    for fi in range(n_files):
        pulses_rows: List[dict] = []
        truth_rows: List[dict] = []
        truth_tracks_rows: List[dict] = []

        kept_events = 0
        attempts = 0
        max_attempts = max(events_per_file * 10, 100)

        while kept_events < events_per_file and attempts < max_attempts:
            attempts += 1

            # Common event vertex within target cylinder
            vtx = sample_target_vertex(0.20, -0.04, 0.04, rng)

            sig_tracks, sig_truth = build_signal_tracks(n_signal_per_event, layer_radii, rng, r_in, r_out, z_half, step_m, beta, vtx)
            cmp_tracks, cmp_truth = build_compton_tracks(n_compton_per_event, surface_mode, p_inner, layer_radii, rng, r_in, r_out, z_half, step_m, beta)

            # Assemble per-event pulses & truth; if none, skip writing this event
            event_pulses: List[dict] = []
            event_truth_tracks: List[dict] = []

            track_id = 0
            for rows, trow in zip(sig_tracks, sig_truth):
                for r in rows:
                    r.update({"event_id": global_event_id, "particle": "signal", "track_id": track_id})
                event_pulses.extend(rows)
                trow.update({"event_id": global_event_id, "track_id": track_id})
                event_truth_tracks.append(trow)
                track_id += 1
            for rows, trow in zip(cmp_tracks, cmp_truth):
                for r in rows:
                    r.update({"event_id": global_event_id, "particle": "compton_electron", "track_id": track_id})
                event_pulses.extend(rows)
                trow.update({"event_id": global_event_id, "track_id": track_id})
                event_truth_tracks.append(trow)
                track_id += 1

            if len(event_pulses) == 0:
                # drop this event entirely (no truth row either)
                continue

            # keep this event
            pulses_rows.extend(event_pulses)
            truth_tracks_rows.extend(event_truth_tracks)
            truth_rows.append({
                "event_id": global_event_id,
                "position_x": float(vtx[0]),
                "position_y": float(vtx[1]),
                "position_z": float(vtx[2]),
            })

            kept_events += 1
            global_event_id += 1

        if kept_events < events_per_file:
            print(f"[warn] file {fi}: only kept {kept_events}/{events_per_file} events after {attempts} attempts.")

        p_path  = pulses_dir       / f"pulses_{fi}.parquet"
        t_path  = truth_dir        / f"truth_{fi}.parquet"
        tt_path = truth_tracks_dir / f"truth_tracks_{fi}.parquet"

        pulses_df = pd.DataFrame(pulses_rows)[["event_id","track_id","particle","dom_x","dom_y","dom_z","dom_t"]]
        truth_df  = pd.DataFrame(truth_rows)[["event_id","position_x","position_y","position_z"]]
        truth_tracks_df = pd.DataFrame(truth_tracks_rows)[["event_id","track_id","particle","start_x","start_y","start_z","dir_x","dir_y","dir_z","n_hits"]]

        safe_to_parquet(pulses_df, p_path)
        safe_to_parquet(truth_df,  t_path)
        safe_to_parquet(truth_tracks_df, tt_path)

        print(f"[write] file {fi}: kept {kept_events} events -> {out_dir}")

# -------------------------------
# CLI
# -------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Generate NEW dataset (signal + fixed-number Compton per event) with labels")
    p.add_argument("--n-files", type=int, default=2, help="Number of output files")
    p.add_argument("--events-per-file", type=int, default=200, help="Kept events per file (events with ≥1 hit)")
    p.add_argument("--n-signal-per-event", type=int, default=2, help="Number of signal tracks per event")
    p.add_argument("--n-compton-per-event", type=int, required=True, help="FIXED number of Compton electrons per event")
    p.add_argument("--surface", choices=["inner","outer","both"], default="outer", help="Compton birth surface (both uses --p-inner)")
    p.add_argument("--p-inner", type=float, default=0.5, help="If surface=both, probability to choose inner")
    p.add_argument("--r-inner", type=float, default=R_INNER_DEFAULT, help="Inner radius [m]")
    p.add_argument("--r-outer", type=float, default=R_OUTER_DEFAULT, help="Outer radius [m]")
    p.add_argument("--z-half",  type=float, default=TPC_HALF_LENGTH_DEFAULT, help="Half-length [m]")
    p.add_argument("--n-layers", type=int, default=N_LAYERS_DEFAULT, help="Number of radial layers")
    p.add_argument("--step-cm", type=float, default=2.0, help="Propagation step [cm] for boundary march")
    p.add_argument("--beta", type=float, default=0.95, help="Assumed v/c for timing")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    return p

def main():
    args = build_argparser().parse_args()

    out_dir = Path("training_data") / f"compton_{args.n_compton_per_event}" / f"particle_{args.n_signal_per_event}"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset(
        out_dir=out_dir,
        n_files=args.n_files,
        events_per_file=args.events_per_file,
        n_signal_per_event=args.n_signal_per_event,
        n_compton_per_event=args.n_compton_per_event,
        surface_mode=args.surface, p_inner=args.p_inner,
        r_in=args.r_inner, r_out=args.r_outer, z_half=args.z_half,
        n_layers=args.n_layers, step_cm=args.step_cm, beta=args.beta, seed=args.seed
    )

    out_dir = Path("validation_data") / f"compton_{args.n_compton_per_event}" / f"particle_{args.n_signal_per_event}"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset(
        out_dir=out_dir,
        n_files=args.n_files,
        events_per_file=args.events_per_file,
        n_signal_per_event=args.n_signal_per_event,
        n_compton_per_event=args.n_compton_per_event,
        surface_mode=args.surface, p_inner=args.p_inner,
        r_in=args.r_inner, r_out=args.r_outer, z_half=args.z_half,
        n_layers=args.n_layers, step_cm=args.step_cm, beta=args.beta, seed=args.seed
    )

if __name__ == "__main__":
    main()


