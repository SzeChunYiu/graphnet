#!/usr/bin/env python3
"""
run_signal_compton_toy_generation_mp.py
---------------------------------------
Multiprocess launcher for tpc_signal_plus_compton_fmt.py over (signal, compton) grids.

- Spawns one subprocess per combo using concurrent.futures.ProcessPoolExecutor
- Derives a unique seed per job (base_seed XOR a small hash of the combo)
- Limits parallelism to avoid CPU oversubscription; configurable via --max-workers
- Streams per-job logs to the console when jobs finish; raises if any job fails
"""

import os
import sys
import hashlib
import itertools
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

GEN_SCRIPT_DEFAULT = "tpc_signal_plus_compton_fmt.py"

def combo_seed(base_seed: int, n_signal: int, n_compton: int) -> int:
    """Derive a deterministic per-combo seed from a base seed."""
    s = f"{base_seed}:{n_signal}:{n_compton}".encode()
    h = int.from_bytes(hashlib.sha256(s).digest()[:4], "big")
    return base_seed ^ h

def run_one_job(gen_script: Path,
                n_files: int,
                events_per_file: int,
                n_signal: int,
                n_compton: int,
                surface: str,
                p_inner: float,
                base_seed: int) -> tuple:
    """Run a single generator invocation; returns (combo_str, returncode, stdout, stderr)."""
    # Unique seed per combo to avoid identical datasets
    seed = combo_seed(base_seed, n_signal, n_compton)

    cmd = [
        sys.executable, str(gen_script),
        "--n-files", str(n_files),
        "--events-per-file", str(events_per_file),
        "--n-signal-per-event", str(n_signal),
        "--n-compton-per-event", str(n_compton),
        "--surface", surface,
        "--p-inner", str(p_inner),
        "--seed", str(seed),
    ]

    # Keep threads per process small to avoid oversubscription of BLAS/OpenMP libs
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    combo_str = f"signal={n_signal}, compton={n_compton}"
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
        return (combo_str, proc.returncode, proc.stdout, proc.stderr)
    except Exception as e:
        return (combo_str, -999, "", f"Launcher exception: {e}")

def main():
    ap = argparse.ArgumentParser(description="Multiprocess toy data generation runner")
    ap.add_argument("--gen-script", type=Path, default=GEN_SCRIPT_DEFAULT,
                    help="Path to tpc_signal_plus_compton_fmt.py")
    ap.add_argument("--compton-min", type=int, default=1, help="Min compton per event (inclusive)")
    ap.add_argument("--compton-max", type=int, default=2, help="Max compton per event (inclusive)")
    ap.add_argument("--signal-min",  type=int, default=2, help="Min signal per event (inclusive)")
    ap.add_argument("--signal-max",  type=int, default=3, help="Max signal per event (inclusive)")
    ap.add_argument("--n-files", type=int, default=20, help="Number of files per job")
    ap.add_argument("--events-per-file", type=int, default=500, help="Kept events per file")
    ap.add_argument("--surface", choices=["inner","outer","both"], default="both",
                    help="Compton birth surface")
    ap.add_argument("--p-inner", type=float, default=0.5,
                    help="If surface='both', probability to choose inner")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    ap.add_argument("--max-workers", type=int, default=0,
                    help="Parallel workers (0 = auto: max(1, cpu_count()-1))")
    args = ap.parse_args()

    gen_script = args.gen_script.resolve()
    if not gen_script.exists():
        print(f"ERROR: generator not found: {gen_script}", file=sys.stderr)
        sys.exit(2)

    # Build the grid of (signal, compton)
    compton_vals = list(range(args.compton_min, args.compton_max + 1))
    signal_vals  = list(range(args.signal_min,  args.signal_max  + 1))
    combos = list(itertools.product(signal_vals, compton_vals))
    if not combos:
        print("No combinations to run.", file=sys.stderr)
        sys.exit(1)

    # Worker count
    if args.max_workers and args.max_workers > 0:
        max_workers = args.max_workers
    else:
        try:
            import os
            cpu = os.cpu_count() or 2
        except Exception:
            cpu = 2
        max_workers = max(1, cpu - 1)

    print(f"Launching {len(combos)} job(s) with max_workers={max_workers}")
    print(f"Generator: {gen_script}")
    print(f"Grid: {combos}")

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for n_signal, n_compton in combos:
            futures.append(
                ex.submit(
                    run_one_job,
                    gen_script,
                    args.n_files,
                    args.events_per_file,
                    n_signal,
                    n_compton,
                    args.surface,
                    args.p_inner,
                    args.seed,
                )
            )

        for fut in as_completed(futures):
            combo_str, rc, out, err = fut.result()
            header = f"\n=== Job finished: {combo_str} (rc={rc}) ==="
            print(header)
            if out:
                print("---- STDOUT ----")
                print(out.rstrip())
            if err:
                print("---- STDERR ----")
                print(err.rstrip())
            results.append((combo_str, rc))

    # Summary & exit code
    bad = [c for c, rc in results if rc != 0]
    if bad:
        print("\nFAILED jobs:", bad, file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll jobs completed successfully.")

if __name__ == "__main__":
    main()
