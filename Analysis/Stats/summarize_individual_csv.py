#!/usr/bin/env python3
"""
Stream the 27.7 GB per-run "individual" CSV into two small summary files that
feed the R1.7 uncertainty figures. The big file is I/O-heavy but this pass is
RAM-light (chunked read, keeps only rows with t % STEP == 0), so it runs on this
box or the cluster.

INPUT cols: t,avg_state,std_states,model_run,scenario,rewiring,type
  avg_state  = per-run MEAN opinion over the 800 agents
  std_states = WITHIN-run agent dispersion (polarisation), NOT across-run spread
The across-run spread R1.7 asks for = percentiles ACROSS the 90 model_run values.

OUTPUTS (to Output/):
  per_run_summary.csv   one row per (type,scenario,rewiring,model_run):
       speed_t95, cooperativity, final_polarization
       -> Fig 3 error bars (IQR of speed/cooperativity across runs) + Table 1 CIs
  trajectory_bands.csv  one row per (type,scenario,rewiring,t):
       opinion_q25/q50/q75 (across-run spread of the per-run mean opinion)
       polar_q25/q50/q75   (across-run spread of within-run dispersion)
       -> Fig 2 IQR ribbons

Usage:
  python summarize_individual_csv.py [--in <csv>] [--step 100] [--chunk 2000000]
                                     [--save-downsampled]
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # repo root (this file lives in Analysis/Stats/)
OUTPUT = os.path.join(ROOT, "Output")
DEFAULT_IN = os.path.join(
    OUTPUT,
    "default_run_individual_N_800_n_90_pNf_0_pc_0.05_sweep_20251014_1704_phased_run_gme_2025-10-15.csv",
)

KEYS = ["type", "scenario", "rewiring", "model_run"]
GROUP = ["type", "scenario", "rewiring"]


def calculate_t95_convergence_speed(trajectory):
    """Time to reach 95% of final value -> speed in [0,1] (higher = faster).
    Copied VERBATIM from Analysis/Plotting/convergence_vs_cooperation.py so the
    per-run speed matches the metric Fig 3 already uses on the mean trajectory.
    CAVEAT: designed for mostly-positive trajectories; runs converging to a
    negative value can register speed~1 (trajectory>=negative target at t=0).
    Kept identical on purpose; the Fig 3 script decides how to summarise."""
    final_val = np.mean(trajectory[-int(len(trajectory) * 0.1):])
    target = 0.95 * final_val
    t95_idx = np.argmax(trajectory >= target) if np.any(trajectory >= target) else len(trajectory)
    speed = 1 - (t95_idx / len(trajectory))
    return max(0, speed)


def stream_downsample(path, step, chunk):
    """Return a downsampled DataFrame keeping only rows with t % step == 0."""
    kept = []
    total = 0
    for ch in pd.read_csv(path, chunksize=chunk,
                          dtype={"scenario": str, "rewiring": str, "type": str},
                          keep_default_na=False):
        ch["t"] = pd.to_numeric(ch["t"], errors="coerce")
        sub = ch[ch["t"] % step == 0]
        if len(sub):
            kept.append(sub)
        total += len(ch)
        print(f"  ...read {total:,} rows, kept {sum(len(k) for k in kept):,}", flush=True)
    df = pd.concat(kept, ignore_index=True)
    for c in ("avg_state", "std_states", "model_run"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def per_run_summary(df):
    rows = []
    for key, g in df.sort_values("t").groupby(KEYS):
        traj = g["avg_state"].to_numpy()
        win = max(1, int(len(traj) * 0.1))
        rows.append({
            "type": key[0], "scenario": key[1], "rewiring": key[2], "model_run": key[3],
            "speed_t95": calculate_t95_convergence_speed(traj),
            "cooperativity": float(np.mean(traj[-win:])),
            "final_polarization": float(np.mean(g["std_states"].to_numpy()[-win:])),
        })
    return pd.DataFrame(rows)


def trajectory_bands(df):
    g = df.groupby(GROUP + ["t"])
    bands = g["avg_state"].quantile([0.25, 0.5, 0.75]).unstack()
    bands.columns = ["opinion_q25", "opinion_q50", "opinion_q75"]
    pol = g["std_states"].quantile([0.25, 0.5, 0.75]).unstack()
    pol.columns = ["polar_q25", "polar_q50", "polar_q75"]
    return bands.join(pol).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--step", type=int, default=100, help="keep rows with t %% step == 0")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--save-downsampled", action="store_true")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    print(f"streaming {args.inp}  (step={args.step}, chunk={args.chunk:,})", flush=True)
    df = stream_downsample(args.inp, args.step, args.chunk)
    print(f"downsampled rows={len(df):,}  combos={df.groupby(GROUP).ngroups}  "
          f"runs/combo~{df.groupby(GROUP)['model_run'].nunique().median():.0f}", flush=True)

    p = args.out_prefix
    if args.save_downsampled:
        dpath = os.path.join(OUTPUT, f"{p}individual_downsampled.csv")
        df.to_csv(dpath, index=False)
        print(f"saved {dpath}", flush=True)

    summary = per_run_summary(df)
    spath = os.path.join(OUTPUT, f"{p}per_run_summary.csv")
    summary.to_csv(spath, index=False)
    print(f"saved {spath}  ({len(summary)} rows)", flush=True)

    bands = trajectory_bands(df)
    bpath = os.path.join(OUTPUT, f"{p}trajectory_bands.csv")
    bands.to_csv(bpath, index=False)
    print(f"saved {bpath}  ({len(bands)} rows)", flush=True)


if __name__ == "__main__":
    main()
