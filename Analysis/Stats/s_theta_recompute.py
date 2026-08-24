#!/usr/bin/env python3
"""
Recompute the parameter-sensitivity index $S_\\theta$ from the CORRECTED heatmap grid (R1.8).

Definition, verbatim from the manuscript (Methods, `-v2.tex` L310):

    $S_{\\theta} = \\sigma(\\{A(\\theta_i)\\}_{i=1}^n)$

where $\\theta_i$ is the $i$-th of $n$ sampled values of parameter $\\theta$ and
$A(\\theta_i)$ the corresponding mean equilibrium opinion. On the stubbornness x
diverger-fraction grid that makes $S_\\theta$ the standard deviation of the ten
*marginal* means along one axis (averaging over the other axis and over runs).

Two axes are reported:
  S_w    -- sensitivity to stubbornness $w$        (SI L43; currently MISLABELLED $S_\\rho$)
  S_rho  -- sensitivity to diverger fraction $\\rho$ (SI L37)

Why this script exists: the numbers in the SI predate the 2026-07-27 horizon correction,
and `heatmap_stats_multi.py` emits sensitivity per topology x algorithm without the
group-level pooling the SI quotes. This reproduces both, old vs corrected, as a durable
artefact.

Pooling caveat: the SI's group figures average the per-topology $S_\\theta$ across
topologies and algorithms ("mean-of-sigma"). Pooling all topologies before taking the
marginal means ("sigma-of-pooled") is also defensible and is reported alongside, since
R1.8 asks precisely how this index is defined.

Usage
-----
    python Analysis/Stats/s_theta_recompute.py
    HEATMAP_CSV=<grid.csv> python Analysis/Stats/s_theta_recompute.py   # single-grid mode
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "Output")
STATS = os.path.join(OUT, "Stats", "stubborness_backfirer")

CORRECTED = os.path.join(OUT, "heatmap_sweep_phased_CORRECTED_2026-07-27.csv")
# The `..._pxc.csv` master is deliberately left buggy for provenance; kept here only as
# the "old" column so the SI's published numbers can be reproduced and diffed.
BUGGY_MASTER = os.path.join(OUT, "heatmap_sweep_phased_sweep_20251014_1511_stubbornness_polarisingNode_f_pxc.csv")

AXES = {"S_w": "stubbornness", "S_rho": "polarisingNode_f"}
NAME = {("biased", "same"): "local (similar)", ("biased", "diff"): "local (opposite)",
        ("bridge", "same"): "bridge (similar)", ("bridge", "diff"): "bridge (opposite)",
        ("wtf", "nan"): "wtf", ("node2vec", "nan"): "node2vec",
        ("random", "nan"): "random", ("nan", "nan"): "static"}
VARIANT = {"local (similar)": "similar", "bridge (similar)": "similar",
           "local (opposite)": "opposite", "bridge (opposite)": "opposite"}


def load(path):
    d = pd.read_csv(path)
    d["mode"] = d["mode"].astype(str)
    d["rewiring"] = d["rewiring"].astype(str)
    d["algorithm"] = [NAME.get((m, r), f"{m}_{r}") for m, r in zip(d["mode"], d["rewiring"])]
    d["variant_type"] = d["algorithm"].map(VARIANT).fillna("other")
    return d


def s_theta(group, axis):
    """sigma over the marginal means of `state` along `axis`."""
    marginal = group.groupby(axis)["state"].mean()
    return float(np.std(marginal.values)) if len(marginal) > 1 else 0.0


def per_cell(d):
    """S_theta per topology x algorithm, both axes ('mean-of-sigma' inputs)."""
    rows = []
    for (topo, algo), g in d.groupby(["topology", "algorithm"]):
        rows.append({"topology": topo, "algorithm": algo,
                     "variant_type": g["variant_type"].iloc[0],
                     "mean_opinion": g["state"].mean(),
                     **{k: s_theta(g, ax) for k, ax in AXES.items()}})
    return pd.DataFrame(rows)


def pooled(d):
    """S_theta with topologies pooled first ('sigma-of-pooled')."""
    rows = []
    for algo, g in d.groupby("algorithm"):
        rows.append({"algorithm": algo, "variant_type": g["variant_type"].iloc[0],
                     "mean_opinion": g["state"].mean(),
                     **{k: s_theta(g, ax) for k, ax in AXES.items()}})
    return pd.DataFrame(rows)


def report(f, label, d):
    cell, pool = per_cell(d), pooled(d)
    f.write(f"\n{'=' * 92}\n{label}\n{'=' * 92}\n")
    f.write("\nPer topology x algorithm (marginal-mean sigma):\n")
    f.write(cell.sort_values(["topology", "algorithm"]).to_string(index=False,
            float_format=lambda v: f"{v:+.3f}") + "\n")
    f.write("\nGroup aggregates:\n")
    for vt in ["opposite", "similar", "other"]:
        c, p = cell[cell.variant_type == vt], pool[pool.variant_type == vt]
        if c.empty:
            continue
        f.write(f"  {vt:9s} mean-of-sigma  S_w={c.S_w.mean():.3f}  S_rho={c.S_rho.mean():.3f}"
                f"  <a>={c.mean_opinion.mean():+.3f}  (n={len(c)} topo x algo cells)\n")
        f.write(f"  {vt:9s} sigma-of-pooled S_w={p.S_w.mean():.3f}  S_rho={p.S_rho.mean():.3f}"
                f"  <a>={p.mean_opinion.mean():+.3f}  (n={len(p)} algorithms)\n")
    return cell, pool


def main():
    override = os.environ.get("HEATMAP_CSV")
    grids = [("CORRECTED grid (override)", override)] if override else \
            [("OLD -- buggy master (SI's published numbers)", BUGGY_MASTER),
             ("CORRECTED grid 2026-07-27", CORRECTED)]

    os.makedirs(STATS, exist_ok=True)
    txt = os.path.join(STATS, "s_theta_recompute_20260730.txt")
    csv = os.path.join(STATS, "s_theta_recompute_20260730.csv")
    frames = []
    with open(txt, "w") as f:
        f.write("S_theta recompute (R1.8) -- sigma of the marginal mean equilibrium opinion\n")
        f.write("Definition: rewiring-manuscript-pnasn-v2.tex L310. S_w = stubbornness axis,\n")
        f.write("S_rho = diverger-fraction axis (SI L43 labels the former S_rho: that is the O7 mislabel).\n")
        for label, path in grids:
            print(f"reading {os.path.relpath(path, ROOT)}")
            cell, _ = report(f, f"{label}\n  {os.path.relpath(path, ROOT)}", load(path))
            cell.insert(0, "grid", label.split(" --")[0].split(" (")[0])
            frames.append(cell)
    pd.concat(frames).to_csv(csv, index=False, float_format="%.4f")
    print(open(txt).read())
    print(f"saved {os.path.relpath(txt, ROOT)}\nsaved {os.path.relpath(csv, ROOT)}")


if __name__ == "__main__":
    main()
